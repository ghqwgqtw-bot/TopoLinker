#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reward-independent VoroIF-GNN-v2 evaluation across TopoLinker stages.

Protocol: whole protein context + PROTAC-proximal contact aggregation.

Unlike the previous cropped-protein version, this script does NOT delete
protein residues before VoroIF. It:
  1) identifies PROTAC heavy atoms from the original DeepTernary complex;
  2) marks protein residues within --local-cutoff (default 10 A) of PROTAC;
  3) removes only HETATM records and runs VoroIF on the WHOLE POI+E3 proteins;
  4) asks VoroIF for detailed results_per_contact via --output-dir;
  5) keeps only VoroIF contacts whose endpoint residues satisfy the local rule
     (default: BOTH endpoints are PROTAC-proximal);
  6) recomputes local pCAD with the same aggregation formula used by the
     official VoroIF-GNN-v2 script.

Candidate main metric: voroif_local_pcad_mean (higher is better).

Smoke test:
  python evaluation/evaluate_voroif_three_stages.py \
    --voroif-app /root/software/voroif-gnn-v2-app \
    --processors 4 \
    --local-cutoff 10 \
    --max-strict-per-stage 100

Formal run (only after smoke-test diagnostics look reasonable):
  python evaluation/evaluate_voroif_three_stages.py \
    --voroif-app /root/software/voroif-gnn-v2-app \
    --processors 4 \
    --local-cutoff 10
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


STAGES = ("pretrain", "rl1", "rl2")
DEFAULT_RECORDS = {
    "pretrain": Path("eval_outputs/test_pretrain/records.pkl"),
    "rl1": Path("eval_outputs/test_rl/records.pkl"),
    "rl2": Path("eval_outputs/test_rl_final/records.pkl"),
}
PROTOCOL_NAME = "whole_context_protac_local_contact_dt_scored_v2"
STRICT_KEYS = ("strict_protac", "precomputed_strict", "strict_topology", "strict")
COMPLEX_PATH_KEYS = (
    "deepternary_complex_pred_path",
    "deepternary_complex_pdb_path",
    "deepternary_pred_complex_path",
)
GLOBAL_SCORE_COLUMNS = (
    "pgoodness", "area", "pgoodness_average", "pcadscore",
    "num_of_residues", "residue_pcadscore",
)
WATER_RESNAMES = {"HOH", "WAT", "DOD", "H2O"}


# -----------------------------------------------------------------------------
# records.pkl loading
# -----------------------------------------------------------------------------

class _DummyRDKitMol:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _RecordsUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "rdkit.Chem.rdchem" and name == "Mol":
            return _DummyRDKitMol
        return super().find_class(module, name)


def load_records(path: Path) -> List[dict]:
    with path.open("rb") as f:
        obj = _RecordsUnpickler(f).load()
    if isinstance(obj, pd.DataFrame):
        records = obj.to_dict(orient="records")
    elif isinstance(obj, list):
        records = obj
    elif isinstance(obj, tuple):
        records = list(obj)
    else:
        raise TypeError(f"Unsupported records object type in {path}: {type(obj).__name__}")
    if not all(isinstance(x, dict) for x in records):
        raise TypeError(f"Expected list[dict] in {path}")
    return records


# -----------------------------------------------------------------------------
# general helpers
# -----------------------------------------------------------------------------

def print_header(title: str) -> None:
    print("")
    print("=" * 100)
    print(title)
    print("=" * 100)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def finite_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def as_bool01(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (int, np.integer)):
        return int(value != 0)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return int(float(value) != 0.0)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n", ""}:
        return 0
    return None


def get_strict(rec: dict) -> Tuple[int, str]:
    for key in STRICT_KEYS:
        if key not in rec:
            continue
        flag = as_bool01(rec.get(key))
        if flag is not None:
            return flag, key
    raise KeyError("Cannot determine strict flag. Tried: " + ", ".join(STRICT_KEYS))


def get_complex_path_value(rec: dict) -> Tuple[str, str]:
    for key in COMPLEX_PATH_KEYS:
        value = safe_str(rec.get(key)).strip()
        if value:
            return value, key
    return "", ""


def first_present(rec: dict, keys: Sequence[str]) -> Any:
    for key in keys:
        if key in rec:
            return rec.get(key)
    return None


def resolve_record_path(value: str | Path, project_root: Path) -> Path:
    text = str(value).strip().strip('"').strip("'").replace("\\", "/")
    win = re.match(r"^([A-Za-z]):/(.*)$", text)
    if win:
        return Path(f"/mnt/{win.group(1).lower()}/{win.group(2)}")
    p = Path(text).expanduser()
    if p.is_absolute():
        return p
    while text.startswith("./"):
        text = text[2:]
    return project_root / text


def file_nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


# -----------------------------------------------------------------------------
# PDB and Voronota residue identifiers
# -----------------------------------------------------------------------------

def pdb_element(line: str) -> str:
    if len(line) >= 78:
        element = line[76:78].strip().upper()
        if element:
            return element
    atom_name = line[12:16].strip()
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    return letters[0].upper() if letters else ""


def is_heavy_pdb_atom(line: str) -> bool:
    return pdb_element(line) not in {"", "H", "D"}


def pdb_xyz(line: str) -> np.ndarray:
    try:
        return np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
    except Exception as exc:
        raise ValueError(f"Cannot parse PDB coordinates: {line!r}") from exc


def canonical_residue_label(resseq: str, icode: str = "") -> str:
    raw = f"{safe_str(resseq).strip()}{safe_str(icode).strip()}".strip()
    m = re.fullmatch(r"([+-]?\d+)([A-Za-z]?)", raw)
    if m:
        return f"{int(m.group(1))}{m.group(2).upper()}"
    return raw.upper()


def pdb_residue_id(line: str) -> Tuple[str, str]:
    chain = line[21:22].strip()
    return chain, canonical_residue_label(line[22:26], line[26:27])


def pdb_hetero_key(line: str) -> Tuple[str, str, str]:
    chain, resid = pdb_residue_id(line)
    resname = line[17:20].strip().upper()
    return chain, resid, resname


_CHAIN_RE = re.compile(r"c<([^>]*)>")
_RES_RE = re.compile(r"r<([^>]*)>")


def parse_voronota_residue_id(value: Any) -> Optional[Tuple[str, str]]:
    """Parse e.g. c<A>r<123>R<LYS> -> ('A', '123')."""
    text = safe_str(value).strip()
    cm = _CHAIN_RE.search(text)
    rm = _RES_RE.search(text)
    if cm is None or rm is None:
        return None
    return cm.group(1).strip(), canonical_residue_label(rm.group(1))


def choose_protac_ligand(
    hetero_heavy: Dict[Tuple[str, str, str], List[np.ndarray]],
    ligand_resname: str,
) -> Tuple[Tuple[str, str, str], np.ndarray]:
    requested = ligand_resname.strip().upper()
    candidates = []
    for key, coords in hetero_heavy.items():
        if key[2] in WATER_RESNAMES or not coords:
            continue
        if requested != "AUTO" and key[2] != requested:
            continue
        candidates.append((key, coords))
    if not candidates:
        available = sorted({k[2] for k, v in hetero_heavy.items() if v and k[2] not in WATER_RESNAMES})
        raise RuntimeError(
            f"No ligand HETATM residue found for selector={ligand_resname!r}; available={available}"
        )
    candidates.sort(key=lambda x: (-len(x[1]), x[0]))
    key, coords = candidates[0]
    return key, np.vstack(coords).astype(np.float64, copy=False)


def prepare_whole_protein_and_local_context(
    src: Path,
    dst: Path,
    *,
    cutoff_A: float,
    ligand_resname: str,
) -> Tuple[Dict[str, Any], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    Write ALL protein ATOM/TER records to dst, while using the original PDB's
    HETATM PROTAC only to identify PROTAC-proximal protein residues.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    protein_heavy: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
    hetero_heavy: Dict[Tuple[str, str, str], List[np.ndarray]] = defaultdict(list)
    all_residues: Set[Tuple[str, str]] = set()
    chains: Set[str] = set()
    atom_count = 0

    with src.open("r", encoding="utf-8", errors="ignore") as fin, \
         dst.open("w", encoding="utf-8", newline="\n") as fout:
        for raw in fin:
            line = raw.rstrip("\r\n")
            if line.startswith("ATOM"):
                fout.write(line + "\n")
                atom_count += 1
                rid = pdb_residue_id(line)
                all_residues.add(rid)
                if rid[0]:
                    chains.add(rid[0])
                if is_heavy_pdb_atom(line):
                    protein_heavy[rid].append(pdb_xyz(line))
            elif line.startswith("TER"):
                fout.write(line + "\n")
            elif line.startswith("HETATM") and is_heavy_pdb_atom(line):
                hetero_heavy[pdb_hetero_key(line)].append(pdb_xyz(line))
        fout.write("END\n")

    ligand_key, ligand_xyz = choose_protac_ligand(hetero_heavy, ligand_resname)
    cutoff2 = float(cutoff_A) ** 2
    local_residues: Set[Tuple[str, str]] = set()
    local_min_distances: List[float] = []

    for rid, coords in protein_heavy.items():
        xyz = np.vstack(coords).astype(np.float64, copy=False)
        diff = xyz[:, None, :] - ligand_xyz[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        min_d2 = float(np.min(d2))
        if min_d2 <= cutoff2:
            local_residues.add(rid)
            local_min_distances.append(math.sqrt(max(0.0, min_d2)))

    by_chain = Counter(chain for chain, _ in local_residues if chain)
    info = {
        "protein_atom_count": atom_count,
        "protein_chain_count": len(chains),
        "protein_chains": ",".join(sorted(chains)),
        "all_protein_residue_count": len(all_residues),
        "protac_proximal_residue_count": len(local_residues),
        "protac_proximal_residue_count_by_chain": ";".join(
            f"{c}:{by_chain[c]}" for c in sorted(by_chain)
        ),
        "selected_ligand_resname": ligand_key[2],
        "selected_ligand_chain": ligand_key[0],
        "selected_ligand_resseq": ligand_key[1],
        "ligand_heavy_atom_count": int(ligand_xyz.shape[0]),
        "closest_local_residue_distance_A": min(local_min_distances) if local_min_distances else np.nan,
        "farthest_local_residue_min_distance_A": max(local_min_distances) if local_min_distances else np.nan,
    }
    return info, local_residues, all_residues


# -----------------------------------------------------------------------------
# VoroIF output + official local aggregation
# -----------------------------------------------------------------------------

def read_whitespace_table(path: Path) -> pd.DataFrame:
    if not file_nonempty(path):
        return pd.DataFrame()
    return pd.read_csv(path, sep=r"\s+", engine="python")


def read_voroif_global_table(path: Path) -> pd.DataFrame:
    df = read_whitespace_table(path)
    if df.empty:
        return df
    if "ID" not in df.columns:
        raise RuntimeError(f"VoroIF global table has no ID column: {path}")
    for col in GLOBAL_SCORE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_voroif_contact_table(path: Path) -> pd.DataFrame:
    df = read_whitespace_table(path)
    if df.empty:
        return df
    required = {"ID1", "ID2", "area", "pgoodness"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"results_per_contact missing {sorted(missing)}: {path}")
    for col in ("area", "pgoodness", "pgoodness_average", "pcadscore"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def official_pcad_from_average(avg: float) -> float:
    if not math.isfinite(avg):
        return float("nan")
    return (math.tanh(((avg + 0.2) / 1.2) - 1.0) + 1.0) / 2.0


def aggregate_local_contacts(
    contacts: pd.DataFrame,
    *,
    local_residues: Set[Tuple[str, str]],
    all_protein_residues: Set[Tuple[str, str]],
    contact_rule: str,
) -> Dict[str, Any]:
    """Aggregate local pCAD from VoroIF's WHOLE-context results_per_contact."""
    if contacts.empty:
        raise RuntimeError("Empty results_per_contact")

    rows = []
    parse_fail_examples = []
    map_fail_examples = []

    for _, crow in contacts.iterrows():
        rid1 = parse_voronota_residue_id(crow["ID1"])
        rid2 = parse_voronota_residue_id(crow["ID2"])
        if rid1 is None or rid2 is None:
            if len(parse_fail_examples) < 3:
                parse_fail_examples.append(f"{crow['ID1']} | {crow['ID2']}")
            continue
        if rid1 not in all_protein_residues or rid2 not in all_protein_residues:
            if len(map_fail_examples) < 3:
                map_fail_examples.append(f"{crow['ID1']}->{rid1}; {crow['ID2']}->{rid2}")
            continue
        area = finite_float(crow["area"])
        pgood = finite_float(crow["pgoodness"])
        if not math.isfinite(area) or not math.isfinite(pgood):
            continue
        rows.append((rid1, rid2, area, pgood))

    total = int(len(contacts))
    parsed_mapped = len(rows)
    if parse_fail_examples:
        raise RuntimeError(
            f"Could not parse all VoroIF residue IDs; examples={parse_fail_examples}"
        )
    if map_fail_examples:
        raise RuntimeError(
            f"VoroIF residue IDs do not map to input PDB; examples={map_fail_examples}"
        )

    selected = []
    for rid1, rid2, area, pgood in rows:
        a = rid1 in local_residues
        b = rid2 in local_residues
        keep = (a and b) if contact_rule == "both" else (a or b)
        if keep:
            selected.append((rid1, rid2, area, pgood))

    result = {
        "voroif_contact_rows_total": total,
        "voroif_contact_rows_mapped": parsed_mapped,
        "local_contact_count": len(selected),
        "local_pgoodness": np.nan,
        "local_area": np.nan,
        "local_pgoodness_average": np.nan,
        "local_pcadscore": np.nan,
        "local_num_of_residues": 0,
        "local_residue_pcadscore": np.nan,
    }
    if not selected:
        return result

    local_area = float(sum(x[2] for x in selected))
    local_pgoodness = float(sum(x[3] for x in selected))
    local_avg = local_pgoodness / local_area if local_area > 0 else float("nan")
    local_pcad = official_pcad_from_average(local_avg)

    residue_area: Dict[Tuple[str, str], float] = defaultdict(float)
    residue_pgood: Dict[Tuple[str, str], float] = defaultdict(float)
    for rid1, rid2, area, pgood in selected:
        for rid in (rid1, rid2):
            residue_area[rid] += area
            residue_pgood[rid] += pgood

    weighted_num = 0.0
    weighted_den = 0.0
    for rid, area in residue_area.items():
        if area <= 0:
            continue
        rpcad = official_pcad_from_average(residue_pgood[rid] / area)
        if math.isfinite(rpcad):
            weighted_num += rpcad * area
            weighted_den += area

    result.update({
        "local_pgoodness": local_pgoodness,
        "local_area": local_area,
        "local_pgoodness_average": local_avg,
        "local_pcadscore": local_pcad,
        "local_num_of_residues": len(residue_area),
        "local_residue_pcadscore": weighted_num / weighted_den if weighted_den > 0 else np.nan,
    })
    return result


def locate_results_per_contact(details_output_dir: Path, input_basename: str) -> Path:
    direct = details_output_dir / "detailed_results" / input_basename / "results_per_contact"
    if file_nonempty(direct):
        return direct
    root = details_output_dir / "detailed_results"
    candidates = list(root.rglob("results_per_contact")) if root.exists() else []
    exact = [p for p in candidates if p.parent.name == input_basename]
    if len(exact) == 1:
        return exact[0]
    stem = Path(input_basename).stem
    stem_matches = [p for p in candidates if Path(p.parent.name).stem == stem]
    if len(stem_matches) == 1:
        return stem_matches[0]
    raise FileNotFoundError(
        f"Cannot locate unique results_per_contact for {input_basename!r}; candidates={len(candidates)}"
    )


def run_voroif_chunk(
    *,
    voroif_exec: Path,
    voroif_app: Path,
    pdb_paths: List[Path],
    processors: int,
    list_file: Path,
    score_file: Path,
    details_output_dir: Path,
    timeout: int,
) -> Tuple[int, str, float]:
    list_file.parent.mkdir(parents=True, exist_ok=True)
    score_file.parent.mkdir(parents=True, exist_ok=True)
    if details_output_dir.exists():
        shutil.rmtree(details_output_dir)
    details_output_dir.mkdir(parents=True, exist_ok=True)
    list_file.write_text("".join(f"{p}\n" for p in pdb_paths), encoding="utf-8")

    cmd = [
        str(voroif_exec),
        "--stdin-file", str(list_file),
        "--processors", str(processors),
        "--output-table-file", str(score_file),
        "--output-dir", str(details_output_dir),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(voroif_app),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )
        return int(proc.returncode), proc.stdout or "", time.perf_counter() - t0
    except subprocess.TimeoutExpired as exc:
        out = ""
        if exc.stdout:
            out = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout)
        return -999, out + "\nTIMEOUT", time.perf_counter() - t0


# -----------------------------------------------------------------------------
# result rows / resume / summaries
# -----------------------------------------------------------------------------

def make_initial_rows(
    stage: str,
    records: List[dict],
    *,
    cutoff_A: float,
    contact_rule: str,
    ligand_resname: str,
) -> List[Dict[str, Any]]:
    rows = []
    for idx, rec in enumerate(records):
        strict, strict_source = get_strict(rec)
        complex_value, complex_source = get_complex_path_value(rec)
        rows.append({
            "stage": stage,
            "record_index": idx,
            "uuid": safe_str(rec.get("uuid")),
            "sample_id": safe_str(rec.get("sample_id")),
            "strict": int(strict),
            "strict_source": strict_source,
            "evaluation_protocol": PROTOCOL_NAME,
            "local_cutoff_A": float(cutoff_A),
            "local_contact_rule": contact_rule,
            "ligand_resname_request": ligand_resname,
            "deepternary_complex_pred_path": complex_value,
            "deepternary_complex_path_source": complex_source,
            "resolved_complex_path": "",
            "deepternary_pred_p2_rmsd": finite_float(first_present(rec, (
                "deepternary_pred_p2_rmsd", "precomputed_deepternary_p2_rmsd",
            ))),
            "deepternary_clash_ratio": finite_float(first_present(rec, (
                "deepternary_clash_ratio", "precomputed_deepternary_clash_ratio",
            ))),
            "deepternary_scored": as_bool01(first_present(rec, (
                "deepternary_scored", "precomputed_deepternary_scored",
            ))),
            "selected_for_run": 0,
            "protein_atom_count": np.nan,
            "protein_chain_count": np.nan,
            "protein_chains": "",
            "all_protein_residue_count": np.nan,
            "protac_proximal_residue_count": np.nan,
            "protac_proximal_residue_count_by_chain": "",
            "selected_ligand_resname": "",
            "selected_ligand_chain": "",
            "selected_ligand_resseq": "",
            "ligand_heavy_atom_count": np.nan,
            "closest_local_residue_distance_A": np.nan,
            "farthest_local_residue_min_distance_A": np.nan,
            "whole_pgoodness": np.nan,
            "whole_area": np.nan,
            "whole_pgoodness_average": np.nan,
            "whole_pcadscore": np.nan,
            "whole_num_of_residues": np.nan,
            "whole_residue_pcadscore": np.nan,
            "voroif_contact_rows_total": np.nan,
            "voroif_contact_rows_mapped": np.nan,
            "local_contact_count": np.nan,
            "local_pgoodness": np.nan,
            "local_area": np.nan,
            "local_pgoodness_average": np.nan,
            "local_pcadscore": np.nan,
            "local_num_of_residues": np.nan,
            "local_residue_pcadscore": np.nan,
            # backward-compatible aliases = LOCAL values in this protocol
            "pgoodness": np.nan,
            "area": np.nan,
            "pgoodness_average": np.nan,
            "pcadscore": np.nan,
            "num_of_residues": np.nan,
            "residue_pcadscore": np.nan,
            "status": "not_strict" if not strict else "pending",
        })
    return rows


def resume_settings_match(old: pd.DataFrame, *, cutoff_A: float, contact_rule: str, ligand_resname: str) -> bool:
    if "evaluation_protocol" not in old.columns:
        return False
    protocols = set(old["evaluation_protocol"].dropna().astype(str))
    if protocols and protocols != {PROTOCOL_NAME}:
        return False
    if "local_cutoff_A" not in old.columns or "local_contact_rule" not in old.columns:
        return False
    cuts = pd.to_numeric(old["local_cutoff_A"], errors="coerce").dropna()
    if len(cuts) and not np.allclose(cuts.to_numpy(float), float(cutoff_A), rtol=0, atol=1e-9):
        return False
    rules = set(old["local_contact_rule"].dropna().astype(str))
    if rules and rules != {contact_rule}:
        return False
    if "ligand_resname_request" in old.columns:
        ligs = set(old["ligand_resname_request"].dropna().astype(str).str.lower())
        if ligs and ligs != {ligand_resname.lower()}:
            return False
    return True


def reuse_existing_scores(
    rows: List[Dict[str, Any]],
    existing_csv: Path,
    *,
    cutoff_A: float,
    contact_rule: str,
    ligand_resname: str,
) -> int:
    if not file_nonempty(existing_csv):
        return 0
    try:
        old = pd.read_csv(existing_csv, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[WARN] Cannot read existing CSV for resume: {exc}")
        return 0
    if not resume_settings_match(old, cutoff_A=cutoff_A, contact_rule=contact_rule, ligand_resname=ligand_resname):
        print("[WARN] Existing CSV uses a different protocol/cutoff/rule; not reused.")
        return 0
    if "record_index" not in old.columns or "local_pcadscore" not in old.columns:
        return 0
    old["record_index"] = pd.to_numeric(old["record_index"], errors="coerce")
    old["local_pcadscore"] = pd.to_numeric(old["local_pcadscore"], errors="coerce")
    reusable = old[old["record_index"].notna() & old["local_pcadscore"].notna()].copy()
    if reusable.empty:
        return 0
    by_idx = {int(r["record_index"]): r for _, r in reusable.iterrows()}
    reused = 0
    for row in rows:
        if int(row["strict"]) != 1:
            continue

        if int(row.get("deepternary_scored") or 0) != 1:
            continue

        old_row = by_idx.get(int(row["record_index"]))
        if old_row is None:
            continue
        for col in row.keys():
            if col in old_row.index and col not in {
                "stage", "record_index", "uuid", "sample_id", "strict", "strict_source",
                "deepternary_complex_pred_path", "deepternary_complex_path_source",
                "deepternary_pred_p2_rmsd", "deepternary_clash_ratio", "deepternary_scored",
            } and not pd.isna(old_row[col]):
                row[col] = old_row[col]
        if math.isfinite(finite_float(row["local_pcadscore"])):
            row["selected_for_run"] = 1
            row["status"] = "scored_reused"
            reused += 1
    return reused


def save_stage_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def summarize_stage(stage: str, rows: List[Dict[str, Any]], *, cutoff_A: float, contact_rule: str) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    strict_mask = (
            pd.to_numeric(
                df["strict"],
                errors="coerce",
            ).fillna(0).astype(int) == 1
    )

    deepternary_scored_mask = (
            pd.to_numeric(
                df["deepternary_scored"],
                errors="coerce",
            ).fillna(0).astype(int) == 1
    )

    eligible_mask = (
            strict_mask
            & deepternary_scored_mask
    )

    selected_mask = (
            pd.to_numeric(
                df["selected_for_run"],
                errors="coerce",
            ).fillna(0).astype(int) == 1
    )

    scored_mask = (
            eligible_mask
            & pd.to_numeric(
        df["local_pcadscore"],
        errors="coerce",
    ).notna()
    )

    def s(col: str) -> pd.Series:
        return pd.to_numeric(df.loc[scored_mask, col], errors="coerce").dropna()

    def stat(x: pd.Series, fn: str) -> float:
        if len(x) == 0:
            return float("nan")
        if fn == "mean": return float(x.mean())
        if fn == "median": return float(x.median())
        if fn == "std": return float(x.std(ddof=1)) if len(x) > 1 else 0.0
        if fn == "min": return float(x.min())
        if fn == "max": return float(x.max())
        if fn == "q05": return float(x.quantile(0.05))
        if fn == "q25": return float(x.quantile(0.25))
        if fn == "q75": return float(x.quantile(0.75))
        if fn == "q95": return float(x.quantile(0.95))
        raise ValueError(fn)

    strict_n = int(
        strict_mask.sum()
    )

    deepternary_scored_strict_n = int(
        eligible_mask.sum()
    )

    selected_n = int(
        (
                eligible_mask
                & selected_mask
        ).sum()
    )

    scored_n = int(
        scored_mask.sum()
    )

    pcad = s("local_pcadscore")

    return {
        "stage": stage,
        "evaluation_protocol": PROTOCOL_NAME,
        "local_cutoff_A": float(cutoff_A),
        "local_contact_rule": contact_rule,
        "generated": int(len(df)),
        "strict": strict_n,

        "deepternary_scored_strict":
            deepternary_scored_strict_n,

        "selected_deepternary_scored_strict_for_run":
            selected_n,

        "successfully_voroif_local_scored_deepternary_strict":
            scored_n,

        "voroif_scored_rate_among_deepternary_scored_strict": (
            scored_n / deepternary_scored_strict_n
            if deepternary_scored_strict_n
            else np.nan
        ),

        "voroif_scored_rate_among_selected": (
            scored_n / selected_n
            if selected_n
            else np.nan
        ),

        "voroif_local_pcad_mean": stat(pcad, "mean"),
        "voroif_local_pcad_median": stat(pcad, "median"),
        "voroif_local_pcad_std": stat(pcad, "std"),
        "voroif_local_pcad_min": stat(pcad, "min"),
        "voroif_local_pcad_q05": stat(pcad, "q05"),
        "voroif_local_pcad_q25": stat(pcad, "q25"),
        "voroif_local_pcad_q75": stat(pcad, "q75"),
        "voroif_local_pcad_q95": stat(pcad, "q95"),
        "voroif_local_pcad_max": stat(pcad, "max"),
        "voroif_local_interface_area_mean": stat(s("local_area"), "mean"),
        "voroif_local_interface_residue_count_mean": stat(s("local_num_of_residues"), "mean"),
        "voroif_local_contact_count_mean": stat(s("local_contact_count"), "mean"),
        "protac_proximal_protein_residue_count_mean": stat(s("protac_proximal_residue_count"), "mean"),
        "voroif_whole_pcad_mean_on_local_scored": stat(s("whole_pcadscore"), "mean"),
        "status_counts": dict(Counter(df["status"].astype(str))),
    }


# -----------------------------------------------------------------------------
# per-stage evaluation
# -----------------------------------------------------------------------------

def evaluate_stage(
    *,
    stage: str,
    records_path: Path,
    project_root: Path,
    out_root: Path,
    voroif_app: Path,
    voroif_exec: Path,
    processors: int,
    chunk_size: int,
    timeout_per_chunk: int,
    max_strict_per_stage: Optional[int],
    keep_work: bool,
    cutoff_A: float,
    ligand_resname: str,
    contact_rule: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print_header(f"VoroIF whole-context + local-contact: {stage}")
    print(f"records: {records_path}")

    records = load_records(records_path)
    rows = make_initial_rows(stage, records, cutoff_A=cutoff_A, contact_rule=contact_rule, ligand_resname=ligand_resname)
    stage_csv = out_root / "stage_results" / f"{stage}_voroif_records.csv"

    reused = reuse_existing_scores(
        rows, stage_csv,
        cutoff_A=cutoff_A,
        contact_rule=contact_rule,
        ligand_resname=ligand_resname,
    )
    if reused:
        print(f"Resume: reused {reused} existing local-contact scores")

    pending: List[Tuple[int, Path]] = []
    strict_seen = 0
    for row in rows:
        if int(row["strict"]) != 1:
            continue

        if int(row.get("deepternary_scored") or 0) != 1:
            row["status"] = "deepternary_not_scored"
            continue
        if max_strict_per_stage is not None:
            if strict_seen >= max_strict_per_stage:
                if row["status"] != "scored_reused":
                    row["status"] = "not_attempted_limit"
                continue
            strict_seen += 1
        if row["status"] == "scored_reused":
            continue
        raw_path = safe_str(row["deepternary_complex_pred_path"]).strip()
        if not raw_path:
            row["status"] = "missing_complex_path"
            continue
        src = resolve_record_path(raw_path, project_root)
        row["resolved_complex_path"] = str(src)
        if not file_nonempty(src):
            row["status"] = "missing_complex_file"
            continue
        row["selected_for_run"] = 1
        pending.append((int(row["record_index"]), src))

    print(
        f"generated={len(rows)} | strict={sum(int(r['strict']) for r in rows)} | "
        f"selected_for_run={sum(int(r['selected_for_run']) for r in rows)} | "
        f"pending_for_voroif={len(pending)} | reused={reused}"
    )

    if not pending:
        save_stage_rows(rows, stage_csv)
        summary = summarize_stage(stage, rows, cutoff_A=cutoff_A, contact_rule=contact_rule)
        return pd.DataFrame(rows), summary

    work_stage = out_root / "work" / stage
    raw_scores_dir = out_root / "raw_scores" / stage
    logs_dir = out_root / "logs" / stage
    raw_scores_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    row_by_idx = {int(r["record_index"]): r for r in rows}
    total_chunks = math.ceil(len(pending) / chunk_size)

    for chunk_id, start in enumerate(range(0, len(pending), chunk_size)):
        chunk = pending[start:start + chunk_size]
        print(f"[{stage}] chunk {chunk_id + 1}/{total_chunks} | records={len(chunk)}")

        chunk_dir = work_stage / f"chunk_{chunk_id:04d}"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        pdb_dir = chunk_dir / "whole_protein_pdbs"
        pdb_dir.mkdir(parents=True, exist_ok=True)
        details_dir = chunk_dir / "voroif_details"

        pdb_paths: List[Path] = []
        basename_to_idx: Dict[str, int] = {}
        local_by_idx: Dict[int, Set[Tuple[str, str]]] = {}
        all_by_idx: Dict[int, Set[Tuple[str, str]]] = {}

        for idx, src in chunk:
            row = row_by_idx[idx]
            dst = pdb_dir / f"{stage}_record_{idx:06d}.pdb"
            try:
                info, local_set, all_set = prepare_whole_protein_and_local_context(
                    src, dst, cutoff_A=cutoff_A, ligand_resname=ligand_resname
                )
            except Exception as exc:
                row["status"] = f"protein_context_prepare_failed:{type(exc).__name__}:{exc}"
                continue
            for k, v in info.items():
                row[k] = v
            if int(info["protein_chain_count"]) < 2:
                row["status"] = "fewer_than_two_protein_chains"
                continue
            if int(info["protac_proximal_residue_count"]) <= 0:
                row["status"] = "no_protac_proximal_residues"
                continue
            row["status"] = "prepared_whole_context"
            pdb_paths.append(dst)
            basename_to_idx[dst.name] = idx
            local_by_idx[idx] = local_set
            all_by_idx[idx] = all_set

        if not pdb_paths:
            save_stage_rows(rows, stage_csv)
            if not keep_work:
                shutil.rmtree(chunk_dir, ignore_errors=True)
            continue

        list_file = raw_scores_dir / f"chunk_{chunk_id:04d}_input.txt"
        score_file = raw_scores_dir / f"chunk_{chunk_id:04d}_global_scores.txt"
        log_file = logs_dir / f"chunk_{chunk_id:04d}.log"

        rc, stdout, elapsed = run_voroif_chunk(
            voroif_exec=voroif_exec,
            voroif_app=voroif_app,
            pdb_paths=pdb_paths,
            processors=processors,
            list_file=list_file,
            score_file=score_file,
            details_output_dir=details_dir,
            timeout=timeout_per_chunk,
        )
        log_file.write_text(
            f"return_code={rc}\nelapsed_s={elapsed:.3f}\n\n{stdout}",
            encoding="utf-8", errors="ignore"
        )

        if rc != 0:
            print(f"[WARN] VoroIF chunk failed: stage={stage}, chunk={chunk_id}, return_code={rc}")
            for p in pdb_paths:
                row_by_idx[basename_to_idx[p.name]]["status"] = f"voroif_failed:returncode={rc}"
            save_stage_rows(rows, stage_csv)
            if not keep_work:
                shutil.rmtree(chunk_dir, ignore_errors=True)
            continue

        try:
            global_scores = read_voroif_global_table(score_file)
            global_by_name = {Path(str(r["ID"])).name: r for _, r in global_scores.iterrows()}
        except Exception as exc:
            print(f"[WARN] Cannot parse global VoroIF table: {exc}")
            for p in pdb_paths:
                row_by_idx[basename_to_idx[p.name]]["status"] = "voroif_global_parse_failed"
            save_stage_rows(rows, stage_csv)
            if not keep_work:
                shutil.rmtree(chunk_dir, ignore_errors=True)
            continue

        for p in pdb_paths:
            idx = basename_to_idx[p.name]
            row = row_by_idx[idx]
            grow = global_by_name.get(p.name)
            if grow is None:
                row["status"] = "voroif_no_global_score"
                continue

            row["whole_pgoodness"] = finite_float(grow.get("pgoodness", np.nan))
            row["whole_area"] = finite_float(grow.get("area", np.nan))
            row["whole_pgoodness_average"] = finite_float(grow.get("pgoodness_average", np.nan))
            row["whole_pcadscore"] = finite_float(grow.get("pcadscore", np.nan))
            row["whole_num_of_residues"] = finite_float(grow.get("num_of_residues", np.nan))
            row["whole_residue_pcadscore"] = finite_float(grow.get("residue_pcadscore", np.nan))

            try:
                contact_file = locate_results_per_contact(details_dir, p.name)
                contacts = read_voroif_contact_table(contact_file)
                local_result = aggregate_local_contacts(
                    contacts,
                    local_residues=local_by_idx[idx],
                    all_protein_residues=all_by_idx[idx],
                    contact_rule=contact_rule,
                )
            except Exception as exc:
                row["status"] = f"local_contact_aggregation_failed:{type(exc).__name__}:{exc}"
                continue

            for k, v in local_result.items():
                row[k] = v

            if not math.isfinite(finite_float(row["local_pcadscore"])):
                row["status"] = (
                    "no_protac_proximal_voroif_contacts"
                    if int(finite_float(row["local_contact_count"]) or 0) == 0
                    else "local_voroif_nonfinite_score"
                )
                continue

            # aliases intentionally represent LOCAL-contact values
            row["pgoodness"] = row["local_pgoodness"]
            row["area"] = row["local_area"]
            row["pgoodness_average"] = row["local_pgoodness_average"]
            row["pcadscore"] = row["local_pcadscore"]
            row["num_of_residues"] = row["local_num_of_residues"]
            row["residue_pcadscore"] = row["local_residue_pcadscore"]
            row["status"] = "scored_local_contact"

        save_stage_rows(rows, stage_csv)
        n_scored = sum(
            1
            for r in rows
            if (
                    int(r["strict"]) == 1
                    and int(r.get("deepternary_scored") or 0) == 1
                    and math.isfinite(
                finite_float(
                    r["local_pcadscore"]
                )
            )
            )
        )
        print(
            f"[{stage}] chunk {chunk_id + 1}/{total_chunks} done in {elapsed:.1f}s | "
            f"cumulative local-scored={n_scored}"
        )
        if not keep_work:
            shutil.rmtree(chunk_dir, ignore_errors=True)

    save_stage_rows(rows, stage_csv)
    summary = summarize_stage(stage, rows, cutoff_A=cutoff_A, contact_rule=contact_rule)
    return pd.DataFrame(rows), summary


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Whole-context VoroIF followed by PROTAC-proximal contact reaggregation.",
    )
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--pretrain", default=str(DEFAULT_RECORDS["pretrain"]))
    ap.add_argument("--rl1", default=str(DEFAULT_RECORDS["rl1"]))
    ap.add_argument("--rl2", default=str(DEFAULT_RECORDS["rl2"]))
    ap.add_argument("--local-cutoff", type=float, default=10.0)
    ap.add_argument(
        "--local-contact-rule", choices=("both", "either"), default="both",
        help="Formal recommendation: both endpoint residues must be PROTAC-proximal.",
    )
    ap.add_argument(
        "--ligand-resname", default="auto",
        help="PROTAC HETATM resname; auto = largest non-water HETATM residue.",
    )
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--voroif-app", default="/root/software/voroif-gnn-v2-app")
    ap.add_argument("--processors", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--timeout-per-chunk", type=int, default=7200)
    ap.add_argument("--max-strict-per-stage", type=int, default=None)
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    if args.local_cutoff <= 0:
        raise ValueError("--local-cutoff must be > 0")
    if args.processors < 1 or args.chunk_size < 1:
        raise ValueError("--processors and --chunk-size must be >= 1")

    project_root = Path(args.project_root).expanduser().resolve()
    records_paths = {
        "pretrain": resolve_record_path(args.pretrain, project_root),
        "rl1": resolve_record_path(args.rl1, project_root),
        "rl2": resolve_record_path(args.rl2, project_root),
    }
    for stage, path in records_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{stage} records.pkl not found: {path}")

    if args.out_root is None:
        cutoff_tag = f"{args.local_cutoff:g}".replace(".", "p")
        out_root = resolve_record_path(
            f"eval_outputs/voroif_protac_contact_{cutoff_tag}A_{args.local_contact_rule}_eval",
            project_root,
        )
    else:
        out_root = resolve_record_path(args.out_root, project_root)
    out_root.mkdir(parents=True, exist_ok=True)

    voroif_app = Path(args.voroif_app).expanduser().resolve()
    voroif_exec = voroif_app / "voronota-js-voroif-gnn-v2"
    if not voroif_app.is_dir():
        raise FileNotFoundError(f"VoroIF app directory not found: {voroif_app}")
    if not voroif_exec.is_file():
        raise FileNotFoundError(f"VoroIF executable not found: {voroif_exec}")

    print_header("CONFIGURATION")
    print(f"project_root : {project_root}")
    print(f"pretrain     : {records_paths['pretrain']}")
    print(f"rl1          : {records_paths['rl1']}")
    print(f"rl2          : {records_paths['rl2']}")
    print(f"out_root     : {out_root}")
    print(f"voroif_app   : {voroif_app}")
    print(f"processors   : {args.processors}")
    print(f"chunk_size   : {args.chunk_size}")
    print(f"protocol     : {PROTOCOL_NAME}")
    print(f"local_cutoff : {args.local_cutoff:g} A")
    print(f"contact_rule : {args.local_contact_rule}")
    print(f"ligand       : {args.ligand_resname}")
    print("VoroIF input : WHOLE protein ATOM/TER context; only HETATM is removed")
    print("localization : post-hoc from official results_per_contact")
    print(
        "selection    : strict_protac + DeepTernary-scored records"
        if args.max_strict_per_stage is None
        else (
            "selection    : smoke test, first "
            f"{args.max_strict_per_stage} "
            "DeepTernary-scored strict records/stage"
        )
    )
    print("main metric  : whole-context PROTAC-proximal VoroIF local pCAD (higher better)")

    summaries = []
    for stage in STAGES:
        _, summary = evaluate_stage(
            stage=stage,
            records_path=records_paths[stage],
            project_root=project_root,
            out_root=out_root,
            voroif_app=voroif_app,
            voroif_exec=voroif_exec,
            processors=args.processors,
            chunk_size=args.chunk_size,
            timeout_per_chunk=args.timeout_per_chunk,
            max_strict_per_stage=args.max_strict_per_stage,
            keep_work=args.keep_work,
            cutoff_A=args.local_cutoff,
            ligand_resname=args.ligand_resname,
            contact_rule=args.local_contact_rule,
        )
        summaries.append(summary)
        write_json(out_root / "stage_results" / f"{stage}_summary.json", summary)

    summary_df = pd.DataFrame([{k: v for k, v in s.items() if k != "status_counts"} for s in summaries])
    summary_csv = out_root / "voroif_three_stage_summary.csv"
    summary_json = out_root / "voroif_three_stage_summary.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    write_json(summary_json, summaries)

    print_header("FINAL SUMMARY")
    display_cols = [
        "stage",
        "generated",
        "strict",
        "deepternary_scored_strict",
        "selected_deepternary_scored_strict_for_run",
        "successfully_voroif_local_scored_deepternary_strict",
        "voroif_scored_rate_among_deepternary_scored_strict",
        "voroif_scored_rate_among_selected",
        "voroif_local_pcad_mean",
        "voroif_local_pcad_median",
        "voroif_local_pcad_std",
        "voroif_local_pcad_q95",
        "voroif_local_pcad_max",
        "voroif_local_interface_area_mean",
        "voroif_local_interface_residue_count_mean",
        "voroif_local_contact_count_mean",
        "voroif_whole_pcad_mean_on_local_scored",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print("")
    print(f"Summary CSV : {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print("")
    print("Candidate main metric: voroif_local_pcad_mean ↑")
    print(
        "For smoke tests, use "
        "voroif_scored_rate_among_selected "
        "as the VoroIF execution success rate."
    )
    print("Do not start the full run until pCAD dynamic range and local-interface diagnostics look reasonable.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
