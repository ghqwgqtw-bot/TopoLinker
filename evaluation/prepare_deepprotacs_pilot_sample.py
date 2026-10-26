#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare ONE strict generated sample for a real DeepPROTACs pilot.

Purpose
-------
This script does NOT run Smina or DeepPROTACs.

It reads one evaluation records.pkl, selects one STRICT generated PROTAC,
maps that exact record to the matching POI-warhead pair and E3-ligand pair
from docking_pairs_poi.csv / docking_pairs_e3.csv, and writes:

    pilot_sample.json
    pilot_sample.csv
    linker.smi
    required_pairs.txt
    generated_protac.sdf          (when pred_mol can be written)
    README.txt

The important rule is:
    POI pair + E3 pair + linker MUST come from the SAME generated record.

This prevents accidentally combining unrelated pilot pairs such as
poi_0001 and e3_0001 just because both were docked independently.

Default stage
-------------
RL2 / final:
    eval_outputs/test_rl_final/records.pkl

Strict flag
-----------
Priority in --strict-key auto mode:
    1. precomputed_strict
    2. strict_protac

The current TopoLinker evaluation workflow stores linker_smiles directly.
If linker_smiles is missing, the script can reconstruct the same linker-only
SMILES using pred_mol + atom_mask + linker_mask with RDKit
MolFragmentToSmiles, matching the evaluation-side extraction logic.

Pilot selection
---------------
Default: --selection prefer-ready

Among valid strict samples:
    1. prefer a sample for which BOTH pair-specific DeepPROTACs MOL2 sides
       are already present;
    2. otherwise prefer an already prepared E3 side;
    3. otherwise prefer an already prepared POI side;
    4. otherwise use the earliest valid strict record.

This is ONLY a pipeline pilot selection rule. It does not use degradation,
DeepTernary, or DeepPROTACs scores and must not be used as a scientific
performance selection criterion.

Examples
--------
Default RL2 pilot:
    python evaluation/prepare_deepprotacs_pilot_sample.py

Force one record:
    python evaluation/prepare_deepprotacs_pilot_sample.py --record-index 123

Use RL1:
    python evaluation/prepare_deepprotacs_pilot_sample.py --stage rl1

Use pretrain:
    python evaluation/prepare_deepprotacs_pilot_sample.py --stage pretrain

Filter UUID:
    python evaluation/prepare_deepprotacs_pilot_sample.py --uuid 19

Outputs default to:
    eval_outputs/deepprotacs_docking/deepprotacs_pilot_sample/
        <stage>_record_<index>/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from rdkit import Chem
except Exception as exc:
    raise RuntimeError(
        "RDKit is required. Please run this script in the linkerDiff environment."
    ) from exc


# =====================================================================
# Basic helpers
# =====================================================================

DEFAULT_RECORDS = {
    "pretrain": Path("eval_outputs/test_pretrain/records.pkl"),
    "rl1": Path("eval_outputs/test_rl/records.pkl"),
    "rl2": Path("eval_outputs/test_rl_final/records.pkl"),
}

MISSING_AF2_UNIPROTS = {
    "P03406",
    "P03436",
    "P36969",
}


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def get_field(rec: Any, key: str, default=None):
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


def scalar_value(x: Any, default=None):
    """
    Convert common Python / NumPy / Torch scalar-like values to Python scalar.
    """
    if x is None:
        return default

    try:
        # torch.Tensor without importing torch explicitly
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            x = x.detach().cpu()
            if hasattr(x, "numel") and x.numel() == 1:
                return x.item()
    except Exception:
        pass

    try:
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return x.reshape(-1)[0].item()
    except Exception:
        pass

    try:
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass

    return x


def as_bool(x: Any, default=False) -> bool:
    x = scalar_value(x, default)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "yes", "y", "t"}:
            return True
        if s in {"0", "false", "no", "n", "f", "", "none", "nan"}:
            return False
    try:
        return bool(x)
    except Exception:
        return bool(default)


def to_list(x: Any) -> List[Any]:
    if x is None:
        return []

    try:
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()

    if isinstance(x, (list, tuple)):
        return list(x)

    try:
        return list(x)
    except Exception:
        return [x]


def load_records(path: Path) -> List[Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, tuple):
        return list(obj)

    if isinstance(obj, dict):
        for key in ["records", "samples", "data", "items"]:
            if key in obj and isinstance(obj[key], (list, tuple)):
                return list(obj[key])

        # dict itself may be index -> record
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())

    raise TypeError(
        f"Cannot recognize top-level structure of records file: {type(obj)}"
    )


def resolve_path(value: Any, project_root: Path) -> Path:
    s = safe_str(value).strip()
    if not s:
        return Path("")

    p = Path(s)
    if p.is_absolute():
        return p

    return (project_root / p).resolve()


def normalized_path(path: Any, project_root: Path) -> str:
    p = resolve_path(path, project_root)
    if not safe_str(p):
        return ""

    try:
        # normcase matters on Windows and is harmless elsewhere.
        return os.path.normcase(os.path.normpath(str(p.resolve())))
    except Exception:
        return os.path.normcase(os.path.normpath(str(p)))


def file_sha1(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""

    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def record_fragment_mols(rec: Any):
    frag_mol = get_field(rec, "frag_mol")

    if frag_mol is None:
        raise ValueError("record has no frag_mol")

    frags = Chem.GetMolFrags(
        frag_mol,
        asMols=True,
        sanitizeFrags=True,
    )

    if len(frags) != 2:
        raise ValueError(
            f"frag_mol should contain exactly 2 fragments, got {len(frags)}"
        )

    poi_lig = Chem.Mol(frags[0])
    e3_lig = Chem.Mol(frags[1])

    if poi_lig.GetNumConformers() == 0:
        raise ValueError("POI fragment has no 3D conformer")

    if e3_lig.GetNumConformers() == 0:
        raise ValueError("E3 fragment has no 3D conformer")

    return poi_lig, e3_lig


def canonical_smiles_from_mol(mol: Any) -> str:
    if mol is None:
        return ""

    try:
        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return ""


def record_structure_eligible(rec: Any) -> Tuple[bool, str]:
    poi_uniprot = safe_str(
        get_field(rec, "poi_uniprot")
    ).strip()

    e3_uniprot = safe_str(
        get_field(rec, "e3_uniprot")
    ).strip()

    missing = []

    if poi_uniprot in MISSING_AF2_UNIPROTS:
        missing.append(poi_uniprot)

    if e3_uniprot in MISSING_AF2_UNIPROTS:
        missing.append(e3_uniprot)

    if missing:
        return False, "missing_af2:" + "+".join(missing)

    return True, "ok"
# =====================================================================
# Molecule / linker helpers
# =====================================================================

def canonical_smiles_from_file(path: Path) -> str:
    """
    Match the canonicalization logic used by prepare_docking_pairs.py
    as closely as possible.
    """
    if not path.exists() or not path.is_file():
        return ""

    ext = path.suffix.lower()
    mol = None

    try:
        if ext in {".sdf", ".sd"}:
            suppl = Chem.SDMolSupplier(
                str(path),
                removeHs=False,
                sanitize=False,
            )
            mol = next((m for m in suppl if m is not None), None)

        elif ext == ".mol":
            mol = Chem.MolFromMolFile(
                str(path),
                removeHs=False,
                sanitize=False,
            )

        elif ext == ".mol2":
            mol = Chem.MolFromMol2File(
                str(path),
                removeHs=False,
                sanitize=False,
            )

        elif ext in {".smi", ".smiles", ".txt"}:
            txt = path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip().split()
            if txt:
                mol = Chem.MolFromSmiles(txt[0])

    except Exception:
        mol = None

    if mol is None:
        return ""

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Same fallback policy as docking-pair preparation.
        pass

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def mask_to_local_atom_ids(atom_mask_i: Any, sub_mask_i: Any) -> List[int]:
    """
    Convert padded model atom indices to RDKit-local atom indices.

    This matches the evaluation logic:
        sub_valid = sub_mask_i[atom_mask_i]
        np.where(sub_valid)[0]
    """
    atom_mask = np.asarray(to_list(atom_mask_i), dtype=bool).reshape(-1)
    sub_mask = np.asarray(to_list(sub_mask_i), dtype=bool).reshape(-1)

    if atom_mask.size == 0 or sub_mask.size == 0:
        return []

    if atom_mask.size != sub_mask.size:
        raise ValueError(
            f"atom_mask length ({atom_mask.size}) != "
            f"linker_mask length ({sub_mask.size})"
        )

    sub_valid = sub_mask[atom_mask]
    return np.where(sub_valid)[0].astype(int).tolist()


def extract_linker_smiles_from_record(rec: Any) -> Tuple[str, str]:
    """
    Returns:
        linker_smiles, source

    Preferred source:
        record['linker_smiles']

    Fallback:
        pred_mol + atom_mask + linker_mask -> MolFragmentToSmiles
    """
    existing = safe_str(get_field(rec, "linker_smiles")).strip()

    if existing and existing.lower() not in {"none", "nan"}:
        mol = Chem.MolFromSmiles(existing)
        if mol is not None:
            try:
                smi = Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                smi = existing
            return smi, "record.linker_smiles"

    mol = get_field(rec, "pred_mol")
    atom_mask = get_field(rec, "atom_mask")
    linker_mask = get_field(rec, "linker_mask")

    if mol is None or atom_mask is None or linker_mask is None:
        return "", "missing"

    try:
        atom_ids = mask_to_local_atom_ids(atom_mask, linker_mask)
    except Exception:
        return "", "mask_conversion_failed"

    if not atom_ids:
        return "", "empty_linker_mask"

    try:
        smi = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=atom_ids,
            canonical=True,
        )
        if smi:
            return smi, "reconstructed_from_pred_mol"
    except Exception:
        pass

    return "", "reconstruction_failed"


def validate_linker_smiles(smi: str) -> Tuple[bool, str]:
    if not smi:
        return False, "empty"

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False, "rdkit_parse_failed"

    # A strict PROTAC should have a connected linker subgraph.
    if "." in smi:
        return False, "disconnected_linker_smiles"

    if mol.GetNumHeavyAtoms() <= 0:
        return False, "zero_heavy_atoms"

    return True, "ok"


def write_pred_mol_sdf(mol: Any, out_path: Path) -> Tuple[bool, str]:
    if mol is None:
        return False, "pred_mol_missing"

    try:
        writer = Chem.SDWriter(str(out_path))
        writer.SetKekulize(False)
        writer.write(mol)
        writer.close()

        if out_path.exists() and out_path.stat().st_size > 0:
            return True, "ok"

        return False, "empty_output"
    except Exception as exc:
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        return False, f"{type(exc).__name__}: {exc}"


# =====================================================================
# Pair table / pair mapping
# =====================================================================

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def prepare_pair_rows(
    rows: List[Dict[str, str]],
    project_root: Path,
) -> List[Dict[str, Any]]:
    out = []

    for row in rows:
        r = dict(row)

        protein_path = resolve_path(
            r.get("protein_path", ""),
            project_root,
        )
        ligand_path = resolve_path(
            r.get("ligand_path", ""),
            project_root,
        )

        r["_protein_norm"] = normalized_path(
            protein_path,
            project_root,
        )
        r["_ligand_norm"] = normalized_path(
            ligand_path,
            project_root,
        )

        r["_protein_sha1"] = file_sha1(protein_path)

        smi = safe_str(
            r.get("ligand_canonical_smiles", "")
        ).strip()

        if not smi:
            smi = canonical_smiles_from_file(ligand_path)

        r["_ligand_smiles"] = smi

        out.append(r)

    return out


def match_pair(
    *,
    rec: Any,
    side: str,
    pair_rows: List[Dict[str, Any]],
    project_root: Path,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Match one generated record to its docking pair using:

        POI: poi_uniprot + frag_mol fragment1
        E3 : e3_uniprot  + frag_mol fragment2

    Experimental p1_path / p2_path and
    unbound_lig1_path / unbound_lig2_path are not used.
    """
    side = side.upper()

    poi_lig_mol, e3_lig_mol = record_fragment_mols(rec)

    if side == "POI":
        uniprot = safe_str(
            get_field(rec, "poi_uniprot")
        ).strip()
        ligand_mol = poi_lig_mol

    elif side == "E3":
        uniprot = safe_str(
            get_field(rec, "e3_uniprot")
        ).strip()
        ligand_mol = e3_lig_mol

    else:
        raise ValueError(side)

    ligand_smiles = canonical_smiles_from_mol(
        ligand_mol
    )

    if not ligand_smiles:
        return None, "fragment_smiles_failed"

    matches = [
        r
        for r in pair_rows
        if (
            safe_str(
                r.get("protein_uniprot")
            ).strip() == uniprot
            and safe_str(
                r.get("_ligand_smiles")
            ).strip() == ligand_smiles
        )
    ]

    if len(matches) == 1:
        return matches[0], "uniprot_fragmol_smiles"

    if len(matches) > 1:
        ids = [
            r.get("pair_id", "")
            for r in matches
        ]
        raise RuntimeError(
            f"{side}: ambiguous UniProt+fragment match: {ids}"
        )

    return None, "not_found"


# =====================================================================
# Strict / readiness
# =====================================================================

def strict_flag(rec: Any, strict_key: str) -> Tuple[bool, str]:
    if strict_key != "auto":
        value = get_field(rec, strict_key, None)
        if value is None:
            return False, f"missing:{strict_key}"
        return as_bool(value), strict_key

    for key in ["precomputed_strict", "strict_protac"]:
        value = get_field(rec, key, None)
        if value is not None:
            return as_bool(value), key

    return False, "missing"


def pair_input_paths(
    pair_id: str,
    side: str,
    deep_inputs_root: Path,
) -> Dict[str, Path]:
    d = deep_inputs_root / pair_id

    if side.upper() == "POI":
        return {
            "ligand": d / "target_ligand.mol2",
            "pocket": d / "target_pocket.mol2",
        }

    if side.upper() == "E3":
        return {
            "ligand": d / "ligase_ligand.mol2",
            "pocket": d / "ligase_pocket.mol2",
        }

    raise ValueError(side)


def side_ready(
    pair_id: str,
    side: str,
    deep_inputs_root: Path,
) -> Tuple[bool, Dict[str, Path]]:
    paths = pair_input_paths(
        pair_id,
        side,
        deep_inputs_root,
    )

    ready = all(
        p.exists() and p.stat().st_size > 0
        for p in paths.values()
    )

    return ready, paths


# =====================================================================
# Candidate generation / selection
# =====================================================================

def candidate_from_record(
    *,
    rec: Any,
    record_index: int,
    strict_key: str,
    poi_rows: List[Dict[str, Any]],
    e3_rows: List[Dict[str, Any]],
    project_root: Path,
    deep_inputs_root: Path,
) -> Tuple[Optional[Dict[str, Any]], str]:

    is_strict, strict_source = strict_flag(
        rec,
        strict_key,
    )

    if not is_strict:
        return None, "not_strict"

    structure_eligible, structure_reason = (
        record_structure_eligible(rec)
    )

    if not structure_eligible:
        return None, structure_reason

    linker_smiles, linker_source = (
        extract_linker_smiles_from_record(rec)
    )

    linker_ok, linker_status = validate_linker_smiles(
        linker_smiles
    )

    if not linker_ok:
        return None, f"bad_linker:{linker_status}"

    try:
        poi_pair, poi_match = match_pair(
            rec=rec,
            side="POI",
            pair_rows=poi_rows,
            project_root=project_root,
        )

        e3_pair, e3_match = match_pair(
            rec=rec,
            side="E3",
            pair_rows=e3_rows,
            project_root=project_root,
        )
    except Exception as exc:
        return None, f"pair_match_error:{exc}"

    if poi_pair is None:
        return None, "poi_pair_not_found"

    if e3_pair is None:
        return None, "e3_pair_not_found"

    poi_pair_id = safe_str(
        poi_pair.get("pair_id")
    ).strip()

    e3_pair_id = safe_str(
        e3_pair.get("pair_id")
    ).strip()

    poi_ready, poi_input_paths = side_ready(
        poi_pair_id,
        "POI",
        deep_inputs_root,
    )

    e3_ready, e3_input_paths = side_ready(
        e3_pair_id,
        "E3",
        deep_inputs_root,
    )

    # Pilot reuse priority:
    # both ready > E3 ready > POI ready > neither
    if poi_ready and e3_ready:
        ready_priority = 3
    elif e3_ready:
        ready_priority = 2
    elif poi_ready:
        ready_priority = 1
    else:
        ready_priority = 0

    uuid = safe_str(get_field(rec, "uuid")).strip()
    sample_id = safe_str(get_field(rec, "sample_id")).strip()

    return {
        "record_index": int(record_index),
        "uuid": uuid,
        "sample_id": sample_id,
        "strict_source": strict_source,
        "linker_smiles": linker_smiles,
        "linker_source": linker_source,
        "poi_pair": poi_pair,
        "e3_pair": e3_pair,
        "poi_pair_id": poi_pair_id,
        "e3_pair_id": e3_pair_id,
        "poi_match_method": poi_match,
        "e3_match_method": e3_match,
        "poi_ready": bool(poi_ready),
        "e3_ready": bool(e3_ready),
        "poi_input_paths": poi_input_paths,
        "e3_input_paths": e3_input_paths,
        "ready_priority": int(ready_priority),
        "record": rec,
    }, "ok"


def select_candidate(
    candidates: List[Dict[str, Any]],
    selection: str,
) -> Dict[str, Any]:
    if not candidates:
        raise RuntimeError(
            "No eligible strict sample could be prepared."
        )

    if selection == "first":
        return min(
            candidates,
            key=lambda x: x["record_index"],
        )

    if selection == "prefer-ready":
        # Higher ready priority first; deterministic earliest record tie-break.
        return sorted(
            candidates,
            key=lambda x: (
                -x["ready_priority"],
                x["record_index"],
            ),
        )[0]

    raise ValueError(selection)


# =====================================================================
# Output
# =====================================================================

def serialize_pair_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in row.items()
        if not k.startswith("_")
    }


def json_safe(value: Any):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            safe_str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    value = scalar_value(value, value)

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    return safe_str(value)


def write_outputs(
    *,
    chosen: Dict[str, Any],
    stage: str,
    records_path: Path,
    out_root: Path,
    deep_inputs_root: Path,
    selection: str,
) -> Path:

    record_index = chosen["record_index"]

    out_dir = (
        out_root
        / f"{stage}_record_{record_index:04d}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    linker_path = out_dir / "linker.smi"
    linker_path.write_text(
        chosen["linker_smiles"].strip() + "\n",
        encoding="utf-8",
    )

    poi_paths = chosen["poi_input_paths"]
    e3_paths = chosen["e3_input_paths"]

    manifest = {
        "purpose": "DeepPROTACs pipeline pilot; not a performance-selected scientific result",
        "stage": stage,
        "records_path": str(records_path.resolve()),
        "record_index": record_index,
        "uuid": chosen["uuid"],
        "sample_id": chosen["sample_id"],
        "selection_rule": selection,
        "strict": True,
        "strict_source": chosen["strict_source"],
        "linker_smiles": chosen["linker_smiles"],
        "linker_source": chosen["linker_source"],
        "linker_smi_path": str(linker_path.resolve()),
        "poi": {
            "pair_id": chosen["poi_pair_id"],
            "match_method": chosen["poi_match_method"],
            "deep_inputs_ready": chosen["poi_ready"],
            "pair_row": serialize_pair_row(
                chosen["poi_pair"]
            ),
            "target_ligand_mol2": str(
                poi_paths["ligand"].resolve()
            ),
            "target_pocket_mol2": str(
                poi_paths["pocket"].resolve()
            ),
        },
        "e3": {
            "pair_id": chosen["e3_pair_id"],
            "match_method": chosen["e3_match_method"],
            "deep_inputs_ready": chosen["e3_ready"],
            "pair_row": serialize_pair_row(
                chosen["e3_pair"]
            ),
            "ligase_ligand_mol2": str(
                e3_paths["ligand"].resolve()
            ),
            "ligase_pocket_mol2": str(
                e3_paths["pocket"].resolve()
            ),
        },
        "all_four_binary_mol2_ready": bool(
            chosen["poi_ready"]
            and chosen["e3_ready"]
        ),
    }

    json_path = out_dir / "pilot_sample.json"
    json_path.write_text(
        json.dumps(
            json_safe(manifest),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csv_path = out_dir / "pilot_sample.csv"

    csv_row = {
        "stage": stage,
        "record_index": record_index,
        "uuid": chosen["uuid"],
        "sample_id": chosen["sample_id"],
        "strict_source": chosen["strict_source"],
        "linker_smiles": chosen["linker_smiles"],
        "linker_source": chosen["linker_source"],
        "poi_pair_id": chosen["poi_pair_id"],
        "e3_pair_id": chosen["e3_pair_id"],
        "poi_ready": int(chosen["poi_ready"]),
        "e3_ready": int(chosen["e3_ready"]),
        "all_four_binary_mol2_ready": int(
            chosen["poi_ready"]
            and chosen["e3_ready"]
        ),
        "target_ligand_mol2": str(
            poi_paths["ligand"].resolve()
        ),
        "target_pocket_mol2": str(
            poi_paths["pocket"].resolve()
        ),
        "ligase_ligand_mol2": str(
            e3_paths["ligand"].resolve()
        ),
        "ligase_pocket_mol2": str(
            e3_paths["pocket"].resolve()
        ),
        "linker_smi": str(linker_path.resolve()),
    }

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(csv_row.keys()),
        )
        writer.writeheader()
        writer.writerow(csv_row)

    required_pairs_path = (
        out_dir / "required_pairs.txt"
    )

    required_pairs_lines = [
        "This DeepPROTACs pilot uses ONE internally consistent generated record.",
        "",
        f"stage:        {stage}",
        f"record_index: {record_index}",
        f"uuid:         {chosen['uuid']}",
        f"sample_id:    {chosen['sample_id']}",
        "",
        "Required binary docking pairs:",
        (
            f"POI: {chosen['poi_pair_id']} | "
            f"ready={chosen['poi_ready']}"
        ),
        (
            f"E3:  {chosen['e3_pair_id']} | "
            f"ready={chosen['e3_ready']}"
        ),
        "",
        "DeepPROTACs inputs:",
        f"target_ligand: {poi_paths['ligand'].resolve()}",
        f"target_pocket: {poi_paths['pocket'].resolve()}",
        f"ligase_ligand: {e3_paths['ligand'].resolve()}",
        f"ligase_pocket: {e3_paths['pocket'].resolve()}",
        f"linker:        {linker_path.resolve()}",
        "",
        (
            "ALL INPUTS READY: "
            + str(
                chosen["poi_ready"]
                and chosen["e3_ready"]
            )
        ),
    ]

    required_pairs_path.write_text(
        "\n".join(required_pairs_lines) + "\n",
        encoding="utf-8",
    )

    # Save generated PROTAC only as a traceable reference.
    pred_mol = get_field(
        chosen["record"],
        "pred_mol",
    )

    pred_sdf_path = (
        out_dir / "generated_protac.sdf"
    )

    sdf_ok, sdf_status = write_pred_mol_sdf(
        pred_mol,
        pred_sdf_path,
    )

    readme_lines = [
        "DeepPROTACs pilot sample",
        "========================",
        "",
        (
            "This directory was prepared from ONE strict generated record. "
            "The POI pair, E3 pair and linker all belong to that same record."
        ),
        "",
        (
            "Selection is for pipeline validation only; it does not use "
            "degradation, DeepTernary or DeepPROTACs scores."
        ),
        "",
        f"stage: {stage}",
        f"record_index: {record_index}",
        f"uuid: {chosen['uuid']}",
        f"sample_id: {chosen['sample_id']}",
        f"strict_source: {chosen['strict_source']}",
        "",
        f"POI pair: {chosen['poi_pair_id']}",
        f"POI binary MOL2 ready: {chosen['poi_ready']}",
        f"E3 pair: {chosen['e3_pair_id']}",
        f"E3 binary MOL2 ready: {chosen['e3_ready']}",
        "",
        f"linker_smiles: {chosen['linker_smiles']}",
        f"linker_source: {chosen['linker_source']}",
        "",
        f"generated_protac.sdf written: {sdf_ok}",
        f"generated_protac.sdf status: {sdf_status}",
        "",
        "Files:",
        "  pilot_sample.json      machine-readable full manifest",
        "  pilot_sample.csv       one-row summary",
        "  linker.smi             exact linker SMILES for this record",
        "  required_pairs.txt     pair IDs and expected MOL2 paths",
        "  generated_protac.sdf   generated full molecule, when writable",
    ]

    (out_dir / "README.txt").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )

    return out_dir


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        choices=["pretrain", "rl1", "rl2"],
        default="rl2",
        help="Evaluation stage from which to choose the pilot sample",
    )

    parser.add_argument(
        "--records",
        default=None,
        help=(
            "Optional explicit records.pkl path. "
            "Overrides the default path associated with --stage."
        ),
    )

    parser.add_argument(
        "--poi-pairs",
        default=(
            "eval_outputs/deepprotacs_docking/"
            "docking_pairs_poi.csv"
        ),
    )

    parser.add_argument(
        "--e3-pairs",
        default=(
            "eval_outputs/deepprotacs_docking/"
            "docking_pairs_e3.csv"
        ),
    )

    parser.add_argument(
        "--deep-inputs-root",
        default=(
            "eval_outputs/deepprotacs_docking/"
            "deepprotacs_inputs"
        ),
        help=(
            "Root containing pair-specific target_* or ligase_* MOL2 files"
        ),
    )

    parser.add_argument(
        "--out-root",
        default=(
            "eval_outputs/deepprotacs_docking/"
            "deepprotacs_pilot_sample"
        ),
    )

    parser.add_argument(
        "--project-root",
        default=".",
    )

    parser.add_argument(
        "--strict-key",
        default="auto",
        help=(
            "Strict flag key. 'auto' prefers precomputed_strict, "
            "then strict_protac."
        ),
    )

    parser.add_argument(
        "--selection",
        choices=["prefer-ready", "first"],
        default="prefer-ready",
        help=(
            "Pilot-only selection rule. prefer-ready reuses already prepared "
            "binary MOL2 sides when possible."
        ),
    )

    parser.add_argument(
        "--record-index",
        type=int,
        default=None,
        help="Force one exact record index after loading records.pkl",
    )

    parser.add_argument(
        "--uuid",
        default=None,
        help="Optional UUID filter",
    )

    parser.add_argument(
        "--sample-id",
        default=None,
        help="Optional sample_id filter",
    )

    args = parser.parse_args()

    project_root = Path(
        args.project_root
    ).resolve()

    if args.records:
        records_path = resolve_path(
            args.records,
            project_root,
        )
    else:
        records_path = resolve_path(
            DEFAULT_RECORDS[args.stage],
            project_root,
        )

    poi_pairs_path = resolve_path(
        args.poi_pairs,
        project_root,
    )

    e3_pairs_path = resolve_path(
        args.e3_pairs,
        project_root,
    )

    deep_inputs_root = resolve_path(
        args.deep_inputs_root,
        project_root,
    )

    out_root = resolve_path(
        args.out_root,
        project_root,
    )

    for p, label in [
        (records_path, "records"),
        (poi_pairs_path, "POI pairs CSV"),
        (e3_pairs_path, "E3 pairs CSV"),
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"{label} does not exist: {p}"
            )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("Prepare DeepPROTACs pilot sample")
    print(f"stage:             {args.stage}")
    print(f"records:           {records_path}")
    print(f"POI pairs:         {poi_pairs_path}")
    print(f"E3 pairs:          {e3_pairs_path}")
    print(f"deep inputs root:  {deep_inputs_root}")
    print(f"strict key:        {args.strict_key}")
    print(f"selection:         {args.selection}")
    print(f"record index:      {args.record_index}")
    print(f"uuid filter:       {args.uuid}")
    print(f"sample_id filter:  {args.sample_id}")
    print("=" * 80)

    records = load_records(records_path)

    poi_rows = prepare_pair_rows(
        read_csv_rows(poi_pairs_path),
        project_root,
    )

    e3_rows = prepare_pair_rows(
        read_csv_rows(e3_pairs_path),
        project_root,
    )

    print("")
    print("Loaded:")
    print(f"  records:   {len(records)}")
    print(f"  POI pairs: {len(poi_rows)}")
    print(f"  E3 pairs:  {len(e3_rows)}")

    # ---------------------------------------------------------------
    # Candidate indices
    # ---------------------------------------------------------------
    if args.record_index is not None:
        if not (
            0 <= args.record_index < len(records)
        ):
            raise IndexError(
                f"--record-index {args.record_index} "
                f"is outside [0, {len(records)-1}]"
            )

        indices = [args.record_index]
    else:
        indices = list(range(len(records)))

    if args.uuid is not None:
        indices = [
            i for i in indices
            if safe_str(
                get_field(records[i], "uuid")
            ).strip() == safe_str(args.uuid).strip()
        ]

    if args.sample_id is not None:
        indices = [
            i for i in indices
            if safe_str(
                get_field(
                    records[i],
                    "sample_id",
                )
            ).strip()
            == safe_str(args.sample_id).strip()
        ]

    if not indices:
        raise RuntimeError(
            "No records remain after the requested filters."
        )

    # ---------------------------------------------------------------
    # Build eligible strict candidates.
    # ---------------------------------------------------------------
    candidates = []
    reject_counts: Dict[str, int] = {}

    for i in indices:
        candidate, status = candidate_from_record(
            rec=records[i],
            record_index=i,
            strict_key=args.strict_key,
            poi_rows=poi_rows,
            e3_rows=e3_rows,
            project_root=project_root,
            deep_inputs_root=deep_inputs_root,
        )

        if candidate is None:
            reject_counts[status] = (
                reject_counts.get(status, 0) + 1
            )
            continue

        candidates.append(candidate)

    print("")
    print("Candidate scan:")
    print(f"  records considered:     {len(indices)}")
    print(f"  eligible strict sample: {len(candidates)}")

    if reject_counts:
        print("  rejected:")
        for key, n in sorted(
            reject_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            print(f"    {key}: {n}")

    if not candidates:
        raise RuntimeError(
            "No eligible strict sample found. "
            "Check strict flags, linker_smiles and pair mapping."
        )

    chosen = select_candidate(
        candidates,
        args.selection,
    )

    # ---------------------------------------------------------------
    # Summarize binary-input readiness among eligible candidates.
    # ---------------------------------------------------------------
    n_both = sum(
        c["poi_ready"] and c["e3_ready"]
        for c in candidates
    )

    n_e3_only = sum(
        (not c["poi_ready"]) and c["e3_ready"]
        for c in candidates
    )

    n_poi_only = sum(
        c["poi_ready"] and (not c["e3_ready"])
        for c in candidates
    )

    n_none = sum(
        (not c["poi_ready"]) and (not c["e3_ready"])
        for c in candidates
    )

    print("")
    print("Existing binary-input coverage among eligible strict samples:")
    print(f"  both sides ready: {n_both}")
    print(f"  E3 only ready:    {n_e3_only}")
    print(f"  POI only ready:   {n_poi_only}")
    print(f"  neither ready:    {n_none}")

    out_dir = write_outputs(
        chosen=chosen,
        stage=args.stage,
        records_path=records_path,
        out_root=out_root,
        deep_inputs_root=deep_inputs_root,
        selection=args.selection,
    )

    print("")
    print("========== Selected pilot sample ==========")
    print(f"stage:          {args.stage}")
    print(f"record_index:   {chosen['record_index']}")
    print(f"uuid:           {chosen['uuid']}")
    print(f"sample_id:      {chosen['sample_id']}")
    print(f"strict source:  {chosen['strict_source']}")
    print(f"linker source:  {chosen['linker_source']}")
    print(f"linker SMILES:  {chosen['linker_smiles']}")

    print("")
    print("Required matching pairs:")
    print(
        f"  POI: {chosen['poi_pair_id']} "
        f"| ready={chosen['poi_ready']} "
        f"| match={chosen['poi_match_method']}"
    )
    print(
        f"  E3:  {chosen['e3_pair_id']} "
        f"| ready={chosen['e3_ready']} "
        f"| match={chosen['e3_match_method']}"
    )

    print("")
    if (
        chosen["poi_ready"]
        and chosen["e3_ready"]
    ):
        print(
            "ALL FOUR binary MOL2 inputs are already ready."
        )
        print(
            "This sample is ready for a DeepPROTACs inference pilot."
        )
    else:
        missing = []
        if not chosen["poi_ready"]:
            missing.append(
                f"POI {chosen['poi_pair_id']}"
            )
        if not chosen["e3_ready"]:
            missing.append(
                f"E3 {chosen['e3_pair_id']}"
            )

        print(
            "Additional binary docking/pocket preparation is required for:"
        )
        for x in missing:
            print(f"  - {x}")

    print("")
    print("Outputs:")
    print(f"  {out_dir / 'pilot_sample.json'}")
    print(f"  {out_dir / 'pilot_sample.csv'}")
    print(f"  {out_dir / 'linker.smi'}")
    print(f"  {out_dir / 'required_pairs.txt'}")
    print(f"  {out_dir / 'generated_protac.sdf'}")
    print(f"  {out_dir / 'README.txt'}")
    print("")
    print("Finished successfully.")


if __name__ == "__main__":
    main()
