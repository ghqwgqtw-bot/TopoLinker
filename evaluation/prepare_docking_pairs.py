#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare shared Smina docking pairs for three TopoLinker evaluation stages.

Workflow:
    1. Extract unique POI-warhead and E3-ligand docking pairs from
       the Pretrain, RL1, and RL2 records.
    2. Check whether the three stages use identical fixed test contexts.
    3. Generate one shared set of docking-pair inputs when the
       consistency checks pass.

Default records:
    Pretrain: eval_outputs/test_pretrain/records.pkl
    RL1:      eval_outputs/test_rl/records.pkl
    RL2:      eval_outputs/test_rl_final/records.pkl

Protein and fragment mapping:
    poi_uniprot -> datasets/PDBs/AF-{poi_uniprot}-F1-model_v6.pdb
    e3_uniprot  -> datasets/PDBs/AF-{e3_uniprot}-F1-model_v6.pdb
    frag_mol fragment1 -> POI warhead
    frag_mol fragment2 -> E3 ligand

Structural eligibility:
    Test contexts containing UniProt IDs P03406, P03436, or P36969
    are excluded from DeepPROTACs external evaluation because the
    required AlphaFold2 structures are unavailable in this workflow.

Outputs:
    docking_pairs_poi.csv
    docking_pairs_e3.csv
    docking_pairs_all.csv
    docking_consistency_report.txt
    docking_pairs_summary.txt

Notes:
    - This script prepares docking-pair inputs but does not run Smina.
    - A single docking set is shared across all three stages when
      their fixed test conditions are consistent.
    - The --allow-mismatch option bypasses the consistency requirement
      and is not recommended for formal evaluation.
"""

import argparse
import csv
import hashlib
import os
import pickle
from collections import defaultdict
from pathlib import Path

try:
    from rdkit import Chem
except ImportError:
    Chem = None


STAGES = ("pretrain", "rl1", "rl2")

MISSING_AF2_UNIPROTS = {
    "P03406",
    "P03436",
    "P36969",
}

def safe_str(x):
    return "" if x is None else str(x)


def resolve_path(path_str, project_root):
    """
    Resolve relative paths against the project root while
    preserving absolute paths.
    """
    path_str = safe_str(path_str).strip()
    if not path_str:
        return Path("")
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def file_sha1(path):
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_smiles_from_file(path):
    """
    Extract canonical SMILES from common small-molecule file formats.

    Use canonical SMILES to identify ligands independently of their
    molecular coordinates. Return an empty string if extraction fails.
    """
    if Chem is None:
        return ""

    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""

    ext = p.suffix.lower()
    mol = None

    try:
        if ext in [".sdf", ".sd"]:
            suppl = Chem.SDMolSupplier(str(p), removeHs=False, sanitize=False)
            mol = next((m for m in suppl if m is not None), None)
        elif ext == ".mol":
            mol = Chem.MolFromMolFile(str(p), removeHs=False, sanitize=False)
        elif ext == ".mol2":
            mol = Chem.MolFromMol2File(str(p), removeHs=False, sanitize=False)
        elif ext in [".smi", ".smiles", ".txt"]:
            txt = p.read_text(encoding="utf-8", errors="ignore").strip().split()
            if txt:
                mol = Chem.MolFromSmiles(txt[0])
    except Exception:
        mol = None

    if mol is None:
        return ""

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Attempt SMILES generation even if sanitization fails.
        pass

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def load_records(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        for key in ["records", "samples", "data", "items"]:
            if key in obj and isinstance(obj[key], (list, tuple)):
                return list(obj[key])

        # Support dictionaries mapping record indices to record objects.
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())

    raise TypeError(
        f"Unsupported top-level records.pkl structure: {type(obj)}. "
        "Inspect the object type and available keys."
    )


def get_field(rec, key):
    if isinstance(rec, dict):
        return rec.get(key)
    return getattr(rec, key, None)

def af2_protein_path(uniprot, project_root):
    uniprot = safe_str(uniprot).strip()
    if not uniprot:
        raise ValueError("Empty UniProt ID")

    path = (
        project_root
        / "datasets"
        / "PDBs"
        / f"AF-{uniprot}-F1-model_v6.pdb"
    ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"AlphaFold2 structure missing unexpectedly: "
            f"{uniprot} -> {path}"
        )

    return path


def record_fragment_mols(rec):
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

    # Dataset convention:
    # fragment1 -> POI warhead
    # fragment2 -> E3 ligand
    poi_lig = Chem.Mol(frags[0])
    e3_lig = Chem.Mol(frags[1])

    if poi_lig.GetNumConformers() == 0:
        raise ValueError("POI fragment has no 3D conformer")

    if e3_lig.GetNumConformers() == 0:
        raise ValueError("E3 fragment has no 3D conformer")

    return poi_lig, e3_lig


def canonical_smiles_from_mol(mol):
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

def protein_key(uniprot, protein_path):
    """
    Generate a protein identity key for deduplication and consistency checks.

    Prefer the UniProt ID combined with the protein file's SHA1 hash.
    Fall back to the UniProt ID and absolute path if the file hash
    is unavailable.
    """
    uniprot = safe_str(uniprot).strip()
    protein_path = Path(protein_path)

    sha1 = file_sha1(protein_path)
    if sha1:
        return f"UNIPROT::{uniprot}::SHA1::{sha1}" if uniprot else f"SHA1::{sha1}"

    abs_path = str(protein_path.resolve()) if str(protein_path) else ""
    return f"UNIPROT::{uniprot}::PATH::{abs_path}" if uniprot else f"PATH::{abs_path}"


def ligand_key(ligand_path):
    """
    Generate a ligand identity key using the following priority:

    1. Canonical SMILES, independent of molecular coordinates.
    2. SHA1 hash of the ligand file.
    3. Absolute file path as the final fallback.
    """
    ligand_path = Path(ligand_path)

    smi = canonical_smiles_from_file(ligand_path)
    if smi:
        return f"SMILES::{smi}", smi

    sha1 = file_sha1(ligand_path)
    if sha1:
        return f"SHA1::{sha1}", ""

    abs_path = str(ligand_path.resolve()) if str(ligand_path) else ""
    return f"PATH::{abs_path}", ""


def stage_records_to_signatures(records, stage, project_root):
    """
    Extract unique docking pairs and test-context signatures
    from the records of one evaluation stage.

    Returns:
        poi_pairs: Dictionary of unique POI-warhead docking pairs.
        e3_pairs: Dictionary of unique E3-ligand docking pairs.
        contexts: Set of paired POI/E3 test-context signatures.
    """
    poi_pairs = {}
    e3_pairs = {}
    contexts = set()

    for idx, rec in enumerate(records):
        # ---------- UniProt ----------
        poi_uniprot = safe_str(
            get_field(rec, "poi_uniprot")
        ).strip()

        e3_uniprot = safe_str(
            get_field(rec, "e3_uniprot")
        ).strip()

        # Exclude test contexts containing UniProt IDs in MISSING_AF2_UNIPROTS.
        # These contexts are excluded from DeepPROTACs external evaluation.
        if (
                poi_uniprot in MISSING_AF2_UNIPROTS
                or e3_uniprot in MISSING_AF2_UNIPROTS
        ):
            continue

        # ---------- AlphaFold2 protein ----------
        p1_path = af2_protein_path(
            poi_uniprot,
            project_root,
        )

        p2_path = af2_protein_path(
            e3_uniprot,
            project_root,
        )

        # ---------- PROTAC terminal ligands ----------
        poi_lig_mol, e3_lig_mol = record_fragment_mols(rec)

        poi_lig_smi = canonical_smiles_from_mol(
            poi_lig_mol
        )
        e3_lig_smi = canonical_smiles_from_mol(
            e3_lig_mol
        )

        if not poi_lig_smi:
            raise RuntimeError(
                f"Cannot build POI ligand SMILES: record={idx}"
            )

        if not e3_lig_smi:
            raise RuntimeError(
                f"Cannot build E3 ligand SMILES: record={idx}"
            )

        # ---------- POI pair ----------
        p1_key = protein_key(
            poi_uniprot,
            p1_path,
        )

        lig1_key = f"SMILES::{poi_lig_smi}"
        poi_key = (p1_key, lig1_key)

        # ---------- E3 pair ----------
        p2_key = protein_key(
            e3_uniprot,
            p2_path,
        )

        lig2_key = f"SMILES::{e3_lig_smi}"
        e3_key = (p2_key, lig2_key)

        # Define each test context by its fixed POI and E3 binding pairs.
        contexts.add((poi_key, e3_key))

        uuid = safe_str(get_field(rec, "uuid"))
        sample_id = safe_str(get_field(rec, "sample_id"))

        if poi_key not in poi_pairs:
            poi_pairs[poi_key] = {
                "side": "POI",
                "protein_uniprot": poi_uniprot,
                "protein_path": str(p1_path),
                "ligand_path": "",
                "ligand_canonical_smiles": poi_lig_smi,
                "_ligand_mol": poi_lig_mol,
                "example_uuid": uuid,
                "example_sample_id": sample_id,
                "first_record_index": idx,
                "stage_record_counts": defaultdict(int),
                "stage_uuids": defaultdict(set),
            }

        poi_pairs[poi_key]["stage_record_counts"][stage] += 1
        if uuid:
            poi_pairs[poi_key]["stage_uuids"][stage].add(uuid)

        if e3_key not in e3_pairs:
            e3_pairs[e3_key] = {
                "side": "E3",
                "protein_uniprot": e3_uniprot,
                "protein_path": str(p2_path),
                "ligand_path": "",
                "ligand_canonical_smiles": e3_lig_smi,
                "_ligand_mol": e3_lig_mol,
                "example_uuid": uuid,
                "example_sample_id": sample_id,
                "first_record_index": idx,
                "stage_record_counts": defaultdict(int),
                "stage_uuids": defaultdict(set),
            }

        e3_pairs[e3_key]["stage_record_counts"][stage] += 1
        if uuid:
            e3_pairs[e3_key]["stage_uuids"][stage].add(uuid)

    return poi_pairs, e3_pairs, contexts


def compare_sets(name, ref_set, other_set, ref_stage, other_stage):
    missing = ref_set - other_set
    extra = other_set - ref_set
    ok = (len(missing) == 0 and len(extra) == 0)

    lines = [
        f"[{name}] {ref_stage} vs {other_stage}",
        f"  {ref_stage}: {len(ref_set)}",
        f"  {other_stage}: {len(other_set)}",
        f"  missing in {other_stage}: {len(missing)}",
        f"  extra in {other_stage}: {len(extra)}",
        f"  result: {'MATCH' if ok else 'MISMATCH'}",
    ]
    return ok, lines


def merge_pair_counts(base_pairs, stage_pairs, stage):
    """
    Merge record counts and UUID sets from another stage into
    the reference stage's docking-pair dictionary.
    """
    for key, row in base_pairs.items():
        if key in stage_pairs:
            row["stage_record_counts"][stage] = stage_pairs[key]["stage_record_counts"][stage]
            row["stage_uuids"][stage] = stage_pairs[key]["stage_uuids"][stage]


def write_csv(pair_dict, path):
    ligand_dir = path.parent / "docking_ligands"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id",
        "side",
        "protein_uniprot",
        "protein_path",
        "ligand_path",
        "ligand_canonical_smiles",
        "pretrain_record_count",
        "rl1_record_count",
        "rl2_record_count",
        "pretrain_unique_uuid_count",
        "rl1_unique_uuid_count",
        "rl2_unique_uuid_count",
        "example_uuid",
        "example_sample_id",
        "first_record_index",
    ]

    rows = list(pair_dict.values())
    rows.sort(
        key=lambda r: (
            r["side"],
            r["protein_uniprot"],
            r["ligand_canonical_smiles"],
            r["ligand_path"],
        )
    )

    counters = {"POI": 0, "E3": 0}

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            counters[row["side"]] += 1

            pair_id = (
                f"{row['side'].lower()}_"
                f"{counters[row['side']]:04d}"
            )

            ligand_path = (
                    ligand_dir / f"{pair_id}.sdf"
            ).resolve()

            ligand_mol = row["_ligand_mol"]

            writer_sdf = Chem.SDWriter(
                str(ligand_path)
            )
            writer_sdf.write(ligand_mol)
            writer_sdf.close()

            if (
                    not ligand_path.is_file()
                    or ligand_path.stat().st_size <= 0
            ):
                raise RuntimeError(
                    f"Failed to write ligand SDF: "
                    f"{ligand_path}"
                )
            out = {
                "pair_id": pair_id,
                "side": row["side"],
                "protein_uniprot": row["protein_uniprot"],
                "protein_path": row["protein_path"],
                "ligand_path": str(ligand_path),
                "ligand_canonical_smiles": row["ligand_canonical_smiles"],
                "pretrain_record_count": row["stage_record_counts"].get("pretrain", 0),
                "rl1_record_count": row["stage_record_counts"].get("rl1", 0),
                "rl2_record_count": row["stage_record_counts"].get("rl2", 0),
                "pretrain_unique_uuid_count": len(row["stage_uuids"].get("pretrain", set())),
                "rl1_unique_uuid_count": len(row["stage_uuids"].get("rl1", set())),
                "rl2_unique_uuid_count": len(row["stage_uuids"].get("rl2", set())),
                "example_uuid": row["example_uuid"],
                "example_sample_id": row["example_sample_id"],
                "first_record_index": row["first_record_index"],
            }
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrain",
        default="eval_outputs/test_pretrain/records.pkl",
        help="Pretrain records.pkl",
    )
    parser.add_argument(
        "--rl1",
        default="eval_outputs/test_rl/records.pkl",
        help="RL stage 1 records.pkl",
    )
    parser.add_argument(
        "--rl2",
        default="eval_outputs/test_rl_final/records.pkl",
        help="RL stage 2/final records.pkl",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve relative paths in records.pkl",
    )
    parser.add_argument(
        "--out-dir",
        default="eval_outputs/deepprotacs_docking",
        help="Output directory for docking-pair tables and ligand files",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help=(
            "Continue generating docking pairs even if the three stages differ. "
            "Not recommended for formal evaluation."
        ),
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_paths = {
        "pretrain": resolve_path(args.pretrain, project_root),
        "rl1": resolve_path(args.rl1, project_root),
        "rl2": resolve_path(args.rl2, project_root),
    }

    stage_data = {}

    print("========== Load records from three stages ==========")
    for stage in STAGES:
        path = stage_paths[stage]
        if not path.exists():
            raise FileNotFoundError(f"{stage} records file not found: {path}")

        records = load_records(path)
        poi_pairs, e3_pairs, contexts = stage_records_to_signatures(
            records, stage, project_root
        )

        stage_data[stage] = {
            "records": records,
            "poi_pairs": poi_pairs,
            "e3_pairs": e3_pairs,
            "contexts": contexts,
        }

        print(
            f"{stage:8s}: records={len(records)}, "
            f"POI-pairs={len(poi_pairs)}, "
            f"E3-pairs={len(e3_pairs)}, "
            f"contexts={len(contexts)}"
        )

    # ---------- Check consistency across the three stages ----------
    report_lines = []
    all_ok = True

    ref = "pretrain"
    for other in ["rl1", "rl2"]:
        ok, lines = compare_sets(
            "POI-warhead pairs",
            set(stage_data[ref]["poi_pairs"].keys()),
            set(stage_data[other]["poi_pairs"].keys()),
            ref,
            other,
        )
        all_ok &= ok
        report_lines.extend(lines)
        report_lines.append("")

        ok, lines = compare_sets(
            "E3-ligand pairs",
            set(stage_data[ref]["e3_pairs"].keys()),
            set(stage_data[other]["e3_pairs"].keys()),
            ref,
            other,
        )
        all_ok &= ok
        report_lines.extend(lines)
        report_lines.append("")

        ok, lines = compare_sets(
            "docking context signatures",
            stage_data[ref]["contexts"],
            stage_data[other]["contexts"],
            ref,
            other,
        )
        all_ok &= ok
        report_lines.extend(lines)
        report_lines.append("")

    report_lines.append(
        f"FINAL CONSISTENCY: "
        f"{'PASS (all three stages match)' if all_ok else 'FAIL (stage mismatch detected)'}"
    )

    consistency_path = out_dir / "docking_consistency_report.txt"
    consistency_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n========== Consistency check ==========")
    print("\n".join(report_lines))

    if not all_ok and not args.allow_mismatch:
        raise RuntimeError(
            "\nTest conditions differ across the three stages; "
            "docking-pair generation has been stopped.\n"
            f"Detailed report: {consistency_path}\n"
            "Inspect the differences before proceeding. "
            "Do not use --allow-mismatch for formal evaluation."
        )

    # ---------- Use pretrain as the reference for shared docking pairs ----------
    poi_pairs = stage_data["pretrain"]["poi_pairs"]
    e3_pairs = stage_data["pretrain"]["e3_pairs"]

    merge_pair_counts(poi_pairs, stage_data["rl1"]["poi_pairs"], "rl1")
    merge_pair_counts(poi_pairs, stage_data["rl2"]["poi_pairs"], "rl2")
    merge_pair_counts(e3_pairs, stage_data["rl1"]["e3_pairs"], "rl1")
    merge_pair_counts(e3_pairs, stage_data["rl2"]["e3_pairs"], "rl2")

    all_pairs = {}
    for k, v in poi_pairs.items():
        all_pairs[("POI",) + k] = v
    for k, v in e3_pairs.items():
        all_pairs[("E3",) + k] = v

    write_csv(poi_pairs, out_dir / "docking_pairs_poi.csv")
    write_csv(e3_pairs, out_dir / "docking_pairs_e3.csv")
    write_csv(all_pairs, out_dir / "docking_pairs_all.csv")

    unique_poi_proteins = len(
        set(
            row["protein_uniprot"] or row["protein_path"]
            for row in poi_pairs.values()
        )
    )
    unique_e3_proteins = len(
        set(
            row["protein_uniprot"] or row["protein_path"]
            for row in e3_pairs.values()
        )
    )

    summary = (
        f"consistency: {'PASS' if all_ok else 'FAIL'}\n"
        f"pretrain records: {len(stage_data['pretrain']['records'])}\n"
        f"rl1 records: {len(stage_data['rl1']['records'])}\n"
        f"rl2 records: {len(stage_data['rl2']['records'])}\n"
        f"unique docking context signatures: {len(stage_data['pretrain']['contexts'])}\n"
        f"unique POI proteins: {unique_poi_proteins}\n"
        f"unique E3 proteins: {unique_e3_proteins}\n"
        f"unique POI-warhead pairs: {len(poi_pairs)}\n"
        f"unique E3-ligand pairs: {len(e3_pairs)}\n"
        f"total docking pairs: {len(poi_pairs) + len(e3_pairs)}\n"
        f"\n"
        "A single set of docking pairs is shared across all three stages.\n"
    )

    summary_path = out_dir / "docking_pairs_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print("\n========== Final summary ==========")
    print(summary)
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
