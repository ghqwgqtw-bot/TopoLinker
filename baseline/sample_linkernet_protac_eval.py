#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LinkerNet sampling and unified PROTAC evaluation; fixed fragments and known anchors.

Uses model-only converted PT for sampling, source PT for reconstruction/evaluation.
Never infers linker bonds from distances/OpenBabel and never edits the shared builder.
"""
from __future__ import annotations
import argparse
import csv
import copy
import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors, Crippen, Lipinski, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Batch
from datasets.linker_data import FragLinkerData
from models.diff_protac_bond import DiffPROTACModel
import utils.transforms as trans
from utils.geometry import find_axes, global_to_local
from protac_eval_metrics import ProtacMetricAccumulator, build_reward_models_from_conf, is_strict_protac_mol
from protac_molecule_builder import (build_molecules as protac_build_molecules,
                                     infer_generated_atom_props_from_current_graph_inplace)
from convert_protac_to_linkernet import ATOM_TYPES

def str2bool(value):
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_load_cpu(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def safe_torch_load_state_dict(path: str | Path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and not any(
        str(k).startswith("dynamics.") or str(k).startswith("gamma.") for k in obj.keys()
    ):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint must contain a state_dict-like mapping: {path}")
    return obj


def load_smiles_list(path: Optional[str]) -> Optional[List[str]]:
    if path is None or str(path).strip().lower() in {"", "none", "null"}:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(line.split()[0])
    return out

def safe_mol_to_smiles(mol):
    if mol is None: return None
    try:
        mol_copy = Chem.Mol(mol); Chem.SanitizeMol(mol_copy); smi = Chem.MolToSmiles(mol_copy); return smi if smi else None
    except Exception:
        return None
def make_pickle_safe_records(records):
    safe = []
    for rec in records:
        item = {}
        for k, v in rec.items():
            item[k] = v.detach().cpu() if torch.is_tensor(v) else v
        safe.append(item)
    return safe


def save_records(records, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "records.pkl"
    with path.open("wb") as f:
        pickle.dump(make_pickle_safe_records(records), f)
    return path


def load_records(outdir: str | Path):
    path = Path(outdir) / "records.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        records = pickle.load(f)
    if not isinstance(records, list):
        raise TypeError(f"{path}: expected list, got {type(records)}")
    return records


def save_metrics(metrics: Dict[str, Any], outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "test_metrics.pkl").open("wb") as f:
        pickle.dump(metrics, f)

    with (outdir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with (outdir / "test_metrics.txt").open("w", encoding="utf-8") as f:
        for k in sorted(metrics):
            f.write(f"{k}: {metrics[k]}\n")

def tensor_to_float(x, default=float("nan")):
    try:
        if x is None:
            return default
        if torch.is_tensor(x):
            if x.numel() == 0:
                return default
            return float(x.detach().cpu().view(-1)[0].item())
        return float(x)
    except Exception:
        return default

def get_batch_item(batch, key, i, default=None):
    if key not in batch:
        return default

    v = batch[key]

    if isinstance(v, (list, tuple)):
        if i < len(v):
            return v[i]
        return default

    if torch.is_tensor(v):
        try:
            vi = v[i]
            if vi.numel() == 1:
                return vi.detach().cpu().item()
            return vi.detach().cpu().numpy().tolist()
        except Exception:
            return default

    return v

def cleanup_deepternary_temp_dirs(deepternary_info):
    """
    Remove temporary DeepTernary sample directories after scoring.

    No PDB files are copied or retained.
    This function only performs filesystem cleanup and does not use RNG.
    """
    if deepternary_info is None:
        return

    if "deepternary_complex_pred_path" not in deepternary_info:
        return

    complex_paths = deepternary_info["deepternary_complex_pred_path"]

    if not isinstance(complex_paths, (list, tuple)):
        return

    removed_dirs = set()

    for src_path in complex_paths:
        if src_path in [None, "", "None", "nan"]:
            continue

        src_path = os.path.abspath(str(src_path))

        sample_dir = os.path.dirname(
            os.path.dirname(src_path)
        )

        if sample_dir in removed_dirs:
            continue

        if os.path.isdir(sample_dir):
            shutil.rmtree(
                sample_dir,
                ignore_errors=True,
            )

        removed_dirs.add(sample_dir)

def mask_to_local_atom_ids(atom_mask_i, sub_mask_i):
    """
    Convert padded atom masks to local RDKit atom indices.

    The reconstructed molecule contains only atoms selected by atom_mask_i.
    MolFragmentToSmiles therefore requires indices in the compact atom order.
    """
    if torch.is_tensor(atom_mask_i):
        atom_mask_i = atom_mask_i.detach().cpu().numpy()
    if torch.is_tensor(sub_mask_i):
        sub_mask_i = sub_mask_i.detach().cpu().numpy()

    atom_mask_i = np.asarray(atom_mask_i).squeeze().astype(bool)
    sub_mask_i = np.asarray(sub_mask_i).squeeze().astype(bool)

    sub_valid = sub_mask_i[atom_mask_i]
    return np.where(sub_valid)[0].astype(int).tolist()

def mol_fragment_smiles(mol, atom_mask_i, linker_mask_i):
    if mol is None:
        return None

    atom_ids = mask_to_local_atom_ids(atom_mask_i, linker_mask_i)

    if len(atom_ids) == 0:
        return None

    try:
        smi = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=atom_ids,
            canonical=True,
        )
        if smi is None or len(smi) == 0:
            return None
        return smi
    except Exception:
        return None

def safe_scaffold_smiles(mol):
    if mol is None:
        return None

    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaf, canonical=True)
        if smi is None or len(smi) == 0:
            return None
        return smi
    except Exception:
        return None

def safe_morgan_bitstring(mol, radius=2, n_bits=2048):
    if mol is None:
        return None

    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=n_bits,
        )
        return fp.ToBitString()
    except Exception:
        return None

def calc_sa_score_safe(mol):
    """
    Calculate the synthetic accessibility (SA) score using RDKit Contrib.
    Return NaN if the scorer is unavailable or the calculation fails.
    """
    if mol is None:
        return float("nan")

    try:
        from rdkit.Chem import RDConfig
        import sys
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        return float("nan")

def calc_basic_sample_metrics(mol, atom_mask_i, linker_mask_i):
    out = {
        "generated_smiles": None,
        "linker_smiles": None,

        "qed": float("nan"),
        "sa_score": float("nan"),
        "mol_weight": float("nan"),
        "logp": float("nan"),
        "tpsa": float("nan"),
        "hbd": float("nan"),
        "hba": float("nan"),
        "rotatable_bonds": float("nan"),
        "num_rings": float("nan"),

        "linker_heavy_atoms": float("nan"),
        "linker_rotatable_bonds": float("nan"),
        "linker_num_rings": float("nan"),

        "mol_scaffold": None,
        "linker_scaffold": None,
        "mol_morgan_fp": None,
        "linker_morgan_fp": None,
    }

    if mol is None:
        return out

    smi = safe_mol_to_smiles(mol)
    out["generated_smiles"] = smi

    if smi is None:
        return out

    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)

        out["qed"] = float(QED.qed(mol_copy))
        out["sa_score"] = calc_sa_score_safe(mol_copy)
        out["mol_weight"] = float(Descriptors.MolWt(mol_copy))
        out["logp"] = float(Crippen.MolLogP(mol_copy))
        out["tpsa"] = float(rdMolDescriptors.CalcTPSA(mol_copy))
        out["hbd"] = float(Lipinski.NumHDonors(mol_copy))
        out["hba"] = float(Lipinski.NumHAcceptors(mol_copy))
        out["rotatable_bonds"] = float(Lipinski.NumRotatableBonds(mol_copy))
        out["num_rings"] = float(rdMolDescriptors.CalcNumRings(mol_copy))
        out["mol_scaffold"] = safe_scaffold_smiles(mol_copy)
        out["mol_morgan_fp"] = safe_morgan_bitstring(mol_copy)

        linker_smi = mol_fragment_smiles(
            mol_copy,
            atom_mask_i=atom_mask_i,
            linker_mask_i=linker_mask_i,
        )
        out["linker_smiles"] = linker_smi

        if linker_smi is not None:
            linker_mol = Chem.MolFromSmiles(linker_smi)
            if linker_mol is not None:
                out["linker_heavy_atoms"] = float(linker_mol.GetNumHeavyAtoms())
                out["linker_rotatable_bonds"] = float(
                    Lipinski.NumRotatableBonds(linker_mol)
                )
                out["linker_num_rings"] = float(
                    rdMolDescriptors.CalcNumRings(linker_mol)
                )
                out["linker_scaffold"] = safe_scaffold_smiles(linker_mol)
                out["linker_morgan_fp"] = safe_morgan_bitstring(linker_mol)

    except Exception:
        pass

    return out

def make_eval_records_for_batch(
        *,
        pred_molecules,
        batch,
        batch_idx,
        sample_id,
        model_name,
        checkpoint_path,
        degradation_scores=None,
        ternary_scores=None,
        deepternary_info=None,
        strict_flags=None,
        strict_infos=None,
        degradation_threshold=0.5,
):
    records = []

    B = len(pred_molecules)

    for i, mol in enumerate(pred_molecules):
        uuid = get_batch_item(batch, "uuid", i, default=f"batch{batch_idx}_idx{i}")
        poi = get_batch_item(batch, "poi_uniprot", i, default=None)
        e3 = get_batch_item(batch, "e3_uniprot", i, default=None)

        if strict_flags is not None:
            strict = bool(strict_flags[i])

            if strict_infos is not None:
                strict_info = strict_infos[i]
            else:
                strict_info = {}

        else:
            strict, strict_info = is_strict_protac_mol(
                mol=mol,
                anchor1_mask=batch["anchor1_mask"][i],
                anchor2_mask=batch["anchor2_mask"][i],
                linker_mask=batch["linker_mask"][i],
                fragment1_mask=batch["fragment1_mask"][i],
                fragment2_mask=batch["fragment2_mask"][i],
            )

        basic = calc_basic_sample_metrics(
            mol=mol,
            atom_mask_i=batch["atom_mask"][i],
            linker_mask_i=batch["linker_mask"][i],
        )

        raw_deg = tensor_to_float(
            degradation_scores[i] if degradation_scores is not None else None,
            default=0.0,
        )

        gated_deg = raw_deg * float(strict)

        ternary = tensor_to_float(
            ternary_scores[i] if ternary_scores is not None else None,
            default=0.0,
        )

        p2_rmsd = -1.0
        clash_ratio = -1.0

        deepternary_scored = 0.0
        deepternary_attempted = 0.0
        deepternary_valid_candidate = 0.0

        if deepternary_info is not None:
            if "deepternary_pred_p2_rmsd_raw" in deepternary_info:
                p2 = deepternary_info["deepternary_pred_p2_rmsd_raw"]

                if (
                        torch.is_tensor(p2)
                        and p2.ndim > 0
                        and p2.numel() > i
                ):
                    p2_rmsd = tensor_to_float(
                        p2.reshape(-1)[i],
                        default=-1.0,
                    )

            if "deepternary_clash_ratio_raw" in deepternary_info:
                clash = deepternary_info["deepternary_clash_ratio_raw"]

                if (
                        torch.is_tensor(clash)
                        and clash.ndim > 0
                        and clash.numel() > i
                ):
                    clash_ratio = tensor_to_float(
                        clash.reshape(-1)[i],
                        default=-1.0,
                    )

            if "deepternary_scored_mask" in deepternary_info:
                m = deepternary_info["deepternary_scored_mask"]
                if (
                        torch.is_tensor(m)
                        and m.ndim > 0
                        and m.numel() > i
                ):
                    deepternary_scored = tensor_to_float(
                        m.reshape(-1)[i],
                        default=0.0,
                    )

            if "deepternary_attempted_mask" in deepternary_info:
                m = deepternary_info["deepternary_attempted_mask"]
                if (
                        torch.is_tensor(m)
                        and m.ndim > 0
                        and m.numel() > i
                ):
                    deepternary_attempted = tensor_to_float(
                        m.reshape(-1)[i],
                        default=0.0,
                    )

            if "deepternary_valid_candidate_mask" in deepternary_info:
                m = deepternary_info["deepternary_valid_candidate_mask"]
                if (
                        torch.is_tensor(m)
                        and m.ndim > 0
                        and m.numel() > i
                ):
                    deepternary_valid_candidate = tensor_to_float(
                        m.reshape(-1)[i],
                        default=0.0,
                    )

        record = {
            "model_name": model_name,
            "checkpoint": checkpoint_path,
            "batch_idx": batch_idx,
            "sample_id": sample_id,
            "case_id": uuid,
            "uuid": uuid,
            "poi_uniprot": poi,
            "e3_uniprot": e3,

            "strict_protac": int(strict),
            "raw_degradation_score": raw_deg,
            "gated_degradation_score": gated_deg,
            "high_degradation": int(strict and raw_deg >= degradation_threshold),

            "ternary_score": ternary,
            "deepternary_pred_p2_rmsd": p2_rmsd,
            "deepternary_clash_ratio": clash_ratio,

            "deepternary_scored": int(deepternary_scored > 0.5),
            "deepternary_attempted": int(deepternary_attempted > 0.5),
            "deepternary_valid_candidate": int(deepternary_valid_candidate > 0.5),
        }

        for k, v in strict_info.items():
            record[f"strict_{k}"] = v

        record.update(basic)

        records.append(record)

    return records

def save_eval_records(records, save_dir, filename="eval_samples.csv"):
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, filename)

    if len(records) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return path

    fieldnames = sorted(records[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    return path

def save_diversity_summary(records, save_dir):
    strict_records = [
        r for r in records
        if int(r.get("strict_protac", 0)) == 1
    ]

    high_deg_records = [
        r for r in strict_records
        if int(r.get("high_degradation", 0)) == 1
    ]

    def nonempty_set(key, rows):
        return set(
            r.get(key)
            for r in rows
            if r.get(key) not in [None, "", "None"]
        )

    summary = {
        "num_total_samples": len(records),
        "num_strict_samples": len(strict_records),
        "num_high_deg_samples": len(high_deg_records),

        "num_unique_mol_smiles_strict": len(
            nonempty_set("generated_smiles", strict_records)
        ),
        "num_unique_linker_smiles_strict": len(
            nonempty_set("linker_smiles", strict_records)
        ),
        "num_unique_mol_scaffold_strict": len(
            nonempty_set("mol_scaffold", strict_records)
        ),
        "num_unique_linker_scaffold_strict": len(
            nonempty_set("linker_scaffold", strict_records)
        ),

        "num_unique_linker_smiles_high_deg": len(
            nonempty_set("linker_smiles", high_deg_records)
        ),
        "num_unique_linker_scaffold_high_deg": len(
            nonempty_set("linker_scaffold", high_deg_records)
        ),
    }

    if len(records) > 0:
        summary["strict_ratio"] = summary["num_strict_samples"] / len(records)
        summary["high_deg_global_ratio"] = summary["num_high_deg_samples"] / len(records)
    else:
        summary["strict_ratio"] = 0.0
        summary["high_deg_global_ratio"] = 0.0

    if len(strict_records) > 0:
        summary["high_deg_among_strict_ratio"] = (
            summary["num_high_deg_samples"] / len(strict_records)
        )
        summary["unique_linker_ratio_strict"] = (
            summary["num_unique_linker_smiles_strict"] / len(strict_records)
        )
    else:
        summary["high_deg_among_strict_ratio"] = 0.0
        summary["unique_linker_ratio_strict"] = 0.0

    path_json = os.path.join(save_dir, "diversity_summary.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    path_txt = os.path.join(save_dir, "diversity_summary.txt")
    with open(path_txt, "w", encoding="utf-8") as f:
        for k in sorted(summary.keys()):
            f.write(f"{k}: {summary[k]}\n")

    return summary
def load_mapping(
    mapping_csv: str | Path,
    converted_records: Sequence[dict],
    source_records: Sequence[dict],
) -> List[Dict[str, Any]]:
    """
    Return rows sorted by output_index and audit:
      converted[output_index] <-> mapping UUID <-> source[original_index].
    """
    mapping_csv = Path(mapping_csv)
    if not mapping_csv.exists():
        raise FileNotFoundError(mapping_csv)

    rows = []
    with mapping_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"output_index", "original_index", "uuid"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"{mapping_csv}: missing columns {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "output_index": int(row["output_index"]),
                    "original_index": int(row["original_index"]),
                    "uuid": str(row["uuid"]),
                    "name": str(row.get("name", "")),
                }
            )

    rows.sort(key=lambda r: r["output_index"])

    if len(rows) != len(converted_records):
        raise ValueError(
            f"mapping rows {len(rows)} != converted records {len(converted_records)}"
        )

    expected_output = list(range(len(converted_records)))
    got_output = [r["output_index"] for r in rows]
    if got_output != expected_output:
        raise ValueError("mapping output_index is not exactly 0..N-1")

    for row in rows:
        oi = row["output_index"]
        si = row["original_index"]
        if si < 0 or si >= len(source_records):
            raise IndexError(
                f"mapping output_index={oi}: original_index={si} outside source range"
            )

        converted_uuid = str(converted_records[oi]["uuid"])
        source_uuid = str(source_records[si]["uuid"])
        mapping_uuid = row["uuid"]

        if converted_uuid != mapping_uuid or source_uuid != mapping_uuid:
            raise ValueError(
                f"UUID mismatch at output_index={oi}: "
                f"converted={converted_uuid}, mapping={mapping_uuid}, source={source_uuid}"
            )

    return rows


def count_skipped_rows(skipped_csv: Optional[str | Path]) -> int:
    if skipped_csv is None:
        return 0
    p = Path(skipped_csv)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def build_fragment_template_with_placeholders(src: dict) -> Chem.Mol:
    """
    Build the fixed-fragment template using the shared molecule builder.
    Linker slots stay as dummy C atoms only to preserve original indexing.
    """
    required = [
        "one_hot",
        "positions",
        "fragment1_mask",
        "fragment2_mask",
        "linker_mask",
        "fragment_bond_types",
        "formal_charges",
        "atom_is_aromatic",
        "num_explicit_hs",
        "no_implicit",
    ]
    missing = [k for k in required if k not in src]
    if missing:
        raise KeyError(f"Missing source fragment-template keys: {missing}")

    one_hot = torch.as_tensor(src["one_hot"]).detach().cpu().float()
    pos = torch.as_tensor(src["positions"]).detach().cpu().float()
    f1 = torch.as_tensor(src["fragment1_mask"]).detach().cpu().bool().view(-1)
    f2 = torch.as_tensor(src["fragment2_mask"]).detach().cpu().bool().view(-1)
    linker = torch.as_tensor(src["linker_mask"]).detach().cpu().bool().view(-1)
    frag = f1 | f2

    n = int(one_hot.shape[0])
    if pos.shape != (n, 3):
        raise ValueError(f"source positions shape {tuple(pos.shape)} != ({n},3)")
    if frag.numel() != n or linker.numel() != n:
        raise ValueError("source masks do not match atom count")
    if bool((frag & linker).any()) or not bool((frag | linker).all()):
        raise ValueError("source fragment/linker masks are invalid")

    frag_bonds = torch.as_tensor(src["fragment_bond_types"]).detach().cpu().long()
    if frag_bonds.shape != (n, n):
        raise ValueError(
            f"fragment_bond_types shape {tuple(frag_bonds.shape)} != ({n},{n})"
        )

    # Keep only f1-f1 / f2-f2 bonds.
    pair_mask = (
        (f1[:, None] & f1[None, :])
        | (f2[:, None] & f2[None, :])
    )
    frag_bonds = frag_bonds * pair_mask.to(frag_bonds.dtype)

    template_h = torch.zeros_like(one_hot)
    template_h[:, 0] = 1.0  # dummy C in every slot
    template_h[frag] = one_hot[frag]

    template_x = torch.zeros_like(pos)
    template_x[frag] = pos[frag]

    formal = torch.zeros(n, dtype=torch.long)
    aromatic = torch.zeros(n, dtype=torch.bool)
    explicit_h = torch.zeros(n, dtype=torch.long)
    no_impl = torch.zeros(n, dtype=torch.bool)

    src_formal = torch.as_tensor(src["formal_charges"]).detach().cpu().long().view(-1)
    src_aromatic = torch.as_tensor(src["atom_is_aromatic"]).detach().cpu().bool().view(-1)
    src_explicit_h = torch.as_tensor(src["num_explicit_hs"]).detach().cpu().long().view(-1)
    src_no_impl = torch.as_tensor(src["no_implicit"]).detach().cpu().bool().view(-1)

    formal[frag] = src_formal[frag]
    aromatic[frag] = src_aromatic[frag]
    explicit_h[frag] = src_explicit_h[frag]
    no_impl[frag] = src_no_impl[frag]

    node_mask = torch.ones(1, n, 1, dtype=torch.float32)

    mols = protac_build_molecules(
        one_hot=template_h.unsqueeze(0),
        x=template_x.unsqueeze(0),
        node_mask=node_mask,
        fragment_bond_types=frag_bonds.unsqueeze(0),
        use_template_fragment_bonds=True,
        formal_charges=formal.unsqueeze(0),
        atom_is_aromatic=aromatic.unsqueeze(0),
        num_explicit_hs=explicit_h.unsqueeze(0),
        no_implicit=no_impl.unsqueeze(0),
        atom_props_mask=frag.unsqueeze(0),
        use_pred_bonds=False,
    )
    if len(mols) != 1 or mols[0] is None:
        raise RuntimeError("Failed to build fragment template")
    if mols[0].GetNumAtoms() != n:
        raise RuntimeError("Fragment template atom count changed")
    return mols[0]


def build_source_reference_molecules(src: dict) -> Tuple[Chem.Mol, Chem.Mol]:
    """
    Build full GT molecule and GT fixed-fragment molecule from the original record.
    """
    required = [
        "one_hot",
        "positions",
        "full_bond_types",
        "fragment1_mask",
        "fragment2_mask",
        "formal_charges",
        "atom_is_aromatic",
        "num_explicit_hs",
        "no_implicit",
    ]
    missing = [k for k in required if k not in src]
    if missing:
        raise KeyError(f"Missing source GT-reference keys: {missing}")

    h = torch.as_tensor(src["one_hot"]).detach().cpu().float()
    x = torch.as_tensor(src["positions"]).detach().cpu().float()
    full_bonds = torch.as_tensor(src["full_bond_types"]).detach().cpu().long()
    f1 = torch.as_tensor(src["fragment1_mask"]).detach().cpu().bool().view(-1)
    f2 = torch.as_tensor(src["fragment2_mask"]).detach().cpu().bool().view(-1)
    frag = f1 | f2
    n = int(h.shape[0])

    formal = torch.as_tensor(src["formal_charges"]).detach().cpu().long().view(-1)
    aromatic = torch.as_tensor(src["atom_is_aromatic"]).detach().cpu().bool().view(-1)
    explicit_h = torch.as_tensor(src["num_explicit_hs"]).detach().cpu().long().view(-1)
    no_impl = torch.as_tensor(src["no_implicit"]).detach().cpu().bool().view(-1)

    if x.shape != (n, 3) or full_bonds.shape != (n, n):
        raise ValueError("Invalid GT source shapes")

    full_mask = torch.ones(1, n, 1, dtype=torch.float32)
    frag_mask = frag.float().view(1, n, 1)

    true_mol = protac_build_molecules(
        one_hot=h.unsqueeze(0),
        x=x.unsqueeze(0),
        node_mask=full_mask,
        fragment_bond_types=full_bonds.unsqueeze(0),
        use_template_fragment_bonds=True,
        formal_charges=formal.unsqueeze(0),
        atom_is_aromatic=aromatic.unsqueeze(0),
        num_explicit_hs=explicit_h.unsqueeze(0),
        no_implicit=no_impl.unsqueeze(0),
        atom_props_mask=None,
        use_pred_bonds=False,
    )[0]

    frag_mol = protac_build_molecules(
        one_hot=h.unsqueeze(0),
        x=x.unsqueeze(0),
        node_mask=frag_mask,
        fragment_bond_types=full_bonds.unsqueeze(0),
        use_template_fragment_bonds=True,
        formal_charges=formal.unsqueeze(0),
        atom_is_aromatic=aromatic.unsqueeze(0),
        num_explicit_hs=explicit_h.unsqueeze(0),
        no_implicit=no_impl.unsqueeze(0),
        atom_props_mask=frag.unsqueeze(0),
        use_pred_bonds=False,
    )[0]

    if true_mol is None or frag_mol is None:
        raise RuntimeError("Failed to build source GT reference molecule(s)")
    return true_mol, frag_mol

def build_model(args, device):
    ckpt = torch_load_cpu(args.model_path)
    if not isinstance(ckpt, dict) or 'model' not in ckpt or 'configs' not in ckpt:
        raise ValueError('Expected official LinkerNet checkpoint {model,configs,...}')
    cfg = ckpt['configs'].model
    if not cfg.train_link or not cfg.train_bond:
        raise ValueError('Checkpoint must support linker atom and bond generation')
    if cfg.known_linker_bond:
        raise ValueError('known_linker_bond=True checkpoint cannot be used for native predicted linker-linker bonds')
    atom_featurizer = trans.FeaturizeAtom('protac', known_anchor=cfg.known_anchor,
                                         add_atom_type=False, add_atom_feat=True)
    graph_builder = trans.BuildCompleteGraph(known_linker_bond=False, known_cand_anchors=True)
    model = DiffPROTACModel(cfg, num_classes=atom_featurizer.num_classes,
                            num_bond_classes=graph_builder.num_bond_classes,
                            atom_feature_dim=atom_featurizer.feature_dim,
                            edge_feature_dim=graph_builder.bond_feature_dim).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model, cfg, atom_featurizer, graph_builder


def make_model_data(converted, model_cfg, atom_featurizer, graph_builder, max_frag_atoms):
    """Build native FragLinkerData *without* GT linker information."""
    p = torch.as_tensor(converted['pos'],dtype=torch.float32).clone()
    fmask = torch.as_tensor(converted['fragment_mask'],dtype=torch.long).clone()
    atom_type = torch.as_tensor(converted['atom_type'],dtype=torch.long).clone()
    anchor_mask = torch.as_tensor(converted['anchor_mask'],dtype=torch.long).clone()
    n = len(fmask)
    f1 = (fmask==1).nonzero(as_tuple=True)[0]
    f2 = (fmask==2).nonzero(as_tuple=True)[0]
    l = (fmask==0).nonzero(as_tuple=True)[0]
    if len(f1)>max_frag_atoms or len(f2)>max_frag_atoms:
        raise ValueError('max_frag_atoms too small for converted input')
    if int(anchor_mask.sum()) != 2 or int(anchor_mask[f1].sum()) != 1 or int(anchor_mask[f2].sum()) != 1:
        raise ValueError('expected exact two known anchors')
    # Entire linker is dummy in the sampler input.
    if not torch.all(atom_type[l] == ATOM_TYPES.index('C4(0)')) or torch.any(p[l] != 0):
        raise RuntimeError('Converted sampling input contains non-placeholder linker labels/coordinates')
    b = torch.as_tensor(converted['fragment_bond_types'],dtype=torch.long)
    if b.shape != (n,n) or torch.any(b[l,:] != 0) or torch.any(b[:,l] != 0):
        raise RuntimeError('Converted sampling input contains hidden linker bonds')
    src, dst = (b != 0).nonzero(as_tuple=True)
    # Construct original positions for *fragments* only and local rigid frames.
    # Global orientation need not be canonicalized; the network is E(3) equivariant
    # and we restore output coordinates by the fragment atom rigid alignment.
    pts = p.numpy()
    Rs, ts, locals_ = [], [], []
    for ids in (f1,f2):
        xyz = pts[ids.numpy()]
        centers = xyz.mean(axis=0)
        z = np.array([Chem.GetPeriodicTable().GetAtomicNumber(
            trans.dataset_info('protac')['number_to_atom'][int(atom_type[int(i)])])
            for i in ids.tolist()],dtype=np.float32)
        axes = np.asarray(find_axes(xyz,z), dtype=np.float32)
        R = torch.from_numpy(axes)
        t = torch.from_numpy(centers.astype(np.float32))
        loc = global_to_local(R,t,p[ids])
        Rs.append(R);ts.append(t);locals_.append(loc.reshape(-1,3))
    data = FragLinkerData(pos=p, fragment_mask=fmask, linker_mask=fmask.eq(0),
                          atom_type=atom_type, anchor_mask=anchor_mask,
                          anchor_indices=anchor_mask.nonzero(as_tuple=True)[0].tolist(),
                          bond_index=torch.stack([src,dst],dim=0), bond_type=b[src,dst],
                          frags_R=torch.stack(Rs), frags_t=torch.stack(ts),
                          frags_local_pos=torch.cat(locals_,dim=0),
                          frags_d=(ts[1][0]-ts[0][0]).reshape(()),
                          uuid=str(converted['uuid']))
    data=atom_featurizer(data)
    data=trans.SelectCandAnchors(mode='exact')(data)
    data=graph_builder(data)
    data=trans.StackFragLocalPos(max_num_atoms=max_frag_atoms)(data)
    # Applying random-rotation augmentation at sampling is unnecessary; keep
    # the two fixed poses and original orientation. The model's relative mode
    # remains determined by checkpoint configuration.
    if str(model_cfg.get('rel_geometry','two_pos_and_rot')) not in ('two_pos_and_rot','distance_and_two_rot'):
        raise ValueError('This fixed-fragment sampler requires two_pos_and_rot-compatible checkpoint')
    return data


def restore_positions(final_xyz, converted, source):
    """Kabsch-align model's FINAL fixed fragment atoms to exact source XYZ;
    use the same rigid transform on generated linker coordinates.
    """
    final_xyz = torch.as_tensor(final_xyz,dtype=torch.float64).cpu().numpy()
    order = np.asarray(converted['original_index_order'],dtype=np.int64)
    srcpos = torch.as_tensor(source['positions'],dtype=torch.float64).cpu().numpy()
    fmask = torch.as_tensor(converted['fragment_mask']).cpu().numpy()
    frag_ids = np.flatnonzero(fmask>0)
    x = final_xyz[frag_ids]
    y = srcpos[order[frag_ids]]
    xc,yc=x.mean(0),y.mean(0)
    U,S,Vt=np.linalg.svd((x-xc).T @ (y-yc))
    D=np.eye(3); D[-1,-1]=np.sign(np.linalg.det(U@Vt))
    R=U@D@Vt
    result=(final_xyz-xc)@R+yc
    rms=float(np.linalg.norm(result[frag_ids]-y,axis=1).max())
    if rms > 2e-3:
        raise RuntimeError(f'Fragment changed during sampling / alignment error: max={rms:.6f} A')
    # EXACT GT fragments, only linker positions are generated.
    result[frag_ids] = y
    return torch.from_numpy(result.astype(np.float32))


def pred_bond_matrix(data, directed_predictions):
    """Exact native directed edge alignment: no distance inference."""
    types=torch.as_tensor(directed_predictions).detach().cpu().long().view(-1)
    ei=data.edge_index.detach().cpu().long()
    n=int(data.num_nodes)
    if types.numel()!=ei.shape[1]:
        raise RuntimeError(f'Bond predictions {types.numel()} != directed edges {ei.shape[1]}')
    if (types<0).any() or (types>4).any():
        raise RuntimeError('LinkerNet predicted bond label not in 0..4')
    mat=torch.zeros((n,n),dtype=torch.long)
    mat[ei[0],ei[1]]=types
    if not torch.equal(mat,mat.T):
        raise RuntimeError('LinkerNet bond predictions are asymmetric')
    return mat


def fill_linkernet_linker_into_template(template_mol, final_x, final_c, final_bond,
                                        converted, source, model_data):
    """PocketXMol style prediction-bond transfer, preserving class formal charges."""
    if template_mol is None: return None
    try:
        order=list(converted['original_index_order'])
        n=len(order)
        if template_mol.GetNumAtoms()!=n:return None
        xyz=restore_positions(final_x,converted,source)
        c=torch.as_tensor(final_c).detach().cpu().long().view(-1)
        if c.numel()!=n:return None
        bonds=pred_bond_matrix(model_data,final_bond)
        native_link=torch.as_tensor(converted['linker_mask']).cpu().bool()
        source_link=torch.as_tensor(source['linker_mask']).cpu().bool()
        anchor=(torch.as_tensor(source['anchor1_mask']).cpu().bool() |
                torch.as_tensor(source['anchor2_mask']).cpu().bool())
        pt=Chem.GetPeriodicTable()
        rw=Chem.RWMol(Chem.Mol(template_mol))
        # Native id -> source id mapping. Source fragment graph unchanged.
        for native_i,source_i in enumerate(order):
            if not bool(native_link[native_i]):continue
            label=ATOM_TYPES[int(c[native_i])]
            atom=Chem.Atom(pt.GetAtomicNumber(label.split('(')[0][:-1] if label.split('(')[0][-1].isdigit() else label.split('(')[0]))
            # The official class string encodes valence and formal charge.
            charge=int(label.split('(')[-1].rstrip(')'))
            atom.SetFormalCharge(charge)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
            atom.SetIsAromatic(False)
            rw.ReplaceAtom(int(source_i),atom,preserveProps=False)
        conf=rw.GetConformer()
        for native_i,source_i in enumerate(order):
            if bool(native_link[native_i]):
                p=xyz[native_i]
                conf.SetAtomPosition(source_i,tuple(float(v) for v in p.tolist()))
        rdtypes={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,
                 3:Chem.BondType.TRIPLE,4:Chem.BondType.AROMATIC}
        for native_i in range(n):
            for native_j in range(native_i):
                si, sj = order[native_i], order[native_j]

                li = bool(source_link[si])
                lj = bool(source_link[sj])

                # Only allow linker-linker or anchor-linker bonds.
                is_linker_linker = li and lj
                is_anchor_linker = (
                        (li and bool(anchor[sj]))
                        or
                        (lj and bool(anchor[si]))
                )

                if not (is_linker_linker or is_anchor_linker):
                    continue

                b = int(bonds[native_i, native_j])

                # No predicted bond.
                if b == 0:
                    continue

                # Reject aromatic anchor-linker bonds.
                # Do not convert aromatic bonds into single bonds.
                if is_anchor_linker and b == 4:
                    continue

                if rw.GetBondBetweenAtoms(si, sj) is not None:
                    return None

                # Aromatic linker-linker bonds are still allowed.
                if b == 4:
                    rw.GetAtomWithIdx(si).SetIsAromatic(True)
                    rw.GetAtomWithIdx(sj).SetIsAromatic(True)

                rw.AddBond(si, sj, rdtypes[b])

                if b == 4:
                    rw.GetBondBetweenAtoms(si, sj).SetIsAromatic(True)
        rw=infer_generated_atom_props_from_current_graph_inplace(
            rwmol=rw,editable_atom_mask=source_link)
        # Preserve the model's formal-charge class even after the common
        # generated-atom property inference stage.
        for native_i,si in enumerate(order):
            if bool(native_link[native_i]):
                charge=int(ATOM_TYPES[int(c[native_i])].split('(')[-1].rstrip(')'))
                rw.GetAtomWithIdx(si).SetFormalCharge(charge)
        try:rw.UpdatePropertyCache(strict=False)
        except Exception:pass
        return rw.GetMol()
    except Exception:
        return None


@torch.no_grad()
def sample_batch_linkernet(model,cfg,converted_batch,atom_featurizer,graph_builder,device,max_frag_atoms):
    samples=[make_model_data(s,cfg,atom_featurizer,graph_builder,max_frag_atoms) for s in converted_batch]
    batch=Batch.from_data_list(samples,follow_batch=['edge_type'],exclude_keys=['nbh_list']).to(device)
    traj, xs,cs,bs=model.sample(batch,sample_frag_rot=False,sample_frag_pos=False,
                               sample_link=True,sample_bond=True,
                               p_init_mode=cfg.get('frag_pos_prior',None),
                               guidance_opt=None,pbar=False)
    del traj
    return samples,xs,cs,bs


def run_generate(args):
    set_seed(args.seed)
    device=torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith('cuda') else 'cpu')
    converted=torch_load_cpu(args.converted_pt)
    source=torch_load_cpu(args.source_pt)
    if not isinstance(converted,list) or not converted or not isinstance(source,list):
        raise TypeError('converted and source PT must contain nonempty lists')
    mapping=load_mapping(args.mapping_csv,converted,source)
    skipped_count=count_skipped_rows(args.skipped_csv)
    print(f'[LINKERNET DATA] converted={len(converted)} original={len(source)} skipped={skipped_count}')
    model,cfg,featurizer,graph_builder=build_model(args,device)
    outdir=Path(args.outdir);outdir.mkdir(parents=True,exist_ok=True)
    acc=ProtacMetricAccumulator()
    cache={}
    attempts=0
    recon_none=0
    for batch_idx,start in enumerate(range(0,len(converted),int(args.batch_size))):
        end=min(start+int(args.batch_size),len(converted))
        ids=list(range(start,end))
        cb=[converted[i] for i in ids]
        source_batch=[];templates=[];trues=[];frags=[]
        for oi in ids:
            si=int(mapping[oi]['original_index'])
            src=source[si]
            source_batch.append(src)
            if oi not in cache:
                template=build_fragment_template_with_placeholders(src)
                true,frag=build_source_reference_molecules(src)
                cache[oi]=(template,true,frag)
            template,true,frag=cache[oi]
            templates.append(Chem.Mol(template))
            trues.append(Chem.Mol(true));frags.append(Chem.Mol(frag))
        meta=metadata_batch(source_batch)
        for sample_id in range(int(args.n_samples)):
            data_list,xs,cs,bs=sample_batch_linkernet(model,cfg,cb,featurizer,graph_builder,
                                                     device,int(args.max_frag_atoms))
            if not(len(xs)==len(cs)==len(bs)==len(cb)):
                raise RuntimeError('LinkerNet sampled batch size does not match cases')
            pred=[]
            for i,oi in enumerate(ids):
                mol=fill_linkernet_linker_into_template(templates[i],xs[i],cs[i],bs[i],cb[i],
                                                       source_batch[i],data_list[i])
                pred.append(mol)
                if mol is None:recon_none+=1
            acc.add_batch(pred_molecules=pred,batch=meta,true_molecules=trues,true_fragments=frags,
                          batch_idx=batch_idx,sample_id=sample_id)
            attempts+=len(pred)
        print(f'[batch {batch_idx:04d}] cases={start}:{end} completed_attempts={attempts} reconstruction_none={recon_none}')
    expected=len(converted)*int(args.n_samples)
    if attempts!=expected or len(acc.records)!=expected:
        raise RuntimeError(f'Incomplete sampling: attempts={attempts} records={len(acc.records)} expected={expected}')
    print(f'[LINKERNET] Finished {attempts} attempts; reconstruction_none={recon_none}')
    return save_records(acc.records,outdir)

def metadata_batch(source_batch: Sequence[dict]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    tensor_keys = [
        "anchor1_mask",
        "anchor2_mask",
        "linker_mask",
        "fragment1_mask",
        "fragment2_mask",
    ]
    value_keys = [
        "poi_uniprot",
        "e3_uniprot",
        "uuid",
        "p1_path",
        "p2_path",
        "unbound_lig1_path",
        "unbound_lig2_path",
        "lig1_mask_path",
        "lig2_mask_path",
    ]

    for key in tensor_keys:
        out[key] = [
            torch.as_tensor(src[key]).detach().cpu().bool().view(-1)
            for src in source_batch
        ]

    out["fragment_mask"] = [
        out["fragment1_mask"][i] | out["fragment2_mask"][i]
        for i in range(len(source_batch))
    ]
    out["atom_mask"] = [
        torch.ones(int(src["num_atoms"]), dtype=torch.bool)
        for src in source_batch
    ]

    for key in value_keys:
        out[key] = [src.get(key, None) for src in source_batch]

    return out


def run_evaluate(args) -> Dict[str, Any]:
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    if not Path(args.model_path).exists():
        raise FileNotFoundError(args.model_path)

    model, cfg, atom_featurizer, graph_builder = build_model(args, device)

    print(
        f"[Evaluation initialization] LinkerNet model loaded: "
        f"{Path(args.model_path).resolve()}"
    )

    outdir = Path(args.outdir)
    records = load_records(outdir)
    if len(records) == 0: raise RuntimeError("records.pkl contains no records.")

    protac_stan_reward = None
    deepternary_reward = None

    if not args.no_protac_stan:
        if args.reward_config is None or str(args.reward_config).lower() in {"", "none", "null"}: raise ValueError("Final functional evaluation requires --reward_config, or use --no_protac_stan.")
        from omegaconf import OmegaConf
        reward_conf = OmegaConf.load(args.reward_config)
        if hasattr(reward_conf, "experiment") and hasattr(reward_conf.experiment, "seed"): reward_conf.experiment.seed = int(args.seed)
        protac_stan_reward, deepternary_reward = build_reward_models_from_conf(conf=reward_conf, device=device, use_deepternary=bool(args.use_deepternary))

    groups = {}
    for rec in records:
        if "batch_idx" not in rec or "sample_id" not in rec: raise KeyError("Evaluation record is missing batch_idx or sample_id.")
        key = (int(rec["batch_idx"]), int(rec["sample_id"]))
        groups.setdefault(key, []).append(rec)

    ordered_groups = sorted(groups.items(), key=lambda x: (x[0][0], x[0][1]))
    print(f"Loaded evaluation groups: {len(ordered_groups)}")
    print(f"Loaded evaluation records: {len(records)}")

    acc = ProtacMetricAccumulator()
    sample_records = []

    model_name = "LinkerNet"

    checkpoint_path = str(
        Path(args.model_path).resolve()
    )
    for (batch_idx, sample_id), group_records in ordered_groups:
        pred_molecules = [rec["pred_mol"] for rec in group_records]
        true_molecules = [rec["true_mol"] for rec in group_records]
        true_fragments = [rec["frag_mol"] for rec in group_records]

        strict_flags = []
        strict_infos = []
        smiles_for_stan = []

        for rec in group_records:
            strict, strict_info = is_strict_protac_mol(mol=rec["pred_mol"], anchor1_mask=rec["anchor1_mask"], anchor2_mask=rec["anchor2_mask"], linker_mask=rec["linker_mask"], fragment1_mask=rec["fragment1_mask"], fragment2_mask=rec["fragment2_mask"])
            smi = safe_mol_to_smiles(rec["pred_mol"])
            if smi is None:
                strict = False
                strict_info["strict_protac"] = 0.0
            strict_flags.append(bool(strict))
            strict_infos.append(strict_info)
            smiles_for_stan.append(smi if strict else None)

        if protac_stan_reward is not None:
            degradation_scores = protac_stan_reward.score(smiles_list=smiles_for_stan, poi_ids=[rec["poi_uniprot"] for rec in group_records], e3_ids=[rec["e3_uniprot"] for rec in group_records], step=0, batch_size=int(args.protac_stan_batch_size))
            degradation_scores = torch.nan_to_num(degradation_scores, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            degradation_scores = torch.zeros(len(group_records), dtype=torch.float32, device=device)

        if deepternary_reward is not None:
            ternary_scores, deepternary_info = deepternary_reward.score(mols=pred_molecules, valid_mask=strict_flags, names=[rec["uuid"] for rec in group_records], p1_paths=[rec["p1_path"] for rec in group_records], p2_paths=[rec["p2_path"] for rec in group_records], unbound_lig1_paths=[rec["unbound_lig1_path"] for rec in group_records], unbound_lig2_paths=[rec["unbound_lig2_path"] for rec in group_records], lig1_mask_paths=[rec["lig1_mask_path"] for rec in group_records], lig2_mask_paths=[rec["lig2_mask_path"] for rec in group_records], step=0, max_samples=args.deepternary_max_samples, device=device)
            ternary_scores = torch.nan_to_num(ternary_scores, nan=0.0, posinf=0.0, neginf=0.0)

            required_dt_fields = ["deepternary_pred_p2_rmsd_raw", "deepternary_clash_ratio_raw", "deepternary_scored_mask", "deepternary_attempted_mask", "deepternary_valid_candidate_mask"]
            missing_dt_fields = [key for key in required_dt_fields if key not in deepternary_info]
            if missing_dt_fields: raise KeyError(f"DeepTernaryReward missing required fields: {missing_dt_fields}")
        else:
            ternary_scores = torch.zeros_like(degradation_scores)
            deepternary_info = {
                "deepternary_pred_p2_rmsd_raw": torch.full_like(degradation_scores, -1.0),
                "deepternary_clash_ratio_raw": torch.full_like(degradation_scores, -1.0),
                "deepternary_scored_mask": torch.zeros_like(degradation_scores),
                "deepternary_attempted_mask": torch.zeros_like(degradation_scores),
                "deepternary_valid_candidate_mask": torch.zeros_like(degradation_scores),
            }
        cleanup_deepternary_temp_dirs(
            deepternary_info
        )
        batch_for_records = {}
        for key in ["anchor1_mask", "anchor2_mask", "linker_mask", "atom_mask", "fragment1_mask", "fragment2_mask", "fragment_mask"]:
            batch_for_records[key] = [rec[key] if torch.is_tensor(rec[key]) else torch.as_tensor(rec[key]) for rec in group_records]

        for key in ["poi_uniprot", "e3_uniprot", "uuid", "p1_path", "p2_path", "unbound_lig1_path", "unbound_lig2_path", "lig1_mask_path", "lig2_mask_path"]:
            batch_for_records[key] = [rec.get(key, None) for rec in group_records]

        acc.add_batch(pred_molecules=pred_molecules, batch=batch_for_records, true_molecules=true_molecules, true_fragments=true_fragments, batch_idx=batch_idx, sample_id=sample_id, strict_flags=strict_flags, strict_infos=strict_infos, degradation_scores=degradation_scores, ternary_scores=ternary_scores, deepternary_info=deepternary_info)
        batch_records = make_eval_records_for_batch(
            pred_molecules=pred_molecules,
            batch=batch_for_records,
            batch_idx=int(batch_idx),
            sample_id=int(sample_id),
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            degradation_scores=degradation_scores,
            ternary_scores=ternary_scores,
            deepternary_info=deepternary_info,
            strict_flags=strict_flags,
            strict_infos=strict_infos,
            degradation_threshold=float(
                args.degradation_threshold
            ),
        )

        sample_records.extend(
            batch_records
        )
    reference_linker_smiles = load_smiles_list(args.reference_linker_smiles_path)
    reference_mol_smiles = load_smiles_list(args.reference_mol_smiles_path)

    metrics = acc.compute(reference_linker_smiles=reference_linker_smiles, reference_mol_smiles=reference_mol_smiles, novelty_on=args.novelty_on, recovery_on=args.recovery_on, uniqueness_group=args.uniqueness_group, sa_filter_mode=args.sa_filter_mode, sa_threshold=float(args.sa_threshold), degradation_threshold=float(args.degradation_threshold), use_precomputed_rewards=True)

    if not isinstance(metrics, dict): metrics = metrics[0]

    metrics["model_name"] = "LinkerNet"
    metrics["seed"] = int(args.seed)
    metrics["checkpoint"] = checkpoint_path
    metrics["test_data_prefix"] = str(args.source_pt)
    metrics["reconstruction"] = (
        "GT_fixed_fragment_template_plus_LinkerNet_generated_linker_"
        "atom_types_coordinates_and_predicted_bonds_"
        "linker_linker_and_nonaromatic_anchor_linker_only_"
        "no_nonanchor_fragment_linker_no_f1_f2_shortcut_"
        "no_largest_component_no_repair"
    )

    save_metrics(
        metrics,
        outdir,
    )

    save_diversity_summary(
        sample_records,
        outdir,
    )

    save_records(
        acc.records,
        outdir,
    )

    eval_csv_path = save_eval_records(
        sample_records,
        outdir,
        filename="eval_samples.csv",
    )

    print(
        f"Per-sample eval records saved to: "
        f"{eval_csv_path}"
    )

    print("=" * 100)
    print("[LINKERNET EVALUATION COMPLETE]")
    print(f"records:   {len(acc.records)}")
    print(f"Validity:  {metrics.get('Validity', 0.0):.12f}")
    print(f"metrics:   {(outdir / 'test_metrics.json').resolve()}")
    print("=" * 100)

    return metrics




def create_argparser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval_mode',default='evaluate',choices=['generate','evaluate','both'])
    p.add_argument('--model_path',default='checkpoints/linkernet_protac_epoch_009.pt')
    p.add_argument('--converted_pt',default='data/protac/linkernet_test.pt')
    p.add_argument('--source_pt',default='data/protac_rltrain_test.pt')
    p.add_argument('--mapping_csv',default='data/protac/linkernet_audit/test_mapping.csv')
    p.add_argument('--skipped_csv',default='data/protac/linkernet_audit/test_skipped.csv')
    p.add_argument('--outdir',default='eval_outputs/linkernet_test_2026')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--batch_size',type=int,default=32)
    p.add_argument('--n_samples',type=int,default=5)
    p.add_argument('--max_frag_atoms',type=int,default=80)
    p.add_argument('--reference_linker_smiles_path',default='./datasets/reference_linkers.smi')
    p.add_argument('--reference_mol_smiles_path',default=None)
    p.add_argument('--novelty_on',default='linker',choices=['linker','molecule'])
    p.add_argument('--recovery_on',default='linker',choices=['linker','molecule'])
    p.add_argument('--uniqueness_group',default='case',choices=['case','global'])
    p.add_argument('--sa_filter_mode',default='threshold',choices=['delinker','threshold'])
    p.add_argument('--sa_threshold',type=float,default=6.0)
    p.add_argument('--reward_config',default='./configs/linkernet_eval_reward.yaml')
    p.add_argument('--no_protac_stan',action='store_true',default=False)
    p.add_argument('--degradation_threshold',type=float,default=0.5)
    p.add_argument('--protac_stan_batch_size',type=int,default=32)
    p.add_argument('--use_deepternary',type=str2bool,default=True)
    p.add_argument('--deepternary_max_samples',type=int,default=None)
    p.add_argument('--five_seeds',action='store_true',default=True,help='Final-test seeds 2024..2028, each with its own output')
    return p


def main():
    args = create_argparser().parse_args()
    runs = [(args.seed, args.outdir)]
    if args.five_seeds:
        runs = [(seed, f'eval_outputs/linkernet_test_{seed}') for seed in (2024, 2025, 2026, 2027, 2028)]
    for seed, outdir in runs:
        runargs = copy.deepcopy(args)
        runargs.seed = seed
        runargs.outdir = outdir
        if runargs.eval_mode in ('generate', 'both'):
            for path in (runargs.model_path, runargs.converted_pt, runargs.source_pt, runargs.mapping_csv):
                if not Path(path).exists():
                    raise FileNotFoundError(path)
            run_generate(runargs)
        if runargs.eval_mode in ('evaluate', 'both'):
            run_evaluate(runargs)

if __name__=='__main__':
    main()
