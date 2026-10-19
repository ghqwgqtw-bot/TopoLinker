#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DesignMaster sampling + unified PROTAC reconstruction/evaluation.

Design:
1) DesignMaster converted .pt is used only for model conditioning/sampling.
2) mapping.csv maps converted output_index -> original TopoLinker original_index.
3) Original TopoLinker records provide fragment bonds, atom properties, anchor1/2
   masks, protein-context paths, and GT reference molecules.
4) DesignMaster generates linker atom types + coordinates under the seven
   physicochemical conditions. Bond perception uses the same unified
   OpenBabel post-sampling route as the other baselines:
      generated full point cloud -> OpenBabel bond perception.
5) Final benchmark molecule is hybrid:
      - GT fixed-fragment atom identities/coordinates/internal bonds/properties;
      - generated linker atom identities/coordinates;
      - only OpenBabel linker-linker + designated anchor-linker bonds;
      - no non-anchor fragment-linker bonds;
      - no fragment1-fragment2 shortcut;
      - aromatic anchor-linker bonds are rejected, never converted to single;
      - linker-linker aromatic bonds are preserved as aromatic.
6) Reconstruction failure remains a record and therefore remains in the metric
   denominator. Sampling failure raises by default rather than silently dropping data.

The same file is used by validation checkpoint selection and final test evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import copy
import pickle
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors, Crippen, Lipinski, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from model_base.edm import EDM as EDM_backbone
from model_mc_v3.edm_mc_v3 import EDM_CFG
from utils.dataset import collate, create_templates_for_linker_generation
from utils.utils import remove_partial_mean_with_mask

from protac_eval_metrics import ProtacMetricAccumulator, build_reward_models_from_conf, is_strict_protac_mol
from protac_molecule_builder import (
    build_molecules as protac_build_molecules,
    infer_generated_atom_props_from_current_graph_inplace,
)


# Released DesignMaster/GEOM channel order.
DESIGNMASTER_Z = [6, 8, 7, 9, 16, 17, 35, 53, 15]


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

        # Same temporary sample-dir level as the previous
        # copy-and-clean implementation.
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
    padded atom index -> RDKit mol local atom index.

    build_molecules 后的 RDKit mol 只包含 atom_mask=True 的原子，
    所以 MolFragmentToSmiles 需要使用去 padding 后的 local index。
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
    RDKit Contrib SA score. 如果环境里没有 sascorer，就返回 nan。
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
    Use the user's shared molecule builder for the fixed fragment template.
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


def resolve_obabel(executable: str) -> str:
    p = Path(executable)
    if p.exists() and p.is_file():
        return str(p.resolve())
    found = shutil.which(executable)
    if found is None:
        raise FileNotFoundError(
            f"OpenBabel executable not found: {executable!r}. "
            f"Set --obabel to the full path of obabel.exe."
        )
    return found


def write_xyz(path: Path, atomic_numbers: torch.Tensor, positions: torch.Tensor) -> None:
    atomic_numbers = atomic_numbers.detach().cpu().long().view(-1)
    positions = positions.detach().cpu().float()
    if positions.shape != (atomic_numbers.numel(), 3):
        raise ValueError("XYZ atom/position length mismatch")

    periodic = Chem.GetPeriodicTable()
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{atomic_numbers.numel()}\n")
        f.write("DesignMaster generated sample\n")
        for z, xyz in zip(atomic_numbers.tolist(), positions.tolist()):
            symbol = periodic.GetElementSymbol(int(z))
            f.write(
                f"{symbol} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n"
            )


def openbabel_bond_perception(
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    obabel: str,
    tmp_root: str | Path,
) -> Optional[Chem.Mol]:
    """
    Use OpenBabel on the generated whole-molecule point cloud.
    Atom count and atom order are audited after conversion.
    """
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="designmaster_ob_", dir=str(tmp_root)) as td:
        td = Path(td)
        xyz_path = td / "sample.xyz"
        sdf_path = td / "sample.sdf"
        write_xyz(xyz_path, atomic_numbers, positions)

        cmd = [
            obabel,
            "-ixyz",
            str(xyz_path),
            "-osdf",
            "-O",
            str(sdf_path),
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0 or not sdf_path.exists():
            return None

        with sdf_path.open("r", encoding="utf-8", errors="replace") as f:
            sdf_text = f.read()

        mol_block = sdf_text.split("$$$$", 1)[0]
        mol = Chem.MolFromMolBlock(mol_block, removeHs=False, sanitize=False)

        if mol is None:
            return None

        expected_z = atomic_numbers.detach().cpu().long().view(-1).tolist()
        if mol.GetNumAtoms() != len(expected_z):
            return None
        got_z = [a.GetAtomicNum() for a in mol.GetAtoms()]
        if got_z != expected_z:
            # Never copy bonds if OpenBabel atom indexing/order changed.
            return None
        return mol


def merge_template_with_openbabel_linker_bonds(
    template_mol: Chem.Mol,
    obabel_mol: Optional[Chem.Mol],
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    fragment1_mask: torch.Tensor,
    fragment2_mask: torch.Tensor,
    linker_mask: torch.Tensor,
    anchor1_mask: torch.Tensor,
    anchor2_mask: torch.Tensor,
) -> Optional[Chem.Mol]:
    """
    Keep GT fragment template; copy only linker-linker and designated anchor-linker
    bonds from OpenBabel. No repair / largest component / f1-f2 shortcut.
    """
    if template_mol is None or obabel_mol is None:
        return None

    try:
        atomic_numbers = torch.as_tensor(atomic_numbers).detach().cpu().long().view(-1)
        positions = torch.as_tensor(positions).detach().cpu().float()
        f1 = torch.as_tensor(fragment1_mask).detach().cpu().bool().view(-1)
        f2 = torch.as_tensor(fragment2_mask).detach().cpu().bool().view(-1)
        linker = torch.as_tensor(linker_mask).detach().cpu().bool().view(-1)
        a1 = torch.as_tensor(anchor1_mask).detach().cpu().bool().view(-1)
        a2 = torch.as_tensor(anchor2_mask).detach().cpu().bool().view(-1)

        n = template_mol.GetNumAtoms()
        if obabel_mol.GetNumAtoms() != n:
            return None
        if atomic_numbers.numel() != n or positions.shape != (n, 3):
            return None

        rwmol = Chem.RWMol(Chem.Mol(template_mol))

        # Defensive cleanup: the template should contain no linker-related bonds.
        to_remove = []
        for bond in rwmol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if bool(linker[i]) or bool(linker[j]):
                to_remove.append((i, j))
        for i, j in to_remove:
            rwmol.RemoveBond(int(i), int(j))

        # Replace linker dummy carbons with generated DesignMaster atom identities.
        for idx_t in torch.where(linker)[0]:
            idx = int(idx_t.item())
            z = int(atomic_numbers[idx].item())
            atom = Chem.Atom(z)
            atom.SetFormalCharge(0)
            atom.SetIsAromatic(False)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
            rwmol.ReplaceAtom(idx, atom, preserveProps=False)

        if rwmol.GetNumConformers() == 0:
            return None
        conf = rwmol.GetConformer()
        for idx in range(n):
            xyz = positions[idx]
            conf.SetAtomPosition(
                idx,
                (
                    float(xyz[0].item()),
                    float(xyz[1].item()),
                    float(xyz[2].item()),
                ),
            )

        anchor = a1 | a2

        for bond in obabel_mol.GetBonds():
            i = int(bond.GetBeginAtomIdx())
            j = int(bond.GetEndAtomIdx())

            i_link = bool(linker[i])
            j_link = bool(linker[j])
            is_linker_linker = i_link and j_link
            is_anchor_linker = (
                (bool(anchor[i]) and j_link)
                or (bool(anchor[j]) and i_link)
            )

            # GT fragment bonds are already in the template. Do not copy anything
            # else from OpenBabel.
            if not (is_linker_linker or is_anchor_linker):
                continue
            if rwmol.GetBondBetweenAtoms(i, j) is not None:
                continue

            bt = bond.GetBondType()
            is_aromatic = bool(
                bond.GetIsAromatic() or bt == Chem.BondType.AROMATIC
            )

            # Benchmark policy: aromatic anchor-linker is invalid and is not
            # silently converted to a single bond.
            if is_anchor_linker and is_aromatic:
                continue

            if is_aromatic:
                rwmol.GetAtomWithIdx(i).SetIsAromatic(True)
                rwmol.GetAtomWithIdx(j).SetIsAromatic(True)
                rwmol.AddBond(i, j, Chem.BondType.AROMATIC)
                new_bond = rwmol.GetBondBetweenAtoms(i, j)
                if new_bond is not None:
                    new_bond.SetIsAromatic(True)
            elif bt in {
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
            }:
                rwmol.AddBond(i, j, bt)
            else:
                # Unsupported/OpenBabel-specific bond type: do not invent a type.
                continue

        # Only generated linker atom properties are inferred from the current graph.
        rwmol = infer_generated_atom_props_from_current_graph_inplace(
            rwmol=rwmol,
            editable_atom_mask=linker,
        )

        try:
            rwmol.UpdatePropertyCache(strict=False)
        except Exception:
            pass

        return rwmol.GetMol()

    except Exception:
        return None


def _ensure_not_git_lfs_pointer(path: str | Path) -> None:
    """Fail early with a clear message if a checkpoint is only a Git-LFS pointer."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        with p.open("rb") as f:
            head = f.read(128)
    except OSError:
        return
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{p} is a Git-LFS pointer, not the real checkpoint. "
            "Download the actual checkpoint before evaluation."
        )


def build_model(args, device: torch.device) -> EDM_CFG:
    """
    Rebuild the same DesignMaster architecture used during training, then load the
    complete trained DesignMaster state_dict. The saved DesignMaster checkpoint
    already contains the backbone parameters, so geom_best.ckpt is not needed here.
    """
    _ensure_not_git_lfs_pointer(args.model_path)

    # Build the backbone structure only. Its temporary random initialization is
    # fully overwritten by the trained DesignMaster checkpoint below.
    backbone_model = EDM_backbone(
        device=str(device),
        in_node_nf=9,
        hidden_nf=128,
        ffn_embedding_dim=1024,
        attention_heads=32,
        n_layers=6,
        tanh=False,
        coords_range=10.0,
        dropout=0.05,
        activation_dropout=0.05,
    )
    backbone = backbone_model.dynamics.dynamics

    model = EDM_CFG(
        device=str(device),
        in_node_nf=9,
        timesteps=500,
        noise_schedule="polynomial_2",
        noise_precision=1e-5,
        loss_type="l2",
        hidden_nf=128,
        ffn_embedding_dim=1024,
        attention_heads=32,
        tanh=False,
        n_layers=6,
        coords_range=10.0,
        dropout=0.05,
        activation_dropout=0.05,
        backbone=backbone,
        guidance_scale=float(args.guidance_scale),
        condition_drop_prob=0.1,
    )

    # This is the only checkpoint loaded for validation/test sampling. It contains
    # the complete trained DesignMaster model, including the fine-tuned backbone.
    state_dict = safe_torch_load_state_dict(args.model_path)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _property_tensor(batch: Dict[str, Any], key: str, device: torch.device) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"DesignMaster batch is missing required property: {key}")
    value = batch[key]
    if torch.is_tensor(value):
        t = value.detach().to(device=device, dtype=torch.float32)
    else:
        t = torch.as_tensor(value, dtype=torch.float32, device=device)
    return t.reshape(-1, 1)


@torch.no_grad()
def sample_batch(model: EDM_CFG, batch: Dict[str, Any], device: torch.device, guidance_scale: float):
    """
    DesignMaster sampling path matching CFGTrainer.sample_chain():
      - ground-truth linker size from linker_mask;
      - fixed fragments;
      - seven physicochemical conditions from the converted PT record;
      - classifier-free guidance at the requested guidance scale.
    """
    linker_sizes = batch["linker_mask"].sum(1).view(-1).int()
    template = create_templates_for_linker_generation(batch, linker_sizes)

    x = template["positions"]
    node_mask = template["atom_mask"]
    edge_mask = template["edge_mask"]
    h = template["one_hot"]
    fragment_mask = template["fragment_mask"]
    linker_mask = template["linker_mask"]
    context = fragment_mask

    numerical_context = [
        _property_tensor(batch, "molecular_weight", device),
        _property_tensor(batch, "H_bond_acceptor_cnt", device),
        _property_tensor(batch, "H_bond_donor_cnt", device),
        _property_tensor(batch, "XLogP3", device),
        _property_tensor(batch, "rotatable_bond_cnt", device),
        _property_tensor(batch, "heavy_atom_cnt", device),
        _property_tensor(batch, "topological_polar_surface_area", device),
    ]

    x = remove_partial_mean_with_mask(
        x,
        node_mask,
        fragment_mask,
    )

    chain = model.sample_chain(
        x=x.to(device),
        h=h.to(device),
        node_mask=node_mask.to(device),
        edge_mask=edge_mask.to(device),
        fragment_mask=fragment_mask.to(device),
        linker_mask=linker_mask.to(device),
        context=context.to(device),
        numerical_context=numerical_context,
        keep_frames=1,
        guidance_scale=float(guidance_scale),
    )
    return chain, node_mask, template


def get_source_context(src: dict):
    n = int(src["num_atoms"])

    pos = torch.as_tensor(src["positions"]).detach().cpu().float()
    charges = torch.as_tensor(src["charges"]).detach().cpu().round().long().view(-1)
    f1 = torch.as_tensor(src["fragment1_mask"]).detach().cpu().bool().view(-1)
    f2 = torch.as_tensor(src["fragment2_mask"]).detach().cpu().bool().view(-1)
    linker = torch.as_tensor(src["linker_mask"]).detach().cpu().bool().view(-1)
    a1 = torch.as_tensor(src["anchor1_mask"]).detach().cpu().bool().view(-1)
    a2 = torch.as_tensor(src["anchor2_mask"]).detach().cpu().bool().view(-1)

    if pos.shape != (n, 3):
        raise ValueError("source positions shape mismatch")
    for name, mask in [
        ("fragment1_mask", f1),
        ("fragment2_mask", f2),
        ("linker_mask", linker),
        ("anchor1_mask", a1),
        ("anchor2_mask", a2),
    ]:
        if mask.numel() != n:
            raise ValueError(f"{name} length mismatch")

    frag = f1 | f2
    if int(a1.sum()) != 1 or int(a2.sum()) != 1:
        raise ValueError("Expected exactly one anchor1 and one anchor2")
    if not bool((a1 <= f1).all()) or not bool((a2 <= f2).all()):
        # This assumes anchor1 belongs to fragment1 and anchor2 to fragment2,
        # matching the user's source schema.
        raise ValueError("anchor1/anchor2 masks do not belong to fragment1/fragment2")

    fragment_com = pos[frag].mean(dim=0, keepdim=True)

    return {
        "n": n,
        "positions": pos,
        "charges": charges,
        "fragment1_mask": f1,
        "fragment2_mask": f2,
        "fragment_mask": frag,
        "linker_mask": linker,
        "anchor1_mask": a1,
        "anchor2_mask": a2,
        "fragment_com": fragment_com,
    }


def reconstruction_from_prediction(
    pred_x_centered: torch.Tensor,
    pred_h: torch.Tensor,
    src: dict,
    template_mol: Chem.Mol,
    obabel: str,
    tmp_root: str | Path,
) -> Optional[Chem.Mol]:
    ctx = get_source_context(src)
    n = ctx["n"]

    pred_x_centered = pred_x_centered.detach().cpu().float()[:n]
    pred_h = pred_h.detach().cpu().float()[:n]

    if pred_x_centered.shape != (n, 3):
        return None
    if pred_h.ndim != 2 or pred_h.shape[0] != n or pred_h.shape[1] != 9:
        return None

    linker = ctx["linker_mask"]
    frag = ctx["fragment_mask"]

    # DesignMaster output coordinates are fragment-COM centered.
    pred_global = pred_x_centered + ctx["fragment_com"]

    hybrid_pos = ctx["positions"].clone()
    hybrid_pos[linker] = pred_global[linker]

    pred_idx = pred_h.argmax(dim=-1)
    pred_z = torch.tensor(
        [DESIGNMASTER_Z[int(i)] for i in pred_idx.tolist()],
        dtype=torch.long,
    )

    # Preserve fixed fragment chemistry exactly; use generated atom types only
    # for linker atoms.
    hybrid_z = ctx["charges"].clone()
    hybrid_z[linker] = pred_z[linker]

    ob_mol = openbabel_bond_perception(
        atomic_numbers=hybrid_z,
        positions=hybrid_pos,
        obabel=obabel,
        tmp_root=tmp_root,
    )

    return merge_template_with_openbabel_linker_bonds(
        template_mol=template_mol,
        obabel_mol=ob_mol,
        atomic_numbers=hybrid_z,
        positions=hybrid_pos,
        fragment1_mask=ctx["fragment1_mask"],
        fragment2_mask=ctx["fragment2_mask"],
        linker_mask=ctx["linker_mask"],
        anchor1_mask=ctx["anchor1_mask"],
        anchor2_mask=ctx["anchor2_mask"],
    )


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


def run_generate(args) -> Dict[str, Any]:
    set_seed(args.seed)

    device = torch.device(
        args.device
        if torch.cuda.is_available() and str(args.device).startswith("cuda")
        else "cpu"
    )

    converted = torch_load_cpu(args.converted_pt)
    source = torch_load_cpu(args.source_pt)
    if not isinstance(converted, list) or not converted:
        raise TypeError("converted_pt must contain a non-empty list")
    if not isinstance(source, list) or not source:
        raise TypeError("source_pt must contain a non-empty list")

    mapping = load_mapping(
        args.mapping_csv,
        converted_records=converted,
        source_records=source,
    )

    skipped_count = count_skipped_rows(args.skipped_csv)

    print("=" * 100)
    print("[DESIGNMASTER DATA AUDIT]")
    print(f"converted_pt:      {Path(args.converted_pt).resolve()}")
    print(f"source_pt:         {Path(args.source_pt).resolve()}")
    print(f"mapping_csv:       {Path(args.mapping_csv).resolve()}")
    print(f"converted records: {len(converted)}")
    print(f"source records:    {len(source)}")
    print(f"mapping rows:      {len(mapping)}")
    print(f"unsupported skip:  {skipped_count}")
    print("=" * 100)

    obabel = resolve_obabel(args.obabel)
    print(f"OpenBabel: {obabel}")

    model = build_model(args, device)
    print(f"Checkpoint: {Path(args.model_path).resolve()}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_root = outdir / "_openbabel_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    acc = ProtacMetricAccumulator()

    # Cache source-side RDKit structures once per converted case.
    source_cache: Dict[int, Dict[str, Any]] = {}

    total_attempts = 0
    reconstruction_none = 0

    batch_size = int(args.batch_size)
    n_samples = int(args.n_samples)

    for batch_idx, start in enumerate(range(0, len(converted), batch_size)):
        end = min(start + batch_size, len(converted))
        output_indices = list(range(start, end))
        converted_batch = [converted[i] for i in output_indices]
        model_batch = collate(converted_batch)

        source_batch = []
        templates = []
        true_molecules = []
        true_fragments = []

        for oi in output_indices:
            row = mapping[oi]
            si = int(row["original_index"])
            src = source[si]
            source_batch.append(src)

            if oi not in source_cache:
                template = build_fragment_template_with_placeholders(src)
                true_mol, frag_mol = build_source_reference_molecules(src)
                source_cache[oi] = {
                    "template": template,
                    "true_mol": true_mol,
                    "frag_mol": frag_mol,
                }

            templates.append(Chem.Mol(source_cache[oi]["template"]))
            true_molecules.append(Chem.Mol(source_cache[oi]["true_mol"]))
            true_fragments.append(Chem.Mol(source_cache[oi]["frag_mol"]))

        batch_meta = metadata_batch(source_batch)

        for sample_id in range(n_samples):
            # Sampling failure is not silently converted into reconstruction failure.
            chain, node_mask, _ = sample_batch(model, model_batch, device, args.guidance_scale)
            pred = chain[0].detach().cpu()
            pred_x = pred[:, :, :3]
            pred_h = pred[:, :, 3:]

            pred_molecules = []

            for local_i, oi in enumerate(output_indices):
                src = source_batch[local_i]
                n = int(src["num_atoms"])

                mask_i = node_mask[local_i].detach().cpu().squeeze().bool()
                if int(mask_i.sum().item()) != n:
                    raise RuntimeError(
                        f"output_index={oi}: sampled node count "
                        f"{int(mask_i.sum())} != source num_atoms {n}"
                    )

                mol = reconstruction_from_prediction(
                    pred_x_centered=pred_x[local_i, :n],
                    pred_h=pred_h[local_i, :n],
                    src=src,
                    template_mol=templates[local_i],
                    obabel=obabel,
                    tmp_root=tmp_root,
                )
                if mol is None:
                    reconstruction_none += 1
                pred_molecules.append(mol)

            # Reconstruction failure remains in records and therefore denominator.
            acc.add_batch(
                pred_molecules=pred_molecules,
                batch=batch_meta,
                true_molecules=true_molecules,
                true_fragments=true_fragments,
                batch_idx=batch_idx,
                sample_id=sample_id,
            )
            total_attempts += len(pred_molecules)

        print(
            f"[batch {batch_idx:04d}] cases={start}:{end} "
            f"completed_attempts={total_attempts} "
            f"reconstruction_none={reconstruction_none}"
        )

    expected = len(converted) * n_samples
    if total_attempts != expected or len(acc.records) != expected:
        raise RuntimeError(
            f"Attempt-count mismatch: attempts={total_attempts}, "
            f"records={len(acc.records)}, expected={expected}"
        )

    records_path = save_records(acc.records, outdir)

    # Remove empty OpenBabel temp root after all TemporaryDirectory instances close.
    try:
        tmp_root.rmdir()
    except OSError:
        pass

    print("=" * 100)
    print("DesignMaster sampling finished.")
    print(f"Completed generation attempts: {total_attempts}")
    print("=" * 100)

    return records_path


def run_evaluate(args) -> Dict[str, Any]:
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    if not Path(args.model_path).exists(): raise FileNotFoundError(args.model_path)
    model = build_model(args, device)
    print(f"[Evaluation initialization] DesignMaster model loaded: {Path(args.model_path).resolve()}")

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

    model_name = "DesignMaster"

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

    metrics["model_name"] = "DesignMaster"
    metrics["seed"] = int(args.seed)
    metrics["checkpoint"] = checkpoint_path
    metrics["test_data_prefix"] = str(args.source_pt)
    metrics["reconstruction"] = (
        "GT_fixed_fragment_template_plus_DesignMaster_generated_linker_"
        "atom_types_and_coordinates_plus_OpenBabel_linker_related_bonds_"
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
    print("[DESIGNMASTER EVALUATION COMPLETE]")
    print(f"records:   {len(acc.records)}")
    print(f"Validity:  {metrics.get('Validity', 0.0):.12f}")
    print(f"metrics:   {(outdir / 'test_metrics.json').resolve()}")
    print("=" * 100)

    return metrics


def create_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--eval_mode",
        default="evaluate",
        choices=["generate", "evaluate", "both"],
    )

    # Final test defaults. The validation selector overrides these arguments.
    p.add_argument(
        "--model_path",
        default="checkpoints/protacs_mc_la_20_300.ckpt",
    )
    p.add_argument(
        "--converted_pt",
        default="datasets/protacs_test_self.pt",
    )
    p.add_argument(
        "--source_pt",
        default="datasets/protac_rltrain_test.pt",
    )
    p.add_argument(
        "--mapping_csv",
        default="datasets/designmaster_conversion_audit/test_mapping.csv",
    )
    p.add_argument(
        "--skipped_csv",
        default="datasets/designmaster_conversion_audit/test_skipped.csv",
    )
    p.add_argument(
        "--outdir",
        default="eval_outputs/designmaster_test_2026",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--guidance_scale", type=float, default=3.0)

    p.add_argument(
        "--obabel",
        default="obabel",
        help="OpenBabel executable name or full path to obabel.exe",
    )

    # Unified metric settings.
    p.add_argument(
        "--reference_linker_smiles_path",
        default="./datasets/reference_linkers.smi",
    )
    p.add_argument(
        "--reference_mol_smiles_path",
        default=None,
    )
    p.add_argument(
        "--novelty_on",
        default="linker",
        choices=["linker", "molecule"],
    )
    p.add_argument(
        "--recovery_on",
        default="linker",
        choices=["linker", "molecule"],
    )
    p.add_argument(
        "--uniqueness_group",
        default="case",
        choices=["case", "global"],
    )
    p.add_argument(
        "--sa_filter_mode",
        default="threshold",
        choices=["delinker", "threshold"],
    )
    p.add_argument("--sa_threshold", type=float, default=6.0)

    # Unified reward configuration for the DesignMaster project; keep its metric/reward parameters identical to the other baselines.
    p.add_argument(
        "--reward_config",
        default="./configs/designmaster_eval_reward.yaml",
    )
    p.add_argument(
        "--no_protac_stan",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--degradation_threshold",
        type=float,
        default=0.5,
    )
    p.add_argument(
        "--protac_stan_batch_size",
        type=int,
        default=32,
    )
    p.add_argument(
        "--use_deepternary",
        type=str2bool,
        default=True,
    )
    p.add_argument(
        "--deepternary_max_samples",
        type=int,
        default=None,
    )

    return p


def main():
    args = create_argparser().parse_args()
    seeds_config = [
        # (2024, "eval_outputs/designmaster_test_2024"),
        # (2025, "eval_outputs/designmaster_test_2025"),
        (2026, "eval_outputs/designmaster_test_2026"),
        # (2027, "eval_outputs/designmaster_test_2027"),
        # (2028, "eval_outputs/designmaster_test_2028"),
    ]
    for seed, outdir in seeds_config:
        run_args = copy.deepcopy(args)
        run_args.seed = seed
        run_args.outdir = outdir
        if run_args.eval_mode in {"generate", "both"}:
            for path in [
                run_args.model_path,
                run_args.converted_pt,
                run_args.source_pt,
                run_args.mapping_csv,
            ]:
                if not Path(path).exists():
                    raise FileNotFoundError(path)

            run_generate(run_args)

        if run_args.eval_mode in {"evaluate", "both"}:
            run_evaluate(run_args)


if __name__ == "__main__":
    main()
