#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate the selected/best DiffLinker checkpoint on the full test set.

Pipeline:
    DiffLinker best .ckpt
    -> full test set sampling
    -> xyz
    -> OpenBabel xyz-to-SDF reconstruction
    -> RDKit Mol, without keeping only the largest connected component
    -> ProtacMetricAccumulator / protac_eval_metrics.py final metrics

Run from DiffLinker project root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors, rdMolDescriptors, Crippen, Lipinski, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from src import utils
from src.lightning import DDPM
from src.protac_molecule_builder import (
    build_molecules as protac_build_molecules,
    infer_generated_atom_props_from_current_graph_inplace,
)

from evaluate_difflinker_checkpoints import (
    make_fragment_internal_bond_types,
    make_hybrid_one_hot_for_template,
    merge_template_fragments_with_openbabel_linker_bonds,
)
from src.visualizer import save_xyz_file

from protac_eval_metrics import (
    ProtacMetricAccumulator,
    build_reward_models_from_conf,
    is_strict_protac_mol,
)

RDLogger.DisableLog("rdApp.warning")


def safe_mol_to_smiles(mol):
    if mol is None:
        return None

    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)

        smi = Chem.MolToSmiles(mol_copy)

        if smi is None or len(smi) == 0:
            return None

        return smi

    except Exception:
        return None
def move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    return batch


def check_obabel_available() -> None:
    ret = subprocess.run(
        "obabel -V",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ret.returncode != 0:
        raise RuntimeError(
            "OpenBabel executable `obabel` was not found or not usable. "
            "Please install OpenBabel and make sure `obabel` is in PATH."
        )


def read_first_sdf_mol(sdf_path: str) -> Optional[Chem.Mol]:
    try:
        supp = Chem.SDMolSupplier(sdf_path, sanitize=False)
        if len(supp) == 0:
            return None
        return supp[0]
    except Exception:
        return None


def save_metrics(metrics_dict: Dict[str, Any], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "test_metrics.pkl"), "wb") as f:
        pickle.dump(metrics_dict, f)
    with open(os.path.join(save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    with open(os.path.join(save_dir, "test_metrics.txt"), "w", encoding="utf-8") as f:
        for k in sorted(metrics_dict.keys()):
            f.write(f"{k}: {metrics_dict[k]}\n")



def make_pickle_safe_records(records):
    safe_records = []

    for rec in records:
        new_rec = {}

        for k, v in rec.items():
            if torch.is_tensor(v):
                new_rec[k] = v.detach().cpu()
            else:
                new_rec[k] = v

        safe_records.append(new_rec)

    return safe_records


def save_records_pkl(records, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    records_pkl_path = os.path.join(
        save_dir,
        "records.pkl",
    )

    with open(records_pkl_path, "wb") as f:
        pickle.dump(
            make_pickle_safe_records(records),
            f,
        )

    print(
        f"Full RDKit records saved to: "
        f"{records_pkl_path}"
    )

    return records_pkl_path


def load_records_pkl(save_dir):
    records_pkl_path = os.path.join(
        save_dir,
        "records.pkl",
    )

    if not os.path.exists(records_pkl_path):
        raise FileNotFoundError(
            f"records.pkl not found: "
            f"{records_pkl_path}"
        )

    with open(records_pkl_path, "rb") as f:
        records = pickle.load(f)

    if not isinstance(records, list):
        raise TypeError(
            "Expected records.pkl to contain list, "
            f"got {type(records)}"
        )

    print(
        f"Loaded records.pkl from: "
        f"{os.path.abspath(records_pkl_path)}"
    )
    print(f"Loaded records: {len(records)}")

    return records

def load_smiles_list(path: Optional[str]) -> Optional[List[str]]:
    if path is None or str(path).lower() == "none":
        return None

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Reference SMILES file not found: {path}"
        )

    values: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            values.append(s.split()[0])

    return values


def str2bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in {"1", "true", "yes", "y", "on"}:
        return True
    if x in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool value: {x}")


def load_reward_config(path: Optional[str]) -> Any:
    if path is None or str(path).lower() == "none":
        return None
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def get_atom_mask_for_mol(model: DDPM, data: Dict[str, Any], node_mask: torch.Tensor) -> torch.Tensor:
    if "." in model.train_data_prefix and "pocket_mask" in data:
        return node_mask - data["pocket_mask"]
    return node_mask


def get_fragment_mask_for_mol(model: DDPM, data: Dict[str, Any]) -> torch.Tensor:
    if "." in model.train_data_prefix and "fragment_only_mask" in data:
        return data["fragment_only_mask"]
    return data["fragment_mask"]


def restore_com_like_difflinker_generate(
    model: DDPM,
    data: Dict[str, Any],
    x: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    if model.center_of_mass == "fragments":
        if "." in model.train_data_prefix and "fragment_only_mask" in data:
            com_mask = data["fragment_only_mask"]
        else:
            com_mask = data["fragment_mask"]
    elif model.center_of_mass == "anchors":
        com_mask = data["anchors"]
    else:
        return x

    pos_masked = data["positions"] * com_mask
    denom = com_mask.sum(1, keepdim=True).clamp(min=1.0)
    mean = torch.sum(pos_masked, dim=1, keepdim=True) / denom
    return x + mean * node_mask


def build_openbabel_molecules_from_prediction(
    *,
    one_hot: torch.Tensor,
    x: torch.Tensor,
    node_mask: torch.Tensor,
    is_geom: bool,
    tmp_dir: str,
    name_prefix: str,
) -> List[Optional[Chem.Mol]]:
    os.makedirs(tmp_dir, exist_ok=True)
    B = int(one_hot.size(0))
    names = [f"{name_prefix}_{i}" for i in range(B)]

    save_xyz_file(
        tmp_dir,
        one_hot.detach().cpu(),
        x.detach().cpu(),
        node_mask.detach().cpu(),
        names=names,
        is_geom=is_geom,
        suffix="",
    )

    mols: List[Optional[Chem.Mol]] = []
    for name in names:
        xyz_path = os.path.join(tmp_dir, f"{name}_.xyz")
        sdf_path = os.path.join(tmp_dir, f"{name}_.sdf")
        ret = subprocess.run(
            ["obabel", xyz_path, "-O", sdf_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ret.returncode != 0 or not os.path.exists(sdf_path):
            mols.append(None)
        else:
            mols.append(read_first_sdf_mol(sdf_path))
    return mols


def require_keys(data: Dict[str, Any], keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(
            f"Test batch is missing required PROTAC masks: {missing}. "
            "Need anchor1_mask, anchor2_mask, linker_mask and atom_mask."
        )


def load_difflinker_model_for_test(
    checkpoint: str,
    device: torch.device,
    batch_size: Optional[int],
    data_path: Optional[str],
    test_data_prefix: Optional[str],
) -> DDPM:
    model = DDPM.load_from_checkpoint(checkpoint, map_location=device)
    model.to(device)
    model.eval()
    model.torch_device = str(device)

    if batch_size is not None:
        model.batch_size = int(batch_size)
    if data_path is not None and str(data_path).lower() != "none":
        model.data_path = data_path
    if test_data_prefix is not None and str(test_data_prefix).lower() != "none":
        model.val_data_prefix = test_data_prefix

    # DiffLinker DDPM.setup(stage='val') builds val_dataset using val_data_prefix.
    # We reuse it for test-set evaluation by overriding val_data_prefix above.
    model.setup(stage="val")
    return model

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

def safe_path_token(text):
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", text)
    return text


def read_deepternary_metric_value(deepternary_info, key, i, default=-1.0):
    if key not in deepternary_info:
        return default

    value = deepternary_info[key]

    if torch.is_tensor(value):
        if value.ndim > 0 and value.numel() > i:
            return tensor_to_float(value.reshape(-1)[i], default=default)
        return default

    return default


def copy_deepternary_complex_pdbs(
        *,
        deepternary_info,
        save_dir,
        batch,
        batch_idx,
        sample_id,
        remove_source_dirs=True,
):
    if deepternary_info is None:
        return deepternary_info

    if "deepternary_complex_pred_path" not in deepternary_info:
        return deepternary_info

    complex_paths = deepternary_info["deepternary_complex_pred_path"]

    if not isinstance(complex_paths, (list, tuple)):
        return deepternary_info

    complex_save_dir = os.path.join(save_dir, "deepternary_complex_pdbs")
    os.makedirs(complex_save_dir, exist_ok=True)

    copied_paths = [None for _ in range(len(complex_paths))]

    for i, src_path in enumerate(complex_paths):
        if src_path in [None, "", "None", "nan"]:
            copied_paths[i] = None
            continue

        src_path = os.path.abspath(str(src_path))

        if not os.path.exists(src_path):
            copied_paths[i] = None
            continue

        uuid = get_batch_item(
            batch,
            "uuid",
            i,
            default=f"batch{batch_idx}_idx{i}",
        )

        p2_rmsd = read_deepternary_metric_value(
            deepternary_info,
            "deepternary_pred_p2_rmsd_raw",
            i,
            default=-1.0,
        )

        clash_ratio = read_deepternary_metric_value(
            deepternary_info,
            "deepternary_clash_ratio_raw",
            i,
            default=-1.0,
        )

        dst_name = safe_path_token(
            f"difflinker_"
            f"batch_{batch_idx}_"
            f"sample_{sample_id}_"
            f"idx_{i}_"
            f"case_{uuid}_"
            f"rmsd_{p2_rmsd:.3f}_"
            f"clash_{clash_ratio:.3f}.pdb"
        )

        dst_path = os.path.join(complex_save_dir, dst_name)

        shutil.copyfile(src_path, dst_path)

        copied_paths[i] = dst_path

        if remove_source_dirs:
            sample_dir = os.path.dirname(os.path.dirname(src_path))

            if os.path.isdir(sample_dir):
                shutil.rmtree(sample_dir, ignore_errors=True)

    deepternary_info["deepternary_complex_pred_path"] = copied_paths

    return deepternary_info

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
        deepternary_complex_pred_path = None

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

            if "deepternary_complex_pred_path" in deepternary_info:
                complex_paths = deepternary_info["deepternary_complex_pred_path"]

                if isinstance(complex_paths, (list, tuple)) and i < len(complex_paths):
                    path_value = complex_paths[i]

                    if path_value not in [None, "", "None", "nan"]:
                        deepternary_complex_pred_path = str(path_value)

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
            "deepternary_complex_pred_path": deepternary_complex_pred_path,
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

@torch.no_grad()
def run_both(args: argparse.Namespace) -> None:
    run_generate(args)
    run_evaluate(args)
@torch.no_grad()
def run_generate(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    )

    utils.set_deterministic(args.seed)
    check_obabel_available()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_difflinker_model_for_test(
        checkpoint=args.checkpoint,
        device=device,
        batch_size=args.batch_size,
        data_path=args.data,
        test_data_prefix=args.test_data_prefix,
    )
    model.FRAMES = 1
    test_loader = model.val_dataloader()

    expected_debug_records = (
                                 min(len(test_loader), int(args.max_test_batches))
                                 if args.max_test_batches is not None
                                 else len(test_loader)
                             ) * int(model.batch_size) * int(args.n_samples)

    print(f"max_test_batches: {args.max_test_batches}")
    print(f"debug expected records upper bound: {expected_debug_records}")

    print("=" * 100)
    print("Final DiffLinker test evaluation")
    print(f"checkpoint: {args.checkpoint}")
    print(f"data_path: {model.data_path}")
    print(f"test_data_prefix via val_data_prefix: {model.val_data_prefix}")
    print(f"test batches: {len(test_loader)}")
    print(f"n_samples per entry: {args.n_samples}")
    print(f"batch_size: {model.batch_size}")
    print(f"device: {device}")
    print(f"out_dir: {args.out_dir}")
    print(
        "Hybrid reconstruction: template fragment bonds + "
        "OpenBabel linker-related bonds, no max component, no f1-f2 shortcut"
    )
    print("=" * 100)

    acc = ProtacMetricAccumulator()

    tmp_root = args.obabel_tmp_root
    if os.path.exists(tmp_root) and not args.keep_obabel_files:
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root, exist_ok=True)

    for batch_idx, batch_cpu in enumerate(tqdm(test_loader, desc="Test batches")):
        if args.max_test_batches is not None and batch_idx >= int(args.max_test_batches):
            break

        data = move_to_device(batch_cpu, device)
        require_keys(data, ["atom_mask", "anchor1_mask", "anchor2_mask", "linker_mask"])

        atom_mask_for_true = get_atom_mask_for_mol(model, data, data["atom_mask"])
        fragment_mask_for_true = get_fragment_mask_for_mol(model, data)

        required_template_keys = [
            "full_bond_types",
            "formal_charges",
            "atom_is_aromatic",
            "num_explicit_hs",
            "no_implicit",
            "fragment1_mask",
            "fragment2_mask",
            "fragment_mask",
        ]

        missing_template_keys = [k for k in required_template_keys if k not in data]
        if missing_template_keys:
            raise KeyError(
                f"Missing keys for template reconstruction: {missing_template_keys}. "
                "Need full_bond_types, atom property tensors, and fragment masks."
            )

        true_molecules = protac_build_molecules(
            one_hot=data["one_hot"],
            x=data["positions"],
            node_mask=atom_mask_for_true,
            fragment_bond_types=data["full_bond_types"],
            use_template_fragment_bonds=True,

            formal_charges=data["formal_charges"],
            atom_is_aromatic=data["atom_is_aromatic"],
            num_explicit_hs=data["num_explicit_hs"],
            no_implicit=data["no_implicit"],

            atom_props_mask=None,
            use_pred_bonds=False,
        )

        true_fragments = protac_build_molecules(
            one_hot=data["one_hot"],
            x=data["positions"],
            node_mask=fragment_mask_for_true,
            fragment_bond_types=data["full_bond_types"],
            use_template_fragment_bonds=True,

            formal_charges=data["formal_charges"],
            atom_is_aromatic=data["atom_is_aromatic"],
            num_explicit_hs=data["num_explicit_hs"],
            no_implicit=data["no_implicit"],

            atom_props_mask=data["fragment_mask"],
            use_pred_bonds=False,
        )

        for sample_id in range(args.n_samples):
            try:
                chain_batch, node_mask = model.sample_chain(
                    data,
                    keep_frames=model.FRAMES,
                )
            except utils.FoundNaNException as e:
                print(
                    f"[WARN] FoundNaNException at "
                    f"batch={batch_idx}, sample={sample_id}: {e}"
                )
                continue

            x, h = utils.split_features(
                z=chain_batch[0],
                n_dims=model.n_dims,
                num_classes=model.num_classes,
                include_charges=model.include_charges,
            )

            node_mask_for_mol = get_atom_mask_for_mol(model, data, node_mask)
            one_hot = h["categorical"]

            x_obabel = restore_com_like_difflinker_generate(
                model=model,
                data=data,
                x=x,
                node_mask=node_mask_for_mol,
            )

            tmp_dir = os.path.join(
                tmp_root,
                f"batch_{batch_idx:05d}_sample_{sample_id:03d}",
            )

            template_one_hot = make_hybrid_one_hot_for_template(
                true_one_hot=data["one_hot"],
                pred_one_hot=one_hot,
                fragment_mask=data["fragment_mask"],
            )

            pred_molecules = build_openbabel_molecules_from_prediction(
                one_hot=template_one_hot,
                x=x_obabel,
                node_mask=node_mask_for_mol,
                is_geom=model.is_geom,
                tmp_dir=tmp_dir,
                name_prefix=f"b{batch_idx:05d}_s{sample_id:03d}",
            )

            fragment_internal_bond_types = make_fragment_internal_bond_types(
                full_bond_types=data["full_bond_types"],
                fragment1_mask=data["fragment1_mask"],
                fragment2_mask=data["fragment2_mask"],
            )

            template_molecules = protac_build_molecules(
                one_hot=template_one_hot,
                x=x_obabel,
                node_mask=node_mask_for_mol,
                fragment_bond_types=fragment_internal_bond_types,
                use_template_fragment_bonds=True,

                formal_charges=data["formal_charges"],
                atom_is_aromatic=data["atom_is_aromatic"],
                num_explicit_hs=data["num_explicit_hs"],
                no_implicit=data["no_implicit"],

                atom_props_mask=data["fragment_mask"],
                use_pred_bonds=False,
            )

            pred_hybrid_molecules = []

            for i in range(int(data["atom_mask"].size(0))):
                valid_mask_i = node_mask_for_mol[i].squeeze().detach().cpu().bool()

                pred_hybrid = merge_template_fragments_with_openbabel_linker_bonds(
                    template_mol=template_molecules[i],
                    obabel_mol=pred_molecules[i],
                    fragment1_mask_i=data["fragment1_mask"][i][valid_mask_i],
                    fragment2_mask_i=data["fragment2_mask"][i][valid_mask_i],
                    linker_mask_i=data["linker_mask"][i][valid_mask_i],
                    anchor1_mask_i=data["anchor1_mask"][i][valid_mask_i],
                    anchor2_mask_i=data["anchor2_mask"][i][valid_mask_i],
                )

                if pred_hybrid is not None:
                    linker_mask_i = (
                        data["linker_mask"][i][valid_mask_i]
                        .detach()
                        .cpu()
                        .bool()
                    )

                    rwmol = Chem.RWMol(
                        Chem.Mol(pred_hybrid)
                    )

                    rwmol = infer_generated_atom_props_from_current_graph_inplace(
                        rwmol=rwmol,
                        editable_atom_mask=linker_mask_i,
                    )

                    pred_hybrid = rwmol.GetMol()

                pred_hybrid_molecules.append(pred_hybrid)

            pred_molecules = pred_hybrid_molecules

            batch_for_records = dict(data)
            batch_for_records["atom_mask"] = node_mask_for_mol
            acc.add_batch(
                pred_molecules=pred_molecules,
                batch=batch_for_records,
                true_molecules=true_molecules,
                true_fragments=true_fragments,
                batch_idx=batch_idx,
                sample_id=sample_id,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_records_pkl(
        acc.records,
        args.out_dir,
    )

    print("=" * 100)
    print("DiffLinker generation finished.")
    print(f"Generated records: {len(acc.records)}")
    print("=" * 100)
@torch.no_grad()
def run_evaluate(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device
        if torch.cuda.is_available() and "cuda" in args.device
        else "cpu"
    )

    utils.set_deterministic(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_difflinker_model_for_test(
        checkpoint=args.checkpoint,
        device=device,
        batch_size=args.batch_size,
        data_path=args.data,
        test_data_prefix=args.test_data_prefix,
    )
    model.FRAMES = 1

    reward_conf = load_reward_config(args.reward_config)

    if reward_conf is not None:
        reward_conf.experiment.seed = int(args.seed)

    protac_stan_reward = None
    deepternary_reward = None

    if reward_conf is not None and not args.no_protac_stan:
        protac_stan_reward, deepternary_reward = build_reward_models_from_conf(
            conf=reward_conf,
            device=device,
            use_deepternary=args.use_deepternary,
        )

    elif args.no_protac_stan:
        print(
            "[INFO] --no_protac_stan is set. "
            "Functional degradation metrics will be 0."
        )

    records = load_records_pkl(args.out_dir)

    groups = {}

    for rec in records:
        key = (
            int(rec["batch_idx"]),
            int(rec["sample_id"]),
        )

        groups.setdefault(key, []).append(rec)

    ordered_groups = sorted(
        groups.items(),
        key=lambda x: (x[0][0], x[0][1]),
    )

    acc = ProtacMetricAccumulator()
    sample_records = []

    model_name = "DiffLinker"
    checkpoint_path = str(args.checkpoint)

    for (batch_idx, sample_id), group_records in tqdm(
        ordered_groups,
        desc="Evaluate generated samples",
    ):
        pred_molecules = [
            rec["pred_mol"]
            for rec in group_records
        ]

        true_molecules = [
            rec["true_mol"]
            for rec in group_records
        ]

        true_fragments = [
            rec["frag_mol"]
            for rec in group_records
        ]

        strict_flags = []
        strict_infos = []
        smiles_for_stan = []

        for rec in group_records:
            strict, strict_info = is_strict_protac_mol(
                mol=rec["pred_mol"],
                anchor1_mask=rec["anchor1_mask"],
                anchor2_mask=rec["anchor2_mask"],
                linker_mask=rec["linker_mask"],
                fragment1_mask=rec["fragment1_mask"],
                fragment2_mask=rec["fragment2_mask"],
            )

            smi = safe_mol_to_smiles(
                rec["pred_mol"]
            )

            if smi is None:
                strict = False
                strict_info["strict_protac"] = 0.0

            strict_flags.append(bool(strict))
            strict_infos.append(strict_info)

            if strict:
                smiles_for_stan.append(smi)
            else:
                smiles_for_stan.append(None)

        if protac_stan_reward is not None:
            degradation_scores = protac_stan_reward.score(
                smiles_list=smiles_for_stan,
                poi_ids=[
                    rec["poi_uniprot"]
                    for rec in group_records
                ],
                e3_ids=[
                    rec["e3_uniprot"]
                    for rec in group_records
                ],
                step=0,
                batch_size=int(
                    args.protac_stan_batch_size
                ),
            )

            degradation_scores = torch.nan_to_num(
                degradation_scores,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        else:
            degradation_scores = torch.zeros(
                len(group_records),
                dtype=torch.float32,
                device=device,
            )

        if deepternary_reward is not None:
            ternary_scores, deepternary_info = deepternary_reward.score(
                mols=pred_molecules,
                valid_mask=strict_flags,
                names=[
                    rec["uuid"]
                    for rec in group_records
                ],
                p1_paths=[
                    rec["p1_path"]
                    for rec in group_records
                ],
                p2_paths=[
                    rec["p2_path"]
                    for rec in group_records
                ],
                unbound_lig1_paths=[
                    rec["unbound_lig1_path"]
                    for rec in group_records
                ],
                unbound_lig2_paths=[
                    rec["unbound_lig2_path"]
                    for rec in group_records
                ],
                lig1_mask_paths=[
                    rec["lig1_mask_path"]
                    for rec in group_records
                ],
                lig2_mask_paths=[
                    rec["lig2_mask_path"]
                    for rec in group_records
                ],
                step=0,
                max_samples=args.deepternary_max_samples,
                device=device,
            )

            ternary_scores = torch.nan_to_num(
                ternary_scores,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            required_dt_fields = [
                "deepternary_pred_p2_rmsd_raw",
                "deepternary_clash_ratio_raw",
                "deepternary_scored_mask",
                "deepternary_attempted_mask",
                "deepternary_valid_candidate_mask",
                "deepternary_context_eligible_mask",
            ]

            missing_dt_fields = [
                key
                for key in required_dt_fields
                if key not in deepternary_info
            ]

            if missing_dt_fields:
                raise KeyError(
                    "DeepTernaryReward missing required fields: "
                    f"{missing_dt_fields}"
                )

        else:
            ternary_scores = torch.zeros_like(
                degradation_scores
            )

            deepternary_info = {
                "deepternary_pred_p2_rmsd_raw":
                    torch.full_like(
                        degradation_scores,
                        -1.0,
                    ),
                "deepternary_clash_ratio_raw":
                    torch.full_like(
                        degradation_scores,
                        -1.0,
                    ),
                "deepternary_scored_mask":
                    torch.zeros_like(
                        degradation_scores
                    ),
                "deepternary_attempted_mask":
                    torch.zeros_like(
                        degradation_scores
                    ),
                "deepternary_valid_candidate_mask":
                    torch.zeros_like(
                        degradation_scores
                    ),
                "deepternary_context_eligible_mask":
                    torch.zeros_like(
                        degradation_scores
                    ),
                "deepternary_complex_pred_path": [
                    None
                    for _ in range(
                        len(group_records)
                    )
                ],
            }

        deepternary_info = copy_deepternary_complex_pdbs(
            deepternary_info=deepternary_info,
            save_dir=args.out_dir,
            batch={
                "uuid": [
                    rec["uuid"]
                    for rec in group_records
                ],
            },
            batch_idx=batch_idx,
            sample_id=sample_id,
            remove_source_dirs=True,
        )

        batch_for_records = {}

        for key in [
            "anchor1_mask",
            "anchor2_mask",
            "linker_mask",
            "atom_mask",
            "fragment1_mask",
            "fragment2_mask",
            "fragment_mask",
        ]:
            batch_for_records[key] = torch.stack(
                [
                    rec[key]
                    if torch.is_tensor(rec[key])
                    else torch.as_tensor(rec[key])
                    for rec in group_records
                ],
                dim=0,
            )

        for key in [
            "poi_uniprot",
            "e3_uniprot",
            "uuid",
            "p1_path",
            "p2_path",
            "unbound_lig1_path",
            "unbound_lig2_path",
            "lig1_mask_path",
            "lig2_mask_path",
        ]:
            batch_for_records[key] = [
                rec.get(key, None)
                for rec in group_records
            ]

        acc.add_batch(
            pred_molecules=pred_molecules,
            batch=batch_for_records,
            true_molecules=true_molecules,
            true_fragments=true_fragments,
            batch_idx=batch_idx,
            sample_id=sample_id,
            strict_flags=strict_flags,
            strict_infos=strict_infos,
            degradation_scores=degradation_scores,
            ternary_scores=ternary_scores,
            deepternary_info=deepternary_info,
        )

        batch_records = make_eval_records_for_batch(
            pred_molecules=pred_molecules,
            batch=batch_for_records,
            batch_idx=batch_idx,
            sample_id=sample_id,
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

        if len(acc.records) >= len(batch_records):
            start_idx = (
                len(acc.records)
                - len(batch_records)
            )

            for local_i, rec in enumerate(
                batch_records
            ):
                acc.records[
                    start_idx + local_i
                ][
                    "deepternary_complex_pred_path"
                ] = rec[
                    "deepternary_complex_pred_path"
                ]

        sample_records.extend(
            batch_records
        )

    reference_linker_smiles = load_smiles_list(
        args.reference_linker_smiles_path
    )

    reference_mol_smiles = load_smiles_list(
        args.reference_mol_smiles_path
    )

    metrics_out = acc.compute(
        reference_linker_smiles=reference_linker_smiles,
        reference_mol_smiles=reference_mol_smiles,
        novelty_on=args.novelty_on,
        recovery_on=args.recovery_on,
        uniqueness_group=args.uniqueness_group,
        sa_filter_mode=args.sa_filter_mode,
        sa_threshold=float(args.sa_threshold),
        degradation_threshold=float(
            args.degradation_threshold
        ),
        use_precomputed_rewards=True,
    )

    metrics_out["checkpoint"] = args.checkpoint
    metrics_out["test_data_prefix"] = str(
        model.val_data_prefix
    )
    metrics_out["n_samples"] = int(
        args.n_samples
    )
    metrics_out["reconstruction"] = (
        "template_fragment_bonds_plus_"
        "openbabel_linker_bonds_plus_shared_atom_props_no_f1_f2_shortcut"
    )

    save_metrics(
        metrics_out,
        args.out_dir,
    )

    save_diversity_summary(
        sample_records,
        args.out_dir,
    )

    save_records_pkl(
        acc.records,
        args.out_dir,
    )

    eval_csv_path = save_eval_records(
        sample_records,
        args.out_dir,
        filename="eval_samples.csv",
    )

    print(
        f"Per-sample eval records saved to: "
        f"{eval_csv_path}"
    )

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval_mode",
        type=str,
        default="generate",
        choices=[
            "generate",
            "evaluate",
            "both",
        ],
    )
    p.add_argument("--checkpoint", type=str, default="./models/protac_difflinker_root_protac_difflinker_scratch_bs16_date03-09_time00-30-37.635075_epoch=0849.ckpt", help="Selected/best DiffLinker .ckpt path.")
    p.add_argument("--data", type=str, default=None, help="Override checkpoint data_path if needed.")
    p.add_argument("--test_data_prefix", type=str, default="protac_rltrain_test")

    p.add_argument("--out_dir", type=str, default="./eval_outputs/difflinker_test_obabel2026")
    p.add_argument("--obabel_tmp_root", type=str, default="./tmp/difflinker_obabel_test")
    p.add_argument("--keep_obabel_files", action="store_true")

    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--max_test_batches",
        type=int,
        default=None,
        help="Only evaluate the first N test batches for debugging. Default: full test set.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)

    p.add_argument("--reference_linker_smiles_path", type=str, default="./datasets/reference_linkers.smi")
    p.add_argument("--reference_mol_smiles_path", type=str, default=None)
    p.add_argument("--novelty_on", type=str, default="linker", choices=["linker", "molecule"])
    p.add_argument("--recovery_on", type=str, default="linker", choices=["linker", "molecule"])
    p.add_argument("--uniqueness_group", type=str, default="case", choices=["case", "global"])
    p.add_argument("--sa_filter_mode", type=str, default="threshold", choices=["delinker", "threshold"])
    p.add_argument("--sa_threshold", type=float, default=6.0)

    p.add_argument("--reward_config", type=str, default="./configs/difflinker_eval_reward.yaml", help="Hydra/OmegaConf YAML containing PROTAC-STAN/DeepTernary paths.")
    p.add_argument("--no_protac_stan", action="store_true", default=False)
    p.add_argument("--degradation_threshold", type=float, default=0.5)
    p.add_argument("--protac_stan_batch_size", type=int, default=32)
    p.add_argument("--use_deepternary", type=str2bool, default=True)
    p.add_argument("--deepternary_max_samples", type=int, default=None)

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.eval_mode == "generate":
        run_generate(args)

    elif args.eval_mode == "evaluate":
        run_evaluate(args)

    elif args.eval_mode == "both":
        run_both(args)

    else:
        raise ValueError(
            f"Unknown eval_mode: {args.eval_mode}"
        )


if __name__ == "__main__":
    main()
