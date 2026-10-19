"""
Final test-set evaluation for pretrained / RL LinkerDiff checkpoints.

Run from project root, for example:

    python experiments/evaluate_protac_test.py \
        experiment.weight_path=./weights/best.pth \
        experiment.test_num_samples=100 \
        experiment.eval_dir=./eval_outputs/pretrained_on_rl_test

    python experiments/evaluate_protac_test.py \
        experiment.weight_path=./checkpoints/xxx_rl/rl_best.pth \
        experiment.test_num_samples=100 \
        experiment.eval_dir=./eval_outputs/rl_on_rl_test

This script uses the RL-stage test set because functional metrics need POI/E3 and
DeepTernary metadata. It does NOT train the model.
"""

import json
import os
import pickle
import sys
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make project root importable when running this file directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data import datasets
from data.datasets import collate
from data.molecule_builder import build_molecules
from src.protac_eval_metrics import ProtacMetricAccumulator, is_strict_protac_mol
import csv

from rdkit.Chem import QED, Descriptors, rdMolDescriptors, Crippen, Lipinski, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from experiments.train_rl import (
    RLExperiment,
    seed_everything,
    safe_mol_to_smiles,
)

RDLogger.DisableLog("rdApp.warning")


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

def subset_batch(batch, mask):
    """
    Select samples along the batch dimension.

    Only batch-major tensors whose first dimension equals
    the batch size are indexed. Flattened/non-batch tensors
    such as [B*N, 1] are kept unchanged.
    """

    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)

    mask = mask.bool()

    batch_size = int(mask.shape[0])

    indices = (
        mask.detach()
        .cpu()
        .nonzero(as_tuple=False)
        .view(-1)
        .tolist()
    )

    out = {}

    for key, value in batch.items():

        if torch.is_tensor(value):

            # Only tensors whose first dimension is the
            # current batch dimension should be subsetted.
            if (
                value.ndim > 0
                and int(value.shape[0]) == batch_size
            ):
                out[key] = value[
                    mask.to(value.device)
                ]
            else:
                # Examples:
                #   [B*N, 1]
                #   global/static tensors
                # These are not indexed by the sample mask.
                out[key] = value

        elif isinstance(value, list):

            if len(value) == batch_size:
                out[key] = [
                    value[i]
                    for i in indices
                ]
            else:
                out[key] = value

        elif isinstance(value, tuple):

            if len(value) == batch_size:
                out[key] = tuple(
                    value[i]
                    for i in indices
                )
            else:
                out[key] = value

        else:
            out[key] = value

    return out


def build_pred_molecules_from_rollout(rollout_out: Dict[str, Any], exp: RLExperiment) -> List[Chem.Mol]:
    """
    Rebuild generated molecules using exactly the same reconstruction settings as
    RLExperiment.compute_reward(...).
    """
    batch = rollout_out["batch"]
    final_xh = rollout_out["final_xh"]

    x = final_xh[:, :, :3]
    h = final_xh[:, :, 3:]

    x = x + rollout_out["anchor_com"] * batch["atom_mask"].float()

    pred_bond_probs = rollout_out["pred_bond_probs"]
    pred_bond_types = rollout_out["pred_bond_types"]

    build_kwargs = dict(
        one_hot=h,
        x=x,
        node_mask=batch["atom_mask"],
        anchors=batch["anchors"],
        linker_mask=batch["linker_mask"],
    )

    if pred_bond_probs is not None and pred_bond_types is not None:
        build_kwargs.update(
            dict(
                pred_bond_probs=pred_bond_probs,
                pred_bond_types=pred_bond_types,
                use_pred_bonds=True,
                pred_bond_threshold=exp.conf.experiment.rl_pred_bond_threshold,
                pred_bond_max_new_bonds_per_atom=exp.conf.experiment.rl_pred_bond_max_new_bonds_per_atom,
                pred_bond_max_dist=exp.conf.experiment.rl_pred_bond_max_dist,
                pred_bond_min_dist=exp.conf.experiment.rl_pred_bond_min_dist,
            )
        )

    if "fragment_bond_types" in batch:
        build_kwargs.update(
            dict(
                fragment_bond_types=batch["fragment_bond_types"],
                use_template_fragment_bonds=True,
            )
        )

    if all(k in batch for k in ["formal_charges", "atom_is_aromatic", "num_explicit_hs", "no_implicit"]):
        build_kwargs.update(
            dict(
                formal_charges=batch["formal_charges"],
                atom_is_aromatic=batch["atom_is_aromatic"],
                num_explicit_hs=batch["num_explicit_hs"],
                no_implicit=batch["no_implicit"],
                atom_props_mask=batch["fragment_mask"],
            )
        )

    return build_molecules(**build_kwargs)


def build_true_molecules_and_fragments(batch: Dict[str, Any]) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
    """
    Rebuild true full molecules and true fragment-only molecules from the batch.
    This follows your pretraining/evaluation reconstruction style.
    """
    fragment_mask = (
        batch["fragment1_mask"].bool() | batch["fragment2_mask"].bool()
    ).to(batch["atom_mask"].dtype)

    true_molecules = build_molecules(
        one_hot=batch["one_hot"],
        x=batch["positions"],
        node_mask=batch["atom_mask"],
        fragment_bond_types=batch["full_bond_types"],
        use_template_fragment_bonds=True,
        formal_charges=batch["formal_charges"],
        atom_is_aromatic=batch["atom_is_aromatic"],
        num_explicit_hs=batch["num_explicit_hs"],
        no_implicit=batch["no_implicit"],
        atom_props_mask=None,
        use_pred_bonds=False,
    )

    true_fragments = build_molecules(
        one_hot=batch["one_hot"],
        x=batch["positions"],
        node_mask=fragment_mask,
        fragment_bond_types=batch["full_bond_types"],
        use_template_fragment_bonds=True,
        formal_charges=batch["formal_charges"],
        atom_is_aromatic=batch["atom_is_aromatic"],
        num_explicit_hs=batch["num_explicit_hs"],
        no_implicit=batch["no_implicit"],
        atom_props_mask=None,
        use_pred_bonds=False,
    )

    return true_molecules, true_fragments


def load_reference_smiles(path: Optional[str]) -> Optional[List[str]]:
    """
    Optional helper for novelty reference sets.
    Accepts .txt/.smi with one SMILES per line, or .pkl containing list/set.
    """
    if path is None or str(path).lower() in {"none", "null", ""}:
        return None

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return list(obj)

    smiles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            smiles.append(line.split()[0])
    return smiles

def conf_to_bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, bool):
        return x

    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False

    return default
def get_eval_mode(conf):
    mode = str(conf.experiment.eval_mode).lower()
    if mode not in {"generate", "evaluate", "both"}:
        raise ValueError(
            f"Unknown eval_mode={mode}. "
            "Expected: generate / evaluate / both"
        )
    return mode

def create_test_loader(conf: DictConfig) -> DataLoader:
    exp_conf = conf.experiment

    test_dataset = datasets.RlDataset(
        data_path=exp_conf.data_path,
        dir="test",
        prefix=exp_conf.prefix_test,
        device="cpu",
    )

    test_batch_size = int(exp_conf.test_batch_size)

    return DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        collate_fn=collate,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
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
    """
    Copy DeepTernary complex_pred_*.pdb from tmp_root to:

        eval_outputs/test_rl_final/deepternary_complex_pdbs/

    Then replace deepternary_info["deepternary_complex_pred_path"] with
    the copied stable path.
    """
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

def save_metrics(metrics: Dict[str, Any], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "test_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    with open(os.path.join(save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_dir, "test_metrics.txt"), "w", encoding="utf-8") as f:
        for k in sorted(metrics.keys()):
            f.write(f"{k}: {metrics[k]}\n")
def save_records_pkl(records, save_dir):
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

    records_pkl_path = os.path.join(save_dir, "records.pkl")

    with open(records_pkl_path, "wb") as f:
        pickle.dump(
            make_pickle_safe_records(records),
            f,
        )

    print(f"Full RDKit records saved to: {records_pkl_path}")
    return records_pkl_path

def clone_conf(conf):
    return OmegaConf.create(
        OmegaConf.to_container(
            conf,
            resolve=False,
        )
    )


def run_both(conf: DictConfig) -> None:
    generate_conf = clone_conf(conf)
    evaluate_conf = clone_conf(conf)

    run_generate(generate_conf)
    run_evaluate(evaluate_conf)

def load_records_pkl(save_dir):
    records_pkl_path = os.path.join(save_dir, "records.pkl")
    print(
        f"Reading records.pkl from: {records_pkl_path}"
    )

    if not os.path.exists(records_pkl_path):
        raise FileNotFoundError(
            f"records.pkl not found: {records_pkl_path}"
        )

    with open(records_pkl_path, "rb") as f:
        records = pickle.load(f)

    if not isinstance(records, list):
        raise TypeError(
            f"Expected records.pkl to contain list, got {type(records)}"
        )

    print(f"Loaded records: {len(records)}")
    return records

def run_evaluate(conf: DictConfig) -> None:
    OmegaConf.set_struct(conf, False)

    seed = int(conf.experiment.seed)
    seed_everything(seed)

    conf.experiment.use_wandb = False
    conf.experiment.ckpt_dir = None

    # ============================================================
    # 1. 保持与原始 evaluate_protac_test.py 完全一致：
    #    先创建 RLExperiment，让它完成 eval_dir 等配置处理，
    #    同时获取 reward model、trained_steps 和 device
    # ============================================================
    exp = RLExperiment(conf=conf)

    save_dir = conf.experiment.eval_dir
    os.makedirs(save_dir, exist_ok=True)

    # ============================================================
    # 2. 读取 generate 阶段保存的 records.pkl
    # ============================================================
    records = load_records_pkl(save_dir)

    degradation_threshold = float(
        conf.experiment.degradation_threshold
    )

    deepternary_eval_max_samples = (
        conf.experiment.test_deepternary_max_samples
    )

    if (
        deepternary_eval_max_samples is not None
        and str(deepternary_eval_max_samples).lower()
        not in {"none", "null"}
    ):
        deepternary_eval_max_samples = int(
            deepternary_eval_max_samples
        )
    else:
        deepternary_eval_max_samples = None

    # ============================================================
    # 3. 按照原始代码的 batch_idx + sample_id 顺序分组
    # ============================================================
    groups = {}

    for rec in records:
        key = (
            int(rec.get("batch_idx", 0)),
            int(rec.get("sample_id", 0)),
        )

        if key not in groups:
            groups[key] = []

        groups[key].append(rec)

    ordered_groups = sorted(
        groups.items(),
        key=lambda x: (x[0][0], x[0][1]),
    )

    # 最终用于生成原来的 eval_samples.csv
    sample_records = []

    # ============================================================
    # 4. 按照原来的 batch/sample 粒度重新计算
    #    PROTAC-STAN / DeepTernary
    # ============================================================
    for (batch_idx, sample_id), group_records in ordered_groups:

        strict_flags = []
        strict_infos = []
        smiles_for_stan = []

        poi_ids = []
        e3_ids = []

        # --------------------------------------------------------
        # 4.1 重新计算 strict
        # --------------------------------------------------------
        for rec in group_records:

            strict, strict_info = is_strict_protac_mol(
                mol=rec["pred_mol"],
                anchor1_mask=rec["anchor1_mask"],
                anchor2_mask=rec["anchor2_mask"],
                linker_mask=rec["linker_mask"],
                fragment1_mask=rec["fragment1_mask"],
                fragment2_mask=rec["fragment2_mask"],
            )

            smi = safe_mol_to_smiles(rec["pred_mol"])

            # 与原代码一致：
            # SMILES 无法得到时，不送入 PROTAC-STAN
            if smi is None:
                strict = False
                strict_info["strict_protac"] = 0.0

            strict_flags.append(bool(strict))
            strict_infos.append(strict_info)

            if strict:
                smiles_for_stan.append(smi)
            else:
                smiles_for_stan.append(None)

            poi_ids.append(rec["poi_uniprot"])
            e3_ids.append(rec["e3_uniprot"])

        # ========================================================
        # 4.2 PROTAC-STAN
        #     与原 evaluate_protac_test.py 完全相同
        # ========================================================
        if exp.protac_stan_reward is not None:

            degradation_scores = exp.protac_stan_reward.score(
                smiles_list=smiles_for_stan,
                poi_ids=poi_ids,
                e3_ids=e3_ids,
                step=int(exp.trained_steps),
                batch_size=int(
                    conf.experiment.protac_stan_batch_size
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
                device=exp.device,
            )

        # ========================================================
        # 4.3 DeepTernary
        #     与原 evaluate_protac_test.py 完全相同
        # ========================================================
        if exp.deepternary_reward is not None:

            ternary_scores, deepternary_info = (
                exp.deepternary_reward.score(
                    mols=[
                        rec["pred_mol"]
                        for rec in group_records
                    ],
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
                    step=int(exp.trained_steps),
                    max_samples=deepternary_eval_max_samples,
                    device=exp.device,
                )
            )

            ternary_scores = torch.nan_to_num(
                ternary_scores,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        else:
            ternary_scores = torch.zeros(
                len(group_records),
                dtype=torch.float32,
                device=exp.device,
            )

            deepternary_info = {
                "deepternary_pred_p2_rmsd_raw":
                    torch.full(
                        (len(group_records),),
                        -1.0,
                    ),
                "deepternary_clash_ratio_raw":
                    torch.full(
                        (len(group_records),),
                        -1.0,
                    ),
                "deepternary_scored_mask":
                    torch.zeros(
                        len(group_records),
                    ),
                "deepternary_attempted_mask":
                    torch.zeros(
                        len(group_records),
                    ),
                "deepternary_valid_candidate_mask":
                    torch.zeros(
                        len(group_records),
                    ),
                "deepternary_complex_pred_path":
                    [
                        None
                        for _ in range(len(group_records))
                    ],
            }

        # ========================================================
        # 4.4 保持原来的 DeepTernary PDB 保存逻辑
        #
        # 原函数只真正需要 batch 中的 uuid，
        # 因此这里构造一个最小 batch 即可。
        # ========================================================
        deepternary_batch = {
            "uuid": [
                rec["uuid"]
                for rec in group_records
            ]
        }

        deepternary_info = copy_deepternary_complex_pdbs(
            deepternary_info=deepternary_info,
            save_dir=save_dir,
            batch=deepternary_batch,
            batch_idx=batch_idx,
            sample_id=sample_id,
            remove_source_dirs=True,
        )

        # ========================================================
        # 4.5 把重新计算后的结果写回原 records
        # ========================================================
        for i, rec in enumerate(group_records):

            rec["precomputed_strict"] = bool(
                strict_flags[i]
            )

            rec["precomputed_strict_info"] = dict(
                strict_infos[i]
            )

            rec["precomputed_degradation_score"] = float(
                degradation_scores[i]
            )

            rec["precomputed_ternary_score"] = float(
                ternary_scores[i]
            )

            rec["precomputed_deepternary_p2_rmsd"] = float(
                deepternary_info[
                    "deepternary_pred_p2_rmsd_raw"
                ][i]
            )

            rec["precomputed_deepternary_clash_ratio"] = float(
                deepternary_info[
                    "deepternary_clash_ratio_raw"
                ][i]
            )

            rec["precomputed_deepternary_scored"] = float(
                deepternary_info.get(
                    "deepternary_scored_mask",
                    torch.zeros(len(group_records)),
                )[i]
            )

            # ----------------------------------------------------
            # 以下字段与原 eval_samples.csv 保持一致
            # ----------------------------------------------------
            rec["strict_protac"] = int(
                strict_flags[i]
            )

            rec["raw_degradation_score"] = float(
                degradation_scores[i]
            )

            rec["gated_degradation_score"] = (
                float(degradation_scores[i])
                * float(strict_flags[i])
            )

            rec["high_degradation"] = int(
                strict_flags[i]
                and float(degradation_scores[i])
                >= degradation_threshold
            )

            rec["ternary_score"] = float(
                ternary_scores[i]
            )

            rec["deepternary_pred_p2_rmsd"] = float(
                deepternary_info[
                    "deepternary_pred_p2_rmsd_raw"
                ][i]
            )

            rec["deepternary_clash_ratio"] = float(
                deepternary_info[
                    "deepternary_clash_ratio_raw"
                ][i]
            )

            scored_mask = deepternary_info.get(
                "deepternary_scored_mask",
                None,
            )

            attempted_mask = deepternary_info.get(
                "deepternary_attempted_mask",
                None,
            )

            valid_candidate_mask = deepternary_info.get(
                "deepternary_valid_candidate_mask",
                None,
            )

            rec["precomputed_deepternary_attempted"] = (
                bool(float(attempted_mask[i]) > 0.5)
                if attempted_mask is not None
                else False
            )

            rec["precomputed_deepternary_valid_candidate"] = (
                bool(float(valid_candidate_mask[i]) > 0.5)
                if valid_candidate_mask is not None
                else False
            )

            rec["deepternary_scored"] = int(
                float(
                    scored_mask[i]
                ) > 0.5
            ) if scored_mask is not None else 0

            rec["deepternary_attempted"] = int(
                float(
                    attempted_mask[i]
                ) > 0.5
            ) if attempted_mask is not None else 0

            rec["deepternary_valid_candidate"] = int(
                float(
                    valid_candidate_mask[i]
                ) > 0.5
            ) if valid_candidate_mask is not None else 0

            rec["deepternary_complex_pred_path"] = (
                deepternary_info[
                    "deepternary_complex_pred_path"
                ][i]
                if isinstance(
                    deepternary_info.get(
                        "deepternary_complex_pred_path"
                    ),
                    (list, tuple),
                )
                else None
            )

        # ========================================================
        # 4.6 用与你原程序相同的 make_eval_records_for_batch
        #     生成 eval_samples.csv 数据
        # ========================================================
        synthetic_batch = {
            "uuid": [
                rec["uuid"]
                for rec in group_records
            ],
            "poi_uniprot": [
                rec["poi_uniprot"]
                for rec in group_records
            ],
            "e3_uniprot": [
                rec["e3_uniprot"]
                for rec in group_records
            ],
            "anchor1_mask": [
                rec["anchor1_mask"]
                for rec in group_records
            ],
            "anchor2_mask": [
                rec["anchor2_mask"]
                for rec in group_records
            ],
            "linker_mask": [
                rec["linker_mask"]
                for rec in group_records
            ],
            "atom_mask": [
                rec["atom_mask"]
                for rec in group_records
            ],
            "fragment1_mask": [
                rec["fragment1_mask"]
                for rec in group_records
            ],
            "fragment2_mask": [
                rec["fragment2_mask"]
                for rec in group_records
            ],
        }

        batch_records = make_eval_records_for_batch(
            pred_molecules=[
                rec["pred_mol"]
                for rec in group_records
            ],
            batch=synthetic_batch,
            batch_idx=batch_idx,
            sample_id=sample_id,
            model_name="ours",
            checkpoint_path=str(
                conf.experiment.weight_path
            ),
            degradation_scores=degradation_scores,
            ternary_scores=ternary_scores,
            deepternary_info=deepternary_info,
            strict_flags=strict_flags,
            strict_infos=strict_infos,
            degradation_threshold=degradation_threshold,
        )

        sample_records.extend(batch_records)

    # ============================================================
    # 5. 最终指标汇总
    #    使用 precomputed=True，与原程序完全一致
    # ============================================================
    acc = ProtacMetricAccumulator()
    acc.records = records

    reference_linker_smiles = load_reference_smiles(
        conf.experiment.reference_linker_smiles_path
    )

    reference_mol_smiles = load_reference_smiles(
        conf.experiment.reference_mol_smiles_path
    )

    metrics = acc.compute(
        reference_linker_smiles=reference_linker_smiles,
        reference_mol_smiles=reference_mol_smiles,
        novelty_on=conf.experiment.novelty_on,
        recovery_on=conf.experiment.recovery_on,
        uniqueness_group=conf.experiment.uniqueness_group,
        sa_filter_mode=conf.experiment.sa_filter_mode,
        sa_threshold=float(
            conf.experiment.sa_threshold
        ),
        degradation_threshold=degradation_threshold,
        use_precomputed_rewards=True,
    )

    # ============================================================
    # 6. 保存原来的全部输出文件
    # ============================================================
    save_metrics(
        metrics,
        save_dir,
    )

    save_diversity_summary(
        sample_records,
        save_dir,
    )

    save_records_pkl(
        records,
        save_dir,
    )

    eval_csv_path = save_eval_records(
        sample_records,
        save_dir,
        filename="eval_samples.csv",
    )

    print(
        f"Per-sample eval records saved to: {eval_csv_path}"
    )

    print("=" * 100)
    print("Evaluation finished.")
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]}")
    print("=" * 100)

def run_generate(conf: DictConfig) -> None:
    OmegaConf.set_struct(conf, False)

    seed = int(conf.experiment.seed)
    seed_everything(seed)

    conf.experiment.use_wandb = False
    conf.experiment.ckpt_dir = None

    n_samples = int(conf.experiment.test_num_samples)
    linker_size_offset = int(conf.experiment.linker_size_offset)

    exp = RLExperiment(conf=conf)
    exp.model.eval()

    test_loader = create_test_loader(conf)
    acc = ProtacMetricAccumulator()

    save_dir = conf.experiment.eval_dir
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch_cpu in enumerate(
            tqdm(test_loader, desc="Generate test samples")
        ):
            batch = batch_to_device(batch_cpu, exp.device)

            if linker_size_offset == 0:
                batch_eval = batch
            else:
                true_linker_sizes = (
                    batch["linker_mask"]
                    .sum(dim=1)
                    .view(-1)
                    .long()
                )

                valid_mask = (
                    true_linker_sizes + linker_size_offset > 0
                )

                if not valid_mask.any():
                    continue

                batch_eval = subset_batch(
                    batch,
                    valid_mask,
                )

            true_molecules, true_fragments = (
                build_true_molecules_and_fragments(batch_eval)
            )

            for sample_id in range(n_samples):

                rollout_out = exp.sample_pretrain_style_from_batch(
                    exp.model,
                    batch_eval,
                    linker_size_offset=linker_size_offset,
                )

                pred_molecules = build_pred_molecules_from_rollout(
                    rollout_out,
                    exp,
                )

                # 这里只保存分子和全部元数据，不运行 reward
                acc.add_batch(
                    pred_molecules=pred_molecules,
                    batch=rollout_out["batch"],
                    true_molecules=true_molecules,
                    true_fragments=true_fragments,
                    batch_idx=batch_idx,
                    sample_id=sample_id,
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_records_pkl(
        acc.records,
        save_dir,
    )

    print("=" * 100)
    print("Generation finished.")
    print("=" * 100)

@hydra.main(version_base=None, config_path="../configs", config_name="base_test")
def run(conf: DictConfig) -> None:
    mode = get_eval_mode(conf)

    if mode == "generate":
        run_generate(conf)

    elif mode == "evaluate":
        run_evaluate(conf)

    elif mode == "both":
        run_both(conf)

    else:
        raise ValueError(
            f"Unknown eval_mode: {mode}"
        )

if __name__ == "__main__":
    run()
