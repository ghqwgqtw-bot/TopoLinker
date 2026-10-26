from copy import deepcopy
import os

import sys

sys.path.append('.')
import argparse
import csv
import gc
import random
import pickle
import json
import shutil
import torch
import numpy as np
from itertools import cycle
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import (
    QED,
    Descriptors,
    rdMolDescriptors,
    Crippen,
    Lipinski,
    AllChem,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.loader import DataLoader
from collections import OrderedDict

from scripts.train_pl import DataModule
from models.maskfill import PMAsymDenoiser
from models.sample import seperate_outputs2, sample_loop3
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
from utils.sample_noise import get_sample_noiser
from protac_eval_metrics import (
    ProtacMetricAccumulator,
    build_reward_models_from_conf,
    is_strict_protac_mol,
)
from protac_molecule_builder import (
    build_molecules as protac_build_molecules,
    infer_generated_atom_props_from_current_graph_inplace,
)

def set_experiment_seed(seed):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got: {value}')


def print_pool_status(pool, logger):
    logger.info(
        '[Pool] Succ/Incomp/Bad: %d/%d/%d'
        % (len(pool.succ), len(pool.incomp), len(pool.bad))
    )


def make_pickle_safe_records(records):
    safe_records = []
    for rec in records:
        new_rec = {}
        for key, value in rec.items():
            new_rec[key] = value.detach().cpu() if torch.is_tensor(value) else value
        safe_records.append(new_rec)
    return safe_records


def load_records_pkl(save_dir):
    records_path = os.path.join(save_dir, 'records.pkl')
    if not os.path.exists(records_path):
        raise FileNotFoundError(f'records.pkl not found: {records_path}')

    with open(records_path, 'rb') as f:
        records = pickle.load(f)

    if not isinstance(records, list):
        raise TypeError(f'Expected records.pkl to contain list, got {type(records)}')

    print(f'Loaded records.pkl from: {os.path.abspath(records_path)}')
    print(f'Loaded records: {len(records)}')
    return records


def save_records_pkl(records, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    records_path = os.path.join(save_dir, 'records.pkl')
    with open(records_path, 'wb') as f:
        pickle.dump(make_pickle_safe_records(records), f)
    print(f'Full RDKit records saved to: {os.path.abspath(records_path)}')
    return records_path

def save_metrics(metrics_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    with open(
        os.path.join(save_dir, 'test_metrics.pkl'),
        'wb',
    ) as f:
        pickle.dump(metrics_dict, f)

    with open(
        os.path.join(save_dir, 'test_metrics.json'),
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            metrics_dict,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        os.path.join(save_dir, 'test_metrics.txt'),
        'w',
        encoding='utf-8',
    ) as f:
        for key in sorted(metrics_dict.keys()):
            f.write(f'{key}: {metrics_dict[key]}\n')

def load_reward_config(path):
    if path is None or str(path).lower() == 'none':
        return None
    from omegaconf import OmegaConf
    return OmegaConf.load(path)


def safe_mol_to_smiles(mol):
    if mol is None:
        return None
    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        smi = Chem.MolToSmiles(mol_copy)
        return smi if smi else None
    except Exception:
        return None


def load_smiles_list(path):
    if path is None or str(path).lower() == 'none':
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f'Reference SMILES file not found: {path}')

    smiles = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Support either "SMILES" or "SMILES <whitespace> optional_name".
            smi = line.split()[0]
            if smi:
                smiles.append(smi)

    print(f'Loaded reference SMILES: {len(smiles)} from {os.path.abspath(path)}')
    return smiles

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
def build_fragment_template_with_placeholders(src):
    """
    Build the fixed-fragment template with the shared protac_molecule_builder.

    Fragment atom identities, coordinates, internal bonds, and atom properties
    come from the source record. Linker slots are retained as dummy carbons with
    zero coordinates and no bonds so the original atom indices are preserved.
    """
    required_keys = [
        'one_hot', 'positions', 'fragment1_mask', 'fragment2_mask', 'linker_mask',
        'fragment_bond_types', 'formal_charges', 'atom_is_aromatic',
        'num_explicit_hs', 'no_implicit',
    ]
    missing_keys = [key for key in required_keys if key not in src]
    if missing_keys:
        raise KeyError(f'Missing source keys for fragment template: {missing_keys}')

    fragment1_mask = torch.as_tensor(src['fragment1_mask']).detach().cpu().bool().view(-1)
    fragment2_mask = torch.as_tensor(src['fragment2_mask']).detach().cpu().bool().view(-1)
    linker_mask = torch.as_tensor(src['linker_mask']).detach().cpu().bool().view(-1)
    fragment_mask = fragment1_mask | fragment2_mask

    source_one_hot = torch.as_tensor(src['one_hot']).detach().cpu().float()
    source_pos = torch.as_tensor(src['positions']).detach().cpu().float()
    source_fragment_bonds = torch.as_tensor(src['fragment_bond_types']).detach().cpu().long()
    n_atoms = int(source_one_hot.shape[0])

    if source_one_hot.ndim != 2:
        raise RuntimeError(f'Expected source one_hot [N,C], got {tuple(source_one_hot.shape)}')
    if source_pos.shape != (n_atoms, 3):
        raise RuntimeError(f'positions shape {tuple(source_pos.shape)} != ({n_atoms}, 3)')
    if source_fragment_bonds.shape != (n_atoms, n_atoms):
        raise RuntimeError(
            f'fragment_bond_types shape {tuple(source_fragment_bonds.shape)} '
            f'!= ({n_atoms}, {n_atoms})'
        )
    if fragment_mask.numel() != n_atoms:
        raise RuntimeError('fragment_mask size does not match source atom count')
    if linker_mask.numel() != n_atoms:
        raise RuntimeError('linker_mask size does not match source atom count')
    if not torch.equal(fragment_mask | linker_mask, torch.ones(n_atoms, dtype=torch.bool)):
        raise RuntimeError('fragment/linker masks do not cover all atoms')
    if bool((fragment_mask & linker_mask).any().item()):
        raise RuntimeError('fragment and linker masks overlap')

    # Keep GT atom types and coordinates only for fragment atoms.
    template_one_hot = torch.zeros_like(source_one_hot)
    template_one_hot[:, 0] = 1.0
    template_one_hot[fragment_mask] = source_one_hot[fragment_mask]

    template_pos = torch.zeros_like(source_pos)
    template_pos[fragment_mask] = source_pos[fragment_mask]

    # Keep only fragment1-fragment1 and fragment2-fragment2 template bonds.
    fragment_pair_mask = (
        (fragment1_mask[:, None] & fragment1_mask[None, :])
        | (fragment2_mask[:, None] & fragment2_mask[None, :])
    )
    template_fragment_bonds = (
        source_fragment_bonds * fragment_pair_mask.to(source_fragment_bonds.dtype)
    )

    # Copy source atom properties only for fragment atoms.
    source_formal_charges = torch.as_tensor(src['formal_charges']).detach().cpu().long().view(-1)
    source_atom_is_aromatic = torch.as_tensor(src['atom_is_aromatic']).detach().cpu().bool().view(-1)
    source_num_explicit_hs = torch.as_tensor(src['num_explicit_hs']).detach().cpu().long().view(-1)
    source_no_implicit = torch.as_tensor(src['no_implicit']).detach().cpu().bool().view(-1)

    template_formal_charges = torch.zeros(n_atoms, dtype=torch.long)
    template_atom_is_aromatic = torch.zeros(n_atoms, dtype=torch.bool)
    template_num_explicit_hs = torch.zeros(n_atoms, dtype=torch.long)
    template_no_implicit = torch.zeros(n_atoms, dtype=torch.bool)

    template_formal_charges[fragment_mask] = source_formal_charges[fragment_mask]
    template_atom_is_aromatic[fragment_mask] = source_atom_is_aromatic[fragment_mask]
    template_num_explicit_hs[fragment_mask] = source_num_explicit_hs[fragment_mask]
    template_no_implicit[fragment_mask] = source_no_implicit[fragment_mask]

    # Preserve every original atom slot so PocketXMol halfedge indices stay valid.
    node_mask = torch.ones(1, n_atoms, 1, dtype=torch.float32)

    template_molecules = protac_build_molecules(
        one_hot=template_one_hot.unsqueeze(0),
        x=template_pos.unsqueeze(0),
        node_mask=node_mask,
        fragment_bond_types=template_fragment_bonds.unsqueeze(0),
        use_template_fragment_bonds=True,
        formal_charges=template_formal_charges.unsqueeze(0),
        atom_is_aromatic=template_atom_is_aromatic.unsqueeze(0),
        num_explicit_hs=template_num_explicit_hs.unsqueeze(0),
        no_implicit=template_no_implicit.unsqueeze(0),
        atom_props_mask=fragment_mask.unsqueeze(0),
        use_pred_bonds=False,
    )

    if len(template_molecules) != 1:
        raise RuntimeError(f'Expected one template molecule, got {len(template_molecules)}')

    template_mol = template_molecules[0]
    if template_mol is None:
        return None
    if template_mol.GetNumAtoms() != n_atoms:
        raise RuntimeError(
            f'Template molecule atom count {template_mol.GetNumAtoms()} '
            f'!= source atom count {n_atoms}'
        )
    return template_mol

def fill_pocketxmol_linker_into_template(
    *,
    template_mol,
    atomic_numbers,
    positions,
    bond_types,
    linker_mask,
    anchor1_mask,
    anchor2_mask,
):
    """
    Fill PocketXMol-generated linker information into the fragment template.

    Fragment atoms/properties/internal bonds remain unchanged. Only linker atom
    identities, coordinates, linker-linker bonds, and designated anchor-linker
    bonds from PocketXMol are inserted. No distance inference, OpenBabel, repair,
    or largest-component extraction is used.
    """
    if template_mol is None:
        return None

    try:
        atomic_numbers = torch.as_tensor(atomic_numbers).detach().cpu().long().view(-1)
        positions = torch.as_tensor(positions).detach().cpu().float()
        bond_types = torch.as_tensor(bond_types).detach().cpu().long()
        linker_mask = torch.as_tensor(linker_mask).detach().cpu().bool().view(-1)
        anchor1_mask = torch.as_tensor(anchor1_mask).detach().cpu().bool().view(-1)
        anchor2_mask = torch.as_tensor(anchor2_mask).detach().cpu().bool().view(-1)
        anchor_mask = anchor1_mask | anchor2_mask
        n_atoms = int(template_mol.GetNumAtoms())

        if atomic_numbers.numel() != n_atoms:
            return None
        if positions.shape != (n_atoms, 3):
            return None
        if bond_types.shape != (n_atoms, n_atoms):
            return None
        if linker_mask.numel() != n_atoms:
            return None

        # Start from the shared fragment template and remove any unexpected
        # linker-related bonds without touching fragment-fragment bonds.
        rwmol = Chem.RWMol(Chem.Mol(template_mol))
        bonds_to_remove = []
        for bond in rwmol.GetBonds():
            atom_i = int(bond.GetBeginAtomIdx())
            atom_j = int(bond.GetEndAtomIdx())
            if bool(linker_mask[atom_i].item()) or bool(linker_mask[atom_j].item()):
                bonds_to_remove.append((atom_i, atom_j))

        for atom_i, atom_j in bonds_to_remove:
            rwmol.RemoveBond(int(atom_i), int(atom_j))

        # Replace dummy linker carbons with PocketXMol-predicted atom identities.
        linker_indices = torch.where(linker_mask)[0]
        for atom_idx_tensor in linker_indices:
            atom_idx = int(atom_idx_tensor.item())
            atomic_number = int(atomic_numbers[atom_idx].item())
            if atomic_number <= 0:
                return None

            new_atom = Chem.Atom(atomic_number)
            new_atom.SetFormalCharge(0)
            new_atom.SetIsAromatic(False)
            new_atom.SetNumExplicitHs(0)
            new_atom.SetNoImplicit(False)
            rwmol.ReplaceAtom(atom_idx, new_atom, preserveProps=False)

        if rwmol.GetNumConformers() == 0:
            return None

        # Fragment coordinates are source coordinates; linker coordinates are
        # PocketXMol outputs restored to the same global coordinate frame.
        conformer = rwmol.GetConformer()
        for atom_idx in range(n_atoms):
            xyz = positions[atom_idx]
            conformer.SetAtomPosition(
                int(atom_idx),
                (float(xyz[0].item()), float(xyz[1].item()), float(xyz[2].item())),
            )

        # Add only linker-related bonds selected from PocketXMol predictions.
        rdkit_bond_types = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.AROMATIC,
        }

        for atom_i in range(n_atoms):
            for atom_j in range(atom_i):
                i_is_linker = bool(linker_mask[atom_i].item())
                j_is_linker = bool(linker_mask[atom_j].item())

                i_is_anchor = bool(anchor_mask[atom_i].item())
                j_is_anchor = bool(anchor_mask[atom_j].item())

                is_linker_linker = (
                        i_is_linker and j_is_linker
                )
                is_anchor_linker = (
                        (i_is_anchor and j_is_linker)
                        or
                        (j_is_anchor and i_is_linker)
                )

                if not (is_linker_linker or is_anchor_linker):
                    continue

                bond_type_idx = int(bond_types[atom_i, atom_j].item())
                if bond_type_idx == 0:
                    continue
                if bond_type_idx not in rdkit_bond_types:
                    return None
                if rwmol.GetBondBetweenAtoms(int(atom_i), int(atom_j)) is not None:
                    return None

                if bond_type_idx == 4:
                    rwmol.GetAtomWithIdx(int(atom_i)).SetIsAromatic(True)
                    rwmol.GetAtomWithIdx(int(atom_j)).SetIsAromatic(True)

                rwmol.AddBond(
                    int(atom_i),
                    int(atom_j),
                    rdkit_bond_types[bond_type_idx],
                )

                if bond_type_idx == 4:
                    new_bond = rwmol.GetBondBetweenAtoms(int(atom_i), int(atom_j))
                    if new_bond is not None:
                        new_bond.SetIsAromatic(True)

        rwmol = infer_generated_atom_props_from_current_graph_inplace(
            rwmol=rwmol,
            editable_atom_mask=linker_mask,
        )

        try:
            rwmol.UpdatePropertyCache(strict=False)
        except Exception:
            pass

        return rwmol.GetMol()

    except Exception:
        return None

def build_source_reference_molecules(src):
    """
    Build GT reference molecules with the same shared
    protac_molecule_builder used by the DiffLinker baseline.

    true_mol:
        all source atoms + full GT bonds + GT atom properties

    true_frag_mol:
        only fragment1 + fragment2 atoms, using the same full GT graph
        restricted by node_mask.
    """
    required_keys = [
        'one_hot',
        'positions',
        'full_bond_types',
        'fragment1_mask',
        'fragment2_mask',
        'formal_charges',
        'atom_is_aromatic',
        'num_explicit_hs',
        'no_implicit',
    ]

    missing_keys = [key for key in required_keys if key not in src]
    if missing_keys:
        raise KeyError(
            f'Missing source reference keys: {missing_keys}'
        )

    one_hot = (
        torch.as_tensor(src['one_hot'])
        .detach()
        .cpu()
        .float()
    )

    positions = (
        torch.as_tensor(src['positions'])
        .detach()
        .cpu()
        .float()
    )

    full_bond_types = (
        torch.as_tensor(src['full_bond_types'])
        .detach()
        .cpu()
        .long()
    )

    fragment1_mask = (
        torch.as_tensor(src['fragment1_mask'])
        .detach()
        .cpu()
        .bool()
        .view(-1)
    )

    fragment2_mask = (
        torch.as_tensor(src['fragment2_mask'])
        .detach()
        .cpu()
        .bool()
        .view(-1)
    )

    fragment_mask = fragment1_mask | fragment2_mask

    formal_charges = (
        torch.as_tensor(src['formal_charges'])
        .detach()
        .cpu()
        .long()
        .view(-1)
    )

    atom_is_aromatic = (
        torch.as_tensor(src['atom_is_aromatic'])
        .detach()
        .cpu()
        .bool()
        .view(-1)
    )

    num_explicit_hs = (
        torch.as_tensor(src['num_explicit_hs'])
        .detach()
        .cpu()
        .long()
        .view(-1)
    )

    no_implicit = (
        torch.as_tensor(src['no_implicit'])
        .detach()
        .cpu()
        .bool()
        .view(-1)
    )

    n_atoms = int(one_hot.shape[0])

    if one_hot.ndim != 2:
        raise RuntimeError(
            f'GT one_hot must be [N,C], got {tuple(one_hot.shape)}'
        )

    if positions.shape != (n_atoms, 3):
        raise RuntimeError(
            f'GT positions shape {tuple(positions.shape)} '
            f'!= ({n_atoms}, 3)'
        )

    if full_bond_types.shape != (n_atoms, n_atoms):
        raise RuntimeError(
            f'GT full_bond_types shape {tuple(full_bond_types.shape)} '
            f'!= ({n_atoms}, {n_atoms})'
        )

    if fragment_mask.numel() != n_atoms:
        raise RuntimeError(
            'GT fragment_mask size does not match atom count'
        )

    # Retain all source atoms in the full reference molecule.
    full_node_mask = torch.ones(
        1,
        n_atoms,
        1,
        dtype=torch.float32,
    )

    # Fragment reference keeps only fragment1 + fragment2 atoms.
    fragment_node_mask = (
        fragment_mask
        .float()
        .view(1, n_atoms, 1)
    )

    true_molecules = protac_build_molecules(
        one_hot=one_hot.unsqueeze(0),
        x=positions.unsqueeze(0),
        node_mask=full_node_mask,
        fragment_bond_types=full_bond_types.unsqueeze(0),
        use_template_fragment_bonds=True,

        formal_charges=formal_charges.unsqueeze(0),
        atom_is_aromatic=atom_is_aromatic.unsqueeze(0),
        num_explicit_hs=num_explicit_hs.unsqueeze(0),
        no_implicit=no_implicit.unsqueeze(0),

        atom_props_mask=None,
        use_pred_bonds=False,
    )

    true_fragments = protac_build_molecules(
        one_hot=one_hot.unsqueeze(0),
        x=positions.unsqueeze(0),
        node_mask=fragment_node_mask,
        fragment_bond_types=full_bond_types.unsqueeze(0),
        use_template_fragment_bonds=True,

        formal_charges=formal_charges.unsqueeze(0),
        atom_is_aromatic=atom_is_aromatic.unsqueeze(0),
        num_explicit_hs=num_explicit_hs.unsqueeze(0),
        no_implicit=no_implicit.unsqueeze(0),

        atom_props_mask=fragment_mask.unsqueeze(0),
        use_pred_bonds=False,
    )

    if len(true_molecules) != 1:
        raise RuntimeError(
            f'Expected one true molecule, got {len(true_molecules)}'
        )

    if len(true_fragments) != 1:
        raise RuntimeError(
            f'Expected one true fragment molecule, '
            f'got {len(true_fragments)}'
        )

    true_mol = true_molecules[0]
    true_frag_mol = true_fragments[0]

    if true_mol is None:
        raise RuntimeError(
            'Shared protac_build_molecules failed for GT true_mol.'
        )

    if true_frag_mol is None:
        raise RuntimeError(
            'Shared protac_build_molecules failed for GT frag_mol.'
        )

    return true_mol, true_frag_mol

def load_pocketxmol_model_for_evaluate(args, device):
    """
    Initialize the PocketXMol generation model before reward-model
    initialization, matching the evaluation-stage ordering used by
    the DiffLinker baseline.

    No sampling is performed here.
    """

    eval_config = make_config(
        args.config_task,
        args.config_model,
    )

    ckpt = torch.load(
        eval_config.model.checkpoint,
        map_location=device,
        weights_only=False,
    )

    cfg_dir = os.path.dirname(
        str(eval_config.model.checkpoint)
    ).replace(
        'checkpoints',
        'train_config',
    )

    train_config_files = os.listdir(cfg_dir)

    train_config = make_config(
        os.path.join(
            cfg_dir,
            ''.join(train_config_files),
        )
    )

    dm = DataModule(train_config)

    in_dims = dm.get_in_dims()

    if train_config.model.name != 'pm_asym_denoiser':
        raise ValueError(
            f'Unsupported PocketXMol model: '
            f'{train_config.model.name}'
        )

    model = PMAsymDenoiser(
        config=train_config.model,
        **in_dims,
    ).to(device)

    model.load_state_dict({
        k[6:]: value
        for k, value in ckpt['state_dict'].items()
        if k.startswith('model.')
    })

    model.eval()

    return model, eval_config
@torch.no_grad()
def run_evaluate(args):
    device = torch.device(
        args.device if torch.cuda.is_available() and 'cuda' in str(args.device) else 'cpu'
    )
    set_experiment_seed(int(args.seed))
    os.makedirs(args.outdir, exist_ok=True)

    model, eval_config = load_pocketxmol_model_for_evaluate(
        args=args,
        device=device,
    )

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
            '[INFO] --no_protac_stan is set. '
            'Functional degradation metrics will be disabled.'
        )

    records = load_records_pkl(args.outdir)
    if len(records) == 0:
        raise RuntimeError('records.pkl contains no records.')

    groups = {}
    for rec in records:
        if 'batch_idx' not in rec:
            raise KeyError('Evaluation record is missing batch_idx.')
        if 'sample_id' not in rec:
            raise KeyError('Evaluation record is missing sample_id.')

        key = (int(rec['batch_idx']), int(rec['sample_id']))
        groups.setdefault(key, []).append(rec)

    ordered_groups = sorted(groups.items(), key=lambda x: (x[0][0], x[0][1]))
    print(f'Loaded evaluation groups: {len(ordered_groups)}')
    print(f'Loaded evaluation records: {len(records)}')

    acc = ProtacMetricAccumulator()

    sample_records = []

    model_name = 'PocketXMol'

    checkpoint_path = str(
        eval_config.model.checkpoint
    )

    for (batch_idx, sample_id), group_records in tqdm(
        ordered_groups,
        desc='Evaluate generated samples',
    ):
        pred_molecules = [rec['pred_mol'] for rec in group_records]
        true_molecules = [rec['true_mol'] for rec in group_records]
        true_fragments = [rec['frag_mol'] for rec in group_records]

        strict_flags = []
        strict_infos = []
        smiles_for_stan = []

        for rec in group_records:
            strict, strict_info = is_strict_protac_mol(
                mol=rec['pred_mol'],
                anchor1_mask=rec['anchor1_mask'],
                anchor2_mask=rec['anchor2_mask'],
                linker_mask=rec['linker_mask'],
                fragment1_mask=rec['fragment1_mask'],
                fragment2_mask=rec['fragment2_mask'],
            )
            smi = safe_mol_to_smiles(rec['pred_mol'])

            if smi is None:
                strict = False
                strict_info['strict_protac'] = 0.0

            strict_flags.append(bool(strict))
            strict_infos.append(strict_info)
            smiles_for_stan.append(smi if strict else None)

        if protac_stan_reward is not None:
            degradation_scores = protac_stan_reward.score(
                smiles_list=smiles_for_stan,
                poi_ids=[rec['poi_uniprot'] for rec in group_records],
                e3_ids=[rec['e3_uniprot'] for rec in group_records],
                step=0,
                batch_size=int(args.protac_stan_batch_size),
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
                names=[rec['uuid'] for rec in group_records],
                p1_paths=[rec['p1_path'] for rec in group_records],
                p2_paths=[rec['p2_path'] for rec in group_records],
                unbound_lig1_paths=[rec['unbound_lig1_path'] for rec in group_records],
                unbound_lig2_paths=[rec['unbound_lig2_path'] for rec in group_records],
                lig1_mask_paths=[rec['lig1_mask_path'] for rec in group_records],
                lig2_mask_paths=[rec['lig2_mask_path'] for rec in group_records],
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
                'deepternary_pred_p2_rmsd_raw',
                'deepternary_clash_ratio_raw',
                'deepternary_scored_mask',
                'deepternary_attempted_mask',
                'deepternary_valid_candidate_mask',
                'deepternary_context_eligible_mask',
            ]
            missing_dt_fields = [
                key for key in required_dt_fields if key not in deepternary_info
            ]
            if missing_dt_fields:
                raise KeyError(
                    f'DeepTernaryReward missing required fields: {missing_dt_fields}'
                )
        else:
            ternary_scores = torch.zeros_like(degradation_scores)
            deepternary_info = {
                'deepternary_pred_p2_rmsd_raw': torch.full_like(degradation_scores, -1.0),
                'deepternary_clash_ratio_raw': torch.full_like(degradation_scores, -1.0),
                'deepternary_scored_mask': torch.zeros_like(degradation_scores),
                'deepternary_attempted_mask': torch.zeros_like(degradation_scores),
                'deepternary_valid_candidate_mask': torch.zeros_like(degradation_scores),
                'deepternary_context_eligible_mask': torch.zeros_like(degradation_scores),
            }

        cleanup_deepternary_temp_dirs(
            deepternary_info
        )

        batch_for_records = {}
        for key in [
            'anchor1_mask',
            'anchor2_mask',
            'linker_mask',
            'atom_mask',
            'fragment1_mask',
            'fragment2_mask',
            'fragment_mask',
        ]:
            batch_for_records[key] = [
                rec[key] if torch.is_tensor(rec[key]) else torch.as_tensor(rec[key])
                for rec in group_records
            ]

        for key in [
            'poi_uniprot',
            'e3_uniprot',
            'uuid',
            'p1_path',
            'p2_path',
            'unbound_lig1_path',
            'unbound_lig2_path',
            'lig1_mask_path',
            'lig2_mask_path',
        ]:
            batch_for_records[key] = [rec.get(key, None) for rec in group_records]

        acc.add_batch(
            pred_molecules=pred_molecules,
            batch=batch_for_records,
            true_molecules=true_molecules,
            true_fragments=true_fragments,
            batch_idx=int(batch_idx),
            sample_id=int(sample_id),
            strict_flags=strict_flags,
            strict_infos=strict_infos,
            degradation_scores=degradation_scores,
            ternary_scores=ternary_scores,
            deepternary_info=deepternary_info,
        )

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

    metrics_out = acc.compute(
        reference_linker_smiles=reference_linker_smiles,
        reference_mol_smiles=reference_mol_smiles,
        novelty_on=args.novelty_on,
        recovery_on=args.recovery_on,
        uniqueness_group=args.uniqueness_group,
        sa_filter_mode=args.sa_filter_mode,
        sa_threshold=float(args.sa_threshold),
        degradation_threshold=float(args.degradation_threshold),
        use_precomputed_rewards=True,
    )
    metrics_out['model_name'] = 'PocketXMol'

    metrics_out['seed'] = int(args.seed)

    metrics_out['checkpoint'] = checkpoint_path

    metrics_out['test_data_prefix'] = str(
        args.source_pt
    )

    metrics_out['reconstruction'] = (
        'true_fragment_template_plus_'
        'native_pocketxmol_linker_coordinates_'
        'atom_types_and_linker_related_bonds_'
        'plus_shared_atom_props_no_f1_f2_shortcut'
    )

    save_metrics(
        metrics_out,
        args.outdir,
    )

    save_diversity_summary(
        sample_records,
        args.outdir,
    )

    save_records_pkl(
        acc.records,
        args.outdir,
    )

    eval_csv_path = save_eval_records(
        sample_records,
        args.outdir,
        filename='eval_samples.csv',
    )

    print(
        f'Per-sample eval records saved to: '
        f'{eval_csv_path}'
    )

    print(
        f'Scored evaluation records: '
        f'{len(acc.records)}'
    )

    print(
        f'Final metrics saved under: '
        f'{os.path.abspath(args.outdir)}'
    )

    return metrics_out

is_vscode = False
if os.environ.get("TERM_PROGRAM") == "vscode":
    is_vscode = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval_mode',
        type=str,
        default='evaluate',
        choices=['generate', 'evaluate', 'both'],
        help=(
            'generate: sample PocketXMol and save records.pkl; '
            'evaluate: evaluate an existing records.pkl; '
            'both: generate first, then evaluate.'
        ),
    )

    # Unified evaluation settings
    parser.add_argument('--seed', type=int, default=2026, help='Seed')
    parser.add_argument('--reference_linker_smiles_path', type=str, default='./datasets/reference_linkers.smi')
    parser.add_argument('--reference_mol_smiles_path', type=str, default=None)
    parser.add_argument('--novelty_on', type=str, default='linker', choices=['linker', 'molecule'])
    parser.add_argument('--recovery_on', type=str, default='linker', choices=['linker', 'molecule'])
    parser.add_argument('--uniqueness_group', type=str, default='case', choices=['case', 'global'])
    parser.add_argument('--sa_filter_mode', type=str, default='threshold', choices=['delinker', 'threshold'])
    parser.add_argument('--sa_threshold', type=float, default=6.0)
    parser.add_argument(
        '--reward_config',
        type=str,
        default='./configs/pocketxmol_eval_reward.yaml',
        help='OmegaConf YAML containing PROTAC-STAN and DeepTernary evaluation settings.',
    )
    parser.add_argument('--no_protac_stan', action='store_true', default=False, help='Disable PROTAC-STAN evaluation.')
    parser.add_argument('--degradation_threshold', type=float, default=0.5, help='PROTAC-STAN threshold for defining high predicted degradation.')
    parser.add_argument('--protac_stan_batch_size', type=int, default=32, help='Batch size used by PROTAC-STAN evaluation.')
    parser.add_argument('--use_deepternary', type=str2bool, default=True, help='Enable DeepTernary evaluation.')
    parser.add_argument('--deepternary_max_samples', type=int, default=None, help='Optional maximum number of samples passed to DeepTernary per evaluation call.')
    parser.add_argument(
        '--config_task',
        type=str,
        default='configs/sample/test/linking_topolinker_rltest/fixed_fragpos.yml',
        help='task config file',
    )
    parser.add_argument(
        '--config_model',
        type=str,
        default='eval_outputs/pocketxmol_finetune_val_selection/_model_configs/pxm_step_000555.yml',
        help='model config file',
    )
    parser.add_argument('--outdir', type=str, default='eval_outputs/pocketxmol_outputs_test2026_555')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument(
        '--source_pt',
        type=str,
        default='data/protac_rltrain_test.pt',
        help='Original TopoLinker test .pt used to build PocketXMol inputs.',
    )
    parser.add_argument(
        '--mapping_csv',
        type=str,
        default='data/topolinker_rltest/conversion_mapping.csv',
        help='data_id -> orig_index mapping produced by topolinker_to_pocketxmol.py.',
    )
    args = parser.parse_args()

    if args.eval_mode == 'evaluate':
        run_evaluate(args)
        raise SystemExit(0)

    # Load original TopoLinker test contexts and PocketXMol mapping.
    source_records = torch.load(
        args.source_pt,
        map_location='cpu',
        weights_only=False,
    )

    if not isinstance(source_records, list):
        raise TypeError(
            f'Expected source_pt to contain a list, got {type(source_records)}'
        )
    if len(source_records) == 0:
        raise ValueError('source_pt is empty')

    data_id_to_source = {}

    with open(
            args.mapping_csv,
            'r',
            encoding='utf-8-sig',
            newline='',
    ) as f:
        reader = csv.DictReader(f)

        required_columns = {'data_id', 'orig_index', 'uuid'}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise KeyError(
                f'Mapping CSV missing columns: {sorted(missing_columns)}'
            )

        for row in reader:
            data_id = str(row['data_id'])
            orig_index = int(row['orig_index'])

            if orig_index < 0 or orig_index >= len(source_records):
                raise IndexError(
                    f'{data_id}: orig_index={orig_index} outside '
                    f'source range [0, {len(source_records) - 1}]'
                )

            if data_id in data_id_to_source:
                raise ValueError(
                    f'Duplicate data_id in mapping CSV: {data_id}'
                )

            src = source_records[orig_index]
            mapping_uuid = str(row['uuid'])
            source_uuid = str(src['uuid'])

            if mapping_uuid != source_uuid:
                raise ValueError(
                    f'UUID mismatch for {data_id}: '
                    f'mapping={mapping_uuid}, source={source_uuid}'
                )

            data_id_to_source[data_id] = {
                'orig_index': orig_index,
                'uuid': source_uuid,
                'record': src,
            }

    if len(data_id_to_source) != len(source_records):
        raise ValueError(
            f'Mapping size {len(data_id_to_source)} '
            f'!= source size {len(source_records)}'
        )

    print(
        f'[OK] Loaded source contexts: {len(source_records)}; '
        f'data_id mappings: {len(data_id_to_source)}'
    )

    # Load configs.
    config = make_config(args.config_task, args.config_model)

    if args.config_model is not None:
        config_name = os.path.basename(
            args.config_task
        ).replace('.yml', '')
        config_name += '_' + os.path.basename(
            args.config_model
        ).replace('.yml', '')
    else:
        config_name = os.path.basename(args.config_task)[
                      :os.path.basename(args.config_task).rfind('.')
                      ]

    seed = int(args.seed)

    # args.seed is the single source of truth for this run.
    # Also overwrite config.sample.seed so any downstream PocketXMol
    # component reading the config sees exactly the same seed.
    config.sample.seed = seed

    set_experiment_seed(seed)

    print(f'[Seed] generation seed: {seed}')

    # Audit the exact files used by this evaluation.
    source_pt_abs = os.path.abspath(args.source_pt)
    mapping_csv_abs = os.path.abspath(args.mapping_csv)
    config_task_abs = os.path.abspath(args.config_task)
    config_model_abs = os.path.abspath(args.config_model)
    dataset_root_abs = os.path.abspath(
        str(config.data.dataset.root)
    )
    assembly_path_abs = os.path.abspath(
        os.path.join(
            dataset_root_abs,
            str(config.data.dataset.assembly_path),
        )
    )
    checkpoint_abs = os.path.abspath(
        str(config.model.checkpoint)
    )
    train_config_dir_abs = os.path.abspath(
        os.path.dirname(
            str(config.model.checkpoint)
        ).replace(
            'checkpoints',
            'train_config',
        )
    )

    print()
    print('=' * 100)
    print('[POCKETXMOL PROTAC DATA AUDIT]')
    print('=' * 100)
    print(f'Source PT:       {source_pt_abs}')
    print(f'Mapping CSV:     {mapping_csv_abs}')
    print(f'Task config:     {config_task_abs}')
    print(f'Model config:    {config_model_abs}')
    print(f'Dataset root:    {dataset_root_abs}')
    print(f'Assembly CSV:    {assembly_path_abs}')
    print(f'Checkpoint:      {checkpoint_abs}')
    print(f'Train cfg dir:   {train_config_dir_abs}')
    print()
    print(f'Source contexts: {len(source_records)}')
    print(f'Mapping entries: {len(data_id_to_source)}')

    if len(data_id_to_source) > 0:
        mapping_keys = list(data_id_to_source.keys())
        print(f'First data_id:   {mapping_keys[0]}')
        print(f'Last data_id:    {mapping_keys[-1]}')

    print()
    print('[PocketXMol dataset DB paths]')

    resolved_db_paths = []

    for db_cfg in config.data.dataset.dbs:
        db_name = str(db_cfg.name)

        lmdb_root_abs = os.path.abspath(
            os.path.join(
                dataset_root_abs,
                str(db_cfg.lmdb_root),
            )
        )
        mols_lmdb_abs = os.path.abspath(
            os.path.join(
                lmdb_root_abs,
                str(db_cfg.lmdb_path.mols),
            )
        )
        linking_lmdb_abs = os.path.abspath(
            os.path.join(
                lmdb_root_abs,
                str(db_cfg.lmdb_path.linking),
            )
        )

        print(f'DB internal name: {db_name}')
        print(f'LMDB root:        {lmdb_root_abs}')
        print(f'mols.lmdb:        {mols_lmdb_abs}')
        print(f'linking.lmdb:     {linking_lmdb_abs}')

        resolved_db_paths.append({
            'db_name': db_name,
            'lmdb_root': lmdb_root_abs,
            'mols': mols_lmdb_abs,
            'linking': linking_lmdb_abs,
        })

    print('=' * 100)

    # Hard checks for the converted TopoLinker test dataset.
    expected_dataset_name = os.path.basename(
        os.path.dirname(mapping_csv_abs)
    )

    allowed_dataset_names = {
        'topolinker_rltest',
        'topolinker_pretrain',
    }

    if expected_dataset_name not in allowed_dataset_names:
        raise RuntimeError(
            f'Unexpected converted dataset name: '
            f'{expected_dataset_name}. '
            f'Expected one of: {sorted(allowed_dataset_names)}'
        )

    if expected_dataset_name not in assembly_path_abs:
        raise RuntimeError(
            'Assembly path does not point to the converted user test set:\n'
            f'{assembly_path_abs}'
        )

    for item in resolved_db_paths:
        if expected_dataset_name not in item['lmdb_root']:
            raise RuntimeError(
                'LMDB path does not point to the converted user test set:\n'
                f'{item["lmdb_root"]}'
            )

    required_files = [
        source_pt_abs,
        mapping_csv_abs,
        config_task_abs,
        config_model_abs,
        assembly_path_abs,
        checkpoint_abs,
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Required evaluation file does not exist:\n{path}'
            )

    for item in resolved_db_paths:
        for path in [item['mols'], item['linking']]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f'Required PocketXMol LMDB does not exist:\n{path}'
                )

    print()
    print('[OK] All audited paths point to the converted TopoLinker test dataset.')
    print('[OK] No official PocketXMol benchmark data path was selected.')

    # Load checkpoint and training config.
    ckpt = torch.load(
        config.model.checkpoint,
        map_location=args.device,
        weights_only=False,
    )

    cfg_dir = os.path.dirname(
        config.model.checkpoint
    ).replace(
        'checkpoints',
        'train_config',
    )

    train_config = os.listdir(cfg_dir)
    train_config = make_config(
        os.path.join(
            cfg_dir,
            ''.join(train_config),
        )
    )

    pocketxmol_atomic_numbers = [
        int(z)
        for z in train_config.transforms.featurizer.chem.atomic_numbers
    ]

    num_pocketxmol_atom_types = len(
        pocketxmol_atomic_numbers
    )

    print(
        f'[PocketXMol atom vocabulary] '
        f'{pocketxmol_atomic_numbers}'
    )

    batch_size = (
        config.sample.batch_size
        if args.batch_size == 0
        else args.batch_size
    )

    num_mols = getattr(
        config.sample,
        'num_mols',
        int(1e10),
    )

    num_repeats = getattr(
        config.sample,
        'num_repeats',
        1,
    )


    # Console-only logging.
    class ConsoleLogger:
        @staticmethod
        def info(msg, *args):
            if args:
                msg = msg % args
            print(msg)

        @staticmethod
        def warning(msg, *args):
            if args:
                msg = msg % args
            print(f'[WARNING] {msg}')


    logger = ConsoleLogger()
    logger.info(
        'Load from %s...'
        % config.model.checkpoint
    )
    logger.info(args)
    logger.info(config)

    # Transforms.
    logger.info('Loading data placeholder...')

    for samp_trans in config.get('transforms', {}).keys():
        if samp_trans in train_config.transforms.keys():
            train_config.transforms.get(
                samp_trans
            ).update(
                config.transforms.get(samp_trans)
            )

    dm = DataModule(train_config)
    featurizer_list = dm.get_featurizers()
    in_dims = dm.get_in_dims()
    task_trans = get_transforms(config.task.transform)
    is_ar = config.task.transform.name

    noiser = get_sample_noiser(
        config.noise,
        in_dims['num_node_types'],
        in_dims['num_edge_types'],
        mode='sample',
        device=args.device,
        ref_config=train_config.noise,
    )

    if 'variable_mol_size' in getattr(
            config,
            'transforms',
            [],
    ):
        transforms = featurizer_list + [
            get_transforms(
                config.transforms.variable_mol_size
            ),
            task_trans,
        ]
    else:
        transforms = featurizer_list + [task_trans]

    addition_transforms = [
        get_transforms(tr)
        for tr in config.data.get(
            'transforms',
            [],
        )
    ]

    transforms = Compose(
        transforms + addition_transforms
    )

    follow_batch = sum([
        getattr(t, 'follow_batch', [])
        for t in transforms.transforms
    ], [])

    exclude_keys = sum([
        getattr(t, 'exclude_keys', [])
        for t in transforms.transforms
    ], [])

    # Data loader.
    logger.info('Loading dataset...')

    data_cfg = config.data

    num_workers = (
        train_config.train.num_workers
        if args.num_workers == -1
        else args.num_workers
    )

    test_set = TestTaskDataset(
        data_cfg.dataset,
        config.task,
        mode='test',
        split=getattr(
            data_cfg,
            'split',
            None,
        ),
        transforms=transforms,
    )

    test_loader = DataLoader(
        test_set,
        batch_size,
        shuffle=args.shuffle,
        num_workers=num_workers,
        pin_memory=train_config.train.pin_memory,
        follow_batch=follow_batch,
        exclude_keys=exclude_keys,
    )

    # Model.
    logger.info('Loading diffusion model...')

    if train_config.model.name == 'pm_asym_denoiser':
        model = PMAsymDenoiser(
            config=train_config.model,
            **in_dims,
        ).to(args.device)

    model.load_state_dict({
        k[6:]: value
        for k, value in ckpt['state_dict'].items()
        if k.startswith('model.')
    })

    model.eval()

    pool = EasyDict({
        'succ': [],
        'bad': [],
        'incomp': [],
    })

    acc = ProtacMetricAccumulator()
    reference_mol_cache = {}

    info_keys = [
        'data_id',
        'db',
        'task',
        'key',
    ]

    i_saved = 0

    logger.info(
        'Start sampling... '
        '(n_repeats=%d, n_mols=%d)'
        % (
            num_repeats,
            num_mols,
        )
    )

    # Batch-first sampling:
    # For each batch, independently sample the same contexts num_repeats times
    # before moving to the next batch.
    #
    # The current fixed-fragment PROTAC config does not use repeat-dependent
    # dataset transforms. Reject them explicitly rather than silently changing
    # their semantics.
    if (
            'overwrite_pos_repeat' in config.sample
            or 'overwrite_mol_repeat' in config.sample
    ):
        raise RuntimeError(
            'Batch-first sampling does not support '
            'overwrite_pos_repeat / overwrite_mol_repeat.'
        )

    for i_batch, batch_base in enumerate(test_loader):
        if i_saved >= num_mols:
            logger.info('Enough molecules. Stop sampling.')
            break

        logger.info(
            f'Generating batch {i_batch}: '
            f'{num_repeats} independent samples per context.'
        )

        for i_repeat in range(num_repeats):
            if i_saved >= num_mols:
                logger.info('Enough molecules. Stop sampling.')
                break

            logger.info(
                f'Generating batch {i_batch}, '
                f'sample {i_repeat + 1}/{num_repeats}.'
            )

            # IMPORTANT:
            # Every repeat must start from the same original conditioning batch.
            # sample_loop3 modifies its input batch in-place, so never reuse
            # the generated batch from the previous repeat.
            batch = batch_base.clone().to(args.device)

            batch, outputs, trajs = sample_loop3(
                batch,
                model,
                noiser,
                args.device,
                is_ar=is_ar,
            )

            data_list = [
                {key: batch[key][i] for key in info_keys}
                for i in range(len(batch))
            ]

            generated_list, outputs_list, traj_list_dict = seperate_outputs2(
                batch,
                outputs,
                trajs,
            )

            if len(generated_list) != len(data_list):
                raise RuntimeError(
                    f'generated_list size {len(generated_list)} '
                    f'!= data_list size {len(data_list)}'
                )

            mol_info_list = []

            for i_mol in tqdm(
                    range(len(generated_list)),
                    desc='Post process generated mols',
            ):
                sample_meta = data_list[i_mol]
                data_id = str(sample_meta['data_id'])

                if data_id not in data_id_to_source:
                    raise KeyError(
                        f'PocketXMol generated unknown data_id: {data_id}'
                    )

                source_entry = data_id_to_source[data_id]
                orig_index = int(source_entry['orig_index'])
                source_uuid = str(source_entry['uuid'])
                src = source_entry['record']

                # Build each reference context once and reuse RDKit copies.
                if orig_index not in reference_mol_cache:
                    reference_true_mol, reference_frag_mol = (
                        build_source_reference_molecules(src)
                    )

                    if reference_true_mol is None:
                        raise RuntimeError(
                            f'{data_id}: failed to build GT true_mol.'
                        )

                    if reference_frag_mol is None:
                        raise RuntimeError(
                            f'{data_id}: failed to build GT frag_mol.'
                        )

                    reference_mol_cache[orig_index] = (
                        reference_true_mol,
                        reference_frag_mol,
                    )

                cached_true_mol, cached_frag_mol = reference_mol_cache[orig_index]
                true_mol = Chem.Mol(cached_true_mol)
                true_frag_mol = Chem.Mol(cached_frag_mol)

                # Keep raw PocketXMol indexing; do not call decode_output().
                generated_raw = generated_list[i_mol]
                n_source = int(src['num_atoms'])
                n_generated = int(generated_raw['node'].shape[0])

                if n_generated != n_source:
                    raise RuntimeError(
                        f'{data_id}: generated atom count '
                        f'{n_generated} != source atom count {n_source}'
                    )

                mol_info = dict(sample_meta)
                mol_info['orig_index'] = orig_index
                mol_info['source_uuid'] = source_uuid
                mol_info['data_id'] = data_id
                mol_info['i_repeat'] = int(i_repeat)
                mol_info['i_batch'] = int(i_batch)

                # Restore linker coordinates from anchor-centered to global frame.
                source_pos = torch.as_tensor(
                    src['positions']
                ).detach().cpu().float()

                if source_pos.shape != (n_source, 3):
                    raise RuntimeError(
                        f'{data_id}: source positions shape '
                        f'{tuple(source_pos.shape)} != ({n_source}, 3)'
                    )

                fragment1_mask = (
                    torch.as_tensor(src['fragment1_mask'])
                    .detach()
                    .cpu()
                    .bool()
                    .view(-1)
                )
                fragment2_mask = (
                    torch.as_tensor(src['fragment2_mask'])
                    .detach()
                    .cpu()
                    .bool()
                    .view(-1)
                )
                linker_mask = (
                    torch.as_tensor(src['linker_mask'])
                    .detach()
                    .cpu()
                    .bool()
                    .view(-1)
                )
                anchor1_mask = (
                    torch.as_tensor(src['anchor1_mask'])
                    .detach()
                    .cpu()
                    .bool()
                    .view(-1)
                )
                anchor2_mask = (
                    torch.as_tensor(src['anchor2_mask'])
                    .detach()
                    .cpu()
                    .bool()
                    .view(-1)
                )

                fragment_mask = fragment1_mask | fragment2_mask

                if int(anchor1_mask.sum()) != 1:
                    raise RuntimeError(
                        f'{data_id}: anchor1_mask must contain exactly one atom'
                    )

                if int(anchor2_mask.sum()) != 1:
                    raise RuntimeError(
                        f'{data_id}: anchor2_mask must contain exactly one atom'
                    )

                anchor1_idx = int(
                    torch.nonzero(anchor1_mask, as_tuple=False)
                    .view(-1)[0]
                    .item()
                )
                anchor2_idx = int(
                    torch.nonzero(anchor2_mask, as_tuple=False)
                    .view(-1)[0]
                    .item()
                )

                anchor_center = source_pos[
                    [anchor1_idx, anchor2_idx]
                ].mean(
                    dim=0,
                    keepdim=True,
                )

                generated_pos_centered = (
                    torch.as_tensor(
                        generated_raw['pos']
                    )
                    .detach()
                    .cpu()
                    .float()
                )

                if generated_pos_centered.shape != (n_source, 3):
                    raise RuntimeError(
                        f'{data_id}: generated positions shape '
                        f'{tuple(generated_pos_centered.shape)} '
                        f'!= ({n_source}, 3)'
                    )

                generated_pos_global = generated_pos_centered + anchor_center

                hybrid_pos = source_pos.clone()
                hybrid_pos[linker_mask] = generated_pos_global[linker_mask]

                # Decode PocketXMol node classes only for linker atoms.
                generated_node_type = (
                    torch.as_tensor(
                        generated_raw['node']
                    )
                    .detach()
                    .cpu()
                    .long()
                    .view(-1)
                )

                if generated_node_type.numel() != n_source:
                    raise RuntimeError(
                        f'{data_id}: generated node count '
                        f'{generated_node_type.numel()} '
                        f'!= source atom count {n_source}'
                    )

                source_atomic_numbers = (
                    torch.as_tensor(src['charges'])
                    .detach()
                    .cpu()
                    .round()
                    .long()
                    .view(-1)
                )

                if source_atomic_numbers.numel() != n_source:
                    raise RuntimeError(
                        f'{data_id}: source atomic-number count '
                        f'{source_atomic_numbers.numel()} != {n_source}'
                    )

                # --------------------------------------------------------
                # Decode PocketXMol linker node types while preserving
                # the original TopoLinker atom indexing.
                #
                # Every linker slot starts as a dummy carbon placeholder.
                #
                # real PXM atom type:
                #     replace the dummy C with the predicted element
                #
                # PXM mask / no-atom:
                #     keep the dummy C only as an index placeholder
                #     and later discard every predicted bond touching it
                # --------------------------------------------------------

                hybrid_atomic_numbers = source_atomic_numbers.clone()

                # All linker slots are carbon placeholders first.
                hybrid_atomic_numbers[linker_mask] = 6

                # Whether the PocketXMol output says that this slot
                # corresponds to a real generated atom.
                generated_node_exists = torch.ones(
                    n_source,
                    dtype=torch.bool,
                )

                linker_indices = torch.where(
                    linker_mask
                )[0]

                linker_node_types = generated_node_type[
                    linker_indices
                ]

                linker_node_valid = (
                        (linker_node_types >= 0)
                        & (linker_node_types < num_pocketxmol_atom_types)
                )

                for local_idx, atom_idx_tensor in enumerate(
                        linker_indices
                ):
                    atom_idx = int(
                        atom_idx_tensor.item()
                    )

                    node_type = int(
                        linker_node_types[
                            local_idx
                        ].item()
                    )

                    if bool(
                            linker_node_valid[
                                local_idx
                            ].item()
                    ):
                        # Real PocketXMol atom:
                        # replace the dummy carbon by the predicted element.
                        hybrid_atomic_numbers[
                            atom_idx
                        ] = int(
                            pocketxmol_atomic_numbers[
                                node_type
                            ]
                        )

                    else:
                        # PocketXMol predicts mask / no atom.
                        #
                        # Keep C only as an index-preserving placeholder.
                        # No predicted bond involving this slot will be used.
                        hybrid_atomic_numbers[
                            atom_idx
                        ] = 6

                        generated_node_exists[
                            atom_idx
                        ] = False

                n_missing_linker_atoms = int(
                    (
                        ~generated_node_exists[
                            linker_mask
                        ]
                    ).sum().item()
                )

                if n_missing_linker_atoms > 0:
                    missing_linker_indices = (
                        torch.where(
                            linker_mask
                            & (~generated_node_exists)
                        )[0]
                        .tolist()
                    )

                    logger.warning(
                        f'{data_id}: PocketXMol predicted '
                        f'{n_missing_linker_atoms} linker slot(s) as '
                        f'mask/no-atom at indices '
                        f'{missing_linker_indices}; '
                        f'these slots are retained as isolated '
                        f'carbon placeholders only.'
                    )

                # Fragment bonds come from source; linker-related bonds come from PXM.
                source_fragment_bonds = (
                    torch.as_tensor(src['fragment_bond_types'])
                    .detach()
                    .cpu()
                    .long()
                )

                if source_fragment_bonds.shape != (n_source, n_source):
                    raise RuntimeError(
                        f'{data_id}: fragment_bond_types shape '
                        f'{tuple(source_fragment_bonds.shape)} '
                        f'!= ({n_source}, {n_source})'
                    )

                fragment_pair_mask = (
                        (fragment1_mask[:, None] & fragment1_mask[None, :])
                        | (fragment2_mask[:, None] & fragment2_mask[None, :])
                )

                hybrid_bond_types = (
                        source_fragment_bonds
                        * fragment_pair_mask.to(source_fragment_bonds.dtype)
                )

                generated_halfedge = (
                    torch.as_tensor(
                        generated_raw['halfedge']
                    )
                    .detach()
                    .cpu()
                    .long()
                    .view(-1)
                )

                generated_halfedge_index = (
                    torch.as_tensor(
                        generated_raw['halfedge_index']
                    )
                    .detach()
                    .cpu()
                    .long()
                )

                if (
                        generated_halfedge_index.ndim != 2
                        or generated_halfedge_index.shape[0] != 2
                ):
                    raise RuntimeError(
                        f'{data_id}: halfedge_index has invalid shape '
                        f'{tuple(generated_halfedge_index.shape)}'
                    )

                if generated_halfedge_index.shape[1] != generated_halfedge.numel():
                    raise RuntimeError(
                        f'{data_id}: halfedge count mismatch: '
                        f'halfedge={generated_halfedge.numel()}, '
                        f'halfedge_index={generated_halfedge_index.shape[1]}'
                    )

                if generated_halfedge_index.numel() > 0:
                    min_index = int(generated_halfedge_index.min().item())
                    max_index = int(generated_halfedge_index.max().item())

                    if min_index < 0 or max_index >= n_source:
                        raise RuntimeError(
                            f'{data_id}: halfedge_index outside '
                            f'atom range [0, {n_source - 1}]'
                        )

                n_linker_linker_bonds = 0
                n_anchor1_linker_bonds = 0
                n_anchor2_linker_bonds = 0
                n_ignored_predicted_bonds = 0
                n_unresolved_halfedges = 0

                for i_edge in range(generated_halfedge.numel()):
                    bond_type = int(
                        generated_halfedge[
                            i_edge
                        ].item()
                    )

                    atom_i = int(
                        generated_halfedge_index[
                            0,
                            i_edge,
                        ].item()
                    )

                    atom_j = int(
                        generated_halfedge_index[
                            1,
                            i_edge,
                        ].item()
                    )

                    # No predicted bond.
                    if bond_type == 0:
                        continue

                    # ----------------------------------------------------
                    # PocketXMol mask/no-atom semantics.
                    #
                    # If either endpoint was predicted as no atom,
                    # discard this edge completely.
                    #
                    # The carbon at that slot exists only to preserve
                    # the original atom index.
                    # ----------------------------------------------------
                    if (
                            not bool(
                                generated_node_exists[
                                    atom_i
                                ].item()
                            )
                            or
                            not bool(
                                generated_node_exists[
                                    atom_j
                                ].item()
                            )
                    ):
                        n_ignored_predicted_bonds += 1
                        continue

                    # PocketXMol bond coding:
                    # 0 none
                    # 1 single
                    # 2 double
                    # 3 triple
                    # 4 aromatic
                    if bond_type not in (1, 2, 3, 4):
                        n_unresolved_halfedges += 1
                        continue

                    i_is_linker = bool(linker_mask[atom_i].item())
                    j_is_linker = bool(linker_mask[atom_j].item())
                    is_linker_linker = i_is_linker and j_is_linker
                    is_anchor1_linker = (
                            (atom_i == anchor1_idx and j_is_linker)
                            or (atom_j == anchor1_idx and i_is_linker)
                    )
                    is_anchor2_linker = (
                            (atom_i == anchor2_idx and j_is_linker)
                            or (atom_j == anchor2_idx and i_is_linker)
                    )

                    if is_linker_linker:
                        hybrid_bond_types[atom_i, atom_j] = bond_type
                        hybrid_bond_types[atom_j, atom_i] = bond_type
                        n_linker_linker_bonds += 1

                    elif is_anchor1_linker:
                        # Aromatic anchor-linker bonds are skipped, never converted.
                        if bond_type == 4:
                            n_ignored_predicted_bonds += 1
                            continue
                        hybrid_bond_types[atom_i, atom_j] = bond_type
                        hybrid_bond_types[atom_j, atom_i] = bond_type
                        n_anchor1_linker_bonds += 1

                    elif is_anchor2_linker:
                        if bond_type == 4:
                            n_ignored_predicted_bonds += 1
                            continue
                        hybrid_bond_types[atom_i, atom_j] = bond_type
                        hybrid_bond_types[atom_j, atom_i] = bond_type
                        n_anchor2_linker_bonds += 1

                    else:
                        # Ignore non-anchor fragment-linker, fragment1-fragment2,
                        # and fragment-internal PocketXMol predictions.
                        n_ignored_predicted_bonds += 1

                # --------------------------------------------------------
                # Unified reconstruction:
                #
                # 1. Build the fixed fragment template.
                # 2. Every linker slot initially exists as a dummy carbon.
                # 3. Real PocketXMol linker nodes replace their dummy C.
                # 4. Mask/no-atom linker slots remain isolated dummy C.
                # 5. Only bonds between existing PocketXMol atoms are used.
                #
                # This preserves the original atom indexing while ensuring
                # that a predicted no-atom slot can never contribute to
                # PROTAC connectivity.
                # --------------------------------------------------------

                fragment_template_mol = (
                    build_fragment_template_with_placeholders(
                        src
                    )
                )

                if fragment_template_mol is None:
                    rdmol = None

                else:
                    rdmol = (
                        fill_pocketxmol_linker_into_template(
                            template_mol=fragment_template_mol,
                            atomic_numbers=hybrid_atomic_numbers,
                            positions=hybrid_pos,
                            bond_types=hybrid_bond_types,
                            linker_mask=linker_mask,
                            anchor1_mask=anchor1_mask,
                            anchor2_mask=anchor2_mask,
                        )
                    )

                # Runtime status only; final validity is computed by the common evaluator.
                if rdmol is None:
                    smiles = ''
                    tag = 'bad'
                    pool.bad.append(mol_info)
                    logger.warning(f'{data_id}: unified hybrid reconstruction failed.')
                else:
                    try:
                        smiles = Chem.MolToSmiles(rdmol, canonical=True)
                    except Exception:
                        smiles = ''

                    try:
                        components = Chem.GetMolFrags(
                            rdmol,
                            asMols=False,
                            sanitizeFrags=False,
                        )
                        is_connected = len(components) == 1
                    except Exception:
                        is_connected = False

                    if is_connected:
                        tag = ''
                        pool.succ.append(mol_info)
                        logger.info(f'Hybrid molecule built: {data_id}')
                    else:
                        tag = 'incomp'
                        pool.incomp.append(mol_info)
                        logger.warning(f'{data_id}: hybrid molecule is disconnected.')

                atom_mask = torch.ones(n_source, dtype=torch.bool)

                mol_info.update({
                    'rdmol': rdmol,
                    'true_mol': true_mol,
                    'frag_mol': true_frag_mol,
                    'smiles': smiles,
                    'tag': tag,
                    'anchor1_mask': anchor1_mask.clone(),
                    'anchor2_mask': anchor2_mask.clone(),
                    'linker_mask': linker_mask.clone(),
                    'atom_mask': atom_mask,
                    'fragment1_mask': fragment1_mask.clone(),
                    'fragment2_mask': fragment2_mask.clone(),
                    'fragment_mask': fragment_mask.clone(),
                    'uuid': src.get('uuid', source_uuid),
                    'poi_uniprot': src.get('poi_uniprot', None),
                    'e3_uniprot': src.get('e3_uniprot', None),
                    'p1_path': src.get('p1_path', None),
                    'p2_path': src.get('p2_path', None),
                    'unbound_lig1_path': src.get('unbound_lig1_path', None),
                    'unbound_lig2_path': src.get('unbound_lig2_path', None),
                    'lig1_mask_path': src.get('lig1_mask_path', None),
                    'lig2_mask_path': src.get('lig2_mask_path', None),
                })

                mol_info_list.append(mol_info)

                # Reconstruction failure still counts as a completed attempt.
                i_saved += 1

            pred_molecules = [item['rdmol'] for item in mol_info_list]
            true_molecules = [item['true_mol'] for item in mol_info_list]
            true_fragments = [item['frag_mol'] for item in mol_info_list]

            record_metadata_keys = [
                'anchor1_mask',
                'anchor2_mask',
                'linker_mask',
                'atom_mask',
                'fragment1_mask',
                'fragment2_mask',
                'fragment_mask',
                'poi_uniprot',
                'e3_uniprot',
                'uuid',
                'p1_path',
                'p2_path',
                'unbound_lig1_path',
                'unbound_lig2_path',
                'lig1_mask_path',
                'lig2_mask_path',
            ]

            batch_for_records = {
                key: [item.get(key, None) for item in mol_info_list]
                for key in record_metadata_keys
            }

            acc.add_batch(
                pred_molecules=pred_molecules,
                batch=batch_for_records,
                true_molecules=true_molecules,
                true_fragments=true_fragments,
                batch_idx=int(i_batch),
                sample_id=int(i_repeat),
            )

            if len(acc.records) != i_saved:
                raise RuntimeError(
                    f'Record-count mismatch: generated={i_saved}, records={len(acc.records)}'
                )

            print_pool_status(pool, logger)

            del batch
            del outputs
            del trajs
            del mol_info_list

            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                torch.cuda.empty_cache()
            gc.collect()

    # Save complete unified evaluation records.
    if len(acc.records) != i_saved:
        raise RuntimeError(
            f'Final record-count mismatch: generated={i_saved}, records={len(acc.records)}'
        )

    records_path = save_records_pkl(acc.records, args.outdir)

    print('=' * 100)
    print('PocketXMol sampling finished.')
    print(f'Completed generation attempts: {i_saved}')
    print('=' * 100)

    if args.eval_mode == 'both':
        run_evaluate(args)