"""
Strict PROTAC generation + RL functional evaluation metrics.

Use after test-set sampling. This module does NOT sample molecules.
It computes metrics from generated RDKit molecules and the corresponding batch metadata.

Main metrics:
    Validity
    Uniqueness
    Novelty
    Recovery
    2D_filters / 2D_filters_SA / 2D_filters_RA / 2D_filters_PAINS
    QED
    SA_score
    SC_RDKit
    Linker_num_rings
    Predicted_degradation_score
    High_degradation_rate
    Ternary_reward
    DeepTernary_pred_p2_rmsd
    DeepTernary_clash_ratio
    DeepTernary_scored_rate_among_eligible_strict

Strict validity follows the same topology idea as train_rl.py:
    1. generated mol can be sanitized;
    2. whole generated mol is a single connected component;
    3. exactly one anchor1-linker bond;
    4. exactly one anchor2-linker bond;
    5. anchor-linker bonds are not aromatic;
    6. anchor1, anchor2 and all linker atoms are connected in the bridge subgraph;
    7. linker atoms are internally connected.

Important:
    - This code does NOT take the largest connected component.
    - Invalid molecules get 0 for count/rate-style metrics.
    - QED, SA_score, SC_RDKit, Linker_num_rings are averaged over strict-valid molecules.
    - PROTAC-STAN and DeepTernary are only called for strict-valid molecules.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import QED
from rdkit.Chem.MolStandardize import rdMolStandardize

# Optional reward models. Metrics can either receive existing reward objects
# or build them from Hydra config via build_reward_models_from_conf(...).
REWARD_IMPORT_ERROR = None

try:
    from model.reward_models import ProtacStanReward, DeepTernaryReward
except Exception as e:
    REWARD_IMPORT_ERROR = e
    ProtacStanReward = None  # type: ignore
    DeepTernaryReward = None  # type: ignore


# ============================================================
# Optional DeLinker utilities: SA score and SC_RDKit
# ============================================================

try:
    from src.delinker_utils import sascorer, calc_SC_RDKit  # type: ignore
except Exception:
    sascorer = None
    calc_SC_RDKit = None

    # Fallback to RDKit contrib SA_Score if available.
    try:
        sa_score_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_score_path not in sys.path:
            sys.path.append(sa_score_path)
        import sascorer  # type: ignore
    except Exception:
        sascorer = None


# ============================================================
# Generic helpers
# ============================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)
def sequence_item_to_float(
    values: Any,
    i: int,
    default: float = 0.0,
) -> float:
    """
    从 tensor / ndarray / list 中安全提取第 i 个标量。
    保存进 records 前统一转为普通 Python float，避免长期持有 GPU tensor。
    """
    if values is None:
        return float(default)

    try:
        if torch.is_tensor(values):
            flat = values.detach().cpu().reshape(-1)
            if i >= flat.numel():
                return float(default)
            return _safe_float(flat[i].item(), default)

        if isinstance(values, np.ndarray):
            flat = values.reshape(-1)
            if i >= flat.size:
                return float(default)
            return _safe_float(flat[i], default)

        if isinstance(values, (list, tuple)):
            if i >= len(values):
                return float(default)
            return _safe_float(values[i], default)

        return _safe_float(values, default)

    except Exception:
        return float(default)

def safe_nanmean(values: Sequence[float], default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(default)
    if not np.isfinite(arr).any():
        return float(default)
    return float(np.nanmean(arr))


def to_numpy_bool_mask(mask: Any, atom_mask: Any = None, n_atoms: Optional[int] = None) -> np.ndarray:
    """
    Convert mask to 1D bool numpy array.

    If atom_mask is provided, first remove padded atoms using atom_mask.
    This is safer when RDKit mol was built after node_mask filtering.
    """
    if mask is None:
        if n_atoms is None:
            raise ValueError("mask is None and n_atoms is not provided.")
        return np.zeros((n_atoms,), dtype=bool)

    if torch.is_tensor(mask):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)

    arr = np.asarray(arr).squeeze()

    if atom_mask is not None:
        if torch.is_tensor(atom_mask):
            am = atom_mask.detach().cpu().numpy()
        else:
            am = np.asarray(atom_mask)
        am = np.asarray(am).squeeze().astype(bool)
        arr = arr[am]

    arr = arr.astype(bool)

    if n_atoms is not None:
        if len(arr) > n_atoms:
            arr = arr[:n_atoms]
        elif len(arr) < n_atoms:
            pad = np.zeros((n_atoms - len(arr),), dtype=bool)
            arr = np.concatenate([arr, pad], axis=0)

    return arr


def get_batch_item(batch: Dict[str, Any], key: str, i: int, default: Any = None) -> Any:
    if batch is None or key not in batch:
        return default

    v = batch[key]

    if torch.is_tensor(v):
        return v[i]

    if isinstance(v, np.ndarray):
        return v[i]

    if isinstance(v, (list, tuple)):
        return v[i]

    # Scalar or shared object.
    return v


def sanitize_copy(mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    mol_copy = Chem.Mol(mol)
    Chem.SanitizeMol(mol_copy)
    return mol_copy


def safe_mol_to_smiles(mol: Optional[Chem.Mol], remove_stereo: bool = True) -> Optional[str]:
    if mol is None:
        return None

    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)

        if remove_stereo:
            Chem.RemoveStereochemistry(mol_copy)

        try:
            mol_copy = Chem.RemoveHs(mol_copy)
        except Exception:
            pass

        smi = Chem.MolToSmiles(
            mol_copy,
            canonical=True,
            isomericSmiles=not remove_stereo,
        )
        if smi is None or len(smi) == 0:
            return None
        return smi
    except Exception:
        return None


def canonical_linker_smiles_from_mol(mol: Optional[Chem.Mol]) -> Optional[str]:
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    try:
        mol_copy = Chem.Mol(mol)
        Chem.RemoveStereochemistry(mol_copy)

        try:
            Chem.SanitizeMol(mol_copy)
        except Exception:
            mol_copy.UpdatePropertyCache(strict=False)

        smi = Chem.MolToSmiles(
            mol_copy,
            canonical=True,
            isomericSmiles=False,
        )

        tmp = Chem.MolFromSmiles(smi)
        if tmp is None:
            return smi

        enumerator = rdMolStandardize.TautomerEnumerator()
        tmp = enumerator.Canonicalize(tmp)
        return Chem.MolToSmiles(tmp, canonical=True, isomericSmiles=False)

    except Exception:
        try:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        except Exception:
            return None


def canonical_linker_smiles_from_smi(smi: Optional[str]) -> Optional[str]:
    if smi is None:
        return None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return canonical_linker_smiles_from_mol(mol)


def is_connected_mol(mol: Optional[Chem.Mol]) -> bool:
    if mol is None:
        return False
    try:
        frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        return len(frags) == 1
    except Exception:
        return False


def submol_from_atom_indices(mol: Optional[Chem.Mol], keep_atom_indices: Sequence[int]) -> Optional[Chem.Mol]:
    """
    Extract a submolecule by keeping only selected atom indices.
    Conformer coordinates are not preserved here because the metrics below only need 2D linker properties.
    """
    if mol is None:
        return None

    keep = sorted(set(int(x) for x in keep_atom_indices))
    if len(keep) == 0:
        return None

    n_atoms = mol.GetNumAtoms()
    if min(keep) < 0 or max(keep) >= n_atoms:
        return None

    try:
        rw = Chem.RWMol(Chem.Mol(mol))
        remove = [idx for idx in range(n_atoms) if idx not in set(keep)]
        for idx in sorted(remove, reverse=True):
            rw.RemoveAtom(int(idx))
        submol = rw.GetMol()
        try:
            Chem.SanitizeMol(submol)
        except Exception:
            submol.UpdatePropertyCache(strict=False)
        return submol
    except Exception:
        return None


def extract_linker_mol_by_mask(
    mol: Optional[Chem.Mol],
    linker_mask: Any,
    atom_mask: Any = None,
) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    linker_mask_np = to_numpy_bool_mask(linker_mask, atom_mask=atom_mask, n_atoms=mol.GetNumAtoms())
    linker_idx = np.where(linker_mask_np)[0].tolist()
    return submol_from_atom_indices(mol, linker_idx)


def _mask_to_bool_numpy(mask):
    """
    mask: torch.Tensor or np.ndarray
        shape can be [N], [N, 1]
    return: np.ndarray bool, shape [N]
    """
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask).squeeze()
    return mask.astype(bool)

# ============================================================
# Strict PROTAC topology
# ============================================================
def is_strict_protac_mol(
    mol,
    anchor1_mask,
    anchor2_mask,
    linker_mask,
    fragment1_mask,
    fragment2_mask,
):
    """
    Strict PROTAC topology check.

    Requirements:
        1. mol can be sanitized
        2. whole molecule is connected
        3. exactly one anchor1-linker bond
        4. exactly one anchor2-linker bond
        5. anchor-linker bonds are non-aromatic
        6. anchor1 + anchor2 + linker atoms form connected bridge
        7. linker atoms internally connected
        8. no forbidden fragment-fragment/linker connections

    Notes:
        - Positive/existence conditions default to failure.
        - Exclusion conditions default to success and are set to failure
          only when a forbidden bond is actually observed.
        - Topology inspection uses a sanitized copy whenever sanitization
          succeeds, avoiding stale RDKit aromatic flags in the stored mol.
    """

    info = {
        # positive / existence conditions
        "sanitize_ok": 0.0,
        "connected": 0.0,
        "exact_one_a1_linker": 0.0,
        "exact_one_a2_linker": 0.0,

        # exclusion condition:
        # if no aromatic anchor-linker bond is observed, this passes
        "anchor_linker_non_aromatic": 1.0,

        # positive connectivity conditions
        "bridge_component_ok": 0.0,
        "linker_internal_connected": 0.0,

        # exclusion conditions:
        # default pass, set to 0 only when a forbidden edge is observed
        "no_fragment1_fragment2_bond": 1.0,
        "no_bad_non_anchor_fragment_linker_bond": 1.0,

        "strict_protac": 0.0,
    }

    # ---------------------------------------------------------
    # No reconstructed molecule:
    # positive conditions fail, but prohibited-bond conditions
    # are not treated as violations.
    # ---------------------------------------------------------
    if mol is None:
        return False, info

    # ---------------------------------------------------------
    # 1. sanitize
    # ---------------------------------------------------------
    # Keep a clean inspection copy.
    # If full sanitization succeeds, all later topology checks
    # use that sanitized graph.
    mol_eval = Chem.Mol(mol)

    try:
        mol_sanitized = Chem.Mol(mol)
        Chem.SanitizeMol(mol_sanitized)

        info["sanitize_ok"] = 1.0
        mol_eval = mol_sanitized

    except Exception:
        info["sanitize_ok"] = 0.0
        mol_eval = Chem.Mol(mol)

    # ---------------------------------------------------------
    # 2. connected
    # ---------------------------------------------------------
    if is_connected_mol(mol_eval):
        info["connected"] = 1.0

    # ---------------------------------------------------------
    # masks
    # ---------------------------------------------------------
    anchor1_mask = _mask_to_bool_numpy(anchor1_mask)
    anchor2_mask = _mask_to_bool_numpy(anchor2_mask)
    linker_mask = _mask_to_bool_numpy(linker_mask)
    fragment1_mask = _mask_to_bool_numpy(fragment1_mask)
    fragment2_mask = _mask_to_bool_numpy(fragment2_mask)

    anchor1_idx = np.where(anchor1_mask)[0].tolist()
    anchor2_idx = np.where(anchor2_mask)[0].tolist()
    linker_idx = np.where(linker_mask)[0].tolist()

    # anchor 数量异常，无法继续
    if len(anchor1_idx) != 1 or len(anchor2_idx) != 1:
        return False, info

    if len(linker_idx) == 0:
        return False, info

    a1 = int(anchor1_idx[0])
    a2 = int(anchor2_idx[0])

    linker_idx = [int(x) for x in linker_idx]

    n_atoms = mol_eval.GetNumAtoms()

    if a1 >= n_atoms or a2 >= n_atoms:
        return False, info

    if max(linker_idx) >= n_atoms:
        return False, info

    # ---------------------------------------------------------
    # 3. anchor-linker bonds
    # ---------------------------------------------------------
    a1_linker_bonds = []
    a2_linker_bonds = []

    for l in linker_idx:

        b1 = mol_eval.GetBondBetweenAtoms(a1, l)

        if b1 is not None:
            a1_linker_bonds.append((a1, l, b1))

            # Only an actual aromatic bond type counts as violation.
            # This avoids stale GetIsAromatic() flags on stored RDKit mols.
            if b1.GetBondType() == Chem.BondType.AROMATIC:
                info["anchor_linker_non_aromatic"] = 0.0

        b2 = mol_eval.GetBondBetweenAtoms(a2, l)

        if b2 is not None:
            a2_linker_bonds.append((a2, l, b2))

            if b2.GetBondType() == Chem.BondType.AROMATIC:
                info["anchor_linker_non_aromatic"] = 0.0

    if len(a1_linker_bonds) == 1:
        info["exact_one_a1_linker"] = 1.0

    if len(a2_linker_bonds) == 1:
        info["exact_one_a2_linker"] = 1.0

    # ---------------------------------------------------------
    # 4. graph
    # ---------------------------------------------------------
    adjacency = {i: [] for i in range(n_atoms)}

    try:
        for bond in mol_eval.GetBonds():

            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            adjacency[i].append(j)
            adjacency[j].append(i)

            # fragment1-fragment2 shortcut
            if (
                (fragment1_mask[i] and fragment2_mask[j])
                or
                (fragment2_mask[i] and fragment1_mask[j])
            ):
                info["no_fragment1_fragment2_bond"] = 0.0

            # non-anchor fragment1-linker
            if (
                (fragment1_mask[i] and linker_mask[j])
                or
                (fragment1_mask[j] and linker_mask[i])
            ):
                frag_atom = i if fragment1_mask[i] else j

                if frag_atom != a1:
                    info[
                        "no_bad_non_anchor_fragment_linker_bond"
                    ] = 0.0

            # non-anchor fragment2-linker
            if (
                (fragment2_mask[i] and linker_mask[j])
                or
                (fragment2_mask[j] and linker_mask[i])
            ):
                frag_atom = i if fragment2_mask[i] else j

                if frag_atom != a2:
                    info[
                        "no_bad_non_anchor_fragment_linker_bond"
                    ] = 0.0

    except Exception:
        pass

    # ---------------------------------------------------------
    # 5. bridge component
    # ---------------------------------------------------------
    bridge_nodes = set(linker_idx + [a1, a2])

    stack = [a1]
    visited = {a1}

    while stack:

        cur = stack.pop()

        for nb in adjacency[cur]:

            if nb not in bridge_nodes:
                continue

            if nb not in visited:
                visited.add(nb)
                stack.append(nb)

    if (
        a2 in visited
        and all(l in visited for l in linker_idx)
    ):
        info["bridge_component_ok"] = 1.0

    # ---------------------------------------------------------
    # 6. linker internal connectivity
    # ---------------------------------------------------------
    if len(linker_idx) == 1:

        info["linker_internal_connected"] = 1.0

    else:

        linker_nodes = set(linker_idx)

        stack = [linker_idx[0]]
        visited_linker = {linker_idx[0]}

        while stack:

            cur = stack.pop()

            for nb in adjacency[cur]:

                if nb not in linker_nodes:
                    continue

                if nb not in visited_linker:
                    visited_linker.add(nb)
                    stack.append(nb)

        if all(l in visited_linker for l in linker_idx):
            info["linker_internal_connected"] = 1.0

    # ---------------------------------------------------------
    # final strict validity
    # ---------------------------------------------------------
    strict_ok = (
        info["sanitize_ok"]
        and info["connected"]
        and info["exact_one_a1_linker"]
        and info["exact_one_a2_linker"]
        and info["anchor_linker_non_aromatic"]
        and info["bridge_component_ok"]
        and info["linker_internal_connected"]
        and info["no_fragment1_fragment2_bond"]
        and info["no_bad_non_anchor_fragment_linker_bond"]
    )

    info["strict_protac"] = float(strict_ok)

    return bool(strict_ok), info


# ============================================================
# 2D filters and continuous metrics
# ============================================================

def calc_sa_score_mol(mol: Optional[Chem.Mol]) -> float:
    if mol is None or sascorer is None:
        return np.nan
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return np.nan


def check_ring_filter(linker_mol: Optional[Chem.Mol]) -> bool:
    """
    RA filter used in DeLinker-style evaluation:
    linker rings should not contain double bonds.
    """
    if linker_mol is None:
        return False
    try:
        ssr = Chem.GetSymmSSSR(linker_mol)
    except Exception:
        return False

    for ring in ssr:
        ring = set(int(x) for x in ring)
        for bond in linker_mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            if a in ring and b in ring:
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    return False
    return True


def load_pains_smarts() -> List[Chem.Mol]:
    pains_path = os.path.join(RDConfig.RDDataDir, "Pains", "wehi_pains.csv")
    pains_smarts: List[Chem.Mol] = []

    if not os.path.exists(pains_path):
        return pains_smarts

    with open(pains_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            patt = Chem.MolFromSmarts(row[0], mergeHs=True)
            if patt is not None:
                pains_smarts.append(patt)

    return pains_smarts


def check_pains(mol: Optional[Chem.Mol], pains_smarts: Sequence[Chem.Mol]) -> bool:
    if mol is None:
        return False
    for patt in pains_smarts:
        try:
            if mol.HasSubstructMatch(patt):
                return False
        except Exception:
            continue
    return True


def calc_2d_filters_for_entry(
    entry: Dict[str, Any],
    pains_smarts: Sequence[Chem.Mol],
    sa_filter_mode: str = "delinker",
    sa_threshold: float = 6.0,
) -> Tuple[bool, bool, bool, bool]:
    """
    Returns: pass_all, pass_SA, pass_RA, pass_PAINS

    sa_filter_mode:
        - "delinker": pass_SA = SA(pred_mol) < SA(frag_mol), with threshold fallback
        - "threshold": pass_SA = SA(pred_mol) < sa_threshold
    """
    pred_mol = entry["pred_mol"]
    frag_mol = entry.get("frag_mol", None)
    linker_mol = entry.get("pred_linker_mol", None)

    pred_sa = calc_sa_score_mol(pred_mol)
    frag_sa = calc_sa_score_mol(frag_mol) if frag_mol is not None else np.nan

    if sa_filter_mode == "delinker":
        if np.isfinite(pred_sa) and np.isfinite(frag_sa):
            pass_SA = pred_sa < frag_sa
        else:
            pass_SA = np.isfinite(pred_sa) and pred_sa < sa_threshold
    elif sa_filter_mode == "threshold":
        pass_SA = np.isfinite(pred_sa) and pred_sa < sa_threshold
    else:
        raise ValueError(f"Unknown sa_filter_mode: {sa_filter_mode}")

    pass_RA = check_ring_filter(linker_mol)
    pass_PAINS = check_pains(pred_mol, pains_smarts)
    pass_all = bool(pass_SA and pass_RA and pass_PAINS)

    return pass_all, bool(pass_SA), bool(pass_RA), bool(pass_PAINS)


def calc_qed(mol: Optional[Chem.Mol]) -> float:
    if mol is None:
        return np.nan
    try:
        return float(QED.qed(mol))
    except Exception:
        return np.nan


def calc_linker_num_rings(linker_mol: Optional[Chem.Mol]) -> float:
    if linker_mol is None:
        return np.nan
    try:
        return float(len(Chem.GetSymmSSSR(linker_mol)))
    except Exception:
        return np.nan


def calc_sc_rdkit_score(
    pred_mol: Optional[Chem.Mol],
    true_mol: Optional[Chem.Mol],
    failed_value: float = -0.5,
) -> float:
    if pred_mol is None or true_mol is None:
        return np.nan
    if calc_SC_RDKit is None:
        return np.nan
    try:
        return float(calc_SC_RDKit.calc_SC_RDKit_score(pred_mol, true_mol))
    except Exception:
        return float(failed_value)


# ============================================================
# Record builders
# ============================================================

def build_records_from_batch(
    pred_molecules: Sequence[Optional[Chem.Mol]],
    batch: Dict[str, Any],
    true_molecules: Optional[Sequence[Optional[Chem.Mol]]] = None,
    true_fragments: Optional[Sequence[Optional[Chem.Mol]]] = None,
    batch_idx: Optional[int] = None,
    sample_id: Optional[Union[int, str]] = None,
    strict_flags: Optional[Sequence[Any]] = None,
    strict_infos: Optional[Sequence[Dict[str, Any]]] = None,
    degradation_scores: Any = None,
    ternary_scores: Any = None,
    deepternary_info: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert one sampled batch into records.

    除分子和数据集元数据外，同时保存本轮已经计算好的：
        - strict 判定
        - PROTAC-STAN 分数
        - DeepTernary 分数
        - DeepTernary RMSD / clash / scored mask

    最终 compute() 可以直接汇总，不再运行奖励模型。
    """
    records: List[Dict[str, Any]] = []
    B = len(pred_molecules)

    for i in range(B):
        strict_value = None
        strict_info_value: Dict[str, Any] = {}

        if strict_flags is not None:
            strict_value = bool(strict_flags[i])

        if strict_infos is not None and i < len(strict_infos):
            strict_info_value = dict(strict_infos[i])

        p2_rmsd = -1.0
        clash_ratio = -1.0
        scored = 0.0
        attempted = 0.0
        valid_candidate = 0.0

        if deepternary_info is not None:
            p2_rmsd = sequence_item_to_float(
                deepternary_info.get(
                    "deepternary_pred_p2_rmsd_raw",
                    None,
                ),
                i,
                default=-1.0,
            )

            clash_ratio = sequence_item_to_float(
                deepternary_info.get(
                    "deepternary_clash_ratio_raw",
                    None,
                ),
                i,
                default=-1.0,
            )

            scored = sequence_item_to_float(
                deepternary_info.get(
                    "deepternary_scored_mask",
                    None,
                ),
                i,
                default=1.0 if p2_rmsd >= 0.0 else 0.0,
            )

            attempted = sequence_item_to_float(
                deepternary_info.get(
                    "deepternary_attempted_mask",
                    None,
                ),
                i,
                default=0.0,
            )

            valid_candidate = sequence_item_to_float(
                deepternary_info.get(
                    "deepternary_valid_candidate_mask",
                    None,
                ),
                i,
                default=0.0,
            )

        rec: Dict[str, Any] = {
            "pred_mol": pred_molecules[i],
            "true_mol": (
                true_molecules[i]
                if true_molecules is not None
                else None
            ),
            "frag_mol": (
                true_fragments[i]
                if true_fragments is not None
                else None
            ),

            "anchor1_mask": get_batch_item(
                batch,
                "anchor1_mask",
                i,
            ),
            "anchor2_mask": get_batch_item(
                batch,
                "anchor2_mask",
                i,
            ),
            "linker_mask": get_batch_item(
                batch,
                "linker_mask",
                i,
            ),
            "atom_mask": get_batch_item(
                batch,
                "atom_mask",
                i,
            ),
            "fragment1_mask": get_batch_item(
                batch,
                "fragment1_mask",
                i,
            ),
            "fragment2_mask": get_batch_item(
                batch,
                "fragment2_mask",
                i,
            ),
            "fragment_mask": get_batch_item(
                batch,
                "fragment_mask",
                i,
            ),

            "poi_uniprot": get_batch_item(
                batch,
                "poi_uniprot",
                i,
            ),
            "e3_uniprot": get_batch_item(
                batch,
                "e3_uniprot",
                i,
            ),

            "uuid": get_batch_item(
                batch,
                "uuid",
                i,
                default=f"idx_{i}",
            ),
            "batch_idx": batch_idx,
            "sample_id": sample_id,

            "p1_path": get_batch_item(
                batch,
                "p1_path",
                i,
            ),
            "p2_path": get_batch_item(
                batch,
                "p2_path",
                i,
            ),
            "unbound_lig1_path": get_batch_item(
                batch,
                "unbound_lig1_path",
                i,
            ),
            "unbound_lig2_path": get_batch_item(
                batch,
                "unbound_lig2_path",
                i,
            ),
            "lig1_mask_path": get_batch_item(
                batch,
                "lig1_mask_path",
                i,
            ),
            "lig2_mask_path": get_batch_item(
                batch,
                "lig2_mask_path",
                i,
            ),

            # 本轮已经得到的 strict 结果
            "precomputed_strict": strict_value,
            "precomputed_strict_info": strict_info_value,

            # 本轮已经得到的奖励模型结果
            "precomputed_degradation_score": (
                sequence_item_to_float(
                    degradation_scores,
                    i,
                    default=0.0,
                )
            ),
            "precomputed_ternary_score": (
                sequence_item_to_float(
                    ternary_scores,
                    i,
                    default=0.0,
                )
            ),
            "precomputed_deepternary_p2_rmsd": p2_rmsd,
            "precomputed_deepternary_clash_ratio": clash_ratio,
            "precomputed_deepternary_scored": (
                bool(scored > 0.5)
            ),
            "precomputed_deepternary_attempted": (
                bool(attempted > 0.5)
            ),
            "precomputed_deepternary_valid_candidate": (
                bool(valid_candidate > 0.5)
            ),
        }

        records.append(rec)

    return records


def _prepare_valid_entries(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[bool], List[Optional[str]], Dict[str, List[float]]]:
    valid_entries: List[Dict[str, Any]] = []
    strict_flags: List[bool] = []
    smiles_for_protac_stan: List[Optional[str]] = []
    strict_info_acc: Dict[str, List[float]] = defaultdict(list)

    for rec in records:
        pred_mol = rec.get("pred_mol", None)
        atom_mask = rec.get("atom_mask", None)

        precomputed_strict = rec.get(
            "precomputed_strict",
            None,
        )

        if precomputed_strict is not None:
            strict_ok = bool(precomputed_strict)

            strict_info = rec.get(
                "precomputed_strict_info",
                {},
            )

            if strict_info is None:
                strict_info = {}

        else:
            strict_ok, strict_info = is_strict_protac_mol(
                mol=pred_mol,
                anchor1_mask=rec.get(
                    "anchor1_mask",
                    None,
                ),
                anchor2_mask=rec.get(
                    "anchor2_mask",
                    None,
                ),
                linker_mask=rec.get(
                    "linker_mask",
                    None,
                ),
                fragment1_mask=rec.get(
                    "fragment1_mask",
                    None,
                ),
                fragment2_mask=rec.get(
                    "fragment2_mask",
                    None,
                ),
            )

        for k, v in strict_info.items():
            strict_info_acc[k].append(float(v))

        if not strict_ok:
            strict_flags.append(False)
            smiles_for_protac_stan.append(None)
            continue

        try:
            pred_mol_san = sanitize_copy(pred_mol)
        except Exception:
            strict_flags.append(False)
            smiles_for_protac_stan.append(None)
            continue

        true_mol = rec.get("true_mol", None)
        frag_mol = rec.get("frag_mol", None)

        true_mol_san = None
        frag_mol_san = None

        if true_mol is not None:
            try:
                true_mol_san = sanitize_copy(true_mol)
            except Exception:
                true_mol_san = None

        if frag_mol is not None:
            try:
                frag_mol_san = sanitize_copy(frag_mol)
            except Exception:
                frag_mol_san = None

        pred_smi = safe_mol_to_smiles(pred_mol_san)
        true_smi = safe_mol_to_smiles(true_mol_san) if true_mol_san is not None else None
        frag_smi = safe_mol_to_smiles(frag_mol_san) if frag_mol_san is not None else None

        pred_linker_mol = extract_linker_mol_by_mask(
            pred_mol_san,
            linker_mask=rec.get("linker_mask", None),
            atom_mask=atom_mask,
        )
        pred_linker_smi = canonical_linker_smiles_from_mol(pred_linker_mol)

        true_linker_mol = None
        true_linker_smi = None

        if true_mol_san is not None:
            true_linker_mol = extract_linker_mol_by_mask(
                true_mol_san,
                linker_mask=rec.get("linker_mask", None),
                atom_mask=atom_mask,
            )
            true_linker_smi = canonical_linker_smiles_from_mol(true_linker_mol)

        if pred_smi is None:
            strict_flags.append(False)
            smiles_for_protac_stan.append(None)
            continue

        entry = dict(rec)
        entry.update(
            {
                "pred_mol": pred_mol_san,
                "true_mol": true_mol_san,
                "frag_mol": frag_mol_san,
                "pred_mol_smi": pred_smi,
                "true_mol_smi": true_smi,
                "frag_smi": frag_smi,
                "pred_linker_mol": pred_linker_mol,
                "true_linker_mol": true_linker_mol,
                "pred_linker_smi": pred_linker_smi,
                "true_linker_smi": true_linker_smi,
            }
        )

        valid_entries.append(entry)
        strict_flags.append(True)
        smiles_for_protac_stan.append(pred_smi)

    return valid_entries, strict_flags, smiles_for_protac_stan, strict_info_acc


# ============================================================
# Novelty reference preparation
# ============================================================

def prepare_reference_mol_smiles(reference_mol_smiles: Optional[Iterable[Any]]) -> Optional[set]:
    if reference_mol_smiles is None:
        return None

    result = set()
    for x in reference_mol_smiles:
        if x is None:
            continue
        if isinstance(x, Chem.Mol):
            smi = safe_mol_to_smiles(x)
        else:
            mol = Chem.MolFromSmiles(str(x))
            smi = safe_mol_to_smiles(mol) if mol is not None else None
        if smi is not None:
            result.add(smi)
    return result


def prepare_reference_linker_smiles(reference_linker_smiles: Optional[Iterable[Any]]) -> Optional[set]:
    if reference_linker_smiles is None:
        return None

    result = set()
    for x in reference_linker_smiles:
        if x is None:
            continue
        if isinstance(x, Chem.Mol):
            smi = canonical_linker_smiles_from_mol(x)
        else:
            smi = canonical_linker_smiles_from_smi(str(x))
        if smi is not None:
            result.add(smi)
    return result


# ============================================================
# Main metrics function
# ============================================================

def compute_metrics_from_records(
    records: Sequence[Dict[str, Any]],
    *,
    reference_mol_smiles: Optional[Iterable[Any]] = None,
    reference_linker_smiles: Optional[Iterable[Any]] = None,
    novelty_on: str = "linker",       # "linker" or "molecule"
    recovery_on: str = "molecule",    # "molecule" or "linker"
    uniqueness_group: str = "case",   # "case" or "global"
    sa_filter_mode: str = "delinker", # "delinker" or "threshold"
    sa_threshold: float = 6.0,
    sc_rdkit_failed_value: float = -0.5,
    protac_stan_reward: Any = None,
    degradation_threshold: float = 0.5,
    protac_stan_batch_size: int = 16,
    deepternary_reward: Any = None,
    deepternary_max_samples: Optional[int] = None,
    step: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    use_precomputed_rewards: bool = False,
    return_entries: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], List[Dict[str, Any]]]]:
    """
    Compute final metrics from generated molecule records.

    Denominators:
        Validity = #strict_valid / #all_generated
        Uniqueness = #strict_valid_and_unique / #all_generated
        Novelty = #strict_valid_and_novel / #all_generated
        Recovery = #strict_valid_and_recovered / #all_generated
        2D_filters = #strict_valid_and_pass / #all_generated

    Continuous metrics:
        QED / SA_score / SC_RDKit / Linker_num_rings are averaged over strict-valid molecules.

    Functional metrics:
        Predicted_degradation_score = mean(strict_valid * PROTAC-STAN score) over all generated.
        High_degradation_rate = #strict_valid_and_score>=threshold / #all_generated.
        Ternary_reward = mean(strict_valid * DeepTernary reward) over all generated.
        DeepTernary_pred_p2_rmsd / DeepTernary_clash_ratio are averaged over successfully scored strict-valid molecules.
    """
    total = len(records)

    result: Dict[str, float] = {
        "Validity": 0.0,
        "Uniqueness": 0.0,
        "Novelty": 0.0,
        "Recovery": 0.0,
        "2D_filters": 0.0,
        "2D_filters_SA": 0.0,
        "2D_filters_RA": 0.0,
        "2D_filters_PAINS": 0.0,
        "QED": 0.0,
        "SA_score": 0.0,
        "SC_RDKit": 0.0,
        "Linker_num_rings": 0.0,
        "Predicted_degradation_score": 0.0,
        "Predicted_degradation_score_valid": 0.0,
        "High_degradation_rate": 0.0,
        "High_degradation_rate_valid": 0.0,
        "Ternary_reward": 0.0,
        "Ternary_reward_scored": 0.0,
        "DeepTernary_pred_p2_rmsd": 0.0,
        "DeepTernary_clash_ratio": 0.0,
        "DeepTernary_scored_rate_among_eligible_strict": 0.0,
        "num_total": float(total),
        "num_valid": 0.0,
        "num_unique_valid": 0.0,
        "num_novel_valid": 0.0,
        "num_recovered_valid": 0.0,
        "num_high_degradation": 0.0,
        "num_deepternary_scored": 0.0,
    }

    if total == 0:
        if return_entries:
            return result, []
        return result

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_entries, strict_flags, smiles_for_protac_stan, strict_info_acc = _prepare_valid_entries(records)

    num_valid = len(valid_entries)
    result["Validity"] = num_valid / total
    result["num_valid"] = float(num_valid)

    for k, vals in strict_info_acc.items():
        result[f"strict_{k}"] = safe_nanmean(vals)

    if num_valid == 0:
        if return_entries:
            return result, valid_entries
        return result

    # ------------------------------
    # Uniqueness
    # ------------------------------
    if uniqueness_group == "global":
        num_unique_valid = len({e["pred_mol_smi"] for e in valid_entries})
    elif uniqueness_group == "case":
        group_to_pred = defaultdict(set)
        for e in valid_entries:
            uuid = str(e.get("uuid", "unknown_uuid"))
            true_key = e.get("true_mol_smi") or "unknown_true"
            frag_key = e.get("frag_smi") or "unknown_frag"
            poi = str(e.get("poi_uniprot", "unknown_poi"))
            e3 = str(e.get("e3_uniprot", "unknown_e3"))
            key = f"{uuid}|{true_key}|{frag_key}|{poi}|{e3}"
            group_to_pred[key].add(e["pred_mol_smi"])
        num_unique_valid = sum(len(v) for v in group_to_pred.values())
    else:
        raise ValueError("uniqueness_group must be 'case' or 'global'.")

    result["num_unique_valid"] = float(num_unique_valid)
    result["Uniqueness"] = num_unique_valid / total

    # ------------------------------
    # Novelty
    # ------------------------------
    if novelty_on == "linker":
        ref_set = prepare_reference_linker_smiles(reference_linker_smiles)
        if ref_set is None:
            # Fallback: DeLinker-style, compare against true linkers in current eval set.
            ref_set = {e["true_linker_smi"] for e in valid_entries if e.get("true_linker_smi") is not None}
        num_novel_valid = sum(
            e.get("pred_linker_smi") is not None and e["pred_linker_smi"] not in ref_set
            for e in valid_entries
        )
    elif novelty_on == "molecule":
        ref_set = prepare_reference_mol_smiles(reference_mol_smiles)
        if ref_set is None:
            ref_set = {e["true_mol_smi"] for e in valid_entries if e.get("true_mol_smi") is not None}
        num_novel_valid = sum(e["pred_mol_smi"] not in ref_set for e in valid_entries)
    else:
        raise ValueError("novelty_on must be 'linker' or 'molecule'.")

    result["num_novel_valid"] = float(num_novel_valid)
    result["Novelty"] = num_novel_valid / total

    # ------------------------------
    # Recovery
    # ------------------------------
    if recovery_on == "molecule":
        num_recovered_valid = sum(
            e.get("true_mol_smi") is not None and e["pred_mol_smi"] == e["true_mol_smi"]
            for e in valid_entries
        )
    elif recovery_on == "linker":
        num_recovered_valid = sum(
            e.get("true_linker_smi") is not None and e["pred_linker_smi"] == e["true_linker_smi"]
            for e in valid_entries
        )
    else:
        raise ValueError("recovery_on must be 'molecule' or 'linker'.")

    result["num_recovered_valid"] = float(num_recovered_valid)
    result["Recovery"] = num_recovered_valid / total

    # ------------------------------
    # 2D filters
    # ------------------------------
    pains_smarts = load_pains_smarts()
    pass_all = 0
    pass_SA = 0
    pass_RA = 0
    pass_PAINS = 0

    for e in valid_entries:
        f_all, f_sa, f_ra, f_pains = calc_2d_filters_for_entry(
            e,
            pains_smarts=pains_smarts,
            sa_filter_mode=sa_filter_mode,
            sa_threshold=sa_threshold,
        )
        pass_all += int(f_all)
        pass_SA += int(f_sa)
        pass_RA += int(f_ra)
        pass_PAINS += int(f_pains)

    result["2D_filters"] = pass_all / total
    result["2D_filters_SA"] = pass_SA / total
    result["2D_filters_RA"] = pass_RA / total
    result["2D_filters_PAINS"] = pass_PAINS / total

    # ------------------------------
    # Continuous metrics over valid molecules
    # ------------------------------
    qed_scores = []
    sa_scores = []
    sc_scores = []
    linker_rings = []

    for e in valid_entries:
        qed_scores.append(calc_qed(e["pred_mol"]))
        sa_scores.append(calc_sa_score_mol(e["pred_mol"]))
        sc_scores.append(calc_sc_rdkit_score(e["pred_mol"], e.get("true_mol", None), failed_value=sc_rdkit_failed_value))
        linker_rings.append(calc_linker_num_rings(e.get("pred_linker_mol", None)))

    result["QED"] = safe_nanmean(qed_scores)
    result["SA_score"] = safe_nanmean(sa_scores)
    result["SC_RDKit"] = safe_nanmean(sc_scores)
    result["Linker_num_rings"] = safe_nanmean(linker_rings)

    # ------------------------------
    # PROTAC-STAN degradation metrics
    # ------------------------------
    strict_arr = np.asarray(
        strict_flags,
        dtype=bool,
    )

    degradation_scores = np.zeros(
        total,
        dtype=float,
    )

    if use_precomputed_rewards:
        missing_degradation = [
            i
            for i, rec in enumerate(records)
            if "precomputed_degradation_score" not in rec
        ]

        if len(missing_degradation) > 0:
            raise ValueError(
                "use_precomputed_rewards=True, but some records "
                "do not contain precomputed_degradation_score. "
                f"First missing index: {missing_degradation[0]}"
            )

        degradation_scores = np.asarray(
            [
                _safe_float(
                    rec["precomputed_degradation_score"],
                    default=0.0,
                )
                for rec in records
            ],
            dtype=float,
        )

        degradation_scores = np.nan_to_num(
            degradation_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        degradation_scores = (
                degradation_scores
                * strict_arr.astype(float)
        )

    elif protac_stan_reward is not None:
        poi_ids = [
            rec.get("poi_uniprot", None)
            for rec in records
        ]

        e3_ids = [
            rec.get("e3_uniprot", None)
            for rec in records
        ]

        if (
                any(x is None for x in poi_ids)
                or any(x is None for x in e3_ids)
        ):
            raise ValueError(
                "PROTAC-STAN scoring requires poi_uniprot "
                "and e3_uniprot in each record."
            )

        deg = protac_stan_reward.score(
            smiles_list=smiles_for_protac_stan,
            poi_ids=poi_ids,
            e3_ids=e3_ids,
            step=step,
            batch_size=protac_stan_batch_size,
        )

        if torch.is_tensor(deg):
            deg_arr = (
                deg.detach()
                .cpu()
                .numpy()
                .astype(float)
            )
        else:
            deg_arr = np.asarray(
                deg,
                dtype=float,
            )

        deg_arr = np.nan_to_num(
            deg_arr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        degradation_scores = (
                deg_arr
                * strict_arr.astype(float)
        )

    num_high_degradation = int(
        np.sum(
            strict_arr
            & (
                    degradation_scores
                    >= float(degradation_threshold)
            )
        )
    )

    result["Predicted_degradation_score"] = float(
        np.mean(degradation_scores)
    )

    result["Predicted_degradation_score_valid"] = (
        float(
            np.mean(
                degradation_scores[strict_arr]
            )
        )
        if strict_arr.any()
        else 0.0
    )

    result["num_high_degradation"] = float(
        num_high_degradation
    )

    result["High_degradation_rate"] = (
            num_high_degradation / total
    )

    result["High_degradation_rate_valid"] = (
        num_high_degradation / num_valid
        if num_valid > 0
        else 0.0
    )

    # ------------------------------
    # DeepTernary metrics
    # ------------------------------
    ternary_rewards = np.zeros(
        total,
        dtype=float,
    )

    ternary_rmsd = np.full(
        total,
        -1.0,
        dtype=float,
    )

    ternary_clash = np.full(
        total,
        -1.0,
        dtype=float,
    )

    scored_mask = np.zeros(
        total,
        dtype=bool,
    )

    valid_candidate_mask = np.zeros(
        total,
        dtype=bool,
    )

    num_deepternary_scored = 0

    if use_precomputed_rewards:
        required_precomputed_keys = [
            "precomputed_ternary_score",
            "precomputed_deepternary_p2_rmsd",
            "precomputed_deepternary_clash_ratio",
            "precomputed_deepternary_scored",
            "precomputed_deepternary_valid_candidate",
        ]

        for key in required_precomputed_keys:
            missing = [
                i
                for i, rec in enumerate(records)
                if key not in rec
            ]

            if len(missing) > 0:
                raise ValueError(
                    "use_precomputed_rewards=True, but some records "
                    f"do not contain {key}. "
                    f"First missing index: {missing[0]}"
                )

        ternary_rewards = np.asarray(
            [
                _safe_float(
                    rec["precomputed_ternary_score"],
                    default=0.0,
                )
                for rec in records
            ],
            dtype=float,
        )

        ternary_rewards = np.nan_to_num(
            ternary_rewards,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        ternary_rewards = (
                ternary_rewards
                * strict_arr.astype(float)
        )

        ternary_rmsd = np.asarray(
            [
                _safe_float(
                    rec[
                        "precomputed_deepternary_p2_rmsd"
                    ],
                    default=-1.0,
                )
                for rec in records
            ],
            dtype=float,
        )

        ternary_clash = np.asarray(
            [
                _safe_float(
                    rec[
                        "precomputed_deepternary_clash_ratio"
                    ],
                    default=-1.0,
                )
                for rec in records
            ],
            dtype=float,
        )

        scored_mask = np.asarray(
            [
                bool(
                    rec[
                        "precomputed_deepternary_scored"
                    ]
                )
                for rec in records
            ],
            dtype=bool,
        )

        scored_mask = scored_mask & strict_arr

        valid_candidate_mask = np.asarray(
            [
                bool(
                    rec[
                        "precomputed_deepternary_valid_candidate"
                    ]
                )
                for rec in records
            ],
            dtype=bool,
        )

        valid_candidate_mask = (
                valid_candidate_mask
                & strict_arr
        )

        num_deepternary_scored = int(
            scored_mask.sum()
        )

    elif deepternary_reward is not None:
        required_keys = [
            "uuid",
            "p1_path",
            "p2_path",
            "unbound_lig1_path",
            "unbound_lig2_path",
            "lig1_mask_path",
            "lig2_mask_path",
        ]

        for key in required_keys:
            if any(
                    rec.get(key, None) is None
                    for rec in records
            ):
                raise ValueError(
                    f"DeepTernary scoring requires {key} "
                    "in each record."
                )

        ternary_tensor, ternary_info = (
            deepternary_reward.score(
                mols=[
                    rec.get("pred_mol", None)
                    for rec in records
                ],
                valid_mask=[
                    bool(x)
                    for x in strict_flags
                ],
                names=[
                    rec.get("uuid")
                    for rec in records
                ],
                p1_paths=[
                    rec.get("p1_path")
                    for rec in records
                ],
                p2_paths=[
                    rec.get("p2_path")
                    for rec in records
                ],
                unbound_lig1_paths=[
                    rec.get("unbound_lig1_path")
                    for rec in records
                ],
                unbound_lig2_paths=[
                    rec.get("unbound_lig2_path")
                    for rec in records
                ],
                lig1_mask_paths=[
                    rec.get("lig1_mask_path")
                    for rec in records
                ],
                lig2_mask_paths=[
                    rec.get("lig2_mask_path")
                    for rec in records
                ],
                step=step,
                max_samples=deepternary_max_samples,
                device=device,
            )
        )

        if torch.is_tensor(ternary_tensor):
            ternary_rewards = (
                ternary_tensor.detach()
                .cpu()
                .numpy()
                .astype(float)
            )
        else:
            ternary_rewards = np.asarray(
                ternary_tensor,
                dtype=float,
            )

        ternary_rewards = np.nan_to_num(
            ternary_rewards,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        ternary_rewards = (
                ternary_rewards
                * strict_arr.astype(float)
        )

        rmsd_tensor = ternary_info.get(
            "deepternary_pred_p2_rmsd",
            None,
        )

        clash_tensor = ternary_info.get(
            "deepternary_clash_ratio",
            None,
        )

        n_scored_tensor = ternary_info.get(
            "deepternary_num_scored",
            None,
        )

        if rmsd_tensor is not None:
            if torch.is_tensor(rmsd_tensor):
                ternary_rmsd = (
                    rmsd_tensor.detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
            else:
                ternary_rmsd = np.asarray(
                    rmsd_tensor,
                    dtype=float,
                )

        if clash_tensor is not None:
            if torch.is_tensor(clash_tensor):
                ternary_clash = (
                    clash_tensor.detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                )
            else:
                ternary_clash = np.asarray(
                    clash_tensor,
                    dtype=float,
                )

        scored_mask = ternary_rmsd >= 0.0

        valid_candidate_tensor = ternary_info.get(
            "deepternary_valid_candidate_mask",
            None,
        )

        if valid_candidate_tensor is not None:
            if torch.is_tensor(valid_candidate_tensor):
                valid_candidate_mask = (
                    valid_candidate_tensor.detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )
            else:
                valid_candidate_mask = np.asarray(
                    valid_candidate_tensor,
                    dtype=bool,
                )

            valid_candidate_mask = (
                    valid_candidate_mask
                    & strict_arr
            )

        if n_scored_tensor is not None:
            if torch.is_tensor(n_scored_tensor):
                num_deepternary_scored = int(
                    float(
                        n_scored_tensor.detach()
                        .cpu()
                        .item()
                    )
                )
            else:
                num_deepternary_scored = int(
                    float(n_scored_tensor)
                )
        else:
            num_deepternary_scored = int(
                scored_mask.sum()
            )

    result["Ternary_reward"] = float(
        np.mean(ternary_rewards)
    )

    result["Ternary_reward_scored"] = (
        float(
            np.mean(
                ternary_rewards[scored_mask]
            )
        )
        if scored_mask.any()
        else 0.0
    )

    result["DeepTernary_pred_p2_rmsd"] = (
        float(
            np.mean(
                ternary_rmsd[scored_mask]
            )
        )
        if scored_mask.any()
        else 0.0
    )

    result["DeepTernary_clash_ratio"] = (
        float(
            np.mean(
                ternary_clash[scored_mask]
            )
        )
        if scored_mask.any()
        else 0.0
    )

    result["num_deepternary_scored"] = float(
        num_deepternary_scored
    )

    num_deepternary_valid_candidate = int(
        valid_candidate_mask.sum()
    )

    result[
        "DeepTernary_scored_rate_among_eligible_strict"
    ] = (
        num_deepternary_scored
        / num_deepternary_valid_candidate
        if num_deepternary_valid_candidate > 0
        else 0.0
    )

    if return_entries:
        return result, valid_entries
    return result




# ============================================================
# Optional reward model builders
# ============================================================

def build_reward_models_from_conf(conf: Any, device: Optional[Union[str, torch.device]] = None, use_deepternary: Optional[bool] = None) -> Tuple[Any, Any]:
    """
    Build PROTAC-STAN and optional DeepTernary reward models from your Hydra config.

    This is only a convenience helper. You can also pass already-created objects
    to compute_metrics_from_records(...):
        protac_stan_reward=rl_exp.protac_stan_reward,
        deepternary_reward=rl_exp.deepternary_reward
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_conf = conf.experiment if hasattr(conf, "experiment") else conf["experiment"]

    if ProtacStanReward is None:
        raise ImportError(
            "Cannot import ProtacStanReward from src.reward_models. "
            f"Real import error: {type(REWARD_IMPORT_ERROR).__name__}: {REWARD_IMPORT_ERROR}"
        ) from REWARD_IMPORT_ERROR

    protac_stan_reward = ProtacStanReward(
        project_dir=exp_conf.protac_stan_project_dir,
        model_path=exp_conf.protac_stan_model_path,
        config_path=exp_conf.protac_stan_config_path,
        esm_map_path=exp_conf.protac_stan_esm_map_path,
        tmp_root=exp_conf.protac_stan_tmp_root,
        device=device,
        descriptor_stats_path=exp_conf.protac_stan_descriptor_stats_path,
    )

    if use_deepternary is None:
        use_deepternary = bool(exp_conf.use_deepternary_reward)

    deepternary_reward = None
    if use_deepternary:
        if DeepTernaryReward is None:
            raise ImportError(
                "Cannot import DeepTernaryReward from model.reward_models. "
                "Either fix the import path or pass deepternary_reward explicitly."
            )

        deepternary_reward = DeepTernaryReward(
            python_exe=exp_conf.deepternary_python,
            project_dir=exp_conf.deepternary_project_dir,
            predict_script=exp_conf.deepternary_predict_script,
            tmp_root=exp_conf.deepternary_tmp_root,
            seeds=exp_conf.deepternary_seeds,
            inference_seed=2026,
            timeout=exp_conf.deepternary_timeout,
            device=device,
        )

    return protac_stan_reward, deepternary_reward



# ============================================================
# Convenience accumulator for test-set sampling loops
# ============================================================

class ProtacMetricAccumulator:
    """
    Accumulate records across all test batches and K sampling rounds.

    Example:
        acc = ProtacMetricAccumulator()
        for batch in test_loader:
            for sample_id in range(K):
                pred_molecules = ...
                true_molecules = ...
                true_fragments = ...
                acc.add_batch(pred_molecules, batch, true_molecules, true_fragments, sample_id)

        metrics = acc.compute(protac_stan_reward=..., deepternary_reward=...)
    """

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []


    def add_batch(
            self,
            pred_molecules: Sequence[Optional[Chem.Mol]],
            batch: Dict[str, Any],
            true_molecules: Optional[
                Sequence[Optional[Chem.Mol]]
            ] = None,
            true_fragments: Optional[
                Sequence[Optional[Chem.Mol]]
            ] = None,
            batch_idx: Optional[int] = None,
            sample_id: Optional[Union[int, str]] = None,
            strict_flags: Optional[Sequence[Any]] = None,
            strict_infos: Optional[
                Sequence[Dict[str, Any]]
            ] = None,
            degradation_scores: Any = None,
            ternary_scores: Any = None,
            deepternary_info: Optional[
                Dict[str, Any]
            ] = None,
    ) -> None:
        self.records.extend(
            build_records_from_batch(
                pred_molecules=pred_molecules,
                batch=batch,
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
        )

    def compute(self, **kwargs: Any) -> Union[Dict[str, float], Tuple[Dict[str, float], List[Dict[str, Any]]]]:
        return compute_metrics_from_records(self.records, **kwargs)

