import os
import pickle
from typing import Dict, List

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import logging
import wandb
import time

from rdkit import Chem
from tqdm import tqdm

from data import datasets
from data.datasets import collate, create_templates_for_linker_generation
from torch.utils.data import DataLoader
from src import const, utils
from model import LinkerDiffusionModel
from omegaconf import DictConfig
from omegaconf import OmegaConf
from datetime import datetime
from collections import defaultdict
from data.molecule_builder import build_molecules
import random
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")


def get_atom_type_ids_by_symbols(symbols):
    symbols = set(symbols)

    if isinstance(const.IDX2ATOM, dict):
        return [
            int(idx)
            for idx, sym in const.IDX2ATOM.items()
            if sym in symbols
        ]

    return [
        idx
        for idx, sym in enumerate(const.IDX2ATOM)
        if sym in symbols
    ]

def get_center_of_mass_mask(exp_conf, anchors, fragment_mask, node_mask):
    """
    Select the center-of-mass reference mask.

    Supported modes are anchors, fragments, and all atoms.
    """
    mode = getattr(exp_conf, "center_of_mass", "fragments")

    if mode == "anchors":
        return anchors

    if mode == "fragments":
        return fragment_mask

    if mode == "all":
        return node_mask

    raise NotImplementedError(f"Unknown center_of_mass mode: {mode}")

def squeeze_node_mask(mask):
    """
    [B, N, 1] or [B, N] -> [B, N]
    """
    if mask.dim() == 3:
        mask = mask.squeeze(-1)
    return mask.float()


def build_bond_type_masks(
        anchor1_mask,
        anchor2_mask,
        linker_mask,
        bond_mask,
):
    """
    Build upper-triangular boolean bond-supervision masks [B, N, N]
    for anchor1-linker, anchor2-linker, and linker-linker pairs.
    """

    anchor1_mask = squeeze_node_mask(anchor1_mask)
    anchor2_mask = squeeze_node_mask(anchor2_mask)
    linker_mask = squeeze_node_mask(linker_mask)

    if bond_mask.dim() == 4:
        bond_mask = bond_mask.squeeze(-1)

    bond_mask = bond_mask.bool()

    B, N = linker_mask.shape
    device = linker_mask.device

    triu_mask = torch.triu(
        torch.ones(N, N, device=device, dtype=torch.bool),
        diagonal=1,
    )[None, :, :]

    # anchor1-linker pair
    a1_l_mask = (
            anchor1_mask[:, :, None] * linker_mask[:, None, :]
            + linker_mask[:, :, None] * anchor1_mask[:, None, :]
    ) > 0.5

    # anchor2-linker pair
    a2_l_mask = (
            anchor2_mask[:, :, None] * linker_mask[:, None, :]
            + linker_mask[:, :, None] * anchor2_mask[:, None, :]
    ) > 0.5

    # linker-linker pair
    l_l_mask = (
            linker_mask[:, :, None] * linker_mask[:, None, :]
    ) > 0.5

    valid_mask = bond_mask & triu_mask

    a1_l_mask = a1_l_mask & valid_mask
    a2_l_mask = a2_l_mask & valid_mask
    l_l_mask = l_l_mask & valid_mask

    return {
        "anchor1_linker": a1_l_mask,
        "anchor2_linker": a2_l_mask,
        "linker_linker": l_l_mask,
    }



def typed_bond_loss_on_pair_mask(
        bond_type_logits,
        bond_type_labels,
        pair_mask,
        max_pos_weight=50.0,
        type_weight=1.0,
        linker_positive_weight=1.2,
):
    """
    Compute linker-linker bond existence and type supervision.

    Reweight all positive linker-linker edges, including ring closures.
    Bond labels represent none, single, double, triple, and aromatic.
    """

    if pair_mask.sum().item() == 0:
        zero = bond_type_logits.sum() * 0.0
        return zero, zero, zero

    logits = bond_type_logits[pair_mask]          # [M, 5]
    labels = bond_type_labels[pair_mask].long()   # [M]

    # existence: no bond vs any bond
    exist_logit = torch.logsumexp(
        logits[:, 1:],
        dim=-1,
    ) - logits[:, 0]

    exist_labels = (labels > 0).float()

    num_pos = exist_labels.sum()
    num_total = exist_labels.numel()
    num_neg = num_total - num_pos

    if num_pos.item() < 0.5:
        pos_weight = torch.tensor(
            1.0,
            device=logits.device,
            dtype=logits.dtype,
        )
    else:
        pos_weight = num_neg / (num_pos + 1e-8)
        pos_weight = pos_weight.clamp(
            min=1.0,
            max=max_pos_weight,
        )

    # Retain unreduced losses for positive-edge reweighting.
    exist_loss_each = F.binary_cross_entropy_with_logits(
        exist_logit,
        exist_labels,
        pos_weight=pos_weight,
        reduction="none",
    )

    # Reweight positive existence examples by linker_positive_weight.
    sample_weight = torch.ones_like(exist_loss_each)

    positive_mask = labels > 0

    sample_weight = torch.where(
        positive_mask,
        torch.full_like(sample_weight, float(linker_positive_weight)),
        sample_weight,
    )

    exist_loss = (
        exist_loss_each * sample_weight
    ).sum() / sample_weight.sum().clamp(min=1.0)

    # Classify all positive bond types, including aromatic,
    # without applying additional type-loss weighting.
    pos_mask = labels > 0

    if pos_mask.sum().item() > 0:
        type_logits = logits[pos_mask, 1:]      # [P, 4]
        type_labels = labels[pos_mask] - 1      # 1..4 -> 0..3

        type_loss = F.cross_entropy(
            type_logits,
            type_labels,
        )
    else:
        type_loss = bond_type_logits.sum() * 0.0

    loss = exist_loss + type_weight * type_loss

    return loss, exist_loss, type_loss

def anchor_linker_bond_loss_on_pair_mask(
        bond_type_logits,
        bond_type_labels,
        pair_mask,
        max_pos_weight=50.0,
        type_weight=1.0,
):
    """
    Compute anchor-linker bond existence and type losses.

    Only single, double, and triple bonds are allowed.
    Aromatic bonds are grouped with no-bond on the negative side
    of the existence classifier.
    """

    if pair_mask.sum().item() == 0:
        zero = bond_type_logits.sum() * 0.0
        return zero, zero, zero

    logits = bond_type_logits[pair_mask]          # [M, 5]
    labels = bond_type_labels[pair_mask].long()   # [M]

    # ------------------------------------------------
    # allowed existence:
    #   allowed bond = single/double/triple
    #   negative/invalid = no bond or aromatic
    # ------------------------------------------------
    allowed_logit = torch.logsumexp(
        logits[:, 1:4],     # single, double, triple
        dim=-1,
    )

    not_allowed_logit = torch.logsumexp(
        torch.stack(
            [
                logits[:, 0],   # no bond
                logits[:, 4],   # aromatic is invalid for anchor-linker pairs
            ],
            dim=-1,
        ),
        dim=-1,
    )

    exist_logit = allowed_logit - not_allowed_logit

    exist_labels = (
        (labels >= 1) & (labels <= 3)
    ).float()

    num_pos = exist_labels.sum()
    num_total = exist_labels.numel()
    num_neg = num_total - num_pos

    if num_pos.item() < 0.5:
        pos_weight = torch.tensor(
            1.0,
            device=logits.device,
            dtype=logits.dtype,
        )
    else:
        pos_weight = num_neg / (num_pos + 1e-8)
        pos_weight = pos_weight.clamp(
            min=1.0,
            max=max_pos_weight,
        )

    exist_loss = F.binary_cross_entropy_with_logits(
        exist_logit,
        exist_labels,
        pos_weight=pos_weight,
    )

    # Classify bond types only on positive anchor-linker edges.
    pos_mask = (labels >= 1) & (labels <= 3)

    if pos_mask.sum().item() > 0:
        type_logits = logits[pos_mask, 1:4]    # [P, 3]
        type_labels = labels[pos_mask] - 1     # 1,2,3 -> 0,1,2

        type_loss = F.cross_entropy(
            type_logits,
            type_labels,
        )
    else:
        type_loss = bond_type_logits.sum() * 0.0

    loss = exist_loss + type_weight * type_loss

    return loss, exist_loss, type_loss
def explicit_linker_bond_type_loss(
        bond_type_logits,
        bond_type_labels,
        bond_mask,
        anchor1_mask,
        anchor2_mask,
        linker_mask,
        anchor1_weight=1.5,
        anchor2_weight=1.5,
        linker_weight=4.0,
        max_pos_weight=50.0,
        type_weight=1.0,
        linker_positive_weight=1.2,
):
    """
    Supervise anchor1-linker, anchor2-linker, and linker-linker bond types.

    Apply additional positive-edge weighting to linker-linker bonds,
    including ring-closing edges, without distinguishing tree and
    cycle edges.
    """

    if bond_type_labels.dim() == 4:
        bond_type_labels = bond_type_labels.squeeze(-1)

    if bond_mask.dim() == 4:
        bond_mask = bond_mask.squeeze(-1)

    bond_type_labels = bond_type_labels.long()
    bond_mask = bond_mask.bool()

    pair_masks = build_bond_type_masks(
        anchor1_mask=anchor1_mask,
        anchor2_mask=anchor2_mask,
        linker_mask=linker_mask,
        bond_mask=bond_mask,
    )

    # Anchor1-linker: allow single, double, and triple; exclude aromatic.
    a1_loss, a1_exist, a1_type = anchor_linker_bond_loss_on_pair_mask(
        bond_type_logits=bond_type_logits,
        bond_type_labels=bond_type_labels,
        pair_mask=pair_masks["anchor1_linker"],
        max_pos_weight=max_pos_weight,
        type_weight=type_weight,
    )

    # Anchor2-linker: allow single, double, and triple; exclude aromatic.
    a2_loss, a2_exist, a2_type = anchor_linker_bond_loss_on_pair_mask(
        bond_type_logits=bond_type_logits,
        bond_type_labels=bond_type_labels,
        pair_mask=pair_masks["anchor2_linker"],
        max_pos_weight=max_pos_weight,
        type_weight=type_weight,
    )


    ll_loss, ll_exist, ll_type = typed_bond_loss_on_pair_mask(
        bond_type_logits=bond_type_logits,
        bond_type_labels=bond_type_labels,
        pair_mask=pair_masks["linker_linker"],
        max_pos_weight=max_pos_weight,
        type_weight=type_weight,
        linker_positive_weight=linker_positive_weight,
    )

    total_weight = (
            anchor1_weight
            + anchor2_weight
            + linker_weight
    )

    total_loss = (
            anchor1_weight * a1_loss
            + anchor2_weight * a2_loss
            + linker_weight * ll_loss
    ) / total_weight

    aux = {
        "bond_anchor1_linker_loss": (anchor1_weight / total_weight * a1_loss).detach(),
        "bond_anchor2_linker_loss": (anchor2_weight / total_weight * a2_loss).detach(),
        "bond_linker_linker_loss": (linker_weight / total_weight * ll_loss).detach(),

        "bond_anchor1_linker_has_pair": float(
            pair_masks["anchor1_linker"].sum().item() > 0
        ),
        "bond_anchor2_linker_has_pair": float(
            pair_masks["anchor2_linker"].sum().item() > 0
        ),
        "bond_linker_linker_has_pair": float(
            pair_masks["linker_linker"].sum().item() > 0
        ),

        "bond_anchor1_exist_loss": a1_exist.detach(),
        "bond_anchor2_exist_loss": a2_exist.detach(),
        "bond_linker_linker_exist_loss": ll_exist.detach(),

        "bond_anchor1_type_loss": a1_type.detach(),
        "bond_anchor2_type_loss": a2_type.detach(),
        "bond_linker_linker_type_loss": ll_type.detach(),

        "bond_linker_positive_weight": float(linker_positive_weight),
    }

    return total_loss, aux



def true_linker_rebuild_bond_length_loss(
    x_pred,
    x_true,
    linker_mask,
    bond_labels,
    anchor1_mask=None,
    anchor2_mask=None,
    bond_mask=None,
    beta=0.02,
):
    """
    Penalize predicted linker-related bond lengths against reference lengths.

    Include true linker-linker and anchor-linker bonds. Average the
    Smooth L1 loss per molecule, then average equally across molecules
    containing eligible supervised bonds.
    """

    if linker_mask.dim() == 3:
        linker_mask = linker_mask.squeeze(-1)
    linker_mask = linker_mask.bool()

    if anchor1_mask is not None and anchor1_mask.dim() == 3:
        anchor1_mask = anchor1_mask.squeeze(-1)
    if anchor2_mask is not None and anchor2_mask.dim() == 3:
        anchor2_mask = anchor2_mask.squeeze(-1)

    if anchor1_mask is not None:
        anchor1_mask = anchor1_mask.bool()
    if anchor2_mask is not None:
        anchor2_mask = anchor2_mask.bool()

    if bond_mask is not None:
        if bond_mask.dim() == 4:
            bond_mask = bond_mask.squeeze(-1)
        bond_mask = bond_mask.bool()

    if bond_labels.dim() == 4:
        if bond_labels.size(-1) == 1:
            bond_labels = bond_labels.squeeze(-1)
        else:
            bond_labels = bond_labels.argmax(dim=-1)

    bond_labels = bond_labels.long()

    B, N, _ = x_pred.shape
    device = x_pred.device

    d_pred = torch.cdist(x_pred, x_pred)

    with torch.no_grad():
        d_true = torch.cdist(x_true, x_true)

    triu_mask = torch.triu(
        torch.ones(N, N, device=device, dtype=torch.bool),
        diagonal=1,
    )

    loss_per_sample = []

    for b in range(B):
        linker = linker_mask[b]

        ll_mask = linker[:, None] & linker[None, :]

        if anchor1_mask is not None:
            a1 = anchor1_mask[b]
            a1_l_mask = (
                (a1[:, None] & linker[None, :])
                | (linker[:, None] & a1[None, :])
            )
        else:
            a1_l_mask = torch.zeros(N, N, device=device, dtype=torch.bool)

        if anchor2_mask is not None:
            a2 = anchor2_mask[b]
            a2_l_mask = (
                (a2[:, None] & linker[None, :])
                | (linker[:, None] & a2[None, :])
            )
        else:
            a2_l_mask = torch.zeros(N, N, device=device, dtype=torch.bool)

        linker_related_mask = ll_mask | a1_l_mask | a2_l_mask
        true_bond_mask = bond_labels[b] > 0

        pair_mask_b = (
            linker_related_mask
            & true_bond_mask
            & triu_mask
        )

        if bond_mask is not None:
            pair_mask_b = pair_mask_b & bond_mask[b]

        if pair_mask_b.sum().item() == 0:
            continue

        loss_b = F.smooth_l1_loss(
            d_pred[b][pair_mask_b],
            d_true[b][pair_mask_b],
            beta=beta,
            reduction="mean",
        )

        loss_per_sample.append(loss_b)

    if len(loss_per_sample) == 0:
        return x_pred.sum() * 0.0

    return torch.stack(loss_per_sample).mean()

def rebuild_consistent_linker_no_isolated_loss(
    x_pred_A,
    x_true_A,
    bond_type_logits,
    bond_type_labels,
    anchors,
    linker_mask,
    distance_margin_A=0.05,
    sharpness=40.0,
):
    """
    Encourage each eligible linker atom to have a plausible neighbor.

    Combine predicted bond existence with a soft distance match to
    reference incident-bond lengths within distance_margin_A.
    No fixed global bond-length window is used.
    """

    # ------------------------------------------------
    # mask shape cleanup
    # ------------------------------------------------
    if anchors.dim() == 3:
        anchors = anchors.squeeze(-1)

    if linker_mask.dim() == 3:
        linker_mask = linker_mask.squeeze(-1)

    if bond_type_labels.dim() == 4:
        if bond_type_labels.size(-1) == 1:
            bond_type_labels = bond_type_labels.squeeze(-1)
        else:
            bond_type_labels = bond_type_labels.argmax(dim=-1)

    anchors = anchors.float()
    linker_mask = linker_mask.float()
    bond_type_labels = bond_type_labels.long()

    B, N, _ = x_pred_A.shape
    device = x_pred_A.device
    dtype = x_pred_A.dtype

    # ------------------------------------------------
    # predicted / true pair distances, Å
    # ------------------------------------------------
    d_pred_A = torch.cdist(x_pred_A, x_pred_A)  # [B, N, N]

    with torch.no_grad():
        d_true_A = torch.cdist(x_true_A, x_true_A)  # [B, N, N]

    # Bond logits: [B, N, N, 5] for none/single/double/triple/aromatic.
    if bond_type_logits.dim() != 4 or bond_type_logits.size(-1) != 5:
        raise ValueError(
            "bond_type_logits should have shape [B, N, N, 5]"
        )

    # Bond existence rules:
    # Linker-linker: single, double, triple, and aromatic are allowed.
    # Anchor-linker: aromatic is grouped with no-bond as invalid.
    anchors_bool = anchors.bool()
    linker_bool = linker_mask.bool()

    anchor_linker_pair_mask = (
            (anchors_bool[:, :, None] & linker_bool[:, None, :])
            |
            (linker_bool[:, :, None] & anchors_bool[:, None, :])
    )  # [B, N, N]

    # General pairs allow all nonzero bond types, including aromatic.
    default_allowed_logit = torch.logsumexp(
        bond_type_logits[..., 1:5],
        dim=-1,
    )

    default_not_allowed_logit = bond_type_logits[..., 0]

    default_exist_logit = (
            default_allowed_logit - default_not_allowed_logit
    )

    anchor_allowed_logit = torch.logsumexp(
        bond_type_logits[..., 1:4],
        dim=-1,
    )

    # Treat aromatic and no-bond logits as invalid for anchor-linker pairs.
    anchor_not_allowed_logit = torch.logsumexp(
        torch.stack(
            [
                bond_type_logits[..., 0],  # no bond
                bond_type_logits[..., 4],  # aromatic, invalid for anchor-linker
            ],
            dim=-1,
        ),
        dim=-1,
    )

    anchor_exist_logit = (
            anchor_allowed_logit - anchor_not_allowed_logit
    )

    # Apply anchor-linker-specific existence logits to anchor-linker pairs.
    exist_logit = torch.where(
        anchor_linker_pair_mask,
        anchor_exist_logit,
        default_exist_logit,
    )

    p_bond = torch.sigmoid(exist_logit)  # [B, N, N]

    # Symmetrize bond probabilities for undirected edges.
    p_bond = 0.5 * (
            p_bond + p_bond.transpose(1, 2)
    )

    # Candidate pairs: linker rows with linker or anchor columns.
    neighbor_mask = linker_mask[:, :, None] * (
        linker_mask[:, None, :] + anchors[:, None, :]
    )
    neighbor_mask = neighbor_mask.clamp(max=1.0)

    eye = torch.eye(
        N,
        device=device,
        dtype=dtype,
    )[None, :, :]

    neighbor_mask = neighbor_mask * (1.0 - eye)

    # Reference linker-linker bonds include aromatic edges.
    # Anchor-linker bonds are restricted to single, double, and triple.

    true_bond_default = bond_type_labels > 0

    true_bond_anchor_linker = (
            (bond_type_labels >= 1)
            & (bond_type_labels <= 3)
    )

    true_bond_mask = torch.where(
        anchor_linker_pair_mask,
        true_bond_anchor_linker,
        true_bond_default,
    ).float()

    # Symmetrize reference bond labels for undirected edges.
    true_bond_mask = torch.maximum(
        true_bond_mask,
        true_bond_mask.transpose(1, 2),
    )

    true_neighbor_bond_mask = neighbor_mask * true_bond_mask  # [B, N, N]

    # Supervise only linker atoms with at least one true linker/anchor bond.
    true_neighbor_count = true_neighbor_bond_mask.sum(dim=-1)

    eligible_mask = linker_mask * (
        true_neighbor_count > 0
    ).float()

    # Match predicted distances to reference incident-bond lengths
    # within +/- distance_margin_A.
    # d_pred_expand: [B, i, j, 1]
    # d_true_expand: [B, i, 1, k]
    # true_neighbor_expand: [B, i, 1, k]
    d_pred_expand = d_pred_A[:, :, :, None]
    d_true_expand = d_true_A[:, :, None, :]

    true_neighbor_expand = true_neighbor_bond_mask[:, :, None, :]

    lower_A = d_true_expand - distance_margin_A
    upper_A = d_true_expand + distance_margin_A

    p_ge_lower = torch.sigmoid(
        sharpness * (d_pred_expand - lower_A)
    )

    p_le_upper = torch.sigmoid(
        sharpness * (upper_A - d_pred_expand)
    )

    p_in_true_window = p_ge_lower * p_le_upper

    p_in_true_window = p_in_true_window * true_neighbor_expand

    # Match each candidate neighbor against any true incident-bond length.
    p_dist = p_in_true_window.max(dim=-1).values  # [B, N, N]

    # Combine bond existence, distance matching, and neighbor masks.
    p_edge = p_bond * p_dist * neighbor_mask

    # Probability of at least one valid neighbor per linker atom:
    # p_has_neighbor(i) = 1 - product_j(1 - p_edge(i, j)).
    log_no_neighbor = torch.log(
        (1.0 - p_edge).clamp(min=1e-8)
    ).sum(dim=-1)

    p_has_neighbor = 1.0 - torch.exp(log_no_neighbor)

    loss = -torch.log(
        p_has_neighbor.clamp(min=1e-8)
    )

    loss = loss * eligible_mask

    return loss.sum() / (eligible_mask.sum() + 1e-8)

def is_rdkit_fully_valid(mol):
    """
    Check full RDKit sanitization on a copy of the input molecule.
    """

    if mol is None:
        return False

    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        return True
    except Exception:
        return False
def diagnose_generated_connectivity(
    mol,
    node_mask_i,
    fragment1_mask_i,
    fragment2_mask_i,
    linker_mask_i,
    anchor1_mask_i,
    anchor2_mask_i,
    name="mol",
    verbose=False,
):
    """
    Diagnose connectivity and strict topology of generated PROTACs.

    Strict topology requires RDKit validity, a connected full molecule,
    an internally connected linker, exactly one attachment to each
    anchor, and no forbidden fragment shortcuts.
    """

    if mol is None:
        return {
            "valid_mol": False,
            "reason": "mol_is_none",
        }

    rdkit_valid = is_rdkit_fully_valid(mol)

    def squeeze_mask(m):
        if m.dim() == 2:
            m = m.squeeze(-1)
        return m.detach().cpu().bool()

    node_mask_i = squeeze_mask(node_mask_i)
    fragment1_mask_i = squeeze_mask(fragment1_mask_i)
    fragment2_mask_i = squeeze_mask(fragment2_mask_i)
    linker_mask_i = squeeze_mask(linker_mask_i)
    anchor1_mask_i = squeeze_mask(anchor1_mask_i)
    anchor2_mask_i = squeeze_mask(anchor2_mask_i)

    valid = node_mask_i

    f1 = fragment1_mask_i[valid].numpy()
    f2 = fragment2_mask_i[valid].numpy()
    linker = linker_mask_i[valid].numpy()
    a1 = anchor1_mask_i[valid].numpy()
    a2 = anchor2_mask_i[valid].numpy()

    n_atoms_mol = mol.GetNumAtoms()
    n_atoms_mask = len(linker)

    if n_atoms_mol != n_atoms_mask:
        n = min(n_atoms_mol, n_atoms_mask)
        f1 = f1[:n]
        f2 = f2[:n]
        linker = linker[:n]
        a1 = a1[:n]
        a2 = a2[:n]

    n_valid_atoms = len(linker)

    direct_a1_linker_bonds = []
    direct_a2_linker_bonds = []
    linker_linker_bonds = []

    fragment1_fragment2_bonds = []
    bad_non_anchor_fragment_linker_bonds = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        if i >= n_valid_atoms or j >= n_valid_atoms:
            continue

        # anchor1-linker
        if (a1[i] and linker[j]) or (a1[j] and linker[i]):
            direct_a1_linker_bonds.append((i, j))

        # anchor2-linker
        if (a2[i] and linker[j]) or (a2[j] and linker[i]):
            direct_a2_linker_bonds.append((i, j))

        # linker-linker
        if linker[i] and linker[j]:
            linker_linker_bonds.append((i, j))

        # Detect direct fragment1-fragment2 bonds.
        if (f1[i] and f2[j]) or (f2[i] and f1[j]):
            fragment1_fragment2_bonds.append((i, j))

        # Detect non-anchor fragment1-linker bonds.
        if (f1[i] and linker[j]) or (f1[j] and linker[i]):
            frag_atom = i if f1[i] else j
            if not a1[frag_atom]:
                bad_non_anchor_fragment_linker_bonds.append((i, j))

        # Detect non-anchor fragment2-linker bonds.
        if (f2[i] and linker[j]) or (f2[j] and linker[i]):
            frag_atom = i if f2[i] else j
            if not a2[frag_atom]:
                bad_non_anchor_fragment_linker_bonds.append((i, j))

    linker_indices = np.where(linker)[0].tolist()

    parent = {idx: idx for idx in linker_indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i, j in linker_linker_bonds:
        if i in parent and j in parent:
            union(i, j)

    linker_components = {}
    for idx in linker_indices:
        root = find(idx)
        linker_components.setdefault(root, []).append(idx)

    n_linker_components = len(linker_components)

    if len(linker_indices) <= 1:
        linker_internal_connected = True
    else:
        linker_internal_connected = n_linker_components == 1

    comps = Chem.GetMolFrags(
        mol,
        asMols=False,
        sanitizeFrags=False,
    )

    full_bridge_component = False
    isolated_linker_component = False

    for comp in comps:
        comp = [idx for idx in comp if idx < n_valid_atoms]

        if len(comp) == 0:
            continue

        comp_f1 = int(f1[comp].sum())
        comp_f2 = int(f2[comp].sum())
        comp_linker = int(linker[comp].sum())

        if comp_linker > 0 and comp_f1 == 0 and comp_f2 == 0:
            isolated_linker_component = True

        if comp_f1 > 0 and comp_f2 > 0 and comp_linker > 0:
            full_bridge_component = True

    whole_mol_connected = len(comps) == 1

    exact_one_a1_linker_bond = len(direct_a1_linker_bonds) == 1
    exact_one_a2_linker_bond = len(direct_a2_linker_bonds) == 1

    has_fragment1_fragment2_bond = len(fragment1_fragment2_bonds) > 0
    has_bad_non_anchor_fragment_linker_bond = (
        len(bad_non_anchor_fragment_linker_bonds) > 0
    )

    strict_protac_topology = (
        rdkit_valid
        and whole_mol_connected
        and full_bridge_component
        and linker_internal_connected
        and exact_one_a1_linker_bond
        and exact_one_a2_linker_bond
        and not has_fragment1_fragment2_bond
        and not has_bad_non_anchor_fragment_linker_bond
    )

    result = {
        "valid_mol": True,
        "rdkit_valid": rdkit_valid,

        # anchor-linker
        "n_a1_linker_bonds": len(direct_a1_linker_bonds),
        "n_a2_linker_bonds": len(direct_a2_linker_bonds),

        # linker-linker
        "n_linker_linker_bonds": len(linker_linker_bonds),
        "n_linker_components": n_linker_components,
        "linker_internal_connected": linker_internal_connected,

        # bridge
        "full_bridge_component": full_bridge_component,
        "isolated_linker_component": isolated_linker_component,

        # strict topology components
        "exact_one_a1_linker_bond": exact_one_a1_linker_bond,
        "exact_one_a2_linker_bond": exact_one_a2_linker_bond,
        "has_fragment1_fragment2_bond": has_fragment1_fragment2_bond,
        "has_bad_non_anchor_fragment_linker_bond": has_bad_non_anchor_fragment_linker_bond,

        # final strict metric
        "strict_protac_topology": strict_protac_topology,
    }

    if verbose:
        print(f"\n========== PROTAC topology diagnosis: {name} ==========")
        print("strict_protac_topology:", strict_protac_topology)
        print("full_bridge_component:", full_bridge_component)
        print("linker_internal_connected:", linker_internal_connected)
        print("n_linker_components:", n_linker_components)
        print("n_a1_linker_bonds:", len(direct_a1_linker_bonds))
        print("n_a2_linker_bonds:", len(direct_a2_linker_bonds))
        print("n_linker_linker_bonds:", len(linker_linker_bonds))
        print("has_fragment1_fragment2_bond:", has_fragment1_fragment2_bond)
        print(
            "has_bad_non_anchor_fragment_linker_bond:",
            has_bad_non_anchor_fragment_linker_bond,
        )
        print("====================================================\n")

    return result

def update_connectivity_debug_counter(counter, diag):
    """
    Accumulate PROTAC topology diagnostics across generated molecules.
    """

    counter["total_diagnosed"] += 1

    if not diag.get("valid_mol", False):
        counter["problem_mol_is_none"] += 1
        return

    n_a1 = int(diag.get("n_a1_linker_bonds", 0))
    n_a2 = int(diag.get("n_a2_linker_bonds", 0))
    n_ll = int(diag.get("n_linker_linker_bonds", 0))
    n_lc = int(diag.get("n_linker_components", 0))

    if diag.get("rdkit_valid", False):
        counter["rdkit_valid"] += 1

    # strict topology
    if diag.get("strict_protac_topology", False):
        counter["strict_protac_topology"] += 1

    # bridge / linker
    if diag.get("full_bridge_component", False):
        counter["full_bridge_component"] += 1

    if diag.get("linker_internal_connected", False):
        counter["linker_internal_connected"] += 1

    if diag.get("isolated_linker_component", False):
        counter["isolated_linker_component"] += 1

    # missing anchor-linker
    if n_a1 == 0:
        counter["missing_a1_linker_bond"] += 1

    if n_a2 == 0:
        counter["missing_a2_linker_bond"] += 1

    # exact one anchor-linker
    if diag.get("exact_one_a1_linker_bond", False):
        counter["exact_one_a1_linker_bond"] += 1

    if diag.get("exact_one_a2_linker_bond", False):
        counter["exact_one_a2_linker_bond"] += 1

    # illegal topology
    if diag.get("has_fragment1_fragment2_bond", False):
        counter["has_fragment1_fragment2_bond"] += 1

    if diag.get("has_bad_non_anchor_fragment_linker_bond", False):
        counter["has_bad_non_anchor_fragment_linker_bond"] += 1

    # average stats
    counter["sum_n_linker_components"] += n_lc
    counter["sum_n_a1_linker_bonds"] += n_a1
    counter["sum_n_a2_linker_bonds"] += n_a2
    counter["sum_n_linker_linker_bonds"] += n_ll


def add_connectivity_debug_results(results, counter, prefix):
    """
    Aggregate connectivity diagnostics into normalized evaluation metrics.

    Report strict topology, linker connectivity, anchor attachments,
    forbidden bonds, and average bond/component counts.
    """

    total = counter.get("total_diagnosed", 0)

    results[f"{prefix}_total_diagnosed"] = float(total)

    if total <= 0:
        return results

    total = float(total)

    results[f"{prefix}_rdkit_valid_ratio"] = (
            float(counter.get("rdkit_valid", 0)) / total
    )

    results[f"{prefix}_strict_protac_topology_ratio"] = (
        float(counter.get("strict_protac_topology", 0)) / total
    )

    results[f"{prefix}_full_bridge_component_ratio"] = (
        float(counter.get("full_bridge_component", 0)) / total
    )

    results[f"{prefix}_linker_internal_connected_ratio"] = (
        float(counter.get("linker_internal_connected", 0)) / total
    )

    results[f"{prefix}_isolated_linker_component_ratio"] = (
        float(counter.get("isolated_linker_component", 0)) / total
    )

    results[f"{prefix}_missing_a1_linker_bond_ratio"] = (
        float(counter.get("missing_a1_linker_bond", 0)) / total
    )

    results[f"{prefix}_missing_a2_linker_bond_ratio"] = (
        float(counter.get("missing_a2_linker_bond", 0)) / total
    )

    results[f"{prefix}_exact_one_a1_linker_bond_ratio"] = (
        float(counter.get("exact_one_a1_linker_bond", 0)) / total
    )

    results[f"{prefix}_exact_one_a2_linker_bond_ratio"] = (
        float(counter.get("exact_one_a2_linker_bond", 0)) / total
    )

    results[f"{prefix}_has_fragment1_fragment2_bond_ratio"] = (
        float(counter.get("has_fragment1_fragment2_bond", 0)) / total
    )

    results[f"{prefix}_has_bad_non_anchor_fragment_linker_bond_ratio"] = (
        float(counter.get("has_bad_non_anchor_fragment_linker_bond", 0)) / total
    )

    results[f"{prefix}_avg_n_linker_components"] = (
        float(counter.get("sum_n_linker_components", 0)) / total
    )

    results[f"{prefix}_avg_n_a1_linker_bonds"] = (
        float(counter.get("sum_n_a1_linker_bonds", 0)) / total
    )

    results[f"{prefix}_avg_n_a2_linker_bonds"] = (
        float(counter.get("sum_n_a2_linker_bonds", 0)) / total
    )

    results[f"{prefix}_avg_n_linker_linker_bonds"] = (
        float(counter.get("sum_n_linker_linker_bonds", 0)) / total
    )

    return results


def position_loss(x_pred, x_true, mask):
    """
    Compute masked coordinate MSE.

    Average over linker atoms and coordinate dimensions within each
    molecule, then average equally across molecules in the batch.
    """

    if mask.dim() == 3:
        mask = mask.squeeze(-1)

    mask = mask.float()  # [B, N]

    sq_error = ((x_pred - x_true) ** 2).sum(dim=-1)  # [B, N]
    sq_error = sq_error * mask

    linker_atom_num = mask.sum(dim=1).clamp(min=1.0)  # [B]
    loss_per_sample = sq_error.sum(dim=1) / (linker_atom_num * x_true.size(-1) + 1e-8)

    return loss_per_sample.mean()

def seed_everything(seed: int = 2026):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Request deterministic cuDNN behavior.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {
            k: move_to_device(v, device)
            for k, v in batch.items()
        }
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    else:
        return batch

class Experiment:
    metrics: Dict[str, List[float]] = {}
    FRAMES = 5

    def __init__(
            self,
            *,
            conf: DictConfig,
        ):

        self._log = logging.getLogger(__name__)

        self._conf = conf
        self._exp_conf = conf.experiment
        self._model_conf = conf.model

        self.seed = int(self._exp_conf.seed)
        seed_everything(self.seed)
        self._log.info(f"Using random seed: {self.seed}")

        self._use_wandb = self._exp_conf.use_wandb
        if self._exp_conf.use_gpu:
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        self.trained_epochs = 0
        self.trained_steps = 0
        self.val_steps = 0
        self.T = self._exp_conf.diffusion_step

        self.model_weights = None
        self._model_conf.num_atom_types = const.NUMBER_OF_ATOM_TYPES

        self._model = LinkerDiffusionModel.LinkerDiffusionModel(
            self._model_conf, self._exp_conf)

        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._exp_conf.num_parameters = num_parameters
        self._log.info(f'Model parameters: {num_parameters}')
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._exp_conf.learning_rate)

        dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")

        if self._exp_conf.ckpt_dir is not None:
            ckpt_dir = os.path.join(
                self._exp_conf.ckpt_dir,
                self._exp_conf.name,
                dt_string)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.ckpt_dir = ckpt_dir
            self._log.info(f'Checkpoints will be saved to: {ckpt_dir}')
        else:
            self._log.info('Checkpoint saving is disabled.')

        if self._exp_conf.eval_dir is not None :
            eval_dir = os.path.join(
                self._exp_conf.eval_dir,
                self._exp_conf.name,
                dt_string)
            self._exp_conf.eval_dir = eval_dir
            self._log.info(f'Evaluation outputs will be saved to: {eval_dir}')
        else:
            self._exp_conf.eval_dir = os.devnull
            self._log.info(f'Evaluation will not be saved.')

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf
    @property
    def device(self):
        return self._device

    @staticmethod
    def sample_combined_position_feature_noise(n_samples, n_nodes, mask):
        z_x = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, 3),
            device=mask.device,
            node_mask=mask
        )

        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, const.NUMBER_OF_ATOM_TYPES),
            device=mask.device,
            node_mask=mask
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z


    def get_aux_loss_scale(self):
        """
        Return the auxiliary-loss scale.

        Keep aux_min_scale during warmup, then increase linearly to 1.0
        over aux_ramp_epochs. Use 1.0 afterward.
        """

        current_epoch = self.trained_epochs + 1

        aux_min_scale = float(self._exp_conf.aux_min_scale)
        aux_warmup_epochs = int(self._exp_conf.aux_warmup_epochs)
        aux_ramp_epochs = int(self._exp_conf.aux_ramp_epochs)

        if current_epoch <= aux_warmup_epochs:
            return aux_min_scale

        if aux_ramp_epochs <= 0:
            return 1.0

        progress = (
                           current_epoch - aux_warmup_epochs
                   ) / float(aux_ramp_epochs)

        progress = max(0.0, min(1.0, progress))

        aux_scale = aux_min_scale + (1.0 - aux_min_scale) * progress

        return aux_scale


    def create_dataset(self):
        train_dataset = datasets.ProtacDataset(
            data_path=self._exp_conf.data_path,
            dir='train',
            prefix=self._exp_conf.prefix_pretrain,
            device="cpu"
        )

        valid_dataset = datasets.ProtacDataset(
            data_path=self._exp_conf.data_path,
            dir='val',
            prefix=self._exp_conf.prefix_preval,
            device="cpu"
        )

        seed = int(self._exp_conf.seed)

        train_generator = torch.Generator()
        train_generator.manual_seed(seed)

        valid_generator = torch.Generator()
        valid_generator.manual_seed(seed + 1)

        # Create a shuffled training DataLoader with deterministic seeding.
        train_loader = DataLoader(dataset=train_dataset, batch_size=self._exp_conf.train_batch_size, collate_fn=collate, shuffle=True, worker_init_fn=seed_worker, generator=train_generator, num_workers=0, pin_memory=False)
        # Create a non-shuffled validation DataLoader.
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=self._exp_conf.val_batch_size, collate_fn=collate, shuffle=False, worker_init_fn=seed_worker, generator=valid_generator, num_workers=0, pin_memory=False)
        return train_loader, valid_loader


    # Run the full training and validation workflow, save checkpoints,
    # and return the accumulated training and evaluation metrics.
    def start_training(self, return_logs=False):
        self._model = self.model.to(self._device)
        self._log.info(f"Using device: {self._device}")
        self._model.train()

        train_loader, valid_loader = self.create_dataset()


        # Initialize the best validation loss tracker.
        best_val_loss = float("inf")


        # Cache up to four validation batches for periodic sampling diagnostics.
        fixed_valid_batches = []
        for i, batch in enumerate(valid_loader):
            fixed_valid_batches.append(batch)
            if i >= 3:
                break


        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):


            self._log.info(
                f"========== Epoch {epoch + 1}/{self._exp_conf.num_epoch} "
            )

            # 1. Train the model for one epoch.
            train_epoch_log = self.train_epoch(
                train_loader=train_loader,
                device=self._device,
                return_logs=return_logs
            )

            # 2. Evaluate validation losses without molecular sampling.
            val_epoch_log = self.validate_epoch(
                valid_loader=valid_loader,
                device=self._device
            )

            train_loss = train_epoch_log["final_loss"]
            val_loss = val_epoch_log["final_loss"]

            self._log.info(
                f"[Epoch {epoch + 1}] "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}"
            )

            # Store epoch-level training and validation metrics.
            for k, v in train_epoch_log.items():
                self.metrics.setdefault(f"{k}/train_epoch", []).append(v)

            for k, v in val_epoch_log.items():
                self.metrics.setdefault(f"{k}/val_epoch", []).append(v)

            # 3. Save the checkpoint with the lowest validation loss.
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                if self._exp_conf.ckpt_dir is not None:
                    os.makedirs(self._exp_conf.ckpt_dir, exist_ok=True)
                    best_ckpt_path = os.path.join(
                        self._exp_conf.ckpt_dir,
                        "best_val_loss.pth"
                    )

                    utils.write_checkpoint(
                        best_ckpt_path,
                        {
                            k: v.detach().cpu()
                            for k, v in self.model.state_dict().items()
                        },
                        self._conf,
                        None,
                        epoch + 1,
                        self.trained_steps,
                        logger=self._log,
                        use_torch=True
                    )

                    self._log.info(
                        f"Best checkpoint saved: {best_ckpt_path}, "
                        f"best_val_loss={best_val_loss:.4f}"
                    )

            # 4. Save periodic checkpoints at the configured epoch frequency.
            if self._exp_conf.ckpt_dir is not None and ((epoch + 1) % self._exp_conf.ckpt_epoch_freq == 0):
                os.makedirs(self._exp_conf.ckpt_dir, exist_ok=True)

                ckpt_path = os.path.join(
                    self._exp_conf.ckpt_dir,
                    f"epoch_{epoch + 1}.pth"
                )

                utils.write_checkpoint(
                    ckpt_path,
                    {
                        k: v.detach().cpu()
                        for k, v in self.model.state_dict().items()
                    },
                    self._conf,
                    None,
                    epoch + 1,
                    self.trained_steps,
                    logger=self._log,
                    use_torch=True
                )

                self._log.info(f"Checkpoint saved: {ckpt_path}")


            # 5. Run periodic sampling diagnostics every 50 epochs.
            if (self._exp_conf.name == "experiment"):
                if (epoch + 1) % 50 == 0 and (epoch + 1) >= 50:
                    self._log.info(f"Running sampling check at epoch {epoch + 1}")

                    eval_dir = os.path.join(
                        self._exp_conf.eval_dir,
                        f"epoch_{epoch + 1}"
                    )
                    os.makedirs(eval_dir, exist_ok=True)

                    start_time = time.time()

                    # Run sampling diagnostics on the cached validation batches.
                    sampling_results = self.sample_and_analyze(
                        eval_dir,
                        fixed_valid_batches,
                        use_diagnose=True,
                        pred_bond_threshold=0.5,
                        pred_bond_max_new_bonds_per_atom=4,
                        pred_bond_min_dist=0.75,
                        pred_bond_max_dist=2.2,
                    )

                    self._log.info(f"Sampling result at epoch {epoch + 1}")
                    for k, v in sampling_results.items():
                        self._log.info(f"{k}: {v}")


                    # if best_fixed_batch_strict_topology < sampling_results["pred_bond_debug_strict_protac_topology_ratio"]:
                    #     best_fixed_batch_strict_topology = sampling_results["pred_bond_debug_strict_protac_topology_ratio"]
                    #
                    #     if self._exp_conf.ckpt_dir is not None:
                    #         os.makedirs(self._exp_conf.ckpt_dir, exist_ok=True)
                    #         best_fixed_batch_strict_topology_ckpt_path = os.path.join(
                    #             self._exp_conf.ckpt_dir,
                    #             "best_fixed_batch_strict_topology.pth"
                    #         )
                    #         utils.write_checkpoint(
                    #             best_fixed_batch_strict_topology_ckpt_path,
                    #             {
                    #                 k: v.detach().cpu()
                    #                 for k, v in self.model.state_dict().items()
                    #             },
                    #             self._conf,
                    #             None,
                    #             epoch + 1,
                    #             self.trained_steps,
                    #             logger=self._log,
                    #             use_torch=True
                    #         )
                    #
                    #         self._log.info(
                    #             f"Best fixed batch strict topology checkpoint saved: {best_fixed_batch_strict_topology_ckpt_path}, "
                    #             f"best_fixed_batch_strict_topology={best_fixed_batch_strict_topology:.4f}"
                    #         )

                    for metric_name, metric_value in sampling_results.items():
                        self.metrics.setdefault(f"{metric_name}/val_sample", []).append(metric_value)

                    eval_time = time.time() - start_time
                    self._log.info(f"Finished sampling check in {eval_time:.2f}s")

            # Update the number of completed training epochs.
            self.trained_epochs = epoch + 1

        self._log.info("Done")

        with open("./metrics.pkl", "wb") as f:
            pickle.dump(self.metrics, f)

        return self.metrics

    # Perform one training update: forward computation, backpropagation,
    # gradient clipping, and optimizer step. Return the loss and auxiliary metrics.
    def update_fn(self, data):
        # Clear accumulated gradients before backpropagation.
        self._optimizer.zero_grad()
        # loss, aux_data = self.loss_fn(data)
        loss, aux_data = self.loss_fn_flow(data)
        # Backpropagate the Flow Matching training loss.
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=float(self._exp_conf.grad_clip_norm)
        )

        # Update model parameters using the optimizer.
        self._optimizer.step()
        return loss, aux_data
    # Run one training epoch and aggregate per-step metrics.
    def train_epoch(self, train_loader, device, return_logs=False):

        self._model.train()

        log_losses = defaultdict(list)
        epoch_losses = defaultdict(list)

        log_time = time.time()

        for train_feats in train_loader:
            # Move training tensors to the computation device.
            train_feats = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in train_feats.items()
            }

            # update_fn: zero_grad -> loss -> backward -> optimizer.step
            loss, aux_data = self.update_fn(train_feats)


            for k, v in aux_data.items():
                value = float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
                log_losses[k].append(value)
                epoch_losses[k].append(value)

            self.trained_steps += 1

            # Log aggregated batch metrics at the configured step frequency.
            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:
                elapsed_time = time.time() - log_time
                log_time = time.time()

                step_per_sec = self._exp_conf.log_freq / max(elapsed_time, 1e-8)

                loss_log = " ".join([
                    f"{k}={np.mean(v):.4f}"
                    for k, v in log_losses.items()
                ])

                self._log.info(
                    f"[step {self.trained_steps}]: {loss_log}, "
                    f"steps/sec={step_per_sec:.5f}"
                )

                for metric_name, values in log_losses.items():
                    self.metrics.setdefault(f"{metric_name}/train_step", []).append(
                        float(np.mean(values))
                    )

                log_losses = defaultdict(list)

        # Compute mean training metrics over the entire epoch.
        epoch_log = {
            k: float(np.mean(v))
            for k, v in epoch_losses.items()
        }

        return epoch_log

    @torch.no_grad()
    def validate_epoch(self, valid_loader, device):
        self._model.eval()

        val_losses = defaultdict(list)

        for valid_feats in valid_loader:
            # Move validation tensors to the computation device.
            valid_feats = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in valid_feats.items()
            }

            # Compute validation losses without updating model parameters.
            # Call loss_fn_flow() directly instead of update_fn().
            loss, aux_data = self.loss_fn_flow(valid_feats)

            for k, v in aux_data.items():
                value = float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
                val_losses[k].append(value)

        val_log = {
            k: float(np.mean(v))
            for k, v in val_losses.items()
        }

        self._model.train()

        return val_log


    def count_true_linker_related_bonds(
            self,
            full_bond_types,
            atom_mask,
            anchor1_mask,
            anchor2_mask,
            linker_mask,
    ):
        """
        Count reference linker-related bonds for each molecule.

        Linker-linker bonds:
            Count all nonzero bond types, including aromatic bonds.

        Anchor-linker bonds:
            Count only single, double, and triple bonds.
            Aromatic bonds (type 4) are excluded.

        Each undirected bond is counted once using the upper triangle
        of the bond adjacency matrix.
        """

        def squeeze_bool(x):
            if x.dim() == 3:
                x = x.squeeze(-1)
            return x.bool()

        atom_mask = squeeze_bool(atom_mask)
        anchor1_mask = squeeze_bool(anchor1_mask)
        anchor2_mask = squeeze_bool(anchor2_mask)
        linker_mask = squeeze_bool(linker_mask)

        if full_bond_types.dim() == 4:
            if full_bond_types.size(-1) == 1:
                full_bond_types = full_bond_types.squeeze(-1)
            else:
                full_bond_types = full_bond_types.argmax(dim=-1)

        full_bond_types = full_bond_types.long()

        B, N = atom_mask.shape
        device = full_bond_types.device

        triu = torch.triu(
            torch.ones(N, N, device=device, dtype=torch.bool),
            diagonal=1,
        )[None, :, :]

        valid_pair = (
                atom_mask[:, :, None]
                & atom_mask[:, None, :]
                & triu
        )

        ll_pair = (
                linker_mask[:, :, None]
                & linker_mask[:, None, :]
        )

        a1_l_pair = (
                (anchor1_mask[:, :, None] & linker_mask[:, None, :])
                |
                (linker_mask[:, :, None] & anchor1_mask[:, None, :])
        )

        a2_l_pair = (
                (anchor2_mask[:, :, None] & linker_mask[:, None, :])
                |
                (linker_mask[:, :, None] & anchor2_mask[:, None, :])
        )

        true_ll = (
                ll_pair
                & valid_pair
                & (full_bond_types > 0)
        )

        true_a1_l = (
                a1_l_pair
                & valid_pair
                & (full_bond_types >= 1)
                & (full_bond_types <= 3)
        )

        true_a2_l = (
                a2_l_pair
                & valid_pair
                & (full_bond_types >= 1)
                & (full_bond_types <= 3)
        )

        true_anchor_linker = true_a1_l | true_a2_l
        true_total = true_ll | true_anchor_linker

        return {
            "true_total": true_total.sum(dim=(1, 2)).float(),
            "true_linker_linker": true_ll.sum(dim=(1, 2)).float(),
            "true_anchor_linker": true_anchor_linker.sum(dim=(1, 2)).float(),
            "true_anchor1_linker": true_a1_l.sum(dim=(1, 2)).float(),
            "true_anchor2_linker": true_a2_l.sum(dim=(1, 2)).float(),
        }


    def sample_and_analyze(
            self,
            eval_dir,
            dataloader,
            use_diagnose=False,
            pred_bond_threshold=0.5,
            pred_bond_max_new_bonds_per_atom=4,
            pred_bond_min_dist=0.75,
            pred_bond_max_dist=2.2,
    ):
        # Initialize topology diagnostics for bond-head-based reconstruction.
        pred_bond_connectivity_debug_counter = defaultdict(int)


        for b, data in tqdm(enumerate(dataloader), total=len(dataloader), desc='Sampling'):
            data = move_to_device(data, self._device)

            atom_mask = data['atom_mask']
            anchor1_mask = data['anchor1_mask']  # [B, N, 1], warhead / target ligand anchor
            anchor2_mask = data['anchor2_mask']  # [B, N, 1], E3 ligand anchor
            fragment1_mask = data['fragment1_mask']  # [B, N, 1], warhead / target ligand fragment
            fragment2_mask = data['fragment2_mask']  # [B, N, 1], E3 ligand fragment
            anchors = (anchor1_mask + anchor2_mask).clamp(max=1.0)
            fragment_mask = (fragment1_mask + fragment2_mask).clamp(max=1.0)

            fragment_atom_props_mask = (
                    data['fragment1_mask'].bool()
                    | data['fragment2_mask'].bool()
            )

            true_bond_counts_batch = self.count_true_linker_related_bonds(
                full_bond_types=data["full_bond_types"],
                atom_mask=data["atom_mask"],
                anchor1_mask=data["anchor1_mask"],
                anchor2_mask=data["anchor2_mask"],
                linker_mask=data["linker_mask"],
            )

            # Reconstruct reference molecules as RDKit Mol objects.
            true_molecules_batch = build_molecules(
                one_hot=data['one_hot'],
                x=data['positions'],
                node_mask=atom_mask,

                # Use complete reference bond types for molecule reconstruction.
                fragment_bond_types=data['full_bond_types'],
                use_template_fragment_bonds=True,

                # Use reference RDKit atom properties for the full molecule.
                formal_charges=data['formal_charges'],
                atom_is_aromatic=data['atom_is_aromatic'],
                num_explicit_hs=data['num_explicit_hs'],
                no_implicit=data['no_implicit'],

                # None applies reference atom properties to all fragment and linker atoms.
                atom_props_mask=None,

                use_pred_bonds=False,
            )

            # Generate multiple samples for each evaluation batch.
            for sample_idx in tqdm(range(self._exp_conf.n_stability_samples)):
                try:
                    # chain_batch, node_mask = self.sample_chain(data, keep_frames=self.FRAMES)
                    chain_batch, node_mask, pred_bond_probs, pred_bond_types = self.sample_chain_flow(
                        data,
                        keep_frames=self.FRAMES,
                    )
                except utils.FoundNaNException as e:
                    for idx in e.x_h_nan_idx:
                        smiles = data['name'][idx]
                        print(f'FoundNaNException: [xh], e={self.current_epoch}, b={b}, i={idx}: {smiles}')
                    for idx in e.only_x_nan_idx:
                        smiles = data['name'][idx]
                        print(f'FoundNaNException: [x ], e={self.current_epoch}, b={b}, i={idx}: {smiles}')
                    for idx in e.only_h_nan_idx:
                        smiles = data['name'][idx]
                        print(f'FoundNaNException: [ h], e={self.current_epoch}, b={b}, i={idx}: {smiles}')
                    continue

                # Get final molecules from chains – for computing metrics
                x, h = utils.split_features(
                    z=chain_batch[0],
                    n_dims=3,
                    num_classes=const.NUMBER_OF_ATOM_TYPES,
                )

                one_hot = h['categorical']


                # ------------------------------------------------
                # 1. Reconstruct molecules using bond-head predictions.
                #    Do not perform linker-component repair.
                # ------------------------------------------------
                pred_molecules_pred_bond_batch = build_molecules(
                    one_hot=one_hot,
                    x=x,
                    node_mask=node_mask,

                    anchors=anchors,
                    linker_mask=data["linker_mask"],

                    # Predicted bond probabilities and types from the bond head.
                    pred_bond_probs=pred_bond_probs,
                    pred_bond_types=pred_bond_types,
                    use_pred_bonds=True,
                    pred_bond_threshold=pred_bond_threshold,
                    pred_bond_max_new_bonds_per_atom=pred_bond_max_new_bonds_per_atom,
                    pred_bond_max_dist=pred_bond_max_dist,
                    pred_bond_min_dist=pred_bond_min_dist,


                    # Use reference fragment-fragment bonds in the fragment template.
                    fragment_bond_types=data["fragment_bond_types"],
                    use_template_fragment_bonds=True,

                    # Preserve reference RDKit atom properties for fragment atoms.
                    formal_charges=data["formal_charges"],
                    atom_is_aromatic=data["atom_is_aromatic"],
                    num_explicit_hs=data["num_explicit_hs"],
                    no_implicit=data["no_implicit"],
                    atom_props_mask=fragment_atom_props_mask,
                )

                for mol_i, (
                        pred_pred_bond,
                        true_mol,
                        # frag,
                ) in enumerate(zip(
                    pred_molecules_pred_bond_batch,
                    true_molecules_batch,
                )):
                    # Include samples only when the reference molecule passes RDKit sanitization.
                    if is_rdkit_fully_valid(true_mol):

                        if use_diagnose:
                            pred_bond_diag = diagnose_generated_connectivity(
                                mol=pred_pred_bond,
                                node_mask_i=node_mask[mol_i],
                                fragment1_mask_i=data["fragment1_mask"][mol_i],
                                fragment2_mask_i=data["fragment2_mask"][mol_i],
                                linker_mask_i=data["linker_mask"][mol_i],
                                anchor1_mask_i=data["anchor1_mask"][mol_i],
                                anchor2_mask_i=data["anchor2_mask"][mol_i],
                                name=f"pred_bond_b{b}_s{sample_idx}_i{mol_i}",
                                verbose=False,
                            )

                            update_connectivity_debug_counter(
                                pred_bond_connectivity_debug_counter,
                                pred_bond_diag,
                            )

        results = {}

        if use_diagnose:
            results = add_connectivity_debug_results(
                results=results,
                counter=pred_bond_connectivity_debug_counter,
                prefix="pred_bond_debug",
            )

        return results
    def bond_type_probs_to_pred_bonds(
            self,
            bond_type_probs,
            anchor1_mask,
            anchor2_mask,
            linker_mask,
    ):
        """
        Convert bond-type probabilities into bond existence probabilities
        and discrete bond-type predictions.

        Args:
            bond_type_probs: Tensor of shape [B, N, N, 5].
                0 = no bond
                1 = single
                2 = double
                3 = triple
                4 = aromatic

        Returns:
            pred_bond_probs: Bond existence probabilities [B, N, N].
            pred_bond_types: Predicted bond types [B, N, N].

        Bond constraints:
            General pairs:
                Single, double, triple, and aromatic are valid bond types.
            Anchor-linker pairs:
                Only single, double, and triple bonds are allowed.
                Aromatic bonds are excluded from allowed bond probability.

        Bond probabilities are symmetrized, and self-loops are removed.
        """

        # Symmetrize bond-type probabilities for undirected bonds.
        bond_type_probs = 0.5 * (
                bond_type_probs
                + bond_type_probs.transpose(1, 2)
        )

        anchor_mask = (
                anchor1_mask + anchor2_mask
        ).clamp(max=1.0)

        if anchor_mask.dim() == 3:
            anchor_mask_2d = anchor_mask.squeeze(-1).bool()
        else:
            anchor_mask_2d = anchor_mask.bool()

        if linker_mask.dim() == 3:
            linker_mask_2d = linker_mask.squeeze(-1).bool()
        else:
            linker_mask_2d = linker_mask.bool()

        anchor_linker_pair_mask = (
                (anchor_mask_2d[:, :, None] & linker_mask_2d[:, None, :])
                |
                (linker_mask_2d[:, :, None] & anchor_mask_2d[:, None, :])
        )

        # 1. Compute bond existence probabilities.
        # Default: all nonzero bond types contribute to bond existence.
        pred_bond_probs_default = 1.0 - bond_type_probs[..., 0]

        # Anchor-linker pairs: sum single, double, and triple probabilities.
        pred_bond_probs_anchor_linker = (
                bond_type_probs[..., 1]
                + bond_type_probs[..., 2]
                + bond_type_probs[..., 3]
        )

        pred_bond_probs = torch.where(
            anchor_linker_pair_mask,
            pred_bond_probs_anchor_linker,
            pred_bond_probs_default,
        )

        # 2. Predict bond types.
        # Default: select the highest-probability nonzero bond type.
        pred_bond_types_default = torch.argmax(
            bond_type_probs[..., 1:5],
            dim=-1,
        ) + 1

        # Anchor-linker pairs: select single, double, or triple; exclude aromatic.
        pred_bond_types_anchor_linker = torch.argmax(
            bond_type_probs[..., 1:4],
            dim=-1,
        ) + 1

        pred_bond_types = torch.where(
            anchor_linker_pair_mask,
            pred_bond_types_anchor_linker,
            pred_bond_types_default,
        )

        # Remove self-loops from predicted bond probabilities and types.
        B, N, _ = pred_bond_probs.shape
        eye = torch.eye(
            N,
            device=pred_bond_probs.device,
            dtype=torch.bool,
        )[None, :, :]

        pred_bond_probs = pred_bond_probs.masked_fill(eye, 0.0)
        pred_bond_types = pred_bond_types.masked_fill(eye, 0)

        return pred_bond_probs, pred_bond_types

    @torch.no_grad()
    def sample_chain_flow(
            self,
            data,
            keep_frames=None,
    ):
        """
        ELF-style / Rectified Flow ODE sampler.

        Start from Gaussian noise on linker atoms.
        Integrate from t = 0 to t = 0.95.
        Predict the clean linker state at each step and derive
        the velocity from the clean-state prediction.
        Discretize atom types only after the final sampling step.
        """

        # 1. Determine linker sizes and construct generation templates.
        linker_sizes = data['linker_mask'].sum(1).view(-1).int()

        template_data = create_templates_for_linker_generation(
            data,
            linker_sizes
        )

        x = template_data['positions']
        h = template_data['one_hot']
        node_mask = template_data['atom_mask']
        edge_mask = template_data['edge_mask']

        anchor1_mask = template_data['anchor1_mask']
        anchor2_mask = template_data['anchor2_mask']
        fragment1_mask = template_data['fragment1_mask']
        fragment2_mask = template_data['fragment2_mask']
        linker_mask = template_data['linker_mask']

        anchors = (anchor1_mask + anchor2_mask).clamp(max=1.0)
        fragment_mask = (fragment1_mask + fragment2_mask).clamp(max=1.0)

        B, N, _ = x.shape
        edge_mask = edge_mask.view(B, N, N)

        center_mask = get_center_of_mass_mask(
            self._exp_conf,
            anchors=anchors,
            fragment_mask=fragment_mask,
            node_mask=node_mask,
        )
        # 2. Center coordinates using the same configured COM mask as training.
        x = utils.remove_partial_mean_with_mask(
            x,
            node_mask,
            center_mask
        )

        # ------------------------------------------------
        # 3. normalize
        # ------------------------------------------------
        x, h = self.normalize(x, h)
        xh = torch.cat([x, h], dim=2)

        # 4. Initialize fixed fragments and Gaussian-noise linker states.
        z_noise = self.sample_combined_position_feature_noise(
            n_samples=B,
            n_nodes=N,
            mask=linker_mask
        )

        z = (
                xh * fragment_mask
                + z_noise * linker_mask
        )

        if keep_frames is None:
            keep_frames = self.FRAMES

        assert keep_frames <= self.T

        chain = torch.zeros(
            (keep_frames,) + z.size(),
            device=z.device
        )

        # 5. ODE sampling from t = 0 to t = 0.95.
        # Stop before t = 1 to avoid division by zero.
        t_end = 0.95
        ts = torch.linspace(0.0, t_end, self.T + 1, device=z.device)

        self.model.eval()


        for k in range(self.T):
            t_now = ts[k]
            t_next = ts[k + 1]
            dt = t_next - t_now

            t_batch = torch.full(
                (B, 1),
                float(t_now.item()),
                device=z.device
            )

            x_t = z[:, :, :3]
            h_t = z[:, :, 3:]

            h_pred, x_pred = self.model(
                h_t,
                x_t,
                anchor1_mask,
                anchor2_mask,
                fragment1_mask,
                fragment2_mask,
                linker_mask,
                node_mask,
                t_batch,
                edge_mask
            )


            xh_pred = torch.cat([x_pred, h_pred], dim=2)

            # velocity field
            v = (xh_pred - z) / (1.0 - t_now + 1e-6)

            z_next = z + dt * v

            # Keep fragment states fixed at every sampling step.
            z = (
                    xh * fragment_mask
                    + z_next * linker_mask
            )

            # Store intermediate sampling frames.
            write_index = keep_frames - 1 - int((k * keep_frames) / self.T)
            write_index = max(1, min(write_index, keep_frames - 1))
            chain[write_index] = self.unnormalize_z(z)

        x_t = z[:, :, :3]
        h_t = z[:, :, 3:]

        t_final = torch.full(
            (B, 1),
            float(t_end),
            device=z.device
        )

        h_pred, x_pred, bond_logits = self.model(
            h_t,
            x_t,
            anchor1_mask,
            anchor2_mask,
            fragment1_mask,
            fragment2_mask,
            linker_mask,
            node_mask,
            t_final,
            edge_mask,
            return_bond=True,
        )

        bond_type_probs = torch.softmax(
            bond_logits,
            dim=-1,
        )  # [B, N, N, 5]

        # Construct the anchor-linker pair mask.
        anchor_mask = (
                anchor1_mask + anchor2_mask
        ).clamp(max=1.0)

        if anchor_mask.dim() == 3:
            anchor_mask_2d = anchor_mask.squeeze(-1).bool()
        else:
            anchor_mask_2d = anchor_mask.bool()

        if linker_mask.dim() == 3:
            linker_mask_2d = linker_mask.squeeze(-1).bool()
        else:
            linker_mask_2d = linker_mask.bool()

        anchor_linker_pair_mask = (
                (anchor_mask_2d[:, :, None] & linker_mask_2d[:, None, :])
                |
                (linker_mask_2d[:, :, None] & anchor_mask_2d[:, None, :])
        )  # [B, N, N]

        # Compute predicted bond existence probabilities.
        # General pairs:
        #   P(any bond) = 1 - P(no bond)
        #
        # Anchor-linker pairs:
        #   P(allowed bond) = P(single) + P(double) + P(triple)
        #   Exclude aromatic bonds from the allowed bond probability.
        pred_bond_probs_default = 1.0 - bond_type_probs[..., 0]

        pred_bond_probs_anchor_linker = (
                bond_type_probs[..., 1]
                + bond_type_probs[..., 2]
                + bond_type_probs[..., 3]
        )

        pred_bond_probs = torch.where(
            anchor_linker_pair_mask,
            pred_bond_probs_anchor_linker,
            pred_bond_probs_default,
        )

        # Select predicted bond types.
        # General pairs:
        #   Choose the highest-scoring type among single, double,
        #   triple, and aromatic.
        #
        # Anchor-linker pairs:
        #   Choose the highest-scoring type among single, double,
        #   and triple.
        bond_type_scores = bond_type_probs[..., 1:5].clone()  # [B, N, N, 4]

        pred_bond_types_default = torch.argmax(
            bond_type_scores,
            dim=-1,
        ) + 1

        bond_type_scores_no_aromatic = bond_type_scores[..., :3].clone()

        pred_bond_types_anchor_linker = torch.argmax(
            bond_type_scores_no_aromatic,
            dim=-1,
        ) + 1

        pred_bond_types = torch.where(
            anchor_linker_pair_mask,
            pred_bond_types_anchor_linker,
            pred_bond_types_default,
        )


        last_xh_pred = torch.cat([x_pred, h_pred], dim=2)

        # ------------------------------------------------
        # 6. final clean state
        # Use the final model prediction as the clean linker state.
        # ------------------------------------------------
        if last_xh_pred is None:
            raise RuntimeError("Flow sampling failed: last_xh_pred is None.")

        # Use the final clean-state prediction for the linker.
        xh_final = (xh * fragment_mask + last_xh_pred * linker_mask)

        x_final_norm = xh_final[:, :, :3]
        h_final_norm = xh_final[:, :, 3:]

        # ------------------------------------------------
        # 7. unnormalize
        # ------------------------------------------------
        x_final, h_final_cont = self.unnormalize(
            x_final_norm,
            h_final_norm
        )

        # 8. Discretize atom types only at the final sampling step.
        #    Restrict linker atom types to C, N, O, S, and F.
        allowed_linker_atom_ids = get_atom_type_ids_by_symbols(
            ("C", "N", "O", "S", "F")
        )

        type_allowed_mask = torch.zeros(
            const.NUMBER_OF_ATOM_TYPES,
            device=h_final_cont.device,
            dtype=torch.bool,
        )
        type_allowed_mask[allowed_linker_atom_ids] = True

        if linker_mask.dim() == 3:
            linker_mask_2d = linker_mask.squeeze(-1).bool()
        else:
            linker_mask_2d = linker_mask.bool()

        h_final_cont_masked = h_final_cont.clone()
        h_final_cont_masked[..., ~type_allowed_mask] = -1e9

        atom_type_default = torch.argmax(
            h_final_cont,
            dim=2,
        )

        atom_type_linker_restricted = torch.argmax(
            h_final_cont_masked,
            dim=2,
        )

        atom_type_final = torch.where(
            linker_mask_2d,
            atom_type_linker_restricted,
            atom_type_default,
        )

        h_final = F.one_hot(
            atom_type_final,
            const.NUMBER_OF_ATOM_TYPES,
        ).float()

        h_final = h_final * node_mask

        chain[0] = torch.cat([x_final, h_final], dim=2)

        self.model.train()

        return chain, node_mask, pred_bond_probs, pred_bond_types


    @staticmethod
    def normalize(x, h):
        new_x = x / 5
        new_h = h.float() / 1
        return new_x, new_h

    @staticmethod
    def unnormalize_z(z):
        assert z.size(2) == 3 + const.NUMBER_OF_ATOM_TYPES
        x, h = z[:, :, :3], z[:, :, 3:]
        x, h = Experiment.unnormalize(x, h)
        return torch.cat([x, h], dim=2)

    @staticmethod
    def unnormalize(x, h):
        new_x = x * 5
        new_h = h * 1
        return new_x, new_h


    def loss_fn_flow(self, batch):
        """
        ELF-style / Rectified Flow Matching loss.

        Time direction:
            t = 0: Gaussian noise
            t = 1: clean linker state

        Model outputs:
            h_pred, x_pred = predicted clean linker state
        """

        x = batch['positions']  # [B, N, 3]
        h = batch['one_hot']  # [B, N, atom_types]

        node_mask = batch['atom_mask']  # [B, N, 1]
        edge_mask = batch['edge_mask']

        anchor1_mask = batch['anchor1_mask']
        anchor2_mask = batch['anchor2_mask']
        fragment1_mask = batch['fragment1_mask']
        fragment2_mask = batch['fragment2_mask']
        linker_mask = batch['linker_mask']

        anchors = (anchor1_mask + anchor2_mask).clamp(max=1.0)
        fragment_mask = (fragment1_mask + fragment2_mask).clamp(max=1.0)

        B, N, _ = x.shape
        edge_mask = edge_mask.view(B, N, N)

        center_mask = get_center_of_mass_mask(
            self._exp_conf,
            anchors=anchors,
            fragment_mask=fragment_mask,
            node_mask=node_mask,
        )

        # 1. Center coordinates using the configured center-of-mass mask.
        x = utils.remove_partial_mean_with_mask(
            x,
            node_mask,
            center_mask
        )

        utils.assert_partial_mean_zero_with_mask(
            x,
            node_mask,
            center_mask
        )

        # ------------------------------------------------
        # 2. normalize
        # ------------------------------------------------
        x, h = self.normalize(x, h)
        xh = torch.cat([x, h], dim=2)  # [B, N, 3 + atom_types]

        # ------------------------------------------------
        # 3. sample t
        # ------------------------------------------------

        t_min = 1e-4
        t_max = 0.95

        high_t_prob = 0.30
        high_t_min = 0.50
        high_t_max = 0.95

        mix = torch.rand(B, 1, device=x.device)

        # Sample 70% of time values uniformly from the full interval.
        u_uniform = torch.rand(B, 1, device=x.device)
        t_uniform = t_min + (t_max - t_min) * u_uniform

        # Sample 30% of time values uniformly from the high-t interval.
        # No square-root transformation is applied.
        u_high = torch.rand(B, 1, device=x.device)
        t_high = high_t_min + (high_t_max - high_t_min) * u_high

        t = torch.where(mix < high_t_prob, t_high, t_uniform)
        t = t.clamp(min=t_min, max=t_max)

        t_expand = t[:, None, :]

        # 4. Sample Gaussian noise only for linker atoms.
        eps = self.sample_combined_position_feature_noise(
            n_samples=B,
            n_nodes=N,
            mask=linker_mask
        )

        # ------------------------------------------------
        # 5. Rectified Flow interpolation
        # z_t = t * xh + (1 - t) * eps
        # Keep fragments fixed and interpolate only linker atoms.
        # ------------------------------------------------
        z_t_linker = t_expand * xh + (1.0 - t_expand) * eps

        z_t = (
                xh * fragment_mask
                + z_t_linker * linker_mask
        )

        x_t = z_t[:, :, :3]
        h_t = z_t[:, :, 3:]

        # 6. Predict the clean linker state with the model.
        h_pred, x_pred, bond_logits = self.model(
            h_t,
            x_t,
            anchor1_mask,
            anchor2_mask,
            fragment1_mask,
            fragment2_mask,
            linker_mask,
            node_mask,
            t,
            edge_mask,
            return_bond=True,
        )

        xh_pred = torch.cat([x_pred, h_pred], dim=2)

        # ------------------------------------------------
        # 7. Flow Matching velocity loss
        # v_target = xh - eps
        # v_pred = (xh_pred - z_t) / (1 - t)
        # ------------------------------------------------
        v_target = xh - eps

        denom = torch.clamp(
            1.0 - t_expand,
            min=float(self._exp_conf.flow_denom_min)
        )

        v_pred = (xh_pred - z_t) / denom

        error = self.sum_except_batch(
            ((v_pred - v_target) ** 2) * linker_mask
        )

        normalization = (
                (3 + const.NUMBER_OF_ATOM_TYPES)
                * self.numbers_of_nodes(linker_mask)
        )

        flow_loss = error / (normalization + 1e-8)
        flow_loss = flow_loss.mean()

        # 8. Geometry auxiliary loss
        # x_pred contains the predicted clean-state coordinates.
        linker_mask_3d = linker_mask
        if linker_mask_3d.dim() == 2:
            linker_mask_3d = linker_mask_3d.unsqueeze(-1)

        # Replace only linker coordinates; keep fragment coordinates fixed.
        x_pred_for_geom = (
                x_pred * linker_mask_3d
                + x * (1.0 - linker_mask_3d)
        )

        x0_loss = position_loss(
            x_pred,
            x,
            linker_mask
        )


        bond_type_labels = batch["full_bond_types"].to(
            self._device,
            non_blocking=True,
        ).long()

        bond_mask = batch["bond_mask"].to(
            self._device,
            non_blocking=True,
        )

        use_no_loss = bool(
            getattr(self._exp_conf, "use_no_loss", False)
        )
        if not use_no_loss:
            linker_bond_length_loss = true_linker_rebuild_bond_length_loss(
                x_pred=x_pred_for_geom,
                x_true=x,
                linker_mask=linker_mask,
                bond_labels=bond_type_labels,
                anchor1_mask=anchor1_mask,
                anchor2_mask=anchor2_mask,
                bond_mask=bond_mask,
                beta=0.02,
            )

        bond_loss, bond_aux = explicit_linker_bond_type_loss(
            bond_type_logits=bond_logits,
            bond_type_labels=bond_type_labels,
            bond_mask=bond_mask,
            anchor1_mask=anchor1_mask,
            anchor2_mask=anchor2_mask,
            linker_mask=linker_mask,
            anchor1_weight=1.5,
            anchor2_weight=1.5,
            linker_weight=8.0,
            max_pos_weight=50.0,
            type_weight=1.0,
            linker_positive_weight=float(self._exp_conf.linker_positive_weight),
        )

        if not use_no_loss:
            x_pred_real = x_pred_for_geom * 5.0
            x_true_real = x * 5.0

            linker_isolated_loss = rebuild_consistent_linker_no_isolated_loss(
                x_pred_A=x_pred_real,
                x_true_A=x_true_real,
                bond_type_logits=bond_logits,
                bond_type_labels=bond_type_labels,
                anchors=anchors,
                linker_mask=linker_mask,
                distance_margin_A=0.10,
                sharpness=40.0,
            )

        # ------------------------------------------------
        # 9. total loss
        # ------------------------------------------------

        aux_scale = self.get_aux_loss_scale()

        x0_weight = float(self._exp_conf.x0_weight)

        topology_weight = float(self._exp_conf.topology_weight) * aux_scale

        topology_bond_coef = float(self._exp_conf.topology_bond_coef)
        if not use_no_loss:
            topology_bond_length_coef = float(self._exp_conf.topology_bond_length_coef)
            topology_isolated_coef = float(self._exp_conf.topology_isolated_coef)
            topology_loss = (
                    topology_bond_coef * bond_loss
                    + topology_bond_length_coef * linker_bond_length_loss
                    + topology_isolated_coef * linker_isolated_loss
            )
        else:
            topology_loss = topology_bond_coef * bond_loss

        final_loss = (
                flow_loss
                + x0_weight * x0_loss
                + topology_weight * topology_loss
        )
        if not use_no_loss:
            aux_data = {
                "final_loss": final_loss.detach(),
                "flow_loss": flow_loss.detach(),
                "weighted_x0_loss": (x0_weight * x0_loss).detach(),
                "weighted_topology_loss": (topology_weight * topology_loss).detach(),
                "topology_loss": topology_loss.detach(),
                "weighted_linker_bond_length_loss": (
                        topology_weight * topology_bond_length_coef * linker_bond_length_loss
                ).detach(),
                "weighted_linker_isolated_loss": (
                        topology_weight * topology_isolated_coef * linker_isolated_loss
                ).detach(),
                # Total weighted typed-bond topology loss.
                "weighted_bond_topology_loss": (
                        topology_weight * topology_bond_coef * bond_loss
                ).detach(),
                # Weighted typed-bond losses for the three supervised pair categories.
                "weighted_bond_anchor1_linker_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_anchor1_linker_loss"]
                ).detach(),
                "weighted_bond_anchor2_linker_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_anchor2_linker_loss"]
                ).detach(),
                "weighted_bond_linker_linker_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_linker_linker_loss"]
                ).detach(),
                # existence loss: no bond vs any bond
                "weighted_bond_anchor1_exist_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_anchor1_exist_loss"]
                ).detach(),
                "weighted_bond_anchor2_exist_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_anchor2_exist_loss"]
                ).detach(),
                "weighted_bond_linker_linker_exist_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_linker_linker_exist_loss"]
                ).detach(),
                # type loss: single / double / triple / aromatic
                "weighted_bond_anchor1_type_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_anchor1_type_loss"]
                ).detach(),
                "weighted_bond_anchor2_type_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_anchor2_type_loss"]
                ).detach(),
                "weighted_bond_linker_linker_type_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_linker_linker_type_loss"]
                ).detach(),
                # Indicators of whether each bond category has supervised pairs.
                "bond_anchor1_linker_has_pair": bond_aux["bond_anchor1_linker_has_pair"],
                "bond_anchor2_linker_has_pair": bond_aux["bond_anchor2_linker_has_pair"],
                "bond_linker_linker_has_pair": bond_aux["bond_linker_linker_has_pair"],
                "aux_scale": torch.tensor(
                    aux_scale,
                    device=self._device,
                    dtype=final_loss.dtype,
                ),
            }
        else:
            aux_data = {
                "final_loss": final_loss.detach(),
                "flow_loss": flow_loss.detach(),
                "weighted_x0_loss": (x0_weight * x0_loss).detach(),
                "weighted_topology_loss": (topology_weight * topology_loss).detach(),
                "topology_loss": topology_loss.detach(),
                # Total weighted typed-bond topology loss.
                "weighted_bond_topology_loss": (
                        topology_weight * topology_bond_coef * bond_loss
                ).detach(),
                # Weighted typed-bond losses for the three supervised pair categories.
                "weighted_bond_anchor1_linker_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_anchor1_linker_loss"]
                ).detach(),
                "weighted_bond_anchor2_linker_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_anchor2_linker_loss"]
                ).detach(),
                "weighted_bond_linker_linker_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_linker_linker_loss"]
                ).detach(),
                # existence loss: no bond vs any bond
                "weighted_bond_anchor1_exist_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_anchor1_exist_loss"]
                ).detach(),
                "weighted_bond_anchor2_exist_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_anchor2_exist_loss"]
                ).detach(),
                "weighted_bond_linker_linker_exist_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_linker_linker_exist_loss"]
                ).detach(),
                # type loss: single / double / triple / aromatic
                "weighted_bond_anchor1_type_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_anchor1_type_loss"]
                ).detach(),
                "weighted_bond_anchor2_type_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_anchor2_type_loss"]
                ).detach(),
                "weighted_bond_linker_linker_type_loss": (
                        topology_weight
                        * topology_bond_coef
                        * bond_aux["bond_linker_linker_type_loss"]
                ).detach(),
                # Indicators of whether each bond category has supervised pairs.
                "bond_anchor1_linker_has_pair": bond_aux["bond_anchor1_linker_has_pair"],
                "bond_anchor2_linker_has_pair": bond_aux["bond_anchor2_linker_has_pair"],
                "bond_linker_linker_has_pair": bond_aux["bond_linker_linker_has_pair"],
                "aux_scale": torch.tensor(
                    aux_scale,
                    device=self._device,
                    dtype=final_loss.dtype,
                ),
            }

        return final_loss, aux_data

    @staticmethod
    def sum_except_batch(x):
        return x.view(x.size(0), -1).sum(-1)

    @staticmethod
    def numbers_of_nodes(mask):
        return torch.sum(mask.squeeze(2), dim=1)



    @torch.no_grad()
    def evaluate_best_sampling_on_full_val(
            self,
            ckpt_path,
            save_dir=None,
            valid_loader=None,
    ):
        """
        Load a model checkpoint, such as best_val_loss.pth, and evaluate
        generated samples using the provided validation data loader.

        This evaluation measures sampling quality rather than
        the validation training loss.
        """

        self._model = self.model.to(self._device)
        self._model.eval()

        self._log.info(f"Loading best checkpoint from: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self._device)

        # Support different checkpoint serialization formats.
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state" in checkpoint:
                state_dict = checkpoint["model_state"]
            else:
                # Handle checkpoints that directly contain a state dictionary.
                possible_state_dict = {
                    k: v for k, v in checkpoint.items()
                    if torch.is_tensor(v)
                }
                if len(possible_state_dict) > 0:
                    state_dict = checkpoint
                else:
                    raise ValueError(
                        f"Cannot find model weights in checkpoint. "
                        f"Checkpoint keys: {list(checkpoint.keys())}"
                    )
        else:
            state_dict = checkpoint

        # Remove the optional "module." prefix from checkpoint parameter names.
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()

        eval_seed = int(self._exp_conf.seed)
        seed_everything(eval_seed)
        self._log.info(f"Using sampling seed: {eval_seed}")

        # Optional recreation of data loaders (currently disabled).
        # _, valid_loader = self.create_dataset()

        if save_dir is None:
            save_dir = os.path.join(
                self._exp_conf.eval_dir,
                "best_val_full_sampling"
            )

        os.makedirs(save_dir, exist_ok=True)

        self._log.info("Start sampling on the full validation set...")
        start_time = time.time()

        sampling_results = self.sample_and_analyze(
            eval_dir=save_dir,
            dataloader=valid_loader,
            use_diagnose=True,
            pred_bond_threshold=0.5,
            pred_bond_max_new_bonds_per_atom=4,
            pred_bond_min_dist=0.75,
            pred_bond_max_dist=2.2,
        )

        eval_time = time.time() - start_time

        self._log.info("========== Full Validation Sampling Results ==========")
        for k, v in sampling_results.items():
            self._log.info(f"{k}: {v}")

        self._log.info(f"Finished full validation sampling in {eval_time:.2f}s")

        # Save sampling evaluation results.
        result_pkl = os.path.join(save_dir, "best_val_sampling_metrics.pkl")
        with open(result_pkl, "wb") as f:
            pickle.dump(sampling_results, f)

        result_txt = os.path.join(save_dir, "best_val_sampling_metrics.txt")
        with open(result_txt, "w") as f:
            for k, v in sampling_results.items():
                f.write(f"{k}: {v}\n")

        self._log.info(f"Saved sampling metrics to: {result_pkl}")
        self._log.info(f"Saved sampling metrics txt to: {result_txt}")

        return sampling_results


# Hydra loads ../configs/base.yaml as an OmegaConf DictConfig and passes it to run().
@hydra.main(version_base=None, config_path="../configs", config_name="base")
def run(conf: DictConfig) -> None:

    os.environ["WANDB_START_METHOD"] = "thread"
    OmegaConf.set_struct(conf, False)
    exp = Experiment(conf=conf)
    exp.start_training(return_logs=True)

if __name__ == '__main__':

    run()