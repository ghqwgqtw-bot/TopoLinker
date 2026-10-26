import numpy as np
import torch
from rdkit import Chem, Geometry

from src import const

def build_template_fragment_molecule(
        positions,
        atom_types,
        fragment_bond_types,
        formal_charges=None,
        atom_is_aromatic=None,
        num_explicit_hs=None,
        no_implicit=None,
        atom_props_mask=None,
):
    """
    Build an initial molecule from atom identities, coordinates and
    provided bond types.

    Retain all atoms and add only the bonds supplied in the template;
    do not infer linker-related bonds from interatomic distances.
    """
    idx2atom = const.IDX2ATOM

    n_atoms = len(atom_types)

    if atom_props_mask is not None:
        atom_props_mask = atom_props_mask.detach().cpu()

        if atom_props_mask.dim() == 2:
            atom_props_mask = atom_props_mask.squeeze(-1)

        atom_props_mask = atom_props_mask.bool()

    mol = Chem.RWMol()

    for k, atom in enumerate(atom_types):
        a = Chem.Atom(idx2atom[int(atom.item())])

        use_props = True
        if atom_props_mask is not None:
            use_props = bool(atom_props_mask[k].item())

        if use_props:
            if formal_charges is not None:
                a.SetFormalCharge(int(formal_charges[k].item()))

            if atom_is_aromatic is not None:
                a.SetIsAromatic(bool(atom_is_aromatic[k].item()))

            if num_explicit_hs is not None:
                a.SetNumExplicitHs(int(num_explicit_hs[k].item()))

            if no_implicit is not None:
                a.SetNoImplicit(bool(no_implicit[k].item()))

        mol.AddAtom(a)

    mol = add_bonds_from_type_matrix(
        rwmol=mol,
        bond_type_matrix=fragment_bond_types,
    )

    mol = mol.GetMol()

    mol.AddConformer(
        create_conformer(
            positions.detach().cpu().numpy().astype(np.float64)
        )
    )

    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
    except Exception as e:
        print(f"[TEMPLATE_PARTIAL_SANITIZE_FAIL] {repr(e)}")

    return mol

def add_bonds_from_type_matrix(
        rwmol,
        bond_type_matrix,
):
    """
    Add bonds specified by an [N, N] bond-type matrix.

    Bond labels: 0=no bond, 1=single, 2=double,
    3=triple, 4=aromatic.

    For aromatic bonds, set the aromaticity flags on both
    endpoint atoms and the bond.
    """
    if bond_type_matrix is None:
        return rwmol

    bond_type_matrix = bond_type_matrix.detach().cpu()

    if bond_type_matrix.dim() == 3:
        bond_type_matrix = bond_type_matrix.squeeze(0)

    n = bond_type_matrix.shape[0]

    for i in range(n):
        for j in range(i):
            bond_idx = int(bond_type_matrix[i, j].item())

            if bond_idx <= 0:
                continue

            if bond_idx >= len(const.BOND_DICT):
                continue

            i = int(i)
            j = int(j)

            if rwmol.GetBondBetweenAtoms(i, j) is not None:
                continue

            bond_type = const.BOND_DICT[bond_idx]

            if bond_type is None:
                continue

            atom_i = rwmol.GetAtomWithIdx(i)
            atom_j = rwmol.GetAtomWithIdx(j)

            old_i_aromatic = atom_i.GetIsAromatic()
            old_j_aromatic = atom_j.GetIsAromatic()

            try:
                if bond_idx == 4:
                    atom_i.SetIsAromatic(True)
                    atom_j.SetIsAromatic(True)

                rwmol.AddBond(
                    i,
                    j,
                    bond_type,
                )

                if bond_idx == 4:
                    bond = rwmol.GetBondBetweenAtoms(i, j)
                    if bond is not None:
                        bond.SetIsAromatic(True)

            except Exception:
                bond = rwmol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    rwmol.RemoveBond(i, j)

                atom_i = rwmol.GetAtomWithIdx(i)
                atom_j = rwmol.GetAtomWithIdx(j)
                atom_i.SetIsAromatic(old_i_aromatic)
                atom_j.SetIsAromatic(old_j_aromatic)

                continue

    return rwmol
def add_predicted_bonds_to_mol(
        mol,
        pred_bond_probs,
        atom_types,
        positions=None,
        anchors=None,
        linker_mask=None,
        pred_bond_types=None,
        threshold=0.5,
        max_new_bonds_per_atom=4,
        only_linker_related=True,
        min_dist=0.75,
        max_dist=2.2,
        final_sanitize=True,
):
    """
    Add predicted linker-related bonds to an RDKit molecule.

    pred_bond_probs: [N, N] bond-existence probabilities, typically
        1 - P(no bond).
    pred_bond_types: optional [N, N] bond-type indices
        (1=single, 2=double, 3=triple, 4=aromatic).
        If absent, all accepted predicted bonds are treated as single.

    Only linker-linker and designated anchor-linker bonds are added.
    Fragment-fragment and non-anchor fragment-linker bonds are excluded.
    Direct fragment1-fragment2 bonds are not added.

    Candidate bonds are filtered by probability, geometry, element
    restrictions, valence, and RDKit sanitization.
    """

    if pred_bond_probs is None:
        return mol

    idx2atom = const.IDX2ATOM

    pred_bond_probs = pred_bond_probs.detach().cpu()

    if pred_bond_probs.dim() == 3:
        pred_bond_probs = pred_bond_probs.squeeze(0)

    if pred_bond_types is not None:
        pred_bond_types = pred_bond_types.detach().cpu()

        if pred_bond_types.dim() == 3:
            pred_bond_types = pred_bond_types.squeeze(0)

    n = pred_bond_probs.shape[0]

    if linker_mask is not None:
        if linker_mask.dim() == 2:
            linker_mask = linker_mask.squeeze(-1)
        linker_mask = linker_mask.detach().cpu().bool()
    else:
        linker_mask = torch.ones(n, dtype=torch.bool)

    if anchors is not None:
        if anchors.dim() == 2:
            anchors = anchors.squeeze(-1)
        anchors = anchors.detach().cpu().bool()
    else:
        anchors = torch.zeros(n, dtype=torch.bool)

    allowed_linker_atoms = {"C", "N", "O", "S", "F"}
    candidates = []

    for i in range(n):
        for j in range(i):
            prob = float(pred_bond_probs[i, j].item())

            if prob <= threshold:
                continue

            i_is_linker = bool(linker_mask[i].item())
            j_is_linker = bool(linker_mask[j].item())

            i_is_anchor = bool(anchors[i].item())
            j_is_anchor = bool(anchors[j].item())

            is_linker_linker = i_is_linker and j_is_linker

            is_anchor_linker = (
                    (i_is_anchor and j_is_linker)
                    or
                    (j_is_anchor and i_is_linker)
            )

            if only_linker_related:
                if not (is_linker_linker or is_anchor_linker):
                    continue

            atom_i_symbol = idx2atom[int(atom_types[i].item())]
            atom_j_symbol = idx2atom[int(atom_types[j].item())]

            # Both atoms in linker-linker bonds must be allowed linker elements.
            if is_linker_linker:
                if atom_i_symbol not in allowed_linker_atoms:
                    continue
                if atom_j_symbol not in allowed_linker_atoms:
                    continue

                has_f_linker_atom = (
                        atom_i_symbol == "F"
                        or atom_j_symbol == "F"
                )

            # For anchor-linker bonds, restrict only the generated linker atom
            # to the allowed elements; the fixed anchor has no such restriction.
            elif is_anchor_linker:
                if i_is_linker:
                    linker_atom_symbol = atom_i_symbol
                else:
                    linker_atom_symbol = atom_j_symbol

                if linker_atom_symbol not in allowed_linker_atoms:
                    continue

                has_f_linker_atom = linker_atom_symbol == "F"

            else:
                continue

            # Fluorine may only appear in linker-linker C-F single bonds.
            if has_f_linker_atom and is_anchor_linker:
                continue
            if has_f_linker_atom and is_linker_linker:
                if atom_i_symbol == "F" and atom_j_symbol == "F":
                    continue

                if atom_i_symbol == "F":
                    f_neighbor_symbol = atom_j_symbol
                else:
                    f_neighbor_symbol = atom_i_symbol

                if f_neighbor_symbol != "C":
                    continue

            dist = None

            if positions is not None:
                dist = float(
                    torch.norm(
                        positions[i] - positions[j],
                        p=2,
                    ).item()
                )

                if min_dist is not None and dist < min_dist:
                    continue

                if max_dist is not None and dist > max_dist:
                    continue

            # --------------------------------------------
            # typed bond type
            # --------------------------------------------
            if pred_bond_types is None:
                # Fallback to single bonds if bond types are unavailable.
                bond_type_idx = 1
            else:
                bond_type_idx = int(pred_bond_types[i, j].item())

            # ------------------------------------------------
            # keep predicted typed bond:
            # if predicted type is aromatic, add aromatic bond
            # ------------------------------------------------
            bond_type_idx = normalize_generated_bond_type_idx(
                bond_type_idx,
                is_anchor_linker,
            )

            if bond_type_idx <= 0:
                continue

            # Fluorine is restricted to single bonds.
            if has_f_linker_atom and bond_type_idx != 1:
                continue

            if bond_type_idx >= len(const.BOND_DICT):
                continue

            candidates.append(
                {
                    "i": i,
                    "j": j,
                    "prob": prob,
                    "dist": dist,
                    "bond_type_idx": bond_type_idx,
                    "is_anchor_linker": is_anchor_linker,
                    "is_linker_linker": is_linker_linker,
                }
            )

    candidates = sorted(
        candidates,
        key=lambda x: (
            -x["prob"],
            999.0 if x["dist"] is None else x["dist"],
        ),
    )

    rwmol = Chem.RWMol(mol)

    new_bond_count = torch.zeros(
        n,
        dtype=torch.long,
    )

    for cand in candidates:
        i = int(cand["i"])
        j = int(cand["j"])

        if rwmol.GetBondBetweenAtoms(i, j) is not None:
            continue

        if new_bond_count[i] >= max_new_bonds_per_atom:
            continue

        if new_bond_count[j] >= max_new_bonds_per_atom:
            continue

        bond_type_idx = int(cand["bond_type_idx"])
        bond_type = const.BOND_DICT[bond_type_idx]

        if bond_type is None:
            continue

        bond_order = bond_idx_to_valence_order(bond_type_idx)

        if not atom_can_accept_bond(rwmol, i, bond_order):
            continue

        if not atom_can_accept_bond(rwmol, j, bond_order):
            continue

        success, rwmol = try_add_typed_bond_with_sanitize(
            rwmol=rwmol,
            i=i,
            j=j,
            bond_type_idx=bond_type_idx,
            editable_atom_mask=linker_mask,
        )

        if success:
            new_bond_count[i] += 1
            new_bond_count[j] += 1
    # Infer linker atom properties from the resulting predicted bond graph.
    rwmol = infer_generated_atom_props_from_current_graph_inplace(
        rwmol=rwmol,
        editable_atom_mask=linker_mask,
    )

    final_mol = rwmol.GetMol()

    if final_sanitize:
        final_mol = final_sanitize_or_keep(
            final_mol,
            tag="FINAL_PRED_BOND_SANITIZE_FAIL",
        )
    else:
        # Update properties without the final kekulization check.
        try_sanitize_mol_no_kekulize(final_mol)

    return final_mol

def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

def normalize_generated_bond_type_idx(
        bond_type_idx,
        is_anchor_linker,
):
    # Preserve aromatic linker-linker predictions, but reject aromatic
    # anchor-linker predictions instead of converting them to single bonds.
    bond_type_idx = int(bond_type_idx)

    if bond_type_idx <= 0 or (is_anchor_linker and bond_type_idx ==4):
        return 0

    return bond_type_idx
def bond_idx_to_valence_order(bond_idx):
    bond_idx = int(bond_idx)

    if bond_idx == 1:
        return 1
    if bond_idx == 2:
        return 2
    if bond_idx == 3:
        return 3
    if bond_idx == 4:
        # Approximate aromatic valence as one; RDKit validates later.
        return 1

    return 1

def _to_1d_bool_mask(mask, n, default_value=True):
    if mask is None:
        return torch.full((n,), default_value, dtype=torch.bool)

    mask = mask.detach().cpu()

    if mask.dim() == 2:
        mask = mask.squeeze(-1)

    return mask.bool()


def infer_generated_atom_props_from_current_graph_inplace(
        rwmol,
        editable_atom_mask=None,
):
    """
    Infer editable-atom properties from the current RDKit bond graph.

    Set formal charges and aromaticity flags while using implicit
    hydrogens (num_explicit_hs=0 and no_implicit=False).
    Non-editable atoms retain their existing properties.
    """

    n = rwmol.GetNumAtoms()
    editable_atom_mask = _to_1d_bool_mask(
        editable_atom_mask,
        n=n,
        default_value=True,
    )

    def is_editable(idx):
        return bool(editable_atom_mask[int(idx)].item())

    def bond_sigma_order(bond):
        bt = bond.GetBondType()

        if bt == Chem.BondType.SINGLE:
            return 1
        if bt == Chem.BondType.DOUBLE:
            return 2
        if bt == Chem.BondType.TRIPLE:
            return 3
        if bt == Chem.BondType.AROMATIC or bond.GetIsAromatic():
            return 1

        return 1

    def atom_has_aromatic_bond(atom):
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.AROMATIC or bond.GetIsAromatic():
                return True
        return False

    # 0. Reset properties only on editable atoms.
    for atom in rwmol.GetAtoms():
        idx = atom.GetIdx()

        if not is_editable(idx):
            continue

        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)

        # Allow RDKit to assign implicit hydrogens.
        atom.SetNoImplicit(False)

        if atom_has_aromatic_bond(atom):
            atom.SetIsAromatic(True)
        else:
            atom.SetIsAromatic(False)

    try:
        rwmol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    # 1. Assign nitro charges: N(=O)-O -> [N+](=O)[O-].
    nitro_n = set()
    nitro_single_o = set()

    for atom in rwmol.GetAtoms():
        if atom.GetSymbol() != "N":
            continue

        n_idx = atom.GetIdx()

        if not is_editable(n_idx):
            continue

        o_double = []
        o_single = []

        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            other_idx = other.GetIdx()

            if other.GetSymbol() != "O":
                continue

            if bond.GetBondType() == Chem.BondType.DOUBLE:
                o_double.append(other_idx)

            elif bond.GetBondType() == Chem.BondType.SINGLE:
                o_single.append(other_idx)

        if len(o_double) >= 1 and len(o_single) >= 1:
            atom.SetFormalCharge(+1)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
            nitro_n.add(n_idx)

            o_idx = int(o_single[0])

            if is_editable(o_idx):
                o_atom = rwmol.GetAtomWithIdx(o_idx)
                o_atom.SetFormalCharge(-1)
                o_atom.SetNumExplicitHs(0)
                o_atom.SetNoImplicit(False)
                nitro_single_o.add(o_idx)

    # 2. Infer formal charges from local valence; leave hydrogens implicit.
    for atom in rwmol.GetAtoms():
        i = atom.GetIdx()

        if not is_editable(i):
            continue

        if i in nitro_n or i in nitro_single_o:
            continue

        sym = atom.GetSymbol()
        bonds = list(atom.GetBonds())

        sigma_valence = sum(bond_sigma_order(bond) for bond in bonds)
        is_aromatic = atom.GetIsAromatic() or atom_has_aromatic_bond(atom)

        # ----------------------------
        # C
        # ----------------------------
        if sym == "C":
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)

        # ----------------------------
        # N
        # ----------------------------
        elif sym == "N":
            if is_aromatic:
                if sigma_valence >= 4:
                    atom.SetFormalCharge(+1)
                else:
                    atom.SetFormalCharge(0)
            else:
                if sigma_valence >= 4:
                    atom.SetFormalCharge(+1)
                else:
                    atom.SetFormalCharge(0)

            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)

        # ----------------------------
        # O
        # ----------------------------
        elif sym == "O":
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)

        # ----------------------------
        # S
        # ----------------------------
        elif sym == "S":
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)

        # ----------------------------
        # P
        # ----------------------------
        elif sym == "P":
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)

        # ----------------------------
        # halogens
        # ----------------------------
        elif sym in {"F", "Cl", "Br", "I"}:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)

        # ----------------------------
        # fallback
        # ----------------------------
        else:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)

    try:
        rwmol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    return rwmol


def atom_can_accept_bond(rwmol, atom_idx, bond_order):
    atom = rwmol.GetAtomWithIdx(int(atom_idx))
    symbol = atom.GetSymbol()

    limit = get_simple_valence_limit(symbol)

    try:
        rwmol.UpdatePropertyCache(strict=False)
        current_valence = atom.GetExplicitValence()
    except Exception:
        return False

    return current_valence + int(bond_order) <= limit

def get_linker_components_in_mol(
        mol,
        linker_mask,
):
    """
    Identify connected components in the linker-induced bond graph.

    Only linker-linker bonds connect components; paths through
    fragment or anchor atoms are excluded.
    """

    if linker_mask is None:
        return []

    if linker_mask.dim() == 2:
        linker_mask = linker_mask.squeeze(-1)

    linker_mask = linker_mask.bool().detach().cpu()

    linker_indices = torch.where(linker_mask)[0].tolist()
    linker_indices = [int(x) for x in linker_indices]

    if len(linker_indices) == 0:
        return []

    linker_set = set(linker_indices)

    parent = {
        idx: idx
        for idx in linker_indices
    }

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx != ry:
            parent[ry] = rx

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        if i in linker_set and j in linker_set:
            union(i, j)

    comp_dict = {}

    for idx in linker_indices:
        root = find(idx)
        if root not in comp_dict:
            comp_dict[root] = []
        comp_dict[root].append(idx)

    components = list(comp_dict.values())

    return components


def build_molecules(
        one_hot,
        x,
        node_mask,
        anchors=None,
        linker_mask=None,
        pred_bond_probs=None,
        use_pred_bonds=False,
        pred_bond_types=None,
        pred_bond_threshold=0.5,
        pred_bond_max_new_bonds_per_atom=4,
        pred_bond_min_dist=0.75,
        pred_bond_max_dist=2.2,

        fragment_bond_types=None,
        use_template_fragment_bonds=False,

        formal_charges=None,
        atom_is_aromatic=None,
        num_explicit_hs=None,
        no_implicit=None,

        atom_props_mask=None,
):
    molecules = []

    for i in range(len(one_hot)):
        mask = node_mask[i].squeeze() == 1

        atom_types = one_hot[i][mask].argmax(dim=1).detach().cpu()
        positions = x[i][mask].detach().cpu()

        anchors_i = None
        linker_mask_i = None
        pred_bond_probs_i = None
        pred_bond_types_i = None
        fragment_bond_types_i = None

        formal_charges_i = None
        atom_is_aromatic_i = None
        num_explicit_hs_i = None
        no_implicit_i = None

        atom_props_mask_i = None

        if anchors is not None:
            anchors_i = anchors[i][mask].detach().cpu()

        if linker_mask is not None:
            linker_mask_i = linker_mask[i][mask].detach().cpu()

        if pred_bond_probs is not None:
            pred_bond_probs_i = pred_bond_probs[i][mask][:, mask].detach().cpu()

        if pred_bond_types is not None:
            pred_bond_types_i = pred_bond_types[i][mask][:, mask].detach().cpu()
        if fragment_bond_types is not None:
            fragment_bond_types_i = fragment_bond_types[i][mask][:, mask].detach().cpu()

        if formal_charges is not None:
            formal_charges_i = formal_charges[i][mask].detach().cpu()

        if atom_is_aromatic is not None:
            atom_is_aromatic_i = atom_is_aromatic[i][mask].detach().cpu()

        if num_explicit_hs is not None:
            num_explicit_hs_i = num_explicit_hs[i][mask].detach().cpu()

        if no_implicit is not None:
            no_implicit_i = no_implicit[i][mask].detach().cpu()

        if atom_props_mask is not None:
            atom_props_mask_i = atom_props_mask[i][mask].detach().cpu()


        mol = build_molecule(
            positions=positions,
            atom_types=atom_types,
            anchors=anchors_i,
            linker_mask=linker_mask_i,
            pred_bond_probs=pred_bond_probs_i,
            pred_bond_types=pred_bond_types_i,
            use_pred_bonds=use_pred_bonds,
            pred_bond_threshold=pred_bond_threshold,
            pred_bond_max_new_bonds_per_atom=pred_bond_max_new_bonds_per_atom,
            pred_bond_min_dist=pred_bond_min_dist,
            pred_bond_max_dist=pred_bond_max_dist,

            fragment_bond_types=fragment_bond_types_i,
            use_template_fragment_bonds=use_template_fragment_bonds,

            formal_charges=formal_charges_i,
            atom_is_aromatic=atom_is_aromatic_i,
            num_explicit_hs=num_explicit_hs_i,
            no_implicit=no_implicit_i,

            atom_props_mask=atom_props_mask_i,
        )

        molecules.append(mol)

    return molecules
def try_sanitize_mol(mol):
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def try_sanitize_mol_no_kekulize(mol):
    """
    Run RDKit sanitization without the kekulization step.

    Other enabled sanitization checks, including atom-property and
    valence checks, still run. The caller should update generated
    atom properties before validating each candidate bond.
    """
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
        return True
    except Exception:
        return False

def fix_common_formal_charges_inplace(rwmol):
    """
    Correct common formal-charge patterns using the current bond graph.

    Handle nitro groups and neutral nitrogen atoms with explicit
    valence four.
    """

    try:
        rwmol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    # 1. Correct nitro charges: N(=O)-O -> [N+](=O)[O-].
    for atom in rwmol.GetAtoms():
        if atom.GetSymbol() != "N":
            continue

        n_idx = atom.GetIdx()

        o_double = []
        o_single = []

        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)

            if other.GetSymbol() != "O":
                continue

            if bond.GetBondType() == Chem.BondType.DOUBLE:
                o_double.append(other.GetIdx())

            elif bond.GetBondType() == Chem.BondType.SINGLE:
                o_single.append(other.GetIdx())

        # Typical nitro pattern: one N=O and one N-O bond.
        if len(o_double) >= 1 and len(o_single) >= 1:
            atom.SetFormalCharge(+1)

            # Assign -1 to one singly bonded oxygen.
            o_atom = rwmol.GetAtomWithIdx(int(o_single[0]))
            o_atom.SetFormalCharge(-1)
            o_atom.SetNoImplicit(True)

    # 2. Assign +1 to neutral nitrogen atoms with explicit valence four.
    try:
        rwmol.UpdatePropertyCache(strict=False)

        for atom in rwmol.GetAtoms():
            if atom.GetSymbol() != "N":
                continue

            if atom.GetFormalCharge() != 0:
                continue

            try:
                explicit_valence = atom.GetExplicitValence()
            except Exception:
                continue

            if explicit_valence == 4:
                atom.SetFormalCharge(+1)

    except Exception:
        pass

    try:
        rwmol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    return rwmol


def fix_common_formal_charges(mol):
    """Apply common formal-charge corrections to an RDKit Mol."""
    rwmol = Chem.RWMol(mol)
    rwmol = fix_common_formal_charges_inplace(rwmol)
    return rwmol.GetMol()
def final_sanitize_or_keep(
        mol,
        tag="FINAL_SANITIZE_FAIL",
):
    """
    Attempt full RDKit sanitization, including kekulization.

    On failure, print the exception and try sanitization without
    kekulization. Return the current molecule in either case.
    """
    try:
        Chem.SanitizeMol(mol)
        return mol

    except Exception as e:
        print(f"[{tag}] {repr(e)}")

        try_sanitize_mol_no_kekulize(mol)

        return mol
def debug_aromatic_status(
        mol,
        tag="MOL",
        max_print_bonds=80,
        verbose=True,
):
    """
    Diagnose aromatic-bond and kekulization issues in an RDKit molecule.

    Compare sanitization with and without kekulization, and report
    aromatic bonds outside rings or inconsistent atom aromaticity flags.
    """
    result = {
        "mol_is_none": mol is None,
        "num_atoms": 0,
        "num_bonds": 0,
        "no_kekulize_sanitize_ok": False,
        "no_kekulize_msg": None,
        "full_sanitize_ok": False,
        "full_sanitize_msg": None,
        "num_aromatic_atoms": 0,
        "num_aromatic_bonds": 0,
        "num_aromatic_bonds_not_in_ring": 0,
        "num_aromatic_atom_flag_mismatch": 0,
    }

    if mol is None:
        if verbose:
            print(f"[{tag}] mol is None")
        return result

    result["num_atoms"] = mol.GetNumAtoms()
    result["num_bonds"] = mol.GetNumBonds()

    # ------------------------------------------------
    # 1. no-kekulize sanitize
    # ------------------------------------------------
    mol_no_kek = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(
            mol_no_kek,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
        result["no_kekulize_sanitize_ok"] = True
        result["no_kekulize_msg"] = "OK"

    except Exception as e:
        result["no_kekulize_sanitize_ok"] = False
        result["no_kekulize_msg"] = repr(e)

    # 2. Full sanitization, including kekulization.
    mol_full = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(mol_full)
        result["full_sanitize_ok"] = True
        result["full_sanitize_msg"] = "OK"

    except Exception as e:
        result["full_sanitize_ok"] = False
        result["full_sanitize_msg"] = repr(e)

    # 3. Count aromatic bonds and inspect atom aromaticity flags.
    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        pass

    aromatic_bonds = []
    aromatic_atoms = set()
    aromatic_bonds_not_in_ring = []
    aromatic_atom_flag_mismatch = []

    for bond in mol.GetBonds():
        is_aromatic_bond = (
                bond.GetIsAromatic()
                or bond.GetBondType() == Chem.BondType.AROMATIC
        )

        if not is_aromatic_bond:
            continue

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        aromatic_bonds.append((i, j))
        aromatic_atoms.add(i)
        aromatic_atoms.add(j)

        try:
            if not bond.IsInRing():
                aromatic_bonds_not_in_ring.append((i, j))
        except Exception:
            aromatic_bonds_not_in_ring.append((i, j))

        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)

        if not atom_i.GetIsAromatic() or not atom_j.GetIsAromatic():
            aromatic_atom_flag_mismatch.append((i, j))

    result["num_aromatic_atoms"] = len(aromatic_atoms)
    result["num_aromatic_bonds"] = len(aromatic_bonds)
    result["num_aromatic_bonds_not_in_ring"] = len(aromatic_bonds_not_in_ring)
    result["num_aromatic_atom_flag_mismatch"] = len(aromatic_atom_flag_mismatch)

    if verbose:
        print("=" * 100)
        print(f"[{tag}]")
        print(f"num_atoms = {result['num_atoms']}")
        print(f"num_bonds = {result['num_bonds']}")
        print(
            f"no_kekulize_sanitize_ok = "
            f"{result['no_kekulize_sanitize_ok']}, "
            f"msg = {result['no_kekulize_msg']}"
        )
        print(
            f"full_sanitize_ok = "
            f"{result['full_sanitize_ok']}, "
            f"msg = {result['full_sanitize_msg']}"
        )
        print(f"num_aromatic_atoms = {result['num_aromatic_atoms']}")
        print(f"num_aromatic_bonds = {result['num_aromatic_bonds']}")
        print(
            f"num_aromatic_bonds_not_in_ring = "
            f"{result['num_aromatic_bonds_not_in_ring']}"
        )
        print(
            f"num_aromatic_atom_flag_mismatch = "
            f"{result['num_aromatic_atom_flag_mismatch']}"
        )

        if len(aromatic_bonds_not_in_ring) > 0:
            print(
                "aromatic_bonds_not_in_ring = "
                f"{aromatic_bonds_not_in_ring[:max_print_bonds]}"
            )

        if len(aromatic_atom_flag_mismatch) > 0:
            print(
                "aromatic_atom_flag_mismatch = "
                f"{aromatic_atom_flag_mismatch[:max_print_bonds]}"
            )

        print("aromatic bonds:")

        for k, (i, j) in enumerate(aromatic_bonds[:max_print_bonds]):
            atom_i = mol.GetAtomWithIdx(int(i))
            atom_j = mol.GetAtomWithIdx(int(j))
            bond = mol.GetBondBetweenAtoms(int(i), int(j))

            print(
                f"  bond {k}: "
                f"{i}-{j}, "
                f"{atom_i.GetSymbol()}-{atom_j.GetSymbol()}, "
                f"bond_type={bond.GetBondType()}, "
                f"bond_aromatic={bond.GetIsAromatic()}, "
                f"atom_i_aromatic={atom_i.GetIsAromatic()}, "
                f"atom_j_aromatic={atom_j.GetIsAromatic()}, "
                f"in_ring={bond.IsInRing()}"
            )

        print("=" * 100)

    return result
def try_add_typed_bond_with_sanitize(
        rwmol,
        i,
        j,
        bond_type_idx,
        editable_atom_mask=None,
):
    """
    Attempt to add a predicted bond and validate the resulting molecule.

    Update editable atom properties from the trial bond graph before
    sanitization without kekulization. If validation fails, discard
    the trial molecule and return (False, rwmol).
    """
    i = int(i)
    j = int(j)
    bond_type_idx = int(bond_type_idx)

    if bond_type_idx <= 0:
        return False, rwmol

    if bond_type_idx >= len(const.BOND_DICT):
        return False, rwmol

    if rwmol.GetBondBetweenAtoms(i, j) is not None:
        return False, rwmol

    bond_type = const.BOND_DICT[bond_type_idx]

    if bond_type is None:
        return False, rwmol

    try:
        trial = Chem.RWMol(rwmol)

        atom_i = trial.GetAtomWithIdx(i)
        atom_j = trial.GetAtomWithIdx(j)

        if bond_type_idx == 4:
            atom_i.SetIsAromatic(True)
            atom_j.SetIsAromatic(True)

        trial.AddBond(
            i,
            j,
            bond_type,
        )

        if bond_type_idx == 4:
            bond = trial.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bond.SetIsAromatic(True)


        trial = infer_generated_atom_props_from_current_graph_inplace(
            rwmol=trial,
            editable_atom_mask=editable_atom_mask,
        )

        test_mol = trial.GetMol()

        if try_sanitize_mol_no_kekulize(test_mol):
            return True, Chem.RWMol(test_mol)

        return False, rwmol

    except Exception:
        return False, rwmol


def get_simple_valence_limit(atom_symbol):

    limits = {
        "C": 4,
        "N": 4,
        "O": 2,
        "F": 1,
        "Cl": 1,
        "Br": 1,
        "I": 1,
        "P": 5,
        "S": 6,
        "B": 3,
    }
    return limits.get(atom_symbol, 4)


def build_molecule(
        positions,
        atom_types,
        anchors=None,
        linker_mask=None,
        pred_bond_probs=None,
        pred_bond_types=None,
        use_pred_bonds=False,
        pred_bond_threshold=0.5,
        pred_bond_max_new_bonds_per_atom=4,
        pred_bond_min_dist=0.75,
        pred_bond_max_dist=2.2,

        fragment_bond_types=None,
        use_template_fragment_bonds=False,

        formal_charges=None,
        atom_is_aromatic=None,
        num_explicit_hs=None,
        no_implicit=None,

        atom_props_mask=None,
):

    if use_template_fragment_bonds and fragment_bond_types is not None:
        # Initialize fixed-fragment bonds from the reference template.
        mol = build_template_fragment_molecule(
            positions=positions,
            atom_types=atom_types,
            fragment_bond_types=fragment_bond_types,

            formal_charges=formal_charges,
            atom_is_aromatic=atom_is_aromatic,
            num_explicit_hs=num_explicit_hs,
            no_implicit=no_implicit,

            atom_props_mask=atom_props_mask,

        )

    # Add predicted linker-related bonds when enabled.
    if use_pred_bonds and pred_bond_probs is not None:

        mol = add_predicted_bonds_to_mol(
            mol=mol,
            pred_bond_probs=pred_bond_probs,
            pred_bond_types=pred_bond_types,
            atom_types=atom_types,
            positions=positions,
            anchors=anchors,
            linker_mask=linker_mask,
            threshold=pred_bond_threshold,
            max_new_bonds_per_atom=pred_bond_max_new_bonds_per_atom,
            only_linker_related=True,
            min_dist=pred_bond_min_dist,
            max_dist=pred_bond_max_dist,
            final_sanitize=True,
        )

    return mol

def get_bond_order(atom1, atom2, distance, check_exists=True, margins=const.MARGINS_EDM):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in const.BONDS_1:
            return 0
        if atom2 not in const.BONDS_1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < const.BONDS_1[atom1][atom2] + margins[0]:

        # Check if atoms in bonds2 dictionary.
        if atom1 in const.BONDS_2 and atom2 in const.BONDS_2[atom1]:
            thr_bond2 = const.BONDS_2[atom1][atom2] + margins[1]
            if distance < thr_bond2:
                if atom1 in const.BONDS_3 and atom2 in const.BONDS_3[atom1]:
                    thr_bond3 = const.BONDS_3[atom1][atom2] + margins[2]
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond

def get_bond_order_symmetric(atom1, atom2, distance, margins=const.MARGINS_EDM):
    """
    Infer bond order using either atom ordering in the bond-length tables.

    Retry with reversed atom symbols when the first lookup yields no bond.
    """
    order = get_bond_order(
        atom1,
        atom2,
        distance,
        check_exists=True,
        margins=margins,
    )

    if order > 0:
        return order

    order = get_bond_order(
        atom2,
        atom1,
        distance,
        check_exists=True,
        margins=margins,
    )

    return order