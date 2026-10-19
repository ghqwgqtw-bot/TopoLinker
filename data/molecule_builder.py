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

        # # 新增
        # infer_atom_props_from_graph=False,
        # atom_props_infer_mask=None,
):
    """
    最终版初始分子:
        1. 添加全部原子；
        2. 只添加 fragment 内部真实键；
        3. 不根据距离添加 linker-linker / anchor-linker 键。
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

    # # ------------------------------------------------
    # # 根据当前 atom type + bond type 确定性推断 atom props
    # # 用于模拟：模型只输出 atom type / coords / bond type
    # # ------------------------------------------------
    # if infer_atom_props_from_graph:
    #     mol = infer_generated_atom_props_from_current_graph_inplace(
    #         rwmol=mol,
    #         editable_atom_mask=atom_props_infer_mask,
    #         aromatic_n_h_mode="zero",
    #     )

    # 修复常见 formal charge，例如 nitro: [N+](=O)[O-]
    # mol = fix_common_formal_charges_inplace(mol)

    mol = mol.GetMol()

    mol.AddConformer(
        create_conformer(
            positions.detach().cpu().numpy().astype(np.float64)
        )
    )

    # try:
    #     mol_copy = Chem.Mol(mol)
    #     Chem.SanitizeMol(mol_copy)
    # except Exception as e:
    #     print(f"[TEMPLATE_FULL_SANITIZE_FAIL] {repr(e)}")

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
    根据 [N,N] bond type matrix 添加 fragment 内部真实键。

    0 = no bond
    1 = SINGLE
    2 = DOUBLE
    3 = TRIPLE
    4 = AROMATIC

    对于 aromatic bond：
        1. 添加 Chem.BondType.AROMATIC；
        2. 同时设置两端 atom 的 aromatic flag；
        3. 设置 bond 的 aromatic flag。
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
    使用 bond_head 预测结果给 RDKit mol 补键。

    pred_bond_probs:
        [N, N]，成键概率，通常为 1 - P(no bond)

    pred_bond_types:
        [N, N]，预测键类型：
            1 = single
            2 = double
            3 = triple
            4 = aromatic

        如果 pred_bond_types=None，则退化为旧 binary 版本：
            所有预测键都按 SINGLE 加。

    只补：
        1. linker-linker
        2. anchor-linker

    不补：
        1. fragment-fragment
        2. non-anchor fragment-linker
        3. fragment1-fragment2
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

            # ------------------------------------------------
            # linker-linker：
            # 两端都是生成 linker atom，所以两端都限制为 C/N/O/S/F
            # ------------------------------------------------
            if is_linker_linker:
                if atom_i_symbol not in allowed_linker_atoms:
                    continue
                if atom_j_symbol not in allowed_linker_atoms:
                    continue

                has_f_linker_atom = (
                        atom_i_symbol == "F"
                        or atom_j_symbol == "F"
                )

            # ------------------------------------------------
            # anchor-linker：
            # 只限制 linker 那一端；
            # anchor 是真实 fragment 连接位点，不强行限制为 C/N/O/S/F
            # ------------------------------------------------
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

            # ------------------------------------------------
            # F 规则：
            # F 只允许作为 linker-linker terminal substituent；
            # 不允许参与 anchor-linker
            # ------------------------------------------------
            if has_f_linker_atom and is_anchor_linker:
                continue
            # F 作为 linker 取代基时，只允许 C-F
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
                # 兼容旧 binary bond head
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

            # F 只能形成 single bond
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
    # predicted bonds 加完后，根据当前预测键图确定性计算 linker props
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
        # 中间态：不做 full sanitize，避免在 repair 前过早触发 kekulize / final 检查
        # 但可以做一次 no-kekulize sanitize，尽量更新 RDKit property cache
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

    # 只用于生成阶段的预测键。
    #
    # 规则：
    #     1. linker-linker aromatic bond 可以保留；
    #     2. anchor-linker aromatic bond 不允许；
    #     3. 若 anchor-linker 被预测为 aromatic，直接返回 0，即不加这条键。
    #
    # 原因：
    #     真实数据中 anchor-linker aromatic = 0；
    #     并且采样阶段 anchor-linker 也只从 single/double/triple 中选择类型。


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
        # aromatic 这里近似按 1 处理，最终由 RDKit sanitize 判断
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
    根据当前 RDKit RWMol 中已有的 atom type + bond type，
    确定性推断 editable 原子的 formal_charge / aromatic flag。

    这里采用 implicit H 模式：
        num_explicit_hs = 0
        no_implicit = False

    目的：
        尽量和数据中真实 RDKit props 的表示方式一致。
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

    # ------------------------------------------------
    # 0. 只重置 editable 原子的 props
    # ------------------------------------------------
    for atom in rwmol.GetAtoms():
        idx = atom.GetIdx()

        if not is_editable(idx):
            continue

        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)

        # implicit H 模式：和真实 RDKit props 更一致
        atom.SetNoImplicit(False)

        if atom_has_aromatic_bond(atom):
            atom.SetIsAromatic(True)
        else:
            atom.SetIsAromatic(False)

    try:
        rwmol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    # ------------------------------------------------
    # 1. 先处理 nitro: N(=O)-O -> [N+](=O)[O-]
    # ------------------------------------------------
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

    # ------------------------------------------------
    # 2. 根据局部价态推断 formal charge
    #    H 不手动补，交给 RDKit implicit H
    # ------------------------------------------------
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
    只看 linker 原子之间的已有键，计算 linker 内部 components。

    注意：
        这里不通过 fragment / anchor 间接连接 linker。
        只统计 linker-linker bond。
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

            # 新增
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
    中间态 sanitize：
        保留大部分 RDKit 检查，
        但跳过 kekulize。

    注意：
        虽然跳过了 kekulize，
        但仍然会检查 valence / formal charge / implicit H /
        property cache / aromaticity / conjugation / hybridization。

    因此逐条添加 predicted bond 时，
    仍然需要先根据当前 graph 推断 linker atom props，
    否则一些本来合理的候选键可能因为 props 未更新而被拒绝。
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
    根据局部键型修复常见 formal charge。

    当前主要修复：
        nitro: N(=O)-O  ->  [N+](=O)[O-]

    作用：
        解决真实数据中 N 显式价态 4 但 formal charge 缺失的问题。
    """

    try:
        rwmol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    # ------------------------------------------------
    # 1. 修复 nitro: N(=O)-O -> [N+](=O)[O-]
    # ------------------------------------------------
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

        # 典型 nitro: 一个 N=O，一个 N-O
        if len(o_double) >= 1 and len(o_single) >= 1:
            atom.SetFormalCharge(+1)

            # 只给其中一个单键 O 设置 -1
            o_atom = rwmol.GetAtomWithIdx(int(o_single[0]))
            o_atom.SetFormalCharge(-1)
            o_atom.SetNoImplicit(True)

    # ------------------------------------------------
    # 2. 保守修复四价 N：普通四价 N 通常应为 [N+]
    # ------------------------------------------------
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
    """
    Mol 版本包装。
    """
    rwmol = Chem.RWMol(mol)
    rwmol = fix_common_formal_charges_inplace(rwmol)
    return rwmol.GetMol()
def final_sanitize_or_keep(
        mol,
        tag="FINAL_SANITIZE_FAIL",
):
    """
    最终分子检查：
        1. 先尝试 full Chem.SanitizeMol，包括 kekulize；
        2. 如果失败，打印错误；
        3. 不回退到旧 mol，保留当前已经加过键的 mol；
        4. 再做一次 no-kekulize sanitize，尽量更新 RDKit 属性缓存。
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
    诊断 molecule 的 aromatic / kekulize 状态。

    主要用于：
        1. 检查真实完整分子 full_bond_types 是否本身有 aromatic 问题；
        2. 检查生成分子最终 sanitize 失败是否由 aromatic 导致。
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

    # ------------------------------------------------
    # 2. full sanitize，包括 kekulize
    # ------------------------------------------------
    mol_full = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(mol_full)
        result["full_sanitize_ok"] = True
        result["full_sanitize_msg"] = "OK"

    except Exception as e:
        result["full_sanitize_ok"] = False
        result["full_sanitize_msg"] = repr(e)

    # ------------------------------------------------
    # 3. aromatic bond / atom 统计
    # ------------------------------------------------
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
    添加 typed bond。

    如果 infer_generated_atom_props=True：
        添加候选键后，先根据当前预测 graph 更新 linker atom props，
        尤其是 formal_charge / implicit H / aromatic flag，
        再做 no-kekulize sanitize。

    这里跳过 kekulize 是为了避免 aromatic 中间态过早失败；
    但 sanitize 仍然会检查价态和 property cache，
    所以 props 推断仍然必要。

    如果失败：
        不修改原始 rwmol，直接返回 False, rwmol。
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

        # 新增
        fragment_bond_types=None,
        use_template_fragment_bonds=False,

        formal_charges=None,
        atom_is_aromatic=None,
        num_explicit_hs=None,
        no_implicit=None,

        atom_props_mask=None,
):

    if use_template_fragment_bonds and fragment_bond_types is not None:
        # 最终版：fragment 键来自真实模板；linker 相关键后面由 bond head 加
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

    # ------------------------------------------------
    # 使用 bond_head 预测结果补 linker 相关键
    # ------------------------------------------------
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
    对称版本的 get_bond_order。
    避免 const.BONDS_1 只存了 C-N 而没存 N-C 这种情况。
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