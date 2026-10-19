import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from src import const
from tqdm import tqdm
import json
from data.molecule_builder import build_molecules
from rdkit import Chem
from src import const


def bond_to_idx_safe(bond):
    """
    RDKit bond -> 你的 bond index:
        0 = no bond
        1 = single
        2 = double
        3 = triple
        4 = aromatic
    """
    if bond is None:
        return 0

    bond_type = bond.GetBondType()
    return int(const.BOND2IDX.get(bond_type, 1))


def check_reconstructed_mol_against_full_mol(
        full_mol,
        recon_mol,
        dataset_to_full_atom_idx,
        name=None,
        check_atom_props=True,
        check_bond_types=True,
        max_print=20,
):
    """
    严格检查 reconstructed mol 是否和 full_mol 在 dataset 顺序下完全一致。

    full_mol:
        原始 RDKit full molecule。

    recon_mol:
        用 one_hot + positions + full_bond_types + atom props 重建出来的 mol。

    dataset_to_full_atom_idx:
        dataset index -> full_mol atom index。
        dataset 顺序通常是：
            [warhead atoms, e3_ligand atoms, linker atoms]
    """

    errors = []

    n = len(dataset_to_full_atom_idx)

    if recon_mol.GetNumAtoms() != n:
        errors.append(
            f"num atoms mismatch: "
            f"recon={recon_mol.GetNumAtoms()}, expected={n}"
        )

    # ------------------------------------------------
    # 1. atom symbol / atom props check
    # ------------------------------------------------
    n_check = min(recon_mol.GetNumAtoms(), n)

    for i in range(n_check):
        full_idx = int(dataset_to_full_atom_idx[i])

        full_atom = full_mol.GetAtomWithIdx(full_idx)
        recon_atom = recon_mol.GetAtomWithIdx(i)

        if full_atom.GetSymbol() != recon_atom.GetSymbol():
            errors.append(
                f"atom symbol mismatch at dataset_idx={i}, full_idx={full_idx}: "
                f"full={full_atom.GetSymbol()}, recon={recon_atom.GetSymbol()}"
            )

        if check_atom_props:
            if full_atom.GetFormalCharge() != recon_atom.GetFormalCharge():
                errors.append(
                    f"formal charge mismatch at dataset_idx={i}, full_idx={full_idx}: "
                    f"full={full_atom.GetFormalCharge()}, "
                    f"recon={recon_atom.GetFormalCharge()}"
                )

            if bool(full_atom.GetIsAromatic()) != bool(recon_atom.GetIsAromatic()):
                errors.append(
                    f"atom aromatic mismatch at dataset_idx={i}, full_idx={full_idx}: "
                    f"full={full_atom.GetIsAromatic()}, "
                    f"recon={recon_atom.GetIsAromatic()}"
                )

            if full_atom.GetNumExplicitHs() != recon_atom.GetNumExplicitHs():
                errors.append(
                    f"explicit H mismatch at dataset_idx={i}, full_idx={full_idx}: "
                    f"full={full_atom.GetNumExplicitHs()}, "
                    f"recon={recon_atom.GetNumExplicitHs()}"
                )

            if bool(full_atom.GetNoImplicit()) != bool(recon_atom.GetNoImplicit()):
                errors.append(
                    f"noImplicit mismatch at dataset_idx={i}, full_idx={full_idx}: "
                    f"full={full_atom.GetNoImplicit()}, "
                    f"recon={recon_atom.GetNoImplicit()}"
                )

    # ------------------------------------------------
    # 2. bond existence / bond type check
    # ------------------------------------------------
    if check_bond_types:
        for i in range(n):
            full_i = int(dataset_to_full_atom_idx[i])

            for j in range(i):
                full_j = int(dataset_to_full_atom_idx[j])

                full_bond = full_mol.GetBondBetweenAtoms(full_i, full_j)
                recon_bond = recon_mol.GetBondBetweenAtoms(i, j)

                full_bond_idx = bond_to_idx_safe(full_bond)
                recon_bond_idx = bond_to_idx_safe(recon_bond)

                if full_bond_idx != recon_bond_idx:
                    errors.append(
                        f"bond type mismatch at dataset_pair=({i}, {j}), "
                        f"full_pair=({full_i}, {full_j}): "
                        f"full={full_bond_idx}, recon={recon_bond_idx}"
                    )

    # ------------------------------------------------
    # 3. full sanitize check
    # ------------------------------------------------
    try:
        mol_copy = Chem.Mol(recon_mol)
        Chem.SanitizeMol(mol_copy)
    except Exception as e:
        errors.append(f"recon sanitize failed: {repr(e)}")

    if len(errors) > 0:
        prefix = f"[RECON_CHECK_FAIL] name={name}" if name is not None else "[RECON_CHECK_FAIL]"
        print(prefix)
        for msg in errors[:max_print]:
            print("  ", msg)

        if len(errors) > max_print:
            print(f"  ... {len(errors) - max_print} more errors")

        return False, errors

    return True, []

def build_rdkit_atom_props_from_full_mol(
        full_mol,
        dataset_to_full_atom_idx,
        device,
):
    """
    按 dataset 顺序保存 RDKit atom props。

    dataset 顺序是：
        [warhead atoms, e3 atoms, linker atoms]

    dataset_to_full_atom_idx:
        dataset index -> full_mol RDKit atom index
    """

    formal_charges = []
    atom_is_aromatic = []
    num_explicit_hs = []
    no_implicit = []

    for full_idx in dataset_to_full_atom_idx:
        atom = full_mol.GetAtomWithIdx(int(full_idx))

        formal_charges.append(atom.GetFormalCharge())
        atom_is_aromatic.append(int(atom.GetIsAromatic()))
        num_explicit_hs.append(atom.GetNumExplicitHs())
        no_implicit.append(int(atom.GetNoImplicit()))

    return {
        "formal_charges": torch.tensor(
            formal_charges,
            dtype=torch.long,
            device=device,
        ),
        "atom_is_aromatic": torch.tensor(
            atom_is_aromatic,
            dtype=torch.bool,
            device=device,
        ),
        "num_explicit_hs": torch.tensor(
            num_explicit_hs,
            dtype=torch.long,
            device=device,
        ),
        "no_implicit": torch.tensor(
            no_implicit,
            dtype=torch.bool,
            device=device,
        ),
    }
def build_fragment_bond_types_from_fragment_mol(
        fragments,
        n_total,
        device,
):
    """
    保存 fragment 内部真实键。

    返回:
        fragment_bond_types: [N, N]
            0 = no bond
            1 = single
            2 = double
            3 = triple
            4 = aromatic
    """
    fragment_bond_types = torch.zeros(
        n_total,
        n_total,
        dtype=torch.long,
        device=device,
    )

    for bond in fragments.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_type = bond.GetBondType()
        bond_idx = const.BOND2IDX.get(bond_type, 1)

        fragment_bond_types[i, j] = bond_idx
        fragment_bond_types[j, i] = bond_idx

    return fragment_bond_types
def read_sdf(sdf_path, strict_full_sanitize=True):
    with Chem.SDMolSupplier(
        sdf_path,
        sanitize=False,
        removeHs=False,
    ) as supplier:
        for molecule in supplier:
            if molecule is None:
                continue

            if strict_full_sanitize:
                try:
                    Chem.SanitizeMol(molecule)
                except Exception as e:
                    print(f"[READ_SDF_FULL_SANITIZE_FAIL] {sdf_path} | {repr(e)}")
                    continue
            else:
                try:
                    Chem.SanitizeMol(
                        molecule,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                    )
                except Exception as e:
                    print(f"[READ_SDF_PARTIAL_SANITIZE_FAIL] {sdf_path} | {repr(e)}")
                    continue

            yield molecule
def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot
def parse_molecule(mol):
    one_hot = []
    charges = []
    atom2idx = const.ATOM2IDX
    charges_dict = const.CHARGES
    for atom in mol.GetAtoms():
        one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
        charges.append(charges_dict[atom.GetSymbol()])
    positions = mol.GetConformer().GetPositions()
    return positions, np.array(one_hot), np.array(charges)
def _squeeze_to_bool(mask):
    if mask.dim() == 3:
        mask = mask.squeeze(-1)
    return mask.bool()


def has_single_bond_entry(atom1, atom2):
    if atom1 in const.BONDS_1 and atom2 in const.BONDS_1[atom1]:
        return True

    if atom2 in const.BONDS_1 and atom1 in const.BONDS_1[atom2]:
        return True

    return False


@torch.no_grad()
def build_true_bond_labels_from_full_mol_single(
        full_mol,
        dataset_to_full_atom_idx,
        atom_types,
        node_mask,
        linker_mask,
        anchor1_mask,
        anchor2_mask,
        use_element_candidate_mask=True,
):
    """
    从 full PROTAC mol 直接读取真实 bond。

    bond_labels:
        真实是否有键。

    bond_mask:
        参与监督的 pair。
        包括 linker-linker 和 anchor-linker 的正负样本。
    """

    if atom_types.dim() == 2:
        atom_types = atom_types.argmax(dim=-1)

    node_mask = _squeeze_to_bool(node_mask)
    linker_mask = _squeeze_to_bool(linker_mask)
    anchor1_mask = _squeeze_to_bool(anchor1_mask)
    anchor2_mask = _squeeze_to_bool(anchor2_mask)

    anchor_mask = anchor1_mask | anchor2_mask

    N = node_mask.shape[0]
    device = node_mask.device

    if len(dataset_to_full_atom_idx) != N:
        raise ValueError(
            f"full atom mapping length mismatch: "
            f"{len(dataset_to_full_atom_idx)} != {N}"
        )

    bond_labels = torch.zeros(
        N, N,
        device=device,
        dtype=torch.float32,
    )

    bond_mask = torch.zeros(
        N, N,
        device=device,
        dtype=torch.float32,
    )

    idx2atom = const.IDX2ATOM

    for i in range(N):
        if not node_mask[i]:
            continue

        ai = idx2atom[int(atom_types[i].item())]
        full_i = int(dataset_to_full_atom_idx[i])

        for j in range(i):
            if not node_mask[j]:
                continue

            aj = idx2atom[int(atom_types[j].item())]
            full_j = int(dataset_to_full_atom_idx[j])

            is_linker_linker = bool(
                linker_mask[i] and linker_mask[j]
            )

            is_anchor_linker = bool(
                (anchor_mask[i] and linker_mask[j])
                or
                (linker_mask[i] and anchor_mask[j])
            )

            if not (is_linker_linker or is_anchor_linker):
                continue

            # 是否是元素上可能成键的候选 pair
            element_candidate = has_single_bond_entry(ai, aj)

            # full mol 中是否真实有键
            true_bond = full_mol.GetBondBetweenAtoms(full_i, full_j)

            has_true_bond = true_bond is not None

            # bond_mask 要包含：
            # 1. 元素上可能成键的负样本；
            # 2. full mol 中真实存在的正样本。
            #
            # 第二条很重要：避免某些真实键因为元素表不全被漏掉。
            if use_element_candidate_mask:
                if not element_candidate and not has_true_bond:
                    continue

            bond_mask[i, j] = 1.0
            bond_mask[j, i] = 1.0

            if has_true_bond:
                bond_labels[i, j] = 1.0
                bond_labels[j, i] = 1.0

    return bond_labels, bond_mask

@torch.no_grad()
def build_full_bond_types_from_full_mol(
        full_mol,
        dataset_to_full_atom_idx,
        n_total,
        device,
):
    """
    构造完整真实 PROTAC bond type matrix。

    注意：
        full_bond_types 的 index 顺序必须是 dataset 顺序：
            [fragment atoms, linker atoms]

        不是 full_mol 原始 RDKit atom index 顺序。

    返回:
        full_bond_types: [N, N]
            0 = no bond
            1 = single
            2 = double
            3 = triple
            4 = aromatic
    """

    if len(dataset_to_full_atom_idx) != n_total:
        raise ValueError(
            f"full atom mapping length mismatch: "
            f"{len(dataset_to_full_atom_idx)} != {n_total}"
        )

    full_bond_types = torch.zeros(
        n_total,
        n_total,
        dtype=torch.long,
        device=device,
    )

    # i, j 是 dataset index
    # full_i, full_j 是 full_mol 里的 RDKit index
    for i in range(n_total):
        full_i = int(dataset_to_full_atom_idx[i])

        for j in range(i):
            full_j = int(dataset_to_full_atom_idx[j])

            true_bond = full_mol.GetBondBetweenAtoms(full_i, full_j)

            if true_bond is None:
                continue

            bond_type = true_bond.GetBondType()
            bond_idx = const.BOND2IDX.get(bond_type, 1)

            full_bond_types[i, j] = bond_idx
            full_bond_types[j, i] = bond_idx

    return full_bond_types
def summarize_bond_labels(
        bond_labels,
        linker_mask,
        anchor1_mask,
        anchor2_mask,
):
    labels = bond_labels.bool()

    linker_mask = linker_mask.bool()
    anchor1_mask = anchor1_mask.bool()
    anchor2_mask = anchor2_mask.bool()

    a1_l = (
        anchor1_mask[:, None] & linker_mask[None, :]
    ) | (
        linker_mask[:, None] & anchor1_mask[None, :]
    )

    a2_l = (
        anchor2_mask[:, None] & linker_mask[None, :]
    ) | (
        linker_mask[:, None] & anchor2_mask[None, :]
    )

    ll = linker_mask[:, None] & linker_mask[None, :]

    n_a1_l = int((labels & a1_l).sum().item() // 2)
    n_a2_l = int((labels & a2_l).sum().item() // 2)
    n_ll = int((labels & ll).sum().item() // 2)

    return n_a1_l, n_a2_l, n_ll

def clean_optional_path(v):
    if pd.isna(v):
        return None

    s = str(v).strip()

    if s == "":
        return None

    return s
def preprocess(data_path, device):
    data = []
    # table_path = os.path.join(data_path, 'protac_processed.csv')
    table_path = os.path.join(data_path, 'protac_processed_deepternary.csv')
    mols_path = os.path.join(data_path, 'mol.sdf')
    fragments_path = os.path.join(data_path, 'fragments.sdf')
    linkers_path = os.path.join(data_path, 'linker.sdf')
    table = pd.read_csv(table_path)
    generator = tqdm(
        zip(
            table.iterrows(),
            read_sdf(mols_path),
            read_sdf(fragments_path),
            read_sdf(linkers_path),
        ),
        total=len(table),
    )

    for (_, row), full_mol, fragments, linker in generator:
        uuid = row['uuid']
        name = row['molecule']
        frag_pos, frag_one_hot, frag_charges = parse_molecule(fragments,)
        link_pos, link_one_hot, link_charges = parse_molecule(linker)

        positions = np.concatenate([frag_pos, link_pos], axis=0)
        one_hot = np.concatenate([frag_one_hot, link_one_hot], axis=0)
        charges = np.concatenate([frag_charges, link_charges], axis=0)

        n_frag = len(frag_charges)
        n_link = len(link_charges)
        n_total = n_frag + n_link
        fragment_bond_types_t = build_fragment_bond_types_from_fragment_mol(
            fragments=fragments,
            n_total=n_total,
            device=device,
        )
        n_warhead = int(row['warhead_num_atoms'])
        n_e3 = int(row['e3_ligand_num_atoms'])
        assert n_warhead + n_e3 == n_frag, (
            f"fragment atom number mismatch: "
            f"warhead {n_warhead} + e3 {n_e3} != fragment {n_frag}"
        )
        # ------------------------------------------------
        # 1. anchor masks
        # ------------------------------------------------
        anchor1_mask = np.zeros_like(charges)  # warhead anchor
        anchor2_mask = np.zeros_like(charges)  # E3 anchor
        anchor1_idx = int(row['warhead_anchor'])
        anchor2_idx = int(row['e3_ligand_anchor'])
        anchor1_mask[anchor1_idx] = 1
        anchor2_mask[anchor2_idx] = 1
        # 兼容原来的 anchors
        anchors = anchor1_mask + anchor2_mask
        # ------------------------------------------------
        # 2. fragment-side masks
        # ------------------------------------------------
        fragment1_mask = np.concatenate([
            np.ones(n_warhead, dtype=np.float32),
            np.zeros(n_e3, dtype=np.float32),
            np.zeros(n_link, dtype=np.float32),
        ])
        fragment2_mask = np.concatenate([
            np.zeros(n_warhead, dtype=np.float32),
            np.ones(n_e3, dtype=np.float32),
            np.zeros(n_link, dtype=np.float32),
        ])
        # 兼容原来的 fragment_mask
        fragment_mask = fragment1_mask + fragment2_mask
        # ------------------------------------------------
        # 3. linker mask
        # ------------------------------------------------
        linker_mask = np.concatenate([
            np.zeros(n_frag, dtype=np.float32),
            np.ones(n_link, dtype=np.float32),
        ])

        # ------------------------------------------------
        # 4. convert to tensors
        # ------------------------------------------------
        positions_t = torch.tensor(
            positions,
            dtype=const.TORCH_FLOAT,
            device=device,
        )

        one_hot_t = torch.tensor(
            one_hot,
            dtype=const.TORCH_FLOAT,
            device=device,
        )

        charges_t = torch.tensor(
            charges,
            dtype=const.TORCH_FLOAT,
            device=device,
        )

        linker_mask_t = torch.tensor(
            linker_mask,
            dtype=const.TORCH_FLOAT,
            device=device,
        )

        anchor1_mask_t = torch.tensor(
            anchor1_mask,
            dtype=const.TORCH_FLOAT,
            device=device,
        )

        anchor2_mask_t = torch.tensor(
            anchor2_mask,
            dtype=const.TORCH_FLOAT,
            device=device,
        )

        fragment1_mask_t = torch.tensor(
            fragment1_mask,
            dtype=const.TORCH_FLOAT,
            device=device,
        )

        fragment2_mask_t = torch.tensor(
            fragment2_mask,
            dtype=const.TORCH_FLOAT,
            device=device,
        )

        anchors_t = torch.tensor(
            anchors,
            dtype=const.TORCH_FLOAT,
            device=device,
        )

        node_mask_t = torch.ones(
            n_total,
            dtype=const.TORCH_FLOAT,
            device=device,
        )

        atom_types_t = one_hot_t.argmax(dim=-1)

        # ------------------------------------------------
        # 5. precompute bond_labels / bond_mask
        # ------------------------------------------------
        # 注意：
        # 这里 positions_t 来自 SDF，是原始 Å 坐标，所以 norm_value=1.0。
        # 如果你这里保存的是 positions / 5.0，则改成 norm_value=5.0。
        dataset_to_full_atom_idx = json.loads(row["full_atom_indices"])
        dataset_to_full_atom_idx = [int(x) for x in dataset_to_full_atom_idx]

        rdkit_atom_props = build_rdkit_atom_props_from_full_mol(
            full_mol=full_mol,
            dataset_to_full_atom_idx=dataset_to_full_atom_idx,
            device=device,
        )

        # 完整真实 PROTAC bond graph
        full_bond_types_t = build_full_bond_types_from_full_mol(
            full_mol=full_mol,
            dataset_to_full_atom_idx=dataset_to_full_atom_idx,
            n_total=n_total,
            device=device,
        )

        bond_labels_t, bond_mask_t = build_true_bond_labels_from_full_mol_single(
            full_mol=full_mol,
            dataset_to_full_atom_idx=dataset_to_full_atom_idx,
            atom_types=atom_types_t,
            node_mask=node_mask_t,
            linker_mask=linker_mask_t,
            anchor1_mask=anchor1_mask_t,
            anchor2_mask=anchor2_mask_t,
            use_element_candidate_mask=True,
        )
        # ------------------------------------------------
        # sanity check:
        # full_bond_types 在 linker-related 监督区域中，
        # 应该和 bond_labels 完全一致。
        # ------------------------------------------------
        full_has_bond_t = full_bond_types_t > 0

        if not torch.equal(
                full_has_bond_t[bond_mask_t.bool()],
                bond_labels_t.bool()[bond_mask_t.bool()],
        ):
            print(
                f"[SKIP] uuid={uuid}, name={name}: "
                f"full_bond_types mismatch with bond_labels"
            )
            continue
        n_a1_l, n_a2_l, n_ll = summarize_bond_labels(
            bond_labels_t,
            linker_mask_t.bool(),
            anchor1_mask_t.bool(),
            anchor2_mask_t.bool(),
        )

        if n_a1_l != 1 or n_a2_l != 1:
            print(
                f"[SKIP] uuid={uuid}, name={name}: "
                f"n_a1_linker_bonds={n_a1_l}, "
                f"n_a2_linker_bonds={n_a2_l}, "
                f"n_linker_linker_bonds={n_ll}"
            )
            continue

        # 节省内存：label/mask 用 bool 存，loss 里面再转 float
        bond_labels_t = bond_labels_t.bool().cpu()
        bond_mask_t = bond_mask_t.bool().cpu()

        # ------------------------------------------------
        # final check:
        # 用当前保存的信息重建 true molecule，
        # 再和原始 full_mol 逐原子、逐键比较。
        # ------------------------------------------------
        with torch.no_grad():
            one_hot_b = one_hot_t.unsqueeze(0)
            positions_b = positions_t.unsqueeze(0)
            node_mask_b = node_mask_t.unsqueeze(0)

            full_bond_types_b = full_bond_types_t.unsqueeze(0)

            formal_charges_b = rdkit_atom_props["formal_charges"].unsqueeze(0)
            atom_is_aromatic_b = rdkit_atom_props["atom_is_aromatic"].unsqueeze(0)
            num_explicit_hs_b = rdkit_atom_props["num_explicit_hs"].unsqueeze(0)
            no_implicit_b = rdkit_atom_props["no_implicit"].unsqueeze(0)

            recon_mol = build_molecules(
                one_hot=one_hot_b,
                x=positions_b,
                node_mask=node_mask_b,

                fragment_bond_types=full_bond_types_b,
                use_template_fragment_bonds=True,

                formal_charges=formal_charges_b,
                atom_is_aromatic=atom_is_aromatic_b,
                num_explicit_hs=num_explicit_hs_b,
                no_implicit=no_implicit_b,

                use_pred_bonds=False,
            )[0]

            ok, errors = check_reconstructed_mol_against_full_mol(
                full_mol=full_mol,
                recon_mol=recon_mol,
                dataset_to_full_atom_idx=dataset_to_full_atom_idx,
                name=name,
                check_atom_props=True,
                check_bond_types=True,
                max_print=20,
            )

            if not ok:
                print(f"[SKIP] uuid={uuid}, name={name}: true reconstruction check failed")
                continue


        data.append({
            "uuid": uuid,
            "name": name,

            # RL reward / PROTAC-STAN 需要的蛋白 ID
            'poi_uniprot': row['Uniprot'],
            'e3_uniprot': row['e3_ligase_Uniprot'],
            'p1_path': clean_optional_path(row['p1_path']),
            'p2_path': clean_optional_path(row['p2_path']),
            'unbound_lig1_path': clean_optional_path(row['unbound_lig1_path']),
            'unbound_lig2_path': clean_optional_path(row['unbound_lig2_path']),
            'lig1_mask_path': clean_optional_path(row['lig1_mask_path']),
            'lig2_mask_path': clean_optional_path(row['lig2_mask_path']),

            "positions": positions_t,
            "one_hot": one_hot_t,
            "charges": charges_t,

            # 兼容原来的字段
            "anchors": anchors_t,
            "linker_mask": linker_mask_t,

            # 新增字段
            "anchor1_mask": anchor1_mask_t,
            "anchor2_mask": anchor2_mask_t,
            "fragment1_mask": fragment1_mask_t,
            "fragment2_mask": fragment2_mask_t,

            # bond topology supervision
            "bond_labels": bond_labels_t,
            "bond_mask": bond_mask_t,

            "fragment_bond_types": fragment_bond_types_t.to(torch.int8).cpu(),
            "full_bond_types": full_bond_types_t.to(torch.int8).cpu(),

            "formal_charges": rdkit_atom_props["formal_charges"].to(torch.int8).cpu(),
            "atom_is_aromatic": rdkit_atom_props["atom_is_aromatic"].cpu(),
            "num_explicit_hs": rdkit_atom_props["num_explicit_hs"].to(torch.int8).cpu(),
            "no_implicit": rdkit_atom_props["no_implicit"].cpu(),


            "num_atoms": len(positions),
        })

    return data

if __name__ == "__main__":
    dirs = ['train', 'val', 'test']
    for dir in dirs:
        data_path = f'./datasets/rl/{dir}'
        device = 'cuda:0'
        dataset_path = f'./datasets/protac_rltrain_{dir}.pt'
        data = preprocess(data_path, device)
        torch.save(data, dataset_path)
