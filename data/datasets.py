from rdkit import Chem
from torch.utils.data import Dataset

from data.molecule_builder import build_molecules
from data.prepaer_final_data import summarize_bond_labels, build_true_bond_labels_from_full_mol_single, \
    build_fragment_bond_types_from_fragment_mol, build_rdkit_atom_props_from_full_mol, \
    build_full_bond_types_from_full_mol, check_reconstructed_mol_against_full_mol
from src import const
import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import json

def read_sdf(sdf_path):
    with Chem.SDMolSupplier(sdf_path, sanitize=False) as supplier:
        for molecule in supplier:
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

class ProtacDataset(Dataset):
    def __init__(self, data_path, dir, prefix, device):
        dataset_path = os.path.join(data_path, f'{prefix}.pt')
        if os.path.exists(dataset_path):
            self.data = torch.load(dataset_path, map_location=device)
        else:
            print(f'Preprocessing dataset with prefix {prefix}')
            self.data = ProtacDataset.preprocess(data_path, dir, prefix, device)
            torch.save(self.data, dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def preprocess(data_path, dir, prefix, device):
        data = []
        table_path = os.path.join(data_path, 'pretrain', dir, f'protac_processed.csv')
        mols_path = os.path.join(data_path, 'pretrain', dir, f'mol.sdf')
        fragments_path = os.path.join(data_path, 'pretrain', dir, f'fragments.sdf')
        linkers_path = os.path.join(data_path, 'pretrain', dir, f'linker.sdf')


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
            frag_pos, frag_one_hot, frag_charges = parse_molecule(fragments)
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

class RlDataset(Dataset):
    def __init__(self, data_path, dir, prefix, device):
        dataset_path = os.path.join(data_path, f"{prefix}.pt")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"RL dataset not found: {dataset_path}. "
                "Please preprocess RL data first."
            )

        self.data = torch.load(
            dataset_path,
            map_location=device,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def pad_square_matrix(mat, max_n):
    """
    mat: [N, N]
    return: [max_n, max_n]
    """
    n = mat.size(0)
    out = mat.new_zeros(max_n, max_n)
    out[:n, :n] = mat
    return out


def collate(batch):
    out = {}

    # Filter out big molecules
    # if 'pocket_mask' not in batch[0].keys():
    #    batch = [data for data in batch if data['num_atoms'] <= 50]
    # else:
    #    batch = [data for data in batch if data['num_atoms'] <= 1000]

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    max_n = max(data["positions"].size(0) for data in batch)

    square_attrs = {
        "bond_labels",
        "bond_mask",
        "fragment_bond_types",
        "full_bond_types",
    }

    for key, value in list(out.items()):

        if key in const.DATA_LIST_ATTRS:
            continue

        # bond_labels / bond_mask 是 [N, N]，不能用 pad_sequence
        if key in square_attrs:
            out[key] = torch.stack(
                [pad_square_matrix(v, max_n) for v in value],
                dim=0,
            )
            continue

        # 普通节点属性是 [N] 或 [N, C]，可以用 pad_sequence
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(
                value,
                batch_first=True,
                padding_value=0,
            )
            continue

        raise Exception(f"Unknown batch key: {key}")

    # 如果 preprocess 里保存了 fragment_mask，直接用
    # 如果没有，就用 fragment1_mask + fragment2_mask 兼容
    if "fragment_mask" not in out:
        if "fragment1_mask" in out and "fragment2_mask" in out:
            out["fragment_mask"] = (
                out["fragment1_mask"].bool() | out["fragment2_mask"].bool()
            ).to(const.TORCH_INT)
        else:
            raise Exception(
                "fragment_mask is missing, and fragment1_mask/fragment2_mask are also missing."
            )

    atom_mask = (
        out["fragment_mask"].bool() | out["linker_mask"].bool()
    ).to(const.TORCH_INT)

    out["atom_mask"] = atom_mask[:, :, None]

    batch_size, n_nodes = atom_mask.size()

    # In case of MOAD edge_mask is batch_idx
    if "pocket_mask" in batch[0].keys():
        batch_mask = torch.cat([
            torch.ones(n_nodes, dtype=const.TORCH_INT) * i
            for i in range(batch_size)
        ]).to(atom_mask.device)

        out["edge_mask"] = batch_mask

    else:
        edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]

        diag_mask = ~torch.eye(
            edge_mask.size(1),
            dtype=torch.bool,
            device=atom_mask.device,
        ).unsqueeze(0)

        edge_mask = edge_mask * diag_mask.to(edge_mask.dtype)

        out["edge_mask"] = edge_mask.view(
            batch_size * n_nodes * n_nodes,
            1,
        )

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out

def create_template(tensor, fragment_size, linker_size, fill=0):
    values_to_keep = tensor[:fragment_size]
    values_to_add = torch.ones(linker_size, tensor.shape[1], dtype=values_to_keep.dtype, device=values_to_keep.device)
    values_to_add = values_to_add * fill
    return torch.cat([values_to_keep, values_to_add], dim=0)

def create_templates_for_linker_generation(data, linker_sizes):
    """
    Takes data batch and new linker size and returns data batch where fragment-related data is the same
    but linker-related data is replaced with zero templates with new linker sizes.

    注意：
        这里不能遍历 const.DATA_ATTRS_TO_PAD。
        因为 DATA_ATTRS_TO_PAD 现在包含 formal_charges / atom_is_aromatic /
        num_explicit_hs / no_implicit 等真实 RDKit props。
        这些字段不应该进入采样 template，否则会泄露 linker 真实信息，
        也会因为它们是 [B, N] 而导致 create_template 的 shape[1] 报错。
    """
    decoupled_data = []
    fragment1_mask = data['fragment1_mask']  # [B, N, 1], warhead / target ligand fragment
    fragment2_mask = data['fragment2_mask']  # [B, N, 1], E3 ligand fragment
    fragment_12_mask = (fragment1_mask + fragment2_mask).clamp(max=1.0)
    for i, linker_size in enumerate(linker_sizes):
        data_dict = {}
        fragment_mask = fragment_12_mask[i].squeeze()
        fragment_size = fragment_mask.sum().int()
        for k, v in data.items():
            if k == 'num_atoms':
                # Computing new number of atoms (fragment_size + linker_size)
                data_dict[k] = fragment_size + linker_size
                continue



            if k == "fragment_bond_types":
                new_n = int(fragment_size + linker_size)

                template = torch.zeros(
                    new_n,
                    new_n,
                    dtype=v[i].dtype,
                    device=v[i].device,
                )

                template[
                :fragment_size,
                :fragment_size
                ] = v[i][
                    :fragment_size,
                    :fragment_size
                    ]

                data_dict[k] = template
                continue



            if k in const.DATA_LIST_ATTRS:
                # These attributes are written without modification
                data_dict[k] = v[i]
                continue
            if k in const.DATA_TEMPLATE_ATTRS_TO_PAD:
                fill_value = 1 if k == "linker_mask" else 0

                template = create_template(
                    v[i],
                    fragment_size,
                    linker_size,
                    fill=fill_value,
                )

                if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                    template = template.squeeze(-1)

                data_dict[k] = template

        decoupled_data.append(data_dict)

    return collate(decoupled_data)