import pandas as pd
from rdkit import Chem
import numpy as np
from tqdm import tqdm
from rdkit import RDLogger
from rdkit.Chem import SaltRemover
RDLogger.DisableLog('rdApp.*')
remover = SaltRemover.SaltRemover()

# ===============================
# 读取数据
# ===============================

protac = pd.read_csv('./datasets/protac.csv', low_memory=False)
linker = pd.read_csv('./datasets/linker.csv')
e3 = pd.read_csv('./datasets/e3_ligand.csv')

protac_df = protac.copy()
linker_df = linker.copy()
e3_df = e3.copy()

# 空字符串 替换为 NaN。 四列中有一个值为 NaN, 删除该行
protac_df[['Uniprot', 'Target', 'E3 ligase', 'Smiles', 'Molecular Weight', 'Exact Mass', 'XLogP3', 'Heavy Atom Count', 'Ring Count', 'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count', 'Rotatable Bond Count', 'Topological Polar Surface Area']] = protac_df[['Uniprot', 'Target', 'E3 ligase', 'Smiles', 'Molecular Weight', 'Exact Mass', 'XLogP3', 'Heavy Atom Count', 'Ring Count', 'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count', 'Rotatable Bond Count', 'Topological Polar Surface Area']].replace('', pd.NA)
linker_df['Smiles'] = linker_df['Smiles'].replace('', pd.NA)
e3_df[['Uniprot', 'Target', 'Smiles']] = e3_df[['Uniprot', 'Target', 'Smiles']].replace('', pd.NA)
protac_df = protac_df.dropna(subset=['Uniprot', 'Target', 'E3 ligase', 'Smiles', 'Molecular Weight', 'Exact Mass', 'XLogP3', 'Heavy Atom Count', 'Ring Count', 'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count', 'Rotatable Bond Count', 'Topological Polar Surface Area'])
linker_df = linker_df.dropna(subset=['Smiles'])
e3_df = e3_df.dropna(subset=['Uniprot', 'Target', 'Smiles'])

# 删除重复行
protac_df = protac_df.drop_duplicates(subset=['Uniprot', 'Target', 'E3 ligase', 'Smiles'])
e3_df = e3_df.drop_duplicates(subset=['Uniprot', 'Target', 'Smiles'])

protac_df = protac_df[['Compound ID', 'Uniprot', 'Target', 'E3 ligase', 'Smiles', 'Molecular Weight', 'Exact Mass', 'XLogP3', 'Heavy Atom Count', 'Ring Count', 'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count', 'Rotatable Bond Count', 'Topological Polar Surface Area']]

# ===============================
# 工具函数
# ===============================

def canon(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        m = remover.StripMol(m)
        if m is None:
            return None
        if len(Chem.GetMolFrags(m)) != 1:
            return None
        return Chem.MolToSmiles(m)
    except:
        return None


def mol_from_mask(pmol, mask):
    """根据 atom mask 生成 linker 子分子"""
    # Chem.Mol(pmol) 复制一份 Mol
    # Chem.RWMol(...) 将其转为 可修改的分子对象（RWMol），可以删除原子。
    new_mask = mask.copy()
    emol = Chem.RWMol(Chem.Mol(pmol))
    # delete 列表里保存 所有需要删除的原子索引。
    delete = [i for i, keep in enumerate(new_mask) if not keep]
    save = [i for i, keep in enumerate(new_mask) if keep]
    anchor_num = 0
    for i in save:
        atom = emol.GetAtomWithIdx(i)
        bonds = atom.GetBonds()
        for bond in bonds:
            source_idx = bond.GetBeginAtomIdx()
            target_idx = bond.GetEndAtomIdx()
            unknown_idx = target_idx if source_idx == i else source_idx
            if unknown_idx in delete:
                anchor_num += 1
                new_mask[unknown_idx] = 1
                emol.GetAtomWithIdx(unknown_idx).SetAtomicNum(0)
    if(anchor_num != 2):
        return None
    delete = [i for i, keep in enumerate(new_mask) if not keep]

    for idx in reversed(delete):
        emol.RemoveAtom(idx)

    mol = emol.GetMol()

    try:
        Chem.SanitizeMol(mol)
    except:
        return None
    return mol


# ===============================
# 规范 SMILES
# ===============================

protac_df['Smiles'] = protac_df['Smiles'].apply(canon)
linker_df['Smiles'] = linker_df['Smiles'].apply(canon)
e3_df['Smiles'] = e3_df['Smiles'].apply(canon)

protac_df = protac_df.dropna(subset=['Smiles'])
linker_df = linker_df.dropna(subset=['Smiles'])
e3_df = e3_df.dropna(subset=['Smiles'])

# protac_df = protac_df.drop_duplicates(subset=['Smiles'])
# ===============================
# Mol 对象
# ===============================

protac_df['mol'] = protac_df['Smiles'].apply(Chem.MolFromSmiles)
linker_df['mol'] = linker_df['Smiles'].apply(Chem.MolFromSmiles)
e3_df['mol'] = e3_df['Smiles'].apply(Chem.MolFromSmiles)

# ===============================
# linker 按原子数排序
# ===============================
# linker_df['atom_num'] = linker_df['Heavy Atom Count'].astype(int)

linker_df['atom_num'] = linker_df['mol'].apply(
    lambda m: m.GetNumAtoms() if m else 0
)

linker_df = linker_df.sort_values('atom_num', ascending=False)


linker_list = []
# 遍历 DataFrame 的每一行
for row in linker_df.itertuples():
    if row.mol:
        linker_list.append(
            (row.mol, row.Smiles)
        )

# ===============================
# 构建 E3 ligand 字典
# ===============================

e3_dict = {}

for row in e3_df.itertuples():
    # 取每行的 'Target' 列作为字典的 key, 如果这一列不存在或为空，就返回 None
    key = getattr(row, 'Target', None)
    # 跳过没有 E3 ligase 名称或没有 Mol 对象的行
    if key is None or row.mol is None:
        continue

    # 初始化这个 ligase 的列表，用于存放它对应的 ligand
    if key not in e3_dict:
        e3_dict[key] = []
    # 把 ligand 的 Mol 对象和 SMILES 放入列表
    e3_dict[key].append(
        (row.mol, row.Smiles, row.Uniprot)
    )

# ===============================
# 初始化输出
# ===============================

protac_df['e3_ligase_Uniprot'] = None
protac_df['linker_smiles'] = None
protac_df['e3_ligand_smiles'] = None
protac_df['warhead_smiles'] = None
protac_df['split_failed_reason'] = None

def remove_anchor(pmol):
    emol = Chem.RWMol(Chem.Mol(pmol))
    atoms_num = len(emol.GetAtoms())
    for idx in reversed(list(range(atoms_num))):
        if emol.GetAtomWithIdx(idx).GetAtomicNum() == 0:
            emol.RemoveAtom(idx)
    mol = emol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except:
        return None
    return mol




# ===============================
# PROTAC 拆分函数
# ===============================

def split_protac(pmol, e3_name):
    # 获取原子数量
    natoms = pmol.GetNumAtoms()
    for linker_mol, linker_smiles in linker_list:
        # 在 PROTAC 中寻找 linker,返回 linker 匹配到的 原子索引
        matches = pmol.GetSubstructMatches(linker_mol)
        # 没有匹配则跳过
        if not matches:
            continue
        # 遍历所有匹配。原因：有时候 linker 在分子中 可能匹配多个位置。
        for match in matches:
            # 建立 linker mask
            mask = np.zeros(natoms, dtype=bool)
            mask[list(match)] = True
            # 生成 linker 分子
            linker_frag = mol_from_mask(pmol, mask)

            if linker_frag is None:
                continue

            # 取反：
            remaining_mask = ~mask

            # 生成剩余分子
            remaining_frag = mol_from_mask(pmol, remaining_mask)

            if remaining_frag is None:
                continue

            # 拆分 fragment。这个函数会把 不连通的分子拆开。
            frags = Chem.GetMolFrags(remaining_frag, asMols=True)
            # fragment 数量必须 =2
            if len(frags) != 2:
                continue

            # 获取该 ligase 的候选 E3 ligand
            e3_candidates = e3_dict.get(e3_name, [])
            # 遍历 fragment
            for frag in frags:
                new_frag = remove_anchor(frag)
                if new_frag is None:
                    break
                # 遍历所有 E3 ligand
                for e3_mol, e3_smiles, e3_ligase_Uniprot in e3_candidates:
                    # 判断 fragment 是否包含 E3 ligand
                    if new_frag.HasSubstructMatch(e3_mol) and e3_mol.HasSubstructMatch(new_frag):
                        # 找到 warhead
                        warhead_frag = [f for f in frags if f is not frag]

                        if not warhead_frag:
                            continue
                        # 取 warhead
                        warhead = warhead_frag[0]
                        # 返回结果
                        return (
                            Chem.MolToSmiles(linker_frag),
                            Chem.MolToSmiles(frag),
                            e3_ligase_Uniprot,
                            Chem.MolToSmiles(warhead)
                        )
    return None, None, None, None


# ===============================
# 主循环
# ===============================
# 逐行遍历 DataFrame
for i, row in tqdm(protac_df.iterrows(), total=len(protac_df)):

    pmol = row.mol

    if pmol is None:
        protac_df.at[i, 'split_failed_reason'] = 'invalid_smiles'
        continue

    linker, e3, e3_ligase_Uniprot, warhead = split_protac(pmol, row['E3 ligase'])

    if linker:
        protac_df.at[i, 'e3_ligase_Uniprot'] = e3_ligase_Uniprot
        protac_df.at[i, 'linker_smiles'] = linker
        protac_df.at[i, 'e3_ligand_smiles'] = e3
        protac_df.at[i, 'warhead_smiles'] = warhead
    else:
        protac_df.at[i, 'split_failed_reason'] = 'split_failed'


# ===============================
# 保存
# ===============================

protac_df = protac_df.dropna(subset=['e3_ligase_Uniprot', 'linker_smiles', 'e3_ligand_smiles', 'warhead_smiles', 'mol'])
protac_df.drop(columns=['mol'], inplace=True)

protac_df.to_csv(
    './datasets/protac_split_result_clean.csv',
    index=False
)

print("PROTAC拆分完成")