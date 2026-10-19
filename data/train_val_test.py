import itertools
import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem, Geometry, DataStructs
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from tqdm import tqdm
import json
# 从输入表格中提取所有唯一的分子 SMILES，并保存为一个列表文件
def create_unique_smiles(input_path):
    smiles = pd.read_csv(input_path)["Smiles"].dropna().unique().tolist()
    return smiles

# 计算两个构象的 RMSD：计算同一个分子的两个构象之间的 RMSD
def getRMS(mol, c1, c2):
    (rms, trans) = AllChem.GetAlignmentTransform(mol, mol, c1, c2)
    return rms

# 把一批 SMILES 分子生成多个 3D 构象（conformers），做能量优化和去重筛选，然后写入 SDF 文件。
def compute_confs_worker(
        smifile,          # SMILES列表
        sdffile,          # 输出 SDF 文件
        maxconfs=50,      # 每个分子默认最大构象数
        sample_mult=1,
        seed=9162006,
        rms_threshold=0.5,
        energy=10.0,      # 能量窗口，保留原参数名
        mmff=True,        # 默认优先使用 MMFF
        nomin=False,
        etkdg=True,       # 默认使用 ETKDG
        jpsettings=True,  # 默认按可旋转键动态增加构象数
):
    print("Worker started")

    outf = open(sdffile, "w+")
    sdwriter = Chem.SDWriter(outf)

    if sdwriter is None:
        print("Could not open ", sdffile)
        sys.exit(-1)

    # 严格数据集：写 SDF 时使用 kekulized 表示
    sdwriter.SetKekulize(True)

    n_total = 0
    n_mol_none = 0
    n_initial_sanitize_fail = 0
    n_embed_fail = 0
    n_opt_fail = 0
    n_remove_h_fail = 0
    n_final_sanitize_fail = 0
    n_written = 0

    generator = tqdm(enumerate(smifile), total=len(smifile))

    for count, smi in generator:
        n_total += 1
        name = smi

        try:
            # ------------------------------------------------
            # 1. 多组分 SMILES：取最大组分
            # ------------------------------------------------
            pieces = str(smi).split(".")
            if len(pieces) > 1:
                smi = max(pieces, key=len)
                print(f"Taking largest component: {smi}")

            # ------------------------------------------------
            # 2. SMILES -> mol
            # ------------------------------------------------
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                n_mol_none += 1
                print(f"[MOL_FROM_SMILES_FAIL] {name}")
                continue

            # ------------------------------------------------
            # 3. 初始 full sanitize
            #    这里失败，说明 SMILES 本身就不适合严格数据集
            # ------------------------------------------------
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                n_initial_sanitize_fail += 1
                print(f"[INITIAL_SANITIZE_FAIL] {name} | {repr(e)}")
                continue

            mol.SetProp("_Name", name)

            # ------------------------------------------------
            # 4. 加氢后生成 3D 构象
            # ------------------------------------------------
            mol_h = Chem.AddHs(mol)

            rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)

            # 不要直接修改 maxconfs，否则会影响后续分子
            if jpsettings:
                if rot_bonds <= 7:
                    n_confs = max(50, maxconfs)
                elif rot_bonds <= 12:
                    n_confs = max(200, maxconfs)
                else:
                    n_confs = max(300, maxconfs)
            else:
                n_confs = maxconfs

            n_confs = int(sample_mult * n_confs)

            # ------------------------------------------------
            # 5. ETKDGv3 构象生成
            # ------------------------------------------------
            if etkdg:
                try:
                    params = AllChem.ETKDGv3()
                except Exception:
                    params = AllChem.ETKDG()

                params.randomSeed = int(seed)
                params.numThreads = 0
                params.pruneRmsThresh = float(rms_threshold)

                # 对小环/大环更稳，部分 RDKit 版本可能没有这些属性
                if hasattr(params, "useSmallRingTorsions"):
                    params.useSmallRingTorsions = True
                if hasattr(params, "useMacrocycleTorsions"):
                    params.useMacrocycleTorsions = True
                if hasattr(params, "useMacrocycle14config"):
                    params.useMacrocycle14config = True

                cids = list(AllChem.EmbedMultipleConfs(
                    mol_h,
                    numConfs=n_confs,
                    params=params,
                ))
            else:
                cids = list(AllChem.EmbedMultipleConfs(
                    mol_h,
                    numConfs=n_confs,
                    randomSeed=int(seed),
                    numThreads=0,
                    pruneRmsThresh=float(rms_threshold),
                ))

            if len(cids) == 0:
                n_embed_fail += 1
                print(f"[EMBED_FAIL] {name}")
                continue

            # ------------------------------------------------
            # 6. 优化构象
            #    优先 MMFF，MMFF 参数不全时退回 UFF
            # ------------------------------------------------
            energy_by_cid = {}
            converged_by_cid = {}
            used_force_field = None

            conf_ids = [conf.GetId() for conf in mol_h.GetConformers()]

            if mmff and AllChem.MMFFHasAllMoleculeParams(mol_h):
                try:
                    results = AllChem.MMFFOptimizeMoleculeConfs(
                        mol_h,
                        numThreads=0,
                        maxIters=1000,
                    )

                    used_force_field = "MMFF"

                    for cid, result in zip(conf_ids, results):
                        not_converged, e = result
                        energy_by_cid[int(cid)] = float(e)
                        converged_by_cid[int(cid)] = (int(not_converged) == 0)

                except Exception as e:
                    print(f"[MMFF_OPT_FAIL] {name} | {repr(e)}")
                    energy_by_cid = {}
                    converged_by_cid = {}

            if len(energy_by_cid) == 0:
                try:
                    results = AllChem.UFFOptimizeMoleculeConfs(
                        mol_h,
                        numThreads=0,
                        maxIters=1000,
                    )

                    used_force_field = "UFF"

                    for cid, result in zip(conf_ids, results):
                        not_converged, e = result
                        energy_by_cid[int(cid)] = float(e)
                        converged_by_cid[int(cid)] = (int(not_converged) == 0)

                except Exception as e:
                    n_opt_fail += 1
                    print(f"[UFF_OPT_FAIL] {name} | {repr(e)}")
                    continue

            if len(energy_by_cid) == 0:
                n_opt_fail += 1
                print(f"[NO_ENERGY_AFTER_OPT] {name}")
                continue

            # ------------------------------------------------
            # 7. 去氢
            # ------------------------------------------------
            try:
                mol_no_h = Chem.RemoveHs(
                    mol_h,
                    updateExplicitCount=True,
                    sanitize=True,
                )
            except Exception as e:
                n_remove_h_fail += 1
                print(f"[REMOVE_H_FAIL] {name} | {repr(e)}")
                continue

            # RemoveHs 后再次 full sanitize
            try:
                mol_check = Chem.Mol(mol_no_h)
                Chem.SanitizeMol(mol_check)
            except Exception as e:
                n_remove_h_fail += 1
                print(f"[AFTER_REMOVE_H_FULL_SANITIZE_FAIL] {name} | {repr(e)}")
                continue

            # ------------------------------------------------
            # 8. 选择最低能量构象
            #    优先选择收敛构象；如果都不收敛，再从全部构象中选最低能量
            # ------------------------------------------------
            final_conf_ids = [conf.GetId() for conf in mol_no_h.GetConformers()]

            valid_cids = [
                int(cid)
                for cid in final_conf_ids
                if int(cid) in energy_by_cid
            ]

            if len(valid_cids) == 0:
                print(f"[NO_VALID_CONF_ID_AFTER_REMOVE_H] {name}")
                continue

            converged_cids = [
                cid for cid in valid_cids
                if converged_by_cid.get(cid, False)
            ]

            if len(converged_cids) > 0:
                candidate_cids = converged_cids
            else:
                candidate_cids = valid_cids

            min_energy = min(energy_by_cid[cid] for cid in candidate_cids)

            # 能量窗口内候选
            window_cids = [
                cid for cid in candidate_cids
                if energy_by_cid[cid] <= min_energy + float(energy)
            ]

            best_cid = sorted(
                window_cids,
                key=lambda cid: energy_by_cid[cid],
            )[0]

            # ------------------------------------------------
            # 9. 写入前最终 full sanitize
            # ------------------------------------------------
            mol_to_write = Chem.Mol(mol_no_h)

            try:
                Chem.SanitizeMol(mol_to_write)
            except Exception as e:
                n_final_sanitize_fail += 1
                print(f"[FINAL_SANITIZE_FAIL_BEFORE_WRITE] {name} | {repr(e)}")
                continue

            mol_to_write.SetProp("_Name", name)
            mol_to_write.SetProp("_Energy", str(energy_by_cid[best_cid]))
            mol_to_write.SetProp("_ForceField", used_force_field)
            mol_to_write.SetProp("_Converged", str(converged_by_cid.get(best_cid, False)))
            mol_to_write.SetProp("_NumConfs", str(len(cids)))
            mol_to_write.SetProp("_RotatableBonds", str(rot_bonds))

            sdwriter.write(mol_to_write, confId=int(best_cid))
            n_written += 1

        except (KeyboardInterrupt, SystemExit):
            raise

        except Exception as e:
            print(f"[UNEXPECTED_EXCEPTION] {name} | {repr(e)}")
            continue

    sdwriter.close()
    outf.close()

    print("\n========== compute_confs_worker summary ==========")
    print(f"total smiles:                       {n_total}")
    print(f"mol from smiles fail:               {n_mol_none}")
    print(f"initial sanitize fail:              {n_initial_sanitize_fail}")
    print(f"embed fail:                         {n_embed_fail}")
    print(f"optimization fail:                  {n_opt_fail}")
    print(f"remove H / after RemoveHs fail:     {n_remove_h_fail}")
    print(f"final sanitize fail before write:   {n_final_sanitize_fail}")
    print(f"written molecules:                  {n_written}")
    print("=================================================\n")

    return None

# 找到所有 exit atom（符号为 * 的原子）
def get_exits(mol):

    exits = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == '*':
            exits.append(atom)
    return exits

# 标记锚点原子（一个锚点原子被标记）
def set_anchor_flags(mol, anchor_idx):

    for atom in mol.GetAtoms():
        if atom.GetIdx() == anchor_idx:
            atom.SetProp('_Anchor', '1')
        else:
            atom.SetProp('_Anchor', '0')

# 读取 分子中所有锚点原子 的 索引
def get_anchors_idx(mol):
    anchors_idx = []
    for atom in mol.GetAtoms():
        if atom.GetProp('_Anchor') == '1':
            anchors_idx.append(atom.GetIdx())

    return anchors_idx

# 找到exit原子，删除exit原子和对应的键，将exit原子连接的另一个原子标记为 _Anchor
def update_fragment(frag):

    # 找 exit atom
    exits = get_exits(frag)
    # 如果片段有多个exit原子 → 抛异常
    if len(exits) > 1:
        raise Exception('Found more than one exits in fragment')
    exit = exits[0]

    bonds = exit.GetBonds()
    # exit 原子只能有 1 条键，否则抛异常
    if len(bonds) > 1:
        raise Exception('Exit atom has more than 1 bond')
    bond = bonds[0]

    exit_idx = exit.GetIdx()
    # 找 exit atom 的连接原子（锚点）
    source_idx = bond.GetBeginAtomIdx()
    target_idx = bond.GetEndAtomIdx()
    anchor_idx = source_idx if target_idx == exit_idx else target_idx
    # 设置 anchor flag
    set_anchor_flags(frag, anchor_idx)

    # 删除 exit atom
    efragment = Chem.EditableMol(frag)
    efragment.RemoveBond(source_idx, target_idx)
    efragment.RemoveAtom(exit_idx)

    return efragment.GetMol()

# 处理 linker：找两个 exit atom，删除 exit bonds， 删除 exit atoms
def update_linker(linker):

    exits = get_exits(linker)
    # 最多 两个exit 原子
    if len(exits) > 2:
        raise Exception('Found more than two exits in linker')

    # 按原子索引降序排序
    exits = sorted(exits, key=lambda e: e.GetIdx(), reverse=True)
    elinker = Chem.EditableMol(linker)

    # 删除exit原子对应的键
    for exit in exits:
        bonds = exit.GetBonds()
        # exit 原子只能有 1 条键
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        elinker.RemoveBond(source_idx, target_idx)

    # 删除exit原子
    for exit in exits:
        elinker.RemoveAtom(exit.GetIdx())

    # 返回处理后的 linker
    return elinker.GetMol()

# 根据给定的三维坐标创建一个 RDKit 的分子构象（conformer）。这个函数 只创建构象，不会自动加入分子。如果要加到分子中，需要：mol.AddConformer(conformer)
def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

# 把一个大分子 mol 的 3D 坐标转移到子结构 frag 上，为 fragment 创建对应的 conformer。
def transfer_conformers(frag, mol):

    # 找出 frag 在 mol 中所有可能的位置。一个match是一种可能的位置
    matches = mol.GetSubstructMatches(frag)
    # 如果没有匹配就报错
    if len(matches) < 1:
        raise Exception('Could not find fragment or linker matches')
    # 创建字典存储结果
    match2conf = {}
    # 遍历所有匹配
    for match in matches:
        # 获取 mol 的坐标
        mol_coords = mol.GetConformer().GetPositions()
        # 提取 fragment 对应坐标
        frag_coords = mol_coords[np.array(match)]
        # 创建 fragment 的 conformer
        frag_conformer = create_conformer(frag_coords)
        match2conf[match] = frag_conformer
    # 返回所有匹配结果
    return match2conf
# 从三组子结构匹配结果中，找到彼此没有原子重叠的组合。
def find_non_intersecting_matches(matches1, matches2, matches3):
    # itertools.product 是 笛卡尔积。得到所有可能组合。
    triplets = list(itertools.product(matches1, matches2, matches3))
    # 创建结果集合
    non_intersecting_matches = set()
    # 遍历每个组合
    for m1, m2, m3 in triplets:
        m1m2 = set(m1) & set(m2)
        m1m3 = set(m1) & set(m3)
        m2m3 = set(m2) & set(m3)
        # 判断是否完全不重叠
        if len(m1m2 | m1m3 | m2m3) == 0:
            # 保存合法组合
            non_intersecting_matches.add((m1, m2, m3))
    return list(non_intersecting_matches)

# 确保 linker 在两个 fragment 中间
def find_matches_with_linker_in_the_middle(non_intersecting_matches, mol):

    matches_with_linker_in_the_middle = []
    for m1, m2, lm in non_intersecting_matches:
        neighbors = set()
        for atom_idx in lm:
            atom_neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            for neighbor in atom_neighbors:
                neighbors.add(neighbor.GetIdx())

        conn1 = set(m1) & neighbors
        conn2 = set(m2) & neighbors
        if len(conn1) == 1 and len(conn2) == 1:
            matches_with_linker_in_the_middle.append((m1, m2, lm))

    return matches_with_linker_in_the_middle
def atom_has_bond_to_any(mol, atom_idx, atom_idx_set):
    """
    判断 full mol 中 atom_idx 是否和 atom_idx_set 中任意原子有键。
    """
    atom_idx_set = set(atom_idx_set)

    atom = mol.GetAtomWithIdx(int(atom_idx))

    bonded_neighbors = []
    for nbr in atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in atom_idx_set:
            bonded_neighbors.append(nbr_idx)

    return bonded_neighbors


def find_matches_with_anchor_connected(
        non_intersecting_matches,
        mol,
        frag1_anchor_idx,
        frag2_anchor_idx,
):
    """
    只保留真正满足：
        fragment1 的 anchor 原子连接 linker
        fragment2 的 anchor 原子连接 linker
    的匹配组合。

    frag1_anchor_idx / frag2_anchor_idx 是 fragment 内部 index，
    不是 full mol index。
    """

    good_matches = []

    for m1, m2, lm in non_intersecting_matches:
        # fragment 内部 anchor index -> full mol atom index
        full_a1 = m1[int(frag1_anchor_idx)]
        full_a2 = m2[int(frag2_anchor_idx)]

        a1_linker_neighbors = atom_has_bond_to_any(
            mol,
            full_a1,
            lm,
        )

        a2_linker_neighbors = atom_has_bond_to_any(
            mol,
            full_a2,
            lm,
        )

        if len(a1_linker_neighbors) == 1 and len(a2_linker_neighbors) == 1:
            good_matches.append((m1, m2, lm))

    return good_matches

# 在 fragment1、fragment2 和 linker 的所有匹配中，找到真正正确的 fragment-linker-fragment 组合。
def find_correct_matches(
        matches_frag1,
        matches_frag2,
        matches_linker,
        mol,
        frag1_anchor_idx=None,
        frag2_anchor_idx=None,
):
    non_intersecting_matches = find_non_intersecting_matches(
        matches_frag1,
        matches_frag2,
        matches_linker,
    )

    if len(non_intersecting_matches) == 0:
        return []

    # 如果提供了 anchor index，优先使用 anchor-aware matching
    if frag1_anchor_idx is not None and frag2_anchor_idx is not None:
        anchor_matches = find_matches_with_anchor_connected(
            non_intersecting_matches=non_intersecting_matches,
            mol=mol,
            frag1_anchor_idx=frag1_anchor_idx,
            frag2_anchor_idx=frag2_anchor_idx,
        )

        if len(anchor_matches) > 0:
            return anchor_matches

        # 如果 anchor-aware 一个都找不到，说明这个样本拆分/匹配有问题
        return []
    else:
        print("**************************************************")

    # # 兼容旧逻辑
    # if len(non_intersecting_matches) == 1:
    #     return non_intersecting_matches
    #
    # return find_matches_with_linker_in_the_middle(
    #     non_intersecting_matches,
    #     mol,
    # )

# 根据完整分子的 3D 构象，为 fragment 和 linker 创建对应的 3D conformer，并设置锚点信息。
def prepare_fragments_and_linker(warhead_smi, e3_ligand_smi, linker_smi, mol):
    warhead = Chem.MolFromSmiles(warhead_smi)
    e3_ligand = Chem.MolFromSmiles(e3_ligand_smi)
    linker = Chem.MolFromSmiles(linker_smi)

    new_warhead = update_fragment(warhead)
    new_e3_ligand = update_fragment(e3_ligand)
    newlinker = update_linker(linker)

    warhead_anchor_idx = get_anchors_idx(new_warhead)
    e3_anchor_idx = get_anchors_idx(new_e3_ligand)
    if len(warhead_anchor_idx) != 1:
        raise Exception(f"found {len(warhead_anchor_idx)} warhead anchors")
    if len(e3_anchor_idx) != 1:
        raise Exception(f"found {len(e3_anchor_idx)} e3 anchors")
    warhead_anchor_idx = warhead_anchor_idx[0]
    e3_anchor_idx = e3_anchor_idx[0]

    match2conf_warhead = transfer_conformers(new_warhead, mol)
    match2conf_e3_ligand = transfer_conformers(new_e3_ligand, mol)
    match2conf_linker = transfer_conformers(newlinker, mol)

    correct_matches = find_correct_matches(
        match2conf_warhead.keys(),
        match2conf_e3_ligand.keys(),
        match2conf_linker.keys(),
        mol,
        frag1_anchor_idx=warhead_anchor_idx,
        frag2_anchor_idx=e3_anchor_idx,
    )

    if len(correct_matches) < 1:
        raise Exception(
            "Could not find anchor-aware fragment-linker-fragment matches"
        )

    if len(correct_matches) > 2:
        raise Exception("Found more than two fragment matches")

    correct_match = correct_matches[0]

    warhead_match, e3_match, linker_match = correct_match

    conf_warhead = match2conf_warhead[warhead_match]
    conf_e3_ligand = match2conf_e3_ligand[e3_match]
    conf_linker = match2conf_linker[linker_match]

    new_warhead.AddConformer(conf_warhead)
    new_e3_ligand.AddConformer(conf_e3_ligand)
    newlinker.AddConformer(conf_linker)

    return new_warhead, new_e3_ligand, newlinker, correct_match

# 读取 SDF 分子文件，根据 table 中给出的 fragment/linker 切分信息，生成带 3D 构象的 fragment 和 linker 数据集。
def process_sdf(sdf_path, table, progress=True):
    # 读取分子
    supplier = Chem.SDMolSupplier(
        sdf_path,
        sanitize=False,
        removeHs=False,
    )

    molecules = []
    fragments = []
    linkers = []
    out_table = []
    uuid = 0
    # 进度条
    supplier = tqdm(supplier, total=len(supplier)) if progress else supplier
    # 遍历每个分子
    for mol in supplier:
        if mol is None:
            continue

        # ------------------------------------------------
        # 1. 先检查 no-kekulize sanitize
        # ------------------------------------------------
        try:
            mol_partial_check = Chem.Mol(mol)
            Chem.SanitizeMol(
                mol_partial_check,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
        except Exception as e:
            print(f"[PROCESS_SDF_PARTIAL_SANITIZE_FAIL] {repr(e)}")
            continue

        # ------------------------------------------------
        # 2. 再检查 full sanitize，包括 Kekulize
        #    这是过滤 KekulizeException 分子的关键
        # ------------------------------------------------
        try:
            mol_full_check = Chem.Mol(mol)
            Chem.SanitizeMol(mol_full_check)
        except Exception as e:
            mol_name_for_log = mol.GetProp("_Name") if mol.HasProp("_Name") else "UNKNOWN"
            print(
                f"[PROCESS_SDF_FULL_SANITIZE_FAIL] "
                f"name={mol_name_for_log} | {repr(e)}"
            )
            continue

        # ------------------------------------------------
        # 3. 检查 conformer
        # ------------------------------------------------
        if mol.GetNumConformers() == 0:
            continue

        # ------------------------------------------------
        # 4. 正式 full sanitize 当前 mol
        # ------------------------------------------------
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"[PROCESS_SDF_FINAL_SANITIZE_FAIL] {repr(e)}")
            continue

        # 读取分子名字
        mol_name = mol.GetProp('_Name')
        # 取 SMILES
        mol_smi = Chem.MolToSmiles(mol)
        # 把 _Name 改成 SMILES
        mol.SetProp('_Name', mol_smi)

        # 找对应 linker + fragments
        for warhead_smi, e3_ligand_smi, linker_smi, uniprot, e3_ligase_uniprot, Molecular_Weight, Exact_Mass, XLogP3, Heavy_Atom_Count, Ring_Count, Hydrogen_Bond_Acceptor_Count, Hydrogen_Bond_Donor_Count, Rotatable_Bond_Count, Topological_Polar_Surface_Area in table[table.Smiles == mol_name][['warhead_smiles', 'e3_ligand_smiles', 'linker_smiles', 'Uniprot', 'e3_ligase_Uniprot', 'Molecular Weight', 'Exact Mass', 'XLogP3', 'Heavy Atom Count', 'Ring Count', 'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count', 'Rotatable Bond Count', 'Topological Polar Surface Area']].values:
            try:
                # 生成 fragment + linker(带3D)
                warhead, e3_ligand, linker, correct_match = prepare_fragments_and_linker(
                    warhead_smi,
                    e3_ligand_smi,
                    linker_smi,
                    mol,
                )
            except Exception as e:
                print(f'{mol_smi} | {linker_smi} | {warhead_smi} | {e3_ligand_smi} : {e}')
                continue

            try:
                Chem.SanitizeMol(warhead)
                Chem.SanitizeMol(e3_ligand)
                Chem.SanitizeMol(linker)
            except Exception as e:
                print(
                    f"[FRAGMENT_LINKER_SANITIZE_FAIL] "
                    f"{mol_smi} | {linker_smi} | {warhead_smi} | {e3_ligand_smi} | {repr(e)}"
                )
                continue


            # 合并两个 fragment。顺序是 warhead 在前，e3_ligand 在后
            n_warhead_atoms = warhead.GetNumAtoms()
            n_e3_ligand_atoms = e3_ligand.GetNumAtoms()
            frags = Chem.CombineMols(warhead, e3_ligand)

            try:
                Chem.SanitizeMol(frags)
            except Exception as e:
                print(f"[COMBINED_FRAGMENTS_SANITIZE_FAIL] {mol_smi} | {repr(e)}")
                continue

            warhead_match, e3_match, linker_match = correct_match

            dataset_to_full_atom_idx = (
                    [int(x) for x in warhead_match]
                    + [int(x) for x in e3_match]
                    + [int(x) for x in linker_match]
            )

            expected_num_atoms = (
                    warhead.GetNumAtoms()
                    + e3_ligand.GetNumAtoms()
                    + linker.GetNumAtoms()
            )

            if len(dataset_to_full_atom_idx) != expected_num_atoms:
                print(
                    f"{mol_smi} | atom mapping length mismatch: "
                    f"mapping={len(dataset_to_full_atom_idx)}, "
                    f"expected={expected_num_atoms}"
                )
                continue

            # 分别读取两个 fragment 内部的 anchor
            warhead_anchor_idx = get_anchors_idx(warhead)
            e3_anchor_idx = get_anchors_idx(e3_ligand)
            if len(warhead_anchor_idx) != 1:
                print(f'{mol_smi} | {linker_smi} | {warhead_smi} | {e3_ligand_smi} : '
                      f'found {len(warhead_anchor_idx)} warhead anchors')
                continue
            if len(e3_anchor_idx) != 1:
                print(f'{mol_smi} | {linker_smi} | {warhead_smi} | {e3_ligand_smi} : '
                      f'found {len(e3_anchor_idx)} e3 anchors')
                continue
            # warhead anchor 是全局 index
            warhead_anchor_global = warhead_anchor_idx[0]
            # e3 anchor 在 CombineMols 后需要加 warhead 原子数偏移
            e3_anchor_global = n_warhead_atoms + e3_anchor_idx[0]
            anchors_idx = [warhead_anchor_global, e3_anchor_global]



            # 保存数据
            molecules.append(mol)
            fragments.append(frags)
            linkers.append(linker)
            # 保存 metadata。生成一条数据记录
            out_table.append({
                'uuid': uuid,
                'molecule': mol_smi,
                'fragments': Chem.MolToSmiles(frags),
                'linker': Chem.MolToSmiles(linker),
                'full_atom_indices': json.dumps(dataset_to_full_atom_idx),
                'anchor_1': anchors_idx[0],
                'anchor_2': anchors_idx[1],

                # 新增字段，后面 preprocess 用来构造 fragment1 / fragment2 mask
                'warhead_num_atoms': n_warhead_atoms,
                'e3_ligand_num_atoms': n_e3_ligand_atoms,
                'warhead_anchor': warhead_anchor_global,
                'e3_ligand_anchor': e3_anchor_global,

                'energy': mol.GetProp('_Energy'),
                'Uniprot': uniprot,
                'e3_ligase_Uniprot': e3_ligase_uniprot,
                'Molecular Weight': Molecular_Weight,
                'Exact Mass': Exact_Mass,
                'XLogP3': XLogP3,
                'Heavy Atom Count': Heavy_Atom_Count,
                'Ring Count': Ring_Count,
                'Hydrogen Bond Acceptor Count': Hydrogen_Bond_Acceptor_Count,
                'Hydrogen Bond Donor Count': Hydrogen_Bond_Donor_Count,
                'Rotatable Bond Count': Rotatable_Bond_Count,
                'Topological Polar Surface Area': Topological_Polar_Surface_Area,
            })
            uuid += 1

    return molecules, fragments, linkers, pd.DataFrame(out_table)

def split_protac_dataset_for_pretrain_and_rl(
    df,
    train_ratio=0.8,
    val_ratio=0.1,
    out_prefix="./datasets",
    seed=42,
    protac_col="Smiles",
    linker_col="linker_smiles",
    similarity_threshold=0.7,
    radius=2,
    n_bits=2048,
    use_local_search=True,
):
    """
    按 linker 相似性聚类划分 PROTAC 数据集。

    适用于：
    1. 预训练阶段：按唯一 PROTAC 去重后的生成数据
    2. RL 阶段：保留 protein context 的一对多数据

    核心原则：
    1. 相同 linker 不能跨 train / val / test
    2. 相似 linker cluster 不能跨 train / val / test
    3. 同一个 PROTAC 的所有 protein context 必须进入同一个 split
    4. train / val / test 尽量接近 8:1:1，但不强行拆 cluster

    参数：
    - similarity_threshold:
        Tanimoto 相似度阈值。
        例如 0.7 表示相似度 >= 0.7 的 linker 尽量聚到同一 cluster。
    - use_local_search:
        是否在 greedy 分配后做局部搜索，使比例更接近 8:1:1。
    """

    df = df.copy()

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio 必须小于 1。")

    # =========================
    # 1. SMILES 标准化
    # =========================
    def canonicalize_smiles(smiles):
        if pd.isna(smiles):
            return None

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        return Chem.MolToSmiles(mol, canonical=True)

    def remove_dummy_atoms_and_canonicalize(smiles):
        """
        linker_smiles 中通常含有 * 作为连接点。
        做 linker 相似性聚类时，建议去掉 *，只比较 linker 主体。
        """
        if pd.isna(smiles):
            return None

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        dummy_idxs = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 0
        ]

        if len(dummy_idxs) > 0:
            emol = Chem.EditableMol(mol)
            for idx in sorted(dummy_idxs, reverse=True):
                emol.RemoveAtom(idx)
            mol = emol.GetMol()

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        if mol.GetNumAtoms() == 0:
            return None

        return Chem.MolToSmiles(mol, canonical=True)

    df["protac_id"] = df[protac_col].apply(canonicalize_smiles)
    df["linker_id"] = df[linker_col].apply(canonicalize_smiles)
    df["linker_core_id"] = df[linker_col].apply(remove_dummy_atoms_and_canonicalize)

    before_num = len(df)

    df = df[
        df["protac_id"].notna()
        & df["linker_core_id"].notna()
    ].copy()

    after_num = len(df)

    print("========== SMILES parsing ==========")
    print(f"原始样本数: {before_num}")
    print(f"有效样本数: {after_num}")
    print(f"被过滤样本数: {before_num - after_num}")

    if len(df) == 0:
        raise ValueError("SMILES 标准化后没有有效样本，请检查输入表格。")

    # =========================
    # 2. 计算 linker scaffold
    # =========================
    def get_linker_scaffold(smiles):
        if pd.isna(smiles):
            return None

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)

        # 对于 PEG / alkyl 这类无环 linker，Murcko scaffold 可能为空
        # 此时退回 linker_core_id 本身
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return Chem.MolToSmiles(mol, canonical=True)

        return Chem.MolToSmiles(scaffold, canonical=True)

    df["linker_scaffold"] = df["linker_core_id"].apply(get_linker_scaffold)

    # =========================
    # 3. 唯一 linker 计算 Morgan fingerprint
    # =========================
    unique_linkers = sorted(
        df["linker_core_id"]
        .dropna()
        .unique()
        .tolist()
    )

    linker_fps = []
    valid_linkers = []

    for smi in unique_linkers:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            radius,
            nBits=n_bits
        )

        linker_fps.append(fp)
        valid_linkers.append(smi)

    if len(valid_linkers) == 0:
        raise ValueError("没有可用于划分的有效 linker。")

    # =========================
    # 4. Butina 聚类
    # =========================
    # Butina 使用距离矩阵，不是相似度矩阵
    # distance = 1 - Tanimoto similarity
    distances = []
    n_linkers = len(valid_linkers)

    for i in range(1, n_linkers):
        sims = DataStructs.BulkTanimotoSimilarity(
            linker_fps[i],
            linker_fps[:i]
        )
        distances.extend([1.0 - s for s in sims])

    distance_threshold = 1.0 - similarity_threshold

    clusters = Butina.ClusterData(
        distances,
        n_linkers,
        distance_threshold,
        isDistData=True
    )

    linker_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for linker_idx in cluster:
            linker_smi = valid_linkers[linker_idx]
            linker_to_cluster[linker_smi] = cluster_id

    df["linker_cluster"] = df["linker_core_id"].map(linker_to_cluster)

    if df["linker_cluster"].isna().any():
        raise ValueError("部分 linker 没有被分配到 cluster，请检查 fingerprint 或 linker_smiles。")

    # =========================
    # 5. 检查一个 PROTAC 是否对应多个 linker cluster
    # =========================
    protac_cluster_count = (
        df.groupby("protac_id")["linker_cluster"]
        .nunique()
    )

    bad_protacs = protac_cluster_count[protac_cluster_count > 1]

    if len(bad_protacs) > 0:
        raise ValueError(
            f"发现 {len(bad_protacs)} 个 protac_id 对应多个 linker_cluster，"
            f"说明同一个 PROTAC 的 linker_smiles 可能不一致，请先检查数据。"
        )

    # =========================
    # 6. 按 linker cluster 划分 train / val / test
    # =========================
    unique_protac_table = (
        df[["protac_id", "linker_cluster"]]
        .drop_duplicates(subset=["protac_id"])
        .reset_index(drop=True)
    )

    cluster_sizes = (
        unique_protac_table
        .groupby("linker_cluster")["protac_id"]
        .nunique()
        .to_dict()
    )

    n_total_unique_protac = len(unique_protac_table)

    target_counts = {
        "train": train_ratio * n_total_unique_protac,
        "val": val_ratio * n_total_unique_protac,
        "test": test_ratio * n_total_unique_protac,
    }

    rng = np.random.default_rng(seed)

    cluster_items = []
    for cid, size in cluster_sizes.items():
        cluster_items.append((cid, size, rng.random()))

    # 大 cluster 先分，避免最后大 cluster 导致比例严重失衡
    cluster_items = sorted(
        cluster_items,
        key=lambda x: (-x[1], x[2])
    )

    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    cluster_to_split = {}

    def split_loss(counts, targets):
        """
        当前划分和目标 8:1:1 的偏差。
        使用平方误差，偏差大的 split 惩罚更重。
        """
        loss = 0.0
        for split in ["train", "val", "test"]:
            loss += (counts[split] - targets[split]) ** 2
        return loss

    # ---------- 6.1 Greedy 分配 ----------
    for cid, size, _ in cluster_items:
        best_split = None
        best_loss = None

        # 尝试把当前 cluster 分别放入 train / val / test
        # 选择整体偏差最小的那个 split
        for split in ["train", "val", "test"]:
            temp_counts = split_counts.copy()
            temp_counts[split] += size

            loss = split_loss(temp_counts, target_counts)

            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_split = split

        cluster_to_split[cid] = best_split
        split_counts[best_split] += size

    # ---------- 6.2 可选：局部搜索微调 ----------
    if use_local_search:
        improved = True
        max_iter = 1000
        n_iter = 0

        while improved and n_iter < max_iter:
            improved = False
            n_iter += 1

            current_loss = split_loss(split_counts, target_counts)

            for cid, size in cluster_sizes.items():
                old_split = cluster_to_split[cid]

                for new_split in ["train", "val", "test"]:
                    if new_split == old_split:
                        continue

                    temp_counts = split_counts.copy()
                    temp_counts[old_split] -= size
                    temp_counts[new_split] += size

                    new_loss = split_loss(temp_counts, target_counts)

                    if new_loss < current_loss:
                        cluster_to_split[cid] = new_split
                        split_counts = temp_counts
                        current_loss = new_loss
                        improved = True
                        break

                if improved:
                    break

        print(f"局部搜索迭代次数: {n_iter}")

    df["split"] = df["linker_cluster"].map(cluster_to_split)

    # =========================
    # 7. 构建 RL 阶段数据
    # =========================
    train_rl_df = df[df["split"] == "train"].copy()
    val_rl_df = df[df["split"] == "val"].copy()
    test_rl_df = df[df["split"] == "test"].copy()

    # =========================
    # 8. 构建预训练阶段数据
    # =========================
    train_pretrain_df = (
        train_rl_df
        .drop_duplicates(subset=["protac_id"])
        .reset_index(drop=True)
    )

    val_pretrain_df = (
        val_rl_df
        .drop_duplicates(subset=["protac_id"])
        .reset_index(drop=True)
    )

    test_pretrain_df = (
        test_rl_df
        .drop_duplicates(subset=["protac_id"])
        .reset_index(drop=True)
    )

    # =========================
    # 9. 数据泄漏检查
    # =========================
    def check_no_overlap(col, name):
        train_set = set(train_rl_df[col].dropna())
        val_set = set(val_rl_df[col].dropna())
        test_set = set(test_rl_df[col].dropna())

        assert len(train_set & val_set) == 0, f"数据泄漏：train 和 val 有相同 {name}"
        assert len(train_set & test_set) == 0, f"数据泄漏：train 和 test 有相同 {name}"
        assert len(val_set & test_set) == 0, f"数据泄漏：val 和 test 有相同 {name}"

    check_no_overlap("protac_id", "PROTAC")
    check_no_overlap("linker_core_id", "linker")
    check_no_overlap("linker_cluster", "linker_cluster")

    # =========================
    # 10. 保存文件
    # =========================
    pretrain_train_dir = os.path.join(out_prefix, "pretrain", "train")
    pretrain_val_dir = os.path.join(out_prefix, "pretrain", "val")
    pretrain_test_dir = os.path.join(out_prefix, "pretrain", "test")

    rl_train_dir = os.path.join(out_prefix, "rl", "train")
    rl_val_dir = os.path.join(out_prefix, "rl", "val")
    rl_test_dir = os.path.join(out_prefix, "rl", "test")

    for d in [
        pretrain_train_dir,
        pretrain_val_dir,
        pretrain_test_dir,
        rl_train_dir,
        rl_val_dir,
        rl_test_dir,
    ]:
        os.makedirs(d, exist_ok=True)

    train_pretrain_csv = os.path.join(
        pretrain_train_dir,
        "protac_train_pretrain.csv"
    )
    val_pretrain_csv = os.path.join(
        pretrain_val_dir,
        "protac_val_pretrain.csv"
    )
    test_pretrain_csv = os.path.join(
        pretrain_test_dir,
        "protac_test_pretrain.csv"
    )

    train_rl_csv = os.path.join(
        rl_train_dir,
        "protac_train_rl.csv"
    )
    val_rl_csv = os.path.join(
        rl_val_dir,
        "protac_val_rl.csv"
    )
    test_rl_csv = os.path.join(
        rl_test_dir,
        "protac_test_rl.csv"
    )

    train_pretrain_df.to_csv(train_pretrain_csv, index=False)
    val_pretrain_df.to_csv(val_pretrain_csv, index=False)
    test_pretrain_df.to_csv(test_pretrain_csv, index=False)

    train_rl_df.to_csv(train_rl_csv, index=False)
    val_rl_df.to_csv(val_rl_csv, index=False)
    test_rl_df.to_csv(test_rl_csv, index=False)

    # 保存完整 split 表，方便之后检查
    split_table_csv = os.path.join(
        out_prefix,
        "protac_split_by_linker_cluster.csv"
    )
    df.to_csv(split_table_csv, index=False)

    # =========================
    # 11. 打印统计信息
    # =========================
    def print_ratio(name, train_df, val_df, test_df):
        n_train = len(train_df)
        n_val = len(val_df)
        n_test = len(test_df)
        n_total = n_train + n_val + n_test

        print(f"\n========== {name} ==========")
        print(f"train: {n_train} ({n_train / n_total:.4f})")
        print(f"val:   {n_val} ({n_val / n_total:.4f})")
        print(f"test:  {n_test} ({n_test / n_total:.4f})")

    print("\n========== Linker cluster split ==========")
    print(f"有效原始样本数: {len(df)}")
    print(f"唯一 PROTAC 数: {n_total_unique_protac}")
    print(f"唯一 linker 数: {len(valid_linkers)}")
    print(f"linker cluster 数: {len(clusters)}")
    print(f"Tanimoto 相似性阈值: {similarity_threshold}")

    print("\n目标 unique PROTAC 数:")
    print(f"train target: {target_counts['train']:.2f}")
    print(f"val target:   {target_counts['val']:.2f}")
    print(f"test target:  {target_counts['test']:.2f}")

    print("\n实际 unique PROTAC 数:")
    print(f"train actual: {len(train_pretrain_df)}")
    print(f"val actual:   {len(val_pretrain_df)}")
    print(f"test actual:  {len(test_pretrain_df)}")

    print_ratio(
        "Pretrain unique PROTAC ratio",
        train_pretrain_df,
        val_pretrain_df,
        test_pretrain_df,
    )

    print_ratio(
        "RL context ratio",
        train_rl_df,
        val_rl_df,
        test_rl_df,
    )

    print("\n========== Saved files ==========")
    print(f"train_pretrain: {train_pretrain_csv}")
    print(f"val_pretrain:   {val_pretrain_csv}")
    print(f"test_pretrain:  {test_pretrain_csv}")
    print(f"train_RL:       {train_rl_csv}")
    print(f"val_RL:         {val_rl_csv}")
    print(f"test_RL:        {test_rl_csv}")
    print(f"完整 split 表:  {split_table_csv}")

    return (
        train_pretrain_df,
        val_pretrain_df,
        test_pretrain_df,
        train_rl_df,
        val_rl_df,
        test_rl_df,
    )

def write_sdf_strict(mols, sdf_path, tag):
    """
    Windows 安全写 SDF。
    不使用 Chem.SDWriter(open(...))，避免文件句柄未释放。
    """
    writer = Chem.SDWriter(sdf_path)
    if writer is None:
        raise RuntimeError(f"[{tag}] Could not open {sdf_path}")

    writer.SetKekulize(True)

    try:
        for i, mol in enumerate(mols):
            try:
                writer.write(mol)
            except Exception as e:
                raise RuntimeError(
                    f"[{tag}_WRITE_FAIL] idx={i} | {repr(e)}"
                )
    finally:
        writer.close()
def run(data_dir, ori_dir="./datasets/pretrain/"):
    # csv 文件路径
    if ori_dir == "./datasets/pretrain/":
        table_path = f"{ori_dir}{data_dir}/protac_{data_dir}_pretrain.csv"
    else:
        table_path = f"{ori_dir}{data_dir}/protac_{data_dir}_rl.csv"

    # 提取唯一的分子 SMILES
    smiles = create_unique_smiles(table_path)
    sdf_file_path = f"{ori_dir}{data_dir}/rdkit_conf.sdf"

    print("csv文件路径", table_path)
    print("sdf文件路径", sdf_file_path)

    # SMILES 分子生成多个 3D 构象, 保存到 sdf 文件
    # compute_confs_worker(smifile=smiles, sdffile=sdf_file_path)
    if not os.path.exists(sdf_file_path):
        compute_confs_worker(
            smifile=smiles,
            sdffile=sdf_file_path
        )
    else:
        print(f"[SKIP] existing rdkit_conf.sdf: {sdf_file_path}")

    out_table_path = f"{ori_dir}{data_dir}/protac_processed.csv"
    out_mol_path = f"{ori_dir}{data_dir}/mol.sdf"
    out_frag_path = f"{ori_dir}{data_dir}/fragments.sdf"
    out_link_path = f"{ori_dir}{data_dir}/linker.sdf"
    # 读 csv 文件
    table = pd.read_csv(table_path)
    # 读取 SDF 分子文件，根据 table 中给出的 fragment/linker 切分信息，生成带 3D 构象的 fragment 和 linker
    molecules, fragments, linkers, out_table = process_sdf(sdf_file_path, table, progress=True)

    # ------------------------------------------------
    # 写文件前先检查四者长度是否一致
    # ------------------------------------------------
    assert len(molecules) == len(fragments) == len(linkers) == len(out_table), (
        f"length mismatch before writing: "
        f"molecules={len(molecules)}, "
        f"fragments={len(fragments)}, "
        f"linkers={len(linkers)}, "
        f"table={len(out_table)}"
    )

    tmp_table_path = out_table_path + ".tmp"
    tmp_mol_path = out_mol_path + ".tmp"
    tmp_frag_path = out_frag_path + ".tmp"
    tmp_link_path = out_link_path + ".tmp"

    # 先写临时文件
    write_sdf_strict(molecules, tmp_mol_path, "MOL")
    write_sdf_strict(fragments, tmp_frag_path, "FRAGMENTS")
    write_sdf_strict(linkers, tmp_link_path, "LINKER")

    out_table.to_csv(tmp_table_path, index=False)

    # Windows 上替换前最好先删除旧文件
    for old_path in [
        out_mol_path,
        out_frag_path,
        out_link_path,
        out_table_path,
    ]:
        if os.path.exists(old_path):
            os.remove(old_path)

    os.replace(tmp_mol_path, out_mol_path)
    os.replace(tmp_frag_path, out_frag_path)
    os.replace(tmp_link_path, out_link_path)
    os.replace(tmp_table_path, out_table_path)

    print(f"[WRITE_DONE] {data_dir} | num_samples={len(out_table)}")

if __name__ == "__main__":
    protac_path = "./datasets/protac_split_result_clean.csv"
    protac = pd.read_csv(protac_path)
    train_pretrain_df, val_pretrain_df, test_pretrain_df, train_rl_df, val_rl_df, test_rl_df = split_protac_dataset_for_pretrain_and_rl(
        df=protac,
        train_ratio=0.8,
        val_ratio=0.1,
        similarity_threshold=0.7,
        use_local_search=True,
    )
    run("train")
    run("val")
    run("test")
    run("train", ori_dir="./datasets/rl/")
    run("val", ori_dir="./datasets/rl/")
    run("test", ori_dir="./datasets/rl/")


