import ast
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Geometry import Point3D


# =========================
# 路径
# =========================

SPLITS = ("train", "val", "test")

# 最终 DeepTernary experimental binary manifest：
# 取得 experimental 六个源文件 + builder 显式 atom mapping
FINAL_MANIFEST_CSV = Path(r"./datasets/deepternary_rl2_unbound_v4/rl2_manifest.csv")
SELECTED_BINARIES_CSV = Path(
    r"./datasets/deepternary_rl2_unbound_v4/selected_binaries.csv"
)



def write_mol_to_sdf(mol, out_path: Path):
    """
    写 RDKit Mol 到 SDF。

    相比 PDB，SDF 更适合小分子 fragment，
    可以避免 MolToPDBFile 的 kekulize 问题。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None

    try:
        mol = Chem.Mol(mol)
        mol.UpdatePropertyCache(strict=False)

        # 尝试跳过 kekulize 的 sanitize
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
        except Exception:
            pass

        writer = Chem.SDWriter(str(out_path))

        if writer is None:
            print(f"[WARN] Could not create SDWriter: {out_path}")
            return False

        # 关闭 kekulize，避免芳香体系写出失败
        if hasattr(writer, "SetKekulize"):
            writer.SetKekulize(False)

        writer.write(mol)
        writer.close()
        writer = None

        return True

    except Exception as e:
        print(f"[WARN] Failed to write sdf {out_path}: {repr(e)}")
        return False

    finally:
        if writer is not None:
            writer.close()


def split_two_fragments(fragment_mol):
    """
    fragments.sdf 中每个 mol 是两个 disconnected fragments。

    当前数据预处理顺序是：
        Chem.CombineMols(warhead, e3_ligand)

    所以正常情况下：
        frag1 = warhead / POI ligand
        frag2 = E3 ligand
    """
    frags = Chem.GetMolFrags(
        fragment_mol,
        asMols=True,
        sanitizeFrags=False,
    )

    if len(frags) != 2:
        return None, None

    return frags[0], frags[1]

def read_sdf_single(path):
    """
    读取一个 SDF 分子，保留原始 atom order 和 3D coordinates。
    不使用几何距离推断 atom identity。
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    supplier = Chem.SDMolSupplier(
        str(path),
        sanitize=False,
        removeHs=False,
    )

    if len(supplier) == 0 or supplier[0] is None:
        raise ValueError(f"Failed to read SDF: {path}")

    mol = Chem.Mol(supplier[0])

    if mol.GetNumConformers() == 0:
        raise ValueError(f"SDF has no conformer: {path}")

    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    # 保留芳香表示，避免某些 mapped core 单独 sanitize 时失败
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=(
                Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ),
        )
    except Exception:
        pass

    return mol


def parse_manifest_indices(value, label):
    """
    rl2_manifest.csv 中 target_indices / bound_indices
    被 pandas 写成类似 "[0, 1, 2]" 的字符串。

    这里只允许安全 literal parsing，不使用 eval。
    """

    if isinstance(value, np.ndarray):
        values = value.tolist()

    elif isinstance(value, (list, tuple)):
        values = list(value)

    else:
        if value is None or pd.isna(value):
            raise ValueError(f"{label}: missing index list")

        text = str(value).strip()

        if text == "":
            raise ValueError(f"{label}: empty index list")

        try:
            values = ast.literal_eval(text)
        except Exception as e:
            raise ValueError(
                f"{label}: cannot parse index list: {text}"
            ) from e

    if not isinstance(values, (list, tuple)):
        raise ValueError(
            f"{label}: parsed value is not a list: {values}"
        )

    values = [int(x) for x in values]

    if len(values) == 0:
        raise ValueError(f"{label}: empty parsed index list")

    if len(set(values)) != len(values):
        raise ValueError(
            f"{label}: duplicated atom indices"
        )

    if min(values) < 0:
        raise ValueError(
            f"{label}: negative atom index"
        )

    return values


def validate_builder_mapping(
    core_mol,
    bound_mol,
    target_indices,
    bound_indices,
    label,
    coord_tol_A=0.02,
):
    """
    验证 builder 保存的显式映射没有在 CSV / 文件复制阶段损坏。

    builder 定义：
        core atom i
            ↔ target_indices[i]
            ↔ bound_indices[i]

    mapped_core 的坐标应来自 bound ligand 的 bound_indices。
    """

    n_core = core_mol.GetNumAtoms()

    if len(target_indices) != n_core:
        raise ValueError(
            f"{label}: target_indices length mismatch: "
            f"{len(target_indices)} != core atoms {n_core}"
        )

    if len(bound_indices) != n_core:
        raise ValueError(
            f"{label}: bound_indices length mismatch: "
            f"{len(bound_indices)} != core atoms {n_core}"
        )

    if max(bound_indices) >= bound_mol.GetNumAtoms():
        raise ValueError(
            f"{label}: bound index outside bound ligand"
        )

    # 化学元素必须逐原子一致
    for core_idx, bound_idx in enumerate(bound_indices):
        core_atom = core_mol.GetAtomWithIdx(core_idx)
        bound_atom = bound_mol.GetAtomWithIdx(bound_idx)

        if core_atom.GetAtomicNum() != bound_atom.GetAtomicNum():
            raise ValueError(
                f"{label}: atom element mismatch at core={core_idx}, "
                f"bound={bound_idx}: "
                f"{core_atom.GetSymbol()} != {bound_atom.GetSymbol()}"
            )

    core_xyz = np.asarray(
        core_mol.GetConformer().GetPositions(),
        dtype=np.float64,
    )

    bound_xyz = np.asarray(
        bound_mol.GetConformer().GetPositions(),
        dtype=np.float64,
    )

    mapped_bound_xyz = bound_xyz[
        np.asarray(bound_indices, dtype=np.int64)
    ]

    distances = np.linalg.norm(
        core_xyz - mapped_bound_xyz,
        axis=1,
    )

    max_diff = float(distances.max())

    if max_diff > float(coord_tol_A):
        raise ValueError(
            f"{label}: mapped_core coordinates do not agree with "
            f"builder bound_indices; max diff={max_diff:.4f} A"
        )

    return max_diff


def build_builder_target_mol(
    terminal_smiles_core,
    terminal_smiles_mapped,
    label,
):
    """
    重建 builder 中 target_indices 实际所索引的 target molecule，
    同时保留 linker exit atom index。

    terminal_smiles_mapped 中：
        atom-map number == 1
    表示 linker attachment / exit atom。

    返回：
        mol
        exit_idx
    """

    mapped_available = (
        terminal_smiles_mapped is not None
        and not pd.isna(terminal_smiles_mapped)
        and str(terminal_smiles_mapped).strip() != ""
    )

    if not mapped_available:
        raise ValueError(
            f"{label}: missing terminal_smiles_mapped; "
            f"cannot verify linker exit atom"
        )

    smi = str(terminal_smiles_mapped).strip()

    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        raise ValueError(
            f"{label}: cannot parse builder target mapped SMILES"
        )

    # --------------------------------------------------
    # atom-map number == 1 是 builder 保存的 linker exit
    # --------------------------------------------------
    exit_indices = [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() == 1
    ]

    if len(exit_indices) != 1:
        raise ValueError(
            f"{label}: expected exactly one linker exit atom, "
            f"got {len(exit_indices)}"
        )

    exit_idx = int(exit_indices[0])

    # --------------------------------------------------
    # 与 builder 原逻辑保持一致：
    # 保存 exit_idx 后清除 atom-map number，
    # 不重新 canonicalize atom order。
    # --------------------------------------------------
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    Chem.SanitizeMol(mol)

    return mol, exit_idx


def map_builder_target_to_fragment(
    fragment_mol,
    target_mol,
    target_indices,
    core_mol,
    target_exit_idx,
    expected_fragment_anchor_idx,
    label,
):
    """
    建立：

        builder target atom index
                    ->
        fixed fragment atom index

    target_indices 是 builder 已审核过的 common-core indices。

    对称等价映射允许存在；
    若出现真正不同的 atom set，则拒绝。

    对等价排列，利用 experimental mapped_core 的坐标，
    选择 Kabsch RMSD 最小的对应。

    注意：
    Kabsch 只用于已通过化学匹配的等价 permutation，
    不用于猜测 atom identity。
    """

    if fragment_mol is None:
        raise ValueError(
            f"{label}: fragment molecule is None"
        )

    if target_mol is None:
        raise ValueError(
            f"{label}: target molecule is None"
        )

    if fragment_mol.GetNumConformers() == 0:
        raise ValueError(
            f"{label}: fragment has no conformer"
        )

    if core_mol.GetNumConformers() == 0:
        raise ValueError(
            f"{label}: mapped core has no conformer"
        )

    target_indices = [
        int(i)
        for i in target_indices
    ]

    if len(target_indices) != core_mol.GetNumAtoms():
        raise ValueError(
            f"{label}: target_indices/core atom-count mismatch: "
            f"{len(target_indices)} != {core_mol.GetNumAtoms()}"
        )

    if max(target_indices) >= target_mol.GetNumAtoms():
        raise ValueError(
            f"{label}: target index outside builder target molecule"
        )

    target_exit_idx = int(target_exit_idx)
    expected_fragment_anchor_idx = int(
        expected_fragment_anchor_idx
    )

    if (
            target_exit_idx < 0
            or target_exit_idx >= target_mol.GetNumAtoms()
    ):
        raise ValueError(
            f"{label}: target exit index outside builder target molecule: "
            f"{target_exit_idx}"
        )

    if (
            expected_fragment_anchor_idx < 0
            or expected_fragment_anchor_idx >= fragment_mol.GetNumAtoms()
    ):
        raise ValueError(
            f"{label}: expected fragment anchor index outside fragment: "
            f"{expected_fragment_anchor_idx}"
        )

    # fixed fragment 和 builder target 都应该表示完整 terminal。
    if fragment_mol.GetNumAtoms() != target_mol.GetNumAtoms():
        raise ValueError(
            f"{label}: full terminal atom-count mismatch: "
            f"fragment={fragment_mol.GetNumAtoms()}, "
            f"builder_target={target_mol.GetNumAtoms()}"
        )

    # --------------------------------------------------
    # 尽量把 fragment 恢复成正常 sanitized chemistry。
    # fragments.sdf 本身原生成时已经 sanitize 过，
    # 这里只是因为当前 supplier 使用 sanitize=False。
    # --------------------------------------------------
    fragment_search = Chem.Mol(fragment_mol)

    try:
        fragment_search.UpdatePropertyCache(
            strict=False
        )
        Chem.SanitizeMol(fragment_search)
    except Exception as e:
        raise ValueError(
            f"{label}: cannot sanitize fixed fragment: {repr(e)}"
        )

    def enumerate_matches(use_chirality):
        matches = fragment_search.GetSubstructMatches(
            target_mol,
            useChirality=use_chirality,
            uniquify=False,
            maxMatches=2048,
        )

        if len(matches) >= 2048:
            raise ValueError(
                f"{label}: terminal atom-map enumeration limit reached"
            )

        return sorted(
            set(
                tuple(int(x) for x in m)
                for m in matches
            )
        )

    # builder target -> complete fixed fragment
    matches = enumerate_matches(True)
    match_mode = "chirality"

    if len(matches) == 0:
        matches = enumerate_matches(False)
        match_mode = "no_chirality"

    if len(matches) == 0:
        raise ValueError(
            f"{label}: no full-terminal chemical match "
            f"between builder target and fixed fragment"
        )

    # --------------------------------------------------
    # linker exit -> fixed-fragment anchor 硬约束
    #
    # match[target_atom_index] = fragment_atom_index
    #
    # 所以 builder target 中的 linker exit 必须映射到
    # 当前固定 fragment 的真实 linker anchor。
    # --------------------------------------------------
    chemical_match_count = len(matches)

    exit_valid_matches = [
        match
        for match in matches
        if int(match[target_exit_idx])
           == expected_fragment_anchor_idx
    ]

    if len(exit_valid_matches) == 0:
        mapped_anchor_candidates = sorted(
            {
                int(match[target_exit_idx])
                for match in matches
            }
        )

        raise ValueError(
            f"{label}: no terminal mapping preserves linker exit -> anchor; "
            f"target_exit_idx={target_exit_idx}, "
            f"expected_fragment_anchor_idx={expected_fragment_anchor_idx}, "
            f"mapped_anchor_candidates={mapped_anchor_candidates}, "
            f"chemical_matches={chemical_match_count}, "
            f"mode={match_mode}"
        )

    matches = exit_valid_matches

    # 因为两边 atom count 相等，一个合法 match 应覆盖整个 fragment。
    full_fragment_set = frozenset(
        range(fragment_search.GetNumAtoms())
    )

    for match in matches:
        if frozenset(match) != full_fragment_set:
            raise ValueError(
                f"{label}: terminal match does not cover entire fragment"
            )

    core_xyz = np.asarray(
        core_mol.GetConformer().GetPositions(),
        dtype=np.float64,
    )

    fragment_xyz = np.asarray(
        fragment_mol.GetConformer().GetPositions(),
        dtype=np.float64,
    )

    best = None

    for match in matches:
        # match[target_atom_index] = fragment_atom_index
        fragment_core_indices = np.asarray(
            [
                match[target_idx]
                for target_idx in target_indices
            ],
            dtype=np.int64,
        )

        # --------------------------------------------------
        # 再做一次元素级检查：
        #
        # core atom i
        #   ↔ target_indices[i]
        #   ↔ fragment_core_indices[i]
        # --------------------------------------------------
        element_ok = True

        for core_idx, fragment_idx in enumerate(
            fragment_core_indices
        ):
            if (
                core_mol
                .GetAtomWithIdx(core_idx)
                .GetAtomicNum()
                !=
                fragment_mol
                .GetAtomWithIdx(int(fragment_idx))
                .GetAtomicNum()
            ):
                element_ok = False
                break

        if not element_ok:
            continue

        target_xyz = fragment_xyz[
            fragment_core_indices
        ]

        _, _, rmsd = calculate_kabsch_transform(
            core_xyz,
            target_xyz,
            f"{label}/terminal_symmetry",
        )

        candidate = (
            float(rmsd),
            tuple(int(x) for x in match),
            fragment_core_indices,
        )

        if (
            best is None
            or candidate[:2] < best[:2]
        ):
            best = candidate

    if best is None:
        raise ValueError(
            f"{label}: no element-consistent "
            f"target-to-fragment atom mapping"
        )

    best_rmsd, best_match, fragment_core_indices = best

    return (
        fragment_core_indices,
        match_mode,
        len(matches),
        best_rmsd,
    )


def calculate_kabsch_transform(source_xyz, target_xyz, label):
    """
    求刚体变换：

        source experimental mapped-core coordinates
                        ->
        target fixed-fragment coordinates

    返回 R, t，使：
        aligned_xyz = xyz @ R.T + t
    """

    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)

    if source_xyz.shape != target_xyz.shape:
        raise ValueError(
            f"{label}: Kabsch shape mismatch: "
            f"source={source_xyz.shape}, target={target_xyz.shape}"
        )

    if source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError(
            f"{label}: invalid coordinate shape {source_xyz.shape}"
        )

    if source_xyz.shape[0] < 3:
        raise ValueError(
            f"{label}: fewer than 3 mapped atoms"
        )

    source_center = source_xyz.mean(axis=0)
    target_center = target_xyz.mean(axis=0)

    source_zero = source_xyz - source_center
    target_zero = target_xyz - target_center

    if np.linalg.matrix_rank(source_zero, tol=1e-6) < 2:
        raise ValueError(
            f"{label}: source mapped-core coordinates are collinear"
        )

    if np.linalg.matrix_rank(target_zero, tol=1e-6) < 2:
        raise ValueError(
            f"{label}: target fragment coordinates are collinear"
        )

    H = source_zero.T @ target_zero

    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # 禁止 reflection，只允许真正的 rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = target_center - R @ source_center

    fitted = source_xyz @ R.T + t

    rmsd = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (fitted - target_xyz) ** 2,
                    axis=1,
                )
            )
        )
    )

    if not np.isfinite(rmsd):
        raise ValueError(
            f"{label}: non-finite Kabsch RMSD"
        )

    return R, t, rmsd


def transform_mol_coordinates(mol, R, t):
    """
    对整个 RDKit molecule 应用同一个刚体变换。
    化学图和 atom order 完全不变，只修改 coordinates。
    """

    out = Chem.Mol(mol)

    if out.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer")

    conf = out.GetConformer()

    for atom_idx in range(out.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)

        xyz = np.asarray(
            [pos.x, pos.y, pos.z],
            dtype=np.float64,
        )

        new_xyz = R @ xyz + t

        conf.SetAtomPosition(
            atom_idx,
            Point3D(
                float(new_xyz[0]),
                float(new_xyz[1]),
                float(new_xyz[2]),
            ),
        )

    return out


def transform_pdb_coordinates(src_path, dst_path, R, t):
    """
    对 PDB 中所有 ATOM/HETATM 坐标应用同一个刚体变换。

    不修改 residue、chain、atom name 等其它信息。
    """

    src_path = Path(src_path)
    dst_path = Path(dst_path)

    if not src_path.is_file():
        raise FileNotFoundError(src_path)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    new_lines = []

    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            if (
                line.startswith("ATOM  ")
                or line.startswith("HETATM")
            ) and len(line) >= 54:

                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    xyz = np.asarray(
                        [x, y, z],
                        dtype=np.float64,
                    )

                    new_xyz = R @ xyz + t

                    line = (
                        line[:30]
                        + f"{new_xyz[0]:8.3f}"
                        + f"{new_xyz[1]:8.3f}"
                        + f"{new_xyz[2]:8.3f}"
                        + line[54:]
                    )

                except ValueError:
                    pass

            new_lines.append(line)

    with open(dst_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

# ============================================================
# 读取最终 DeepTernary experimental binary 路径和显式 atom mapping；
# 不改变原始 molecular dataset，只生成对齐后的 DeepTernary 输入
# ============================================================
_FINAL_PATH_COLS = [
    "poi_protein_path",
    "e3_protein_path",
    "poi_bound_ligand_path",
    "e3_bound_ligand_path",
    "poi_mapped_core_path",
    "e3_mapped_core_path",
]
_FINAL_MAPPING_COLS = [
    "poi_target_indices",
    "poi_bound_indices",
    "e3_target_indices",
    "e3_bound_indices",
]

_FINAL_PAYLOAD_COLS = (
    _FINAL_PATH_COLS
    + _FINAL_MAPPING_COLS
)


def _canon_smiles(s, isomeric=True):
    mol = Chem.MolFromSmiles(str(s))
    if mol is None:
        return None
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=isomeric,
    )


def _same_final_payload(df):
    if len(df) == 0:
        return False

    return all(
        df[c].astype(str).nunique(dropna=False) == 1
        for c in _FINAL_PAYLOAD_COLS
    )


def find_final_manifest_row(manifest_df, split, molecule, poi_uniprot, e3_uniprot):
    """
    只做“当前 processed 样本 -> 最终 manifest 路径”的定位。

    优先 exact isomeric SMILES；
    若 processed 3D SMILES 比原始 SMILES 多了未指定手性标记，则退回到
    相同 connectivity + 相同 POI/E3，并要求原始已明确的手性不冲突。

    这个函数只决定六个 path 字段，不参与原来的数据预处理逻辑。
    """
    split = str(split).strip().lower()
    poi_uniprot = str(poi_uniprot).strip().upper()
    e3_uniprot = str(e3_uniprot).strip().upper()

    candidates = manifest_df[
        (manifest_df["split"].astype(str).str.lower() == split)
        & (manifest_df["poi_protein_uniprot"].astype(str).str.upper() == poi_uniprot)
        & (manifest_df["e3_protein_uniprot"].astype(str).str.upper() == e3_uniprot)
    ].copy()

    if len(candidates) == 0:
        return None

    mol_iso = _canon_smiles(molecule, isomeric=True)
    mol_noniso = _canon_smiles(molecule, isomeric=False)
    if mol_iso is None or mol_noniso is None:
        return None

    # 1. 先尝试完全一致的 isomeric SMILES
    exact_mask = candidates["protac_id"].map(
        lambda s: _canon_smiles(s, isomeric=True) == mol_iso
    )
    exact = candidates[exact_mask]
    if len(exact) > 0:
        if not _same_final_payload(exact):
            raise RuntimeError(
                f"同一 exact identity 对应多套 DeepTernary 路径: "
                f"split={split}, molecule={molecule}"
            )
        return exact.iloc[0]

    # 2. connectivity 必须一致
    conn_mask = candidates["protac_id"].map(
        lambda s: _canon_smiles(s, isomeric=False) == mol_noniso
    )
    conn = candidates[conn_mask].copy()

    if len(conn) == 0:
        return None

    # 3. 原始 manifest 中明确指定的手性不能与 processed molecule 冲突。
    processed_mol = Chem.MolFromSmiles(str(molecule))

    def stereo_compatible(original_smi):
        original_mol = Chem.MolFromSmiles(str(original_smi))
        if original_mol is None or processed_mol is None:
            return False
        return bool(processed_mol.HasSubstructMatch(original_mol, useChirality=True))

    compatible = conn[conn["protac_id"].map(stereo_compatible)]

    if len(compatible) == 0:
        return None

    if not _same_final_payload(compatible):
        raise RuntimeError(
            f"同一 processed identity 可对应多套 DeepTernary 路径，无法安全选择: "
            f"split={split}, molecule={molecule}, "
            f"POI={poi_uniprot}, E3={e3_uniprot}"
        )

    return compatible.iloc[0]

def to_relative_path(p):
    if p is None or pd.isna(p):
        return None
    return os.path.relpath(str(p), start=Path.cwd()).replace("\\", "/")

def print_geometry_summary(
    label,
    rmsds,
    max_errors,
    exit_errors,
):
    rmsds = np.asarray(
        rmsds,
        dtype=np.float64,
    )

    max_errors = np.asarray(
        max_errors,
        dtype=np.float64,
    )

    exit_errors = np.asarray(
        exit_errors,
        dtype=np.float64,
    )

    if len(rmsds) == 0:
        print(
            f"{label}: no successful alignments"
        )
        return

    print(f"\n{label} geometry summary:")

    print(
        "  Kabsch RMSD [A]: "
        f"median={np.median(rmsds):.4f}, "
        f"q95={np.quantile(rmsds, 0.95):.4f}, "
        f"max={np.max(rmsds):.4f}"
    )

    print(
        "  max mapped-atom error [A]: "
        f"median={np.median(max_errors):.4f}, "
        f"q95={np.quantile(max_errors, 0.95):.4f}, "
        f"max={np.max(max_errors):.4f}"
    )

    print(
        "  max mapped-atom error > 1 A: "
        f"{int(np.sum(max_errors > 1.0))}"
        f"/{len(max_errors)}"
    )

    print(
        "  exit-anchor error [A]: "
        f"median={np.median(exit_errors):.4f}, "
        f"q95={np.quantile(exit_errors, 0.95):.4f}, "
        f"max={np.max(exit_errors):.4f}"
    )

    print(
        "  exit-anchor error > 1 A: "
        f"{int(np.sum(exit_errors > 1.0))}"
        f"/{len(exit_errors)}"
    )

def main(split):
    PROTAC_CSV = Path(f"./datasets/rl/{split}/protac_processed.csv")
    FRAGMENTS_SDF = Path(f"./datasets/rl/{split}/fragments.sdf")
    OUT_ROOT = Path(f"./datasets/rl/{split}/deepternary_inputs")
    OUT_CSV = Path(f"./datasets/rl/{split}/protac_processed_deepternary.csv")

    print("\n" + "=" * 80)
    print(f"Processing split: {split}")
    print("=" * 80)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 加载最终 manifest：
    # 获取 experimental 六个源文件和 builder 显式 atom mapping
    final_manifest = pd.read_csv(FINAL_MANIFEST_CSV)

    selected_binaries = pd.read_csv(
        SELECTED_BINARIES_CSV
    )

    required_selected_cols = {
        "pair_id",
        "terminal_smiles_core",
        "terminal_smiles_mapped",
    }

    missing_selected_cols = (
            required_selected_cols
            - set(selected_binaries.columns)
    )

    if missing_selected_cols:
        raise ValueError(
            f"{SELECTED_BINARIES_CSV} 缺少列: "
            f"{missing_selected_cols}"
        )

    if selected_binaries["pair_id"].duplicated().any():
        duplicated = selected_binaries.loc[
            selected_binaries["pair_id"].duplicated(
                keep=False
            ),
            "pair_id",
        ].tolist()

        raise ValueError(
            "selected_binaries.csv 中 pair_id 不唯一: "
            f"{duplicated[:20]}"
        )

    selected_by_pair = selected_binaries.set_index(
        "pair_id",
        drop=False,
    )

    current_split = split.lower()

    # 关键：reset_index，保证 CSV 第 k 行对应 fragments.sdf 第 k 个 mol
    df = pd.read_csv(PROTAC_CSV).reset_index(drop=True)

    required_cols = {
        "uuid",
        "Uniprot",
        "e3_ligase_Uniprot",
        "warhead_num_atoms",
        "e3_ligand_num_atoms",
        "warhead_anchor",
        "e3_ligand_anchor",
    }

    missing_cols = required_cols - set(df.columns)

    if len(missing_cols) > 0:
        raise ValueError(f"{PROTAC_CSV} 缺少列: {missing_cols}")

    frag_supplier = Chem.SDMolSupplier(
        str(FRAGMENTS_SDF),
        removeHs=False,
        sanitize=False,
    )

    if len(frag_supplier) != len(df):
        print("[WARN] fragments.sdf 数量和 CSV 行数不一致")
        print("csv rows:", len(df))
        print("sdf mols:", len(frag_supplier))

    new_rows = []
    missing = []
    alignment_successes = 0
    alignment_failures = 0

    # 只用于轻量几何诊断，不参与筛选
    poi_kabsch_rmsds = []
    e3_kabsch_rmsds = []

    poi_max_mapped_errors = []
    e3_max_mapped_errors = []

    poi_exit_errors = []
    e3_exit_errors = []

    for sdf_idx, row in tqdm(df.iterrows(), total=len(df)):
        uuid = str(row["uuid"])

        poi_uniprot = str(row["Uniprot"]).strip()
        e3_uniprot = str(row["e3_ligase_Uniprot"]).strip()

        frag_mol = frag_supplier[int(sdf_idx)]

        if frag_mol is None:
            missing.append((uuid, "fragment_sdf_none", sdf_idx))
            continue

        if frag_mol.GetNumConformers() == 0:
            missing.append((uuid, "fragment_no_conformer", sdf_idx))
            continue

        frag1, frag2 = split_two_fragments(frag_mol)

        if frag1 is None or frag2 is None:
            missing.append((uuid, "not_two_fragments", sdf_idx))
            continue

        # ==================================================
        # 强校验 fragment 顺序：
        # frag1 应该是 warhead / POI ligand
        # frag2 应该是 E3 ligand
        # ==================================================
        n_warhead = int(row["warhead_num_atoms"])
        n_e3 = int(row["e3_ligand_num_atoms"])

        poi_anchor_local = int(
            row["warhead_anchor"]
        )

        e3_anchor_local = (
                int(row["e3_ligand_anchor"])
                - n_warhead
        )

        if not (0 <= poi_anchor_local < n_warhead):
            raise ValueError(
                f"{split}/{uuid}: invalid POI anchor local index: "
                f"{poi_anchor_local}"
            )

        if not (0 <= e3_anchor_local < n_e3):
            raise ValueError(
                f"{split}/{uuid}: invalid E3 anchor local index: "
                f"{e3_anchor_local}"
            )

        if frag1.GetNumAtoms() != n_warhead or frag2.GetNumAtoms() != n_e3:
            missing.append((
                uuid,
                "fragment_order_or_row_mismatch",
                (
                    f"frag1_atoms={frag1.GetNumAtoms()}, expected_warhead={n_warhead}; "
                    f"frag2_atoms={frag2.GetNumAtoms()}, expected_e3={n_e3}; "
                    f"sdf_idx={sdf_idx}"
                ),
            ))
            continue
        # ==================================================
        # 这里只替换六个 DeepTernary 路径
        # ==================================================
        final_meta = find_final_manifest_row(
            final_manifest,
            current_split,
            row["molecule"],
            poi_uniprot,
            e3_uniprot,
        )

        # 每次重新构建该 uuid，避免旧版未对齐文件残留
        sample_dir = OUT_ROOT / uuid

        if sample_dir.exists():
            shutil.rmtree(sample_dir)

        if final_meta is None:
            final_p1_path = None
            final_p2_path = None
            final_unbound_lig1_path = None
            final_unbound_lig2_path = None
            final_lig1_mask_path = None
            final_lig2_mask_path = None
        else:
            # ==================================================
            # 数据源：
            #   deepternary_rl2_unbound_v4
            #
            # 输出：
            #   deepternary_inputs/<uuid>
            #
            # 这里不改变任何 chemistry / atom order，
            # 只把 experimental binary 整体刚体变换到
            # 当前 PT fixed-fragment 坐标系。
            #
            # 历史 PT 方向保持：
            #   p1 / lig1 = POI
            #   p2 / lig2 = E3
            # ==================================================

            sample_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            p1_path = sample_dir / "p1_poi.pdb"
            p2_path = sample_dir / "p2_e3.pdb"

            unbound_lig1_path = (
                    sample_dir / "unbound_lig1_poi.sdf"
            )
            unbound_lig2_path = (
                    sample_dir / "unbound_lig2_e3.sdf"
            )

            lig1_mask_path = (
                    sample_dir / "lig1_mask_poi.sdf"
            )
            lig2_mask_path = (
                    sample_dir / "lig2_mask_e3.sdf"
            )

            try:
                # ==========================================
                # 1. 读取 builder 已审核的 experimental
                #    bound ligand + mapped core
                # ==========================================

                poi_bound = read_sdf_single(
                    final_meta["poi_bound_ligand_path"]
                )
                poi_core = read_sdf_single(
                    final_meta["poi_mapped_core_path"]
                )

                e3_bound = read_sdf_single(
                    final_meta["e3_bound_ligand_path"]
                )
                e3_core = read_sdf_single(
                    final_meta["e3_mapped_core_path"]
                )

                # ==========================================
                # 2. 读取 builder 保存的 explicit map
                # ==========================================

                poi_target_indices = parse_manifest_indices(
                    final_meta["poi_target_indices"],
                    f"{split}/{uuid}/POI target_indices",
                )
                poi_bound_indices = parse_manifest_indices(
                    final_meta["poi_bound_indices"],
                    f"{split}/{uuid}/POI bound_indices",
                )

                e3_target_indices = parse_manifest_indices(
                    final_meta["e3_target_indices"],
                    f"{split}/{uuid}/E3 target_indices",
                )
                e3_bound_indices = parse_manifest_indices(
                    final_meta["e3_bound_indices"],
                    f"{split}/{uuid}/E3 bound_indices",
                )

                # ==========================================
                # 2.5 根据 pair_id 取得 builder 当时实际使用的
                #     POI / E3 terminal chemistry
                # ==========================================

                poi_pair_id = str(
                    final_meta["poi_pair_id"]
                ).strip()

                e3_pair_id = str(
                    final_meta["e3_pair_id"]
                ).strip()

                if poi_pair_id not in selected_by_pair.index:
                    raise ValueError(
                        f"{split}/{uuid}/POI: "
                        f"pair_id not found in selected_binaries: "
                        f"{poi_pair_id}"
                    )

                if e3_pair_id not in selected_by_pair.index:
                    raise ValueError(
                        f"{split}/{uuid}/E3: "
                        f"pair_id not found in selected_binaries: "
                        f"{e3_pair_id}"
                    )

                poi_selected = selected_by_pair.loc[
                    poi_pair_id
                ]

                e3_selected = selected_by_pair.loc[
                    e3_pair_id
                ]

                poi_target, poi_exit_idx = build_builder_target_mol(
                    terminal_smiles_core=(
                        poi_selected["terminal_smiles_core"]
                    ),
                    terminal_smiles_mapped=(
                        poi_selected["terminal_smiles_mapped"]
                    ),
                    label=f"{split}/{uuid}/POI",
                )

                e3_target, e3_exit_idx = build_builder_target_mol(
                    terminal_smiles_core=(
                        e3_selected["terminal_smiles_core"]
                    ),
                    terminal_smiles_mapped=(
                        e3_selected["terminal_smiles_mapped"]
                    ),
                    label=f"{split}/{uuid}/E3",
                )

                # ==========================================
                # 3. 强校验：
                #
                # mapped_core atom i
                #     ↔ target_indices[i]
                #     ↔ bound_indices[i]
                #
                # 确认 builder 输出没有损坏。
                # ==========================================

                validate_builder_mapping(
                    core_mol=poi_core,
                    bound_mol=poi_bound,
                    target_indices=poi_target_indices,
                    bound_indices=poi_bound_indices,
                    label=f"{split}/{uuid}/POI",
                )

                validate_builder_mapping(
                    core_mol=e3_core,
                    bound_mol=e3_bound,
                    target_indices=e3_target_indices,
                    bound_indices=e3_bound_indices,
                    label=f"{split}/{uuid}/E3",
                )

                # ==========================================
                # 4. builder target -> 当前 fixed fragment
                #
                # frag1 = POI / warhead
                # frag2 = E3 ligand
                # ==========================================

                (
                    poi_fragment_indices,
                    poi_match_mode,
                    poi_num_matches,
                    poi_symmetry_rmsd,
                ) = map_builder_target_to_fragment(
                    fragment_mol=frag1,
                    target_mol=poi_target,
                    target_indices=poi_target_indices,
                    core_mol=poi_core,
                    target_exit_idx=poi_exit_idx,
                    expected_fragment_anchor_idx=poi_anchor_local,
                    label=f"{split}/{uuid}/POI",
                )

                (
                    e3_fragment_indices,
                    e3_match_mode,
                    e3_num_matches,
                    e3_symmetry_rmsd,
                ) = map_builder_target_to_fragment(
                    fragment_mol=frag2,
                    target_mol=e3_target,
                    target_indices=e3_target_indices,
                    core_mol=e3_core,
                    target_exit_idx=e3_exit_idx,
                    expected_fragment_anchor_idx=e3_anchor_local,
                    label=f"{split}/{uuid}/E3",
                )

                # ==========================================
                # 5. Kabsch:
                #
                # experimental mapped_core xyz
                #               ->
                # PT fixed fragment xyz
                # ==========================================

                poi_core_xyz = np.asarray(
                    poi_core
                    .GetConformer()
                    .GetPositions(),
                    dtype=np.float64,
                )

                poi_fragment_xyz_all = np.asarray(
                    frag1
                    .GetConformer()
                    .GetPositions(),
                    dtype=np.float64,
                )

                poi_target_xyz = poi_fragment_xyz_all[
                    poi_fragment_indices
                ]

                poi_R, poi_t, poi_kabsch_rmsd = (
                    calculate_kabsch_transform(
                        poi_core_xyz,
                        poi_target_xyz,
                        f"{split}/{uuid}/POI",
                    )
                )

                poi_fitted_xyz = (
                        poi_core_xyz @ poi_R.T
                        + poi_t
                )

                poi_atom_errors = np.linalg.norm(
                    poi_fitted_xyz - poi_target_xyz,
                    axis=1,
                )
                if poi_exit_idx not in poi_target_indices:
                    raise ValueError(
                        f"{split}/{uuid}/POI: "
                        f"linker exit atom not present in mapped core"
                    )

                poi_exit_core_idx = poi_target_indices.index(
                    poi_exit_idx
                )

                poi_exit_error = float(
                    poi_atom_errors[poi_exit_core_idx]
                )

                if not np.isfinite(poi_exit_error):
                    raise ValueError(
                        f"{split}/{uuid}/POI: "
                        f"non-finite exit-anchor error"
                    )

                poi_max_mapped_error = float(
                    np.max(poi_atom_errors)
                )

                if not np.isfinite(poi_max_mapped_error):
                    raise ValueError(
                        f"{split}/{uuid}/POI: "
                        f"non-finite max mapped-atom error"
                    )

                e3_core_xyz = np.asarray(
                    e3_core
                    .GetConformer()
                    .GetPositions(),
                    dtype=np.float64,
                )

                e3_fragment_xyz_all = np.asarray(
                    frag2
                    .GetConformer()
                    .GetPositions(),
                    dtype=np.float64,
                )

                e3_target_xyz = e3_fragment_xyz_all[
                    e3_fragment_indices
                ]

                e3_R, e3_t, e3_kabsch_rmsd = (
                    calculate_kabsch_transform(
                        e3_core_xyz,
                        e3_target_xyz,
                        f"{split}/{uuid}/E3",
                    )
                )

                e3_fitted_xyz = (
                        e3_core_xyz @ e3_R.T
                        + e3_t
                )

                e3_atom_errors = np.linalg.norm(
                    e3_fitted_xyz - e3_target_xyz,
                    axis=1,
                )

                if e3_exit_idx not in e3_target_indices:
                    raise ValueError(
                        f"{split}/{uuid}/E3: "
                        f"linker exit atom not present in mapped core"
                    )

                e3_exit_core_idx = e3_target_indices.index(
                    e3_exit_idx
                )

                e3_exit_error = float(
                    e3_atom_errors[e3_exit_core_idx]
                )

                if not np.isfinite(e3_exit_error):
                    raise ValueError(
                        f"{split}/{uuid}/E3: "
                        f"non-finite exit-anchor error"
                    )

                e3_max_mapped_error = float(
                    np.max(e3_atom_errors)
                )

                if not np.isfinite(e3_max_mapped_error):
                    raise ValueError(
                        f"{split}/{uuid}/E3: "
                        f"non-finite max mapped-atom error"
                    )

                # ==========================================
                # 6. 同一个刚体变换同时作用于：
                #
                # protein
                # bound ligand
                # mapped core
                #
                # 绝不能只移动 ligand。
                # ==========================================

                transform_pdb_coordinates(
                    final_meta["poi_protein_path"],
                    p1_path,
                    poi_R,
                    poi_t,
                )

                transform_pdb_coordinates(
                    final_meta["e3_protein_path"],
                    p2_path,
                    e3_R,
                    e3_t,
                )

                poi_bound_aligned = (
                    transform_mol_coordinates(
                        poi_bound,
                        poi_R,
                        poi_t,
                    )
                )

                poi_core_aligned = (
                    transform_mol_coordinates(
                        poi_core,
                        poi_R,
                        poi_t,
                    )
                )

                e3_bound_aligned = (
                    transform_mol_coordinates(
                        e3_bound,
                        e3_R,
                        e3_t,
                    )
                )

                e3_core_aligned = (
                    transform_mol_coordinates(
                        e3_core,
                        e3_R,
                        e3_t,
                    )
                )

                # ==========================================
                # 7. 写出 SDF。
                #
                # 保留 chemistry 和 atom order，
                # 仅坐标发生刚体变换。
                # ==========================================

                if not write_mol_to_sdf(
                        poi_bound_aligned,
                        unbound_lig1_path,
                ):
                    raise RuntimeError(
                        "failed to write aligned POI bound ligand"
                    )

                if not write_mol_to_sdf(
                        poi_core_aligned,
                        lig1_mask_path,
                ):
                    raise RuntimeError(
                        "failed to write aligned POI mapped core"
                    )

                if not write_mol_to_sdf(
                        e3_bound_aligned,
                        unbound_lig2_path,
                ):
                    raise RuntimeError(
                        "failed to write aligned E3 bound ligand"
                    )

                if not write_mol_to_sdf(
                        e3_core_aligned,
                        lig2_mask_path,
                ):
                    raise RuntimeError(
                        "failed to write aligned E3 mapped core"
                    )

                # ==========================================
                # 8. CSV/PT 继续保存原来的相对路径格式
                # ==========================================

                final_p1_path = to_relative_path(p1_path)
                final_p2_path = to_relative_path(p2_path)

                final_unbound_lig1_path = (
                    to_relative_path(unbound_lig1_path)
                )
                final_unbound_lig2_path = (
                    to_relative_path(unbound_lig2_path)
                )

                final_lig1_mask_path = (
                    to_relative_path(lig1_mask_path)
                )
                final_lig2_mask_path = (
                    to_relative_path(lig2_mask_path)
                )

                # 成功完成整个 alignment 后才纳入几何统计
                poi_kabsch_rmsds.append(
                    float(poi_kabsch_rmsd)
                )
                e3_kabsch_rmsds.append(
                    float(e3_kabsch_rmsd)
                )

                poi_max_mapped_errors.append(
                    float(poi_max_mapped_error)
                )
                e3_max_mapped_errors.append(
                    float(e3_max_mapped_error)
                )

                poi_exit_errors.append(
                    float(poi_exit_error)
                )

                e3_exit_errors.append(
                    float(e3_exit_error)
                )
                alignment_successes += 1

            except Exception as e:
                # ------------------------------------------
                # 对齐失败：
                # 不能留下旧的、未经正确对齐的六文件。
                #
                # 这个 molecular sample 本身仍保留，
                # 只是 DeepTernary input 变为 unavailable。
                # ------------------------------------------

                shutil.rmtree(
                    sample_dir,
                    ignore_errors=True,
                )

                final_p1_path = None
                final_p2_path = None

                final_unbound_lig1_path = None
                final_unbound_lig2_path = None

                final_lig1_mask_path = None
                final_lig2_mask_path = None

                alignment_failures += 1

                print(
                    f"[ALIGNMENT_FAIL] split={split}, "
                    f"uuid={uuid}, error={repr(e)}"
                )

        new_row = row.to_dict()
        new_row.update({
            "p1_path": final_p1_path,
            "p2_path": final_p2_path,
            "unbound_lig1_path": final_unbound_lig1_path,
            "unbound_lig2_path": final_unbound_lig2_path,
            "lig1_mask_path": final_lig1_mask_path,
            "lig2_mask_path": final_lig2_mask_path,
        })

        new_rows.append(new_row)

    out_df = pd.DataFrame(new_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print("saved:", OUT_CSV)
    print("valid rows:", len(out_df))
    print("missing rows:", len(missing))

    print(
        "alignment successes:",
        alignment_successes,
    )

    print(
        "alignment failures:",
        alignment_failures,
    )

    print_geometry_summary(
        f"{split}/POI",
        poi_kabsch_rmsds,
        poi_max_mapped_errors,
        poi_exit_errors,
    )

    print_geometry_summary(
        f"{split}/E3",
        e3_kabsch_rmsds,
        e3_max_mapped_errors,
        e3_exit_errors,
    )

    if len(missing) > 0:
        missing_path = OUT_ROOT / "missing_deepternary_inputs.csv"

        missing_df = pd.DataFrame(
            missing,
            columns=["uuid", "reason", "value"],
        )

        missing_df.to_csv(missing_path, index=False)

        print("missing saved:", missing_path)
        print("\nmissing reason counts:")
        print(missing_df["reason"].value_counts())
        print("\nmissing head:")
        print(missing_df.head(20))


if __name__ == "__main__":
    for split in SPLITS:
        main(split)
