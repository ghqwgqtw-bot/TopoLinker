#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
汇总 Smina Stage 1 结果、提取 affinity + pose 空间中心、进行空间聚类，
并选出 Top-K candidate binding regions 供 Stage 2 使用。

当前设计用于：
    tiled whole-protein blind docking
        -> Stage 1 多 box
        -> 汇总所有 pose
        -> 以 ligand heavy-atom centroid 做空间聚类
        -> 选 Top 3 candidate regions
        -> Stage 2 高精度 docking

默认适配当前 pilot：
    eval_outputs/deepprotacs_docking/stage1/e3_0001
    eval_outputs/deepprotacs_docking/smina_grids/e3_0001/e3_0001_boxes_active.csv

重要说明
--------
1) 本脚本只读取 Stage 1 结果，不修改、不删除已有 box_*.sdf / box_*.log。
2) 可以对“部分完成的 box”运行，因此适合当前 pilot。
3) 正式全量时，应在该 pair 的所有 active boxes 完成后再运行一次。
4) 聚类对象是每个 docking pose 的 ligand heavy-atom centroid。
5) 默认空间聚类半径为 5 Å。
6) 聚类采用“按 affinity 从优到劣的 greedy radius clustering”：
   - 最低 affinity pose 先作为一个 cluster seed；
   - seed 周围 5 Å 内、尚未分配的 pose 归入该 cluster；
   - 然后继续下一个尚未分配的最低 affinity pose。
   这样可以避免单链式连接造成一个 cluster 被拉得过长。
7) cluster 排名默认：
   best affinity（越低越好）
   -> median affinity（越低越好）
   -> 支持 box 数（越多越好）
8) candidate region 的 Stage 2 中心使用该 cluster 所有 pose centroid 的平均坐标。
9) 默认写出 Top 3 candidate regions；Stage 2 box_size 默认记录为 30 Å。

依赖：
    numpy
    pandas
    rdkit
"""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


AFFINITY_PROP_CANDIDATES = (
    "minimizedAffinity",
    "affinity",
    "vina_affinity",
    "VINA_AFFINITY",
    "SCORE",
    "score",
)


def parse_box_id(path: Path) -> int:
    m = re.search(r"box_(\d+)", path.stem)
    if not m:
        raise ValueError(f"无法从文件名解析 box_id: {path.name}")
    return int(m.group(1))


def parse_affinities_from_log(log_path: Path):
    """
    从 Smina 日志的 mode/affinity 表中解析：
        mode -> affinity (kcal/mol)

    示例：
        1       -6.2       0.000      0.000
        2       -6.2       4.732      7.488
    """
    if not log_path.exists():
        return {}

    text = log_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    affinities = {}
    in_table = False

    for line in text.splitlines():
        if "mode" in line.lower() and "affinity" in line.lower():
            in_table = True
            continue

        if not in_table:
            continue

        # 跳过表头分隔线、单位行等
        if not line.strip():
            if affinities:
                break
            continue

        m = re.match(
            r"^\s*(\d+)\s+"
            r"(-?\d+(?:\.\d+)?)"
            r"(?:\s+[-+]?\d+(?:\.\d+)?)?"
            r"(?:\s+[-+]?\d+(?:\.\d+)?)?"
            r"\s*$",
            line,
        )

        if m:
            mode = int(m.group(1))
            affinity = float(m.group(2))
            affinities[mode] = affinity
        elif affinities:
            # 已经开始解析 pose 表，后面遇到非表格行即可停止
            break

    return affinities


def get_affinity_from_mol(mol):
    """
    如果日志未给出对应 mode 的 affinity，
    尝试从 SDF property 中读取。
    """
    for key in AFFINITY_PROP_CANDIDATES:
        if mol.HasProp(key):
            try:
                return float(mol.GetProp(key))
            except Exception:
                pass

    # 再做一次大小写不敏感搜索
    prop_names = list(mol.GetPropNames())
    lower_map = {x.lower(): x for x in prop_names}

    for key in AFFINITY_PROP_CANDIDATES:
        actual = lower_map.get(key.lower())
        if actual is not None:
            try:
                return float(mol.GetProp(actual))
            except Exception:
                pass

    return math.nan


def ligand_heavy_atom_centroid(mol):
    """
    返回 ligand heavy atoms 的几何中心。
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()

    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([float(pos.x), float(pos.y), float(pos.z)])

    if not coords:
        # 极端情况下没有 heavy atom，则退化为所有 atoms
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([float(pos.x), float(pos.y), float(pos.z)])

    if not coords:
        return None

    arr = np.asarray(coords, dtype=float)
    return arr.mean(axis=0)


def read_sdf_poses(sdf_path: Path):
    """
    sanitize=False：这里主要读取 Smina 输出坐标，
    避免不必要的化学 sanitize 导致 pose 丢失。
    """
    suppl = Chem.SDMolSupplier(
        str(sdf_path),
        removeHs=False,
        sanitize=False,
    )

    mols = []
    for i, mol in enumerate(suppl, start=1):
        if mol is not None:
            mols.append((i, mol))

    return mols


def collect_stage1_poses(stage1_dir: Path, max_poses_per_box=None):
    """
    收集目录中所有 box_*.sdf。

    返回：
        pose_df
        pose_mols: {(box_id, mode): RDKit Mol}
        diagnostics
    """
    sdf_files = sorted(
        stage1_dir.glob("box_*.sdf"),
        key=parse_box_id,
    )

    if not sdf_files:
        raise FileNotFoundError(
            f"在 {stage1_dir} 中没有找到 box_*.sdf"
        )

    rows = []
    pose_mols = {}

    missing_logs = []
    no_affinity = []
    sdf_log_count_mismatch = []

    for sdf_path in sdf_files:
        box_id = parse_box_id(sdf_path)
        log_path = stage1_dir / f"box_{box_id:04d}.log"

        log_aff = parse_affinities_from_log(log_path)

        if not log_path.exists():
            missing_logs.append(box_id)

        mols = read_sdf_poses(sdf_path)

        if max_poses_per_box is not None:
            mols = mols[:max_poses_per_box]

        if log_aff and len(mols) != len(log_aff):
            sdf_log_count_mismatch.append(
                (box_id, len(mols), len(log_aff))
            )

        for mode, mol in mols:
            center = ligand_heavy_atom_centroid(mol)
            if center is None:
                continue

            affinity = log_aff.get(mode, math.nan)
            affinity_source = "log"

            if not np.isfinite(affinity):
                affinity = get_affinity_from_mol(mol)
                affinity_source = "sdf_property"

            if not np.isfinite(affinity):
                no_affinity.append((box_id, mode))
                continue

            pose_mols[(box_id, mode)] = mol

            rows.append(
                {
                    "box_id": int(box_id),
                    "mode": int(mode),
                    "affinity_kcal_mol": float(affinity),
                    "pose_center_x": float(center[0]),
                    "pose_center_y": float(center[1]),
                    "pose_center_z": float(center[2]),
                    "sdf_file": str(sdf_path),
                    "log_file": str(log_path),
                    "affinity_source": affinity_source,
                }
            )

    if not rows:
        raise RuntimeError(
            "没有成功解析到带 affinity 和 3D 坐标的 docking pose。"
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["affinity_kcal_mol", "box_id", "mode"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    diagnostics = {
        "n_sdf_files": len(sdf_files),
        "missing_logs": missing_logs,
        "no_affinity": no_affinity,
        "sdf_log_count_mismatch": sdf_log_count_mismatch,
    }

    return df, pose_mols, diagnostics


def greedy_radius_cluster(pose_df: pd.DataFrame, radius: float):
    """
    affinity-prioritized greedy radius clustering。

    过程：
      1. pose 已按 affinity 从低到高排序；
      2. 选择当前未分配的最佳 affinity pose 作为 seed；
      3. 与 seed centroid 距离 <= radius 的所有未分配 pose 加入 cluster；
      4. 重复直到全部 pose 被分配。

    这不是 DBSCAN，也不是 single-linkage。
    其优点是不会因“链式连接”让一个 cluster 延伸得过长。
    """
    coords = pose_df[
        ["pose_center_x", "pose_center_y", "pose_center_z"]
    ].to_numpy(dtype=float)

    n = len(pose_df)
    unassigned = np.ones(n, dtype=bool)
    cluster_ids = np.full(n, -1, dtype=int)
    cluster_seed_index = {}

    cluster_id = 0

    for seed_idx in range(n):
        if not unassigned[seed_idx]:
            continue

        cluster_id += 1
        seed = coords[seed_idx]

        candidate_idx = np.where(unassigned)[0]
        dists = np.linalg.norm(
            coords[candidate_idx] - seed[None, :],
            axis=1,
        )

        members = candidate_idx[dists <= radius]

        cluster_ids[members] = cluster_id
        unassigned[members] = False
        cluster_seed_index[cluster_id] = seed_idx

    out = pose_df.copy()
    out["cluster_id"] = cluster_ids

    # 记录每个 pose 到其 seed 的距离
    seed_distances = []
    for i, row in out.iterrows():
        cid = int(row["cluster_id"])
        seed_idx = cluster_seed_index[cid]
        d = float(np.linalg.norm(coords[i] - coords[seed_idx]))
        seed_distances.append(d)

    out["distance_to_cluster_seed_A"] = seed_distances

    return out, cluster_seed_index


def summarize_clusters(clustered_df: pd.DataFrame, cluster_seed_index):
    rows = []

    for cluster_id, group in clustered_df.groupby("cluster_id"):
        group = group.sort_values(
            ["affinity_kcal_mol", "box_id", "mode"],
            ascending=[True, True, True],
        )

        best = group.iloc[0]

        centers = group[
            ["pose_center_x", "pose_center_y", "pose_center_z"]
        ].to_numpy(dtype=float)

        mean_center = centers.mean(axis=0)

        # 距 cluster mean center 的最大距离，辅助判断 cluster 是否过宽
        max_radius_from_mean = float(
            np.max(
                np.linalg.norm(
                    centers - mean_center[None, :],
                    axis=1,
                )
            )
        )

        affinities = group["affinity_kcal_mol"].to_numpy(dtype=float)

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_poses": int(len(group)),
                "n_unique_boxes": int(group["box_id"].nunique()),
                "best_affinity_kcal_mol": float(np.min(affinities)),
                "median_affinity_kcal_mol": float(np.median(affinities)),
                "mean_affinity_kcal_mol": float(np.mean(affinities)),
                "center_x": float(mean_center[0]),
                "center_y": float(mean_center[1]),
                "center_z": float(mean_center[2]),
                "max_pose_center_distance_from_cluster_mean_A": max_radius_from_mean,
                "representative_box_id": int(best["box_id"]),
                "representative_mode": int(best["mode"]),
                "representative_sdf_file": best["sdf_file"],
                "representative_log_file": best["log_file"],
            }
        )

    cdf = pd.DataFrame(rows)

    # 先按 best affinity，随后 median affinity，再按支持 box 数
    cdf = cdf.sort_values(
        [
            "best_affinity_kcal_mol",
            "median_affinity_kcal_mol",
            "n_unique_boxes",
            "n_poses",
        ],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    cdf["cluster_rank"] = np.arange(1, len(cdf) + 1)

    # 调整列顺序
    cols = [
        "cluster_rank",
        "cluster_id",
        "n_poses",
        "n_unique_boxes",
        "best_affinity_kcal_mol",
        "median_affinity_kcal_mol",
        "mean_affinity_kcal_mol",
        "center_x",
        "center_y",
        "center_z",
        "max_pose_center_distance_from_cluster_mean_A",
        "representative_box_id",
        "representative_mode",
        "representative_sdf_file",
        "representative_log_file",
    ]

    return cdf[cols]


def save_representative_pose(mol, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))
    writer.SetKekulize(False)
    writer.write(mol)
    writer.close()


def get_expected_box_ids(boxes_csv):
    if boxes_csv is None:
        return None

    p = Path(boxes_csv)
    if not p.exists():
        raise FileNotFoundError(f"boxes CSV 不存在: {p}")

    df = pd.read_csv(p)
    if "box_id" not in df.columns:
        raise ValueError(
            f"{p} 中缺少 box_id 列。columns={list(df.columns)}"
        )

    return set(df["box_id"].astype(int).tolist())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage1-dir",
        default=(
            "eval_outputs/deepprotacs_docking/"
            "stage1/e3_0001"
        ),
        help="包含 box_*.sdf / box_*.log 的 Stage 1 目录",
    )

    parser.add_argument(
        "--boxes",
        default=(
            "eval_outputs/deepprotacs_docking/"
            "smina_grids/e3_0001/"
            "e3_0001_boxes_active.csv"
        ),
        help=(
            "该 protein 的 active boxes CSV。"
            "用于报告当前完成比例；不要求全部完成。"
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "分析输出目录。默认自动写到 "
            "eval_outputs/deepprotacs_docking/"
            "stage1_analysis/<pair_id>"
        ),
    )

    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=5.0,
        help="pose centroid 空间聚类半径，单位 Å；默认 5.0",
    )

    parser.add_argument(
        "--top-regions",
        type=int,
        default=3,
        help="输出给 Stage 2 的 candidate region 数；默认 3",
    )

    parser.add_argument(
        "--stage2-box-size",
        type=float,
        default=30.0,
        help="写入 candidate_regions.csv 的 Stage 2 box size；默认 30 Å",
    )

    parser.add_argument(
        "--max-poses-per-box",
        type=int,
        default=None,
        help=(
            "每个 box 最多使用多少个 pose；"
            "默认 None=使用 Smina 输出的全部 pose。"
        ),
    )

    args = parser.parse_args()

    stage1_dir = Path(args.stage1_dir).resolve()
    if not stage1_dir.exists():
        raise FileNotFoundError(
            f"Stage 1 目录不存在: {stage1_dir}"
        )

    pair_id = stage1_dir.name

    if args.out_dir is None:
        # stage1/e3_0001 -> ../../stage1_analysis/e3_0001
        docking_root = stage1_dir.parent.parent
        out_dir = docking_root / "stage1_analysis" / pair_id
    else:
        out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cluster_radius <= 0:
        raise ValueError("--cluster-radius 必须 > 0")

    if args.top_regions <= 0:
        raise ValueError("--top-regions 必须 > 0")

    print("=" * 72)
    print("Smina Stage 1 analysis")
    print(f"pair_id:          {pair_id}")
    print(f"stage1_dir:       {stage1_dir}")
    print(f"cluster_radius:   {args.cluster_radius:.2f} Å")
    print(f"top_regions:      {args.top_regions}")
    print(f"out_dir:          {out_dir}")
    print("=" * 72)

    pose_df, pose_mols, diagnostics = collect_stage1_poses(
        stage1_dir,
        max_poses_per_box=args.max_poses_per_box,
    )

    completed_box_ids = set(
        pose_df["box_id"].astype(int).unique().tolist()
    )

    expected_box_ids = get_expected_box_ids(args.boxes)
    if expected_box_ids is not None:
        missing_box_ids = sorted(expected_box_ids - completed_box_ids)
        extra_box_ids = sorted(completed_box_ids - expected_box_ids)
    else:
        missing_box_ids = []
        extra_box_ids = []

    clustered_df, cluster_seed_index = greedy_radius_cluster(
        pose_df,
        radius=float(args.cluster_radius),
    )

    cluster_df = summarize_clusters(
        clustered_df,
        cluster_seed_index,
    )

    top_n = min(args.top_regions, len(cluster_df))
    top_df = cluster_df.head(top_n).copy()

    top_df.insert(
        0,
        "region_rank",
        np.arange(1, top_n + 1),
    )

    top_df["size_x"] = float(args.stage2_box_size)
    top_df["size_y"] = float(args.stage2_box_size)
    top_df["size_z"] = float(args.stage2_box_size)

    # 保存 representative pose
    representative_paths = []

    for _, row in top_df.iterrows():
        region_rank = int(row["region_rank"])
        box_id = int(row["representative_box_id"])
        mode = int(row["representative_mode"])

        mol = pose_mols.get((box_id, mode))
        if mol is None:
            raise RuntimeError(
                f"找不到代表 pose: box={box_id}, mode={mode}"
            )

        rep_path = (
            out_dir
            / f"region_{region_rank:02d}_representative.sdf"
        )

        save_representative_pose(mol, rep_path)
        representative_paths.append(str(rep_path))

    top_df["representative_pose_file"] = representative_paths

    # 输出 CSV
    all_poses_path = out_dir / "all_stage1_poses.csv"
    clusters_path = out_dir / "stage1_clusters.csv"
    candidates_path = out_dir / "candidate_regions.csv"

    clustered_df.to_csv(
        all_poses_path,
        index=False,
        encoding="utf-8-sig",
    )

    cluster_df.to_csv(
        clusters_path,
        index=False,
        encoding="utf-8-sig",
    )

    candidate_cols = [
        "region_rank",
        "cluster_rank",
        "cluster_id",
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "best_affinity_kcal_mol",
        "median_affinity_kcal_mol",
        "mean_affinity_kcal_mol",
        "n_poses",
        "n_unique_boxes",
        "representative_box_id",
        "representative_mode",
        "representative_pose_file",
        "representative_sdf_file",
        "representative_log_file",
    ]

    top_df[candidate_cols].to_csv(
        candidates_path,
        index=False,
        encoding="utf-8-sig",
    )

    # diagnostics / summary
    summary_lines = [
        f"pair_id: {pair_id}",
        f"stage1_dir: {stage1_dir}",
        "",
        "========== completion ==========",
        f"completed boxes with parsed poses: {len(completed_box_ids)}",
    ]

    if expected_box_ids is not None:
        summary_lines.extend(
            [
                f"expected active boxes: {len(expected_box_ids)}",
                (
                    "completion rate: "
                    f"{len(completed_box_ids) / len(expected_box_ids):.2%}"
                    if expected_box_ids
                    else "completion rate: NA"
                ),
                f"missing active boxes: {len(missing_box_ids)}",
                (
                    "missing box ids: "
                    + ",".join(map(str, missing_box_ids))
                    if missing_box_ids
                    else "missing box ids: none"
                ),
                f"extra boxes not in active CSV: {len(extra_box_ids)}",
            ]
        )

    summary_lines.extend(
        [
            "",
            "========== parsed Stage 1 ==========",
            f"SDF files: {diagnostics['n_sdf_files']}",
            f"poses used: {len(clustered_df)}",
            f"clusters: {len(cluster_df)}",
            f"cluster radius: {args.cluster_radius:.2f} Å",
            f"missing log files: {len(diagnostics['missing_logs'])}",
            f"poses without affinity: {len(diagnostics['no_affinity'])}",
            (
                "SDF/log pose-count mismatches: "
                f"{len(diagnostics['sdf_log_count_mismatch'])}"
            ),
            "",
            "========== Top candidate regions ==========",
        ]
    )

    for _, row in top_df.iterrows():
        summary_lines.append(
            (
                f"Region {int(row['region_rank'])}: "
                f"center=({row['center_x']:.3f}, "
                f"{row['center_y']:.3f}, "
                f"{row['center_z']:.3f}) Å; "
                f"best={row['best_affinity_kcal_mol']:.3f} kcal/mol; "
                f"median={row['median_affinity_kcal_mol']:.3f} kcal/mol; "
                f"poses={int(row['n_poses'])}; "
                f"boxes={int(row['n_unique_boxes'])}; "
                f"representative=box "
                f"{int(row['representative_box_id'])}, "
                f"mode {int(row['representative_mode'])}"
            )
        )

    if missing_box_ids:
        summary_lines.extend(
            [
                "",
                "NOTE:",
                (
                    "当前不是完整 Stage 1：仍有 active boxes 未完成。"
                    "这些 candidate regions 仅可用于 pilot 流程验证，"
                    "不能作为正式论文中的最终 blind-docking 结果。"
                ),
            ]
        )
    else:
        summary_lines.extend(
            [
                "",
                "NOTE:",
                (
                    "当前 active boxes 已全部完成，"
                    "candidate regions 可用于正式 Stage 2。"
                ),
            ]
        )

    if diagnostics["sdf_log_count_mismatch"]:
        summary_lines.extend(
            [
                "",
                "SDF/log pose-count mismatches:",
            ]
        )
        for box_id, n_sdf, n_log in diagnostics[
            "sdf_log_count_mismatch"
        ]:
            summary_lines.append(
                f"  box {box_id}: SDF={n_sdf}, log={n_log}"
            )

    summary_path = out_dir / "stage1_analysis_summary.txt"
    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("")
    print("========== Stage 1 summary ==========")
    print(f"completed boxes: {len(completed_box_ids)}")

    if expected_box_ids is not None:
        print(f"expected boxes:  {len(expected_box_ids)}")
        print(f"missing boxes:   {len(missing_box_ids)}")

    print(f"poses used:      {len(clustered_df)}")
    print(f"clusters:        {len(cluster_df)}")

    print("")
    print("========== Top candidate regions ==========")

    for _, row in top_df.iterrows():
        print(
            f"Region {int(row['region_rank'])}: "
            f"center=({row['center_x']:.2f}, "
            f"{row['center_y']:.2f}, "
            f"{row['center_z']:.2f}) Å | "
            f"best={row['best_affinity_kcal_mol']:.2f} | "
            f"median={row['median_affinity_kcal_mol']:.2f} | "
            f"poses={int(row['n_poses'])} | "
            f"boxes={int(row['n_unique_boxes'])}"
        )

    print("")
    print("Outputs:")
    print(f"  {all_poses_path}")
    print(f"  {clusters_path}")
    print(f"  {candidates_path}")
    print(f"  {summary_path}")

    if missing_box_ids:
        print("")
        print(
            "WARNING: 当前只完成了部分 active boxes。"
            "本次 Top regions 仅用于 pilot 验证。"
        )


if __name__ == "__main__":
    main()
