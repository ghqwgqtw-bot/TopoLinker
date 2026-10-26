#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze Smina Stage 1 blind-docking results and identify candidate
binding regions for Stage 2 refinement.

Workflow:
    Whole-protein tiled blind docking
        -> Collect poses from multiple Stage 1 boxes
        -> Extract docking affinities and ligand heavy-atom centroids
        -> Cluster poses by spatial proximity
        -> Rank clusters and select the top candidate regions
        -> Refine selected regions in Stage 2

Default pilot inputs:
    eval_outputs/deepprotacs_docking/stage1/e3_0001
    eval_outputs/deepprotacs_docking/smina_grids/e3_0001/e3_0001_boxes_active.csv

Notes:
    1. Existing Stage 1 SDF and log files are read without modification.
    2. Partial Stage 1 results can be analyzed for pilot validation.
    3. Final analysis should use all expected active docking boxes.
    4. Each docking pose is represented by its ligand heavy-atom centroid.
    5. The default spatial clustering radius is 5 Å.
    6. Affinity-prioritized greedy radius clustering is used:
       - The unassigned pose with the lowest affinity becomes a seed.
       - All unassigned poses within the specified seed radius are grouped.
       - The process repeats until all poses are assigned.
       This avoids transitive chaining across distant pose centers.
    7. Clusters are ranked by best affinity, median affinity,
       number of supporting boxes, and total pose count.
    8. Each candidate region is centered at the mean centroid
       of all poses assigned to its cluster.
    9. The default output contains the top three candidate regions,
       each with a Stage 2 box size of 30 Å.

Dependencies:
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
        raise ValueError(f"Cannot parse box_id from filename: {path.name}")
    return int(m.group(1))


def parse_affinities_from_log(log_path: Path):
    """
    Parse mode-specific docking affinities from a Smina log file.

    Returns:
        Dictionary mapping docking mode to affinity (kcal/mol).

    Example:
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

        # Skip blank lines and non-data lines before the affinity entries.
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
            # Stop when a non-table line appears after parsing affinity entries.
            break

    return affinities


def get_affinity_from_mol(mol):
    """
    Extract docking affinity from SDF molecule properties.

    Used as a fallback when affinity is unavailable from the Smina log.
    """
    for key in AFFINITY_PROP_CANDIDATES:
        if mol.HasProp(key):
            try:
                return float(mol.GetProp(key))
            except Exception:
                pass

    # Retry with case-insensitive property-name matching.
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
    Return the geometric centroid of the ligand's heavy atoms.
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
        # Fall back to all atoms if no heavy atoms are available.
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([float(pos.x), float(pos.y), float(pos.z)])

    if not coords:
        return None

    arr = np.asarray(coords, dtype=float)
    return arr.mean(axis=0)


def read_sdf_poses(sdf_path: Path):
    """
    Read Smina docking poses without RDKit sanitization.

    Preserve pose coordinates without discarding molecules due to
    sanitization failures.
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
    Collect docking poses from all box_*.sdf files in a Stage 1 directory.

    Returns:
        pose_df: DataFrame containing affinity, coordinates, and pose metadata.
        pose_mols: Dictionary mapping (box_id, mode) to RDKit molecules.
        diagnostics: Information about missing logs and parsing issues.
    """
    sdf_files = sorted(
        stage1_dir.glob("box_*.sdf"),
        key=parse_box_id,
    )

    if not sdf_files:
        raise FileNotFoundError(
            f"No box_*.sdf files found in {stage1_dir}"
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
            "No docking poses with valid affinity and 3D coordinates were parsed."
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
    Cluster docking poses using affinity-prioritized greedy radius clustering.

    The input poses are ordered by ascending affinity. Each unassigned
    pose becomes a cluster seed, and all remaining poses within the
    specified radius of that seed are assigned to the same cluster.

    Continue until every pose has been assigned. Unlike DBSCAN or
    single-linkage clustering, membership is determined by distance
    to the cluster seed rather than by transitive connectivity.
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

    # Record each pose centroid's distance to its cluster seed.
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

        # Measure the maximum pose-centroid distance from the cluster mean.
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

    # Rank by best affinity, median affinity, unique box count, and pose count.
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

    # Arrange columns for the cluster summary.
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
        raise FileNotFoundError(f"Boxes CSV does not exist: {p}")

    df = pd.read_csv(p)
    if "box_id" not in df.columns:
        raise ValueError(
            f"Missing 'box_id' column in {p}. columns={list(df.columns)}"
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
        help="Stage 1 directory containing box_*.sdf and box_*.log files",
    )

    parser.add_argument(
        "--boxes",
        default=(
            "eval_outputs/deepprotacs_docking/"
            "smina_grids/e3_0001/"
            "e3_0001_boxes_active.csv"
        ),
        help=(
            "CSV listing active docking boxes for this protein. "
            "Used to report Stage 1 coverage; partial results are supported."
        ),
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Analysis output directory. By default, results are saved under "
            "eval_outputs/deepprotacs_docking/"
            "stage1_analysis/<pair_id>"
        ),
    )

    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=5.0,
        help="Spatial clustering radius for pose centroids in Å (default: 5.0)",
    )

    parser.add_argument(
        "--top-regions",
        type=int,
        default=3,
        help="Number of candidate regions selected for Stage 2 (default: 3)",
    )

    parser.add_argument(
        "--stage2-box-size",
        type=float,
        default=30.0,
        help="Stage 2 box size written to candidate_regions.csv (default: 30 Å)",
    )

    parser.add_argument(
        "--max-poses-per-box",
        type=int,
        default=None,
        help=(
            "Maximum number of poses to use per box. "
            "By default, all available Smina poses are used."
        ),
    )

    args = parser.parse_args()

    stage1_dir = Path(args.stage1_dir).resolve()
    if not stage1_dir.exists():
        raise FileNotFoundError(
            f"Stage 1 directory does not exist: {stage1_dir}"
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
        raise ValueError("--cluster-radius must be greater than 0")

    if args.top_regions <= 0:
        raise ValueError("--top-regions must be greater than 0")

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

    # Save representative docking poses for the selected regions.
    representative_paths = []

    for _, row in top_df.iterrows():
        region_rank = int(row["region_rank"])
        box_id = int(row["representative_box_id"])
        mode = int(row["representative_mode"])

        mol = pose_mols.get((box_id, mode))
        if mol is None:
            raise RuntimeError(
                f"Representative pose not found: box={box_id}, mode={mode}"
            )

        rep_path = (
            out_dir
            / f"region_{region_rank:02d}_representative.sdf"
        )

        save_representative_pose(mol, rep_path)
        representative_paths.append(str(rep_path))

    top_df["representative_pose_file"] = representative_paths

    # Save pose, cluster, and candidate-region CSV files.
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
                    "Stage 1 coverage is incomplete: some expected active "
                    "boxes have no usable poses. These candidate regions "
                    "are suitable for pilot validation, not final "
                    "blind-docking results."
                ),
            ]
        )
    else:
        summary_lines.extend(
            [
                "",
                "NOTE:",
                (
                    "All expected active boxes have at least one usable pose. "
                    "Candidate regions are available for Stage 2 refinement."
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
            "WARNING: Some active boxes have no usable Stage 1 poses. "
            "The selected regions should be treated as pilot results only."
        )


if __name__ == "__main__":
    main()
