#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate overlapping Smina search boxes that cover an entire protein.

Default protocol:
- cubic box: 30 x 30 x 30 Å
- step: 15 Å (50% overlap)
- margin around protein bounds: 5 Å
- keep candidate boxes whose center is within 10 Å of any protein heavy atom
- verify that every protein heavy atom is still covered by at least one retained box

Outputs:
  <prefix>_boxes_all.csv
  <prefix>_boxes_active.csv
  <prefix>_summary.txt
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError as e:
    raise SystemExit(
        "This script requires scipy. Install it with: conda install scipy"
    ) from e


def read_pdb_heavy_atoms(pdb_path: Path, include_hetatm: bool = False):
    coords = []
    record_types = ("ATOM  ", "HETATM") if include_hetatm else ("ATOM  ",)

    with pdb_path.open("r", errors="ignore") as f:
        for line in f:
            if not line.startswith(record_types):
                continue

            element = line[76:78].strip().upper()
            if not element:
                # Fallback from atom name.
                atom_name = line[12:16].strip()
                element = "".join(ch for ch in atom_name if ch.isalpha())[:2].upper()

            if element == "H":
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            coords.append((x, y, z))

    if not coords:
        raise ValueError(f"No protein heavy atoms found in: {pdb_path}")

    return np.asarray(coords, dtype=np.float64)


def make_axis(vmin, vmax, box_size, step, margin):
    lo = vmin - margin
    hi = vmax + margin
    half = box_size / 2.0

    # One box is enough if the expanded span is smaller than the box.
    if (hi - lo) <= box_size:
        return np.asarray([(lo + hi) / 2.0], dtype=np.float64)

    start = lo + half
    end = hi - half

    values = list(np.arange(start, end + 1e-8, step, dtype=np.float64))

    # Add a final shifted box if the regular grid does not fully cover the high end.
    if not values:
        values = [(lo + hi) / 2.0]
    elif values[-1] + half < hi - 1e-8:
        values.append(hi - half)

    return np.asarray(values, dtype=np.float64)


def count_atoms_in_cube(tree, coords, center, half):
    # Spherical prefilter, then exact cube test.
    radius = math.sqrt(3.0) * half
    ids = tree.query_ball_point(center, r=radius)
    if not ids:
        return 0
    pts = coords[np.asarray(ids, dtype=int)]
    return int(np.sum(np.all(np.abs(pts - center) <= half, axis=1)))


def atom_covered_by_any_box(atom, centers, half):
    if len(centers) == 0:
        return False
    return bool(np.any(np.all(np.abs(centers - atom) <= half, axis=1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True, help="AlphaFold/protein PDB file")
    parser.add_argument("--out-dir", default="smina_grid", help="Output directory")
    parser.add_argument("--prefix", default="protein", help="Output filename prefix")
    parser.add_argument("--box-size", type=float, default=30.0, help="Box size in Å")
    parser.add_argument("--step", type=float, default=15.0, help="Grid step in Å")
    parser.add_argument("--margin", type=float, default=5.0, help="Margin around protein in Å")
    parser.add_argument(
        "--max-center-distance",
        type=float,
        default=10.0,
        help="Retain boxes whose center is within this distance (Å) of a protein heavy atom",
    )
    parser.add_argument(
        "--include-hetatm",
        action="store_true",
        help="Also use HETATM records when constructing the protein bounds",
    )
    args = parser.parse_args()

    pdb_path = Path(args.pdb)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords = read_pdb_heavy_atoms(pdb_path, include_hetatm=args.include_hetatm)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = maxs - mins

    axes = [
        make_axis(mins[i], maxs[i], args.box_size, args.step, args.margin)
        for i in range(3)
    ]

    grid = np.array(
        np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    ).reshape(3, -1).T

    tree = cKDTree(coords)
    nearest_dist, _ = tree.query(grid, k=1)

    half = args.box_size / 2.0
    atom_counts = np.asarray(
        [count_atoms_in_cube(tree, coords, c, half) for c in grid],
        dtype=np.int64,
    )

    active = nearest_dist <= args.max_center_distance

    # Safety check: retained boxes should still cover every protein heavy atom.
    active_centers = grid[active]
    covered = np.asarray(
        [atom_covered_by_any_box(atom, active_centers, half) for atom in coords],
        dtype=bool,
    )

    # If any protein atoms are not covered, activate the nearest grid box(es).
    if not np.all(covered):
        uncovered = coords[~covered]
        grid_tree = cKDTree(grid)
        _, nearest_grid_idx = grid_tree.query(uncovered, k=1)
        active[np.unique(nearest_grid_idx)] = True
        active_centers = grid[active]

        covered = np.asarray(
            [atom_covered_by_any_box(atom, active_centers, half) for atom in coords],
            dtype=bool,
        )

    all_csv = out_dir / f"{args.prefix}_boxes_all.csv"
    active_csv = out_dir / f"{args.prefix}_boxes_active.csv"
    summary_txt = out_dir / f"{args.prefix}_summary.txt"

    header = [
        "box_id",
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "nearest_protein_atom_A",
        "protein_atoms_in_box",
        "active",
    ]

    rows = []
    for i, (c, d, n, is_active) in enumerate(
        zip(grid, nearest_dist, atom_counts, active), start=1
    ):
        rows.append(
            [
                i,
                f"{c[0]:.3f}",
                f"{c[1]:.3f}",
                f"{c[2]:.3f}",
                f"{args.box_size:.3f}",
                f"{args.box_size:.3f}",
                f"{args.box_size:.3f}",
                f"{d:.3f}",
                int(n),
                int(is_active),
            ]
        )

    with all_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    with active_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows([r for r in rows if r[-1] == 1])

    summary = f"""PDB: {pdb_path}
Protein heavy atoms: {len(coords)}

Protein coordinate bounds:
  X: {mins[0]:.3f} to {maxs[0]:.3f} Å   span={spans[0]:.3f} Å
  Y: {mins[1]:.3f} to {maxs[1]:.3f} Å   span={spans[1]:.3f} Å
  Z: {mins[2]:.3f} to {maxs[2]:.3f} Å   span={spans[2]:.3f} Å

Grid parameters:
  box_size = {args.box_size:.1f} Å
  step = {args.step:.1f} Å
  overlap = {100.0 * (1.0 - args.step / args.box_size):.1f}%
  margin = {args.margin:.1f} Å
  max_center_distance = {args.max_center_distance:.1f} Å

Grid dimensions:
  nx={len(axes[0])}, ny={len(axes[1])}, nz={len(axes[2])}

All boxes: {len(grid)}
Active boxes: {int(active.sum())}
Protein heavy-atom coverage by active boxes: {100.0 * covered.mean():.2f}%

Outputs:
  {all_csv}
  {active_csv}
"""
    summary_txt.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
