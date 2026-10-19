#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final one-command DeepPROTACs external evaluation for TopoLinker.

Reads:
    eval_outputs/test_pretrain/records.pkl
    eval_outputs/test_rl/records.pkl
    eval_outputs/test_rl_final/records.pkl

Locked formal protocol:
    Grid: 30 Å box, 15 Å step, 5 Å margin, active-center distance 10 Å
    Stage1: up to 24 deterministic FPS boxes, ex=4, modes=10, seed=2026
    Stage1 clustering: radius=5 Å, top-1 region
    Stage2: top-1, ex=32, modes=20, seed=2026
    Pocket: whole residues with any heavy atom <=5 Å from docked ligand
    DeepPROTACs: official single_prediction.py via run_deepprotacs_pilot_v4.py

Failure policy:
    Pair-preparation failures are recorded, not manually repaired.
    Samples depending on failed pair inputs are missing_pair_input.
    They remain in the strict/all-generated denominators.
    Test contexts whose required AlphaFold2 structures are unavailable
(P03406, P03436, P36969) are excluded before DeepPROTACs denominators
are constructed.

For the remaining eligible contexts, pair-preparation failures are
recorded as missing_pair_input and remain in the denominators.

Resumable:
    Existing Smina outputs, pocket MOL2s and successful DeepPROTACs results
    are reused automatically.

Required helper files kept in evaluation/:
    prepare_docking_pairs.py
    prepare_smina_inputs.py
    make_smina_grid.py
    analyze_smina_stage1.py
    extract_deepprotacs_pocket.py
    prepare_deepprotacs_pilot_sample.py
    run_deepprotacs_pilot_v4.py
        (or the identical v4-compatible code under run_deepprotacs_pilot.py)

Usage:
    python evaluation/evaluate_deepprotacs_three_stages.py

Quick inference smoke test:
    python evaluation/evaluate_deepprotacs_three_stages.py --max-inference-samples 5

Prepare docking/pockets only:
    python evaluation/evaluate_deepprotacs_three_stages.py --skip-inference

Inference only from already prepared pockets:
    python evaluation/evaluate_deepprotacs_three_stages.py --skip-docking
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


STAGES = ("pretrain", "rl1", "rl2")
DEFAULT_RECORDS = {
    "pretrain": Path("eval_outputs/test_pretrain/records.pkl"),
    "rl1": Path("eval_outputs/test_rl/records.pkl"),
    "rl2": Path("eval_outputs/test_rl_final/records.pkl"),
}
DEFAULT_OUT_ROOT = Path("eval_outputs/deepprotacs_external_eval")


def resolve_path(value: str | Path, root: Path) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def utf8_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=utf8_env(),
        timeout=timeout,
        shell=False,
    )


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def file_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def import_module_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_header(title: str) -> None:
    print("")
    print("=" * 100)
    print(title)
    print("=" * 100)


# -----------------------------------------------------------------------------
# Pair tables + Smina input preparation
# -----------------------------------------------------------------------------

def prepare_pair_tables(
    project_root: Path,
    evaluation_dir: Path,
    records_paths: Dict[str, Path],
    out_root: Path,
) -> Tuple[Path, Path, Path]:
    script = evaluation_dir / "prepare_docking_pairs.py"
    require_file(script)

    print_header("1/8 Prepare shared docking-pair tables")

    proc = run_cmd(
        [
            sys.executable,
            str(script),
            "--pretrain", str(records_paths["pretrain"]),
            "--rl1", str(records_paths["rl1"]),
            "--rl2", str(records_paths["rl2"]),
            "--project-root", str(project_root),
            "--out-dir", str(out_root),
        ],
        cwd=project_root,
        timeout=1800,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        raise RuntimeError("prepare_docking_pairs.py failed")

    all_csv = out_root / "docking_pairs_all.csv"
    poi_csv = out_root / "docking_pairs_poi.csv"
    e3_csv = out_root / "docking_pairs_e3.csv"
    for p in [all_csv, poi_csv, e3_csv]:
        require_file(p)
    return all_csv, poi_csv, e3_csv


def prepare_smina_inputs_all(
    project_root: Path,
    evaluation_dir: Path,
    pairs_csv: Path,
    out_root: Path,
) -> pd.DataFrame:
    script = evaluation_dir / "prepare_smina_inputs.py"
    require_file(script)

    print_header("2/8 Prepare Smina PDBQT inputs")

    proc = run_cmd(
        [
            sys.executable,
            str(script),
            "--pairs", str(pairs_csv),
            "--out-dir", str(out_root),
            "--all",
        ],
        cwd=project_root,
        timeout=7200,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())

    status_csv = out_root / "prepare_smina_inputs_status.csv"
    require_file(status_csv)
    return read_csv(status_csv)


# -----------------------------------------------------------------------------
# Grid + deterministic FPS
# -----------------------------------------------------------------------------

def fps_select(active: pd.DataFrame, n_select: int) -> pd.DataFrame:
    g = active.copy()
    for c in ["box_id", "center_x", "center_y", "center_z"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna(
        subset=["box_id", "center_x", "center_y", "center_z"]
    ).copy()
    if g.empty:
        raise RuntimeError("No active boxes")

    if len(g) <= n_select:
        out = g.sort_values("box_id", kind="mergesort").copy()
        out["fps_rank"] = np.arange(1, len(out) + 1)
        return out

    xyz = g[["center_x", "center_y", "center_z"]].to_numpy(float)
    box_ids = g["box_id"].astype(int).to_numpy()

    centroid = xyz.mean(axis=0)
    d0 = np.linalg.norm(xyz - centroid[None, :], axis=1)
    cand = np.where(np.isclose(d0, d0.min()))[0]
    first = int(cand[np.argmin(box_ids[cand])])

    chosen = [first]
    min_dist = np.linalg.norm(xyz - xyz[first][None, :], axis=1)
    min_dist[first] = -np.inf

    while len(chosen) < n_select:
        m = np.max(min_dist)
        cand = np.where(np.isclose(min_dist, m))[0]
        nxt = int(cand[np.argmin(box_ids[cand])])
        chosen.append(nxt)

        d_new = np.linalg.norm(xyz - xyz[nxt][None, :], axis=1)
        min_dist = np.minimum(min_dist, d_new)
        for idx in chosen:
            min_dist[idx] = -np.inf

    out = g.iloc[chosen].copy()
    out["fps_rank"] = np.arange(1, len(out) + 1)
    return out


def make_grids_and_fps(
    project_root: Path,
    evaluation_dir: Path,
    pair_rows: pd.DataFrame,
    good_pair_ids: set[str],
    grids_root: Path,
    fps_root: Path,
    box_size: float,
    step: float,
    margin: float,
    max_center_distance: float,
    fps_boxes: int,
) -> Dict[str, Path]:
    script = evaluation_dir / "make_smina_grid.py"
    require_file(script)

    print_header("3/8 Generate active grids and select FPS boxes")

    fps_paths: Dict[str, Path] = {}

    for i, row in pair_rows.iterrows():
        pair_id = str(row["pair_id"])
        if pair_id not in good_pair_ids:
            continue

        try:
            protein = resolve_path(str(row["protein_path"]), project_root)
            grid_dir = grids_root / pair_id
            active_csv = grid_dir / f"{pair_id}_boxes_active.csv"

            if not file_nonempty(active_csv):
                grid_dir.mkdir(parents=True, exist_ok=True)
                proc = run_cmd(
                    [
                        sys.executable,
                        str(script),
                        "--pdb", str(protein),
                        "--out-dir", str(grid_dir),
                        "--prefix", pair_id,
                        "--box-size", str(box_size),
                        "--step", str(step),
                        "--margin", str(margin),
                        "--max-center-distance", str(max_center_distance),
                    ],
                    cwd=project_root,
                    timeout=1800,
                )
                if proc.returncode != 0 or not file_nonempty(active_csv):
                    raise RuntimeError("grid generation failed")

            active = read_csv(active_csv)
            selected = fps_select(active, fps_boxes)

            fps_root.mkdir(parents=True, exist_ok=True)
            fps_csv = fps_root / f"{pair_id}_fps{fps_boxes}.csv"
            selected.to_csv(fps_csv, index=False, encoding="utf-8-sig")
            fps_paths[pair_id] = fps_csv

            if (i + 1) % 25 == 0 or i == len(pair_rows) - 1:
                print(
                    f"[{i+1}/{len(pair_rows)}] "
                    f"FPS-ready={len(fps_paths)}"
                )

        except Exception as exc:
            print(f"[FAIL] {pair_id}: {type(exc).__name__}: {exc}")

    return fps_paths


# -----------------------------------------------------------------------------
# Smina job pool
# -----------------------------------------------------------------------------

def parse_best_affinity(log_path: Path) -> float:
    if not log_path.is_file():
        return float("nan")
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    in_table = False
    for line in text.splitlines():
        if "mode" in line.lower() and "affinity" in line.lower():
            in_table = True
            continue
        if not in_table:
            continue
        m = re.match(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", line)
        if m:
            return float(m.group(1))
    return float("nan")


def run_smina_job(task: Dict[str, Any], smina: str, project_root: Path, timeout: int):
    out_file = Path(task["output_file"])
    log_file = Path(task["log_file"])
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if file_nonempty(out_file):
        affinity = parse_best_affinity(log_file)

        if np.isfinite(affinity):
            return {
                **task,
                "status": "ok_reused",
                "return_code": 0,
                "elapsed_s": float("nan"),
                "best_affinity": affinity,
            }

        # SDF 存在但日志不完整：
        # 删除损坏/中断输出，然后重新 docking
        try:
            out_file.unlink()
        except Exception:
            pass

        try:
            log_file.unlink()
        except Exception:
            pass

    cmd = [
        smina,
        "--receptor", task["receptor"],
        "--ligand", task["ligand"],
        "--center_x", str(task["center_x"]),
        "--center_y", str(task["center_y"]),
        "--center_z", str(task["center_z"]),
        "--size_x", str(task["size_x"]),
        "--size_y", str(task["size_y"]),
        "--size_z", str(task["size_z"]),
        "--exhaustiveness", str(task["exhaustiveness"]),
        "--num_modes", str(task["num_modes"]),
        "--energy_range", str(task["energy_range"]),
        "--min_rmsd_filter", str(task["min_rmsd_filter"]),
        "--seed", str(task["seed"]),
        "--cpu", str(task["cpu_per_job"]),
        "--out", str(out_file),
        "--log", str(log_file),
    ]

    t0 = time.perf_counter()
    try:
        proc = run_cmd(cmd, cwd=project_root, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            out_file.unlink()
        except Exception:
            pass

        try:
            log_file.unlink()
        except Exception:
            pass

        return {
            **task,
            "status": "timeout",
            "return_code": -999,
            "elapsed_s": round(
                time.perf_counter() - t0,
                3,
            ),
            "best_affinity": float("nan"),
        }


    elapsed = time.perf_counter() - t0
    affinity = parse_best_affinity(log_file)

    ok = (
            proc.returncode == 0
            and file_nonempty(out_file)
            and np.isfinite(affinity)
    )

    if not ok:
        out_file.with_suffix(".stdout.txt").write_text(
            proc.stdout or "",
            encoding="utf-8",
            errors="ignore",
        )

    return {
        **task,
        "status": "ok" if ok else "failed",
        "return_code": int(proc.returncode),
        "elapsed_s": round(elapsed, 3),
        "best_affinity": affinity if ok else float("nan"),
    }


def run_pool(
    tasks: List[Dict[str, Any]],
    *,
    workers: int,
    smina: str,
    project_root: Path,
    timeout: int,
    status_csv: Path,
    title: str,
) -> pd.DataFrame:
    print_header(title)

    if not tasks:
        return pd.DataFrame()

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut_map = {
            pool.submit(
                run_smina_job,
                task,
                smina,
                project_root,
                timeout,
            ): task
            for task in tasks
        }

        for n, fut in enumerate(as_completed(fut_map), start=1):
            task = fut_map[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {
                    **task,
                    "status": f"wrapper_exception:{type(exc).__name__}:{exc}",
                    "return_code": -998,
                    "elapsed_s": float("nan"),
                    "best_affinity": float("nan"),
                }
            rows.append(row)

            if n <= 20 or n % 100 == 0 or n == len(tasks):
                print(
                    f"[{n}/{len(tasks)}] {row['pair_id']} | "
                    f"{row['status']} | {row.get('elapsed_s')} s"
                )

            if n % 50 == 0 or n == len(tasks):
                pd.DataFrame(rows).to_csv(
                    status_csv,
                    index=False,
                    encoding="utf-8-sig",
                )

    df = pd.DataFrame(rows)
    df.to_csv(status_csv, index=False, encoding="utf-8-sig")
    return df


def build_stage1_tasks(
    pair_rows: pd.DataFrame,
    fps_paths: Dict[str, Path],
    smina_inputs_root: Path,
    stage1_root: Path,
    box_size: float,
    ex: int,
    modes: int,
    energy_range: float,
    min_rmsd: float,
    seed: int,
    cpu_per_job: int,
) -> List[Dict[str, Any]]:
    tasks = []

    for _, pair in pair_rows.iterrows():
        pair_id = str(pair["pair_id"])
        if pair_id not in fps_paths:
            continue

        receptor = smina_inputs_root / pair_id / "receptor.pdbqt"
        ligand = smina_inputs_root / pair_id / "ligand.pdbqt"
        if not file_nonempty(receptor) or not file_nonempty(ligand):
            continue

        boxes = read_csv(fps_paths[pair_id])

        for _, box in boxes.iterrows():
            box_id = int(box["box_id"])
            out_dir = stage1_root / pair_id
            tasks.append({
                "phase": "stage1",
                "pair_id": pair_id,
                "box_id": box_id,
                "receptor": str(receptor),
                "ligand": str(ligand),
                "center_x": float(box["center_x"]),
                "center_y": float(box["center_y"]),
                "center_z": float(box["center_z"]),
                "size_x": box_size,
                "size_y": box_size,
                "size_z": box_size,
                "exhaustiveness": ex,
                "num_modes": modes,
                "energy_range": energy_range,
                "min_rmsd_filter": min_rmsd,
                "seed": seed,
                "cpu_per_job": cpu_per_job,
                "output_file": str(out_dir / f"box_{box_id:04d}.sdf"),
                "log_file": str(out_dir / f"box_{box_id:04d}.log"),
            })

    return tasks


# -----------------------------------------------------------------------------
# Stage1 analysis + Stage2 + pockets
# -----------------------------------------------------------------------------

def analyze_stage1(
    project_root: Path,
    evaluation_dir: Path,
    pair_rows: pd.DataFrame,
    fps_paths: Dict[str, Path],
    stage1_root: Path,
    analysis_root: Path,
    cluster_radius: float,
    stage2_box_size: float,
) -> Dict[str, Path]:
    script = evaluation_dir / "analyze_smina_stage1.py"
    require_file(script)

    print_header("5/8 Analyze Stage1 and select top-1 region")

    out = {}

    for i, row in pair_rows.iterrows():
        pair_id = str(row["pair_id"])
        if pair_id not in fps_paths:
            continue

        boxes = read_csv(fps_paths[pair_id])
        stage1_dir = stage1_root / pair_id

        complete = all(
            file_nonempty(stage1_dir / f"box_{int(b):04d}.sdf")
            for b in boxes["box_id"].tolist()
        )
        if not complete:
            print(f"[SKIP] {pair_id}: incomplete Stage1")
            continue

        analysis_dir = analysis_root / pair_id
        candidate_csv = analysis_dir / "candidate_regions.csv"

        if not file_nonempty(candidate_csv):
            proc = run_cmd(
                [
                    sys.executable,
                    str(script),
                    "--stage1-dir", str(stage1_dir),
                    "--boxes", str(fps_paths[pair_id]),
                    "--out-dir", str(analysis_dir),
                    "--cluster-radius", str(cluster_radius),
                    "--top-regions", "1",
                    "--stage2-box-size", str(stage2_box_size),
                ],
                cwd=project_root,
                timeout=1800,
            )
            if proc.returncode != 0 or not file_nonempty(candidate_csv):
                print(f"[FAIL] {pair_id}: Stage1 analysis")
                continue

        out[pair_id] = candidate_csv

        if (i + 1) % 25 == 0 or i == len(pair_rows) - 1:
            print(
                f"[{i+1}/{len(pair_rows)}] "
                f"candidate-ready={len(out)}"
            )

    return out


def build_stage2_tasks(
    candidate_paths: Dict[str, Path],
    smina_inputs_root: Path,
    stage2_root: Path,
    ex: int,
    modes: int,
    energy_range: float,
    min_rmsd: float,
    seed: int,
    cpu_per_job: int,
) -> List[Dict[str, Any]]:
    tasks = []

    for pair_id, path in candidate_paths.items():
        df = read_csv(path)
        top = df[pd.to_numeric(df["region_rank"], errors="coerce") == 1]
        if len(top) != 1:
            continue
        r = top.iloc[0]

        out_dir = stage2_root / pair_id / "region_01"
        tasks.append({
            "phase": "stage2",
            "pair_id": pair_id,
            "receptor": str(
                smina_inputs_root / pair_id / "receptor.pdbqt"
            ),
            "ligand": str(
                smina_inputs_root / pair_id / "ligand.pdbqt"
            ),
            "center_x": float(r["center_x"]),
            "center_y": float(r["center_y"]),
            "center_z": float(r["center_z"]),
            "size_x": float(r["size_x"]),
            "size_y": float(r["size_y"]),
            "size_z": float(r["size_z"]),
            "exhaustiveness": ex,
            "num_modes": modes,
            "energy_range": energy_range,
            "min_rmsd_filter": min_rmsd,
            "seed": seed,
            "cpu_per_job": cpu_per_job,
            "output_file": str(out_dir / f"seed_{seed}.sdf"),
            "log_file": str(out_dir / f"seed_{seed}.log"),
        })

    return tasks


def extract_pockets(
    project_root: Path,
    evaluation_dir: Path,
    pair_rows: pd.DataFrame,
    stage2_root: Path,
    deep_inputs_root: Path,
    seed: int,
    cutoff: float,
) -> pd.DataFrame:
    script = evaluation_dir / "extract_deepprotacs_pocket.py"
    require_file(script)

    print_header("7/8 Extract 5 Å pockets")

    rows = []

    for i, pair in pair_rows.iterrows():
        pair_id = str(pair["pair_id"])
        side = str(pair["side"]).strip().lower()
        protein = resolve_path(str(pair["protein_path"]), project_root)
        docked = (
            stage2_root / pair_id / "region_01" / f"seed_{seed}.sdf"
        )
        out_dir = deep_inputs_root / pair_id

        if side == "poi":
            lig_out = out_dir / "target_ligand.mol2"
            pocket_out = out_dir / "target_pocket.mol2"
        else:
            lig_out = out_dir / "ligase_ligand.mol2"
            pocket_out = out_dir / "ligase_pocket.mol2"

        if file_nonempty(lig_out) and file_nonempty(pocket_out):
            status = "ok_reused"
        elif not file_nonempty(docked):
            status = "missing_stage2_pose"
        else:
            proc = run_cmd(
                [
                    sys.executable,
                    str(script),
                    "--protein-pdb", str(protein),
                    "--docked-sdf", str(docked),
                    "--mode", "1",
                    "--cutoff", str(cutoff),
                    "--side", side,
                    "--pair-id", pair_id,
                    "--out-dir", str(out_dir),
                ],
                cwd=project_root,
                timeout=1800,
            )
            status = (
                "ok"
                if (
                    proc.returncode == 0
                    and file_nonempty(lig_out)
                    and file_nonempty(pocket_out)
                )
                else "pocket_failed"
            )

        rows.append({
            "pair_id": pair_id,
            "side": pair["side"],
            "status": status,
            "ligand_mol2": str(lig_out),
            "pocket_mol2": str(pocket_out),
        })

        if (i + 1) % 25 == 0 or i == len(pair_rows) - 1:
            n_ok = sum(
                str(r["status"]).startswith("ok")
                for r in rows
            )
            print(
                f"[{i+1}/{len(pair_rows)}] pocket-ready={n_ok}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(
        deep_inputs_root.parent / "pair_pocket_status.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return df


# -----------------------------------------------------------------------------
# Final sample manifests + DeepPROTACs
# -----------------------------------------------------------------------------

def build_manifest(candidate, stage, records_path, manifest_dir):
    manifest_dir.mkdir(parents=True, exist_ok=True)

    linker_path = manifest_dir / "linker.smi"
    linker_path.write_text(
        candidate["linker_smiles"].strip() + "\n",
        encoding="utf-8",
    )

    poi_paths = candidate["poi_input_paths"]
    e3_paths = candidate["e3_input_paths"]

    manifest = {
        "purpose": "Final reward-independent DeepPROTACs external evaluation",
        "stage": stage,
        "records_path": str(records_path.resolve()),
        "record_index": int(candidate["record_index"]),
        "uuid": candidate["uuid"],
        "sample_id": candidate["sample_id"],
        "selection_rule": "all_strict_samples",
        "strict": True,
        "strict_source": candidate["strict_source"],
        "linker_smiles": candidate["linker_smiles"],
        "linker_source": candidate["linker_source"],
        "linker_smi_path": str(linker_path.resolve()),
        "poi": {
            "pair_id": candidate["poi_pair_id"],
            "match_method": candidate["poi_match_method"],
            "deep_inputs_ready": bool(candidate["poi_ready"]),
            "target_ligand_mol2": str(
                poi_paths["ligand"].resolve()
            ),
            "target_pocket_mol2": str(
                poi_paths["pocket"].resolve()
            ),
        },
        "e3": {
            "pair_id": candidate["e3_pair_id"],
            "match_method": candidate["e3_match_method"],
            "deep_inputs_ready": bool(candidate["e3_ready"]),
            "ligase_ligand_mol2": str(
                e3_paths["ligand"].resolve()
            ),
            "ligase_pocket_mol2": str(
                e3_paths["pocket"].resolve()
            ),
        },
        "all_four_binary_mol2_ready": bool(
            candidate["poi_ready"] and candidate["e3_ready"]
        ),
    }

    path = manifest_dir / "sample_manifest.json"
    write_json(path, manifest)
    return path


def load_prediction(result_json: Path):
    if not file_nonempty(result_json):
        return None

    try:
        obj = json.loads(
            result_json.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None

    if obj.get("status") != "success":
        return None

    if "prediction" not in obj:
        return None

    # Formal evaluation only reuses results produced
    # under the deterministic inference protocol.
    if obj.get(
        "deterministic_inference"
    ) is not True:
        return None

    if int(
        obj.get(
            "deterministic_seed",
            -1,
        )
    ) != 2026:
        return None

    return obj


def evaluate_stage(
    *,
    stage: str,
    records_path: Path,
    helper,
    project_root: Path,
    poi_rows,
    e3_rows,
    deep_inputs_root: Path,
    manifests_root: Path,
    inference_root: Path,
    runner: Path,
    deepprotacs_dir: Optional[Path],
    deepprotacs_python: Optional[Path],
    timeout: int,
    inference_workers: int,
    max_inference_samples: Optional[int],
    skip_inference: bool,
    out_csv: Path,
):
    records = helper.load_records(records_path)
    rows = []
    attempts = 0
    inference_jobs = []

    excluded_missing_af2_records = 0
    excluded_missing_af2_uuids = set()

    print_header(f"8/8 DeepPROTACs inference: {stage}")

    for idx, rec in enumerate(records):
        structure_eligible, structure_reason = (
            helper.record_structure_eligible(rec)
        )

        if not structure_eligible:
            excluded_missing_af2_records += 1

            uuid = helper.safe_str(
                helper.get_field(rec, "uuid")
            ).strip()

            if uuid:
                excluded_missing_af2_uuids.add(uuid)

            continue
        is_strict, strict_source = helper.strict_flag(rec, "auto")

        row = {
            "stage": stage,
            "record_index": idx,
            "uuid": helper.safe_str(helper.get_field(rec, "uuid")),
            "sample_id": helper.safe_str(
                helper.get_field(rec, "sample_id")
            ),
            "strict": int(is_strict),
            "strict_source": strict_source,
            "status": "",
            "linker_smiles": "",
            "linker_source": "",
            "poi_pair_id": "",
            "e3_pair_id": "",
            "prediction": "",
            "predicted_active": "",
        }

        if not is_strict:
            row["status"] = "not_strict"
            rows.append(row)
            continue

        candidate, reason = helper.candidate_from_record(
            rec=rec,
            record_index=idx,
            strict_key="auto",
            poi_rows=poi_rows,
            e3_rows=e3_rows,
            project_root=project_root,
            deep_inputs_root=deep_inputs_root,
        )

        if candidate is None:
            row["status"] = reason
            rows.append(row)
            continue

        row.update({
            "linker_smiles": candidate["linker_smiles"],
            "linker_source": candidate["linker_source"],
            "poi_pair_id": candidate["poi_pair_id"],
            "e3_pair_id": candidate["e3_pair_id"],
        })

        if not candidate["poi_ready"] or not candidate["e3_ready"]:
            missing = []
            if not candidate["poi_ready"]:
                missing.append("poi")
            if not candidate["e3_ready"]:
                missing.append("e3")
            row["status"] = "missing_pair_input:" + "+".join(missing)
            rows.append(row)
            continue

        manifest_dir = manifests_root / stage / f"record_{idx:04d}"
        manifest = build_manifest(
            candidate,
            stage,
            records_path,
            manifest_dir,
        )

        run_dir = inference_root / f"{stage}_record_{idx:04d}"
        result_json = run_dir / "deepprotacs_pilot_result.json"

        existing = load_prediction(result_json)
        if existing is not None:
            pred = int(existing["prediction"])
            row["status"] = "scored_reused"
            row["prediction"] = pred
            row["predicted_active"] = int(pred == 1)
            rows.append(row)
            continue

        if skip_inference:
            row["status"] = "ready_not_scored"
            rows.append(row)
            continue

        if (
            max_inference_samples is not None
            and attempts >= max_inference_samples
        ):
            row["status"] = "ready_not_scored_limit"
            rows.append(row)
            continue

        cmd = [
            sys.executable,
            str(runner),
            "--manifest", str(manifest),
            "--out-root", str(inference_root),
            "--timeout", str(timeout),
            "--deterministic-seed", "2026",
            "--script-mode", "portable",
        ]

        if deepprotacs_dir is not None:
            cmd += [
                "--deepprotacs-dir",
                str(deepprotacs_dir),
            ]

        if deepprotacs_python is not None:
            cmd += [
                "--deepprotacs-python",
                str(deepprotacs_python),
            ]

        # Keep the row at its deterministic original position.
        row_pos = len(rows)

        row["status"] = "queued"
        rows.append(row)

        inference_jobs.append(
            {
                "row_pos": row_pos,
                "record_index": idx,
                "cmd": cmd,
                "result_json": result_json,
            }
        )

        attempts += 1

        # cmd = [
        #     sys.executable,
        #     str(runner),
        #     "--manifest", str(manifest),
        #     "--out-root", str(inference_root),
        #     "--timeout", str(timeout),
        # ]
        # if deepprotacs_dir is not None:
        #     cmd += ["--deepprotacs-dir", str(deepprotacs_dir)]
        # if deepprotacs_python is not None:
        #     cmd += ["--deepprotacs-python", str(deepprotacs_python)]
        #
        # proc = run_cmd(
        #     cmd,
        #     cwd=project_root,
        #     timeout=timeout + 120,
        # )
        # attempts += 1
        #
        # result = load_prediction(result_json)
        # if proc.returncode == 0 and result is not None:
        #     pred = int(result["prediction"])
        #     row["status"] = "scored"
        #     row["prediction"] = pred
        #     row["predicted_active"] = int(pred == 1)
        # else:
        #     row["status"] = (
        #         f"deepprotacs_failed:returncode={proc.returncode}"
        #     )
        #
        # rows.append(row)
        #
        # if attempts % 10 == 0:
        #     pd.DataFrame(rows).to_csv(
        #         out_csv,
        #         index=False,
        #         encoding="utf-8-sig",
        #     )
        #     print(
        #         f"{stage}: new inference attempts={attempts}, "
        #         f"record={idx}/{len(records)-1}"
        #     )

    # ============================================================
    # Parallel DeepPROTACs inference
    # ============================================================

    if inference_jobs:

        worker_count = min(
            max(
                1,
                int(inference_workers),
            ),
            len(inference_jobs),
        )

        print(
            f"{stage}: queued="
            f"{len(inference_jobs)}, "
            f"inference_workers="
            f"{worker_count}"
        )

        completed = 0

        with ThreadPoolExecutor(
            max_workers=worker_count
        ) as pool:

            future_map = {
                pool.submit(
                    run_cmd,
                    job["cmd"],
                    cwd=project_root,
                    timeout=timeout + 120,
                ): job
                for job in inference_jobs
            }

            for future in as_completed(
                future_map
            ):

                job = future_map[
                    future
                ]

                row = rows[
                    job["row_pos"]
                ]

                try:
                    proc = future.result()

                    result = load_prediction(
                        job["result_json"]
                    )

                    if (
                        proc.returncode == 0
                        and result is not None
                    ):
                        pred = int(
                            result["prediction"]
                        )

                        row["status"] = "scored"
                        row["prediction"] = pred
                        row[
                            "predicted_active"
                        ] = int(
                            pred == 1
                        )


                    else:

                        row["status"] = (

                            "deepprotacs_failed:"

                            f"returncode="

                            f"{proc.returncode}"

                        )

                        print(

                            f"\n[{stage} record "

                            f"{job['record_index']}] "

                            f"DeepPROTACs failed, "

                            f"returncode={proc.returncode}"

                        )

                        if proc.stdout:
                            print(proc.stdout.rstrip())

                except subprocess.TimeoutExpired:
                    row["status"] = (
                        "deepprotacs_failed:"
                        "outer_timeout"
                    )

                except Exception as exc:
                    row["status"] = (
                        "deepprotacs_failed:"
                        f"{type(exc).__name__}:"
                        f"{exc}"
                    )

                rows[
                    job["row_pos"]
                ] = row

                completed += 1

                if (
                    completed % 10 == 0
                    or completed
                    == len(inference_jobs)
                ):
                    pd.DataFrame(
                        rows
                    ).to_csv(
                        out_csv,
                        index=False,
                        encoding="utf-8-sig",
                    )

                    print(
                        f"{stage}: completed="
                        f"{completed}/"
                        f"{len(inference_jobs)}"
                    )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    strict_n = int(df["strict"].sum())
    scored = df["status"].astype(str).isin(
        ["scored", "scored_reused"]
    )
    active = scored & (
        pd.to_numeric(
            df["predicted_active"],
            errors="coerce",
        ).fillna(0).astype(int) == 1
    )

    status_s = df["status"].astype(str)
    invalid_linker = status_s.str.startswith("bad_linker:")
    valid_linker_strict = int(
        ((df["strict"] == 1) & ~invalid_linker).sum()
    )

    summary = {
        "stage": stage,

        "original_generated": int(len(records)),

        "excluded_missing_af2_records": int(
            excluded_missing_af2_records
        ),

        "excluded_missing_af2_contexts": int(
            len(excluded_missing_af2_uuids)
        ),

        "generated": int(len(df)),
        "eligible_contexts": int(
            df["uuid"].astype(str).nunique()
        ),

        "strict": strict_n,
        "valid_linker_strict": valid_linker_strict,
        "successfully_scored_strict": int(scored.sum()),
        "active": int(active.sum()),

        "scored_rate_among_strict": (
            float(scored.sum()) / strict_n
            if strict_n
            else float("nan")
        ),

        "active_rate_among_scored_strict": (
            float(active.sum()) / float(scored.sum())
            if int(scored.sum())
            else float("nan")
        ),

        "strict_and_active_rate_all_generated": (
            float(active.sum()) / len(df)
            if len(df)
            else float("nan")
        ),

        "status_counts": dict(Counter(status_s)),
    }

    return df, summary


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument("--project-root", default=".")
    ap.add_argument("--evaluation-dir", default="evaluation")
    ap.add_argument("--pretrain", default=str(DEFAULT_RECORDS["pretrain"]))
    ap.add_argument("--rl1", default=str(DEFAULT_RECORDS["rl1"]))
    ap.add_argument("--rl2", default=str(DEFAULT_RECORDS["rl2"]))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))

    ap.add_argument("--box-size", type=float, default=30.0)
    ap.add_argument("--grid-step", type=float, default=15.0)
    ap.add_argument("--grid-margin", type=float, default=5.0)
    ap.add_argument("--max-center-distance", type=float, default=10.0)
    ap.add_argument("--fps-boxes", type=int, default=24)

    ap.add_argument("--stage1-ex", type=int, default=4)
    ap.add_argument("--stage1-modes", type=int, default=10)
    ap.add_argument("--cluster-radius", type=float, default=5.0)

    ap.add_argument("--stage2-ex", type=int, default=32)
    ap.add_argument("--stage2-modes", type=int, default=20)

    ap.add_argument("--energy-range", type=float, default=5.0)
    ap.add_argument("--min-rmsd-filter", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pocket-cutoff", type=float, default=5.0)

    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cpu-per-job", type=int, default=1)
    ap.add_argument("--smina-timeout", type=int, default=7200)

    ap.add_argument("--deepprotacs-dir", default=None)
    ap.add_argument("--deepprotacs-python", default=None)
    ap.add_argument("--deepprotacs-timeout", type=int, default=600)

    ap.add_argument(
        "--inference-workers",
        type=int,
        default=18,
        help=(
            "Concurrent DeepPROTACs inference "
            "subprocesses"
        ),
    )

    ap.add_argument(
        "--max-inference-samples",
        type=int,
        default=None,
    )

    # Default behavior:
    #   - reuse the already completed docking/pocket results
    #   - run DeepPROTACs inference directly
    #
    # --run-docking can still be used later if docking needs to be rerun.
    docking_group = ap.add_mutually_exclusive_group()

    docking_group.add_argument(
        "--skip-docking",
        dest="skip_docking",
        action="store_true",
    )

    docking_group.add_argument(
        "--run-docking",
        dest="skip_docking",
        action="store_false",
    )

    ap.set_defaults(
        skip_docking=True,
    )

    ap.add_argument(
        "--skip-inference",
        action="store_true",
    )

    args = ap.parse_args()

    if args.inference_workers < 1:
        ap.error(
            "--inference-workers must be >= 1"
        )

    project_root = Path(args.project_root).resolve()
    evaluation_dir = resolve_path(args.evaluation_dir, project_root)
    out_root = resolve_path(args.out_root, project_root)
    out_root.mkdir(parents=True, exist_ok=True)

    records_paths = {
        "pretrain": resolve_path(args.pretrain, project_root),
        "rl1": resolve_path(args.rl1, project_root),
        "rl2": resolve_path(args.rl2, project_root),
    }
    for p in records_paths.values():
        require_file(p)

    helper_files = [
        "prepare_docking_pairs.py",
        "prepare_smina_inputs.py",
        "make_smina_grid.py",
        "analyze_smina_stage1.py",
        "extract_deepprotacs_pocket.py",
        "prepare_deepprotacs_pilot_sample.py",
    ]
    for name in helper_files:
        require_file(evaluation_dir / name)

    # Use ONLY the successful v4-compatible DeepPROTACs wrapper.
    # The older run_deepprotacs_pilot.py may lack the PyG 2.7 compatibility
    # repairs (_cls / aggr_module reconstruction), so do not silently use it
    # unless it contains the same v4 markers.
    deepprotacs_runner_candidates = [
        evaluation_dir / "run_deepprotacs_pilot_v4.py",
        evaluation_dir / "run_deepprotacs_pilot.py",
    ]
    deepprotacs_runner = None
    for candidate in deepprotacs_runner_candidates:
        if not candidate.is_file():
            continue
        try:
            runner_text = candidate.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        v4_markers = [
            "aggregation_resolver",
            "aggr_module",
            "torch_geometric.nn.conv.utils.inspector",
        ]
        if all(marker in runner_text for marker in v4_markers):
            deepprotacs_runner = candidate
            break

    if deepprotacs_runner is None:
        raise FileNotFoundError(
            "The successful DeepPROTACs v4 compatibility wrapper was not found. "
            "Place run_deepprotacs_pilot_v4.py in evaluation/. The older "
            "run_deepprotacs_pilot.py is accepted only if it contains the same "
            "PyG compatibility fixes."
        )

    print(f"DeepPROTACs compatibility runner: {deepprotacs_runner}")

    smina = shutil.which("smina")
    if not args.skip_docking and not smina:
        raise RuntimeError("Cannot find smina")

    smina_inputs_root = out_root / "smina_inputs"
    grids_root = out_root / "grids"
    fps_root = out_root / "fps_boxes"
    stage1_root = out_root / "stage1"
    stage1_analysis_root = out_root / "stage1_analysis"
    stage2_root = out_root / "stage2"
    deep_inputs_root = out_root / "deepprotacs_inputs"
    manifests_root = out_root / "sample_manifests"
    inference_root = out_root / "deepprotacs_inference"
    stage_results_root = out_root / "stage_results"
    stage_results_root.mkdir(parents=True, exist_ok=True)

    pairs_all_csv, pairs_poi_csv, pairs_e3_csv = prepare_pair_tables(
        project_root,
        evaluation_dir,
        records_paths,
        out_root,
    )
    pair_rows = read_csv(pairs_all_csv)

    if not args.skip_docking:
        prep_status = prepare_smina_inputs_all(
            project_root,
            evaluation_dir,
            pairs_all_csv,
            smina_inputs_root,
        )
        good_ids = set(
            prep_status.loc[
                prep_status["status"].astype(str).str.lower() == "ok",
                "pair_id",
            ].astype(str)
        )

        fps_paths = make_grids_and_fps(
            project_root,
            evaluation_dir,
            pair_rows,
            good_ids,
            grids_root,
            fps_root,
            args.box_size,
            args.grid_step,
            args.grid_margin,
            args.max_center_distance,
            args.fps_boxes,
        )

        stage1_tasks = build_stage1_tasks(
            pair_rows,
            fps_paths,
            smina_inputs_root,
            stage1_root,
            args.box_size,
            args.stage1_ex,
            args.stage1_modes,
            args.energy_range,
            args.min_rmsd_filter,
            args.seed,
            args.cpu_per_job,
        )
        run_pool(
            stage1_tasks,
            workers=args.workers,
            smina=smina,
            project_root=project_root,
            timeout=args.smina_timeout,
            status_csv=out_root / "stage1_jobs.csv",
            title=(
                f"4/8 Stage1: FPS={args.fps_boxes}, "
                f"ex={args.stage1_ex}, modes={args.stage1_modes}"
            ),
        )

        candidates = analyze_stage1(
            project_root,
            evaluation_dir,
            pair_rows,
            fps_paths,
            stage1_root,
            stage1_analysis_root,
            args.cluster_radius,
            args.box_size,
        )

        stage2_tasks = build_stage2_tasks(
            candidates,
            smina_inputs_root,
            stage2_root,
            args.stage2_ex,
            args.stage2_modes,
            args.energy_range,
            args.min_rmsd_filter,
            args.seed,
            args.cpu_per_job,
        )
        run_pool(
            stage2_tasks,
            workers=args.workers,
            smina=smina,
            project_root=project_root,
            timeout=args.smina_timeout,
            status_csv=out_root / "stage2_jobs.csv",
            title=(
                f"6/8 Stage2: ex={args.stage2_ex}, "
                f"modes={args.stage2_modes}"
            ),
        )

        extract_pockets(
            project_root,
            evaluation_dir,
            pair_rows,
            stage2_root,
            deep_inputs_root,
            args.seed,
            args.pocket_cutoff,
        )
    else:
        print_header("Docking skipped")
        if not deep_inputs_root.is_dir():
            raise RuntimeError(
                f"Missing prepared pair inputs: {deep_inputs_root}"
            )

    helper = import_module_from_file(
        "deepprotacs_sample_helper",
        evaluation_dir / "prepare_deepprotacs_pilot_sample.py",
    )
    poi_rows = helper.prepare_pair_rows(
        helper.read_csv_rows(pairs_poi_csv),
        project_root,
    )
    e3_rows = helper.prepare_pair_rows(
        helper.read_csv_rows(pairs_e3_csv),
        project_root,
    )

    # Resolve official DeepPROTACs repository.  The user's project layout is:
    #   LinkerDiff/external/DeepPROTACs
    # so that location is checked first when --deepprotacs-dir is omitted.
    if args.deepprotacs_dir:
        dp_dir = resolve_path(args.deepprotacs_dir, project_root)
    else:
        dp_candidates = [
            project_root / "external" / "DeepPROTACs",
            project_root / "DeepPROTACs",
            project_root.parent / "DeepPROTACs",
        ]
        dp_dir = next(
            (
                c.resolve()
                for c in dp_candidates
                if (c / "single_prediction.py").is_file()
                and (c / "model.py").is_file()
                and (c / "model" / "test.pt").is_file()
            ),
            None,
        )

    if dp_dir is None:
        raise FileNotFoundError(
            "Official DeepPROTACs repository not found. Expected e.g. "
            "external/DeepPROTACs containing single_prediction.py, model.py, "
            "and model/test.pt; or pass --deepprotacs-dir explicitly."
        )

    print(f"Official DeepPROTACs repo: {dp_dir}")
    dp_python = (
        resolve_path(args.deepprotacs_python, project_root)
        if args.deepprotacs_python else None
    )

    summaries = []

    for stage in STAGES:
        _, summary = evaluate_stage(
            stage=stage,
            records_path=records_paths[stage],
            helper=helper,
            project_root=project_root,
            poi_rows=poi_rows,
            e3_rows=e3_rows,
            deep_inputs_root=deep_inputs_root,
            manifests_root=manifests_root,
            inference_root=inference_root,
            runner=deepprotacs_runner,
            deepprotacs_dir=dp_dir,
            deepprotacs_python=dp_python,
            timeout=args.deepprotacs_timeout,
            inference_workers=args.inference_workers,
            max_inference_samples=args.max_inference_samples,
            skip_inference=args.skip_inference,
            out_csv=(
                stage_results_root / f"{stage}_deepprotacs_records.csv"
            ),
        )
        summaries.append(summary)
        write_json(
            stage_results_root / f"{stage}_summary.json",
            summary,
        )

    flat = [
        {k: v for k, v in s.items() if k != "status_counts"}
        for s in summaries
    ]
    summary_df = pd.DataFrame(flat)
    summary_csv = out_root / "deepprotacs_three_stage_summary.csv"
    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )
    write_json(
        out_root / "deepprotacs_three_stage_summary.json",
        summaries,
    )

    print_header("FINAL SUMMARY")
    print(summary_df.to_string(index=False))
    print("")
    print(f"Summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
