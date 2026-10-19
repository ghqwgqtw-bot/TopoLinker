#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
为 Smina 准备 receptor / ligand PDBQT。

默认只处理 docking_pairs_all.csv 中第 1 个 pair，先做小规模试跑。
不会修改原始 PDB、SDF、records.pkl 或 docking_pairs_*.csv。
"""

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


def run_cmd(cmd, log_lines):
    log_lines.append("$ " + " ".join(f'\"{x}\"' if " " in str(x) else str(x) for x in cmd))
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if p.stdout:
        log_lines.append(p.stdout.rstrip())
    log_lines.append(f"[returncode] {p.returncode}")
    return p.returncode


def count_pdbqt_atoms(path):
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                n += 1
    return n


def has_any_tag(path, tags):
    text = path.read_text(encoding="utf-8", errors="ignore")
    return any(tag in text for tag in tags)


def prepare_one(row, out_root, obabel, overwrite=False):
    pair_id = row["pair_id"].strip()
    protein_path = Path(row["protein_path"])
    ligand_path = Path(row["ligand_path"])

    pair_dir = out_root / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    receptor_out = pair_dir / "receptor.pdbqt"
    ligand_out = pair_dir / "ligand.pdbqt"
    log_path = pair_dir / "prepare.log"

    log_lines = [
        f"pair_id: {pair_id}",
        f"side: {row.get('side', '')}",
        f"protein_uniprot: {row.get('protein_uniprot', '')}",
        f"protein_path: {protein_path}",
        f"ligand_path: {ligand_path}",
        "",
    ]

    if not protein_path.exists():
        log_lines.append("[ERROR] protein file not found")
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        return False, "protein_missing"

    if not ligand_path.exists():
        log_lines.append("[ERROR] ligand file not found")
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        return False, "ligand_missing"

    if overwrite:
        for p in (receptor_out, ligand_out):
            if p.exists():
                p.unlink()

    # receptor: 添加极性氢、Gasteiger 电荷，并强制写成刚性 PDBQT。
    if not receptor_out.exists() or receptor_out.stat().st_size == 0:
        receptor_cmd = [
            obabel,
            str(protein_path),
            "-O", str(receptor_out),
            "--addpolarh",
            "--partialcharge", "gasteiger",
            "-xr",   # PDBQT output option: rigid molecule
            "-xh",   # preserve hydrogens in PDBQT
        ]
        if run_cmd(receptor_cmd, log_lines) != 0:
            log_lines.append("[ERROR] receptor conversion failed")
            log_path.write_text("\n".join(log_lines), encoding="utf-8")
            return False, "receptor_conversion_failed"
    else:
        log_lines.append("[SKIP] receptor.pdbqt already exists")

    # ligand: 保留原有重原子坐标，只加氢和 Gasteiger 电荷；保持默认柔性 torsion tree。
    # 不使用 -p 7.4，避免主动改变原始 fragment 的质子化/形式电荷状态。
    if not ligand_out.exists() or ligand_out.stat().st_size == 0:
        ligand_cmd = [
            obabel,
            str(ligand_path),
            "-O", str(ligand_out),
            "-h",
            "--partialcharge", "gasteiger",
            "-xh",
        ]
        if run_cmd(ligand_cmd, log_lines) != 0:
            log_lines.append("[ERROR] ligand conversion failed")
            log_path.write_text("\n".join(log_lines), encoding="utf-8")
            return False, "ligand_conversion_failed"
    else:
        log_lines.append("[SKIP] ligand.pdbqt already exists")

    receptor_atoms = count_pdbqt_atoms(receptor_out)
    ligand_atoms = count_pdbqt_atoms(ligand_out)
    receptor_has_torsion = has_any_tag(receptor_out, ["BRANCH", "ROOT", "TORSDOF"])
    ligand_has_torsion = has_any_tag(ligand_out, ["ROOT", "BRANCH", "TORSDOF"])

    log_lines += [
        "",
        "========== validation ==========",
        f"receptor_pdbqt_atoms: {receptor_atoms}",
        f"ligand_pdbqt_atoms: {ligand_atoms}",
        f"receptor_has_torsion_tags: {receptor_has_torsion}",
        f"ligand_has_torsion_tags: {ligand_has_torsion}",
    ]

    if receptor_atoms <= 0:
        log_lines.append("[ERROR] receptor.pdbqt contains no atoms")
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        return False, "empty_receptor"

    if ligand_atoms <= 0:
        log_lines.append("[ERROR] ligand.pdbqt contains no atoms")
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        return False, "empty_ligand"

    if receptor_has_torsion:
        log_lines.append("[ERROR] receptor should be rigid but torsion-tree tags were found")
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        return False, "receptor_not_rigid"

    log_lines.append("[OK] Smina inputs prepared")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        default="eval_outputs/deepprotacs_docking/docking_pairs_all.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="eval_outputs/deepprotacs_docking/smina_inputs",
    )
    parser.add_argument("--pair-id", default=None, help="例如 poi_0001")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not pairs_path.exists():
        raise FileNotFoundError(f"pairs CSV 不存在: {pairs_path}")

    obabel = shutil.which("obabel")
    if not obabel:
        raise RuntimeError("当前环境找不到 obabel，请先运行 obabel -V。")

    print(f"obabel: {obabel}")
    print(f"pairs:  {pairs_path.resolve()}")
    print(f"out:    {out_root.resolve()}")

    with pairs_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if args.pair_id:
        rows = [r for r in rows if r.get("pair_id", "").strip() == args.pair_id]
        if not rows:
            raise ValueError(f"找不到 pair_id={args.pair_id}")
    elif not args.all:
        rows = rows[:max(0, args.limit)]

    print(f"本次处理 pair 数: {len(rows)}")

    summary_rows = []
    ok_count = 0

    for i, row in enumerate(rows, start=1):
        pair_id = row.get("pair_id", "")
        print(f"[{i}/{len(rows)}] {pair_id} ...", end=" ", flush=True)
        ok, status = prepare_one(row, out_root, obabel, overwrite=args.overwrite)
        print("OK" if ok else f"FAIL ({status})")
        summary_rows.append({
            "pair_id": pair_id,
            "side": row.get("side", ""),
            "protein_uniprot": row.get("protein_uniprot", ""),
            "status": status,
        })
        if ok:
            ok_count += 1

    summary_path = out_root / "prepare_smina_inputs_status.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair_id", "side", "protein_uniprot", "status"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n========== summary ==========")
    print(f"processed: {len(rows)}")
    print(f"success:   {ok_count}")
    print(f"failed:    {len(rows) - ok_count}")
    print(f"status:    {summary_path.resolve()}")


if __name__ == "__main__":
    main()
