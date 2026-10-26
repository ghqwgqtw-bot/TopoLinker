#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract a DeepPROTACs-style binary binding pocket from a Smina Stage 2 pose.

Workflow
--------
Original protein PDB
    +
Stage 2 docked ligand SDF
        ↓
take one docking mode (default: mode 1)
        ↓
find protein residues having at least one heavy atom within cutoff
(default: 5.0 Å) of any ligand heavy atom
        ↓
keep the WHOLE selected residues
        ↓
write:
    - selected docked ligand SDF
    - selected docked ligand MOL2
    - pocket PDB
    - pocket MOL2
    - residue list CSV
    - extraction summary TXT

This follows the DeepPROTACs pocket definition used in its official example:
protein residues within 5 Å of the bound ligand are retained as the pocket.

Important
---------
1) Coordinates are NOT aligned or transformed here.
   The Stage 2 Smina pose must already be in the same coordinate frame as
   the original protein PDB. In the current workflow this should be true,
   because receptor.pdbqt was generated directly from the original protein
   coordinates and Smina preserves the receptor coordinate frame.

2) Residue selection is "by residue":
   if ANY heavy atom of a residue is within cutoff of ANY ligand heavy atom,
   all ATOM records of that residue are written to the pocket PDB.

3) HETATM records are excluded by default from pocket selection/writing.
   This is intentional for the initial DeepPROTACs protein-pocket workflow.
   Use --include-hetatm only if you explicitly decide to retain protein-side
   cofactors/metals/waters in a later protocol.

4) OpenBabel is used to convert:
      ligand SDF -> MOL2
      pocket PDB -> MOL2
   because DeepPROTACs expects MOL2 inputs.

Default pilot (current e3_0001)
-------------------------------
protein:
    datasets/rl/test/deepternary_inputs/19/p2_e3.pdb

stage2 pose:
    eval_outputs/deepprotacs_docking/stage2/e3_0001/
    region_01/seed_2026.sdf

side:
    E3

output:
    eval_outputs/deepprotacs_docking/deepprotacs_inputs/e3_0001/

For E3 side, final aliases are:
    ligase_ligand.mol2
    ligase_pocket.mol2

For POI side, final aliases are:
    target_ligand.mol2
    target_pocket.mol2
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def run_command(cmd: Sequence[str], *, label: str) -> str:
    """
    Run a subprocess and fail loudly with captured stdout/stderr.
    """
    proc = subprocess.run(
        list(map(str, cmd)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed with return code {proc.returncode}\n"
            f"Command:\n{' '.join(map(str, cmd))}\n\n"
            f"Output:\n{proc.stdout}"
        )

    return proc.stdout or ""


def check_obabel(obabel: str) -> None:
    proc = subprocess.run(
        [obabel, "-V"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"OpenBabel executable is not usable: {obabel}\n"
            f"{proc.stdout}"
        )


def parse_pdb_atom_line(line: str) -> Optional[dict]:
    """
    Parse standard PDB ATOM/HETATM columns.

    Returns None if coordinates cannot be parsed.
    """
    if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
        return None

    try:
        record = line[0:6].strip()
        atom_name = line[12:16].strip()
        altloc = line[16:17].strip()
        resname = line[17:20].strip()
        chain_id = line[21:22].strip()
        resseq = line[22:26].strip()
        icode = line[26:27].strip()

        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])

        element = line[76:78].strip()
        if not element:
            # Conservative fallback from atom name.
            stripped = atom_name.lstrip("0123456789")
            if stripped:
                # Protein atom names usually start with element symbol.
                # We only need hydrogen-vs-heavy discrimination here.
                element = stripped[0].upper()
            else:
                element = ""

        return {
            "record": record,
            "atom_name": atom_name,
            "altloc": altloc,
            "resname": resname,
            "chain_id": chain_id,
            "resseq": resseq,
            "icode": icode,
            "x": x,
            "y": y,
            "z": z,
            "element": element.upper(),
            "line": line.rstrip("\n"),
        }
    except Exception:
        return None


def residue_key(atom: dict) -> Tuple[str, str, str, str]:
    """
    chain, residue number, insertion code, residue name
    """
    return (
        atom["chain_id"],
        atom["resseq"],
        atom["icode"],
        atom["resname"],
    )


def is_hydrogen_atom(atom: dict) -> bool:
    """
    Identify H/D atoms for distance calculations.
    """
    element = atom["element"].upper()
    if element in {"H", "D"}:
        return True

    atom_name = atom["atom_name"].strip().upper()
    # PDB atom names can be 1H, 2HA, etc.
    atom_name = atom_name.lstrip("0123456789")
    return atom_name.startswith("H") or atom_name.startswith("D")


def load_protein_atoms(
    pdb_path: Path,
    *,
    include_hetatm: bool = False,
    keep_altloc: Sequence[str] = ("", "A"),
):
    """
    Read protein records. By default only ATOM records are used.

    Alternative locations:
      keep blank and A by default.
    """
    atoms = []

    with pdb_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        for line in f:
            if line.startswith("ATOM  "):
                pass
            elif include_hetatm and line.startswith("HETATM"):
                pass
            else:
                continue

            atom = parse_pdb_atom_line(line)
            if atom is None:
                continue

            if atom["altloc"] not in keep_altloc:
                continue

            atoms.append(atom)

    if not atoms:
        raise RuntimeError(
            f"No usable protein atoms were parsed from: {pdb_path}"
        )

    return atoms


def load_docking_mode(sdf_path: Path, mode: int):
    """
    Smina SDF generally stores docking modes as sequential molecules.
    mode=1 means the first molecule.
    """
    if mode < 1:
        raise ValueError("--mode must be >= 1")

    suppl = Chem.SDMolSupplier(
        str(sdf_path),
        removeHs=False,
        sanitize=False,
    )

    mols = [m for m in suppl if m is not None]

    if len(mols) < mode:
        raise ValueError(
            f"Requested mode {mode}, but only {len(mols)} readable "
            f"molecules are present in {sdf_path}"
        )

    mol = mols[mode - 1]

    if mol.GetNumConformers() == 0:
        raise RuntimeError(
            f"Docking mode {mode} has no 3D conformer: {sdf_path}"
        )

    return mol, len(mols)


def get_heavy_atom_coords_from_mol(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()

    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue

        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append(
            [float(p.x), float(p.y), float(p.z)]
        )

    if not coords:
        raise RuntimeError("Docked ligand has no heavy atoms.")

    return np.asarray(coords, dtype=float)


def select_residues_within_cutoff(
    protein_atoms: List[dict],
    ligand_heavy_xyz: np.ndarray,
    cutoff: float,
):
    """
    A residue is selected when at least one of its heavy atoms is
    within cutoff Å of any ligand heavy atom.

    Returns:
      selected_residue_keys
      residue_stats
      global_min_distance
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be > 0")

    # Group all protein atoms by residue.
    residues: Dict[Tuple[str, str, str, str], List[dict]] = {}

    for atom in protein_atoms:
        key = residue_key(atom)
        residues.setdefault(key, []).append(atom)

    selected = set()
    residue_stats = []

    global_min_distance = math.inf

    for key, atoms in residues.items():
        heavy = [
            atom for atom in atoms
            if not is_hydrogen_atom(atom)
        ]

        if not heavy:
            continue

        xyz = np.asarray(
            [
                [a["x"], a["y"], a["z"]]
                for a in heavy
            ],
            dtype=float,
        )

        # shape: protein_atoms x ligand_atoms
        diff = xyz[:, None, :] - ligand_heavy_xyz[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        min_dist = float(np.sqrt(np.min(d2)))

        global_min_distance = min(
            global_min_distance,
            min_dist,
        )

        if min_dist <= cutoff:
            selected.add(key)
            residue_stats.append(
                {
                    "chain_id": key[0],
                    "resseq": key[1],
                    "icode": key[2],
                    "resname": key[3],
                    "min_distance_A": min_dist,
                    "num_atoms_written": len(atoms),
                    "num_heavy_atoms": len(heavy),
                }
            )

    residue_stats.sort(
        key=lambda r: (
            r["min_distance_A"],
            r["chain_id"],
            int(r["resseq"])
            if str(r["resseq"]).lstrip("-").isdigit()
            else str(r["resseq"]),
            r["icode"],
        )
    )

    return selected, residue_stats, global_min_distance


def write_selected_residue_pdb(
    original_pdb: Path,
    out_pdb: Path,
    selected_residues,
    *,
    include_hetatm: bool = False,
    keep_altloc: Sequence[str] = ("", "A"),
):
    """
    Preserve original PDB ATOM lines for selected whole residues.

    TER is inserted between chains when possible.
    """
    selected_residues = set(selected_residues)

    written = 0
    last_chain = None

    with original_pdb.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as fin, out_pdb.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fout:

        for line in fin:
            if line.startswith("ATOM  "):
                pass
            elif include_hetatm and line.startswith("HETATM"):
                pass
            else:
                continue

            atom = parse_pdb_atom_line(line)
            if atom is None:
                continue

            if atom["altloc"] not in keep_altloc:
                continue

            key = residue_key(atom)

            if key not in selected_residues:
                continue

            chain = atom["chain_id"]

            if (
                last_chain is not None
                and chain != last_chain
            ):
                fout.write("TER\n")

            fout.write(atom["line"] + "\n")
            written += 1
            last_chain = chain

        if written > 0:
            fout.write("TER\n")
            fout.write("END\n")

    if written == 0:
        raise RuntimeError(
            "Pocket PDB contains zero atoms after residue selection."
        )

    return written


def write_single_mol_sdf(
    mol: Chem.Mol,
    out_sdf: Path,
):
    writer = Chem.SDWriter(str(out_sdf))
    writer.SetKekulize(False)
    writer.write(mol)
    writer.close()

    if (
        not out_sdf.exists()
        or out_sdf.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Failed to write ligand SDF: {out_sdf}"
        )


def convert_with_obabel(
    obabel: str,
    input_path: Path,
    output_path: Path,
    *,
    add_h: bool = False,
):
    """
    Generic OpenBabel conversion.

    We intentionally do not re-protonate at a target pH here.
    The docked ligand/protein coordinates should remain unchanged.
    """
    cmd = [
        obabel,
        str(input_path),
        "-O",
        str(output_path),
    ]

    if add_h:
        cmd.append("-h")

    stdout = run_command(
        cmd,
        label=f"OpenBabel conversion {input_path.name} -> {output_path.name}",
    )

    if (
        not output_path.exists()
        or output_path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"OpenBabel produced no output: {output_path}\n"
            f"{stdout}"
        )

    return stdout


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protein-pdb",
        default=(
            "datasets/rl/test/deepternary_inputs/"
            "19/p2_e3.pdb"
        ),
        help="Original protein PDB in the same coordinate frame as Smina",
    )

    parser.add_argument(
        "--docked-sdf",
        default=(
            "eval_outputs/deepprotacs_docking/"
            "stage2/e3_0001/region_01/seed_2026.sdf"
        ),
        help="Stage 2 Smina output SDF",
    )

    parser.add_argument(
        "--mode",
        type=int,
        default=1,
        help="Docking mode to extract from SDF. Default: 1",
    )

    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Residue-selection cutoff in Å. Default: 5.0",
    )

    parser.add_argument(
        "--side",
        choices=["e3", "poi"],
        default="e3",
        help=(
            "DeepPROTACs side. "
            "e3 -> ligase_ligand/pocket; "
            "poi -> target_ligand/pocket"
        ),
    )

    parser.add_argument(
        "--pair-id",
        default="e3_0001",
        help="Pair identifier used in output summary",
    )

    parser.add_argument(
        "--out-dir",
        default=(
            "eval_outputs/deepprotacs_docking/"
            "deepprotacs_inputs/e3_0001"
        ),
        help="Output directory",
    )

    parser.add_argument(
        "--obabel",
        default="obabel",
        help="OpenBabel executable",
    )

    parser.add_argument(
        "--include-hetatm",
        action="store_true",
        help=(
            "Include HETATM records in residue selection/writing. "
            "Default: false."
        ),
    )

    args = parser.parse_args()

    protein_pdb = Path(args.protein_pdb).resolve()
    docked_sdf = Path(args.docked_sdf).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not protein_pdb.exists():
        raise FileNotFoundError(
            f"Protein PDB does not exist: {protein_pdb}"
        )

    if not docked_sdf.exists():
        raise FileNotFoundError(
            f"Docked SDF does not exist: {docked_sdf}"
        )

    if args.cutoff <= 0:
        raise ValueError("--cutoff must be > 0")

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    check_obabel(args.obabel)

    print("=" * 76)
    print("DeepPROTACs pocket extraction")
    print(f"pair_id:         {args.pair_id}")
    print(f"side:            {args.side}")
    print(f"protein_pdb:     {protein_pdb}")
    print(f"docked_sdf:      {docked_sdf}")
    print(f"docking mode:    {args.mode}")
    print(f"cutoff:          {args.cutoff:.2f} Å")
    print(f"include HETATM:  {args.include_hetatm}")
    print(f"out_dir:         {out_dir}")
    print("=" * 76)

    # -----------------------------------------------------------------
    # 1. Read the selected Smina docking mode.
    # -----------------------------------------------------------------
    ligand_mol, n_modes = load_docking_mode(
        docked_sdf,
        args.mode,
    )

    ligand_xyz = get_heavy_atom_coords_from_mol(
        ligand_mol
    )

    # -----------------------------------------------------------------
    # 2. Read protein atoms in the original receptor coordinate frame.
    # -----------------------------------------------------------------
    protein_atoms = load_protein_atoms(
        protein_pdb,
        include_hetatm=args.include_hetatm,
    )

    protein_heavy_count = sum(
        not is_hydrogen_atom(a)
        for a in protein_atoms
    )

    # -----------------------------------------------------------------
    # 3. Select whole residues within cutoff.
    # -----------------------------------------------------------------
    (
        selected_residues,
        residue_stats,
        global_min_distance,
    ) = select_residues_within_cutoff(
        protein_atoms,
        ligand_xyz,
        args.cutoff,
    )

    if not selected_residues:
        raise RuntimeError(
            "No protein residue was found within the requested cutoff.\n"
            f"Minimum ligand-protein heavy-atom distance: "
            f"{global_min_distance:.3f} Å\n"
            "This usually indicates either:\n"
            "  1) the cutoff is too small, or\n"
            "  2) the docked ligand and original PDB are not in the "
            "same coordinate frame."
        )

    # -----------------------------------------------------------------
    # 4. Output names.
    # -----------------------------------------------------------------
    selected_ligand_sdf = out_dir / "selected_docked_ligand.sdf"
    pocket_pdb = out_dir / "selected_pocket.pdb"
    residue_csv = out_dir / "selected_pocket_residues.csv"

    if args.side == "e3":
        final_ligand_mol2 = out_dir / "ligase_ligand.mol2"
        final_pocket_mol2 = out_dir / "ligase_pocket.mol2"
    else:
        final_ligand_mol2 = out_dir / "target_ligand.mol2"
        final_pocket_mol2 = out_dir / "target_pocket.mol2"

    # -----------------------------------------------------------------
    # 5. Write selected ligand and pocket PDB.
    # -----------------------------------------------------------------
    write_single_mol_sdf(
        ligand_mol,
        selected_ligand_sdf,
    )

    pocket_atom_count = write_selected_residue_pdb(
        protein_pdb,
        pocket_pdb,
        selected_residues,
        include_hetatm=args.include_hetatm,
    )

    # -----------------------------------------------------------------
    # 6. Write residue table.
    # -----------------------------------------------------------------
    with residue_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        fields = [
            "chain_id",
            "resseq",
            "icode",
            "resname",
            "min_distance_A",
            "num_atoms_written",
            "num_heavy_atoms",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in residue_stats:
            writer.writerow(
                {
                    **row,
                    "min_distance_A": (
                        f"{row['min_distance_A']:.6f}"
                    ),
                }
            )

    # -----------------------------------------------------------------
    # 7. Convert both structures to MOL2 for DeepPROTACs.
    # -----------------------------------------------------------------
    ligand_obabel_stdout = convert_with_obabel(
        args.obabel,
        selected_ligand_sdf,
        final_ligand_mol2,
        add_h=False,
    )

    pocket_obabel_stdout = convert_with_obabel(
        args.obabel,
        pocket_pdb,
        final_pocket_mol2,
        add_h=False,
    )

    # Save conversion logs for reproducibility/debugging.
    (out_dir / "ligand_obabel.log").write_text(
        ligand_obabel_stdout,
        encoding="utf-8",
        errors="ignore",
    )

    (out_dir / "pocket_obabel.log").write_text(
        pocket_obabel_stdout,
        encoding="utf-8",
        errors="ignore",
    )

    # -----------------------------------------------------------------
    # 8. Summary / sanity checks.
    # -----------------------------------------------------------------
    chain_ids = sorted(
        set(row["chain_id"] for row in residue_stats)
    )

    summary_lines = [
        f"pair_id: {args.pair_id}",
        f"side: {args.side}",
        f"protein_pdb: {protein_pdb}",
        f"docked_sdf: {docked_sdf}",
        f"selected_mode: {args.mode}",
        f"available_modes_in_sdf: {n_modes}",
        f"cutoff_A: {args.cutoff:.3f}",
        f"include_hetatm: {args.include_hetatm}",
        "",
        "========== coordinate sanity ==========",
        (
            "minimum ligand-protein heavy-atom distance_A: "
            f"{global_min_distance:.3f}"
        ),
        (
            "ligand heavy atoms: "
            f"{ligand_xyz.shape[0]}"
        ),
        (
            "protein atoms parsed: "
            f"{len(protein_atoms)}"
        ),
        (
            "protein heavy atoms parsed: "
            f"{protein_heavy_count}"
        ),
        "",
        "========== pocket ==========",
        (
            "selected residues: "
            f"{len(selected_residues)}"
        ),
        (
            "selected chains: "
            + (
                ",".join(chain_ids)
                if chain_ids
                else "none"
            )
        ),
        (
            "pocket atoms written: "
            f"{pocket_atom_count}"
        ),
        "",
        "========== DeepPROTACs outputs ==========",
        f"ligand_mol2: {final_ligand_mol2}",
        f"pocket_mol2: {final_pocket_mol2}",
        f"pocket_pdb: {pocket_pdb}",
        f"residue_csv: {residue_csv}",
        "",
        (
            "Pocket definition: whole protein residues with at least "
            f"one heavy atom within {args.cutoff:.1f} Å of any "
            "docked ligand heavy atom."
        ),
    ]

    summary_path = out_dir / "pocket_extraction_summary.txt"
    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("")
    print("========== Pocket summary ==========")
    print(
        f"minimum ligand-protein distance: "
        f"{global_min_distance:.3f} Å"
    )
    print(
        f"selected residues:                "
        f"{len(selected_residues)}"
    )
    print(
        f"pocket atoms written:             "
        f"{pocket_atom_count}"
    )
    print(
        f"selected chains:                  "
        f"{','.join(chain_ids) if chain_ids else 'none'}"
    )

    print("")
    print("DeepPROTACs outputs:")
    print(f"  ligand: {final_ligand_mol2}")
    print(f"  pocket: {final_pocket_mol2}")
    print(f"  summary: {summary_path}")

    print("")
    print("Finished successfully.")


if __name__ == "__main__":
    main()
