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
# Extract unique non-null SMILES from the input CSV.
def create_unique_smiles(input_path):
    smiles = pd.read_csv(input_path)["Smiles"].dropna().unique().tolist()
    return smiles

# Align two conformers of the same molecule and return their RMSD.
def getRMS(mol, c1, c2):
    (rms, trans) = AllChem.GetAlignmentTransform(mol, mol, c1, c2)
    return rms

# Generate, optimize, and select 3D conformers from SMILES inputs.
# Save the selected conformers as SDF records.
def compute_confs_worker(
        smifile,          # Input SMILES collection
        sdffile,          # Output SDF path
        maxconfs=50,      # Base conformer count
        sample_mult=1,
        seed=9162006,
        rms_threshold=0.5,
        energy=10.0,      # Candidate energy window
        mmff=True,        # Prefer MMFF when available
        nomin=False,
        etkdg=True,       # Enable ETKDG-based embedding
        jpsettings=True,  # Adjust conformer count using rotatable bonds
):
    print("Worker started")

    outf = open(sdffile, "w+")
    sdwriter = Chem.SDWriter(outf)

    if sdwriter is None:
        print("Could not open ", sdffile)
        sys.exit(-1)

    # Write SDF records using kekulized bond representations.
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
            # 1. Select the longest SMILES component when multiple components are present
            pieces = str(smi).split(".")
            if len(pieces) > 1:
                smi = max(pieces, key=len)
                print(f"Taking longest SMILES component: {smi}")

            # ------------------------------------------------
            # 2. SMILES -> mol
            # ------------------------------------------------
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                n_mol_none += 1
                print(f"[MOL_FROM_SMILES_FAIL] {name}")
                continue

            # ------------------------------------------------
            # 3. Perform initial full sanitization
            #    Reject molecules that fail sanitization.
            # ------------------------------------------------
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                n_initial_sanitize_fail += 1
                print(f"[INITIAL_SANITIZE_FAIL] {name} | {repr(e)}")
                continue

            mol.SetProp("_Name", name)

            # 4. Add hydrogens and generate 3D conformers
            mol_h = Chem.AddHs(mol)

            rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)

            # Keep maxconfs unchanged across molecules.
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

            # 5. Generate conformers using ETKDGv3 when available
            if etkdg:
                try:
                    params = AllChem.ETKDGv3()
                except Exception:
                    params = AllChem.ETKDG()

                params.randomSeed = int(seed)
                params.numThreads = 0
                params.pruneRmsThresh = float(rms_threshold)

                # Enable ring torsion options when supported by RDKit.
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
            # 6. Optimize conformers
            #    Prefer MMFF and fall back to UFF if needed.
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

            # 7. Remove hydrogens
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

            # Sanitize again after removing hydrogens.
            try:
                mol_check = Chem.Mol(mol_no_h)
                Chem.SanitizeMol(mol_check)
            except Exception as e:
                n_remove_h_fail += 1
                print(f"[AFTER_REMOVE_H_FULL_SANITIZE_FAIL] {name} | {repr(e)}")
                continue

            # ------------------------------------------------
            # 8. Select the lowest-energy conformer
            #    Prefer converged conformers; otherwise consider all valid conformers.
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

            # Candidate conformers within the energy window.
            window_cids = [
                cid for cid in candidate_cids
                if energy_by_cid[cid] <= min_energy + float(energy)
            ]

            best_cid = sorted(
                window_cids,
                key=lambda cid: energy_by_cid[cid],
            )[0]

            # 9. Perform full sanitization before writing
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

# Identify dummy exit atoms with the symbol "*".
def get_exits(mol):

    exits = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == '*':
            exits.append(atom)
    return exits

# Mark the selected anchor atom.
def set_anchor_flags(mol, anchor_idx):

    for atom in mol.GetAtoms():
        if atom.GetIdx() == anchor_idx:
            atom.SetProp('_Anchor', '1')
        else:
            atom.SetProp('_Anchor', '0')

# Return indices of atoms carrying the _Anchor flag.
def get_anchors_idx(mol):
    anchors_idx = []
    for atom in mol.GetAtoms():
        if atom.GetProp('_Anchor') == '1':
            anchors_idx.append(atom.GetIdx())

    return anchors_idx

# Remove the fragment exit atom and mark its neighbor as the anchor.
def update_fragment(frag):

    exits = get_exits(frag)
    # Reject fragments containing more than one exit atom.
    if len(exits) > 1:
        raise Exception('Found more than one exits in fragment')
    exit = exits[0]

    bonds = exit.GetBonds()
    # Each exit atom may have at most one bond.
    if len(bonds) > 1:
        raise Exception('Exit atom has more than 1 bond')
    bond = bonds[0]

    exit_idx = exit.GetIdx()
    # Identify the atom adjacent to the exit atom.
    source_idx = bond.GetBeginAtomIdx()
    target_idx = bond.GetEndAtomIdx()
    anchor_idx = source_idx if target_idx == exit_idx else target_idx
    set_anchor_flags(frag, anchor_idx)

    efragment = Chem.EditableMol(frag)
    efragment.RemoveBond(source_idx, target_idx)
    efragment.RemoveAtom(exit_idx)

    return efragment.GetMol()

# Remove linker exit atoms and their incident bonds.
def update_linker(linker):

    exits = get_exits(linker)
    # Reject linkers containing more than two exit atoms.
    if len(exits) > 2:
        raise Exception('Found more than two exits in linker')

    # Process exit atoms in descending index order.
    exits = sorted(exits, key=lambda e: e.GetIdx(), reverse=True)
    elinker = Chem.EditableMol(linker)

    # Remove bonds incident to exit atoms.
    for exit in exits:
        bonds = exit.GetBonds()
        # Each exit atom may have at most one bond.
        if len(bonds) > 1:
            raise Exception('Exit atom has more than 1 bond')
        bond = bonds[0]
        source_idx = bond.GetBeginAtomIdx()
        target_idx = bond.GetEndAtomIdx()
        elinker.RemoveBond(source_idx, target_idx)

    # Remove exit atoms.
    for exit in exits:
        elinker.RemoveAtom(exit.GetIdx())

    return elinker.GetMol()

# Create an RDKit conformer from 3D coordinates without attaching it to a molecule.
def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

# Transfer 3D coordinates from the parent molecule to matching fragments.
def transfer_conformers(frag, mol):

    # Find all substructure matches in the parent molecule.
    matches = mol.GetSubstructMatches(frag)
    if len(matches) < 1:
        raise Exception('Could not find fragment or linker matches')

    match2conf = {}
    for match in matches:
        mol_coords = mol.GetConformer().GetPositions()
        # Extract coordinates in fragment atom order.
        frag_coords = mol_coords[np.array(match)]
        frag_conformer = create_conformer(frag_coords)
        match2conf[match] = frag_conformer

    return match2conf
# Find pairwise disjoint fragment1, fragment2, and linker matches.
def find_non_intersecting_matches(matches1, matches2, matches3):
    # Enumerate the Cartesian product of the three match sets.
    triplets = list(itertools.product(matches1, matches2, matches3))
    non_intersecting_matches = set()
    for m1, m2, m3 in triplets:
        m1m2 = set(m1) & set(m2)
        m1m3 = set(m1) & set(m3)
        m2m3 = set(m2) & set(m3)
        # Keep only combinations with no overlapping atoms.
        if len(m1m2 | m1m3 | m2m3) == 0:
            non_intersecting_matches.add((m1, m2, m3))
    return list(non_intersecting_matches)

# Require the linker match to contact both fragment matches.
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
    Return the neighbors of atom_idx that belong to atom_idx_set.
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
    Retain matches where each designated fragment anchor has
    exactly one bonded neighbor in the linker.

    Anchor indices are local to their respective fragments,
    not atom indices in the full molecule.
    """

    good_matches = []

    for m1, m2, lm in non_intersecting_matches:
        # Map fragment-local anchor indices to full-molecule atom indices.
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

# Select non-overlapping fragment-linker matches with valid anchor connections.
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

    # Use anchor-aware matching when both anchor indices are provided.
    if frag1_anchor_idx is not None and frag2_anchor_idx is not None:
        anchor_matches = find_matches_with_anchor_connected(
            non_intersecting_matches=non_intersecting_matches,
            mol=mol,
            frag1_anchor_idx=frag1_anchor_idx,
            frag2_anchor_idx=frag2_anchor_idx,
        )

        if len(anchor_matches) > 0:
            return anchor_matches

        # Reject the sample if no anchor-consistent match is found.
        return []
    else:
        print("**************************************************")

# Construct fragment and linker conformers from the reference 3D structure.
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

# Build 3D fragment and linker records from SDF molecules
# and the corresponding decomposition metadata.
def process_sdf(sdf_path, table, progress=True):

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
    supplier = tqdm(supplier, total=len(supplier)) if progress else supplier
    for mol in supplier:
        if mol is None:
            continue

        # 1. Sanitize without kekulization
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

        # 2. Check full sanitization, including kekulization.
        #    Reject molecules that fail the complete RDKit checks.
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

        # 3. Check that a conformer is present
        if mol.GetNumConformers() == 0:
            continue

        # 4. Sanitize the source molecule before decomposition
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"[PROCESS_SDF_FINAL_SANITIZE_FAIL] {repr(e)}")
            continue

        # Read the original molecule name.
        mol_name = mol.GetProp('_Name')
        # Generate canonical SMILES and update the molecule name.
        mol_smi = Chem.MolToSmiles(mol)
        mol.SetProp('_Name', mol_smi)

        # Find corresponding linker and fragment records.
        for warhead_smi, e3_ligand_smi, linker_smi, uniprot, e3_ligase_uniprot, Molecular_Weight, Exact_Mass, XLogP3, Heavy_Atom_Count, Ring_Count, Hydrogen_Bond_Acceptor_Count, Hydrogen_Bond_Donor_Count, Rotatable_Bond_Count, Topological_Polar_Surface_Area in table[table.Smiles == mol_name][['warhead_smiles', 'e3_ligand_smiles', 'linker_smiles', 'Uniprot', 'e3_ligase_Uniprot', 'Molecular Weight', 'Exact Mass', 'XLogP3', 'Heavy Atom Count', 'Ring Count', 'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count', 'Rotatable Bond Count', 'Topological Polar Surface Area']].values:
            try:
                # Construct fragment and linker molecules with 3D conformers.
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

            # Combine fragments in warhead-then-E3-ligand order.
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

            # Read anchor indices within each fragment.
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
            # The warhead anchor already uses the combined fragment index space.
            warhead_anchor_global = warhead_anchor_idx[0]
            # Offset the E3 anchor index by the number of warhead atoms.
            e3_anchor_global = n_warhead_atoms + e3_anchor_idx[0]
            anchors_idx = [warhead_anchor_global, e3_anchor_global]

            # Store processed molecules and fragments.
            molecules.append(mol)
            fragments.append(frags)
            linkers.append(linker)
            # Record molecular metadata and atom mappings.
            out_table.append({
                'uuid': uuid,
                'molecule': mol_smi,
                'fragments': Chem.MolToSmiles(frags),
                'linker': Chem.MolToSmiles(linker),
                'full_atom_indices': json.dumps(dataset_to_full_atom_idx),
                'anchor_1': anchors_idx[0],
                'anchor_2': anchors_idx[1],

                # Atom counts and anchor indices used to construct fragment-specific masks.
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
    Split PROTAC records by linker similarity using Butina clustering.

    Pretraining splits contain unique PROTACs, while RL splits retain
    multiple protein-context records associated with the same PROTAC.

    Assign each complete linker cluster to one split, ensuring that
    identical linker cores and all protein contexts of the same PROTAC
    remain in the same split.

    Approximate the requested train/validation/test ratios without
    splitting clusters. Butina clustering does not guarantee that every
    pair of linkers above the similarity threshold belongs to the same
    cluster.

    Args:
        similarity_threshold: Tanimoto similarity used to define the
            Butina distance threshold (1 - similarity_threshold).
        use_local_search: Whether to refine greedy cluster assignments
            to reduce deviations from the requested split ratios.
    """

    df = df.copy()

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    # 1. Canonicalize SMILES
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
        Remove dummy attachment atoms before linker similarity clustering.

        Canonicalize the remaining linker core for fingerprint generation.
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
    print(f"Original records: {before_num}")
    print(f"Valid records: {after_num}")
    print(f"Filtered records: {before_num - after_num}")

    if len(df) == 0:
        raise ValueError("No valid samples remain after SMILES canonicalization.")

    # 2. Compute linker scaffolds
    def get_linker_scaffold(smiles):
        if pd.isna(smiles):
            return None

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)

        # Acyclic linkers may have empty Murcko scaffolds.
        # Fall back to the canonical linker core.
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return Chem.MolToSmiles(mol, canonical=True)

        return Chem.MolToSmiles(scaffold, canonical=True)

    df["linker_scaffold"] = df["linker_core_id"].apply(get_linker_scaffold)

    # 3. Compute Morgan fingerprints for unique linker cores
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
        raise ValueError("No valid linkers are available for dataset splitting.")

    # 4. Cluster linkers using Butina
    # Butina clustering uses distances rather than similarities.
    # Distance = 1 - Tanimoto similarity.
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
        raise ValueError(
            "Some linkers were not assigned to a cluster; "
            "check fingerprints and linker SMILES."
        )

    # 5. Check whether a PROTAC maps to multiple linker clusters
    protac_cluster_count = (
        df.groupby("protac_id")["linker_cluster"]
        .nunique()
    )

    bad_protacs = protac_cluster_count[protac_cluster_count > 1]

    if len(bad_protacs) > 0:
        raise ValueError(
            f"{len(bad_protacs)} PROTAC IDs map to multiple linker clusters; "
            f"check for inconsistent linker SMILES for the same PROTAC."
        )

    # 6. Assign linker clusters to train, val, and test
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

    # Assign larger clusters first to reduce split-ratio imbalance.
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
        Measure deviation from the requested split sizes.

        Use squared errors to penalize larger deviations.
        """
        loss = 0.0
        for split in ["train", "val", "test"]:
            loss += (counts[split] - targets[split]) ** 2
        return loss

    # ---------- 6.1 Greedy assignment ----------
    for cid, size, _ in cluster_items:
        best_split = None
        best_loss = None

        # Evaluate assigning the current cluster to each split.
        # Select the split minimizing the overall squared error.
        for split in ["train", "val", "test"]:
            temp_counts = split_counts.copy()
            temp_counts[split] += size

            loss = split_loss(temp_counts, target_counts)

            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_split = split

        cluster_to_split[cid] = best_split
        split_counts[best_split] += size

    # ---------- 6.2 Optional local search ----------
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

        print(f"Local search iterations: {n_iter}")

    df["split"] = df["linker_cluster"].map(cluster_to_split)

    # 7. Build RL datasets
    train_rl_df = df[df["split"] == "train"].copy()
    val_rl_df = df[df["split"] == "val"].copy()
    test_rl_df = df[df["split"] == "test"].copy()

    # 8. Build pretraining datasets
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

    # 9. Check for data leakage across splits
    def check_no_overlap(col, name):
        train_set = set(train_rl_df[col].dropna())
        val_set = set(val_rl_df[col].dropna())
        test_set = set(test_rl_df[col].dropna())

        assert len(train_set & val_set) == 0, f"Data leakage: train and val share {name} values"
        assert len(train_set & test_set) == 0, f"Data leakage: train and test share {name} values"
        assert len(val_set & test_set) == 0, f"Data leakage: val and test share {name} values"

    check_no_overlap("protac_id", "PROTAC")
    check_no_overlap("linker_core_id", "linker")
    check_no_overlap("linker_cluster", "linker_cluster")

    # 10. Save split datasets
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

    # Save the complete split-assignment table for auditing.
    split_table_csv = os.path.join(
        out_prefix,
        "protac_split_by_linker_cluster.csv"
    )
    df.to_csv(split_table_csv, index=False)

    # 11. Report split statistics
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
    print(f"Valid input records: {len(df)}")
    print(f"Unique PROTACs: {n_total_unique_protac}")
    print(f"Unique linkers: {len(valid_linkers)}")
    print(f"Linker clusters: {len(clusters)}")
    print(f"Tanimoto similarity threshold: {similarity_threshold}")

    print("\nTarget unique PROTAC counts:")
    print(f"train target: {target_counts['train']:.2f}")
    print(f"val target:   {target_counts['val']:.2f}")
    print(f"test target:  {target_counts['test']:.2f}")

    print("\nActual unique PROTAC counts:")
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
    print(f"Complete split table: {split_table_csv}")

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
    Write molecules to an SDF file with proper resource cleanup.

    Ensure that the RDKit writer is closed even if writing fails.
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
    # Input CSV path.
    if ori_dir == "./datasets/pretrain/":
        table_path = f"{ori_dir}{data_dir}/protac_{data_dir}_pretrain.csv"
    else:
        table_path = f"{ori_dir}{data_dir}/protac_{data_dir}_rl.csv"

    # Extract unique molecule SMILES.
    smiles = create_unique_smiles(table_path)
    sdf_file_path = f"{ori_dir}{data_dir}/rdkit_conf.sdf"

    print("CSV path:", table_path)
    print("SDF path:", sdf_file_path)

    # Generate 3D conformers and save them to an SDF file if needed.
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
    # Load the input table.
    table = pd.read_csv(table_path)
    # Build 3D fragment and linker records from the SDF and decomposition table.
    molecules, fragments, linkers, out_table = process_sdf(sdf_file_path, table, progress=True)

    # Verify that all output collections have matching lengths.
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

    # Write temporary output files first.
    write_sdf_strict(molecules, tmp_mol_path, "MOL")
    write_sdf_strict(fragments, tmp_frag_path, "FRAGMENTS")
    write_sdf_strict(linkers, tmp_link_path, "LINKER")

    out_table.to_csv(tmp_table_path, index=False)

    # Remove existing output files before replacing them.
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


