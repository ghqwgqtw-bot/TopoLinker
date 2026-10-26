import pandas as pd
from rdkit import Chem
import numpy as np
from tqdm import tqdm
from rdkit import RDLogger
from rdkit.Chem import SaltRemover
RDLogger.DisableLog('rdApp.*')
remover = SaltRemover.SaltRemover()

# Load input tables
protac = pd.read_csv('./datasets/protac.csv', low_memory=False)
linker = pd.read_csv('./datasets/linker.csv')
e3 = pd.read_csv('./datasets/e3_ligand.csv')

protac_df = protac.copy()
linker_df = linker.copy()
e3_df = e3.copy()

# Treat empty required fields as missing and drop incomplete records.
protac_df[['Uniprot', 'Target', 'E3 ligase', 'Smiles', 'Molecular Weight', 'Exact Mass', 'XLogP3', 'Heavy Atom Count', 'Ring Count', 'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count', 'Rotatable Bond Count', 'Topological Polar Surface Area']] = protac_df[['Uniprot', 'Target', 'E3 ligase', 'Smiles', 'Molecular Weight', 'Exact Mass', 'XLogP3', 'Heavy Atom Count', 'Ring Count', 'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count', 'Rotatable Bond Count', 'Topological Polar Surface Area']].replace('', pd.NA)
linker_df['Smiles'] = linker_df['Smiles'].replace('', pd.NA)
e3_df[['Uniprot', 'Target', 'Smiles']] = e3_df[['Uniprot', 'Target', 'Smiles']].replace('', pd.NA)
protac_df = protac_df.dropna(subset=['Uniprot', 'Target', 'E3 ligase', 'Smiles', 'Molecular Weight', 'Exact Mass', 'XLogP3', 'Heavy Atom Count', 'Ring Count', 'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count', 'Rotatable Bond Count', 'Topological Polar Surface Area'])
linker_df = linker_df.dropna(subset=['Smiles'])
e3_df = e3_df.dropna(subset=['Uniprot', 'Target', 'Smiles'])

# Remove duplicate PROTAC and E3 ligand records
protac_df = protac_df.drop_duplicates(subset=['Uniprot', 'Target', 'E3 ligase', 'Smiles'])
e3_df = e3_df.drop_duplicates(subset=['Uniprot', 'Target', 'Smiles'])

protac_df = protac_df[['Compound ID', 'Uniprot', 'Target', 'E3 ligase', 'Smiles', 'Molecular Weight', 'Exact Mass', 'XLogP3', 'Heavy Atom Count', 'Ring Count', 'Hydrogen Bond Acceptor Count', 'Hydrogen Bond Donor Count', 'Rotatable Bond Count', 'Topological Polar Surface Area']]

# Helper functions
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
    """
    Extract masked atoms while retaining boundary atoms as dummies.

    Require exactly two bonds between retained and excluded atoms.
    """
    new_mask = mask.copy()
    emol = Chem.RWMol(Chem.Mol(pmol))
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

# Canonicalize SMILES
protac_df['Smiles'] = protac_df['Smiles'].apply(canon)
linker_df['Smiles'] = linker_df['Smiles'].apply(canon)
e3_df['Smiles'] = e3_df['Smiles'].apply(canon)

protac_df = protac_df.dropna(subset=['Smiles'])
linker_df = linker_df.dropna(subset=['Smiles'])
e3_df = e3_df.dropna(subset=['Smiles'])

# Construct RDKit molecule objects
protac_df['mol'] = protac_df['Smiles'].apply(Chem.MolFromSmiles)
linker_df['mol'] = linker_df['Smiles'].apply(Chem.MolFromSmiles)
e3_df['mol'] = e3_df['Smiles'].apply(Chem.MolFromSmiles)

# Sort linkers by descending atom count
linker_df['atom_num'] = linker_df['mol'].apply(
    lambda m: m.GetNumAtoms() if m else 0
)

linker_df = linker_df.sort_values('atom_num', ascending=False)


linker_list = []
for row in linker_df.itertuples():
    if row.mol:
        linker_list.append(
            (row.mol, row.Smiles)
        )

# Build the E3 ligand lookup table
e3_dict = {}

for row in e3_df.itertuples():
    # Use the Target field as the E3 ligase lookup key.
    key = getattr(row, 'Target', None)
    # Skip entries with a missing E3 ligase lookup key or RDKit molecule.
    if key is None or row.mol is None:
        continue
    if key not in e3_dict:
        e3_dict[key] = []
    e3_dict[key].append(
        (row.mol, row.Smiles, row.Uniprot)
    )

# Initialize decomposition results
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

# PROTAC decomposition
def split_protac(pmol, e3_name):
    natoms = pmol.GetNumAtoms()
    for linker_mol, linker_smiles in linker_list:
        # Find linker substructure matches in the PROTAC.
        matches = pmol.GetSubstructMatches(linker_mol)
        if not matches:
            continue
        # Check every linker match because a linker can occur more than once.
        for match in matches:
            # Mark the atoms belonging to this linker match.
            mask = np.zeros(natoms, dtype=bool)
            mask[list(match)] = True
            linker_frag = mol_from_mask(pmol, mask)

            if linker_frag is None:
                continue

            # Select atoms outside the linker match.
            remaining_mask = ~mask

            remaining_frag = mol_from_mask(pmol, remaining_mask)

            if remaining_frag is None:
                continue

            # Split the remaining structure into disconnected components.
            frags = Chem.GetMolFrags(remaining_frag, asMols=True)
            # Require exactly two disconnected fragment components.
            if len(frags) != 2:
                continue

            # Retrieve candidate E3 ligands for the specified ligase.
            e3_candidates = e3_dict.get(e3_name, [])
            for frag in frags:
                new_frag = remove_anchor(frag)
                if new_frag is None:
                    break
                for e3_mol, e3_smiles, e3_ligase_Uniprot in e3_candidates:
                    # Check whether the fragment matches an E3 ligand in both directions.
                    if new_frag.HasSubstructMatch(e3_mol) and e3_mol.HasSubstructMatch(new_frag):
                        warhead_frag = [f for f in frags if f is not frag]

                        if not warhead_frag:
                            continue
                        warhead = warhead_frag[0]
                        # Return linker, E3 ligand, E3 UniProt ID, and warhead.
                        return (
                            Chem.MolToSmiles(linker_frag),
                            Chem.MolToSmiles(frag),
                            e3_ligase_Uniprot,
                            Chem.MolToSmiles(warhead)
                        )
    return None, None, None, None

# Process PROTAC records
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

# Save matched PROTAC records
protac_df = protac_df.dropna(subset=['e3_ligase_Uniprot', 'linker_smiles', 'e3_ligand_smiles', 'warhead_smiles', 'mol'])
protac_df.drop(columns=['mol'], inplace=True)

protac_df.to_csv(
    './datasets/protac_split_result_clean.csv',
    index=False
)

print("PROTAC splitting completed.")