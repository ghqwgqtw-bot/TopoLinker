from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

input_sdf = "./datasets/pretrain/train/linker.sdf"
output_smi = "./datasets/reference_linkers.smi"

supplier = Chem.SDMolSupplier(
    input_sdf,
    sanitize=False,
    removeHs=False,
)

reference_smiles = set()

for mol in supplier:
    if mol is None:
        continue

    mol = Chem.Mol(mol)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        continue

    # 与当前 Novelty 计算保持一致：忽略立体化学
    Chem.RemoveStereochemistry(mol)

    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        pass

    # 与 evaluator 的 linker canonicalization 保持一致
    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        mol = enumerator.Canonicalize(mol)
    except Exception:
        pass

    smi = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )

    if smi:
        reference_smiles.add(smi)

reference_smiles = sorted(reference_smiles)

with open(output_smi, "w", encoding="utf-8") as f:
    for smi in reference_smiles:
        f.write(smi + "\n")

print("Input SDF:", input_sdf)
print("Unique training linkers:", len(reference_smiles))
print("Saved to:", output_smi)