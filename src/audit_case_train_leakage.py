import os
import json
import csv
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from data.molecule_builder import build_molecules


RDLogger.DisableLog("rdApp.*")


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASETS_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
)

# 正式的数据划分文件
SPLIT_CSV = os.path.join(
    DATASETS_DIR,
    "protac_split_by_linker_cluster.csv",
)

# 三个 retrospective case
CASE_FILES = {
    "MZ1": os.path.join(
        DATASETS_DIR,
        "cases",
        "case_5T35_MZ1",
        "single_batch.pt",
    ),
    "A1ANN": os.path.join(
        DATASETS_DIR,
        "cases",
        "case_9B9W_A1ANN",
        "single_batch.pt",
    ),
    "NEP108": os.path.join(
        DATASETS_DIR,
        "cases",
        "case_8X7G_NEP108",
        "single_batch.pt",
    ),
}

CASE_PDB = {
    "MZ1": "5T35",
    "A1ANN": "9B9W",
    "NEP108": "8X7G",
}

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "paper_tables",
    "case_leakage_audit",
)

# 必须和实际 linker clustering 保持一致
TANIMOTO_THRESHOLD = 0.70
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


# ============================================================
# Chemistry helpers
# ============================================================

def canonicalize_smiles(
    smiles: Optional[str],
) -> Optional[str]:
    if smiles is None:
        return None

    smiles = str(smiles).strip()

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
    )


def canonicalize_linker_core(
    smiles: Optional[str],
) -> Optional[str]:
    """
    与数据划分代码保持一致：

    1. SMILES -> Mol
    2. 删除 dummy atoms (*)
    3. sanitize
    4. canonical SMILES
    """

    if smiles is None:
        return None

    smiles = str(smiles).strip()

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    dummy_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]

    if dummy_indices:
        rw_mol = Chem.RWMol(
            Chem.Mol(mol)
        )

        for idx in sorted(
            dummy_indices,
            reverse=True,
        ):
            rw_mol.RemoveAtom(idx)

        mol = rw_mol.GetMol()

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    if mol.GetNumAtoms() == 0:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
    )


def get_morgan_fp(
    smiles: str,
):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise RuntimeError(
            f"Cannot parse linker SMILES:\n{smiles}"
        )

    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        MORGAN_RADIUS,
        nBits=MORGAN_BITS,
    )


def tanimoto_similarity(
    fp1,
    fp2,
) -> float:
    return float(
        DataStructs.TanimotoSimilarity(
            fp1,
            fp2,
        )
    )


# ============================================================
# Load official split table
# ============================================================

def load_split_table() -> pd.DataFrame:

    if not os.path.exists(SPLIT_CSV):
        raise FileNotFoundError(
            f"Cannot find official split table:\n{SPLIT_CSV}"
        )

    df = pd.read_csv(
        SPLIT_CSV,
        low_memory=False,
    )

    required_columns = {
        "Smiles",
        "linker_smiles",
        "protac_id",
        "linker_core_id",
        "linker_cluster",
        "split",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            "The official split table is missing columns:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    return df


# ============================================================
# Load case single_batch.pt
# ============================================================

def load_case_file(
    case_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    path = CASE_FILES[case_name]

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{case_name}: single_batch.pt not found:\n{path}"
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            f"{case_name}: single_batch.pt is not a dict."
        )

    if "batch" not in data:
        raise KeyError(
            f"{case_name}: single_batch.pt has no 'batch'."
        )

    if "reference" not in data:
        raise KeyError(
            f"{case_name}: single_batch.pt has no 'reference'."
        )

    return (
        data["batch"],
        data["reference"],
    )


def unwrap_single_sample(
    batch: Dict[str, Any],
) -> Dict[str, Any]:

    sample = {}

    for key, value in batch.items():

        if torch.is_tensor(value):

            if (
                value.dim() > 0
                and value.size(0) == 1
            ):
                sample[key] = value[0]
            else:
                sample[key] = value

        else:
            sample[key] = value

    return sample


# ============================================================
# Reconstruct reference molecule from single_batch.pt
# ============================================================

def _ensure_batch_dim(
    value: torch.Tensor,
) -> torch.Tensor:
    return value.unsqueeze(0)


def build_reference_molecule(
    sample: Dict[str, Any],
) -> Chem.Mol:

    required_keys = [
        "one_hot",
        "positions",
        "atom_mask",
        "full_bond_types",
        "formal_charges",
        "atom_is_aromatic",
        "num_explicit_hs",
        "no_implicit",
    ]

    missing = [
        k
        for k in required_keys
        if k not in sample
    ]

    if missing:
        raise KeyError(
            "Reference sample missing keys: "
            + ", ".join(missing)
        )

    try:
        mols = build_molecules(
            one_hot=_ensure_batch_dim(
                sample["one_hot"]
            ),
            x=_ensure_batch_dim(
                sample["positions"]
            ),
            node_mask=_ensure_batch_dim(
                sample["atom_mask"]
            ),
            fragment_bond_types=_ensure_batch_dim(
                sample["full_bond_types"]
            ),
            use_template_fragment_bonds=True,
            formal_charges=_ensure_batch_dim(
                sample["formal_charges"]
            ),
            atom_is_aromatic=_ensure_batch_dim(
                sample["atom_is_aromatic"]
            ),
            num_explicit_hs=_ensure_batch_dim(
                sample["num_explicit_hs"]
            ),
            no_implicit=_ensure_batch_dim(
                sample["no_implicit"]
            ),
            atom_props_mask=None,
            use_pred_bonds=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to reconstruct reference molecule."
        ) from exc

    if not mols:
        raise RuntimeError(
            "Reference molecule reconstruction returned no molecules."
        )

    mol = mols[0]

    if mol is None:
        raise RuntimeError(
            "Reference molecule reconstruction returned None."
        )

    return mol


# ============================================================
# Extract case reference linker
# ============================================================

def extract_case_reference_linker(
    batch: Dict[str, Any],
) -> Tuple[str, int]:

    sample = unwrap_single_sample(
        batch
    )

    mol = build_reference_molecule(
        sample
    )

    if "linker_mask" not in sample:
        raise KeyError(
            "Case sample does not contain linker_mask."
        )

    linker_mask = (
        sample["linker_mask"]
        .detach()
        .cpu()
        .bool()
        .view(-1)
    )

    linker_indices = (
        torch.nonzero(
            linker_mask,
            as_tuple=False,
        )
        .view(-1)
        .tolist()
    )

    if len(linker_indices) == 0:
        raise RuntimeError(
            "Reference linker contains zero atoms."
        )

    raw_linker_smiles = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=[
            int(x)
            for x in linker_indices
        ],
        canonical=True,
    )

    if not raw_linker_smiles:
        raise RuntimeError(
            "Could not extract reference linker SMILES."
        )

    linker_core = canonicalize_linker_core(
        raw_linker_smiles
    )

    if linker_core is None:
        raise RuntimeError(
            "Could not canonicalize reference linker."
        )

    return (
        linker_core,
        len(linker_indices),
    )


# ============================================================
# Extract reference full PROTAC
# ============================================================

def extract_case_reference_protac(
    reference: Dict[str, Any],
) -> str:

    reference_smiles = reference.get(
        "reference_smiles",
        None,
    )

    if reference_smiles is None:
        raise RuntimeError(
            "reference_smiles is missing."
        )

    canonical = canonicalize_smiles(
        reference_smiles
    )

    if canonical is None:
        raise RuntimeError(
            "reference_smiles cannot be canonicalized."
        )

    return canonical


# ============================================================
# Split libraries
# ============================================================

def prepare_split_libraries(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    train_df = df.loc[
        train_mask
    ].copy()

    val_df = df.loc[
        val_mask
    ].copy()

    test_df = df.loc[
        test_mask
    ].copy()

    train_protacs = set(
        train_df["protac_id"]
        .dropna()
        .astype(str)
    )

    val_protacs = set(
        val_df["protac_id"]
        .dropna()
        .astype(str)
    )

    test_protacs = set(
        test_df["protac_id"]
        .dropna()
        .astype(str)
    )

    train_linkers = set(
        train_df["linker_core_id"]
        .dropna()
        .astype(str)
    )

    val_linkers = set(
        val_df["linker_core_id"]
        .dropna()
        .astype(str)
    )

    test_linkers = set(
        test_df["linker_core_id"]
        .dropna()
        .astype(str)
    )

    # --------------------------------------------------------
    # linker_core_id -> linker_cluster
    # --------------------------------------------------------

    linker_cluster_map_df = (
        df[
            [
                "linker_core_id",
                "linker_cluster",
            ]
        ]
        .dropna(
            subset=[
                "linker_core_id",
                "linker_cluster",
            ]
        )
        .drop_duplicates(
            subset=[
                "linker_core_id",
            ]
        )
    )

    linker_to_cluster = {}

    for _, row in linker_cluster_map_df.iterrows():

        linker = str(
            row["linker_core_id"]
        )

        cluster = int(
            row["linker_cluster"]
        )

        linker_to_cluster[
            linker
        ] = cluster

    # --------------------------------------------------------
    # Save complete split dataframes for later overlap tracing
    # --------------------------------------------------------

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,

        "train_protacs": train_protacs,
        "val_protacs": val_protacs,
        "test_protacs": test_protacs,

        "train_linkers": train_linkers,
        "val_linkers": val_linkers,
        "test_linkers": test_linkers,

        "linker_to_cluster": linker_to_cluster,
    }


# ============================================================
# Max Tanimoto against train
# ============================================================

def calculate_max_train_tanimoto(
    reference_linker_core: str,
    train_linkers: set,
) -> Tuple[
    float,
    Optional[str],
]:

    ref_fp = get_morgan_fp(
        reference_linker_core
    )

    best_score = -1.0
    best_linker = None

    for linker_smiles in train_linkers:

        train_fp = get_morgan_fp(
            linker_smiles
        )

        score = tanimoto_similarity(
            ref_fp,
            train_fp,
        )

        if score > best_score:
            best_score = score
            best_linker = linker_smiles

    return (
        best_score,
        best_linker,
    )


# ============================================================
# Audit one retrospective case
# ============================================================

def audit_case(
    case_name: str,
    split_libraries: Dict[str, Any],
) -> Dict[str, Any]:

    batch, reference = load_case_file(
        case_name
    )

    # --------------------------------------------------------
    # Reference identities
    # --------------------------------------------------------

    reference_protac = (
        extract_case_reference_protac(
            reference
        )
    )

    reference_linker, linker_atom_count = (
        extract_case_reference_linker(
            batch
        )
    )

    # --------------------------------------------------------
    # Exact PROTAC presence
    # --------------------------------------------------------

    protac_in_train = (
        reference_protac
        in split_libraries[
            "train_protacs"
        ]
    )

    protac_in_val = (
        reference_protac
        in split_libraries[
            "val_protacs"
        ]
    )

    protac_in_test = (
        reference_protac
        in split_libraries[
            "test_protacs"
        ]
    )

    # --------------------------------------------------------
    # Exact linker presence
    # --------------------------------------------------------

    linker_in_train = (
        reference_linker
        in split_libraries[
            "train_linkers"
        ]
    )

    linker_in_val = (
        reference_linker
        in split_libraries[
            "val_linkers"
        ]
    )

    linker_in_test = (
        reference_linker
        in split_libraries[
            "test_linkers"
        ]
    )

    # --------------------------------------------------------
    # Official cluster assignment
    #
    # Important:
    # reference cases may not be present in the split CSV.
    # In that case reference_cluster is None.
    # --------------------------------------------------------

    reference_cluster = (
        split_libraries[
            "linker_to_cluster"
        ].get(
            reference_linker,
            None,
        )
    )

    train_df = split_libraries[
        "train_df"
    ]

    val_df = split_libraries[
        "val_df"
    ]

    test_df = split_libraries[
        "test_df"
    ]

    # --------------------------------------------------------
    # Exact linker rows in train
    # --------------------------------------------------------

    exact_linker_train_rows = (
        train_df[
            train_df["linker_core_id"]
            .astype(str)
            == str(reference_linker)
        ]
        .copy()
    )

    exact_linker_train_protacs = (
        exact_linker_train_rows[
            "protac_id"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    # --------------------------------------------------------
    # Same cluster rows in train
    #
    # Only possible if reference linker itself is represented
    # in the official split table and has an official cluster ID.
    # --------------------------------------------------------

    if reference_cluster is not None:

        same_cluster_train_rows = (
            train_df[
                train_df["linker_cluster"]
                == reference_cluster
            ]
            .copy()
        )

    else:

        same_cluster_train_rows = (
            train_df.iloc[0:0]
            .copy()
        )

    same_cluster_train_protacs = (
        same_cluster_train_rows[
            "protac_id"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    # --------------------------------------------------------
    # Exact linker rows in val/test
    # --------------------------------------------------------

    exact_linker_val_rows = (
        val_df[
            val_df["linker_core_id"]
            .astype(str)
            == str(reference_linker)
        ]
        .copy()
    )

    exact_linker_test_rows = (
        test_df[
            test_df["linker_core_id"]
            .astype(str)
            == str(reference_linker)
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Max Tanimoto vs ALL unique train linkers
    # --------------------------------------------------------

    (
        max_train_tanimoto,
        closest_train_linker,
    ) = calculate_max_train_tanimoto(
        reference_linker_core=reference_linker,
        train_linkers=split_libraries[
            "train_linkers"
        ],
    )

    # --------------------------------------------------------
    # Count train linkers >= 0.70
    # --------------------------------------------------------

    reference_fp = get_morgan_fp(
        reference_linker
    )

    train_similar_linkers = []

    for linker_smiles in split_libraries[
        "train_linkers"
    ]:

        train_fp = get_morgan_fp(
            linker_smiles
        )

        score = tanimoto_similarity(
            reference_fp,
            train_fp,
        )

        if score >= TANIMOTO_THRESHOLD:
            train_similar_linkers.append(
                {
                    "linker_core_id": linker_smiles,
                    "tanimoto": score,
                }
            )

    train_similar_linkers.sort(
        key=lambda x: x["tanimoto"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Cluster overlap
    # --------------------------------------------------------

    cluster_in_train = (
        reference_cluster is not None
        and len(same_cluster_train_rows) > 0
    )

    cluster_in_val = False
    cluster_in_test = False

    if reference_cluster is not None:

        cluster_in_val = bool(
            (
                val_df["linker_cluster"]
                == reference_cluster
            ).any()
        )

        cluster_in_test = bool(
            (
                test_df["linker_cluster"]
                == reference_cluster
            ).any()
        )

    # --------------------------------------------------------
    # Full hold-out classification
    # --------------------------------------------------------

    completely_held_out = (
        not protac_in_train
        and not linker_in_train
        and not cluster_in_train
    )

    # --------------------------------------------------------
    # More useful classification
    # --------------------------------------------------------

    if completely_held_out:
        holdout_category = (
            "fully held-out"
        )
    elif (
        protac_in_train
        and linker_in_train
    ):
        holdout_category = (
            "seen PROTAC / seen linker"
        )
    elif (
        not protac_in_train
        and linker_in_train
    ):
        holdout_category = (
            "unseen PROTAC / seen linker"
        )
    elif (
        not protac_in_train
        and not linker_in_train
        and cluster_in_train
    ):
        holdout_category = (
            "unseen linker identity / "
            "seen linker cluster"
        )
    else:
        holdout_category = (
            "partial overlap"
        )

    # --------------------------------------------------------
    # Prepare exact train rows
    # --------------------------------------------------------

    detail_columns = [
        c
        for c in [
            "Compound ID",
            "Smiles",
            "linker_smiles",
            "protac_id",
            "linker_core_id",
            "linker_cluster",
            "split",
            "Uniprot",
            "Target",
            "E3 ligase",
        ]
        if c in train_df.columns
    ]

    exact_linker_train_rows_output = (
        exact_linker_train_rows[
            detail_columns
        ]
        .drop_duplicates()
        .to_dict("records")
    )

    same_cluster_train_rows_output = (
        same_cluster_train_rows[
            detail_columns
        ]
        .drop_duplicates()
        .to_dict("records")
    )

    # --------------------------------------------------------
    # Similar train linker list
    # --------------------------------------------------------

    train_similar_linkers_output = (
        train_similar_linkers
    )

    return {
        "case": case_name,
        "pdb": CASE_PDB[case_name],

        "reference_protac_smiles": (
            reference_protac
        ),

        "reference_linker_core_smiles": (
            reference_linker
        ),

        "reference_linker_atoms": (
            linker_atom_count
        ),

        # ----------------------------------------------------
        # Exact identity
        # ----------------------------------------------------

        "reference_protac_in_train": (
            bool(protac_in_train)
        ),

        "reference_protac_in_val": (
            bool(protac_in_val)
        ),

        "reference_protac_in_test": (
            bool(protac_in_test)
        ),

        "reference_linker_in_train": (
            bool(linker_in_train)
        ),

        "reference_linker_in_val": (
            bool(linker_in_val)
        ),

        "reference_linker_in_test": (
            bool(linker_in_test)
        ),

        # ----------------------------------------------------
        # Official cluster
        # ----------------------------------------------------

        "reference_linker_cluster": (
            reference_cluster
        ),

        "reference_linker_cluster_in_train": (
            bool(cluster_in_train)
        ),

        "reference_linker_cluster_in_val": (
            bool(cluster_in_val)
        ),

        "reference_linker_cluster_in_test": (
            bool(cluster_in_test)
        ),

        # ----------------------------------------------------
        # Tanimoto
        # ----------------------------------------------------

        "max_tanimoto_vs_train": (
            float(max_train_tanimoto)
        ),

        "closest_training_linker": (
            closest_train_linker
        ),

        "train_linkers_tanimoto_ge_0.70": (
            int(
                len(
                    train_similar_linkers
                )
            )
        ),

        "train_similar_linkers": (
            train_similar_linkers_output
        ),

        # ----------------------------------------------------
        # Exact overlap details
        # ----------------------------------------------------

        "exact_linker_train_protac_count": (
            len(
                exact_linker_train_protacs
            )
        ),

        "exact_linker_train_protacs": (
            exact_linker_train_protacs
        ),

        "exact_linker_train_rows": (
            exact_linker_train_rows_output
        ),

        "same_cluster_train_protac_count": (
            len(
                same_cluster_train_protacs
            )
        ),

        "same_cluster_train_protacs": (
            same_cluster_train_protacs
        ),

        "same_cluster_train_rows": (
            same_cluster_train_rows_output
        ),

        "exact_linker_val_rows": (
            exact_linker_val_rows[
                [
                    c
                    for c in detail_columns
                    if c in exact_linker_val_rows.columns
                ]
            ]
            .drop_duplicates()
            .to_dict("records")
        ),

        "exact_linker_test_rows": (
            exact_linker_test_rows[
                [
                    c
                    for c in detail_columns
                    if c in exact_linker_test_rows.columns
                ]
            ]
            .drop_duplicates()
            .to_dict("records")
        ),

        # ----------------------------------------------------
        # Final classification
        # ----------------------------------------------------

        "holdout_category": (
            holdout_category
        ),

        "completely_held_out": (
            bool(completely_held_out)
        ),
    }


# ============================================================
# Save summary CSV
# ============================================================

def save_csv(
    results: List[Dict[str, Any]],
    path: str,
) -> None:

    fieldnames = [
        "case",
        "pdb",
        "reference_linker_atoms",

        "reference_protac_in_train",
        "reference_protac_in_val",
        "reference_protac_in_test",

        "reference_linker_in_train",
        "reference_linker_in_val",
        "reference_linker_in_test",

        "reference_linker_cluster",
        "reference_linker_cluster_in_train",
        "reference_linker_cluster_in_val",
        "reference_linker_cluster_in_test",

        "max_tanimoto_vs_train",
        "closest_training_linker",
        "train_linkers_tanimoto_ge_0.70",

        "exact_linker_train_protac_count",
        "same_cluster_train_protac_count",

        "holdout_category",
        "completely_held_out",

        "reference_protac_smiles",
        "reference_linker_core_smiles",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                result
            )


# ============================================================
# Save overlap details CSV
# ============================================================

def save_overlap_details(
    results: List[Dict[str, Any]],
    path: str,
) -> None:

    rows = []

    for result in results:

        case_name = result["case"]

        # ----------------------------------------------------
        # Exact linker overlap
        # ----------------------------------------------------

        for row in result[
            "exact_linker_train_rows"
        ]:

            output_row = {
                "case": case_name,
                "pdb": result["pdb"],
                "overlap_type": (
                    "exact_linker"
                ),
            }

            output_row.update(
                row
            )

            rows.append(
                output_row
            )

        # ----------------------------------------------------
        # Same cluster overlap
        # ----------------------------------------------------

        for row in result[
            "same_cluster_train_rows"
        ]:

            output_row = {
                "case": case_name,
                "pdb": result["pdb"],
                "overlap_type": (
                    "same_linker_cluster"
                ),
            }

            output_row.update(
                row
            )

            rows.append(
                output_row
            )

    if not rows:

        # 即使没有行，也写一个空 CSV
        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "case",
                    "pdb",
                    "overlap_type",
                ]
            )

        return

    details_df = pd.DataFrame(
        rows
    )

    details_df = details_df.drop_duplicates()

    details_df.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


# ============================================================
# Save similar-linker details CSV
# ============================================================

def save_similarity_details(
    results: List[Dict[str, Any]],
    path: str,
) -> None:

    rows = []

    for result in results:

        for item in result[
            "train_similar_linkers"
        ]:

            rows.append(
                {
                    "case": result["case"],
                    "pdb": result["pdb"],
                    "reference_linker_core_smiles": (
                        result[
                            "reference_linker_core_smiles"
                        ]
                    ),
                    "training_linker_core_smiles": (
                        item["linker_core_id"]
                    ),
                    "tanimoto": item["tanimoto"],
                    "threshold": TANIMOTO_THRESHOLD,
                }
            )

    if not rows:

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "case",
                    "pdb",
                    "reference_linker_core_smiles",
                    "training_linker_core_smiles",
                    "tanimoto",
                    "threshold",
                ]
            )

        return

    pd.DataFrame(
        rows
    ).sort_values(
        [
            "case",
            "tanimoto",
        ],
        ascending=[
            True,
            False,
        ],
    ).to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


# ============================================================
# Save JSON
# ============================================================

def save_json(
    results: List[Dict[str, Any]],
    path: str,
) -> None:

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Save Markdown
# ============================================================

def save_markdown(
    results: List[Dict[str, Any]],
    path: str,
) -> None:

    lines = [
        "# Case-specific held-out / leakage audit",
        "",
        (
            "Official "
            "`protac_split_by_linker_cluster.csv` "
            "was used directly. No new clustering was performed."
        ),
        "",
        (
            f"Morgan fingerprint: radius={MORGAN_RADIUS}, "
            f"bits={MORGAN_BITS}"
        ),
        (
            f"Linker similarity threshold: "
            f"Tanimoto >= {TANIMOTO_THRESHOLD:.2f}"
        ),
        "",
        (
            "| Case | PROTAC in train | "
            "Linker in train | Cluster in train | "
            "Max Tanimoto vs train | "
            "Exact linker train PROTACs | "
            "Same-cluster train PROTACs | "
            "Hold-out category | Completely held-out |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---|---:|"
        ),
    ]

    for result in results:

        lines.append(
            "| "
            f"{result['case']} | "
            f"{result['reference_protac_in_train']} | "
            f"{result['reference_linker_in_train']} | "
            f"{result['reference_linker_cluster_in_train']} | "
            f"{result['max_tanimoto_vs_train']:.6f} | "
            f"{result['exact_linker_train_protac_count']} | "
            f"{result['same_cluster_train_protac_count']} | "
            f"{result['holdout_category']} | "
            f"{result['completely_held_out']} |"
        )

    lines.append("")

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
        )


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "CASE-SPECIFIC HELD-OUT / LEAKAGE AUDIT"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # 1. Load official split
    # --------------------------------------------------------

    print()
    print(
        "Loading official linker-cluster split..."
    )

    split_df = load_split_table()

    print(
        f"Rows in split table : {len(split_df)}"
    )

    print(
        "Splits               : "
        + ", ".join(
            sorted(
                split_df["split"]
                .dropna()
                .astype(str)
                .unique()
            )
        )
    )

    # --------------------------------------------------------
    # 2. Prepare libraries
    # --------------------------------------------------------

    libraries = prepare_split_libraries(
        split_df
    )

    print()
    print(
        f"Train PROTACs : "
        f"{len(libraries['train_protacs'])}"
    )

    print(
        f"Val PROTACs   : "
        f"{len(libraries['val_protacs'])}"
    )

    print(
        f"Test PROTACs  : "
        f"{len(libraries['test_protacs'])}"
    )

    print(
        f"Train linkers : "
        f"{len(libraries['train_linkers'])}"
    )

    print(
        f"Val linkers   : "
        f"{len(libraries['val_linkers'])}"
    )

    print(
        f"Test linkers  : "
        f"{len(libraries['test_linkers'])}"
    )

    # --------------------------------------------------------
    # 3. Audit each retrospective case
    # --------------------------------------------------------

    results = []

    for case_name in [
        "MZ1",
        "A1ANN",
        "NEP108",
    ]:

        result = audit_case(
            case_name=case_name,
            split_libraries=libraries,
        )

        results.append(
            result
        )

        print()
        print("-" * 100)
        print(
            f"{case_name} ({result['pdb']})"
        )
        print("-" * 100)

        print(
            "Reference linker atoms          : "
            f"{result['reference_linker_atoms']}"
        )

        print(
            "Reference linker cluster        : "
            f"{result['reference_linker_cluster']}"
        )

        print(
            "Reference PROTAC in train       : "
            f"{result['reference_protac_in_train']}"
        )

        print(
            "Reference PROTAC in val         : "
            f"{result['reference_protac_in_val']}"
        )

        print(
            "Reference PROTAC in test        : "
            f"{result['reference_protac_in_test']}"
        )

        print(
            "Reference linker in train       : "
            f"{result['reference_linker_in_train']}"
        )

        print(
            "Reference linker in val         : "
            f"{result['reference_linker_in_val']}"
        )

        print(
            "Reference linker in test        : "
            f"{result['reference_linker_in_test']}"
        )

        print(
            "Reference cluster in train      : "
            f"{result['reference_linker_cluster_in_train']}"
        )

        print(
            "Reference cluster in val        : "
            f"{result['reference_linker_cluster_in_val']}"
        )

        print(
            "Reference cluster in test       : "
            f"{result['reference_linker_cluster_in_test']}"
        )

        print(
            "Max Tanimoto vs train           : "
            f"{result['max_tanimoto_vs_train']:.6f}"
        )

        print(
            "Closest training linker         : "
            f"{result['closest_training_linker']}"
        )

        print(
            "Train linkers with T >= 0.70    : "
            f"{result['train_linkers_tanimoto_ge_0.70']}"
        )

        print(
            "Exact-linker train PROTACs      : "
            f"{result['exact_linker_train_protac_count']}"
        )

        for protac in result[
            "exact_linker_train_protacs"
        ]:

            print(
                f"    EXACT LINKER -> {protac}"
            )

        print(
            "Same-cluster train PROTACs      : "
            f"{result['same_cluster_train_protac_count']}"
        )

        for protac in result[
            "same_cluster_train_protacs"
        ][:20]:

            print(
                f"    SAME CLUSTER -> {protac}"
            )

        if (
            result[
                "same_cluster_train_protac_count"
            ] > 20
        ):

            print(
                "    ... "
                f"{result['same_cluster_train_protac_count'] - 20} "
                "more in JSON/CSV"
            )

        print(
            "Hold-out category               : "
            f"{result['holdout_category']}"
        )

        print(
            "Completely held-out             : "
            f"{result['completely_held_out']}"
        )

    # --------------------------------------------------------
    # 4. Save
    # --------------------------------------------------------

    summary_csv_path = os.path.join(
        OUTPUT_DIR,
        "case_train_leakage_audit.csv",
    )

    overlap_csv_path = os.path.join(
        OUTPUT_DIR,
        "case_train_overlap_details.csv",
    )

    similarity_csv_path = os.path.join(
        OUTPUT_DIR,
        "case_train_similarity_details.csv",
    )

    json_path = os.path.join(
        OUTPUT_DIR,
        "case_train_leakage_audit.json",
    )

    md_path = os.path.join(
        OUTPUT_DIR,
        "case_train_leakage_audit.md",
    )

    save_csv(
        results,
        summary_csv_path,
    )

    save_overlap_details(
        results,
        overlap_csv_path,
    )

    save_similarity_details(
        results,
        similarity_csv_path,
    )

    save_json(
        results,
        json_path,
    )

    save_markdown(
        results,
        md_path,
    )

    # --------------------------------------------------------
    # 5. Final summary
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 100)

    print(
        "Case        PROTAC_train  Linker_train  "
        "Cluster_train  Tmax_train  Category"
    )

    print(
        "-" * 100
    )

    for result in results:

        print(
            f"{result['case']:<10}"
            f"{str(result['reference_protac_in_train']):<14}"
            f"{str(result['reference_linker_in_train']):<14}"
            f"{str(result['reference_linker_cluster_in_train']):<15}"
            f"{result['max_tanimoto_vs_train']:<12.4f}"
            f"{result['holdout_category']}"
        )

    print()
    print(
        f"Summary CSV      : {summary_csv_path}"
    )

    print(
        f"Overlap details  : {overlap_csv_path}"
    )

    print(
        f"Similarity detail: {similarity_csv_path}"
    )

    print(
        f"JSON             : {json_path}"
    )

    print(
        f"Markdown         : {md_path}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()