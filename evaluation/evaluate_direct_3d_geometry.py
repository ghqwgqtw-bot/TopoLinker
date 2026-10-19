import csv
import json
import math
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolTransforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "DiffLinker": (
        PROJECT_ROOT / "eval_outputs" / "difflinker_seeds"
        / "difflinker_test_obabel2026" / "records.pkl"
    ),
    "DiffPROTACs": (
        PROJECT_ROOT / "eval_outputs" / "diffprotacs_seeds"
        / "diffprotacs_test_2026" / "records.pkl"
    ),
    "LinkerNet": (
        PROJECT_ROOT / "eval_outputs" / "linkernet_seeds"
        / "linkernet_test_2026" / "records.pkl"
    ),
    "DesignMaster": (
        PROJECT_ROOT / "eval_outputs" / "designmaster_seeds"
        / "designmaster_test_2026" / "records.pkl"
    ),
    "PocketXMol": (
        PROJECT_ROOT / "eval_outputs" / "pocketxmol_seeds"
        / "pocketxmol_outputs_test2026" / "records.pkl"
    ),
    "TopoLinker": (
        PROJECT_ROOT / "eval_outputs" / "test_seeds"
        / "seed2026" / "records.pkl"
    ),
}

PAPER_TABLES_DIR = (
    PROJECT_ROOT
    / "paper_tables"
    / "direct_3d_geometry"
)

PAPER_TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV = (
    PAPER_TABLES_DIR
    / "direct_3d_geometry_full_summary.csv"
)

MAIN_TABLE_CSV = (
    PAPER_TABLES_DIR
    / "direct_3d_geometry_main_table.csv"
)

MAIN_TABLE_TSV = (
    PAPER_TABLES_DIR
    / "direct_3d_geometry_main_table.tsv"
)

MAIN_TABLE_MD = (
    PAPER_TABLES_DIR
    / "direct_3d_geometry_main_table.md"
)

SAMPLE_CSV = (
    PAPER_TABLES_DIR
    / "direct_3d_geometry_per_sample.csv"
)

PLOT_DATA_CSV = (
    PAPER_TABLES_DIR
    / "direct_3d_geometry_plot_data.csv"
)

SUMMARY_JSON = (
    PAPER_TABLES_DIR
    / "direct_3d_geometry_summary.json"
)


SEVERE_OVERLAP_THRESHOLD_A = 0.40
MAX_EXCLUDED_GRAPH_DISTANCE = 3

ANGLE_MIN_DEG = 0.0
ANGLE_MAX_DEG = 180.0
ANGLE_BIN_WIDTH_DEG = 5.0

BOND_MIN_A = 0.0
BOND_MAX_A = 3.0
BOND_BIN_WIDTH_A = 0.05

TORSION_N_BINS = 18

TORSION_EDGES = np.linspace(
    -180.0,
    180.0,
    TORSION_N_BINS + 1,
)


def mask_to_bool(mask):
    if mask is None:
        return None

    if torch.is_tensor(mask):
        mask = (
            mask
            .detach()
            .cpu()
            .numpy()
        )

    mask = np.asarray(mask)
    mask = np.squeeze(mask)

    if mask.ndim != 1:
        raise ValueError(
            f"Expected 1D mask after squeeze, "
            f"got shape={mask.shape}"
        )

    return mask.astype(bool)


def get_effective_mask(
    record,
    key,
):
    atom_mask = mask_to_bool(
        record["atom_mask"]
    )

    target_mask = mask_to_bool(
        record[key]
    )

    if atom_mask is None:
        raise ValueError(
            "atom_mask is None"
        )

    if target_mask is None:
        raise ValueError(
            f"{key} is None"
        )

    if len(atom_mask) != len(target_mask):
        raise ValueError(
            f"{key} length != atom_mask length: "
            f"{len(target_mask)} vs {len(atom_mask)}"
        )

    return target_mask[
        atom_mask
    ]


def get_coords(mol):
    if mol is None:
        raise ValueError(
            "Molecule is None"
        )

    if mol.GetNumConformers() == 0:
        raise ValueError(
            "Molecule has no conformer"
        )

    conf = mol.GetConformer(0)

    coords = np.asarray(
        conf.GetPositions(),
        dtype=np.float64,
    )

    expected_shape = (
        mol.GetNumAtoms(),
        3,
    )

    if coords.shape != expected_shape:
        raise ValueError(
            f"Coordinate shape mismatch: "
            f"{coords.shape} vs {expected_shape}"
        )

    if not np.isfinite(
        coords
    ).all():
        raise ValueError(
            "Coordinates contain NaN/Inf"
        )

    return coords


def summarize(
    values,
    suffix,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "n": 0,
            f"mean_{suffix}": None,
            f"sd_{suffix}": None,
            f"median_{suffix}": None,
            f"q1_{suffix}": None,
            f"q3_{suffix}": None,
            f"iqr_{suffix}": None,
            f"min_{suffix}": None,
            f"max_{suffix}": None,
        }

    q1 = float(
        np.percentile(
            values,
            25,
        )
    )

    q3 = float(
        np.percentile(
            values,
            75,
        )
    )

    return {
        "n": int(
            len(values)
        ),

        f"mean_{suffix}": float(
            values.mean()
        ),

        f"sd_{suffix}": (
            float(
                values.std(
                    ddof=1
                )
            )
            if len(values) > 1
            else 0.0
        ),

        f"median_{suffix}": float(
            np.median(values)
        ),

        f"q1_{suffix}": q1,

        f"q3_{suffix}": q3,

        f"iqr_{suffix}": float(
            q3 - q1
        ),

        f"min_{suffix}": float(
            values.min()
        ),

        f"max_{suffix}": float(
            values.max()
        ),
    }


def get_region_sets(record):
    linker_mask = get_effective_mask(
        record,
        "linker_mask",
    )

    anchor1_mask = get_effective_mask(
        record,
        "anchor1_mask",
    )

    anchor2_mask = get_effective_mask(
        record,
        "anchor2_mask",
    )

    linker = set(
        np.flatnonzero(
            linker_mask
        ).tolist()
    )

    anchor1 = set(
        np.flatnonzero(
            anchor1_mask
        ).tolist()
    )

    anchor2 = set(
        np.flatnonzero(
            anchor2_mask
        ).tolist()
    )

    return (
        linker,
        anchor1,
        anchor2,
    )


def classify_linker_related_bond(
    i,
    j,
    linker,
    anchor1,
    anchor2,
):
    i_linker = (
        i in linker
    )

    j_linker = (
        j in linker
    )

    if (
        i_linker
        and
        j_linker
    ):
        return "linker-linker"

    if (
        (
            i in anchor1
            and
            j_linker
        )
        or
        (
            j in anchor1
            and
            i_linker
        )
    ):
        return "anchor1-linker"

    if (
        (
            i in anchor2
            and
            j_linker
        )
        or
        (
            j in anchor2
            and
            i_linker
        )
    ):
        return "anchor2-linker"

    return None


def kabsch_transform(
    P,
    Q,
):
    if P.shape != Q.shape:
        raise ValueError(
            f"Kabsch shape mismatch: "
            f"{P.shape} vs {Q.shape}"
        )

    if (
        P.ndim != 2
        or
        P.shape[1] != 3
    ):
        raise ValueError(
            f"Invalid coordinate shape: "
            f"{P.shape}"
        )

    P_centroid = P.mean(
        axis=0
    )

    Q_centroid = Q.mean(
        axis=0
    )

    P_centered = (
        P
        - P_centroid
    )

    Q_centered = (
        Q
        - Q_centroid
    )

    H = (
        P_centered.T
        @
        Q_centered
    )

    U, _, Vt = np.linalg.svd(
        H
    )

    R = (
        U
        @
        Vt
    )

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0

        R = (
            U
            @
            Vt
        )

    return (
        R,
        P_centroid,
        Q_centroid,
    )


def apply_transform(
    coords,
    R,
    P_centroid,
    Q_centroid,
):
    return (
        (
            coords
            - P_centroid
        )
        @
        R
        +
        Q_centroid
    )


def rmsd(
    coords_a,
    coords_b,
):
    if coords_a.shape != coords_b.shape:
        raise ValueError(
            f"RMSD shape mismatch: "
            f"{coords_a.shape} vs {coords_b.shape}"
        )

    diff = (
        coords_a
        - coords_b
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    diff * diff,
                    axis=1,
                )
            )
        )
    )


def compute_linker_rmsd(record):
    pred_mol = record[
        "pred_mol"
    ]

    true_mol = record[
        "true_mol"
    ]

    pred_n = (
        pred_mol
        .GetNumAtoms()
    )

    true_n = (
        true_mol
        .GetNumAtoms()
    )

    if pred_n != true_n:
        raise ValueError(
            f"Atom count mismatch: "
            f"pred={pred_n}, true={true_n}"
        )

    pred_coords = get_coords(
        pred_mol
    )

    true_coords = get_coords(
        true_mol
    )

    (
        fragment1_mask,
        fragment2_mask,
        linker_mask,
        anchor1_mask,
        anchor2_mask,
    ) = (
        get_effective_mask(
            record,
            "fragment1_mask",
        ),
        get_effective_mask(
            record,
            "fragment2_mask",
        ),
        get_effective_mask(
            record,
            "linker_mask",
        ),
        get_effective_mask(
            record,
            "anchor1_mask",
        ),
        get_effective_mask(
            record,
            "anchor2_mask",
        ),
    )

    fragment_mask = (
        fragment1_mask
        |
        fragment2_mask
    )

    fragment_indices = np.flatnonzero(
        fragment_mask
    )

    linker_indices = np.flatnonzero(
        linker_mask
    )

    anchor1_indices = np.flatnonzero(
        anchor1_mask
    )

    anchor2_indices = np.flatnonzero(
        anchor2_mask
    )

    if len(fragment_indices) < 3:
        raise ValueError(
            "Too few fragment atoms for Kabsch alignment"
        )

    if len(linker_indices) == 0:
        raise ValueError(
            "No linker atoms found"
        )

    if len(anchor1_indices) != 1:
        raise ValueError(
            f"Expected exactly one anchor1, "
            f"got {len(anchor1_indices)}"
        )

    if len(anchor2_indices) != 1:
        raise ValueError(
            f"Expected exactly one anchor2, "
            f"got {len(anchor2_indices)}"
        )

    pred_fragment = pred_coords[
        fragment_indices
    ]

    true_fragment = true_coords[
        fragment_indices
    ]

    (
        R,
        P_centroid,
        Q_centroid,
    ) = kabsch_transform(
        pred_fragment,
        true_fragment,
    )

    pred_aligned = apply_transform(
        pred_coords,
        R,
        P_centroid,
        Q_centroid,
    )

    fragment_rmsd = rmsd(
        pred_aligned[
            fragment_indices
        ],
        true_coords[
            fragment_indices
        ],
    )

    if fragment_rmsd > 1e-3:
        raise ValueError(
            "Unexpectedly large fragment RMSD "
            "after alignment: "
            f"{fragment_rmsd:.6e} A"
        )

    linker_rmsd = rmsd(
        pred_aligned[
            linker_indices
        ],
        true_coords[
            linker_indices
        ],
    )

    return {
        "linker_rmsd_A":
            linker_rmsd,

        "fragment_rmsd_after_alignment_A":
            fragment_rmsd,

        "linker_heavy_atoms":
            int(
                len(
                    linker_indices
                )
            ),

        "fragment_atoms":
            int(
                len(
                    fragment_indices
                )
            ),

        "anchor1_index":
            int(
                anchor1_indices[0]
            ),

        "anchor2_index":
            int(
                anchor2_indices[0]
            ),
    }


def extract_bond_lengths(
    mol,
    coords,
    linker,
    anchor1,
    anchor2,
):
    lengths = {
        "linker-linker": [],
        "anchor1-linker": [],
        "anchor2-linker": [],
    }

    for bond in mol.GetBonds():
        i = int(
            bond.GetBeginAtomIdx()
        )

        j = int(
            bond.GetEndAtomIdx()
        )

        category = (
            classify_linker_related_bond(
                i,
                j,
                linker,
                anchor1,
                anchor2,
            )
        )

        if category is None:
            continue

        d = float(
            np.linalg.norm(
                coords[i]
                -
                coords[j]
            )
        )

        if np.isfinite(d):
            lengths[
                category
            ].append(
                d
            )

    return lengths


def bond_angle_deg(
    coords,
    i,
    j,
    k,
):
    v1 = (
        coords[int(i)]
        -
        coords[int(j)]
    )

    v2 = (
        coords[int(k)]
        -
        coords[int(j)]
    )

    norm1 = np.linalg.norm(
        v1
    )

    norm2 = np.linalg.norm(
        v2
    )

    if (
        norm1 <= 1e-12
        or
        norm2 <= 1e-12
    ):
        raise ValueError(
            "Degenerate angle geometry"
        )

    cos_theta = (
        np.dot(
            v1,
            v2,
        )
        /
        (
            norm1
            *
            norm2
        )
    )

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                cos_theta
            )
        )
    )


def classify_angle(
    i,
    j,
    k,
    linker,
    anchor1,
    anchor2,
):
    i_linker = (
        i in linker
    )

    j_linker = (
        j in linker
    )

    k_linker = (
        k in linker
    )

    if (
        i_linker
        and
        j_linker
        and
        k_linker
    ):
        return (
            "linker-linker-linker"
        )

    if j_linker:
        if (
            (
                i in anchor1
                and
                k_linker
            )
            or
            (
                k in anchor1
                and
                i_linker
            )
        ):
            return (
                "anchor1-linker-linker"
            )

        if (
            (
                i in anchor2
                and
                k_linker
            )
            or
            (
                k in anchor2
                and
                i_linker
            )
        ):
            return (
                "anchor2-linker-linker"
            )

    return None


def extract_bond_angles(
    mol,
    coords,
    linker,
    anchor1,
    anchor2,
):
    angles = {
        "linker-linker-linker": [],
        "anchor1-linker-linker": [],
        "anchor2-linker-linker": [],
    }

    for j in sorted(
        linker
    ):
        atom_j = (
            mol.GetAtomWithIdx(
                int(j)
            )
        )

        neighbors = [
            int(
                nbr.GetIdx()
            )
            for nbr
            in atom_j.GetNeighbors()
            if nbr.GetAtomicNum() > 1
        ]

        if len(neighbors) < 2:
            continue

        for a in range(
            len(neighbors)
        ):
            i = neighbors[a]

            for b in range(
                a + 1,
                len(neighbors),
            ):
                k = neighbors[b]

                category = classify_angle(
                    i,
                    j,
                    k,
                    linker,
                    anchor1,
                    anchor2,
                )

                if category is None:
                    continue

                angle = bond_angle_deg(
                    coords,
                    i,
                    j,
                    k,
                )

                if np.isfinite(angle):
                    angles[
                        category
                    ].append(
                        angle
                    )

    return angles


def torsion_angle_deg(
    coords,
    i,
    j,
    k,
    l,
):
    p0 = coords[
        int(i)
    ]

    p1 = coords[
        int(j)
    ]

    p2 = coords[
        int(k)
    ]

    p3 = coords[
        int(l)
    ]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(
        b1
    )

    if b1_norm <= 1e-12:
        raise ValueError(
            "Degenerate central bond"
        )

    b1_unit = (
        b1
        /
        b1_norm
    )

    v = (
        b0
        -
        np.dot(
            b0,
            b1_unit,
        )
        *
        b1_unit
    )

    w = (
        b2
        -
        np.dot(
            b2,
            b1_unit,
        )
        *
        b1_unit
    )

    v_norm = np.linalg.norm(
        v
    )

    w_norm = np.linalg.norm(
        w
    )

    if (
        v_norm <= 1e-12
        or
        w_norm <= 1e-12
    ):
        raise ValueError(
            "Degenerate torsion geometry"
        )

    v_unit = (
        v
        /
        v_norm
    )

    w_unit = (
        w
        /
        w_norm
    )

    x = np.dot(
        v_unit,
        w_unit,
    )

    y = np.dot(
        np.cross(
            b1_unit,
            v_unit,
        ),
        w_unit,
    )

    angle = np.degrees(
        np.arctan2(
            y,
            x,
        )
    )

    return float(
        angle
    )


def classify_torsion_category(
    j,
    k,
    linker,
    anchor1,
    anchor2,
):
    if (
        j in linker
        and
        k in linker
    ):
        return (
            "linker-linker"
        )

    if (
        (
            j in anchor1
            and
            k in linker
        )
        or
        (
            k in anchor1
            and
            j in linker
        )
    ):
        return (
            "anchor1-linker"
        )

    if (
        (
            j in anchor2
            and
            k in linker
        )
        or
        (
            k in anchor2
            and
            j in linker
        )
    ):
        return (
            "anchor2-linker"
        )

    return None


def get_linker_associated_torsions(
    mol,
    linker_mask,
    anchor1_mask,
    anchor2_mask,
):
    linker = set(
        np.flatnonzero(
            linker_mask
        ).tolist()
    )

    anchor1 = set(
        np.flatnonzero(
            anchor1_mask
        ).tolist()
    )

    anchor2 = set(
        np.flatnonzero(
            anchor2_mask
        ).tolist()
    )

    anchors = (
        anchor1
        |
        anchor2
    )

    allowed = (
        linker
        |
        anchors
    )

    torsions = []

    for bond in mol.GetBonds():
        j = bond.GetBeginAtomIdx()
        k = bond.GetEndAtomIdx()

        if (
            j not in allowed
            or
            k not in allowed
        ):
            continue

        if (
            j not in linker
            and
            k not in linker
        ):
            continue

        if (
            bond.GetBondType()
            !=
            Chem.BondType.SINGLE
        ):
            continue

        if bond.IsInRing():
            continue

        atom_j = (
            mol.GetAtomWithIdx(
                int(j)
            )
        )

        atom_k = (
            mol.GetAtomWithIdx(
                int(k)
            )
        )

        j_neighbors = []

        for nbr in atom_j.GetNeighbors():
            i = nbr.GetIdx()

            if i == k:
                continue

            if i not in allowed:
                continue

            if nbr.GetAtomicNum() <= 1:
                continue

            j_neighbors.append(
                int(i)
            )

        k_neighbors = []

        for nbr in atom_k.GetNeighbors():
            l = nbr.GetIdx()

            if l == j:
                continue

            if l not in allowed:
                continue

            if nbr.GetAtomicNum() <= 1:
                continue

            k_neighbors.append(
                int(l)
            )

        if len(j_neighbors) == 0:
            continue

        if len(k_neighbors) == 0:
            continue

        category = (
            classify_torsion_category(
                j,
                k,
                linker,
                anchor1,
                anchor2,
            )
        )

        if category is None:
            continue

        for i in j_neighbors:
            for l in k_neighbors:
                if len(
                    {
                        i,
                        j,
                        k,
                        l,
                    }
                ) < 4:
                    continue

                torsion = (
                    int(i),
                    int(j),
                    int(k),
                    int(l),
                )

                torsions.append(
                    {
                        "quartet":
                            torsion,

                        "category":
                            category,
                    }
                )

    return torsions


def extract_torsions(
    mol,
    linker_mask,
    anchor1_mask,
    anchor2_mask,
):
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "Molecule has no conformer"
        )

    coords = get_coords(
        mol
    )

    torsion_records = (
        get_linker_associated_torsions(
            mol=mol,
            linker_mask=linker_mask,
            anchor1_mask=anchor1_mask,
            anchor2_mask=anchor2_mask,
        )
    )

    torsions = {
        "linker-linker": [],
        "anchor1-linker": [],
        "anchor2-linker": [],
    }

    for item in torsion_records:
        (
            i,
            j,
            k,
            l,
        ) = item[
            "quartet"
        ]

        category = item[
            "category"
        ]

        try:
            angle = torsion_angle_deg(
                coords,
                i,
                j,
                k,
                l,
            )

        except Exception:
            continue

        if not np.isfinite(
            angle
        ):
            continue

        angle = (
            (
                angle
                +
                180.0
            )
            %
            360.0
        ) - 180.0

        torsions[
            category
        ].append(
            angle
        )

    return torsions


PERIODIC_TABLE = (
    Chem.GetPeriodicTable()
)


def get_heavy_atom_indices(mol):
    return [
        int(
            atom.GetIdx()
        )
        for atom
        in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]


def shortest_path_distances_limited(
    mol,
    source,
    max_distance=MAX_EXCLUDED_GRAPH_DISTANCE,
):
    source = int(
        source
    )

    distances = {
        source: 0
    }

    queue = deque([
        source
    ])

    while queue:
        current = (
            queue.popleft()
        )

        current_distance = (
            distances[
                current
            ]
        )

        if (
            current_distance
            >=
            max_distance
        ):
            continue

        atom = (
            mol.GetAtomWithIdx(
                int(current)
            )
        )

        for nbr in atom.GetNeighbors():
            nxt = int(
                nbr.GetIdx()
            )

            if nxt in distances:
                continue

            distances[nxt] = (
                current_distance
                + 1
            )

            queue.append(
                nxt
            )

    return distances


def get_eligible_heavy_atom_pairs(
    mol,
    heavy_atom_indices,
):
    distance_cache = {}

    for i in heavy_atom_indices:
        distance_cache[i] = (
            shortest_path_distances_limited(
                mol,
                i,
                MAX_EXCLUDED_GRAPH_DISTANCE,
            )
        )

    pairs = []

    for a in range(
        len(
            heavy_atom_indices
        )
    ):
        i = (
            heavy_atom_indices[a]
        )

        for b in range(
            a + 1,
            len(
                heavy_atom_indices
            ),
        ):
            j = (
                heavy_atom_indices[b]
            )

            graph_distance = (
                distance_cache[i]
                .get(
                    j,
                    None,
                )
            )

            if graph_distance is not None:
                if graph_distance <= 3:
                    continue

            pairs.append(
                (
                    i,
                    j,
                )
            )

    return pairs


def get_vdw_radius(atom):
    radius = float(
        PERIODIC_TABLE.GetRvdw(
            int(
                atom.GetAtomicNum()
            )
        )
    )

    if (
        not np.isfinite(radius)
        or
        radius <= 0.0
    ):
        return None

    return radius


def calculate_clash_statistics(
    mol,
    coords,
):
    heavy_atoms = (
        get_heavy_atom_indices(
            mol
        )
    )

    if len(heavy_atoms) < 2:
        raise ValueError(
            "Too few heavy atoms"
        )

    pairs = (
        get_eligible_heavy_atom_pairs(
            mol,
            heavy_atoms,
        )
    )

    if len(pairs) == 0:
        return None

    eligible_pairs = 0
    skipped_pairs = 0

    positive_overlap_pairs = 0
    severe_clashes = 0

    auxiliary_counts = {
        0.20: 0,
        0.30: 0,
    }

    positive_overlap_values = []

    for i, j in pairs:
        atom_i = (
            mol.GetAtomWithIdx(
                int(i)
            )
        )

        atom_j = (
            mol.GetAtomWithIdx(
                int(j)
            )
        )

        r_i = get_vdw_radius(
            atom_i
        )

        r_j = get_vdw_radius(
            atom_j
        )

        if (
            r_i is None
            or
            r_j is None
        ):
            skipped_pairs += 1
            continue

        distance = float(
            np.linalg.norm(
                coords[i]
                -
                coords[j]
            )
        )

        if not np.isfinite(
            distance
        ):
            skipped_pairs += 1
            continue

        eligible_pairs += 1

        overlap = (
            r_i
            +
            r_j
            -
            distance
        )

        if overlap > 0.0:
            positive_overlap_pairs += 1

            positive_overlap_values.append(
                float(
                    overlap
                )
            )

        for threshold in (
            0.20,
            0.30,
        ):
            if overlap >= threshold:
                auxiliary_counts[
                    threshold
                ] += 1

        if (
            overlap
            >=
            SEVERE_OVERLAP_THRESHOLD_A
        ):
            severe_clashes += 1

    if eligible_pairs == 0:
        return None

    return {
        "eligible_pairs":
            int(
                eligible_pairs
            ),

        "skipped_pairs":
            int(
                skipped_pairs
            ),

        "positive_overlap_pairs":
            int(
                positive_overlap_pairs
            ),

        "severe_clashes":
            int(
                severe_clashes
            ),

        "auxiliary_counts":
            auxiliary_counts,

        "positive_overlap_values":
            positive_overlap_values,
    }


def histogram_edges(
    minimum,
    maximum,
    width,
):
    return np.arange(
        minimum,
        maximum + width,
        width,
    )


BOND_EDGES = histogram_edges(
    BOND_MIN_A,
    BOND_MAX_A,
    BOND_BIN_WIDTH_A,
)

ANGLE_EDGES = histogram_edges(
    ANGLE_MIN_DEG,
    ANGLE_MAX_DEG,
    ANGLE_BIN_WIDTH_DEG,
)


def histogram_rows(
    values,
    edges,
    unit,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    hist, edge_values = np.histogram(
        values,
        bins=edges,
    )

    total = int(
        hist.sum()
    )

    rows = []

    for i in range(
        len(hist)
    ):
        count = int(
            hist[i]
        )

        rows.append(
            {
                "bin_left":
                    float(
                        edge_values[i]
                    ),

                "bin_right":
                    float(
                        edge_values[
                            i + 1
                        ]
                    ),

                "count":
                    count,

                "percent":
                    (
                        100.0
                        *
                        count
                        /
                        total
                        if total > 0
                        else 0.0
                    ),

                "unit":
                    unit,
            }
        )

    return rows


def evaluate_model(
    model_name,
    pkl_path,
):
    print()
    print(
        "=" * 80
    )
    print(
        f"Loading {model_name}"
    )
    print(
        "=" * 80
    )
    print(
        pkl_path
    )

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"PKL file not found:\n"
            f"{pkl_path}"
        )

    with open(
        pkl_path,
        "rb",
    ) as f:
        records = pickle.load(
            f
        )

    if not isinstance(
        records,
        (
            list,
            tuple,
        ),
    ):
        raise TypeError(
            f"Expected list/tuple records, "
            f"got {type(records)}"
        )

    total_records = len(
        records
    )

    strict_valid_records = 0

    geometry_success = 0
    geometry_failed = 0

    failures = []

    linker_rmsds = []

    bond_lengths = []

    bond_lengths_by_category = {
        "linker-linker": [],
        "anchor1-linker": [],
        "anchor2-linker": [],
    }

    bond_angles = []

    bond_angles_by_category = {
        "linker-linker-linker": [],
        "anchor1-linker-linker": [],
        "anchor2-linker-linker": [],
    }

    torsions = []

    torsions_by_category = {
        "linker-linker": [],
        "anchor1-linker": [],
        "anchor2-linker": [],
    }

    total_eligible_pairs = 0
    total_skipped_pairs = 0
    total_positive_overlap_pairs = 0
    total_severe_clashes = 0

    molecules_with_positive_overlap = 0
    molecules_with_severe_clash = 0

    auxiliary_pair_counts = {
        0.20: 0,
        0.30: 0,
    }

    auxiliary_molecule_counts = {
        0.20: 0,
        0.30: 0,
    }

    max_overlap_values = []

    positive_overlap_values = []

    sample_rows = []

    for sample_idx, record in enumerate(
        records
    ):
        if not bool(
            record.get(
                "precomputed_strict",
                False,
            )
        ):
            continue

        strict_valid_records += 1

        try:
            pred_mol = record[
                "pred_mol"
            ]

            coords = get_coords(
                pred_mol
            )

            (
                linker,
                anchor1,
                anchor2,
            ) = get_region_sets(
                record
            )

            if len(linker) == 0:
                raise ValueError(
                    "No linker atoms found"
                )

            if len(anchor1) != 1:
                raise ValueError(
                    f"Expected exactly one anchor1, "
                    f"got {len(anchor1)}"
                )

            if len(anchor2) != 1:
                raise ValueError(
                    f"Expected exactly one anchor2, "
                    f"got {len(anchor2)}"
                )

            rmsd_result = (
                compute_linker_rmsd(
                    record
                )
            )

            linker_rmsds.append(
                rmsd_result[
                    "linker_rmsd_A"
                ]
            )

            lengths = (
                extract_bond_lengths(
                    pred_mol,
                    coords,
                    linker,
                    anchor1,
                    anchor2,
                )
            )

            sample_bond_lengths = []

            for category in lengths:
                values = (
                    lengths[
                        category
                    ]
                )

                bond_lengths_by_category[
                    category
                ].extend(
                    values
                )

                sample_bond_lengths.extend(
                    values
                )

            if len(
                sample_bond_lengths
            ) == 0:
                raise ValueError(
                    "Generated molecule contains no "
                    "linker-related bonds"
                )

            bond_lengths.extend(
                sample_bond_lengths
            )

            angles = (
                extract_bond_angles(
                    pred_mol,
                    coords,
                    linker,
                    anchor1,
                    anchor2,
                )
            )

            sample_angles = []

            for category in angles:
                values = (
                    angles[
                        category
                    ]
                )

                bond_angles_by_category[
                    category
                ].extend(
                    values
                )

                sample_angles.extend(
                    values
                )

            if len(
                sample_angles
            ) == 0:
                raise ValueError(
                    "Generated molecule contains no "
                    "linker-associated bond angles"
                )

            bond_angles.extend(
                sample_angles
            )

            linker_mask_eff = (
                get_effective_mask(
                    record,
                    "linker_mask",
                )
            )

            anchor1_mask_eff = (
                get_effective_mask(
                    record,
                    "anchor1_mask",
                )
            )

            anchor2_mask_eff = (
                get_effective_mask(
                    record,
                    "anchor2_mask",
                )
            )

            torsion_values = (
                extract_torsions(
                    mol=pred_mol,
                    linker_mask=linker_mask_eff,
                    anchor1_mask=anchor1_mask_eff,
                    anchor2_mask=anchor2_mask_eff,
                )
            )

            sample_torsions = []

            for category in (
                "linker-linker",
                "anchor1-linker",
                "anchor2-linker",
            ):
                values = (
                    torsion_values[
                        category
                    ]
                )

                torsions_by_category[
                    category
                ].extend(
                    values
                )

                sample_torsions.extend(
                    values
                )

            torsions.extend(
                sample_torsions
            )

            clash = (
                calculate_clash_statistics(
                    pred_mol,
                    coords,
                )
            )

            if clash is None:
                raise ValueError(
                    "No eligible heavy-atom pairs "
                    "for clash evaluation"
                )

            eligible_pairs = (
                clash[
                    "eligible_pairs"
                ]
            )

            positive_pairs = (
                clash[
                    "positive_overlap_pairs"
                ]
            )

            severe_pairs = (
                clash[
                    "severe_clashes"
                ]
            )

            total_eligible_pairs += (
                eligible_pairs
            )

            total_skipped_pairs += (
                clash[
                    "skipped_pairs"
                ]
            )

            total_positive_overlap_pairs += (
                positive_pairs
            )

            total_severe_clashes += (
                severe_pairs
            )

            if positive_pairs > 0:
                molecules_with_positive_overlap += 1

            if severe_pairs > 0:
                molecules_with_severe_clash += 1

            sample_aux = {}

            for threshold in (
                0.20,
                0.30,
            ):
                count = (
                    clash[
                        "auxiliary_counts"
                    ][
                        threshold
                    ]
                )

                auxiliary_pair_counts[
                    threshold
                ] += count

                sample_aux[
                    threshold
                ] = count

                if count > 0:
                    auxiliary_molecule_counts[
                        threshold
                    ] += 1

            ov = (
                clash[
                    "positive_overlap_values"
                ]
            )

            if ov:
                positive_overlap_values.extend(
                    ov
                )

                max_overlap_values.append(
                    max(ov)
                )

            else:
                max_overlap_values.append(
                    0.0
                )

            sample_rows.append(
                {
                    "model":
                        model_name,

                    "sample_index":
                        sample_idx,

                    "sample_id":
                        record.get(
                            "sample_id",
                            sample_idx,
                        ),

                    "uuid":
                        record.get(
                            "uuid",
                            "",
                        ),

                    "strict_valid":
                        True,

                    "linker_rmsd_A":
                        rmsd_result[
                            "linker_rmsd_A"
                        ],

                    "fragment_rmsd_after_alignment_A":
                        rmsd_result[
                            "fragment_rmsd_after_alignment_A"
                        ],

                    "linker_heavy_atoms":
                        rmsd_result[
                            "linker_heavy_atoms"
                        ],

                    "linker_bond_count":
                        len(
                            sample_bond_lengths
                        ),

                    "linker_angle_count":
                        len(
                            sample_angles
                        ),

                    "linker_torsion_count":
                        len(
                            sample_torsions
                        ),

                    "bond_length_mean_A":
                        float(
                            np.mean(
                                sample_bond_lengths
                            )
                        ),

                    "bond_angle_mean_deg":
                        float(
                            np.mean(
                                sample_angles
                            )
                        ),

                    "eligible_heavy_atom_pairs":
                        eligible_pairs,

                    "positive_overlap_pairs":
                        positive_pairs,

                    "severe_clash_pairs":
                        severe_pairs,

                    "severe_clash_pair_rate":
                        (
                            severe_pairs
                            /
                            eligible_pairs
                        ),

                    "has_severe_clash":
                        (
                            severe_pairs
                            > 0
                        ),

                    "overlap_ge_0p20_pairs":
                        sample_aux[
                            0.20
                        ],

                    "overlap_ge_0p30_pairs":
                        sample_aux[
                            0.30
                        ],

                    "max_positive_overlap_A":
                        (
                            max(ov)
                            if ov
                            else 0.0
                        ),
                }
            )

            geometry_success += 1

        except Exception as exc:
            geometry_failed += 1

            failures.append(
                {
                    "sample_index":
                        sample_idx,

                    "error":
                        repr(exc),
                }
            )

    if geometry_success == 0:
        raise RuntimeError(
            f"{model_name}: no successfully evaluated "
            f"strict-valid samples"
        )

    if geometry_failed > 0:
        raise RuntimeError(
            f"{model_name}: "
            f"{geometry_failed} strict-valid samples "
            "failed direct-geometry evaluation. "
            "Inspect failures before generating "
            "publication results."
        )

    severe_pair_rate = (
        total_severe_clashes
        /
        total_eligible_pairs
    )

    severe_molecule_rate = (
        molecules_with_severe_clash
        /
        geometry_success
    )

    positive_overlap_pair_rate = (
        total_positive_overlap_pairs
        /
        total_eligible_pairs
    )

    positive_overlap_molecule_rate = (
        molecules_with_positive_overlap
        /
        geometry_success
    )

    auxiliary_pair_rates = {}

    auxiliary_molecule_rates = {}

    for threshold in (
        0.20,
        0.30,
    ):
        auxiliary_pair_rates[
            threshold
        ] = (
            auxiliary_pair_counts[
                threshold
            ]
            /
            total_eligible_pairs
        )

        auxiliary_molecule_rates[
            threshold
        ] = (
            auxiliary_molecule_counts[
                threshold
            ]
            /
            geometry_success
        )

    rmsd_summary = summarize(
        linker_rmsds,
        "A",
    )

    bond_summary = summarize(
        bond_lengths,
        "A",
    )

    angle_summary = summarize(
        bond_angles,
        "deg",
    )

    torsion_summary = summarize(
        torsions,
        "deg",
    )

    bond_category_summary = {
        category:
            summarize(
                values,
                "A",
            )
        for category, values
        in bond_lengths_by_category.items()
    }

    angle_category_summary = {
        category:
            summarize(
                values,
                "deg",
            )
        for category, values
        in bond_angles_by_category.items()
    }

    torsion_category_summary = {
        category:
            summarize(
                values,
                "deg",
            )
        for category, values
        in torsions_by_category.items()
    }

    return {
        "model":
            model_name,

        "pkl_path":
            str(
                pkl_path
            ),

        "total_records":
            total_records,

        "strict_valid_records":
            strict_valid_records,

        "geometry_success":
            geometry_success,

        "geometry_failed":
            geometry_failed,

        "rmsd":
            rmsd_summary,

        "bond_length":
            bond_summary,

        "bond_length_by_category":
            bond_category_summary,

        "bond_angle":
            angle_summary,

        "bond_angle_by_category":
            angle_category_summary,

        "torsion":
            torsion_summary,

        "torsion_by_category":
            torsion_category_summary,

        "clash": {
            "eligible_heavy_atom_pairs":
                total_eligible_pairs,

            "skipped_unsupported_pairs":
                total_skipped_pairs,

            "positive_overlap_pairs":
                total_positive_overlap_pairs,

            "positive_overlap_pair_rate":
                positive_overlap_pair_rate,

            "positive_overlap_molecules":
                molecules_with_positive_overlap,

            "positive_overlap_molecule_rate":
                positive_overlap_molecule_rate,

            "severe_threshold_A":
                SEVERE_OVERLAP_THRESHOLD_A,

            "severe_clashing_pairs":
                total_severe_clashes,

            "severe_pair_rate":
                severe_pair_rate,

            "severe_molecules":
                molecules_with_severe_clash,

            "severe_molecule_rate":
                severe_molecule_rate,

            "auxiliary_pair_rates":
                auxiliary_pair_rates,

            "auxiliary_molecule_rates":
                auxiliary_molecule_rates,

            "mean_max_overlap_A":
                float(
                    np.mean(
                        max_overlap_values
                    )
                ),

            "median_max_overlap_A":
                float(
                    np.median(
                        max_overlap_values
                    )
                ),

            "max_overlap_A":
                float(
                    np.max(
                        max_overlap_values
                    )
                ),

            "mean_positive_overlap_A":
                (
                    float(
                        np.mean(
                            positive_overlap_values
                        )
                    )
                    if positive_overlap_values
                    else 0.0
                ),

            "median_positive_overlap_A":
                (
                    float(
                        np.median(
                            positive_overlap_values
                        )
                    )
                    if positive_overlap_values
                    else 0.0
                ),
        },

        "values": {
            "linker_rmsd_A":
                linker_rmsds,

            "bond_lengths_A":
                bond_lengths,

            "bond_lengths_by_category_A":
                bond_lengths_by_category,

            "bond_angles_deg":
                bond_angles,

            "bond_angles_by_category_deg":
                bond_angles_by_category,

            "torsions_deg":
                torsions,

            "torsions_by_category_deg":
                torsions_by_category,
        },

        "sample_rows":
            sample_rows,

        "failures":
            failures[:50],
    }


def build_outputs(
    all_results,
):
    rows = []

    for result in all_results:
        model = result[
            "model"
        ]

        for (
            metric_name,
            metric_summary,
        ) in [
            (
                "linker_rmsd",
                result["rmsd"],
            ),
            (
                "bond_length",
                result["bond_length"],
            ),
            (
                "bond_angle",
                result["bond_angle"],
            ),
            (
                "torsion",
                result["torsion"],
            ),
        ]:
            if metric_name == "linker_rmsd":
                unit = "A"

            elif metric_name == "bond_length":
                unit = "A"

            else:
                unit = "deg"

            rows.append(
                {
                    "model":
                        model,

                    "metric":
                        metric_name,

                    "scope":
                        "overall",

                    "n":
                        metric_summary[
                            "n"
                        ],

                    "mean":
                        metric_summary[
                            f"mean_{unit}"
                        ],

                    "sd":
                        metric_summary[
                            f"sd_{unit}"
                        ],

                    "median":
                        metric_summary[
                            f"median_{unit}"
                        ],

                    "q1":
                        metric_summary[
                            f"q1_{unit}"
                        ],

                    "q3":
                        metric_summary[
                            f"q3_{unit}"
                        ],

                    "iqr":
                        metric_summary[
                            f"iqr_{unit}"
                        ],

                    "min":
                        metric_summary[
                            f"min_{unit}"
                        ],

                    "max":
                        metric_summary[
                            f"max_{unit}"
                        ],

                    "unit":
                        unit,
                }
            )

        for (
            category,
            metric_summary,
        ) in result[
            "bond_length_by_category"
        ].items():
            rows.append(
                {
                    "model":
                        model,

                    "metric":
                        "bond_length",

                    "scope":
                        category,

                    "n":
                        metric_summary[
                            "n"
                        ],

                    "mean":
                        metric_summary[
                            "mean_A"
                        ],

                    "sd":
                        metric_summary[
                            "sd_A"
                        ],

                    "median":
                        metric_summary[
                            "median_A"
                        ],

                    "q1":
                        metric_summary[
                            "q1_A"
                        ],

                    "q3":
                        metric_summary[
                            "q3_A"
                        ],

                    "iqr":
                        metric_summary[
                            "iqr_A"
                        ],

                    "min":
                        metric_summary[
                            "min_A"
                        ],

                    "max":
                        metric_summary[
                            "max_A"
                        ],

                    "unit":
                        "A",
                }
            )

        for (
            category,
            metric_summary,
        ) in result[
            "bond_angle_by_category"
        ].items():
            rows.append(
                {
                    "model":
                        model,

                    "metric":
                        "bond_angle",

                    "scope":
                        category,

                    "n":
                        metric_summary[
                            "n"
                        ],

                    "mean":
                        metric_summary[
                            "mean_deg"
                        ],

                    "sd":
                        metric_summary[
                            "sd_deg"
                        ],

                    "median":
                        metric_summary[
                            "median_deg"
                        ],

                    "q1":
                        metric_summary[
                            "q1_deg"
                        ],

                    "q3":
                        metric_summary[
                            "q3_deg"
                        ],

                    "iqr":
                        metric_summary[
                            "iqr_deg"
                        ],

                    "min":
                        metric_summary[
                            "min_deg"
                        ],

                    "max":
                        metric_summary[
                            "max_deg"
                        ],

                    "unit":
                        "deg",
                }
            )

        for (
            category,
            metric_summary,
        ) in result[
            "torsion_by_category"
        ].items():
            rows.append(
                {
                    "model":
                        model,

                    "metric":
                        "torsion",

                    "scope":
                        category,

                    "n":
                        metric_summary[
                            "n"
                        ],

                    "mean":
                        metric_summary[
                            "mean_deg"
                        ],

                    "sd":
                        metric_summary[
                            "sd_deg"
                        ],

                    "median":
                        metric_summary[
                            "median_deg"
                        ],

                    "q1":
                        metric_summary[
                            "q1_deg"
                        ],

                    "q3":
                        metric_summary[
                            "q3_deg"
                        ],

                    "iqr":
                        metric_summary[
                            "iqr_deg"
                        ],

                    "min":
                        metric_summary[
                            "min_deg"
                        ],

                    "max":
                        metric_summary[
                            "max_deg"
                        ],

                    "unit":
                        "deg",
                }
            )

        clash = result[
            "clash"
        ]

        rows.append(
            {
                "model":
                    model,

                "metric":
                    "severe_clash_pair_rate",

                "scope":
                    "overall",

                "n":
                    clash[
                        "eligible_heavy_atom_pairs"
                    ],

                "mean":
                    (
                        clash[
                            "severe_pair_rate"
                        ]
                        *
                        100.0
                    ),

                "sd":
                    None,

                "median":
                    None,

                "q1":
                    None,

                "q3":
                    None,

                "iqr":
                    None,

                "min":
                    None,

                "max":
                    None,

                "unit":
                    "%",
            }
        )

        rows.append(
            {
                "model":
                    model,

                "metric":
                    "severe_clash_molecule_rate",

                "scope":
                    "overall",

                "n":
                    result[
                        "geometry_success"
                    ],

                "mean":
                    (
                        clash[
                            "severe_molecule_rate"
                        ]
                        *
                        100.0
                    ),

                "sd":
                    None,

                "median":
                    None,

                "q1":
                    None,

                "q3":
                    None,

                "iqr":
                    None,

                "min":
                    None,

                "max":
                    None,

                "unit":
                    "%",
            }
        )

    fieldnames = [
        "model",
        "metric",
        "scope",
        "n",
        "mean",
        "sd",
        "median",
        "q1",
        "q3",
        "iqr",
        "min",
        "max",
        "unit",
    ]

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    main_rows = []

    for result in all_results:
        rmsd_result = result[
            "rmsd"
        ]

        bond_result = result[
            "bond_length"
        ]

        angle_result = result[
            "bond_angle"
        ]

        clash = result[
            "clash"
        ]

        main_rows.append(
            {
                "Model":
                    result[
                        "model"
                    ],

                "Strict-valid N":
                    result[
                        "strict_valid_records"
                    ],

                "Linker RMSD (A)":
                    (
                        f"{rmsd_result['mean_A']:.4f} "
                        f"± "
                        f"{rmsd_result['sd_A']:.4f}"
                    ),

                "Bond length (A)":
                    (
                        f"{bond_result['mean_A']:.4f} "
                        f"± "
                        f"{bond_result['sd_A']:.4f}"
                    ),

                "Bond angle (deg)":
                    (
                        f"{angle_result['mean_deg']:.2f} "
                        f"± "
                        f"{angle_result['sd_deg']:.2f}"
                    ),

                "Severe clash pair rate (%)":
                    (
                        f"{clash['severe_pair_rate'] * 100.0:.6f}"
                    ),
            }
        )

    main_fields = [
        "Model",
        "Strict-valid N",
        "Linker RMSD (A)",
        "Bond length (A)",
        "Bond angle (deg)",
        "Severe clash pair rate (%)",
    ]

    with open(
        MAIN_TABLE_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=main_fields,
        )

        writer.writeheader()

        writer.writerows(
            main_rows
        )

    with open(
        MAIN_TABLE_TSV,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\t".join(
                main_fields
            )
            +
            "\n"
        )

        for row in main_rows:
            f.write(
                "\t".join(
                    str(
                        row[field]
                    )
                    for field
                    in main_fields
                )
                +
                "\n"
            )

    with open(
        MAIN_TABLE_MD,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "| "
            +
            " | ".join(
                main_fields
            )
            +
            " |\n"
        )

        f.write(
            "| "
            +
            " | ".join(
                [
                    "---"
                ]
                *
                len(
                    main_fields
                )
            )
            +
            " |\n"
        )

        for row in main_rows:
            f.write(
                "| "
                +
                " | ".join(
                    str(
                        row[field]
                    )
                    for field
                    in main_fields
                )
                +
                " |\n"
            )

    all_sample_rows = []

    for result in all_results:
        all_sample_rows.extend(
            result[
                "sample_rows"
            ]
        )

    sample_fields = [
        "model",
        "sample_index",
        "sample_id",
        "uuid",
        "strict_valid",
        "linker_rmsd_A",
        "fragment_rmsd_after_alignment_A",
        "linker_heavy_atoms",
        "linker_bond_count",
        "linker_angle_count",
        "linker_torsion_count",
        "bond_length_mean_A",
        "bond_angle_mean_deg",
        "eligible_heavy_atom_pairs",
        "positive_overlap_pairs",
        "severe_clash_pairs",
        "severe_clash_pair_rate",
        "has_severe_clash",
        "overlap_ge_0p20_pairs",
        "overlap_ge_0p30_pairs",
        "max_positive_overlap_A",
    ]

    with open(
        SAMPLE_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=sample_fields,
        )

        writer.writeheader()

        writer.writerows(
            all_sample_rows
        )

    plot_rows = []

    for result in all_results:
        model = result[
            "model"
        ]

        for value in result[
            "values"
        ][
            "linker_rmsd_A"
        ]:
            plot_rows.append(
                {
                    "model":
                        model,

                    "metric":
                        "linker_rmsd",

                    "category":
                        "overall",

                    "value":
                        float(
                            value
                        ),

                    "unit":
                        "A",
                }
            )

        for (
            category,
            values,
        ) in result[
            "values"
        ][
            "bond_lengths_by_category_A"
        ].items():
            for value in values:
                plot_rows.append(
                    {
                        "model":
                            model,

                        "metric":
                            "bond_length",

                        "category":
                            category,

                        "value":
                            float(
                                value
                            ),

                        "unit":
                            "A",
                    }
                )

        for (
            category,
            values,
        ) in result[
            "values"
        ][
            "bond_angles_by_category_deg"
        ].items():
            for value in values:
                plot_rows.append(
                    {
                        "model":
                            model,

                        "metric":
                            "bond_angle",

                        "category":
                            category,

                        "value":
                            float(
                                value
                            ),

                        "unit":
                            "deg",
                    }
                )

        for (
            category,
            values,
        ) in result[
            "values"
        ][
            "torsions_by_category_deg"
        ].items():
            for value in values:
                plot_rows.append(
                    {
                        "model":
                            model,

                        "metric":
                            "torsion",

                        "category":
                            category,

                        "value":
                            float(
                                value
                            ),

                        "unit":
                            "deg",
                    }
                )

    plot_fields = [
        "model",
        "metric",
        "category",
        "value",
        "unit",
    ]

    with open(
        PLOT_DATA_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=plot_fields,
        )

        writer.writeheader()

        writer.writerows(
            plot_rows
        )

    json_summary = {
        "project_root":
            str(
                PROJECT_ROOT
            ),

        "paper_tables_dir":
            str(
                PAPER_TABLES_DIR
            ),

        "metric_scope": {
            "strict_valid_only":
                True,

            "reference_geometry_used_for_linker_rmsd":
                True,

            "reference_geometry_used_for_other_direct_metrics":
                False,

            "protac_stan_used":
                False,

            "deepternary_used":
                False,
        },

        "definitions": {
            "linker_rmsd":
                (
                    "Node-corresponded linker coordinate RMSD "
                    "after rigid Kabsch alignment using the "
                    "fixed fragment atoms only; generated linker "
                    "atom types and bond types are not used "
                    "for alignment."
                ),

            "bond_length":
                (
                    "Generated pred_mol only; linker-linker, "
                    "anchor1-linker, anchor2-linker bonds."
                ),

            "bond_angle":
                (
                    "Generated pred_mol only; heavy-atom angles "
                    "centered on linker atoms."
                ),

            "torsion":
                (
                    "Generated pred_mol only; linker-associated "
                    "heavy-atom proper torsions around eligible "
                    "non-ring single bonds."
                ),

            "severe_clash":
                (
                    "Whole generated PROTAC heavy-atom pairs; "
                    "exclude graph distances 1-3; "
                    "vdW overlap >= 0.40 A is severe."
                ),
        },

        "models": {
            result["model"]: {
                key:
                    value

                for key, value
                in result.items()

                if key not in (
                    "values",
                    "sample_rows",
                )
            }

            for result
            in all_results
        },
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_summary,
            f,
            indent=2,
            ensure_ascii=False,
        )


def print_summary(
    all_results,
):
    print()
    print(
        "=" * 100
    )

    print(
        "INTEGRATED DIRECT 3D GEOMETRY EVALUATION"
    )

    print(
        "=" * 100
    )

    for result in all_results:
        print()

        print(
            f"[{result['model']}]"
        )

        print(
            f"  Total records                 : "
            f"{result['total_records']}"
        )

        print(
            f"  Strict-valid records          : "
            f"{result['strict_valid_records']}"
        )

        print(
            f"  Geometry successful           : "
            f"{result['geometry_success']}"
        )

        print(
            f"  Geometry failed               : "
            f"{result['geometry_failed']}"
        )

        rmsd_result = result[
            "rmsd"
        ]

        print()

        print(
            "  Linker RMSD:"
        )

        print(
            f"    Mean                        : "
            f"{rmsd_result['mean_A']:.6f} A"
        )

        print(
            f"    SD                          : "
            f"{rmsd_result['sd_A']:.6f} A"
        )

        print(
            f"    Median                      : "
            f"{rmsd_result['median_A']:.6f} A"
        )

        bond_result = result[
            "bond_length"
        ]

        print()

        print(
            "  Generated bond length:"
        )

        print(
            f"    N                           : "
            f"{bond_result['n']}"
        )

        print(
            f"    Mean                        : "
            f"{bond_result['mean_A']:.6f} A"
        )

        print(
            f"    SD                          : "
            f"{bond_result['sd_A']:.6f} A"
        )

        print(
            f"    Median                      : "
            f"{bond_result['median_A']:.6f} A"
        )

        angle_result = result[
            "bond_angle"
        ]

        print()

        print(
            "  Generated bond angle:"
        )

        print(
            f"    N                           : "
            f"{angle_result['n']}"
        )

        print(
            f"    Mean                        : "
            f"{angle_result['mean_deg']:.6f} deg"
        )

        print(
            f"    SD                          : "
            f"{angle_result['sd_deg']:.6f} deg"
        )

        print(
            f"    Median                      : "
            f"{angle_result['median_deg']:.6f} deg"
        )

        torsion_result = result[
            "torsion"
        ]

        print()

        print(
            "  Generated linker torsion:"
        )

        print(
            f"    N                           : "
            f"{torsion_result['n']}"
        )

        if (
            torsion_result[
                "n"
            ]
            > 0
        ):
            print(
                f"    Mean                        : "
                f"{torsion_result['mean_deg']:.6f} deg"
            )

            print(
                f"    SD                          : "
                f"{torsion_result['sd_deg']:.6f} deg"
            )

            print(
                f"    Median                      : "
                f"{torsion_result['median_deg']:.6f} deg"
            )

        else:
            print(
                "    No linker-associated rotatable torsions found."
            )

        clash = result[
            "clash"
        ]

        print()

        print(
            "  Severe steric clash:"
        )

        print(
            f"    Eligible heavy-atom pairs  : "
            f"{clash['eligible_heavy_atom_pairs']}"
        )

        print(
            f"    Severe clashing pairs      : "
            f"{clash['severe_clashing_pairs']}"
        )

        print(
            f"    Severe pair clash rate     : "
            f"{clash['severe_pair_rate'] * 100.0:.6f}%"
        )

        print(
            f"    Severe molecule rate       : "
            f"{clash['severe_molecule_rate'] * 100.0:.4f}%"
        )

    print()

    print(
        "=" * 100
    )

    print(
        "MAIN TABLE PREVIEW"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Model':<15}"
        f"{'Strict N':>10}"
        f"{'RMSD (A)':>20}"
        f"{'Bond length (A)':>24}"
        f"{'Bond angle (deg)':>24}"
        f"{'Severe clash (%)':>20}"
    )

    print(
        "-" * 115
    )

    for result in all_results:
        rmsd_result = result[
            "rmsd"
        ]

        bond_result = result[
            "bond_length"
        ]

        angle_result = result[
            "bond_angle"
        ]

        clash = result[
            "clash"
        ]

        print(
            f"{result['model']:<15}"
            f"{result['strict_valid_records']:>10d}"
            +
            (
                f"{rmsd_result['mean_A']:.4f} "
                f"± "
                f"{rmsd_result['sd_A']:.4f}"
            ).rjust(
                20
            )
            +
            (
                f"{bond_result['mean_A']:.4f} "
                f"± "
                f"{bond_result['sd_A']:.4f}"
            ).rjust(
                24
            )
            +
            (
                f"{angle_result['mean_deg']:.2f} "
                f"± "
                f"{angle_result['sd_deg']:.2f}"
            ).rjust(
                24
            )
            +
            (
                f"{clash['severe_pair_rate'] * 100.0:.6f}"
            ).rjust(
                20
            )
        )

    print()

    print(
        "Torsion distributions are saved for the "
        "Supplementary three-panel figure; torsion is not "
        "forced into the main scalar comparison table."
    )

    print()

    print(
        "=" * 100
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 100
    )

    print(
        f"Full summary CSV    : "
        f"{SUMMARY_CSV}"
    )

    print(
        f"Main table CSV      : "
        f"{MAIN_TABLE_CSV}"
    )

    print(
        f"Main table TSV      : "
        f"{MAIN_TABLE_TSV}"
    )

    print(
        f"Main table Markdown : "
        f"{MAIN_TABLE_MD}"
    )

    print(
        f"Per-sample CSV      : "
        f"{SAMPLE_CSV}"
    )

    print(
        f"Plot data CSV       : "
        f"{PLOT_DATA_CSV}"
    )

    print(
        f"Summary JSON        : "
        f"{SUMMARY_JSON}"
    )


def main():
    all_results = []

    for (
        model_name,
        pkl_path,
    ) in MODELS.items():
        result = evaluate_model(
            model_name,
            pkl_path,
        )

        all_results.append(
            result
        )

    build_outputs(
        all_results
    )

    print_summary(
        all_results
    )

    for result in all_results:
        if result[
            "failures"
        ]:
            print()

            print(
                "=" * 80
            )

            print(
                f"FIRST FAILURES — "
                f"{result['model']}"
            )

            print(
                "=" * 80
            )

            for item in result[
                "failures"
            ][
                :20
            ]:
                print(
                    item
                )


if __name__ == "__main__":
    main()