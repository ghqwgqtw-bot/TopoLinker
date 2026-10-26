#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Build auditable DeepTernary binary inputs for second-stage RL.

The output preserves experimental protein/ligand coordinates, records explicit
chemical atom maps, and automatically accepts the first structurally valid
same-UniProt exact or conservative similar ligand. It never infers atom
identity from coordinate proximity; mapped pocket masks use target chemistry
with coordinates copied through explicit experimental atom maps.  A
stereochemically specified experimental
ligand may resolve an unspecified target stereocentre, but two explicitly
conflicting stereoisomers are never merged.
"""
import hashlib
import json
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFMCS, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

VERSION = "dt-input-rl2-4.2-protac-unbound-final"
CONTACT_CUTOFF_A = 4.0
# DeepTernary's PROTAC benchmark uses a known-pocket *unbound* protocol.
# The X-ray/resolution/R-free/same-chain filters in the paper belong to
# TernaryDB training-set construction, not to PROTAC unbound-input selection.
# We therefore retain experimental quality metadata for ranking/audit only.
PREFERRED_MAX_RESOLUTION_A = 3.5
PREFERRED_MAX_RFREE = 0.26
MIN_PROTEIN_RESIDUES_SANITY = 7
SIMILAR_TARGET_COVERAGE_MIN = 0.80
SIMILAR_CORE_HEAVY_ATOMS_MIN = 6

# Rooted-MCS timeout policy: keep a fast first pass, and only retry
# candidates that genuinely hit the MCS time limit. Chemistry/QC thresholds
# are unchanged by this retry.
MCS_TIMEOUT_INITIAL_S = 3
MCS_TIMEOUT_RETRY_S = 15

def entry_quality_rank(method, resolution, rfree, preferred_max_resolution=PREFERRED_MAX_RESOLUTION_A):
    """Ranking only; never a hard acceptance gate for PROTAC unbound inputs."""
    method = str(method or "").upper()
    resolution = safe_float(resolution) if "safe_float" in globals() else (float(resolution) if resolution is not None else None)
    rfree = safe_float(rfree) if "safe_float" in globals() else (float(rfree) if rfree is not None else None)
    is_xray = "X-RAY DIFFRACTION" in method
    good_res = resolution is not None and resolution <= float(preferred_max_resolution)
    good_rfree = rfree is not None and rfree <= PREFERRED_MAX_RFREE
    if is_xray and good_res and good_rfree:
        return 0
    if is_xray and good_res:
        return 1
    if good_res:
        return 2
    if resolution is not None:
        return 3
    return 4

_local = threading.local()
_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)

# Components that cannot serve as a drug-like terminal ligand.  Filtering is
# performed once while building the protein catalogue, rather than once for
# every target pair.  The exclusion is written to catalog_exclusions.csv.
COMMON_NON_DRUGLIKE = {
    "HOH", "WAT", "DOD", "NA", "K", "CL", "CA", "MG", "ZN", "MN", "FE",
    "CO", "CU", "NI", "CD", "HG", "IOD", "BR", "F", "LI", "CS", "SR",
    "ACT", "ACE", "FMT", "GOL", "PEG", "SO4", "PO4", "TRS", "XYP", "BME",
    "EDO", "PG4", "PG5", "DMS", "MPD", "IPA", "EOH", "UNX",
}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)

def write_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(value)
    os.replace(tmp, path)

def atomic_copy(src, dst):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    import shutil
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def request(url, *, payload=None, params=None, binary=False, retries=3, timeout=(10, 45), session=None):
    """204 is a successful empty search. Only transient failures are retried."""
    if session is None:
        if not hasattr(_local, "session"):
            _local.session = requests.Session()
            _local.session.headers.update({"User-Agent": VERSION})
        session = _local.session
    for attempt in range(retries):
        delay = min(2 ** attempt, 8)
        try:
            response = session.request("POST" if payload is not None else "GET", url,
                                       json=payload, params=params, timeout=timeout)
            if response.status_code == 204:
                if binary:
                    raise ValueError(f"Empty structure response: {url}")
                return {"result_set": [], "total_count": 0}
            if response.status_code in (429, 500, 502, 503, 504):
                try:
                    delay = min(30, max(delay, float(response.headers.get("Retry-After", delay))))
                except ValueError:
                    pass
                response.raise_for_status()
            elif response.status_code != 200:
                raise ValueError(f"Permanent HTTP {response.status_code}: {url}: {response.text[:240]}")
            else:
                return response.content if binary else response.json()
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
            if attempt + 1 == retries:
                raise
            time.sleep(delay)
    raise RuntimeError("Unreachable HTTP state")

def canonical(mol):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def _nonisomeric(mol):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def _largest_fragment(mol):
    frags = Chem.GetMolFrags(Chem.Mol(mol), asMols=True, sanitizeFrags=True)
    if not frags:
        raise ValueError("No molecular fragment after parsing")
    return max(frags, key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms(), canonical(m)))

def standardize_mol(mol, *, fragment_parent=True):
    """Conservative comparison-only standardization; coordinates are not used here."""
    x = Chem.Mol(mol)
    for atom in x.GetAtoms():
        atom.SetAtomMapNum(0)
    # RDKit diagnostics are captured as structured failures by the caller.
    # Blocking its stderr here prevents millions of duplicate warning lines
    # when the same RCSB component is compared with many target pairs.
    with rdBase.BlockLogs():
        x = rdMolStandardize.Cleanup(x)
        if fragment_parent:
            x = rdMolStandardize.FragmentParent(x)
        x = rdMolStandardize.Uncharger().uncharge(x)
        x = rdMolStandardize.TautomerEnumerator().Canonicalize(x)
        Chem.SanitizeMol(x)
        x = Chem.RemoveHs(x)
    if x.GetNumHeavyAtoms() < 3:
        raise ValueError("Standardized ligand has fewer than 3 heavy atoms")
    return x

@lru_cache(maxsize=50000)
def standardized_candidate_smiles(smiles):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid candidate SMILES: {smiles}")
    return canonical(standardize_mol(mol, fragment_parent=True))

@lru_cache(maxsize=20000)
def standardized_target_smiles(smiles):
    """Cache target standardization; one target is compared with many ligands."""
    return canonical(standardize_mol(terminal(smiles), fragment_parent=False))

def _assigned_stereo_count(mol):
    atom_count = sum(
        a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()
    )
    bond_count = sum(
        b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
        for b in mol.GetBonds()
    )
    return int(atom_count + bond_count)

def _stereo_relation(target, candidate):
    """Classify stereo after constitution equality has been established.

    Chirality is directional: stereo specified by the target is a constraint;
    extra stereo specified only by the experimental candidate resolves an
    otherwise unspecified target and is permitted.  This is not the same as
    accepting two explicitly conflicting stereoisomers.
    """
    if _nonisomeric(target) != _nonisomeric(candidate):
        return "different_constitution"
    if canonical(target) == canonical(candidate):
        return "exact"
    target_stereo = _assigned_stereo_count(target)
    candidate_stereo = _assigned_stereo_count(candidate)
    if candidate_stereo < target_stereo:
        return "candidate_missing_target_stereo"
    if not candidate.HasSubstructMatch(target, useChirality=True):
        return "explicit_conflict"
    if candidate_stereo > target_stereo:
        return "candidate_resolves_unspecified"
    return "compatible_by_symmetry"

def _exact_constitution_result(target, candidate, representation):
    """Return (same constitution, accepted screen result or None).

    ``exact`` is a provenance label, not a quality tier.  All accepted records
    later pass the same coordinate, contact, clash, occupancy and mapping audit.
    """
    if _nonisomeric(target) != _nonisomeric(candidate):
        return False, None
    relation = _stereo_relation(target, candidate)
    if relation in {"explicit_conflict", "candidate_missing_target_stereo"}:
        return True, None
    detail = {
        ("raw", "exact"): "raw_exact",
        ("standardized", "exact"): "standardized_exact",
        ("raw", "candidate_resolves_unspecified"): "raw_stereo_resolved",
        ("standardized", "candidate_resolves_unspecified"): "standardized_stereo_resolved",
    }.get((representation, relation), f"{representation}_stereo_compatible")
    return True, {
        "match_type": "exact",
        "match_detail": detail,
        "screen_rank": {
            "raw_exact": 0,
            "raw_stereo_resolved": 1,
            "raw_stereo_compatible": 1,
            "standardized_exact": 2,
            "standardized_stereo_resolved": 3,
            "standardized_stereo_compatible": 3,
        }.get(detail, 3),
        "similarity": 1.0,
        "target_coverage": 1.0,
        "candidate_coverage": 1.0,
        "core_heavy_atoms": int(target.GetNumHeavyAtoms()),
        "chemistry_equivalence": f"{representation}_constitution",
        "stereo_relation": relation,
        "mcs_smarts": "",
        "target_core_indices_std": list(range(target.GetNumAtoms())),
        "candidate_core_indices_std": list(range(candidate.GetNumAtoms())),
    }

def catalog_exclusion_reason(comp_id, smiles):
    """Cheap, once-per-catalogue exclusion for ions and invalid descriptors."""
    comp_id = str(comp_id or "").strip().upper()
    if comp_id in COMMON_NON_DRUGLIKE:
        return "common_non_druglike_component"
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "invalid_metadata_smiles"
    if mol.GetNumHeavyAtoms() < 3:
        return "fewer_than_3_heavy_atoms"
    return None

def relaxed_query(mol):
    """Element/connectivity query that ignores formal charge, aromatic flags and bond order."""
    rw = Chem.RWMol()
    for atom in mol.GetAtoms():
        rw.AddAtom(Chem.AtomFromSmarts(f"[#{atom.GetAtomicNum()}]"))
    for bond in mol.GetBonds():
        rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.UNSPECIFIED)
    return rw.GetMol()

def relaxed_unique_match(whole, query, allowed=None):
    q = relaxed_query(query)
    matches = whole.GetSubstructMatches(q, useChirality=False, uniquify=False, maxMatches=2048)
    if len(matches) >= 2048:
        raise ValueError("Relaxed atom-map enumeration limit reached")
    allowed = None if allowed is None else set(allowed)
    matches = [tuple(m) for m in matches if allowed is None or set(m) == allowed]
    if not matches:
        raise ValueError("No standardized element/connectivity substructure map")
    if len({frozenset(m) for m in matches}) != 1:
        raise ValueError("Ambiguous non-equivalent standardized terminal occurrences")
    return list(min(matches))

def _raw_sdf_molecule(path):
    with rdBase.BlockLogs():
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        mols = list(supplier)
    if len(mols) != 1 or mols[0] is None or mols[0].GetNumConformers() != 1:
        raise ValueError(f"Expected one coordinate-bearing raw molecule: {path}")
    mol = Chem.RWMol(mols[0])
    for idx in reversed([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]):
        mol.RemoveAtom(idx)
    return mol.GetMol()

def _template_candidates(candidate_smiles):
    parsed = Chem.MolFromSmiles(str(candidate_smiles))
    if parsed is None:
        raise ValueError("Candidate metadata SMILES cannot form repair template")
    variants = []
    for frag in Chem.GetMolFrags(parsed, asMols=True, sanitizeFrags=True):
        if frag.GetNumHeavyAtoms() >= 3:
            variants.append(frag)
            try:
                variants.append(standardize_mol(frag, fragment_parent=False))
            except Exception:
                pass
    dedup = {}
    for mol in variants:
        dedup.setdefault(canonical(mol), mol)
    return list(dedup.values())

def repair_sdf_from_metadata_template(path, candidate_smiles):
    """Repair RCSB SDF bond-order/valence encoding using metadata chemistry + raw coordinates."""
    raw = _raw_sdf_molecule(path)
    raw_xyz = raw.GetConformer().GetPositions()
    raw_elements = sorted(a.GetAtomicNum() for a in raw.GetAtoms())
    errors = []
    for template in _template_candidates(candidate_smiles):
        if template.GetNumAtoms() != raw.GetNumAtoms():
            continue
        if sorted(a.GetAtomicNum() for a in template.GetAtoms()) != raw_elements:
            continue
        try:
            mapping = relaxed_unique_match(raw, template, allowed=range(raw.GetNumAtoms()))
            repaired = Chem.Mol(template)
            repaired.RemoveAllConformers()
            conf = Chem.Conformer(repaired.GetNumAtoms())
            conf.Set3D(True)
            for i, j in enumerate(mapping):
                x, y, z = map(float, raw_xyz[j])
                conf.SetAtomPosition(i, (x, y, z))
            repaired.AddConformer(conf, assignId=True)
            Chem.SanitizeMol(repaired)
            return repaired
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("SDF template repair failed: " + " | ".join(errors)[:600])

def read_sdf_with_repair(path, candidate_smiles):
    try:
        return read_sdf(path), "strict_sdf"
    except Exception as strict_exc:
        try:
            return repair_sdf_from_metadata_template(path, candidate_smiles), "metadata_template_repair"
        except Exception as repair_exc:
            raise ValueError(f"strict_sdf_failed:{strict_exc}; template_repair_failed:{repair_exc}")

def classify_reason(error):
    s = str(error)
    rules = [
        ("exit_not_in_common_core", "Similar-ligand common core does not contain the linker exit atom"),
        ("exit_vector_not_preserved", "Similar-ligand common core does not preserve exit-vector direction"),
        ("mapped_subgraph_invalid", "Cannot build chemically valid mapped subgraph"),
        ("no_substructure_map", "No chemical substructure map"),
        ("no_standardized_substructure_map", "No standardized element/connectivity substructure map"),
        ("ambiguous_terminal_mapping", "Ambiguous non-equivalent terminal occurrences"),
        ("ambiguous_standardized_mapping", "Ambiguous non-equivalent standardized terminal occurrences"),
        ("uniprot_chain_mapping_failed", "uniprot_chain_mapping_failed"),
        ("not_xray", "Experimental method is not X-RAY DIFFRACTION"),
        ("resolution_above_limit", "Resolution exceeds"),
        ("missing_r_free", "R-free is missing"),
        ("r_free_above_limit", "R-free "),
        ("not_unbound_binary_source", "Source PDB contains context partner protein(s)"),
        ("rooted_mcs_timeout", "Exit-aware rooted MCS timed out"),
        ("rooted_mcs_missing_exit", "Similar-ligand matching requires an explicit linker exit atom"),
        ("protein_ligand_clash", "Protein-ligand heavy-atom clash"),
        ("low_occupancy", "Ligand occupancy below 0.5"),
        ("short_protein_construct", "Protein construct has fewer than"),
        ("insufficient_contact", "contact protein within 4 A"),
        ("sdf_sanitize_or_repair_failed", "strict_sdf_failed:"),
        ("sdf_instance_mismatch", "SDF is not the observed ligand coordinate instance"),
        ("missing_or_extra_ligand_atoms", "Missing/extra heavy atoms in experimental ligand"),
        ("invalid_metadata_smiles", "Invalid candidate SMILES"),
        ("too_few_heavy_atoms", "fewer than 3 heavy atoms"),
        ("stereo_conflict", "Explicit stereochemistry conflict"),
        ("candidate_missing_target_stereo", "Candidate does not specify target stereochemistry"),
        ("http_failure", "HTTP"),
        ("timeout", "Timeout"),
        ("connection_failure", "Connection"),
    ]
    for code, needle in rules:
        if needle in s:
            return code
    return "other_validation_failure"

def terminal(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid terminal SMILES: {smiles}")
    edit = Chem.RWMol(mol)
    for i in sorted([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0], reverse=True):
        edit.RemoveAtom(i)
    mol = edit.GetMol()
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        # Allow the severed dummy bond to be capped by implicit hydrogen.
        atom.SetNoImplicit(False)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    if len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumAtoms() < 3:
        raise ValueError("Terminal must be one connected fragment with >=3 heavy atoms")
    return Chem.MolFromSmiles(canonical(mol))

def terminal_with_exit(smiles):
    """Return canonical terminal SMILES and a version marking the linker exit atom.

    A single dummy atom (``*``) is interpreted as the linker attachment.  Its
    neighbouring real atom receives atom-map number 1 before the dummy is
    removed.  Keeping this marker prevents two fragments with the same capped
    graph but different exit positions from being merged into one target.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid terminal SMILES: {smiles}")
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) > 1:
        raise ValueError("Terminal must contain at most one linker dummy atom")
    exit_old = None
    if dummies:
        neighbours = list(dummies[0].GetNeighbors())
        if len(neighbours) != 1 or neighbours[0].GetAtomicNum() == 0:
            raise ValueError("Linker dummy atom must have exactly one real neighbour")
        exit_old = neighbours[0].GetIdx()
    dummy_indices = sorted((a.GetIdx() for a in dummies), reverse=True)
    edit = Chem.RWMol(mol)
    for idx in dummy_indices:
        edit.RemoveAtom(idx)
    core = edit.GetMol()
    for atom in core.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetNoImplicit(False)
    Chem.SanitizeMol(core)
    core = Chem.RemoveHs(core)
    if len(Chem.GetMolFrags(core)) != 1 or core.GetNumAtoms() < 3:
        raise ValueError("Terminal must be one connected fragment with >=3 heavy atoms")
    exit_new = None
    if exit_old is not None:
        exit_new = exit_old - sum(1 for idx in dummy_indices if idx < exit_old)
        core.GetAtomWithIdx(exit_new).SetAtomMapNum(1)
    marked_smiles = Chem.MolToSmiles(core, isomericSmiles=True, canonical=True)
    if exit_new is not None:
        core.GetAtomWithIdx(exit_new).SetAtomMapNum(0)
    return canonical(core), marked_smiles, exit_new is not None

def pair_id(side, uniprot, smiles, marked_smiles=None):
    identity = marked_smiles or smiles
    return side.lower() + "_" + digest([str(uniprot).strip().upper(), identity])[:16]

def read_sdf(path):
    with rdBase.BlockLogs():
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
        mols = list(supplier)
    if len(mols) != 1 or mols[0] is None or mols[0].GetNumConformers() != 1:
        raise ValueError(f"Expected one sanitized, coordinate-bearing molecule: {path}")
    mol = mols[0]
    xyz = mol.GetConformer().GetPositions()
    if not np.isfinite(xyz).all() or np.linalg.matrix_rank(xyz - xyz.mean(0), tol=1e-3) < 2:
        raise ValueError(f"Invalid/degenerate coordinates: {path}")
    return mol

def write_sdf(path, mol):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".{os.getpid()}.{threading.get_ident()}.tmp.sdf")
    with Chem.SDWriter(str(tmp)) as writer:
        writer.write(mol)
    read_sdf(tmp)
    os.replace(tmp, path)

def submol(mol, indices):
    """Extract an induced atom subset while preserving a chemically valid graph.

    The normal path preserves the original aromatic representation. A
    kekulized fallback is used only when deleting atoms leaves RDKit with an
    otherwise valid fragment whose aromatic flags cannot be sanitized as a
    standalone molecule. Atom identity is never inferred from geometry.
    """
    indices = [int(i) for i in indices]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Mapped subgraph indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= mol.GetNumAtoms():
        raise ValueError("Mapped subgraph index outside molecule")

    def _build(source):
        keep = set(indices)
        edit = Chem.RWMol(Chem.Mol(source))
        for i in reversed(range(source.GetNumAtoms())):
            if i not in keep:
                edit.RemoveAtom(i)
        result = edit.GetMol()
        for atom in result.GetAtoms():
            atom.SetNoImplicit(False)
        Chem.SanitizeMol(result)
        old_sorted = sorted(indices)
        return Chem.RenumberAtoms(
            result, [old_sorted.index(i) for i in indices]
        )

    try:
        return _build(mol)
    except Exception as first_exc:
        try:
            kek = Chem.Mol(mol)
            Chem.Kekulize(kek, clearAromaticFlags=True)
            return _build(kek)
        except Exception as second_exc:
            raise ValueError(
                "Cannot build chemically valid mapped subgraph: "
                f"{first_exc}; kekulized_fallback:{second_exc}"
            ) from second_exc

def graph_check(query, whole, mapping, *, induced=True):
    """Check injective element/charge/aromatic/bond correspondence."""
    if len(mapping) != query.GetNumAtoms() or len(set(mapping)) != len(mapping):
        raise ValueError("Atom map is not one-to-one")
    if not mapping or min(mapping) < 0 or max(mapping) >= whole.GetNumAtoms():
        raise ValueError("Atom map outside molecule")
    for a, idx in zip(query.GetAtoms(), mapping):
        b = whole.GetAtomWithIdx(int(idx))
        if (a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsAromatic()) != (b.GetAtomicNum(), b.GetFormalCharge(), b.GetIsAromatic()):
            raise ValueError("Atom identity/charge/aromaticity mismatch")
    for i in range(len(mapping)):
        for j in range(i):
            a, b = query.GetBondBetweenAtoms(i, j), whole.GetBondBetweenAtoms(int(mapping[i]), int(mapping[j]))
            if a is None and not induced:
                continue
            if (a is None) != (b is None) or (a is not None and a.GetBondType() != b.GetBondType()):
                raise ValueError("Bond graph mismatch")

def unique_match(whole, query, allowed=None, chirality=True):
    matches = whole.GetSubstructMatches(query, useChirality=chirality, uniquify=False, maxMatches=1024)
    if len(matches) >= 1024:
        raise ValueError("Atom-map enumeration limit reached")
    allowed = None if allowed is None else set(allowed)
    matches = [tuple(m) for m in matches if allowed is None or set(m) == allowed]
    if not matches:
        raise ValueError("No chemical substructure map")
    if len({frozenset(m) for m in matches}) != 1:
        raise ValueError("Ambiguous non-equivalent terminal occurrences")
    answer = min(matches)  # symmetry-equivalent maps: deterministic index tie break
    graph_check(query, whole, answer)
    return list(answer)

def audit_mapping(query, whole, mapping):
    """Post-audit a relaxed map in standardized chemical space.

    This closes the v1.5 loophole where the fallback matcher ignored bond
    order, charge, aromaticity and chirality.  Standardization may reconcile a
    representation difference, but it may not change constitution or violate
    target-specified stereochemistry.
    """
    if len(mapping) != query.GetNumAtoms() or len(set(mapping)) != len(mapping):
        raise ValueError("Post-audit atom map is not one-to-one")
    mapped = submol(whole, mapping)
    q_std = standardize_mol(query, fragment_parent=False)
    m_std = standardize_mol(mapped, fragment_parent=False)
    if _nonisomeric(q_std) != _nonisomeric(m_std):
        raise ValueError("Post-audit standardized bond/charge graph mismatch")
    relation = _stereo_relation(q_std, m_std)
    if relation == "explicit_conflict":
        raise ValueError("Explicit stereochemistry conflict after atom mapping")
    if relation == "candidate_missing_target_stereo":
        raise ValueError("Candidate does not specify target stereochemistry after atom mapping")
    return relation

@lru_cache(maxsize=20000)
class _ExitRootAcceptance(rdFMCS.MCSAcceptance):
    """Require an MCS to contain the marked linker-exit atom and one neighbour."""

    ROOT_PROP = "_DT_EXIT_ROOT"

    def __call__(self, mol1, mol2, atom_idx_match, bond_idx_match, params):
        pairs = [(int(a), int(b)) for a, b in atom_idx_match]
        roots1 = [a.GetIdx() for a in mol1.GetAtoms() if a.HasProp(self.ROOT_PROP)]
        roots2 = [a.GetIdx() for a in mol2.GetAtoms() if a.HasProp(self.ROOT_PROP)]
        if len(roots1) + len(roots2) != 1:
            return False
        if roots1:
            root = roots1[0]
            matched = {a for a, _ in pairs}
            neighbours = {n.GetIdx() for n in mol1.GetAtomWithIdx(root).GetNeighbors()}
        else:
            root = roots2[0]
            matched = {b for _, b in pairs}
            neighbours = {n.GetIdx() for n in mol2.GetAtomWithIdx(root).GetNeighbors()}
        return root in matched and bool(neighbours & matched)


def _standardized_target_and_exit(target_smiles, target_smiles_mapped):
    """Return standardized target and the exit index in standardized atom order."""
    raw_target, raw_exit = _target_and_exit(target_smiles, target_smiles_mapped)
    t_std = Chem.MolFromSmiles(standardized_target_smiles(target_smiles))
    if t_std is None:
        raise ValueError("Cannot standardize target terminal")
    if raw_exit is None:
        return t_std, None
    if raw_target.GetNumAtoms() != t_std.GetNumAtoms():
        raise ValueError("Target standardization changed atom count; exit cannot be audited")
    std_to_raw = relaxed_unique_match(
        raw_target, t_std, allowed=range(raw_target.GetNumAtoms())
    )
    hits = [i for i, raw_i in enumerate(std_to_raw) if int(raw_i) == int(raw_exit)]
    if len(hits) != 1:
        raise ValueError("Cannot map linker exit into standardized target")
    return t_std, int(hits[0])


def _unique_rooted_match_sets(mol, query, *, exit_idx=None):
    matches = mol.GetSubstructMatches(
        query, useChirality=False, uniquify=False, maxMatches=2048
    )
    if len(matches) >= 2048:
        raise ValueError("Rooted MCS atom-map enumeration limit reached")
    if exit_idx is not None:
        neighbours = {
            n.GetIdx() for n in mol.GetAtomWithIdx(int(exit_idx)).GetNeighbors()
        }
        matches = [
            tuple(m) for m in matches
            if int(exit_idx) in m and bool(neighbours & set(m))
        ]
    else:
        matches = [tuple(m) for m in matches]
    if not matches:
        return []
    by_set = {}
    for m in matches:
        by_set.setdefault(frozenset(m), m)
    return [by_set[k] for k in sorted(by_set, key=lambda x: tuple(sorted(x)))]


def exit_aware_rooted_mcs(target_std, candidate_std, target_exit_idx, timeout_s=3):
    """Find the largest conservative common core rooted at the linker exit.

    The MCS search itself is constrained: an accepted MCS must contain the
    target exit atom and at least one directly bonded target neighbour.  This
    avoids the old failure mode where an unconstrained larger MCS was found in
    an unrelated part of the ligand and rejected only afterwards.
    """
    if target_exit_idx is None:
        raise ValueError("Similar-ligand matching requires an explicit linker exit atom")

    params = rdFMCS.MCSParameters()
    params.Timeout = int(timeout_s)
    params.MaximizeBonds = True
    params.StoreAll = True
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    params.AtomCompareParameters.MatchFormalCharge = True
    params.AtomCompareParameters.MatchChiralTag = False
    params.AtomCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = True
    params.BondCompareParameters.MatchStereo = False
    params.ShouldAcceptMCS = _ExitRootAcceptance()

    target_search = Chem.Mol(target_std)
    target_search.GetAtomWithIdx(int(target_exit_idx)).SetBoolProp(
        _ExitRootAcceptance.ROOT_PROP, True
    )
    result = rdFMCS.FindMCS([target_search, candidate_std], params)
    if result.canceled:
        raise ValueError("Exit-aware rooted MCS timed out")
    if result.numAtoms <= 0:
        return None

    queries = []
    if getattr(result, "degenerateSmartsQueryMolDict", None):
        queries.extend(result.degenerateSmartsQueryMolDict.items())
    elif result.smartsString:
        q = Chem.MolFromSmarts(result.smartsString)
        if q is not None:
            queries.append((result.smartsString, q))

    valid = []
    for smarts, query in queries:
        if query is None:
            continue
        target_sets = _unique_rooted_match_sets(
            target_std, query, exit_idx=target_exit_idx
        )
        candidate_sets = _unique_rooted_match_sets(candidate_std, query)
        if not target_sets or not candidate_sets:
            continue
        # More than one chemically distinct atom set makes the pocket atom map
        # non-unique. Symmetry-equivalent permutations of the same set are fine.
        if len(target_sets) != 1:
            continue
        if len(candidate_sets) != 1:
            continue
        ti = list(target_sets[0])
        ci = list(candidate_sets[0])
        core_atoms = len(ti)
        coverage = core_atoms / max(1, target_std.GetNumHeavyAtoms())
        candidate_coverage = core_atoms / max(1, candidate_std.GetNumHeavyAtoms())
        if core_atoms < SIMILAR_CORE_HEAVY_ATOMS_MIN:
            continue
        if coverage < SIMILAR_TARGET_COVERAGE_MIN:
            continue
        valid.append({
            "mcs_smarts": str(smarts),
            "target_core_indices_std": ti,
            "candidate_core_indices_std": ci,
            "core_heavy_atoms": int(core_atoms),
            "target_coverage": float(coverage),
            "candidate_coverage": float(candidate_coverage),
        })

    if not valid:
        return None
    valid.sort(key=lambda x: (
        -x["core_heavy_atoms"],
        -x["target_coverage"],
        tuple(x["target_core_indices_std"]),
        tuple(x["candidate_core_indices_std"]),
    ))
    return valid[0]


@lru_cache(maxsize=40000)
def chemistry_screen(
    target_smiles,
    candidate_smiles,
    target_smiles_mapped=None,
    mcs_timeout_s=MCS_TIMEOUT_INITIAL_S,
):
    """Chemistry screen with no A/B/C tiers.

    Exact constitutional matches are preferred.  Otherwise a candidate is
    accepted only if an exit-aware rooted MCS covers >=80% of the target and
    contains >=6 heavy atoms. Tanimoto is recorded only for ranking; it is not
    a hard rejection threshold. Candidate coverage is diagnostic only.
    """
    t_raw = terminal(target_smiles)
    c_raw = None
    try:
        c_raw = terminal(candidate_smiles)
    except Exception:
        pass
    if c_raw is not None:
        same_constitution, result = _exact_constitution_result(t_raw, c_raw, "raw")
        if same_constitution:
            return result

    t_std, exit_std = _standardized_target_and_exit(
        target_smiles, target_smiles_mapped
    )
    c_std = Chem.MolFromSmiles(standardized_candidate_smiles(candidate_smiles))
    if c_std is None:
        return None
    same_constitution, result = _exact_constitution_result(t_std, c_std, "standardized")
    if same_constitution:
        return result

    sim = DataStructs.TanimotoSimilarity(
        _fpgen.GetFingerprint(t_std), _fpgen.GetFingerprint(c_std)
    )
    rooted = exit_aware_rooted_mcs(
        t_std,
        c_std,
        exit_std,
        timeout_s=int(mcs_timeout_s),
    )
    if rooted is None:
        return None
    return {
        "match_type": "similar_rooted_mcs",
        "match_detail": "exit_aware_standardized_rooted_mcs",
        "screen_rank": 4,
        "similarity": float(sim),
        **rooted,
        "chemistry_equivalence": "standardized_exit_rooted_common_core",
        "stereo_relation": "partial_core_pending_coordinate_audit",
    }


def chemistry_screen_with_timeout_retry(
    target_smiles,
    candidate_smiles,
    target_smiles_mapped=None,
):
    """Run the same chemistry policy with a longer retry only on MCS timeout.

    This does NOT relax chemistry acceptance criteria. The target-coverage,
    minimum-core-size, exit-root, ring, charge, bond-order and stereochemistry
    requirements remain exactly the same.
    """
    try:
        return chemistry_screen(
            target_smiles,
            candidate_smiles,
            target_smiles_mapped,
            MCS_TIMEOUT_INITIAL_S,
        )
    except Exception as exc:
        if classify_reason(exc) != "rooted_mcs_timeout":
            raise

        return chemistry_screen(
            target_smiles,
            candidate_smiles,
            target_smiles_mapped,
            MCS_TIMEOUT_RETRY_S,
        )


def _target_and_exit(target_smiles, target_smiles_mapped=None):
    exit_idx = None
    if target_smiles_mapped:
        target = Chem.MolFromSmiles(target_smiles_mapped)
        if target is None:
            raise ValueError("Invalid mapped terminal SMILES")
        marked = [a.GetIdx() for a in target.GetAtoms() if a.GetAtomMapNum() == 1]
        if len(marked) > 1:
            raise ValueError("Multiple linker exit atoms in mapped terminal")
        exit_idx = marked[0] if marked else None
        for atom in target.GetAtoms():
            atom.SetAtomMapNum(0)
    else:
        target = terminal(target_smiles)
    return target, exit_idx

def map_to_bound(target_smiles, bound, screen, target_smiles_mapped=None):
    target, exit_idx = _target_and_exit(target_smiles, target_smiles_mapped)
    map_mode = "strict"
    if screen["match_type"] == "exact":
        try:
            bi = unique_match(bound, target)
        except Exception:
            bi = relaxed_unique_match(bound, target)
            map_mode = "standardized_graph_fallback"
        mapping_stereo_relation = audit_mapping(target, bound, bi)
        return list(range(target.GetNumAtoms())), bi, map_mode, mapping_stereo_relation

    if screen["match_type"] != "similar_rooted_mcs":
        raise ValueError(f"Unsupported chemistry match type: {screen.get('match_type')}")

    # IMPORTANT: ``screen["target_core_indices_std"]`` is indexed against the
    # canonical standardized target created by _standardized_target_and_exit()
    # inside chemistry_screen().  Do not rebuild a merely standardized target
    # here with its original atom order: the chemistry can be identical while
    # the atom indices differ, which can move the linker-exit index onto the
    # wrong raw atom and generate false mapping failures.
    t_std, exit_std = _standardized_target_and_exit(
        target_smiles, target_smiles_mapped
    )
    std_to_raw = relaxed_unique_match(
        target, t_std, allowed=range(target.GetNumAtoms())
    )
    if exit_idx is not None:
        if exit_std is None or int(std_to_raw[int(exit_std)]) != int(exit_idx):
            raise ValueError(
                "Canonical standardized target index map does not preserve linker exit"
            )
    ti_std = [int(i) for i in screen.get("target_core_indices_std", [])]
    if not ti_std or max(ti_std) >= t_std.GetNumAtoms():
        raise ValueError("Invalid rooted-MCS target atom indices")
    ti = tuple(int(std_to_raw[i]) for i in ti_std)

    if exit_idx is None:
        raise ValueError("Similar-ligand mapping requires an explicit linker exit atom")
    core_set = set(ti)
    if exit_idx not in core_set:
        raise ValueError("Exit-aware rooted MCS lost the linker exit atom")
    exit_neighbours = [a.GetIdx() for a in target.GetAtomWithIdx(exit_idx).GetNeighbors()]
    if not exit_neighbours or not any(idx in core_set for idx in exit_neighbours):
        raise ValueError("Exit-aware rooted MCS lost the exit-vector neighbour")

    core = submol(target, ti)
    try:
        bi = unique_match(bound, core, chirality=True)
    except Exception:
        bi = relaxed_unique_match(bound, core)
        map_mode = "standardized_graph_fallback"
    mapping_stereo_relation = audit_mapping(core, bound, bi)
    return list(ti), bi, map_mode, mapping_stereo_relation


def mapped_core_from_target_with_bound_coords(
    target_smiles,
    target_smiles_mapped,
    target_indices,
    bound,
    bound_indices,
):
    """Build a DeepTernary pocket mask in target chemistry, experimental pose.

    Topology comes from the PROTAC terminal/common core. Coordinates are copied
    only through the explicit target_indices <-> bound_indices correspondence
    from the experimental bound ligand.
    """
    ti = [int(i) for i in target_indices]
    bi = [int(i) for i in bound_indices]
    if not ti or len(ti) != len(bi):
        raise ValueError("Mapped-core target/bound atom counts do not agree")
    if len(set(ti)) != len(ti) or len(set(bi)) != len(bi):
        raise ValueError("Mapped-core atom correspondence is not one-to-one")

    target, _ = _target_and_exit(target_smiles, target_smiles_mapped)
    core = submol(target, ti)
    if core.GetNumAtoms() != len(bi):
        raise ValueError("Mapped-core topology/coordinate atom counts do not agree")

    bconf = bound.GetConformer()
    core.RemoveAllConformers()
    conf = Chem.Conformer(core.GetNumAtoms())
    conf.Set3D(True)
    for i, bound_idx in enumerate(bi):
        if bound_idx < 0 or bound_idx >= bound.GetNumAtoms():
            raise ValueError("Mapped-core bound atom index outside ligand")
        pos = bconf.GetAtomPosition(bound_idx)
        conf.SetAtomPosition(i, (float(pos.x), float(pos.y), float(pos.z)))
    core.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(core)
    return core


def uid(value):
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s

# Geometry extraction helpers retained from the supplied builder.
import copy, math, re
from pathlib import Path
from typing import Optional, List, Sequence, Tuple
import numpy as np
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
WATER_NAMES = {"HOH", "WAT", "DOD"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def base_uniprot(uniprot: str) -> str:
    return str(uniprot).split("-")[0].upper()

def listify(x) -> List:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def first_nonempty(d: dict, keys: Sequence[str]):
    for k in keys:
        if k in d:
            vals = listify(d[k])
            for v in vals:
                if v not in (None, "", ".", "?"):
                    return v
    return None

def safe_float(x) -> Optional[float]:
    try:
        if x in (None, "", ".", "?"):
            return None
        return float(x)
    except Exception:
        return None

def _split_chain_field(x: str) -> List[str]:
    if x is None:
        return []
    # Support comma-, semicolon-, or whitespace-separated chain IDs.
    vals = re.split(r"[,;\s]+", str(x).strip())
    return [v for v in vals if v and v not in {".", "?"}]

def get_uniprot_auth_chains(cif_path: Path, uniprot: str) -> List[str]:
    """
    Resolve UniProt accessions to author chain IDs from mmCIF metadata.

    Two mapping routes are supported:
    1. Match _struct_ref.pdbx_db_accession through _struct_ref_seq.ref_id.
    2. Use _struct_ref_seq.pdbx_db_accession directly when available.
    """
    d = MMCIF2Dict(str(cif_path))
    wanted = {uniprot.upper()}

    chains = set()

    ref_ids = listify(d.get("_struct_ref.id"))
    db_names = listify(d.get("_struct_ref.db_name"))
    accessions = listify(d.get("_struct_ref.pdbx_db_accession"))

    target_ref_ids = set()
    n = min(len(ref_ids), len(db_names), len(accessions))
    for i in range(n):
        dbn = str(db_names[i]).upper()
        acc = str(accessions[i]).upper()
        if dbn in {"UNP", "UNIPROT", "UNIPROTKB"} and (
            acc in wanted
        ):
            target_ref_ids.add(str(ref_ids[i]))

    seq_ref_ids = listify(d.get("_struct_ref_seq.ref_id"))
    seq_strands = listify(d.get("_struct_ref_seq.pdbx_strand_id"))
    n = min(len(seq_ref_ids), len(seq_strands))
    for i in range(n):
        if str(seq_ref_ids[i]) in target_ref_ids:
            chains.update(_split_chain_field(seq_strands[i]))

    direct_acc = listify(d.get("_struct_ref_seq.pdbx_db_accession"))
    direct_strand = listify(d.get("_struct_ref_seq.pdbx_strand_id"))
    n = min(len(direct_acc), len(direct_strand))
    for i in range(n):
        acc = str(direct_acc[i]).upper()
        if acc in wanted:
            chains.update(_split_chain_field(direct_strand[i]))

    return sorted(chains)

def get_entry_metadata(cif_path: Path) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    try:
        d = MMCIF2Dict(str(cif_path))
    except Exception:
        return None, None, None

    methods = [
        str(x).strip()
        for x in listify(d.get("_exptl.method"))
        if x not in (None, "", ".", "?")
    ]

    resolution_values = []
    for key in ("_refine.ls_d_res_high", "_reflns.d_resolution_high", "_em_3d_reconstruction.resolution"):
        for value in listify(d.get(key)):
            x = safe_float(value)
            if x is not None:
                resolution_values.append(x)
    resolution = min(resolution_values) if resolution_values else None

    rfree_values = []
    for value in listify(d.get("_refine.ls_R_factor_R_free")):
        x = safe_float(value)
        if x is not None:
            rfree_values.append(x)
    rfree = max(rfree_values) if rfree_values else None

    return (
        ";".join(sorted(set(methods))) if methods else None,
        resolution,
        rfree,
    )


def atom_is_hydrogen(atom) -> bool:
    e = str(getattr(atom, "element", "") or "").strip().upper()
    if e == "H":
        return True
    name = str(atom.get_name()).strip().upper()
    return name.startswith("H")

def heavy_atom_coords_from_chain(chain) -> np.ndarray:
    coords = []
    for residue in chain.get_residues():
        # Use only ATOM protein residues for pocket proximity,
        # excluding HETATM ligands from protein atom coordinates.
        if residue.id[0] != " ":
            continue
        for atom in residue.get_atoms():
            if not atom_is_hydrogen(atom):
                coords.append(np.asarray(atom.coord, dtype=float))
    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.vstack(coords)

def heavy_atoms_of_residue(residue):
    return [a for a in residue.get_atoms() if not atom_is_hydrogen(a)]

def min_distance_coords(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("inf")
    # Iterate over ligand atoms to avoid allocating a large pairwise distance matrix.
    best = float("inf")
    for xyz in b:
        d2 = np.sum((a - xyz[None, :]) ** 2, axis=1)
        x = float(np.min(d2))
        if x < best:
            best = x
    return float(math.sqrt(best))

def enumerate_bound_ligands_for_uniprot(
    cif_path: Path,
    pdb_id: str,
    uniprot: str,
    pocket_cutoff: float,
    min_ligand_heavy_atoms: int,
    fallback_auth_chains=None,
) -> Tuple[List[dict], Optional[str]]:
    """Enumerate observed ligands contacting the requested exact-UniProt chain.

    Chain localization uses mmCIF _struct_ref/_struct_ref_seq first. If those
    annotations are absent or do not resolve to a parsed author chain, exact-
    UniProt RCSB polymer-entity auth_asym_ids may rescue localization.
    """
    fallback_auth_chains = [
        str(x).strip() for x in (fallback_auth_chains or []) if str(x).strip()
    ]
    try:
        auth_chains = get_uniprot_auth_chains(cif_path, uniprot)
    except Exception as e:
        auth_chains = []
        mmcif_mapping_error = f"mmcif_mapping_parse_failed:{repr(e)}"
    else:
        mmcif_mapping_error = None

    parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    try:
        structure = parser.get_structure(pdb_id, str(cif_path))
    except Exception as e:
        return [], f"mmcif_parse_failed:{repr(e)}"

    model = next(structure.get_models())

    chain_by_id = {str(ch.id): ch for ch in model.get_chains()}
    target_chain_ids = [c for c in auth_chains if c in chain_by_id]
    chain_mapping_mode = "mmcif_struct_ref"

    if not target_chain_ids and fallback_auth_chains:
        target_chain_ids = [
            c for c in fallback_auth_chains if c in chain_by_id
        ]
        if target_chain_ids:
            chain_mapping_mode = "rcsb_polymer_entity_metadata"

    if not target_chain_ids:
        if mmcif_mapping_error and not fallback_auth_chains:
            return [], mmcif_mapping_error
        return [], (
            "uniprot_chain_mapping_failed:"
            f"mmcif={auth_chains},fallback={fallback_auth_chains},"
            f"available={sorted(chain_by_id)[:30]}"
        )

    target_chain_coords = {
        cid: heavy_atom_coords_from_chain(chain_by_id[cid])
        for cid in target_chain_ids
    }

    method, resolution, rfree = get_entry_metadata(cif_path)

    out = []
    seen = set()

    for lig_chain in model.get_chains():
        ligand_chain_id = str(lig_chain.id)

        for residue in lig_chain.get_residues():
            hetflag = str(residue.id[0])
            resname = str(residue.get_resname()).strip().upper()

            if hetflag == " ":
                continue
            if resname in WATER_NAMES or resname in COMMON_NON_DRUGLIKE:
                continue

            latoms = heavy_atoms_of_residue(residue)
            if len(latoms) < min_ligand_heavy_atoms:
                continue

            lig_coords = np.vstack(
                [np.asarray(a.coord, dtype=float) for a in latoms]
            )

            best_chain = None
            best_dist = float("inf")
            for target_chain_id, pcoords in target_chain_coords.items():
                dist = min_distance_coords(pcoords, lig_coords)
                if dist < best_dist:
                    best_dist = dist
                    best_chain = target_chain_id

            if best_chain is None or best_dist > pocket_cutoff:
                continue
            # PROTAC unbound protocol only requires a ligand genuinely bound to
            # the requested protein; PDB author chain IDs for HETATM ligands do
            # not have to equal the protein chain ID.

            auth_seq_id = residue.id[1]
            insertion_code = str(residue.id[2]).strip()

            key = (
                pdb_id,
                uniprot,
                best_chain,
                ligand_chain_id,
                str(auth_seq_id),
                insertion_code,
                resname,
            )
            if key in seen:
                continue
            seen.add(key)

            out.append({
                "protein_uniprot": uniprot,
                "pdb_id": pdb_id,
                "target_auth_chain": best_chain,
                "protein_chain_mapping_mode": chain_mapping_mode,
                "ligand_comp_id": resname,
                "ligand_auth_chain": ligand_chain_id,
                "ligand_auth_seq_id": auth_seq_id,
                "ligand_insertion_code": insertion_code,
                "protein_ligand_min_distance_A": best_dist,
                "ligand_heavy_atoms_in_model": len(latoms),
                "experimental_method": method,
                "resolution_A": resolution,
                "r_free": rfree,
                "raw_cif_path": str(cif_path.as_posix()),
            })

    return out, None

def extract_clean_protein_chain_to_pdb(
    cif_path: Path,
    auth_chain_id: str,
    out_path: Path,
) -> None:
    """
    Extract the specified target protein chain from an experimental mmCIF.

    Retain only standard ATOM protein residues, excluding water,
    ligands, and other HETATM components. Rename the output chain
    to A for compatibility with PDB parsers.
    """
    parser = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    structure = parser.get_structure("binary", str(cif_path))
    model = next(structure.get_models())

    target_chain = None
    for ch in model.get_chains():
        if str(ch.id) == str(auth_chain_id):
            target_chain = ch
            break
    if target_chain is None:
        raise RuntimeError(
            f"target chain {auth_chain_id} not found in {cif_path}"
        )

    ch = copy.deepcopy(target_chain)
    ch.detach_parent()

    for residue in list(ch.get_residues()):
        if residue.id[0] != " ":
            ch.detach_child(residue.id)

    ch.id = "A"

    s = Structure("binary")
    m = Model(0)
    s.add(m)
    m.add(ch)

    ensure_dir(out_path.parent)
    io = PDBIO()
    io.set_structure(s)
    io.save(str(out_path))

"""Metadata-first experimental binary library. Run from the LinkerDiff root."""
import argparse
import csv
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from Bio.PDB import MMCIFParser


SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL = "https://data.rcsb.org/graphql"
FIELDS = '''rcsb_id rcsb_entry_info { resolution_combined } exptl { method } refine { ls_R_factor_R_free }
polymer_entities { rcsb_polymer_entity_container_identifiers {
entity_id auth_asym_ids reference_sequence_identifiers { database_name database_accession } } }
nonpolymer_entities { pdbx_entity_nonpoly { comp_id }
nonpolymer_comp { rcsb_chem_comp_descriptor { SMILES SMILES_stereo } } }'''


def rcsb_auth_chains_for_uniprot(entry, uniprot):
    """Exact-UniProt author-chain fallback from RCSB polymer-entity metadata.

    This only localizes a chain inside an already exact-UniProt RCSB hit when
    mmCIF _struct_ref/_struct_ref_seq annotation is incomplete. It never
    broadens protein identity to homologues or related accessions.
    """
    wanted = str(uniprot).strip().upper()
    chains = set()
    for pe in entry.get("polymer_entities") or []:
        ids = pe.get("rcsb_polymer_entity_container_identifiers") or {}
        refs = ids.get("reference_sequence_identifiers") or []
        exact = any(
            str(ref.get("database_name", "")).upper() == "UNIPROT"
            and str(ref.get("database_accession", "")).upper() == wanted
            for ref in refs
        )
        if exact:
            for chain in ids.get("auth_asym_ids") or []:
                chain = str(chain).strip()
                if chain:
                    chains.add(chain)
    return sorted(chains)


_locks, _guard = {}, threading.Lock()

def lock(key):
    with _guard:
        return _locks.setdefault(str(key), threading.Lock())

def csv_out(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns) if columns is not None else pd.DataFrame(rows)
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)

def uniprot_entries(uniprot, root):
    p = root / "cache_v1" / "search" / (uniprot + ".json")
    if p.exists():
        return read_json(p)["pdb_ids"]
    payload = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
            "operator": "exact_match", "value": uniprot}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.nonpolymer_entity_count", "operator": "greater", "value": 0}}]},
        "return_type": "entry", "request_options": {"return_all_hits": True,
        "results_content_type": ["experimental"]}}
    obj = request(SEARCH, payload=payload)
    ids = sorted({r["identifier"].upper() for r in obj.get("result_set", [])})
    # Failures never reach this write and are therefore retried on the next run.
    write_json(p, {"version": VERSION, "status": "ok", "pdb_ids": ids})
    return ids

def metadata_batch(ids, root):
    query = "{ entries(entry_ids:" + json.dumps(ids) + ") { " + FIELDS + " } }"
    obj = request(GRAPHQL, payload={"query": query})
    if obj.get("errors"):
        raise ValueError("GraphQL errors: " + json.dumps(obj["errors"])[:500])
    entries = [e for e in obj.get("data", {}).get("entries", []) if e]
    returned = {e["rcsb_id"].upper() for e in entries}
    if returned != set(ids):
        raise ValueError("Incomplete GraphQL batch; missing: " + str(set(ids) - returned))
    for entry in entries:
        entry["_builder_metadata_version"] = VERSION
        write_json(root / "cache_v1" / "metadata" / (entry["rcsb_id"] + ".json"), entry)
    return entries

def looks_like_complete_cif(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size < 1000:
        return False
    with path.open("rb") as handle:
        head = handle.read(524288).lstrip()
        handle.seek(max(0, path.stat().st_size - 4096))
        tail = handle.read().rstrip()
    return head.startswith(b"data_") and b"_atom_site." in head and tail.endswith(b"#")

def get_cif(pdb, root, old_cache, force_download=False):
    out = root / "raw_mmcif" / (pdb.lower() + ".cif")
    with lock(out):
        if force_download and out.exists():
            out.unlink()
        if looks_like_complete_cif(out):
            return out
        if out.exists():
            out.unlink()
        out.parent.mkdir(parents=True, exist_ok=True)
        if old_cache:
            for folder in [Path(old_cache) / "raw_mmcif", Path(old_cache)]:
                for name in [pdb.lower() + ".cif", pdb.upper() + ".cif"]:
                    old = folder / name
                    if looks_like_complete_cif(old):
                        tmp = out.with_suffix(".copy.tmp")
                        shutil.copyfile(old, tmp)
                        os.replace(tmp, out)
                        return out
        raw = request(f"https://files.rcsb.org/download/{pdb}.cif", binary=True)
        if (not raw.lstrip().startswith(b"data_") or b"_atom_site." not in raw
                or not raw.rstrip().endswith(b"#")):
            raise ValueError("Invalid mmCIF response")
        write_bytes(out, raw)
    return out

def validate_instance(cif, row, bound):
    """The SDF heavy atoms must equal the observed instance in the SAME frame."""
    model = next(MMCIFParser(QUIET=True).get_structure("x", str(cif)).get_models())
    matches = [r for r in model[str(row["ligand_auth_chain"])].get_residues()
               if r.id[1] == int(row["ligand_auth_seq_id"])
               and str(r.id[2]).strip() == str(row["ligand_insertion_code"])
               and r.resname == row["ligand_comp_id"]]
    if len(matches) != 1:
        raise ValueError("Ambiguous ligand residue")
    atoms = [a for a in matches[0].get_atoms() if a.element not in ("H", "D")]
    if len(atoms) != bound.GetNumAtoms():
        raise ValueError("Missing/extra heavy atoms in experimental ligand")
    if any((a.occupancy or 0) < .5 for a in atoms):
        raise ValueError("Ligand occupancy below 0.5")
    xyz = np.asarray([a.coord for a in atoms])
    sdfxyz = bound.GetConformer().GetPositions()
    dist = np.linalg.norm(sdfxyz[:, None, :] - xyz[None, :, :], axis=2)
    indices = dist.argmin(1)
    if len(set(indices)) != len(indices) or dist[np.arange(len(indices)), indices].max() > .02:
        raise ValueError("SDF is not the observed ligand coordinate instance")
    for i, j in enumerate(indices):
        if bound.GetAtomWithIdx(i).GetSymbol().upper() != atoms[j].element.upper():
            raise ValueError("SDF/CIF element mismatch")

def pocket_check(
    protein,
    bound,
    bound_indices,
    *,
    pdb_id="",
    protein_auth_chain="",
):
    ca_coords, heavy_coords, residue_ids = [], [], set()
    for line in Path(protein).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not element:
            element = re.sub(r"[^A-Za-z]", "", line[12:16]).upper()[:1]
        if element in ("H", "D"):
            continue
        coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        heavy_coords.append(coord)
        residue_ids.add((line[21:22], line[22:26], line[26:27]))
        if line[12:16].strip() == "CA":
            ca_coords.append(coord)

    if len(residue_ids) < MIN_PROTEIN_RESIDUES_SANITY:
        raise ValueError(
            f"Protein construct has fewer than {MIN_PROTEIN_RESIDUES_SANITY} amino-acid residues"
        )
    protein_xyz = np.asarray(heavy_coords, dtype=float)
    ligand_xyz = bound.GetConformer().GetPositions()
    full_d = np.linalg.norm(
        ligand_xyz[:, None, :] - protein_xyz[None, :, :], axis=2
    )
    min_distance = float(full_d.min())
    # Do not apply the TernaryDB <2 A clash filter here.  These are observed
    # binary structures used by the PROTAC known-pocket unbound protocol; a
    # short contact can be chemically legitimate (e.g. covalent binding).
    # We record the minimum distance for audit and rely on exact-instance,
    # occupancy, chemistry and mapped-pocket checks for acceptance.

    full_contacts = full_d.min(1) < CONTACT_CUTOFF_A
    if int(full_contacts.sum()) < 3:
        raise ValueError("Fewer than 3 ligand heavy atoms contact protein within 4 A")

    mapped_xyz = ligand_xyz[bound_indices]
    mapped_d = np.linalg.norm(
        mapped_xyz[:, None, :] - protein_xyz[None, :, :], axis=2
    )
    mapped_contacts = mapped_d.min(1) < CONTACT_CUTOFF_A
    # This is stricter than the paper's whole-ligand contact criterion, but is
    # required here because these mapped atoms define the known pocket used to
    # align the unbound binary input to the PROTAC terminal.
    if (
        int(mapped_contacts.sum()) < 3
        or np.linalg.matrix_rank(
            mapped_xyz[mapped_contacts] - mapped_xyz[mapped_contacts].mean(0),
            tol=1e-3,
        ) < 2
    ):
        raise ValueError(
            "Fewer than 3 non-collinear mapped-core atoms contact protein within 4 A"
        )
    return {
        "protein_residue_count": int(len(residue_ids)),
        "protein_ca_count": int(len(ca_coords)),
        "mapped_pocket_atoms": int(mapped_contacts.sum()),
        "ligand_contact_atoms": int(full_contacts.sum()),
        "protein_ligand_min_distance_A": min_distance,
    }


def export_candidate(target, cand, root, old_cache, policy_id, preferred_max_resolution=3.5):
    cif = get_cif(cand["pdb_id"], root, old_cache)
    fallback_auth_chains = [
        x for x in str(cand.get("target_auth_chains", "")).split(";") if x
    ]
    rows, error = enumerate_bound_ligands_for_uniprot(
        cif,
        cand["pdb_id"],
        target["protein_uniprot"],
        CONTACT_CUTOFF_A,
        3,
        fallback_auth_chains=fallback_auth_chains,
    )
    if error and str(error).startswith("mmcif_"):
        cif = get_cif(cand["pdb_id"], root, old_cache, force_download=True)
        rows, error = enumerate_bound_ligands_for_uniprot(
            cif,
            cand["pdb_id"],
            target["protein_uniprot"],
            CONTACT_CUTOFF_A,
            3,
            fallback_auth_chains=fallback_auth_chains,
        )
    if error:
        raise ValueError(error)
    rows = [r for r in rows if r["ligand_comp_id"] == cand["comp_id"]]
    if not rows:
        raise ValueError(
            "Ligand does not contact the requested UniProt protein chain"
        )

    errors = []
    for row in sorted(
        rows,
        key=lambda x: (
            x["protein_ligand_min_distance_A"],
            x["target_auth_chain"],
        ),
    ):
        try:
            # DeepTernary's PROTAC known-pocket unbound protocol does not impose
            # the TernaryDB X-ray/resolution/R-free/same-chain filters.  Keep
            # these fields as provenance and ranking metadata, while the final
            # acceptance gate is based on target identity, observed-instance
            # integrity, chemistry/atom mapping and pocket geometry.
            method = str(row.get("experimental_method") or "")
            resolution = safe_float(row.get("resolution_A"))
            rfree = safe_float(row.get("r_free"))
            row["entry_quality_rank"] = entry_quality_rank(
                method, resolution, rfree, preferred_max_resolution
            )
            cid = digest([
                target["pair_id"],
                cand["pdb_id"],
                cand["comp_id"],
                row["target_auth_chain"],
                row["ligand_auth_chain"],
                row["ligand_auth_seq_id"],
                row.get("ligand_insertion_code", ""),
            ])[:20]
            dest = root / "binaries" / target["pair_id"] / cid
            dest.mkdir(parents=True, exist_ok=True)
            lig_path = dest / "bound_ligand.sdf"
            core_path = dest / "mapped_core.sdf"
            protein_path = dest / "protein.pdb"
            instance_key = digest([
                cand["pdb_id"],
                row["ligand_auth_chain"],
                row["ligand_auth_seq_id"],
                row.get("ligand_insertion_code", ""),
            ])
            cached_sdf = root / "cache_v1" / "instances" / (instance_key + ".sdf")
            with lock(cached_sdf):
                bound = None
                if cached_sdf.exists():
                    try:
                        bound, sdf_parse_mode = read_sdf_with_repair(
                            cached_sdf, cand["smiles"]
                        )
                    except Exception:
                        cached_sdf.unlink()
                if bound is None:
                    raw = request(
                        f"https://models.rcsb.org/v1/{cand['pdb_id'].lower()}/ligand",
                        params={
                            "auth_asym_id": row["ligand_auth_chain"],
                            "auth_seq_id": str(row["ligand_auth_seq_id"]),
                            "encoding": "sdf",
                        },
                        binary=True,
                    )
                    write_bytes(cached_sdf, raw)
                    bound, sdf_parse_mode = read_sdf_with_repair(
                        cached_sdf, cand["smiles"]
                    )

            validate_instance(cif, row, bound)
            # Re-screen the actual observed coordinate-bearing ligand. Catalogue
            # descriptors are never trusted as the final chemistry identity.
            actual_screen = chemistry_screen_with_timeout_retry(
                target["terminal_smiles_core"],
                canonical(bound),
                target.get("terminal_smiles_mapped"),
            )
            if not actual_screen:
                raise ValueError("Actual ligand chemistry is outside candidate policy")
            if (
                cand.get("match_type") == "exact"
                and actual_screen.get("match_type") != "exact"
            ):
                raise ValueError(
                    "Observed ligand no longer preserves the catalogue exact match"
                )

            ti, bi, map_mode, mapping_stereo_relation = map_to_bound(
                target["terminal_smiles_core"],
                bound,
                actual_screen,
                target.get("terminal_smiles_mapped"),
            )
            mapped_core = mapped_core_from_target_with_bound_coords(
                target["terminal_smiles_core"],
                target.get("terminal_smiles_mapped"),
                ti,
                bound,
                bi,
            )
            extract_clean_protein_chain_to_pdb(
                cif, row["target_auth_chain"], protein_path
            )
            contact_stats = pocket_check(
                protein_path,
                bound,
                bi,
                pdb_id=cand["pdb_id"],
                protein_auth_chain=row["target_auth_chain"],
            )
            write_sdf(lig_path, bound)
            # DeepTernary pocket alignment consumes only the explicitly audited
            # target-corresponding atoms for similar ligands.
            write_sdf(core_path, mapped_core)

            screen_keys = (
                "match_type",
                "match_detail",
                "screen_rank",
                "similarity",
                "target_coverage",
                "candidate_coverage",
                "core_heavy_atoms",
                "mcs_smarts",
                "chemistry_equivalence",
                "stereo_relation",
                "target_core_indices_std",
                "candidate_core_indices_std",
            )
            catalog_screen = {
                "catalog_" + k: cand.get(k, "") for k in screen_keys
            }
            observed_screen = {
                "observed_" + k: actual_screen.get(k, "") for k in screen_keys
            }
            metadata_path = dest / "metadata.json"
            meta = {
                **target,
                **cand,
                **row,
                **catalog_screen,
                **observed_screen,
                **actual_screen,
                "candidate_id": cid,
                "policy_id": policy_id,
                "target_indices": ti,
                "bound_indices": bi,
                "sdf_parse_mode": sdf_parse_mode,
                "atom_map_mode": map_mode,
                "mapping_stereo_relation": mapping_stereo_relation,
                **contact_stats,
                "protein_path": str(protein_path.resolve()),
                "bound_ligand_path": str(lig_path.resolve()),
                "mapped_core_path": str(core_path.resolve()),
                "protein_sha256": file_hash(protein_path),
                "ligand_sha256": file_hash(lig_path),
                "mapped_core_sha256": file_hash(core_path),
                "metadata_path": str(metadata_path.resolve()),
            }
            write_json(metadata_path, meta)
            return meta
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("; ".join(errors)[:1600])

FILE_HASH_KEYS = (
    ("protein_path", "protein_sha256"),
    ("bound_ligand_path", "ligand_sha256"),
    ("mapped_core_path", "mapped_core_sha256"),
)

def metadata_unbound_source_ok(meta):
    """Minimal provenance check for a cached PROTAC unbound binary.

    Experimental method/resolution/R-free are intentionally not hard gates here;
    the DeepTernary PROTAC protocol requires a valid unbound binary with a
    reliably aligned anchor/warhead pocket, not TernaryDB training-set QC.
    """
    return bool(
        str(meta.get("protein_uniprot", "")).strip()
        and str(meta.get("pdb_id", "")).strip()
        and str(meta.get("match_type", "")).strip()
    )


def selected_record_files_ok(meta):
    """A selected record is reusable only when every payload hash still agrees."""
    try:
        return all(
            meta.get(path_key) and meta.get(hash_key)
            and Path(meta[path_key]).is_file()
            and file_hash(meta[path_key]) == meta[hash_key]
            for path_key, hash_key in FILE_HASH_KEYS
        )
    except (OSError, TypeError, ValueError):
        return False

def persist_selected(done_path, meta):
    """Keep pair cache and candidate metadata byte-for-byte consistent."""
    if meta.get("metadata_path"):
        write_json(meta["metadata_path"], meta)
    write_json(done_path, meta)

def _selected_json_records(base):
    base = Path(base)
    records = []
    aggregate = base / "selected_binaries.json"
    if aggregate.is_file():
        try:
            obj = read_json(aggregate)
            if isinstance(obj, list):
                records.extend(x for x in obj if isinstance(x, dict))
        except Exception:
            pass
    selected_dir = base / "cache_v1" / "selected"
    if selected_dir.is_dir():
        for path in sorted(selected_dir.glob("*.json")):
            try:
                obj = read_json(path)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                pass
    return records

def load_validated_donors(root, old_cache):
    """Index already validated exact-match experimental binaries.

    Reuse is only a discovery/caching optimization.  The requested target is
    re-screened, remapped, pocket-checked and re-hashed before acceptance.
    Older records may be reused only as coordinate donors; all target-specific
    chemistry, mapping, unbound-source and pocket checks are repeated.
    """
    sources = [Path(root)]
    if old_cache:
        old = Path(old_cache).resolve()
        if old != Path(root).resolve():
            sources.append(old)
    donors, seen = {}, set()
    for base in sources:
        for meta in _selected_json_records(base):
            if meta.get("match_type") != "exact":
                continue
            if not selected_record_files_ok(meta):
                continue
            if not metadata_unbound_source_ok(meta):
                continue
            key = (
                str(meta.get("protein_uniprot", "")).upper(),
                meta.get("protein_sha256", ""),
                meta.get("ligand_sha256", ""),
            )
            if not key[0] or key in seen:
                continue
            try:
                bound = read_sdf(meta["bound_ligand_path"])
                observed_smiles = canonical(bound)
            except Exception:
                continue
            seen.add(key)
            donor = dict(meta)
            donor["_observed_smiles"] = observed_smiles
            donors.setdefault(key[0], []).append(donor)
    for values in donors.values():
        values.sort(key=lambda m: (
            entry_quality_rank(
                m.get("experimental_method") or m.get("method"),
                m.get("resolution_A"), m.get("r_free")
            ),
            float(m.get("resolution_A") or float("inf")),
            str(m.get("pdb_id", "")),
            str(m.get("candidate_id", "")),
        ))
    return donors


def export_reused_binary(target, donor, root, policy_id):
    """Materialize a same-UniProt exact-ligand pose for a new target pair."""
    if str(donor.get("protein_uniprot", "")).upper() != target["protein_uniprot"]:
        raise ValueError("Validated donor UniProt mismatch")
    if not selected_record_files_ok(donor):
        raise ValueError("Validated donor payload hash mismatch")
    if not metadata_unbound_source_ok(donor):
        raise ValueError("Validated donor lacks required unbound-source provenance")
    donor_uniprots = {
        x for x in str(donor.get("entry_uniprots", "")).split(";") if x
    }
    forbidden = set(target.get("forbidden_context_partner_uniprots", []))
    leaked = sorted(donor_uniprots & forbidden)
    if leaked:
        raise ValueError(
            "Source PDB contains context partner protein(s), not an unbound "
            "binary template: " + ",".join(leaked)
        )

    actual_screen = chemistry_screen_with_timeout_retry(
        target["terminal_smiles_core"],
        donor["_observed_smiles"],
        target.get("terminal_smiles_mapped"),
    )
    if not actual_screen or actual_screen["match_type"] != "exact":
        raise ValueError("Validated donor is not an exact match for the requested target")

    bound = read_sdf(donor["bound_ligand_path"])
    ti, bi, map_mode, mapping_stereo_relation = map_to_bound(
        target["terminal_smiles_core"],
        bound,
        actual_screen,
        target.get("terminal_smiles_mapped"),
    )
    cid = digest([
        target["pair_id"],
        "validated_binary_reuse",
        donor.get("pair_id", ""),
        donor.get("candidate_id", ""),
        donor.get("protein_sha256", ""),
        donor.get("ligand_sha256", ""),
    ])[:20]
    dest = root / "binaries" / target["pair_id"] / cid
    protein_path = dest / "protein.pdb"
    lig_path = dest / "bound_ligand.sdf"
    core_path = dest / "mapped_core.sdf"
    metadata_path = dest / "metadata.json"
    mapped_core = mapped_core_from_target_with_bound_coords(
        target["terminal_smiles_core"],
        target.get("terminal_smiles_mapped"),
        ti,
        bound,
        bi,
    )
    atomic_copy(donor["protein_path"], protein_path)
    write_sdf(lig_path, bound)
    write_sdf(core_path, mapped_core)
    contact_stats = pocket_check(
        protein_path,
        bound,
        bi,
        pdb_id=donor.get("pdb_id", ""),
        protein_auth_chain=donor.get("target_auth_chain", ""),
    )

    provenance_keys = (
        "pdb_id", "comp_id", "entry_uniprots", "smiles", "resolution_A",
        "r_free", "method", "target_auth_chain", "ligand_comp_id",
        "ligand_auth_chain", "ligand_auth_seq_id", "ligand_insertion_code",
        "protein_chain_mapping_mode", "ligand_heavy_atoms_in_model",
        "experimental_method", "raw_cif_path",
        "sdf_parse_mode",
    )
    provenance = {k: donor.get(k, "") for k in provenance_keys}
    screen_keys = (
        "match_type", "match_detail", "screen_rank", "similarity",
        "target_coverage", "candidate_coverage", "core_heavy_atoms",
        "mcs_smarts", "chemistry_equivalence", "stereo_relation",
        "target_core_indices_std", "candidate_core_indices_std",
    )
    catalog_screen = {
        "catalog_" + k: donor.get("catalog_" + k, donor.get(k, ""))
        for k in screen_keys
    }
    observed_screen = {
        "observed_" + k: actual_screen.get(k, "") for k in screen_keys
    }
    meta = {
        **target,
        **provenance,
        **catalog_screen,
        **observed_screen,
        **actual_screen,
        "candidate_id": cid,
        "policy_id": policy_id,
        "target_indices": ti,
        "bound_indices": bi,
        "atom_map_mode": map_mode,
        "mapping_stereo_relation": mapping_stereo_relation,
        **contact_stats,
        "protein_path": str(protein_path.resolve()),
        "bound_ligand_path": str(lig_path.resolve()),
        "mapped_core_path": str(core_path.resolve()),
        "protein_sha256": file_hash(protein_path),
        "ligand_sha256": file_hash(lig_path),
        "mapped_core_sha256": file_hash(core_path),
        "metadata_path": str(metadata_path.resolve()),
        "selection_mode": "validated_binary_reuse",
        "accepted_automatically": True,
        "reused_from_pair_id": donor.get("pair_id", ""),
        "reused_from_candidate_id": donor.get("candidate_id", ""),
    }
    write_json(metadata_path, meta)
    return meta

def make_targets(df):
    needed = ["Uniprot", "e3_ligase_Uniprot", "warhead_smiles", "e3_ligand_smiles", "split"]
    if any(k not in df for k in needed):
        raise ValueError("Input CSV requires: " + ", ".join(needed))

    def clean_uniprot(value):
        u = str(value).strip().upper()
        if not u or u == "NAN" or any(c in u for c in ",; /\\"):
            raise ValueError("Missing or multiple UniProt identifiers")
        return u

    targets, contexts = {}, []
    for i, row in df.iterrows():
        context = {"context_row": i, "split": str(row["split"])}
        for key in ("uuid", "protac_id", "molecule"):
            if key in df.columns:
                context[key] = uid(row[key])

        cleaned = {}
        for side, col in (("POI", "Uniprot"), ("E3", "e3_ligase_Uniprot")):
            try:
                cleaned[side] = clean_uniprot(row[col])
            except Exception as exc:
                cleaned[side] = ""
                context[side.lower() + "_error"] = str(exc)

        for side, ucol, scol, other_side in [
            ("POI", "Uniprot", "warhead_smiles", "E3"),
            ("E3", "e3_ligase_Uniprot", "e3_ligand_smiles", "POI"),
        ]:
            key = side.lower() + "_pair_id"
            try:
                u = cleaned[side]
                if not u:
                    raise ValueError("Missing or invalid UniProt identifier")
                smi, mapped_smi, has_exit = terminal_with_exit(row[scol])
                pid = pair_id(side, u, smi, mapped_smi)
                target = targets.setdefault(pid, {
                    "pair_id": pid,
                    "side": side,
                    "protein_uniprot": u,
                    "terminal_smiles_core": smi,
                    "terminal_smiles_mapped": mapped_smi,
                    "has_linker_exit_atom": has_exit,
                    "forbidden_context_partner_uniprots": set(),
                })
                partner = cleaned.get(other_side, "")
                if partner and partner != u:
                    target["forbidden_context_partner_uniprots"].add(partner)
                context[key] = pid
            except Exception as exc:
                context[key] = ""
                context[side.lower() + "_error"] = str(exc)
        contexts.append(context)

    out_targets = []
    for target in targets.values():
        t = dict(target)
        t["forbidden_context_partner_uniprots"] = sorted(
            target.get("forbidden_context_partner_uniprots", set())
        )
        out_targets.append(t)
    return out_targets, contexts


def make_rl2_manifest(contexts, selected):
    """Return only contexts for which both coordinate-bearing binaries exist."""
    by_pair = {m["pair_id"]: m for m in selected}
    fields = (
        "pair_id", "match_type", "match_detail", "protein_uniprot", "pdb_id", "comp_id",
        "protein_path", "bound_ligand_path", "mapped_core_path",
        "protein_sha256", "ligand_sha256", "mapped_core_sha256",
        "target_indices", "bound_indices", "selection_mode", "stereo_relation",
        "target_coverage", "candidate_coverage", "core_heavy_atoms",
        "resolution_A", "r_free", "protein_residue_count", "protein_ca_count",
    )
    manifest = []
    for context in contexts:
        poi = by_pair.get(context.get("poi_pair_id"))
        e3 = by_pair.get(context.get("e3_pair_id"))
        if not poi or not e3:
            continue
        row = dict(context)
        for prefix, meta in (("poi", poi), ("e3", e3)):
            for field in fields:
                row[f"{prefix}_{field}"] = meta.get(field, "")
        row["coordinate_pair_group_id"] = digest([
            poi.get("protein_sha256", ""), poi.get("ligand_sha256", ""),
            e3.get("protein_sha256", ""), e3.get("ligand_sha256", ""),
        ])[:20]
        manifest.append(row)
    group_sizes = {}
    for row in manifest:
        key = row["coordinate_pair_group_id"]
        group_sizes[key] = group_sizes.get(key, 0) + 1
    for row in manifest:
        size = group_sizes[row["coordinate_pair_group_id"]]
        row["coordinate_pair_group_size"] = size
        row["inverse_coordinate_pair_frequency"] = 1.0 / size
    return manifest

def rl2_readiness_report(contexts, manifest, selected, args):
    """Quantify sample count separately from independent structural diversity."""
    complete_by_split = {}
    total_by_split = {}
    for row in contexts:
        split = str(row.get("split", ""))
        total_by_split[split] = total_by_split.get(split, 0) + 1
    for row in manifest:
        split = str(row.get("split", ""))
        complete_by_split[split] = complete_by_split.get(split, 0) + 1

    def e3_proteins(splits):
        return sorted({
            str(row.get("e3_protein_uniprot", ""))
            for row in manifest if str(row.get("split", "")) in splits
            and row.get("e3_protein_uniprot")
        })

    train_e3 = e3_proteins({"train"})
    eval_e3 = e3_proteins({"val", "test"})
    checks = {
        "complete_contexts": len(manifest) >= args.min_rl2_contexts,
        "validation_contexts_present": complete_by_split.get("val", 0) > 0,
        "test_contexts_present": complete_by_split.get("test", 0) > 0,
        "train_e3_protein_diversity": len(train_e3) >= args.min_train_e3_proteins,
        "evaluation_e3_protein_diversity": len(eval_e3) >= args.min_eval_e3_proteins,
    }
    unique_payloads = {
        (m.get("protein_sha256", ""), m.get("ligand_sha256", "")) for m in selected
    }
    report = {
        "version": VERSION,
        "ready": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_rl2_contexts": args.min_rl2_contexts,
            "min_train_e3_proteins": args.min_train_e3_proteins,
            "min_eval_e3_proteins": args.min_eval_e3_proteins,
        },
        "contexts": {
            "complete": len(manifest), "total": len(contexts),
            "fraction": len(manifest) / len(contexts) if contexts else 0.0,
            "complete_by_split": complete_by_split,
            "total_by_split": total_by_split,
        },
        "binary_records": len(selected),
        "independent_coordinate_payloads": len(unique_payloads),
        "independent_coordinate_pairs": len({
            row.get("coordinate_pair_group_id", "") for row in manifest
            if row.get("coordinate_pair_group_id")
        }),
        "unique_pdb_entries": len({m.get("pdb_id", "") for m in selected if m.get("pdb_id")}),
        "unique_proteins": len({m.get("protein_uniprot", "") for m in selected}),
        "train_e3_proteins": train_e3,
        "evaluation_e3_proteins": eval_e3,
        "match_type_counts": {
            match_type: sum(m.get("match_type") == match_type for m in selected)
            for match_type in ("exact", "similar_rooted_mcs")
        },
        "interpretation": (
            "Complete contexts are training examples; independent_coordinate_payloads measures "
            "structural diversity. Reused exact binaries increase context coverage but not the "
            "number of independent experimental poses. rl2_manifest.csv therefore also reports "
            "coordinate-pair group size and inverse frequency for optional balanced sampling."
        ),
    }
    return report

def build(args):
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets, contexts = make_targets(pd.read_csv(args.input_csv))
    policy = {
        "version": VERSION,
        "deepternary_protac_unbound_alignment": {
            "protocol": "known-pocket unbound PROTAC inference",
            "protein_inputs": (
                "mono/unbound E3 and POI structures carrying a bound ligand "
                "corresponding to or similar to the PROTAC anchor/warhead"
            ),
            "same_target_operationalization": (
                "exact UniProt accession; no homolog/related-accession substitution"
            ),
            "chain_localization": (
                "mmCIF struct_ref/struct_ref_seq first; exact-UniProt RCSB "
                "polymer-entity auth_asym_ids fallback only when needed"
            ),
            "atom_index_alignment_required": True,
            "unbound_pocket_coordinates_used": True,
            "bound_ligand_may_be_similar_not_exact": True,
            "TernaryDB_training_filters_are_not_hard_binary_filters": [
                "X-ray only", "resolution <=3.5 A", "R-free <=0.26",
                "ligand/protein same author chain ID",
                "TernaryDB peptide-length criterion",
                "TernaryDB <2 A clash exclusion",
            ],
        },
        "similar_ligand_operationalization": {
            "method": "exit-aware rooted MCS",
            "target_coverage_min": SIMILAR_TARGET_COVERAGE_MIN,
            "core_heavy_atoms_min": SIMILAR_CORE_HEAVY_ATOMS_MIN,
            "candidate_coverage_is_hard_filter": False,
            "tanimoto_is_hard_filter": False,
            "tanimoto_use": "ranking only",
            "require_exit_atom": True,
            "require_exit_neighbour_bond": True,
            "ring_policy": "ring-to-ring and complete rings only",
        },
        "reliability_audits": {
            "minimum_protein_residues_sanity": MIN_PROTEIN_RESIDUES_SANITY,
            "minimum_ligand_contact_atoms_within_4A": 3,
            "occupancy_min": 0.5,
            "sdf_mmcif_instance_coordinate_tolerance_A": 0.02,
            "mapped_core_contact_requirement": (
                ">=3 non-collinear mapped-core atoms within 4 A of protein"
            ),
            "mapped_core_representation": (
                "PROTAC-terminal/common-core topology with coordinates copied "
                "through explicit target_indices<->bound_indices mapping from "
                "the experimental bound ligand"
            ),
            "insertion_coded_ligands": (
                "attempted; accepted only if fetched SDF is uniquely verified "
                "against the exact mmCIF residue instance in the same frame"
            ),
            "stereo": (
                "target-specified stereo mandatory; experimental ligand may "
                "resolve target-unspecified stereo but explicit conflicts reject"
            ),
            "structure_quality_metadata": (
                "method/resolution/R-free recorded and used for candidate ranking only"
            ),
            "preferred_resolution_for_ranking_A": args.max_resolution,
            "preferred_rfree_for_ranking": PREFERRED_MAX_RFREE,
        },
        "chemical_standardization": (
            "cleanup+fragment-parent(candidate)+uncharger+canonical-tautomer"
        ),
        "sdf_repair": (
            "strict sanitized SDF, then metadata-template topology repair preserving coordinates"
        ),
        "fingerprint": "Morgan-r2-2048-chiral; ranking only for similar candidates",
        "max_pdb_per_target": args.max_pdb_per_target,
        "selection": (
            "first structurally valid same-UniProt candidate; exact matches ranked "
            "before similar rooted-MCS matches; experimental quality breaks ties"
        ),
        "validated_binary_reuse": not args.disable_validated_reuse,
        "exact_only_diagnostic": bool(args.exact_only),
        "protein_ligand_contact_cutoff_A": CONTACT_CUTOFF_A,
        "source_sha256": file_hash(args.input_csv),
    }
    policy_id = digest(policy)
    write_json(root / "policy.json", policy)
    csv_out(root / "search_targets.csv", targets)
    by_u, failures = {}, []
    accessions = sorted({t["protein_uniprot"] for t in targets})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(uniprot_entries, u, root): u for u in accessions}
        for n, future in enumerate(as_completed(futures), 1):
            u = futures[future]
            try:
                by_u[u] = future.result()
            except Exception as exc:
                failures.append({"stage": "search", "key": u, "error": str(exc)})
            print(f"[search] {n}/{len(accessions)} {u}: {len(by_u.get(u, []))} entries", flush=True)
    ids = sorted({p for ps in by_u.values() for p in ps})
    missing = []
    for pdb_id in ids:
        mp = root / "cache_v1" / "metadata" / (pdb_id + ".json")
        if not mp.exists():
            missing.append(pdb_id)
            continue
        try:
            cached_meta = read_json(mp)
        except Exception:
            missing.append(pdb_id)
            continue
        if cached_meta.get("_builder_metadata_version") != VERSION:
            missing.append(pdb_id)
    batches = [missing[i:i + args.batch_size] for i in range(0, len(missing), args.batch_size)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(metadata_batch, b, root): b for b in batches}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                failures.append({"stage": "metadata", "key": ",".join(futures[future]), "error": str(exc)})
            print(f"[metadata] {n}/{len(batches)} batches", flush=True)
    catalogs = {u: [] for u in accessions}
    catalog_exclusions, seen_exclusions = [], set()
    entry_quality_metadata = []
    wanted_uniprots = set(catalogs)
    for pdb in ids:
        path = root / "cache_v1" / "metadata" / (pdb + ".json")
        if not path.exists():
            continue
        e = read_json(path)
        uniprots = {
            ref["database_accession"]
            for pe in e.get("polymer_entities") or []
            for ref in pe["rcsb_polymer_entity_container_identifiers"].get(
                "reference_sequence_identifiers"
            ) or []
            if ref["database_name"] == "UniProt"
        }
        relevant_uniprots = sorted(uniprots & wanted_uniprots)
        if not relevant_uniprots:
            continue

        auth_chains_by_uniprot = {
            u: rcsb_auth_chains_for_uniprot(e, u) for u in relevant_uniprots
        }

        res = (e.get("rcsb_entry_info") or {}).get("resolution_combined") or []
        numeric_res = [safe_float(x) for x in res]
        numeric_res = [x for x in numeric_res if x is not None]
        resolution = min(numeric_res) if numeric_res else None
        methods = [str(x.get("method", "")).upper() for x in e.get("exptl") or []]
        refine_rows = listify(e.get("refine"))
        rfree_values = [safe_float(x.get("ls_R_factor_R_free")) for x in refine_rows if isinstance(x, dict)]
        rfree_values = [x for x in rfree_values if x is not None]
        # Multiple refinement records are rare. Using the maximum is conservative:
        # every reported R-free must satisfy the accepted entry threshold.
        rfree = max(rfree_values) if rfree_values else None

        # PROTAC unbound-input selection: structure-quality fields are ranking
        # metadata only.  They are not TernaryDB-style hard exclusion rules.
        quality_rank = entry_quality_rank(
            ";".join(methods), resolution, rfree, args.max_resolution
        )
        entry_quality_metadata.append({
            "pdb_id": pdb,
            "protein_uniprots": ";".join(relevant_uniprots),
            "experimental_methods": ";".join(methods),
            "resolution_A": "" if resolution is None else resolution,
            "r_free": "" if rfree is None else rfree,
            "entry_quality_rank": quality_rank,
            "hard_filtered": False,
        })

        for lig in e.get("nonpolymer_entities") or []:
            desc = (lig.get("nonpolymer_comp") or {}).get(
                "rcsb_chem_comp_descriptor"
            ) or {}
            smi = desc.get("SMILES_stereo") or desc.get("SMILES")
            if not smi:
                continue
            comp_id = lig["pdbx_entity_nonpoly"]["comp_id"]
            exclusion = catalog_exclusion_reason(comp_id, smi)
            for u in relevant_uniprots:
                if exclusion:
                    exclusion_key = (u, pdb, comp_id, exclusion)
                    if exclusion_key not in seen_exclusions:
                        seen_exclusions.add(exclusion_key)
                        catalog_exclusions.append({
                            "protein_uniprot": u,
                            "pdb_id": pdb,
                            "comp_id": comp_id,
                            "reason_code": exclusion,
                        })
                    continue
                catalogs[u].append({
                    "pdb_id": pdb,
                    "comp_id": comp_id,
                    "entry_uniprots": ";".join(sorted(uniprots)),
                    "target_auth_chains": ";".join(
                        auth_chains_by_uniprot.get(u, [])
                    ),
                    "smiles": smi,
                    "resolution_A": resolution,
                    "r_free": rfree,
                    "method": ";".join(methods),
                    "entry_quality_rank": quality_rank,
                })
    csv_out(
        root / "catalog_exclusions.csv",
        catalog_exclusions,
        columns=["protein_uniprot", "pdb_id", "comp_id", "reason_code"],
    )
    csv_out(
        root / "entry_quality_metadata.csv",
        entry_quality_metadata,
        columns=[
            "pdb_id", "protein_uniprots", "experimental_methods",
            "resolution_A", "r_free", "entry_quality_rank", "hard_filtered",
        ],
    )
    donors = (
        {}
        if args.disable_validated_reuse
        else load_validated_donors(root, args.reuse_cache)
    )
    print(
        f"[reuse] validated exact coordinate donors: {sum(map(len, donors.values()))}",
        flush=True,
    )
    selected, outcomes, candidate_attempts, screen_failures = [], [], [], []

    def target_job(t):
        done = root / "cache_v1" / "selected" / (t["pair_id"] + ".json")
        if done.exists():
            m = read_json(done)
            if (
                selected_record_files_ok(m)
                and metadata_unbound_source_ok(m)
                and m.get("match_type") in {"exact", "similar_rooted_mcs"}
                and m.get("policy_id") == policy_id
                and not args.revalidate_existing
            ):
                return m, [], {
                    "pair_id": t["pair_id"],
                    "status": "usable",
                    "reused_existing": True,
                    "selected_match_type": m.get("match_type", ""),
                }

        local_failures = []
        # Reuse is limited to hash-verified exact experimental binaries from
        # the same UniProt. Chemistry, atom mapping, unbound-source and pocket
        # geometry are recomputed for the requested target before acceptance.
        for donor in donors.get(t["protein_uniprot"], []):
            try:
                preliminary = chemistry_screen_with_timeout_retry(
                    t["terminal_smiles_core"],
                    donor["_observed_smiles"],
                    t.get("terminal_smiles_mapped"),
                )
            except Exception:
                continue
            if not preliminary or preliminary["match_type"] != "exact":
                continue
            try:
                m = export_reused_binary(
                    t,
                    donor,
                    root,
                    policy_id,
                )
                m["selection_mode"] = "validated_binary_reuse"
                m["accepted_automatically"] = True
                m["reused_from_pair_id"] = donor.get("pair_id", "")
                m["reused_from_candidate_id"] = donor.get("candidate_id", "")
                persist_selected(done, m)
                return m, local_failures, {
                    "pair_id": t["pair_id"],
                    "status": "usable_reused_binary",
                    "selected_match_type": "exact",
                    "attempted_candidates": 0,
                    "reused_from_pair_id": donor.get("pair_id", ""),
                }
            except Exception as exc:
                local_failures.append({
                    "pair_id": t["pair_id"],
                    "side": t["side"],
                    "protein_uniprot": t["protein_uniprot"],
                    "pdb_id": donor.get("pdb_id", ""),
                    "comp_id": donor.get("comp_id", ""),
                    "match_type": "exact",
                    "similarity": 1.0,
                    "stage": "validated_binary_reuse",
                    "reason_code": classify_reason(exc),
                    "error": str(exc),
                })

        if t["protein_uniprot"] not in by_u:
            return None, local_failures, {
                "pair_id": t["pair_id"], "status": "search_failed"
            }
        if not by_u[t["protein_uniprot"]]:
            return None, local_failures, {
                "pair_id": t["pair_id"],
                "status": "no_pdb_for_uniprot",
                "attempted_candidates": 0,
            }

        candidates, policy_rejections = [], 0
        forbidden_partners = set(t.get("forbidden_context_partner_uniprots", []))
        for c in catalogs[t["protein_uniprot"]]:
            try:
                # DeepTernary's PROTAC benchmark uses unbound/binary templates
                # from other PDB entries, not the corresponding ternary complex.
                # If this source entry already contains a context partner protein,
                # do not use it as the known-pocket binary source for that target.
                entry_uniprots = {
                    x for x in str(c.get("entry_uniprots", "")).split(";") if x
                }
                leaked_partners = sorted(entry_uniprots & forbidden_partners)
                if leaked_partners:
                    raise ValueError(
                        "Source PDB contains context partner protein(s), not an unbound "
                        "binary template: " + ",".join(leaked_partners)
                    )
                screen = chemistry_screen_with_timeout_retry(
                    t["terminal_smiles_core"],
                    c["smiles"],
                    t.get("terminal_smiles_mapped"),
                )
                if screen:
                    candidates.append({**c, **screen})
                else:
                    policy_rejections += 1
            except Exception as exc:
                local_failures.append({
                    "pair_id": t["pair_id"],
                    "side": t["side"],
                    "protein_uniprot": t["protein_uniprot"],
                    "pdb_id": c.get("pdb_id", ""),
                    "comp_id": c.get("comp_id", ""),
                    "stage": "chemistry_screen",
                    "reason_code": classify_reason(exc),
                    "error": str(exc),
                })

        candidates.sort(key=lambda c: (
            c.get("screen_rank", 99),
            -float(c.get("target_coverage", 0.0)),
            -float(c.get("similarity", 0.0)),
            int(c.get("entry_quality_rank", 4)),
            float(c.get("resolution_A") or float("inf")),
            float(c.get("r_free") or float("inf")),
            c["pdb_id"],
            c["comp_id"],
        ))
        if args.exact_only:
            candidates = [c for c in candidates if c["match_type"] == "exact"]

        attempted, seen, errors = 0, set(), []
        for c in candidates:
            key = (c["pdb_id"], c["comp_id"])
            if key in seen:
                continue
            if (
                args.max_pdb_per_target > 0
                and attempted >= args.max_pdb_per_target
            ):
                break
            seen.add(key)
            attempted += 1
            try:
                m = export_candidate(
                    t,
                    c,
                    root,
                    args.reuse_cache,
                    policy_id,
                    preferred_max_resolution=args.max_resolution,
                )
                m["selection_mode"] = m.get("match_type", c["match_type"])
                m["accepted_automatically"] = True
                persist_selected(done, m)
                return m, local_failures, {
                    "pair_id": t["pair_id"],
                    "status": "usable",
                    "selected_match_type": m["match_type"],
                    "attempted_candidates": attempted,
                }
            except Exception as exc:
                msg = str(exc)
                errors.append(c["pdb_id"] + "/" + c["comp_id"] + ":" + msg)
                local_failures.append({
                    "pair_id": t["pair_id"],
                    "side": t["side"],
                    "protein_uniprot": t["protein_uniprot"],
                    "pdb_id": c["pdb_id"],
                    "comp_id": c["comp_id"],
                    "match_type": c.get("match_type", ""),
                    "similarity": c.get("similarity", ""),
                    "target_coverage": c.get("target_coverage", ""),
                    "stage": "candidate_validation_or_fetch",
                    "reason_code": classify_reason(exc),
                    "error": msg,
                })

        missing_metadata = any(
            not (root / "cache_v1" / "metadata" / (pdb + ".json")).exists()
            for pdb in by_u[t["protein_uniprot"]]
        )
        if errors:
            status = "candidate_validation_or_fetch_failed"
        elif not catalogs[t["protein_uniprot"]]:
            status = "pdb_present_no_eligible_ligand_metadata"
        else:
            status = "no_candidate_under_policy"
        if missing_metadata:
            status = "metadata_incomplete"
        elif (
            args.max_pdb_per_target > 0
            and attempted >= args.max_pdb_per_target
            and len(candidates) > attempted
        ):
            status = "budget_exhausted"
        return None, local_failures, {
            "pair_id": t["pair_id"],
            "status": status,
            "catalog_candidates": len(catalogs[t["protein_uniprot"]]),
            "policy_candidates": len(candidates),
            "policy_rejections": policy_rejections,
            "screen_failures": sum(
                r["stage"] == "chemistry_screen" for r in local_failures
            ),
            "attempted_candidates": attempted,
            "errors": " | ".join(errors),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(target_job, t): t for t in targets}
        for n, future in enumerate(as_completed(futures), 1):
            t = futures[future]
            try:
                chosen, rev, outcome = future.result()
            except Exception as exc:
                chosen, rev, outcome = None, [], {"pair_id": t["pair_id"], "status": "internal_error", "errors": str(exc)}
            if chosen:
                selected.append(chosen)
            for rec in rev:
                if rec.get("stage") == "chemistry_screen":
                    screen_failures.append(rec)
                else:
                    candidate_attempts.append(rec)
            outcomes.append(outcome)
            print(f"[binary] {n}/{len(targets)} usable={len(selected)} {outcome['status']}", flush=True)
    selected.sort(key=lambda r: r["pair_id"])
    write_json(root / "selected_binaries.json", selected)
    csv_out(root / "selected_binaries.csv", selected)
    # No quality tiers are used in this policy. Remove the stale legacy review file
    # if it was produced by an older A/B workflow.
    legacy_review = root / "B_candidates_for_review.csv"
    if legacy_review.exists():
        legacy_review.unlink()
    csv_out(root / "target_status.csv", outcomes)
    csv_out(root / "candidate_screen_failures.csv", screen_failures)
    csv_out(root / "candidate_attempt_failures.csv", candidate_attempts)
    csv_out(root / "request_failures.csv", failures, columns=["stage", "key", "error"])
    good = {m["pair_id"]: m["match_type"] for m in selected}
    for r in contexts:
        r["poi_match_type"] = good.get(r["poi_pair_id"], "")
        r["e3_match_type"] = good.get(r["e3_pair_id"], "")
        r["both_usable"] = bool(r["poi_match_type"] and r["e3_match_type"])
    csv_out(root / "context_coverage.csv", contexts)
    manifest = make_rl2_manifest(contexts, selected)
    csv_out(root / "rl2_manifest.csv", manifest)
    readiness = rl2_readiness_report(contexts, manifest, selected, args)
    write_json(root / "rl2_readiness.json", readiness)
    if selected:
        print("\nSelected binary match types:")
        print(
            pd.Series([m["match_type"] for m in selected])
            .value_counts()
            .sort_index()
            .to_string()
        )
    print(pd.DataFrame(contexts).groupby("split")["both_usable"].agg(["sum", "count", "mean"]).to_string())
    verdict = "READY" if readiness["ready"] else "NOT_READY"
    print(f"RL2 readiness: {verdict}; complete contexts={len(manifest)}/{len(contexts)}")
    print("Next input: rl2_manifest.csv; audit details: rl2_readiness.json")
    if args.require_rl2_ready and not readiness["ready"]:
        raise SystemExit(2)

def run_self_tests():
    """Fast offline regression tests for the high-risk chemistry decisions."""
    unspecified = "CC(O)C(=O)O"
    stereo_r = "C[C@H](O)C(=O)O"
    stereo_s = "C[C@@H](O)C(=O)O"

    resolved = chemistry_screen(unspecified, stereo_r)
    assert resolved and resolved["match_type"] == "exact"
    assert resolved["stereo_relation"] == "candidate_resolves_unspecified"
    assert chemistry_screen(stereo_r, stereo_s) is None
    assert chemistry_screen(stereo_r, unspecified) is None

    raw_terminal = "*Nc1ccccc1"
    core, marked, has_exit = terminal_with_exit(raw_terminal)
    assert has_exit and ":1" in marked and Chem.MolFromSmiles(core) is not None

    similar = chemistry_screen(core, "CCNc1ccccc1", marked)
    assert similar and similar["match_type"] == "similar_rooted_mcs"
    assert similar["target_coverage"] >= SIMILAR_TARGET_COVERAGE_MIN
    assert similar["core_heavy_atoms"] >= SIMILAR_CORE_HEAVY_ATOMS_MIN

    bound_similar = Chem.MolFromSmiles("CCNc1ccccc1")
    ti2, bi2, _, _ = map_to_bound(core, bound_similar, similar, marked)
    target_for_map, exit_idx = _target_and_exit(core, marked)
    assert exit_idx in set(ti2)
    assert any(
        n.GetIdx() in set(ti2)
        for n in target_for_map.GetAtomWithIdx(exit_idx).GetNeighbors()
    )
    assert len(ti2) == len(bi2) >= SIMILAR_CORE_HEAVY_ATOMS_MIN

    bound = Chem.MolFromSmiles(stereo_r)
    ti, bi, mode, relation = map_to_bound(unspecified, bound, resolved)
    assert len(ti) == len(bi) == bound.GetNumAtoms()
    assert len(set(bi)) == len(bi)
    assert mode in {"strict", "standardized_graph_fallback"}
    assert relation in {
        "candidate_resolves_unspecified", "exact", "compatible_by_symmetry"
    }

    assert catalog_exclusion_reason("NA", "[Na+]") == "common_non_druglike_component"
    assert catalog_exclusion_reason("XXX", "not-a-smiles") == "invalid_metadata_smiles"
    assert catalog_exclusion_reason("XXX", "CCO") is None
    assert entry_quality_rank("X-RAY DIFFRACTION", 2.0, 0.20) == 0
    assert entry_quality_rank("X-RAY DIFFRACTION", 2.0, None) == 1
    assert entry_quality_rank("ELECTRON MICROSCOPY", 3.0, None) == 2
    assert entry_quality_rank("ELECTRON MICROSCOPY", 4.0, None) == 3

    bound3d = Chem.MolFromSmiles("CCNc1ccccc1")
    conf3d = Chem.Conformer(bound3d.GetNumAtoms())
    conf3d.Set3D(True)
    for i in range(bound3d.GetNumAtoms()):
        conf3d.SetAtomPosition(
            i, (float(i), float(i % 3), float((i * 2) % 5))
        )
    bound3d.AddConformer(conf3d)
    core3d = mapped_core_from_target_with_bound_coords(
        core, marked, ti2, bound3d, bi2
    )
    assert core3d.GetNumAtoms() == len(ti2) == len(bi2)
    assert core3d.GetNumConformers() == 1

    mock_entry = {
        "polymer_entities": [{
            "rcsb_polymer_entity_container_identifiers": {
                "auth_asym_ids": ["A", "C"],
                "reference_sequence_identifiers": [{
                    "database_name": "UniProt",
                    "database_accession": "P12345",
                }],
            }
        }]
    }
    assert rcsb_auth_chains_for_uniprot(mock_entry, "P12345") == ["A", "C"]
    assert rcsb_auth_chains_for_uniprot(mock_entry, "Q99999") == []

    print(
        "Self-test passed: stereo directionality, exit-rooted MCS, atom map, "
        "target-topology mapped core, exact-UniProt chain fallback, "
        "PROTAC-unbound quality ranking, catalogue filter."
    )

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", default="datasets/protac_split_by_linker_cluster.csv")
    p.add_argument("--out-dir", default="datasets/deepternary_rl2_unbound_v4")
    p.add_argument("--reuse-cache", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-pdb-per-target", type=int, default=0,
                   help="Maximum ranked candidates tried per pair; 0 (default) tries until success or exhaustion.")
    p.add_argument("--max-resolution", type=float, default=3.5,
                   help="Preferred resolution used for candidate ranking only; not a hard PROTAC-unbound filter.")
    p.add_argument("--exact-only", action="store_true",
                   help="Diagnostic mode: accept exact constitutional matches only. Default also accepts validated exit-rooted similar ligands.")
    p.add_argument("--revalidate-existing", action="store_true",
                   help="Re-run current-policy selections; older-policy selections are always revalidated.")
    p.add_argument("--disable-validated-reuse", action="store_true",
                   help="Disable same-UniProt reuse of hash-verified exact experimental binary payloads.")
    p.add_argument("--min-rl2-contexts", type=int, default=500,
                   help="Minimum complete POI/E3 contexts for the RL2 readiness check (default: 500).")
    p.add_argument("--min-train-e3-proteins", type=int, default=3,
                   help="Minimum distinct E3 UniProt accessions in complete training contexts (default: 3).")
    p.add_argument("--min-eval-e3-proteins", type=int, default=2,
                   help="Minimum distinct E3 UniProt accessions across complete val/test contexts (default: 2).")
    p.add_argument("--require-rl2-ready", action="store_true",
                   help="Exit with status 2 after writing outputs when the RL2 readiness gate fails.")
    p.add_argument("--self-test", action="store_true",
                   help="Run fast offline chemistry regression tests and exit.")
    a = p.parse_args()
    if a.self_test:
        run_self_tests()
        return
    if (min(a.workers, a.batch_size, a.min_rl2_contexts,
            a.min_train_e3_proteins, a.min_eval_e3_proteins) < 1
            or a.max_pdb_per_target < 0):
        p.error("worker/batch/readiness minima must be positive; max-pdb-per-target must be >= 0")
    if not (a.max_resolution > 0):
        p.error("--max-resolution must be >0 A (ranking preference only)")
    build(a)

if __name__ == "__main__":
    main()

