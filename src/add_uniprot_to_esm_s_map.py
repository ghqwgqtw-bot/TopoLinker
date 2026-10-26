import pickle
import shutil
import sys
import time
from io import StringIO
from pathlib import Path

import requests
from Bio import SeqIO
from torchdrug import data
from torchdrug.transforms import Compose, ProteinView, TruncateProtein


PROJECT_ROOT = Path(".")
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.get_embed_s import PROTACTargets, ESMSEmbedder


# ============================================================
# Settings
# ============================================================
UNIPROT_ID = "Q8IVV7"   # GID4, used by 8X7G chain A

ROOT = PROJECT_ROOT / "datasets" / "rl"
ESM_PATH = ROOT / "esm_s_map.pkl"

TMP_ROOT = PROJECT_ROOT / "tmp" / f"add_esm_s_{UNIPROT_ID}"
TMP_PMAP = TMP_ROOT / "p_map.pkl"

# Optional copied map used by some reward / temporary scripts.
TMP_REWARD_ESM_PATH = PROJECT_ROOT / "tmp" / "protac_stan_reward" / "esm_s_map.pkl"


def fetch_uniprot_fasta(uniprot_id):
    """
    Fetch canonical UniProt FASTA sequence.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    fasta_io = StringIO(response.text)
    record = SeqIO.read(fasta_io, "fasta")

    seq = str(record.seq).upper().strip()

    if len(seq) == 0:
        raise RuntimeError(f"Empty sequence for {uniprot_id}")

    return seq


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def backup_file(path):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(path.stem + f".backup_before_add_{UNIPROT_ID}_{timestamp}" + path.suffix)

    shutil.copy2(path, backup_path)

    return backup_path


def main():
    print("=" * 100)
    print("Adding UniProt embedding to esm_s_map.pkl")
    print("UniProt:", UNIPROT_ID)
    print("ESM path:", ESM_PATH)
    print("=" * 100)

    if not ESM_PATH.exists():
        raise FileNotFoundError(ESM_PATH)

    esm_map = load_pickle(ESM_PATH)
    esm_keys = set(str(k).strip() for k in esm_map.keys())

    print("Existing esm_s_map keys:", len(esm_keys))
    print(f"{UNIPROT_ID} already exists:", UNIPROT_ID in esm_keys)

    if UNIPROT_ID in esm_keys:
        print("[OK] Existing esm_s_map.pkl already contains:", UNIPROT_ID)
        return

    # Backup first.
    backup_path = backup_file(ESM_PATH)
    print("Backup saved:", backup_path)

    # Fetch sequence.
    seq = fetch_uniprot_fasta(UNIPROT_ID)
    print("Fetched sequence length:", len(seq))
    print("First 30 aa:", seq[:30])

    # Create temporary p_map.pkl containing only the new UniProt.
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    save_pickle({UNIPROT_ID: seq}, TMP_PMAP)
    print("Temporary p_map saved:", TMP_PMAP)

    # Build one-protein dataset and infer embedding.
    protein_view = ProteinView(view="residue")
    truncate_protein = TruncateProtein(max_length=1022)
    transforms = Compose([protein_view, truncate_protein])

    dataset = PROTACTargets(
        path=str(TMP_ROOT),
        transform=transforms,
        verbose=1,
        lazy=True,
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    embedder = ESMSEmbedder()
    embeddings = embedder.get_embed(dataloader)

    if len(embeddings) != 1:
        raise RuntimeError(f"Expected 1 embedding, got {len(embeddings)}")

    uid, feature = embeddings[0]

    uid = str(uid).strip()

    if uid != UNIPROT_ID:
        raise RuntimeError(f"Unexpected UniProt ID: {uid}, expected {UNIPROT_ID}")

    print("New embedding shape:", feature.shape)

    # Merge instead of overwrite.
    esm_map[UNIPROT_ID] = feature

    save_pickle(esm_map, ESM_PATH)

    # Verify.
    updated_map = load_pickle(ESM_PATH)
    updated_keys = set(str(k).strip() for k in updated_map.keys())

    print("Updated esm_s_map keys:", len(updated_keys))
    print(f"{UNIPROT_ID} in updated map:", UNIPROT_ID in updated_keys)

    if UNIPROT_ID not in updated_keys:
        raise RuntimeError(f"Failed to add {UNIPROT_ID} to {ESM_PATH}")

    # Optional: sync temporary reward copy if it already exists.
    if TMP_REWARD_ESM_PATH.exists():
        shutil.copy2(ESM_PATH, TMP_REWARD_ESM_PATH)
        print("Synced to:", TMP_REWARD_ESM_PATH)

    print("=" * 100)
    print("[DONE] Added", UNIPROT_ID, "to", ESM_PATH)
    print("=" * 100)


if __name__ == "__main__":
    main()