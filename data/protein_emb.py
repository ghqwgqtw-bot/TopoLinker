import pickle
import time
from io import StringIO
from Bio import SeqIO

import pandas as pd
import requests
from requests.exceptions import HTTPError, RequestException

train_path = './datasets/rl/train/protac_processed.csv'
val_path = './datasets/rl/val/protac_processed.csv'
test_path = './datasets/rl/test/protac_processed.csv'

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

uniprot_ids = set(train_df['Uniprot'].tolist() + train_df['e3_ligase_Uniprot'].tolist() + val_df['Uniprot'].tolist() + val_df['e3_ligase_Uniprot'].tolist() + test_df['Uniprot'].tolist() + test_df['e3_ligase_Uniprot'].tolist())

try:
    seq_cache = pickle.load(open('./datasets/rl/seq_cache.pkl'), 'rb')
except:
    seq_cache = {}

def get_aa_seq(uniprot_id):
    """
    Retrieve and cache the amino acid sequence for a UniProt accession.

    Args:
        uniprot_id (str): UniProt protein accession.

    Returns:
        str or None: Protein sequence, or None if retrieval fails.
    """
    try:
        if uniprot_id in seq_cache:
            return seq_cache[uniprot_id]

        url = f'https://www.uniprot.org/uniprot/{uniprot_id}.fasta'
        r = requests.get(url)
        r.raise_for_status()
        fasta = r.text
        fasta_io = StringIO(fasta)
        seq_record = SeqIO.read(fasta_io, 'fasta')
        time.sleep(1)
        seq = str(seq_record.seq)
        seq_cache[uniprot_id] = seq
        pickle.dump(seq_cache, open('./datasets/rl/seq_cache.pkl', 'wb'))
        return seq
    except HTTPError as http_err:
        print(f'HTTP error: {http_err}')
    except RequestException as req_err:
        print(f'Request error: {req_err}')
    except Exception as e:
        print(f'Unexpected error: {e}')

p_map = {uniprot_id: get_aa_seq(uniprot_id) for uniprot_id in uniprot_ids}

with open('./datasets/rl/p_map.pkl', 'wb') as f:
    pickle.dump(p_map, f)


# # PROTAC descriptor statistics
#
# columns = [
#     "Smiles",
#     "Molecular Weight",
#     "Exact Mass",
#     "XLogP3",
#     "Heavy Atom Count",
#     "Ring Count",
#     "Hydrogen Bond Acceptor Count",
#     "Hydrogen Bond Donor Count",
#     "Rotatable Bond Count",
#     "Topological Polar Surface Area",
# ]
#
# train_csv = "./datasets/rl/train/protac_processed.csv"
# out_path = "./datasets/rl/descriptor_stats.pkl"
#
# df = pd.read_csv(train_csv)
#
# feat_cols = columns[1:]
# df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
#
# mean = df[feat_cols].mean()
# std = df[feat_cols].std().replace(0, 1.0).fillna(1.0)
#
# stats = {
#     "mean": mean.to_dict(),
#     "std": std.to_dict(),
# }
#
# with open(out_path, "wb") as f:
#     pickle.dump(stats, f)
#
# print("Saved descriptor statistics:", out_path)


