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


esm_path = './datasets/rl/esm_s_map.pkl'

with open(esm_path, 'rb') as f:
    esm_map = pickle.load(f)

esm_keys = set(str(k).strip() for k in esm_map.keys())

missing = [uid for uid in uniprot_ids if uid not in esm_keys]
print(len(missing))

try:
    seq_cache = pickle.load(open('./datasets/rl/seq_cache.pkl'), 'rb')
except:
    seq_cache = {}


def get_aa_seq(uniprot_id):
    """
    Download fasta file from Uniprot and extract amino acid sequence

    Parameters:
        uniprot_id (str): UniProt ID of the protein.

    Returns:
        str: Amino acid sequence of the protein.

    Example:
        get_aa_seq('P04637')
        >>> 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD'
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
        print(f'发生HTTP错误: {http_err}')
    except RequestException as req_err:
        print(f'发生请求错误: {req_err}')
    except Exception as e:
        print(f'发生意外错误: {e}')

p_map = {uniprot_id: get_aa_seq(uniprot_id) for uniprot_id in uniprot_ids}

with open('./datasets/rl/p_map.pkl', 'wb') as f:
    pickle.dump(p_map, f)





# import pickle
# import pandas as pd
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
# train_csv = r"D:\pycharm\Pycharm_Projects\LinkerDiff\datasets\rl\train\protac_processed.csv"
# out_path = r"D:\pycharm\Pycharm_Projects\LinkerDiff\datasets\rl\descriptor_stats.pkl"
#
# df = pd.read_csv(train_csv)
#
# feat_cols = columns[1:]
# df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
#
# mean = df[feat_cols].mean()
# std = df[feat_cols].std().replace(0, 1.0).fillna(1.0)

#     "mean": mean.to_dict(),
#     "std": std.to_dict(),
# }
#
# with open(out_path, "wb") as f:
#     pickle.dump(stats, f)
#
# print("saved:", out_path)


# import pickle
# import os
#
# src = r"./datasets/rl/esm_s_map.pkl"
# dst = r"./tmp/protac_stan_reward/esm_s_map.pkl"
#
# for path in [src, dst]:
#     print("=" * 80)
#     print(path)
#     print("exists:", os.path.exists(path))
#
#     if os.path.exists(path):
#         with open(path, "rb") as f:
#             m = pickle.load(f)
#
#         keys = set(str(k).strip() for k in m.keys())
#         print("num keys:", len(keys))
#         print("Q8TEB1 in keys:", "Q8TEB1" in keys)
# from pathlib import Path
# from rdkit import Chem
# from collections import Counter
#
#
# PROJECT_ROOT = Path(r"D:\pycharm\Pycharm_Projects\LinkerDiff")
#
# ROOTS = [
#     PROJECT_ROOT / "datasets" / "rl" / "train" / "deepternary_inputs",
#     PROJECT_ROOT / "datasets" / "rl" / "val" / "deepternary_inputs",
#     PROJECT_ROOT / "datasets" / "rl" / "test" / "deepternary_inputs",
# ]
#
#
# def check_sdf(sdf_path: Path):
#     if not sdf_path.exists():
#         return False, "missing"
#
#     if sdf_path.stat().st_size <= 0:
#         return False, "empty"
#
#     try:
#         suppl = Chem.SDMolSupplier(
#             str(sdf_path),
#             sanitize=False,
#             removeHs=False,
#         )
#
#         mol = suppl[0] if len(suppl) > 0 else None
#
#         if mol is None:
#             return False, "mol is None"
#
#         if mol.GetNumConformers() == 0:
#             return False, "no conformer"
#
#         try:
#             mol.UpdatePropertyCache(strict=False)
#         except Exception:
#             pass
#
#         return True, "ok"
#
#     except Exception as e:
#         return False, repr(e)
#
#
# def main():
#     all_bad = []
#
#     for root in ROOTS:
#         print("\n" + "=" * 100)
#         print("checking root:", root)
#         print("exists:", root.exists())
#
#         if not root.exists():
#             continue
#
#         sdf_files = sorted(root.rglob("*.sdf"))
#
#         print("sdf file count:", len(sdf_files))
#
#         bad = []
#         reason_counter = Counter()
#
#         for sdf_path in sdf_files:
#             ok, reason = check_sdf(sdf_path)
#
#             if not ok:
#                 bad.append((sdf_path, reason))
#                 reason_counter[reason] += 1
#
#         print("bad count:", len(bad))
#         print("bad reason counts:", dict(reason_counter))
#
#         if bad:
#             print("\nFirst 50 bad SDF:")
#             for p, reason in bad[:50]:
#                 print("[BAD]", reason, p)
#
#         all_bad.extend(bad)
#
#     print("\n" + "=" * 100)
#     print("TOTAL bad:", len(all_bad))
#
#     if len(all_bad) == 0:
#         print("[OK] 所有 deepternary_inputs 下的 SDF 都能被 RDKit 读取，并且都有 conformer。")
#
#
# if __name__ == "__main__":
#     main()


# from pathlib import Path
# import torch
#
#
# PROJECT_ROOT = Path(r"D:\pycharm\Pycharm_Projects\LinkerDiff")
#
# PATH_COLS = [
#     "p1_path",
#     "p2_path",
#     "unbound_lig1_path",
#     "unbound_lig2_path",
#     "lig1_mask_path",
#     "lig2_mask_path",
# ]
#
#
# def load_pt(pt_path):
#     try:
#         return torch.load(str(pt_path), map_location="cpu")
#     except TypeError:
#         return torch.load(str(pt_path), map_location="cpu", weights_only=False)
#
#
# def main():
#     for split in ["train", "val", "test"]:
#         pt_path = PROJECT_ROOT / "datasets" / f"protac_rltrain_{split}.pt"
#
#         print("\n" + "=" * 120)
#         print("split:", split)
#         print("pt_path:", pt_path)
#         print("exists:", pt_path.exists())
#
#         if not pt_path.exists():
#             continue
#
#         data = load_pt(pt_path)
#
#         print("num samples:", len(data))
#
#         # 打印前 20 个样本的路径
#         print("\n" + "-" * 120)
#         print("First 20 samples paths:")
#
#         for i, item in enumerate(data[:20]):
#             print("\n" + "-" * 80)
#             print("idx:", i)
#             print("uuid:", item.get("uuid", None))
#             print("name:", item.get("name", None))
#
#             for col in PATH_COLS:
#                 print(f"{col}: {item.get(col, None)}")
#
#         # 统计路径格式
#         windows_abs_count = 0
#         linux_abs_count = 0
#         backslash_count = 0
#         missing_count = 0
#         none_count = 0
#
#         print("\n" + "-" * 120)
#         print("Checking path format and existence...")
#
#         for i, item in enumerate(data):
#             uuid = item.get("uuid", f"idx_{i}")
#
#             for col in PATH_COLS:
#                 p = item.get(col, None)
#
#                 if p is None:
#                     none_count += 1
#                     if none_count <= 20:
#                         print("[NONE]", i, uuid, col)
#                     continue
#
#                 raw_p = str(p)
#
#                 if "\\" in raw_p:
#                     backslash_count += 1
#                     if backslash_count <= 20:
#                         print("[BACKSLASH]", i, uuid, col, raw_p)
#
#                 # Windows 绝对路径，例如 D:/xxx 或 D:\xxx
#                 normalized = raw_p.replace("\\", "/")
#
#                 if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/":
#                     windows_abs_count += 1
#
#                 # Linux 绝对路径，例如 /root/xxx
#                 if normalized.startswith("/"):
#                     linux_abs_count += 1
#                     if linux_abs_count <= 20:
#                         print("[LINUX ABS]", i, uuid, col, raw_p)
#
#                 # 在 Windows 本地检查文件是否存在
#                 if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/":
#                     check_path = Path(normalized)
#                 elif normalized.startswith("/"):
#                     # Linux 绝对路径，Windows 本地无法检查
#                     continue
#                 else:
#                     check_path = PROJECT_ROOT / normalized
#
#                 if not check_path.exists():
#                     missing_count += 1
#                     if missing_count <= 20:
#                         print("[MISSING]", i, uuid, col)
#                         print(" raw:", raw_p)
#                         print(" win:", check_path)
#
#         print("\n" + "-" * 120)
#         print("Summary:")
#         print("none_count:", none_count)
#         print("backslash_count:", backslash_count)
#         print("windows_abs_count:", windows_abs_count)
#         print("linux_abs_count:", linux_abs_count)
#         print("missing_count:", missing_count)
#
#
# if __name__ == "__main__":
#     main()

