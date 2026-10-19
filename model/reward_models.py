# import math
# import os
# import re
# import subprocess
# import sys
# import shutil
# import importlib
# import contextlib
# import pandas as pd
# import toml
# import torch
# from rdkit import Chem
# from rdkit.Chem import Descriptors
#
# PROTAC_STAN_DESCRIPTOR_COLUMNS = [
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
#
# def calculate_molecular_descriptors(smiles):
#     """
#     根据 SMILES 计算 PROTAC-STAN 需要的分子描述符。
#     """
#     mol = Chem.MolFromSmiles(smiles)
#
#     if mol is None:
#         raise ValueError(f"Invalid SMILES: {smiles}")
#
#     desc = {
#         "Molecular Weight": Descriptors.MolWt(mol),
#         "Exact Mass": Descriptors.ExactMolWt(mol),
#         "XLogP3": Descriptors.MolLogP(mol),
#         "Heavy Atom Count": Descriptors.HeavyAtomCount(mol),
#         "Ring Count": Descriptors.RingCount(mol),
#         "Hydrogen Bond Acceptor Count": Descriptors.NumHAcceptors(mol),
#         "Hydrogen Bond Donor Count": Descriptors.NumHDonors(mol),
#         "Rotatable Bond Count": Descriptors.NumRotatableBonds(mol),
#         "Topological Polar Surface Area": Descriptors.TPSA(mol),
#     }
#
#     return desc
# def import_module_from_file(module_name, file_path, extra_sys_path=None):
#     """
#     从指定文件路径导入模块，避免和当前项目里的 model / data 模块冲突。
#     """
#     old_sys_path = list(sys.path)
#
#     if extra_sys_path is not None:
#         sys.path.insert(0, extra_sys_path)
#
#     try:
#         spec = importlib.util.spec_from_file_location(module_name, file_path)
#         module = importlib.util.module_from_spec(spec)
#
#         # 注册到 sys.modules，方便模块内部相互 import
#         sys.modules[module_name] = module
#
#         spec.loader.exec_module(module)
#
#     finally:
#         sys.path = old_sys_path
#
#     return module
# @contextlib.contextmanager
# def suppress_stdout_stderr(enabled=True):
#     if not enabled:
#         yield
#         return
#
#     with open(os.devnull, "w") as devnull:
#         with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
#             yield
# class ProtacStanReward:
#     def __init__(
#             self,
#             project_dir,
#             model_path,
#             config_path,
#             esm_map_path,
#             tmp_root,
#             device,
#             descriptor_stats_path,
#             quiet=True,
#     ):
#         """
#         project_dir: PROTAC-STAN-main 路径
#         model_path: saved_models/protac-stan.pt
#         config_path: config.toml
#         esm_map_path: 你刚生成的 esm_s_map.pkl
#         tmp_root: RL 过程中临时保存 PROTAC-STAN 输入的位置
#         """
#
#         self.project_dir = os.path.abspath(project_dir)
#         self.model_path = os.path.abspath(model_path)
#         self.config_path = os.path.abspath(config_path)
#         self.esm_map_path = os.path.abspath(esm_map_path)
#         self.tmp_root = os.path.abspath(tmp_root)
#         self.device = device
#         self.descriptor_stats_path = os.path.abspath(descriptor_stats_path)
#         self.quiet = quiet
#
#         os.makedirs(self.tmp_root, exist_ok=True)
#
#         # PROTACData.process() 会读取 root/esm_s_map.pkl
#         dst_esm = os.path.join(self.tmp_root, "esm_s_map.pkl")
#
#         # 每次初始化都覆盖，避免 tmp_root 里残留旧 esm_s_map.pkl
#         shutil.copyfile(self.esm_map_path, dst_esm)
#
#         print(f"[ProtacStanReward] copied esm map:")
#         print(f"  from: {self.esm_map_path}")
#         print(f"  to:   {dst_esm}")
#
#         # 动态导入 PROTAC-STAN 的 model 和 data_loader
#         model_file = os.path.join(self.project_dir, "model.py")
#         data_file = os.path.join(self.project_dir, "data.py")
#         loader_file = os.path.join(self.project_dir, "data_loader.py")
#
#         if not os.path.exists(model_file):
#             raise FileNotFoundError(f"Cannot find PROTAC-STAN model.py: {model_file}")
#
#         if not os.path.exists(data_file):
#             raise FileNotFoundError(f"Cannot find PROTAC-STAN data.py: {data_file}")
#
#         if not os.path.exists(loader_file):
#             raise FileNotFoundError(f"Cannot find PROTAC-STAN data_loader.py: {loader_file}")
#
#         # 备份可能已经存在的 data 模块，避免污染当前项目
#         old_data_module = sys.modules.get("data", None)
#
#         try:
#             # 关键：先把 PROTAC-STAN 的 data.py 注册成 data
#             # 这样 data_loader.py 里的 from data import PROTACData 才会导入正确
#             self.protac_data_module = import_module_from_file(
#                 module_name="data",
#                 file_path=data_file,
#                 extra_sys_path=self.project_dir,
#             )
#
#             self.protac_loader_module = import_module_from_file(
#                 module_name="protac_stan_data_loader",
#                 file_path=loader_file,
#                 extra_sys_path=self.project_dir,
#             )
#
#         finally:
#             # 还原原来的 data 模块，避免影响 LinkerDiff 自己的 data
#             if old_data_module is not None:
#                 sys.modules["data"] = old_data_module
#             else:
#                 sys.modules.pop("data", None)
#
#         # model.py 用独立名字导入，避免和 LinkerDiff/model 冲突
#         self.protac_model_module = import_module_from_file(
#             module_name="protac_stan_model",
#             file_path=model_file,
#             extra_sys_path=self.project_dir,
#         )
#
#         PROTAC_STAN = self.protac_model_module.PROTAC_STAN
#         self.PROTACLoader = self.protac_loader_module.PROTACLoader
#
#         cfg = toml.load(self.config_path)
#         model_cfg = cfg["model"]
#
#         self.model = PROTAC_STAN(model_cfg)
#
#         state_dict = torch.load(
#             self.model_path,
#             map_location=self.device
#         )
#
#         self.model.load_state_dict(state_dict)
#         self.model.to(self.device)
#         self.model.eval()
#         dst_stats = os.path.join(self.tmp_root, "descriptor_stats.pkl")
#         shutil.copyfile(self.descriptor_stats_path, dst_stats)
#
#         for p in self.model.parameters():
#             p.requires_grad_(False)
#
#     def _write_csv(self, smiles_list, poi_ids, e3_ids, name):
#         """
#         写 PROTAC-STAN 需要的 {name}.csv。
#         必须包含：
#             Smiles
#             Uniprot
#             E3 ligase Uniprot
#             9 个分子描述符
#         """
#
#         rows = []
#
#         for smi, poi, e3 in zip(smiles_list, poi_ids, e3_ids):
#             desc = calculate_molecular_descriptors(smi)
#
#             row = {
#                 "Smiles": smi,
#                 "Uniprot": str(poi),
#                 "E3 ligase Uniprot": str(e3),
#             }
#
#             for col in PROTAC_STAN_DESCRIPTOR_COLUMNS:
#                 row[col] = desc[col]
#
#             rows.append(row)
#
#         df = pd.DataFrame(rows)
#
#         csv_path = os.path.join(self.tmp_root, f"{name}.csv")
#         df.to_csv(csv_path, index=False)
#
#         return csv_path
#
#     @torch.no_grad()
#     def score(self, smiles_list, poi_ids, e3_ids, step=0, batch_size=16):
#         """
#         smiles_list: 长度 B，其中无效分子为 None
#         poi_ids:     长度 B，batch["poi_uniprot"]
#         e3_ids:      长度 B，batch["e3_uniprot"]
#
#         返回:
#             scores: [B] tensor
#         """
#
#         B = len(smiles_list)
#
#         scores = torch.zeros(
#             B,
#             device=self.device,
#             dtype=torch.float32
#         )
#
#         valid_indices = []
#
#         for i, smi in enumerate(smiles_list):
#             if smi is not None:
#                 valid_indices.append(i)
#
#         if len(valid_indices) == 0:
#             return scores
#
#         valid_smiles = [smiles_list[i] for i in valid_indices]
#         valid_poi = [poi_ids[i] for i in valid_indices]
#         valid_e3 = [e3_ids[i] for i in valid_indices]
#
#         name = f"rl_reward_{step}"
#
#         self._write_csv(
#             smiles_list=valid_smiles,
#             poi_ids=valid_poi,
#             e3_ids=valid_e3,
#             name=name,
#         )
#
#         # 删除旧 processed，防止 PyG 复用缓存
#         processed_dir = os.path.join(self.tmp_root, "processed", name)
#         if os.path.exists(processed_dir):
#             shutil.rmtree(processed_dir)
#
#         with suppress_stdout_stderr(self.quiet):
#             _, test_loader = self.PROTACLoader(
#                 root=self.tmp_root,
#                 name=name,
#                 batch_size=min(batch_size, len(valid_indices)),
#                 train_ratio=0.0,
#             )
#
#         valid_scores = []
#
#         self.model.eval()
#
#         for data in test_loader:
#             protac_data = data["protac"].to(self.device)
#             e3_ligase_data = data["e3_ligase"].to(self.device)
#             poi_data = data["poi"].to(self.device)
#
#             if not torch.isfinite(protac_data.x).all():
#                 print("[WARN] protac_data.x contains NaN/Inf. Set this batch degradation score to 0.")
#                 pos_probs = torch.zeros(
#                     e3_ligase_data.size(0),
#                     device=self.device,
#                     dtype=torch.float32,
#                 )
#                 valid_scores.extend(pos_probs.detach().cpu().tolist())
#                 continue
#
#             num_graphs = e3_ligase_data.size(0)
#
#             bad_graph_mask = torch.zeros(
#                 num_graphs,
#                 device=self.device,
#                 dtype=torch.bool,
#             )
#
#             # 检查 protac_data.x 中哪些 graph 有 NaN/Inf
#             if not torch.isfinite(protac_data.x).all():
#                 print("[WARN] protac_data.x contains NaN/Inf.")
#
#                 bad_node_mask = ~torch.isfinite(protac_data.x).all(dim=1)
#
#                 if hasattr(protac_data, "batch"):
#                     bad_graph_idx = protac_data.batch[bad_node_mask].unique()
#                     bad_graph_mask[bad_graph_idx] = True
#                     print("bad graph idx:", bad_graph_idx.detach().cpu().tolist())
#                 else:
#                     bad_graph_mask[:] = True
#
#                 # 打印坏特征维度，方便定位来源
#                 bad_dim_idx = torch.where((~torch.isfinite(protac_data.x)).any(dim=0))[0]
#                 print("bad feature dims:", bad_dim_idx.detach().cpu().tolist())
#                 print("num bad values:", (~torch.isfinite(protac_data.x)).sum().item())
#
#                 # 为了让 model forward 不崩，先把 NaN/Inf 替换成 0
#                 protac_data.x = torch.nan_to_num(
#                     protac_data.x,
#                     nan=0.0,
#                     posinf=0.0,
#                     neginf=0.0,
#                 )
#
#             # edge_attr 也检查一下
#             if hasattr(protac_data, "edge_attr") and protac_data.edge_attr is not None:
#                 if not torch.isfinite(protac_data.edge_attr).all():
#                     print("[WARN] protac_data.edge_attr contains NaN/Inf.")
#                     protac_data.edge_attr = torch.nan_to_num(
#                         protac_data.edge_attr,
#                         nan=0.0,
#                         posinf=0.0,
#                         neginf=0.0,
#                     )
#
#             outputs, _ = self.model(
#                 protac_data,
#                 e3_ligase_data,
#                 poi_data,
#                 mode="eval",
#             )
#
#             # 输出也按样本检查
#             finite_output_mask = torch.isfinite(outputs).all(dim=1)
#
#             valid_score_mask = finite_output_mask & (~bad_graph_mask)
#
#             pos_probs = torch.zeros(
#                 outputs.size(0),
#                 device=self.device,
#                 dtype=torch.float32,
#             )
#
#             if valid_score_mask.any():
#                 probs = torch.softmax(outputs[valid_score_mask], dim=1)
#                 probs = torch.nan_to_num(
#                     probs,
#                     nan=0.0,
#                     posinf=0.0,
#                     neginf=0.0,
#                 )
#                 pos_probs[valid_score_mask] = probs[:, 1]
#
#             if (~valid_score_mask).any():
#                 print(
#                     "[WARN] Some PROTAC-STAN samples are invalid. "
#                     "Set their degradation scores to 0."
#                 )
#
#             valid_scores.extend(pos_probs.detach().cpu().tolist())
#
#         for idx, score in zip(valid_indices, valid_scores):
#             scores[idx] = float(score)
#
#         return scores
# def safe_name(name):
#     name = str(name)
#     name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
#     return name
#
#
# def write_mol_to_sdf(mol, out_path):
#     """
#     把生成的 RDKit Mol 写成 SDF，供 DeepTernary 读取。
#     SDF 比 PDB 更适合小分子，可以减少 kekulize / 键类型问题。
#     """
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#
#     try:
#         mol = Chem.Mol(mol)
#         mol.UpdatePropertyCache(strict=False)
#
#         writer = Chem.SDWriter(str(out_path))
#
#         if hasattr(writer, "SetKekulize"):
#             writer.SetKekulize(False)
#
#         writer.write(mol)
#         writer.close()
#
#         return True
#
#     except Exception as e:
#         print(f"[DeepTernaryReward] Failed to write ligand sdf: {e}")
#         return False
#
# def write_mol_to_pdb(mol, out_path):
#     """
#     把生成的完整 PROTAC ligand 写成 PDB，供 DeepTernary 的 --lig 使用。
#     注意：--lig 不能用 SDF，因为 predict.py 后面 set_new_coords() 只接受 PDB。
#     """
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#
#     try:
#         mol = Chem.Mol(mol)
#         mol.UpdatePropertyCache(strict=False)
#
#         Chem.MolToPDBFile(mol, str(out_path))
#         return True
#
#     except Exception as e:
#         print(f"[DeepTernaryReward] Failed to write ligand pdb: {e}")
#         return False
#
# class DeepTernaryReward:
#     def __init__(
#         self,
#         python_exe,
#         project_dir,
#         predict_script,
#         tmp_root,
#         seeds=1,
#         timeout=600,
#         device="cuda",
#     ):
#         """
#         python_exe:
#             DeepTernary 所在环境的 python.exe
#             例如 D:/study/Anaconda3/envs/linker/python.exe
#
#         project_dir:
#             DeepTernary-master 根目录
#
#         predict_script:
#             DeepTernary-master/predict.py
#
#         tmp_root:
#             临时输出目录
#
#         seeds:
#             DeepTernary 采样种子数量。RL 阶段先用 1。
#
#         timeout:
#             subprocess 超时时间，单位秒。
#         """
#
#         if python_exe is None or str(python_exe).lower() == "current":
#             self.python_exe = sys.executable
#         else:
#             self.python_exe = os.path.abspath(python_exe)
#         self.project_dir = os.path.abspath(project_dir)
#         self.predict_script = os.path.abspath(predict_script)
#         self.tmp_root = os.path.abspath(tmp_root)
#         self.seeds = int(seeds)
#         self.timeout = int(timeout)
#         self.device = device
#
#         os.makedirs(self.tmp_root, exist_ok=True)
#
#     def score_one(
#         self,
#         mol,
#         name,
#         p1_path,
#         p2_path,
#         unbound_lig1_path,
#         unbound_lig2_path,
#         lig1_mask_path,
#         lig2_mask_path,
#         step,
#         index,
#     ):
#         name = safe_name(name)
#         sample_name = f"{name}_step{step}_idx{index}"
#         sample_dir = os.path.join(self.tmp_root, sample_name)
#
#         # 每次同名样本先清空，避免旧 out/ 结果残留
#         if os.path.exists(sample_dir):
#             shutil.rmtree(sample_dir)
#
#         outdir = os.path.join(sample_dir, "out")
#         os.makedirs(sample_dir, exist_ok=True)
#         os.makedirs(outdir, exist_ok=True)
#
#         lig_pdb_path = os.path.join(sample_dir, "generated_ligand.pdb")
#         lig_sdf_path = os.path.join(sample_dir, "generated_ligand.sdf")
#
#         ok_pdb = write_mol_to_pdb(mol, lig_pdb_path)
#         ok_sdf = write_mol_to_sdf(mol, lig_sdf_path)
#
#         if not ok_pdb or not ok_sdf:
#             return 0.0, None, None
#
#         lig_path = os.path.abspath(lig_pdb_path)
#         lig_read_path = os.path.abspath(lig_sdf_path)
#
#         # 注意：batch 中的路径是相对 LinkerDiff 项目的，所以这里转成绝对路径
#         p1_path = os.path.abspath(str(p1_path))
#         p2_path = os.path.abspath(str(p2_path))
#         unbound_lig1_path = os.path.abspath(str(unbound_lig1_path))
#         unbound_lig2_path = os.path.abspath(str(unbound_lig2_path))
#         lig1_mask_path = os.path.abspath(str(lig1_mask_path))
#         lig2_mask_path = os.path.abspath(str(lig2_mask_path))
#         outdir = os.path.abspath(outdir)
#
#         cmd = [
#             self.python_exe,
#             self.predict_script,
#             "--task", "PROTAC",
#             "--lig", lig_path,
#             "--lig-read", lig_read_path,
#             "--p1", p1_path,
#             "--p2", p2_path,
#             "--unbound-lig1", unbound_lig1_path,
#             "--unbound-lig2", unbound_lig2_path,
#             "--lig1-mask", lig1_mask_path,
#             "--lig2-mask", lig2_mask_path,
#             "--outdir", outdir,
#             "--name", sample_name,
#             "--seeds", str(self.seeds),
#             "--device", 'cpu',
#             "--no-correct",
#         ]
#
#         try:
#             result = subprocess.run(
#                 cmd,
#                 cwd=self.project_dir,
#                 capture_output=True,
#                 text=True,
#                 timeout=self.timeout,
#             )
#
#             # if result.returncode != 0:
#             #     print("[DeepTernaryReward] subprocess failed")
#             #     print(result.stderr[-2000:])
#             #     return 0.0, None, None
#             #
#             # summary_path = os.path.join(outdir, f"summary_{sample_name}.csv")
#
#             summary_path = os.path.join(outdir, f"summary_{sample_name}.csv")
#
#             if result.returncode != 0 and not os.path.exists(summary_path):
#                 print("=" * 100)
#                 print("[DeepTernaryReward] subprocess failed and summary csv not found")
#                 print("[returncode]", result.returncode)
#                 print("[cwd]", self.project_dir)
#                 print("[cmd]", " ".join(cmd))
#                 print("-" * 100)
#                 print("[stdout]")
#                 print(result.stdout[-4000:] if result.stdout else "<EMPTY>")
#                 print("-" * 100)
#                 print("[stderr]")
#                 print(result.stderr[-4000:] if result.stderr else "<EMPTY>")
#                 print("=" * 100)
#                 return 0.0, None, None
#             if result.returncode != 0 and os.path.exists(summary_path):
#                 print(
#                     f"[DeepTernaryReward] WARN: subprocess returncode={result.returncode}, "
#                     f"but summary csv exists. Use summary csv anyway."
#                 )
#
#             if not os.path.exists(summary_path):
#                 print("[DeepTernaryReward] summary csv not found:", summary_path)
#                 return 0.0, None, None
#
#
#             if not os.path.exists(summary_path):
#                 print("[DeepTernaryReward] summary csv not found:", summary_path)
#                 return 0.0, None, None
#
#             df = pd.read_csv(summary_path)
#
#             if len(df) == 0:
#                 return 0.0, None, None
#
#             pred_p2_rmsd = float(df.iloc[0]["pred_p2_rmsd"])
#             clash_ratio = float(df.iloc[0]["clash_ratio"])
#
#             if not math.isfinite(pred_p2_rmsd) or not math.isfinite(clash_ratio):
#                 return 0.0, pred_p2_rmsd, clash_ratio
#
#             # RMSD 越小越好，clash_ratio 越小越好
#             reward = math.exp(-pred_p2_rmsd / 2.0) * math.exp(-5.0 * clash_ratio)
#
#             reward = float(max(0.0, min(1.0, reward)))
#
#             return reward, pred_p2_rmsd, clash_ratio
#
#         except subprocess.TimeoutExpired:
#             print("[DeepTernaryReward] timeout")
#             return 0.0, None, None
#
#         except Exception as e:
#             print(f"[DeepTernaryReward] exception: {e}")
#             return 0.0, None, None
#
#     @torch.no_grad()
#     def score(
#         self,
#         mols,
#         valid_mask,
#         names,
#         p1_paths,
#         p2_paths,
#         unbound_lig1_paths,
#         unbound_lig2_paths,
#         lig1_mask_paths,
#         lig2_mask_paths,
#         step,
#         max_samples=None,
#         device="cuda",
#     ):
#         """
#         mols: list[RDKit Mol], 长度 B
#         valid_mask: list[bool]，只有合法且连通的分子才调用 DeepTernary
#         """
#
#         B = len(mols)
#
#         rewards = torch.zeros(B, device=device, dtype=torch.float32)
#         rmsds = torch.full((B,), -1.0, device=device, dtype=torch.float32)
#         clashes = torch.full((B,), -1.0, device=device, dtype=torch.float32)
#
#         valid_indices = [
#             i for i in range(B)
#             if valid_mask[i]
#         ]
#
#         if max_samples is not None:
#             max_samples = int(max_samples)
#             if len(valid_indices) > max_samples:
#                 perm = torch.randperm(len(valid_indices)).tolist()
#                 valid_indices = [valid_indices[j] for j in perm[:max_samples]]
#
#         for i in valid_indices:
#             reward, rmsd, clash = self.score_one(
#                 mol=mols[i],
#                 name=names[i],
#                 p1_path=p1_paths[i],
#                 p2_path=p2_paths[i],
#                 unbound_lig1_path=unbound_lig1_paths[i],
#                 unbound_lig2_path=unbound_lig2_paths[i],
#                 lig1_mask_path=lig1_mask_paths[i],
#                 lig2_mask_path=lig2_mask_paths[i],
#                 step=step,
#                 index=i,
#             )
#
#             rewards[i] = float(reward)
#
#             if rmsd is not None and math.isfinite(rmsd):
#                 rmsds[i] = float(rmsd)
#
#             if clash is not None and math.isfinite(clash):
#                 clashes[i] = float(clash)
#
#         valid_metric_mask = (rmsds >= 0) & (clashes >= 0)
#
#         num_scored = valid_metric_mask.float().sum()
#
#         if valid_metric_mask.any():
#             rmsd_mean = rmsds[valid_metric_mask].mean()
#             clash_mean = clashes[valid_metric_mask].mean()
#         else:
#             rmsd_mean = torch.tensor(-1.0, device=device, dtype=torch.float32)
#             clash_mean = torch.tensor(-1.0, device=device, dtype=torch.float32)
#
#         scored_rate = num_scored / max(float(B), 1.0)
#
#         return rewards, {
#             # 每个样本的 reward：失败样本仍然是 0
#             "ternary_reward": rewards.detach(),
#
#             # 保留 per-sample 原始值，供最终 test 统计使用
#             "deepternary_pred_p2_rmsd_raw": rmsds.detach(),
#             "deepternary_clash_ratio_raw": clashes.detach(),
#
#             # 训练日志用：只在 successfully scored 样本上平均
#             "deepternary_pred_p2_rmsd": rmsd_mean.detach(),
#             "deepternary_clash_ratio": clash_mean.detach(),
#
#             # 成功 scored 数量和比例
#             "deepternary_num_scored": num_scored.detach(),
#             "deepternary_scored_rate": scored_rate.detach(),
#         }

import math
import os
import re
import subprocess
import sys
import shutil
import importlib
import contextlib
import pandas as pd
import toml
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from concurrent.futures import ThreadPoolExecutor, as_completed

PROTAC_STAN_DESCRIPTOR_COLUMNS = [
    "Molecular Weight",
    "Exact Mass",
    "XLogP3",
    "Heavy Atom Count",
    "Ring Count",
    "Hydrogen Bond Acceptor Count",
    "Hydrogen Bond Donor Count",
    "Rotatable Bond Count",
    "Topological Polar Surface Area",
]


def calculate_molecular_descriptors(smiles):
    """
    根据 SMILES 计算 PROTAC-STAN 需要的分子描述符。
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    desc = {
        "Molecular Weight": Descriptors.MolWt(mol),
        "Exact Mass": Descriptors.ExactMolWt(mol),
        "XLogP3": Descriptors.MolLogP(mol),
        "Heavy Atom Count": Descriptors.HeavyAtomCount(mol),
        "Ring Count": Descriptors.RingCount(mol),
        "Hydrogen Bond Acceptor Count": Descriptors.NumHAcceptors(mol),
        "Hydrogen Bond Donor Count": Descriptors.NumHDonors(mol),
        "Rotatable Bond Count": Descriptors.NumRotatableBonds(mol),
        "Topological Polar Surface Area": Descriptors.TPSA(mol),
    }

    return desc
def import_module_from_file(module_name, file_path, extra_sys_path=None):
    """
    从指定文件路径导入模块，避免和当前项目里的 model / data 模块冲突。
    """
    old_sys_path = list(sys.path)

    if extra_sys_path is not None:
        sys.path.insert(0, extra_sys_path)

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)

        # 注册到 sys.modules，方便模块内部相互 import
        sys.modules[module_name] = module

        spec.loader.exec_module(module)

    finally:
        sys.path = old_sys_path

    return module
@contextlib.contextmanager
def suppress_stdout_stderr(enabled=True):
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
class ProtacStanReward:
    def __init__(
            self,
            project_dir,
            model_path,
            config_path,
            esm_map_path,
            tmp_root,
            device,
            descriptor_stats_path,
            quiet=True,
    ):
        """
        project_dir: PROTAC-STAN-main 路径
        model_path: saved_models/protac-stan.pt
        config_path: config.toml
        esm_map_path: 你刚生成的 esm_s_map.pkl
        tmp_root: RL 过程中临时保存 PROTAC-STAN 输入的位置
        """

        self.project_dir = os.path.abspath(project_dir)
        self.model_path = os.path.abspath(model_path)
        self.config_path = os.path.abspath(config_path)
        self.esm_map_path = os.path.abspath(esm_map_path)
        self.tmp_root = os.path.abspath(tmp_root)
        self.device = device
        self.descriptor_stats_path = os.path.abspath(descriptor_stats_path)
        self.quiet = quiet

        os.makedirs(self.tmp_root, exist_ok=True)

        # PROTACData.process() 会读取 root/esm_s_map.pkl
        dst_esm = os.path.join(self.tmp_root, "esm_s_map.pkl")

        # 每次初始化都覆盖，避免 tmp_root 里残留旧 esm_s_map.pkl
        shutil.copyfile(self.esm_map_path, dst_esm)

        print(f"[ProtacStanReward] copied esm map:")
        print(f"  from: {self.esm_map_path}")
        print(f"  to:   {dst_esm}")

        # 动态导入 PROTAC-STAN 的 model 和 data_loader
        model_file = os.path.join(self.project_dir, "model.py")
        data_file = os.path.join(self.project_dir, "data.py")
        loader_file = os.path.join(self.project_dir, "data_loader.py")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Cannot find PROTAC-STAN model.py: {model_file}")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Cannot find PROTAC-STAN data.py: {data_file}")

        if not os.path.exists(loader_file):
            raise FileNotFoundError(f"Cannot find PROTAC-STAN data_loader.py: {loader_file}")

        # 备份可能已经存在的 data 模块，避免污染当前项目
        old_data_module = sys.modules.get("data", None)

        try:
            # 关键：先把 PROTAC-STAN 的 data.py 注册成 data
            # 这样 data_loader.py 里的 from data import PROTACData 才会导入正确
            self.protac_data_module = import_module_from_file(
                module_name="data",
                file_path=data_file,
                extra_sys_path=self.project_dir,
            )

            self.protac_loader_module = import_module_from_file(
                module_name="protac_stan_data_loader",
                file_path=loader_file,
                extra_sys_path=self.project_dir,
            )

        finally:
            # 还原原来的 data 模块，避免影响 LinkerDiff 自己的 data
            if old_data_module is not None:
                sys.modules["data"] = old_data_module
            else:
                sys.modules.pop("data", None)

        # model.py 用独立名字导入，避免和 LinkerDiff/model 冲突
        self.protac_model_module = import_module_from_file(
            module_name="protac_stan_model",
            file_path=model_file,
            extra_sys_path=self.project_dir,
        )

        PROTAC_STAN = self.protac_model_module.PROTAC_STAN
        self.PROTACLoader = self.protac_loader_module.PROTACLoader

        cfg = toml.load(self.config_path)
        model_cfg = cfg["model"]

        self.model = PROTAC_STAN(model_cfg)

        state_dict = torch.load(
            self.model_path,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        dst_stats = os.path.join(self.tmp_root, "descriptor_stats.pkl")
        shutil.copyfile(self.descriptor_stats_path, dst_stats)

        for p in self.model.parameters():
            p.requires_grad_(False)

    def _write_csv(self, smiles_list, poi_ids, e3_ids, name):
        """
        写 PROTAC-STAN 需要的 {name}.csv。
        必须包含：
            Smiles
            Uniprot
            E3 ligase Uniprot
            9 个分子描述符
        """

        rows = []

        for smi, poi, e3 in zip(smiles_list, poi_ids, e3_ids):
            desc = calculate_molecular_descriptors(smi)

            row = {
                "Smiles": smi,
                "Uniprot": str(poi),
                "E3 ligase Uniprot": str(e3),
            }

            for col in PROTAC_STAN_DESCRIPTOR_COLUMNS:
                row[col] = desc[col]

            rows.append(row)

        df = pd.DataFrame(rows)

        csv_path = os.path.join(self.tmp_root, f"{name}.csv")
        df.to_csv(csv_path, index=False)

        return csv_path

    @torch.no_grad()
    def score(self, smiles_list, poi_ids, e3_ids, step=0, batch_size=16):
        """
        smiles_list: 长度 B，其中无效分子为 None
        poi_ids:     长度 B，batch["poi_uniprot"]
        e3_ids:      长度 B，batch["e3_uniprot"]

        返回:
            scores: [B] tensor
        """

        B = len(smiles_list)

        scores = torch.zeros(
            B,
            device=self.device,
            dtype=torch.float32
        )

        valid_indices = []

        for i, smi in enumerate(smiles_list):
            if smi is not None:
                valid_indices.append(i)

        if len(valid_indices) == 0:
            return scores

        valid_smiles = [smiles_list[i] for i in valid_indices]
        valid_poi = [poi_ids[i] for i in valid_indices]
        valid_e3 = [e3_ids[i] for i in valid_indices]

        name = f"rl_reward_{step}"

        self._write_csv(
            smiles_list=valid_smiles,
            poi_ids=valid_poi,
            e3_ids=valid_e3,
            name=name,
        )

        # 删除旧 processed，防止 PyG 复用缓存
        processed_dir = os.path.join(self.tmp_root, "processed", name)
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)

        with suppress_stdout_stderr(self.quiet):
            _, test_loader = self.PROTACLoader(
                root=self.tmp_root,
                name=name,
                batch_size=min(batch_size, len(valid_indices)),
                train_ratio=0.0,
            )

        valid_scores = []

        self.model.eval()

        for data in test_loader:
            protac_data = data["protac"].to(self.device)
            e3_ligase_data = data["e3_ligase"].to(self.device)
            poi_data = data["poi"].to(self.device)

            if not torch.isfinite(protac_data.x).all():
                print("[WARN] protac_data.x contains NaN/Inf. Set this batch degradation score to 0.")
                pos_probs = torch.zeros(
                    e3_ligase_data.size(0),
                    device=self.device,
                    dtype=torch.float32,
                )
                valid_scores.extend(pos_probs.detach().cpu().tolist())
                continue

            num_graphs = e3_ligase_data.size(0)

            bad_graph_mask = torch.zeros(
                num_graphs,
                device=self.device,
                dtype=torch.bool,
            )

            # 检查 protac_data.x 中哪些 graph 有 NaN/Inf
            if not torch.isfinite(protac_data.x).all():
                print("[WARN] protac_data.x contains NaN/Inf.")

                bad_node_mask = ~torch.isfinite(protac_data.x).all(dim=1)

                if hasattr(protac_data, "batch"):
                    bad_graph_idx = protac_data.batch[bad_node_mask].unique()
                    bad_graph_mask[bad_graph_idx] = True
                    print("bad graph idx:", bad_graph_idx.detach().cpu().tolist())
                else:
                    bad_graph_mask[:] = True

                # 打印坏特征维度，方便定位来源
                bad_dim_idx = torch.where((~torch.isfinite(protac_data.x)).any(dim=0))[0]
                print("bad feature dims:", bad_dim_idx.detach().cpu().tolist())
                print("num bad values:", (~torch.isfinite(protac_data.x)).sum().item())

                # 为了让 model forward 不崩，先把 NaN/Inf 替换成 0
                protac_data.x = torch.nan_to_num(
                    protac_data.x,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

            # edge_attr 也检查一下
            if hasattr(protac_data, "edge_attr") and protac_data.edge_attr is not None:
                if not torch.isfinite(protac_data.edge_attr).all():
                    print("[WARN] protac_data.edge_attr contains NaN/Inf.")
                    protac_data.edge_attr = torch.nan_to_num(
                        protac_data.edge_attr,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

            outputs, _ = self.model(
                protac_data,
                e3_ligase_data,
                poi_data,
                mode="eval",
            )

            # 输出也按样本检查
            finite_output_mask = torch.isfinite(outputs).all(dim=1)

            valid_score_mask = finite_output_mask & (~bad_graph_mask)

            pos_probs = torch.zeros(
                outputs.size(0),
                device=self.device,
                dtype=torch.float32,
            )

            if valid_score_mask.any():
                probs = torch.softmax(outputs[valid_score_mask], dim=1)
                probs = torch.nan_to_num(
                    probs,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                pos_probs[valid_score_mask] = probs[:, 1]

            if (~valid_score_mask).any():
                print(
                    "[WARN] Some PROTAC-STAN samples are invalid. "
                    "Set their degradation scores to 0."
                )

            valid_scores.extend(pos_probs.detach().cpu().tolist())

        for idx, score in zip(valid_indices, valid_scores):
            scores[idx] = float(score)

        return scores

DEEPTERNARY_PATH_KEYS = (
    "p1_path",
    "p2_path",
    "unbound_lig1_path",
    "unbound_lig2_path",
    "lig1_mask_path",
    "lig2_mask_path",
)


def is_valid_optional_path(v):
    if v is None:
        return False

    try:
        if pd.isna(v):
            return False
    except Exception:
        pass

    s = str(v).strip()

    if s == "":
        return False

    if s.lower() in {"none", "nan", "null"}:
        return False

    return True


def has_complete_deepternary_paths(
        p1_path,
        p2_path,
        unbound_lig1_path,
        unbound_lig2_path,
        lig1_mask_path,
        lig2_mask_path,
):
    return all([
        is_valid_optional_path(p1_path),
        is_valid_optional_path(p2_path),
        is_valid_optional_path(unbound_lig1_path),
        is_valid_optional_path(unbound_lig2_path),
        is_valid_optional_path(lig1_mask_path),
        is_valid_optional_path(lig2_mask_path),
    ])

def safe_name(name):
    name = str(name)
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    return name


def write_mol_to_sdf(mol, out_path):
    """
    把生成的 RDKit Mol 写成 SDF，供 DeepTernary 读取。
    SDF 比 PDB 更适合小分子，可以减少 kekulize / 键类型问题。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        mol = Chem.Mol(mol)
        mol.UpdatePropertyCache(strict=False)

        writer = Chem.SDWriter(str(out_path))

        if hasattr(writer, "SetKekulize"):
            writer.SetKekulize(False)

        writer.write(mol)
        writer.close()

        return True

    except Exception as e:
        print(f"[DeepTernaryReward] Failed to write ligand sdf: {e}")
        return False

def write_mol_to_pdb(mol, out_path):
    """
    把生成的完整 PROTAC ligand 写成 PDB，供 DeepTernary 的 --lig 使用。
    注意：--lig 不能用 SDF，因为 predict.py 后面 set_new_coords() 只接受 PDB。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        mol = Chem.Mol(mol)
        mol.UpdatePropertyCache(strict=False)

        Chem.MolToPDBFile(mol, str(out_path))
        return True

    except Exception as e:
        print(f"[DeepTernaryReward] Failed to write ligand pdb: {e}")
        return False

class DeepTernaryReward:
    def __init__(
            self,
            python_exe,
            project_dir,
            predict_script,
            tmp_root,
            seeds=1,
            inference_seed=None,
            timeout=600,
            device="cuda",
            num_workers=4,
    ):
        """
        python_exe:
            DeepTernary 所在环境的 python.exe
            例如 D:/study/Anaconda3/envs/linker/python.exe

        project_dir:
            DeepTernary-master 根目录

        predict_script:
            DeepTernary-master/predict.py

        tmp_root:
            临时输出目录

        seeds:
            DeepTernary 采样种子数量。RL 阶段先用 1。

        timeout:
            subprocess 超时时间，单位秒。
        """

        if python_exe is None or str(python_exe).lower() == "current":
            self.python_exe = sys.executable
        else:
            self.python_exe = os.path.abspath(python_exe)
        self.project_dir = os.path.abspath(project_dir)
        self.predict_script = os.path.abspath(predict_script)
        self.tmp_root = os.path.abspath(tmp_root)
        self.seeds = int(seeds)
        if inference_seed is None:
            raise ValueError(
                "DeepTernary inference_seed must be explicitly provided. "
                "Refusing to use an implicit/default seed."
            )
        self.inference_seed = int(inference_seed)
        self.timeout = int(timeout)
        self.device = device
        self.num_workers = max(1, int(num_workers))

        os.makedirs(self.tmp_root, exist_ok=True)

    def score_one(
        self,
        mol,
        name,
        p1_path,
        p2_path,
        unbound_lig1_path,
        unbound_lig2_path,
        lig1_mask_path,
        lig2_mask_path,
        step,
        index,
    ):
        name = safe_name(name)
        sample_name = f"{name}_step{step}_idx{index}"
        sample_dir = os.path.join(self.tmp_root, sample_name)

        # 每次同名样本先清空，避免旧 out/ 结果残留
        if os.path.exists(sample_dir):
            shutil.rmtree(sample_dir)

        outdir = os.path.join(sample_dir, "out")
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)

        lig_pdb_path = os.path.join(sample_dir, "generated_ligand.pdb")
        lig_sdf_path = os.path.join(sample_dir, "generated_ligand.sdf")

        ok_pdb = write_mol_to_pdb(mol, lig_pdb_path)
        ok_sdf = write_mol_to_sdf(mol, lig_sdf_path)

        if not ok_pdb or not ok_sdf:
            return 0.0, None, None, None

        lig_path = os.path.abspath(lig_pdb_path)
        lig_read_path = os.path.abspath(lig_sdf_path)

        input_paths = [
            p1_path,
            p2_path,
            unbound_lig1_path,
            unbound_lig2_path,
            lig1_mask_path,
            lig2_mask_path,
        ]

        if not all(is_valid_optional_path(p) for p in input_paths):
            return 0.0, None, None, None

        # 注意：batch 中的路径是相对 LinkerDiff 项目的，所以这里转成绝对路径
        p1_path = os.path.abspath(str(p1_path))
        p2_path = os.path.abspath(str(p2_path))
        unbound_lig1_path = os.path.abspath(str(unbound_lig1_path))
        unbound_lig2_path = os.path.abspath(str(unbound_lig2_path))
        lig1_mask_path = os.path.abspath(str(lig1_mask_path))
        lig2_mask_path = os.path.abspath(str(lig2_mask_path))
        outdir = os.path.abspath(outdir)

        resolved_paths = [
            p1_path,
            p2_path,
            unbound_lig1_path,
            unbound_lig2_path,
            lig1_mask_path,
            lig2_mask_path,
        ]

        if not all(os.path.isfile(p) for p in resolved_paths):
            print(
                "[DeepTernaryReward] missing experimental input file:",
                [p for p in resolved_paths if not os.path.isfile(p)]
            )
            return 0.0, None, None, None

        cmd = [
            self.python_exe,
            self.predict_script,
            "--task", "PROTAC",
            "--lig", lig_path,
            "--lig-read", lig_read_path,
            # PT p2 = E3 -> DeepTernary p1
            "--p1", p2_path,

            # PT p1 = POI -> DeepTernary p2
            "--p2", p1_path,

            # PT lig2 = E3 -> DeepTernary lig1
            "--unbound-lig1", unbound_lig2_path,

            # PT lig1 = POI -> DeepTernary lig2
            "--unbound-lig2", unbound_lig1_path,

            "--lig1-mask", lig2_mask_path,
            "--lig2-mask", lig1_mask_path,
            "--outdir", outdir,
            "--name", sample_name,
            "--seeds", str(self.seeds),
            # Actual base RNG seed
            "--seed", str(self.inference_seed),
            "--device", 'cpu',
            "--no-correct",
        ]

        try:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"

            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )

            summary_path = os.path.join(outdir, f"summary_{sample_name}.csv")

            if result.returncode != 0 and not os.path.exists(summary_path):
                print("=" * 100)
                print("[DeepTernaryReward] subprocess failed and summary csv not found")
                # print("[returncode]", result.returncode)
                # print("[cwd]", self.project_dir)
                # print("[cmd]", " ".join(cmd))
                # print("-" * 100)
                # print("[stdout]")
                # print(result.stdout[-4000:] if result.stdout else "<EMPTY>")
                # print("-" * 100)
                # print("[stderr]")
                # print(result.stderr[-4000:] if result.stderr else "<EMPTY>")
                print("=" * 100)
                return 0.0, None, None, None

            # if result.returncode != 0 and os.path.exists(summary_path):
            #     print(
            #         f"[DeepTernaryReward] WARN: subprocess returncode={result.returncode}, "
            #         f"but summary csv exists. Use summary csv anyway."
            #     )

            if not os.path.exists(summary_path):
                print("[DeepTernaryReward] summary csv not found:", summary_path)
                return 0.0, None, None, None

            df = pd.read_csv(summary_path)

            if len(df) == 0:
                return 0.0, None, None, None

            pred_p2_rmsd = float(df.iloc[0]["pred_p2_rmsd"])
            clash_ratio = float(df.iloc[0]["clash_ratio"])

            complex_pred_path = None

            if "complex_pred" in df.columns:
                complex_pred_value = str(df.iloc[0]["complex_pred"])

                if complex_pred_value not in ["", "None", "nan"]:
                    if os.path.isabs(complex_pred_value):
                        complex_pred_path = complex_pred_value
                    else:
                        complex_pred_path = os.path.abspath(
                            os.path.join(outdir, complex_pred_value)
                        )

                    if not os.path.exists(complex_pred_path):
                        complex_pred_path = None

            if not math.isfinite(pred_p2_rmsd) or not math.isfinite(clash_ratio):
                return 0.0, pred_p2_rmsd, clash_ratio, complex_pred_path

            # RMSD 越小越好，clash_ratio 越小越好
            reward = math.exp(-pred_p2_rmsd / 2.0) * math.exp(-5.0 * clash_ratio)

            reward = float(max(0.0, min(1.0, reward)))

            return reward, pred_p2_rmsd, clash_ratio, complex_pred_path

        except subprocess.TimeoutExpired:
            print("[DeepTernaryReward] timeout")
            return 0.0, None, None, None

        except Exception as e:
            print(f"[DeepTernaryReward] exception: {e}")
            return 0.0, None, None, None

    @torch.no_grad()
    def score(
        self,
        mols,
        valid_mask,
        names,
        p1_paths,
        p2_paths,
        unbound_lig1_paths,
        unbound_lig2_paths,
        lig1_mask_paths,
        lig2_mask_paths,
        step,
        max_samples=None,
        device="cuda",
    ):
        """
        mols: list[RDKit Mol], 长度 B
        valid_mask: list[bool]，只有合法且连通的分子才调用 DeepTernary
        """

        B = len(mols)
        context_eligible_mask = [
            has_complete_deepternary_paths(
                p1_paths[i],
                p2_paths[i],
                unbound_lig1_paths[i],
                unbound_lig2_paths[i],
                lig1_mask_paths[i],
                lig2_mask_paths[i],
            )
            for i in range(B)
        ]

        rewards = torch.zeros(B, device=device, dtype=torch.float32)
        rmsds = torch.full((B,), -1.0, device=device, dtype=torch.float32)
        clashes = torch.full((B,), -1.0, device=device, dtype=torch.float32)
        complex_pred_paths = [None for _ in range(B)]

        attempted_mask = torch.zeros(B, device=device, dtype=torch.float32)
        eligible_context_mask = torch.tensor(
            context_eligible_mask,
            device=device,
            dtype=torch.float32,
        )

        strict_eligible_mask = [
            bool(valid_mask[i]) and bool(context_eligible_mask[i])
            for i in range(B)
        ]

        valid_candidate_mask = torch.tensor(
            strict_eligible_mask,
            device=device,
            dtype=torch.float32,
        )

        valid_indices = [
            i for i in range(B)
            if strict_eligible_mask[i]
        ]

        if max_samples is not None:
            max_samples = int(max_samples)
            if len(valid_indices) > max_samples:
                perm = torch.randperm(len(valid_indices)).tolist()
                valid_indices = [valid_indices[j] for j in perm[:max_samples]]

        for i in valid_indices:
            attempted_mask[i] = 1.0

        def _run_one(i):
            reward, rmsd, clash, complex_pred_path = self.score_one(
                mol=mols[i],
                name=names[i],
                p1_path=p1_paths[i],
                p2_path=p2_paths[i],
                unbound_lig1_path=unbound_lig1_paths[i],
                unbound_lig2_path=unbound_lig2_paths[i],
                lig1_mask_path=lig1_mask_paths[i],
                lig2_mask_path=lig2_mask_paths[i],
                step=step,
                index=i,
            )

            return i, reward, rmsd, clash, complex_pred_path

        if len(valid_indices) > 0:
            n_workers = min(self.num_workers, len(valid_indices))

            if n_workers <= 1:
                results_iter = [_run_one(i) for i in valid_indices]
            else:
                results_iter = []
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    future_to_i = {
                        executor.submit(_run_one, i): i
                        for i in valid_indices
                    }

                    for future in as_completed(future_to_i):
                        i = future_to_i[future]
                        try:
                            results_iter.append(future.result())
                        except Exception as e:
                            print(f"[DeepTernaryReward] parallel worker failed at index={i}: {e}")
                            results_iter.append((i, 0.0, None, None, None))

            for i, reward, rmsd, clash, complex_pred_path in results_iter:
                rewards[i] = float(reward)

                if rmsd is not None and math.isfinite(rmsd):
                    rmsds[i] = float(rmsd)

                if clash is not None and math.isfinite(clash):
                    clashes[i] = float(clash)

                complex_pred_paths[i] = complex_pred_path

        valid_metric_mask = (rmsds >= 0) & (clashes >= 0)
        scored_mask = valid_metric_mask.float()

        num_scored = scored_mask.sum()

        num_valid_candidates = valid_candidate_mask.sum().clamp(min=1.0)
        num_attempted = attempted_mask.sum().clamp(min=1.0)

        scored_rate = num_scored / max(float(B), 1.0)
        scored_rate_among_valid = num_scored / num_valid_candidates
        scored_rate_among_attempted = num_scored / num_attempted
        attempted_rate_among_valid = attempted_mask.sum() / num_valid_candidates

        if valid_metric_mask.any():
            rmsd_mean = rmsds[valid_metric_mask].mean()
            clash_mean = clashes[valid_metric_mask].mean()
        else:
            rmsd_mean = torch.tensor(-1.0, device=device, dtype=torch.float32)
            clash_mean = torch.tensor(-1.0, device=device, dtype=torch.float32)

        return rewards, {
            # 每个样本的 reward：失败样本仍然是 0
            "ternary_reward": rewards.detach(),

            # 保留 per-sample 原始值，供最终 test 统计使用
            "deepternary_pred_p2_rmsd_raw": rmsds.detach(),
            "deepternary_clash_ratio_raw": clashes.detach(),

            # 训练日志用：只在 successfully scored 样本上平均
            "deepternary_pred_p2_rmsd": rmsd_mean.detach(),
            "deepternary_clash_ratio": clash_mean.detach(),

            # 成功 scored 数量和比例
            "deepternary_num_scored": num_scored.detach(),
            "deepternary_scored_rate": scored_rate.detach(),

            # mask
            "deepternary_scored_mask": scored_mask.detach(),
            "deepternary_attempted_mask": attempted_mask.detach(),
            "deepternary_valid_candidate_mask": valid_candidate_mask.detach(),

            # predicted ternary complex pdb path
            "deepternary_complex_pred_path": complex_pred_paths,

            "deepternary_context_eligible_mask":
                eligible_context_mask.detach(),

            # 数量和比例
            "deepternary_scored_rate_among_valid": scored_rate_among_valid.detach(),
            "deepternary_scored_rate_among_attempted": scored_rate_among_attempted.detach(),
            "deepternary_attempted_rate_among_valid": attempted_rate_among_valid.detach(),
        }