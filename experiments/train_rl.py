import logging
import math
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import pickle
from collections import defaultdict

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from torch.utils.data import DataLoader

from data import datasets
from data.datasets import collate
from data.molecule_builder import build_molecules
from experiments.train import Experiment
from model import LinkerDiffusionModel
from src import const, utils
from model.reward_models import ProtacStanReward, DeepTernaryReward
import random
from experiments.train import get_atom_type_ids_by_symbols

def seed_everything(seed: int):
    """
    Set random seed for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
def safe_mol_to_smiles(mol):
    if mol is None:
        return None

    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        smi = Chem.MolToSmiles(mol_copy)

        if smi is None or len(smi) == 0:
            return None

        return smi

    except Exception:
        return None


def is_connected_mol(mol):
    try:
        frags = Chem.GetMolFrags(
            mol,
            asMols=False,
            sanitizeFrags=False
        )
        return len(frags) == 1
    except Exception:
        return False
def _mask_to_bool_numpy(mask):
    """
    mask: torch.Tensor or np.ndarray
        shape can be [N], [N, 1]
    return: np.ndarray bool, shape [N]
    """
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    mask = np.asarray(mask).squeeze()
    return mask.astype(bool)


def is_strict_protac_mol(
        mol,
        anchor1_mask,
        anchor2_mask,
        linker_mask,
        fragment1_mask,
        fragment2_mask,
):
    """
    Check strict PROTAC topology validity.

    Require RDKit sanitization, full-molecule connectivity,
    exactly one non-aromatic linker attachment per anchor,
    a connected anchor-linker bridge, and an internally connected linker.

    Reject direct fragment1-fragment2 bonds and non-anchor
    fragment-linker attachments.
    """

    info = {
        "sanitize_ok": 0.0,
        "connected": 0.0,
        "exact_one_a1_linker": 0.0,
        "exact_one_a2_linker": 0.0,
        "anchor_linker_non_aromatic": 0.0,
        "bridge_component_ok": 0.0,
        "linker_internal_connected": 0.0,
        "no_fragment1_fragment2_bond": 0.0,
        "no_bad_non_anchor_fragment_linker_bond": 0.0,
        "strict_protac": 0.0,
    }

    if mol is None:
        return False, info

    # ------------------------------------------------
    # 1. sanitize
    # ------------------------------------------------
    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        info["sanitize_ok"] = 1.0
    except Exception:
        return False, info

    # ------------------------------------------------
    # 2. whole molecule connected
    # ------------------------------------------------
    if not is_connected_mol(mol):
        return False, info

    info["connected"] = 1.0

    anchor1_mask = _mask_to_bool_numpy(anchor1_mask)
    anchor2_mask = _mask_to_bool_numpy(anchor2_mask)
    linker_mask = _mask_to_bool_numpy(linker_mask)
    fragment1_mask = _mask_to_bool_numpy(fragment1_mask)
    fragment2_mask = _mask_to_bool_numpy(fragment2_mask)

    anchor1_idx = np.where(anchor1_mask)[0].tolist()
    anchor2_idx = np.where(anchor2_mask)[0].tolist()
    linker_idx = np.where(linker_mask)[0].tolist()

    if len(anchor1_idx) != 1 or len(anchor2_idx) != 1:
        return False, info

    if len(linker_idx) == 0:
        return False, info

    a1 = int(anchor1_idx[0])
    a2 = int(anchor2_idx[0])
    linker_idx = [int(x) for x in linker_idx]

    n_atoms = mol.GetNumAtoms()

    if a1 >= n_atoms or a2 >= n_atoms:
        return False, info

    if max(linker_idx) >= n_atoms:
        return False, info

    # ------------------------------------------------
    # 3. anchor-linker bonds
    # ------------------------------------------------
    a1_linker_bonds = []
    a2_linker_bonds = []

    anchor_linker_non_aromatic = True

    for l in linker_idx:
        b1 = mol.GetBondBetweenAtoms(a1, l)
        if b1 is not None:
            a1_linker_bonds.append((a1, l, b1))

            if b1.GetIsAromatic() or b1.GetBondType() == Chem.BondType.AROMATIC:
                anchor_linker_non_aromatic = False

        b2 = mol.GetBondBetweenAtoms(a2, l)
        if b2 is not None:
            a2_linker_bonds.append((a2, l, b2))

            if b2.GetIsAromatic() or b2.GetBondType() == Chem.BondType.AROMATIC:
                anchor_linker_non_aromatic = False

    if len(a1_linker_bonds) == 1:
        info["exact_one_a1_linker"] = 1.0

    if len(a2_linker_bonds) == 1:
        info["exact_one_a2_linker"] = 1.0

    if anchor_linker_non_aromatic:
        info["anchor_linker_non_aromatic"] = 1.0

    if len(a1_linker_bonds) != 1:
        return False, info

    if len(a2_linker_bonds) != 1:
        return False, info

    if not anchor_linker_non_aromatic:
        return False, info

    # ------------------------------------------------
    # 4. build graph adjacency
    # ------------------------------------------------
    adjacency = {i: [] for i in range(n_atoms)}

    has_fragment1_fragment2_bond = False
    has_bad_non_anchor_fragment_linker_bond = False

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        adjacency[i].append(j)
        adjacency[j].append(i)
        if i >= n_atoms or j >= n_atoms:
            continue

            # Direct fragment1-fragment2 bonds are forbidden.
        if (fragment1_mask[i] and fragment2_mask[j]) or (fragment2_mask[i] and fragment1_mask[j]):
            has_fragment1_fragment2_bond = True

            # Fragment1 may connect to the linker only through anchor1.
        if (fragment1_mask[i] and linker_mask[j]) or (fragment1_mask[j] and linker_mask[i]):
            frag_atom = i if fragment1_mask[i] else j
            if frag_atom != a1:
                has_bad_non_anchor_fragment_linker_bond = True

            # Fragment2 may connect to the linker only through anchor2.
        if (fragment2_mask[i] and linker_mask[j]) or (fragment2_mask[j] and linker_mask[i]):
            frag_atom = i if fragment2_mask[i] else j
            if frag_atom != a2:
                has_bad_non_anchor_fragment_linker_bond = True

    # Verify that both anchors and all linker atoms are connected
    # within the anchor-linker induced subgraph.
    bridge_nodes = set(linker_idx + [a1, a2])

    stack = [a1]
    visited = set([a1])

    while stack:
        cur = stack.pop()

        for nb in adjacency[cur]:
            if nb not in bridge_nodes:
                continue

            if nb not in visited:
                visited.add(nb)
                stack.append(nb)

    if a2 not in visited:
        return False, info

    if not all(l in visited for l in linker_idx):
        return False, info

    info["bridge_component_ok"] = 1.0

    # ------------------------------------------------
    # 6. linker internal connected
    # ------------------------------------------------
    if len(linker_idx) == 1:
        info["linker_internal_connected"] = 1.0
    else:
        linker_nodes = set(linker_idx)

        stack = [linker_idx[0]]
        visited_linker = set([linker_idx[0]])

        while stack:
            cur = stack.pop()

            for nb in adjacency[cur]:
                if nb not in linker_nodes:
                    continue

                if nb not in visited_linker:
                    visited_linker.add(nb)
                    stack.append(nb)

        if all(l in visited_linker for l in linker_idx):
            info["linker_internal_connected"] = 1.0
        else:
            return False, info

    if has_fragment1_fragment2_bond:
        return False, info
    info["no_fragment1_fragment2_bond"] = 1.0

    if has_bad_non_anchor_fragment_linker_bond:
        return False, info
    info["no_bad_non_anchor_fragment_linker_bond"] = 1.0

    info["strict_protac"] = 1.0
    return True, info
def masked_gaussian_log_prob(z, mu, sigma, mask):
    """
    z:     [B, N, D]
    mu:    [B, N, D]
    sigma: [B, N, D] or broadcastable
    mask:  [B, N, 1]

    return:
        log_prob: [B]
    """
    log_prob = -0.5 * ((z - mu) / sigma) ** 2
    log_prob = log_prob - torch.log(sigma)
    log_prob = log_prob - 0.5 * math.log(2.0 * math.pi)

    log_prob = log_prob * mask

    return log_prob.sum(dim=(1, 2))

class RLExperiment:
    def __init__(
            self,
            *,
            conf: DictConfig,
        ):

        self._log = logging.getLogger(__name__)
        self._conf = conf
        self._exp_conf = conf.experiment
        self._model_conf = conf.model
        self._use_wandb = self._exp_conf.use_wandb
        if self._exp_conf.use_gpu:
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        # Initialize RL model weights from a pretrained checkpoint.
        # Optimizer state and training counters start fresh.
        ckpt_model = None
        self.trained_epochs = 0
        self.trained_steps = 0
        self.val_steps = 0
        self.T = self._exp_conf.diffusion_step

        if conf.experiment.weight_path is not None:
            ckpt_path = conf.experiment.weight_path
            self._log.info(f"Loading pretrained checkpoint from {ckpt_path}")
            ckpt_pkl = torch.load(ckpt_path, map_location=self._device)
            ckpt_model = ckpt_pkl["model"]
            conf.experiment.weight_path = ckpt_path

        self._model_conf.num_atom_types = const.NUMBER_OF_ATOM_TYPES
        self._model = LinkerDiffusionModel.LinkerDiffusionModel(
            self._model_conf, self._exp_conf)
        self._model = self._model.to(self._device)
        if ckpt_model is not None:
            # Remove the optional "module." prefix from checkpoint keys.
            ckpt_model = {k.replace('module.', ''):v for k,v in ckpt_model.items()}
            self._model.load_state_dict(ckpt_model, strict=True)

        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._exp_conf.num_parameters = num_parameters
        self._log.info(f'Model parameters: {num_parameters}')

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._exp_conf.rl_learning_rate)

        self._log.info(f"RL learning rate: {self._exp_conf.rl_learning_rate}")

        if self._exp_conf.ckpt_dir is not None:
            ckpt_dir = os.path.join(
                self._exp_conf.ckpt_dir,
                self._exp_conf.name,
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.ckpt_dir = ckpt_dir
            self._log.info(f"RL checkpoints will be saved to: {ckpt_dir}")
        else:
            self._log.info("RL checkpoint saving is disabled.")
        if self._exp_conf.eval_dir is not None:
            eval_dir = os.path.join(
                self._exp_conf.eval_dir,
                self._exp_conf.name,
            )
            os.makedirs(eval_dir, exist_ok=True)
            self._exp_conf.eval_dir = eval_dir
            self._log.info(f"RL evaluation outputs will be saved to: {eval_dir}")
        else:
            self._exp_conf.eval_dir = os.devnull
            self._log.info("RL evaluation will not be saved.")

        self.metrics = {}

        self.ppo_clip = self._exp_conf.ppo_clip
        self.max_grad_norm = self._exp_conf.max_grad_norm

        self.use_protac_stan_reward = bool(self._exp_conf.use_protac_stan_reward)

        self.protac_stan_reward = None

        if self.use_protac_stan_reward:
            self.protac_stan_reward = ProtacStanReward(
                project_dir=self._exp_conf.protac_stan_project_dir,
                model_path=self._exp_conf.protac_stan_model_path,
                config_path=self._exp_conf.protac_stan_config_path,
                esm_map_path=self._exp_conf.protac_stan_esm_map_path,
                tmp_root=self._exp_conf.protac_stan_tmp_root,
                device=self._device,
                descriptor_stats_path=self._exp_conf.protac_stan_descriptor_stats_path,
            )
        self.deepternary_reward = None

        if self._exp_conf.use_deepternary_reward:
            self.deepternary_reward = DeepTernaryReward(
                python_exe=self._exp_conf.deepternary_python,
                project_dir=self._exp_conf.deepternary_project_dir,
                predict_script=self._exp_conf.deepternary_predict_script,
                tmp_root=self._exp_conf.deepternary_tmp_root,
                seeds=self._exp_conf.deepternary_seeds,
                inference_seed=2026,
                timeout=self._exp_conf.deepternary_timeout,
                device=self._device,
            )

        self._log.info("RLExperiment initialized.")
    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf
    @property
    def device(self):
        return self._device

    def start_rl_training(self):
        self.model.train()
        self._log.info(f"Using device: {self._device}")
        train_loader, _ = self.create_dataset()
        num_rl_epoch = self._exp_conf.num_rl_epoch

        for epoch in range(num_rl_epoch):
            self._log.info(
                f"========== RL Epoch {epoch + 1}/{num_rl_epoch} =========="
            )

            # =========================
            # 1. RL train
            # =========================
            epoch_log = self.rl_train_epoch(train_loader)

            for k, v in epoch_log.items():
                self.metrics.setdefault(f"{k}/rl_train_epoch", []).append(v)

            self._log.info(
                "[RL Epoch {}] ".format(epoch + 1)
                + " ".join([f"{k}={v:.4f}" for k, v in epoch_log.items()])
            )

            # =========================
            # 4. Save epoch checkpoint
            # =========================
            if self._exp_conf.ckpt_dir is not None:

                ckpt_path = os.path.join(
                    self._exp_conf.ckpt_dir,
                    f"rl_epoch_{epoch + 1}.pth"
                )

                utils.write_checkpoint(
                    ckpt_path,
                    {
                        k: v.detach().cpu()
                        for k, v in self.model.state_dict().items()
                    },
                    self._conf,
                    None,
                    epoch + 1,
                    self.trained_steps,
                    logger=self._log,
                    use_torch=True
                )

                self._log.info(f"RL checkpoint saved: {ckpt_path}")

        with open("./rl_metrics.pkl", "wb") as f:
            pickle.dump(self.metrics, f)


    def rl_train_epoch(self, train_loader):
        log_losses = defaultdict(list)

        for batch_idx, batch in enumerate(train_loader):

            # Use eval mode for both rollout and PPO policy re-evaluation.
            self.model.eval()

            rollout_out = self.rollout_sample_from_batch(
                self.model,
                batch,
                keep_frames=2,
            )

            reward, reward_info = self.compute_reward(
                rollout_out
            )

            loss, aux_data = self.ppo_update(
                rollout_out=rollout_out,
                reward=reward,
            )

            # Restore training mode after the PPO update.
            self.model.train()

            log_dict = {
                **aux_data,
                **reward_info,
            }

            for k, v in log_dict.items():
                value = float(v.mean().detach().cpu()) if torch.is_tensor(v) else float(v)
                log_losses[k].append(value)

            self.trained_steps += 1

            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:
                msg = " ".join([
                    f"{k}={np.mean(v):.4f}"
                    for k, v in log_losses.items()
                ])
                self._log.info(f"[RL step {self.trained_steps}] {msg}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            k: float(np.mean(v))
            for k, v in log_losses.items()
        }


    def create_dataset(self):
        train_dataset = datasets.RlDataset(
            data_path=self._exp_conf.data_path,
            dir='train',
            prefix=self._exp_conf.prefix_rltrain,
            device="cpu"
        )

        valid_dataset = datasets.RlDataset(
            data_path=self._exp_conf.data_path,
            dir='val',
            prefix=self._exp_conf.prefix_rlval,
            device="cpu"
        )

        seed = int(self._exp_conf.seed)

        train_generator = torch.Generator()
        train_generator.manual_seed(seed)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self._exp_conf.rl_batch_size,
            collate_fn=collate,
            shuffle=True,
            generator=train_generator,
            num_workers=0,
            pin_memory=False,
        )

        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=self._exp_conf.rl_batch_size,
            collate_fn=collate,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        return train_loader, valid_loader

    def ppo_update(self, rollout_out, reward):
        reward = torch.nan_to_num(
            reward.detach(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        zero = torch.tensor(0.0, device=self._device)

        if not torch.isfinite(reward).all():
            print("[WARN] reward is not finite after nan_to_num, skip PPO update.")
            return zero, {
                "ppo_loss": zero,
                "policy_loss": zero,
                "critic_loss": zero,
                "approx_kl": zero,
                "entropy": zero,
                "clip_frac": zero,
                "ratio_mean": zero,
                "reward_mean": reward.mean(),
                "reward_std": reward.std(unbiased=False),
                "advantage_mean": zero,
                "advantage_std": zero,
                "value_mean": zero,
                "value_std": zero,
            }

        traj = rollout_out["traj"]
        batch = rollout_out["batch"]

        node_mask = batch["atom_mask"]
        linker_mask = batch["linker_mask"]

        anchor1_mask = batch["anchor1_mask"]
        anchor2_mask = batch["anchor2_mask"]
        fragment1_mask = batch["fragment1_mask"]
        fragment2_mask = batch["fragment2_mask"]

        fragment_mask = (
                fragment1_mask.bool() | fragment2_mask.bool()
        ).to(linker_mask.dtype)

        B = batch["positions"].size(0)
        N = batch["positions"].size(1)
        edge_mask = batch["edge_mask"].view(B, N, N)

        target_value = reward.detach()

        num_total_steps = len(traj)

        k = int(self._exp_conf.ppo_t_indices)
        k = min(k, num_total_steps)

        t_indices = torch.linspace(
            0,
            num_total_steps - 1,
            steps=k,
            device=self._device,
        ).long().tolist()

        ppo_epochs = int(self._exp_conf.ppo_epochs)

        kl_coef = self._exp_conf.kl_coef
        entropy_coef = self._exp_conf.entropy_coef
        clip_eps = self._exp_conf.ppo_clip
        vf_coef = self._exp_conf.vf_coef

        # Compute advantages once using critic values before PPO updates.
        # Reuse these fixed advantages across PPO epochs while
        # recomputing the value loss with the current critic.
        fixed_advantages = {}

        with torch.no_grad():

            for idx in t_indices:

                step = traj[idx]

                z_t = step["z_t"]
                t = step["t"]

                _, _, old_value = self.flow_policy_params(
                    model=self.model,
                    z_t=z_t,
                    t=t,
                    node_mask=node_mask,
                    linker_mask=linker_mask,
                    anchor1_mask=anchor1_mask,
                    anchor2_mask=anchor2_mask,
                    fragment1_mask=fragment1_mask,
                    fragment2_mask=fragment2_mask,
                    edge_mask=edge_mask,
                    return_value=True,
                )

                old_value = torch.nan_to_num(
                    old_value,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                advantage = (
                    target_value
                    - old_value
                )

                adv_std = advantage.std(
                    unbiased=False
                )

                if (
                    advantage.numel() > 1
                    and adv_std > 1e-6
                ):
                    advantage = (
                        advantage
                        - advantage.mean()
                    ) / (
                        adv_std + 1e-8
                    )
                else:
                    advantage = torch.clamp(
                        advantage,
                        min=-5.0,
                        max=5.0,
                    )

                fixed_advantages[idx] = (
                    advantage.detach()
                )

        # ------------------------------------------------
        # PPO multi-epoch update
        # ------------------------------------------------
        log_total_loss = 0.0
        log_policy_loss = 0.0
        log_value_loss = 0.0
        log_kl = 0.0
        log_entropy = 0.0
        log_clip_frac = 0.0
        ratio_mean_accum = 0.0
        log_value_mean = 0.0
        log_value_std = 0.0
        log_advantage_mean = 0.0
        log_advantage_std = 0.0

        num_update_terms = 0
        last_total_loss = None

        for ppo_epoch in range(ppo_epochs):

            self._optimizer.zero_grad(set_to_none=True)

            total_loss_detached = torch.tensor(
                0.0,
                device=self._device,
                dtype=torch.float32,
            )

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_kl = 0.0
            total_entropy = 0.0
            total_clip_frac = 0.0
            total_ratio = 0.0
            total_value_mean = 0.0
            total_value_std = 0.0
            total_advantage_mean = 0.0
            total_advantage_std = 0.0

            valid_ppo_update = True

            for idx in t_indices:
                step = traj[idx]

                z_t = step["z_t"]
                action = step["action"]
                t = step["t"]
                old_log_prob = step["log_prob_old"].detach()

                v_mean, _, value = self.flow_policy_params(
                    model=self.model,
                    z_t=z_t,
                    t=t,
                    node_mask=node_mask,
                    linker_mask=linker_mask,
                    anchor1_mask=anchor1_mask,
                    anchor2_mask=anchor2_mask,
                    fragment1_mask=fragment1_mask,
                    fragment2_mask=fragment2_mask,
                    edge_mask=edge_mask,
                    return_value=True,
                )

                value = torch.nan_to_num(
                    value,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                action_std = float(self._exp_conf.rl_action_std)

                new_log_prob = masked_gaussian_log_prob(
                    z=action,
                    mu=v_mean,
                    sigma=torch.full_like(action, action_std),
                    mask=linker_mask,
                )

                step_log_ratio = (
                        new_log_prob
                        - old_log_prob
                )

                step_log_ratio = torch.nan_to_num(
                    step_log_ratio,
                    nan=0.0,
                    posinf=10.0,
                    neginf=-10.0,
                )

                step_log_ratio = torch.clamp(
                    step_log_ratio,
                    min=-10.0,
                    max=10.0,
                )

                ratio = torch.exp(
                    step_log_ratio
                )

                ratio = torch.nan_to_num(
                    ratio,
                    nan=1.0,
                    posinf=1.0,
                    neginf=1.0,
                )

                # Reuse the fixed rollout-time advantage across PPO epochs.
                advantage = fixed_advantages[idx]

                surr1 = ratio * advantage
                surr2 = torch.clamp(
                    ratio,
                    1.0 - clip_eps,
                    1.0 + clip_eps,
                ) * advantage

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, target_value)

                approx_kl = (
                        (ratio - 1.0)
                        - step_log_ratio
                ).mean()

                action_dim = action.size(-1)
                n_linker_atoms = linker_mask.sum(dim=(1, 2)).float()

                entropy_per_dim = 0.5 * math.log(
                    2.0 * math.pi * math.e * (action_std ** 2)
                )

                entropy = torch.tensor(
                    entropy_per_dim,
                    device=self._device,
                    dtype=torch.float32,
                ) * action_dim * n_linker_atoms.mean()

                step_loss = (
                        policy_loss
                        + vf_coef * value_loss
                        + kl_coef * approx_kl
                        - entropy_coef * entropy
                )

                if not torch.isfinite(step_loss):
                    print("[WARN] step_loss is NaN/Inf, skip PPO update.")
                    valid_ppo_update = False
                    break

                # Backpropagate each sampled time step separately to limit memory use.
                (step_loss / float(k)).backward()

                total_loss_detached = total_loss_detached + step_loss.detach() / float(k)

                total_policy_loss = total_policy_loss + policy_loss.detach()
                total_value_loss = total_value_loss + value_loss.detach()
                total_kl = total_kl + approx_kl.detach()
                total_entropy = total_entropy + entropy.detach()
                total_clip_frac = total_clip_frac + (
                        (ratio - 1.0).abs() > clip_eps
                ).float().mean().detach()
                total_ratio = total_ratio + ratio.mean().detach()
                total_value_mean = total_value_mean + value.detach().mean()
                total_value_std = total_value_std + value.detach().std(unbiased=False)
                total_advantage_mean = total_advantage_mean + advantage.detach().mean()
                total_advantage_std = total_advantage_std + advantage.detach().std(unbiased=False)

            if not valid_ppo_update:
                self._optimizer.zero_grad(set_to_none=True)
                continue

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm,
            )

            self._optimizer.step()

            log_total_loss += total_loss_detached.detach()
            log_policy_loss += total_policy_loss / k
            log_value_loss += total_value_loss / k
            log_kl += total_kl / k
            log_entropy += total_entropy / k
            log_clip_frac += total_clip_frac / k
            ratio_mean_accum += total_ratio / k
            log_value_mean += total_value_mean / k
            log_value_std += total_value_std / k
            log_advantage_mean += total_advantage_mean / k
            log_advantage_std += total_advantage_std / k

            num_update_terms += 1
            last_total_loss = total_loss_detached.detach()

        denom_log = max(float(num_update_terms), 1.0)

        if last_total_loss is None:
            last_total_loss = zero

        aux_data = {
            "ppo_loss": log_total_loss / denom_log,
            "policy_loss": log_policy_loss / denom_log,
            "critic_loss": log_value_loss / denom_log,
            "approx_kl": log_kl / denom_log,
            "entropy": log_entropy / denom_log,
            "clip_frac": log_clip_frac / denom_log,
            "ratio_mean": ratio_mean_accum / denom_log,
            "reward_mean": reward.mean().detach(),
            "reward_std": reward.std(unbiased=False).detach(),
            "advantage_mean": log_advantage_mean / denom_log,
            "advantage_std": log_advantage_std / denom_log,
            "value_mean": log_value_mean / denom_log,
            "value_std": log_value_std / denom_log,
        }

        return last_total_loss, aux_data

    @torch.no_grad()
    def compute_reward(self, rollout_out):
        """
        reward =
            strict_topology_reward
            + strict-gated PROTAC-STAN degradation reward
            + strict-gated high-degradation indicator reward
            + optional strict-gated DeepTernary reward
        """

        batch = rollout_out["batch"]
        final_xh = rollout_out["final_xh"]

        x = final_xh[:, :, :3]
        h = final_xh[:, :, 3:]

        # Restore the original coordinate frame using the saved anchor center.
        x = x + rollout_out["anchor_com"] * batch["atom_mask"].float()

        pred_bond_probs = rollout_out["pred_bond_probs"]
        pred_bond_types = rollout_out["pred_bond_types"]

        build_kwargs = dict(
            one_hot=h,
            x=x,
            node_mask=batch["atom_mask"],
            anchors=batch["anchors"],
            linker_mask=batch["linker_mask"],
        )

        # Reconstruct generated molecules using bond-head predictions
        # when predicted bond probabilities and types are available.
        if pred_bond_probs is not None and pred_bond_types is not None:
            build_kwargs.update(
                dict(
                    pred_bond_probs=pred_bond_probs,
                    pred_bond_types=pred_bond_types,
                    use_pred_bonds=True,

                    pred_bond_threshold=self._exp_conf.rl_pred_bond_threshold,
                    pred_bond_max_new_bonds_per_atom=self._exp_conf.rl_pred_bond_max_new_bonds_per_atom,
                    pred_bond_min_dist=self._exp_conf.rl_pred_bond_min_dist,
                    pred_bond_max_dist=self._exp_conf.rl_pred_bond_max_dist,
                )
            )

        # Preserve reference fragment bonds during reconstruction.
        if "fragment_bond_types" in batch:
            build_kwargs.update(
                dict(
                    fragment_bond_types=batch["fragment_bond_types"],
                    use_template_fragment_bonds=True,
                )
            )

        # Preserve reference fragment atom properties while leaving
        # linker atom properties to the reconstruction procedure.
        if all(k in batch for k in [
            "formal_charges",
            "atom_is_aromatic",
            "num_explicit_hs",
            "no_implicit",
        ]):
            build_kwargs.update(
                dict(
                    formal_charges=batch["formal_charges"],
                    atom_is_aromatic=batch["atom_is_aromatic"],
                    num_explicit_hs=batch["num_explicit_hs"],
                    no_implicit=batch["no_implicit"],

                    # Apply reference atom properties only to fragment atoms.
                    atom_props_mask=batch["fragment_mask"],
                )
            )

        pred_molecules = build_molecules(**build_kwargs)

        smiles_list = []
        strict_protac_flags = []

        strict_info_acc = {
            "strict_sanitize_ok": [],
            "strict_connected": [],
            "strict_exact_one_a1_linker": [],
            "strict_exact_one_a2_linker": [],
            "strict_anchor_linker_non_aromatic": [],
            "strict_bridge_component_ok": [],
            "strict_linker_internal_connected": [],
            "strict_no_fragment1_fragment2_bond": [],
            "strict_no_bad_non_anchor_fragment_linker_bond": [],
        }

        for i, mol in enumerate(pred_molecules):
            smi = safe_mol_to_smiles(mol)

            if smi is None:
                smiles_list.append(None)
                strict_protac_flags.append(0.0)

                for k in strict_info_acc:
                    strict_info_acc[k].append(0.0)

                continue

            strict_protac, strict_info = is_strict_protac_mol(
                mol=mol,
                anchor1_mask=batch["anchor1_mask"][i],
                anchor2_mask=batch["anchor2_mask"][i],
                linker_mask=batch["linker_mask"][i],
                fragment1_mask=batch["fragment1_mask"][i],
                fragment2_mask=batch["fragment2_mask"][i],
            )

            # Submit only strict-valid PROTAC SMILES for degradation prediction.
            if strict_protac:
                smiles_list.append(smi)
            else:
                smiles_list.append(None)

            strict_protac_flags.append(1.0 if strict_protac else 0.0)

            strict_info_acc["strict_sanitize_ok"].append(
                strict_info["sanitize_ok"]
            )
            strict_info_acc["strict_connected"].append(
                strict_info["connected"]
            )
            strict_info_acc["strict_exact_one_a1_linker"].append(
                strict_info["exact_one_a1_linker"]
            )
            strict_info_acc["strict_exact_one_a2_linker"].append(
                strict_info["exact_one_a2_linker"]
            )
            strict_info_acc["strict_anchor_linker_non_aromatic"].append(
                strict_info["anchor_linker_non_aromatic"]
            )
            strict_info_acc["strict_bridge_component_ok"].append(
                strict_info["bridge_component_ok"]
            )
            strict_info_acc["strict_linker_internal_connected"].append(
                strict_info["linker_internal_connected"]
            )
            strict_info_acc["strict_no_fragment1_fragment2_bond"].append(
                strict_info["no_fragment1_fragment2_bond"]
            )
            strict_info_acc["strict_no_bad_non_anchor_fragment_linker_bond"].append(
                strict_info["no_bad_non_anchor_fragment_linker_bond"]
            )

        strict_protac_reward = torch.tensor(
            strict_protac_flags,
            device=self._device,
            dtype=torch.float32,
        )

        strict_info_tensors = {
            k: torch.tensor(
                v,
                device=self._device,
                dtype=torch.float32,
            )
            for k, v in strict_info_acc.items()
        }

        if self.protac_stan_reward is not None:
            raw_degradation_reward = self.protac_stan_reward.score(
                smiles_list=smiles_list,
                poi_ids=batch["poi_uniprot"],
                e3_ids=batch["e3_uniprot"],
                step=self.trained_steps,
                batch_size=self._exp_conf.protac_stan_batch_size,
            )

            raw_degradation_reward = raw_degradation_reward.to(
                device=self._device,
                dtype=torch.float32,
            )

            raw_degradation_reward = torch.nan_to_num(
                raw_degradation_reward,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        else:
            raw_degradation_reward = torch.zeros_like(strict_protac_reward)

        deepternary_valid_mask = [
            strict_flag > 0.5
            for strict_flag in strict_protac_flags
        ]
        if self.deepternary_reward is not None:
            ternary_reward, ternary_info = self.deepternary_reward.score(
                mols=pred_molecules,
                valid_mask=deepternary_valid_mask,
                names=batch["uuid"],
                p1_paths=batch["p1_path"],
                p2_paths=batch["p2_path"],
                unbound_lig1_paths=batch["unbound_lig1_path"],
                unbound_lig2_paths=batch["unbound_lig2_path"],
                lig1_mask_paths=batch["lig1_mask_path"],
                lig2_mask_paths=batch["lig2_mask_path"],
                step=self.trained_steps,
                max_samples=self._exp_conf.deepternary_max_samples,
                device=self._device,
            )
        else:
            ternary_reward = torch.zeros_like(raw_degradation_reward)
            ternary_info = {
                "ternary_reward": ternary_reward.detach(),

                "deepternary_pred_p2_rmsd": torch.tensor(
                    -1.0,
                    device=self._device,
                    dtype=torch.float32,
                ),
                "deepternary_clash_ratio": torch.tensor(
                    -1.0,
                    device=self._device,
                    dtype=torch.float32,
                ),

                "deepternary_pred_p2_rmsd_raw": torch.full_like(
                    raw_degradation_reward,
                    -1.0,
                ),
                "deepternary_clash_ratio_raw": torch.full_like(
                    raw_degradation_reward,
                    -1.0,
                ),

                "deepternary_scored_mask": torch.zeros_like(raw_degradation_reward),
                "deepternary_attempted_mask": torch.zeros_like(raw_degradation_reward),
                "deepternary_valid_candidate_mask": torch.zeros_like(raw_degradation_reward),
                "deepternary_context_eligible_mask": torch.zeros_like(
                    raw_degradation_reward
                ),

                "deepternary_num_scored": torch.tensor(
                    0.0,
                    device=self._device,
                    dtype=torch.float32,
                ),
                "deepternary_scored_rate": torch.tensor(
                    0.0,
                    device=self._device,
                    dtype=torch.float32,
                ),
                "deepternary_scored_rate_among_valid": torch.tensor(
                    0.0,
                    device=self._device,
                    dtype=torch.float32,
                ),
                "deepternary_scored_rate_among_attempted": torch.tensor(
                    0.0,
                    device=self._device,
                    dtype=torch.float32,
                ),
                "deepternary_attempted_rate_among_valid": torch.tensor(
                    0.0,
                    device=self._device,
                    dtype=torch.float32,
                ),
            }

        ternary_reward = ternary_reward.to(
            device=self._device,
            dtype=torch.float32,
        )

        ternary_reward = torch.nan_to_num(
            ternary_reward,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        deepternary_scored_mask = ternary_info["deepternary_scored_mask"].to(
            device=self._device,
            dtype=torch.float32,
        )

        deepternary_attempted_mask = ternary_info["deepternary_attempted_mask"].to(
            device=self._device,
            dtype=torch.float32,
        )

        deepternary_valid_candidate_mask = ternary_info["deepternary_valid_candidate_mask"].to(
            device=self._device,
            dtype=torch.float32,
        )

        # ============================================================
        # DeepTernary reward scale
        # ============================================================

        ternary_reward = (
                ternary_reward
                * float(self._exp_conf.ternary_reward_scale)
        )

        ternary_reward = torch.clamp(
            ternary_reward,
            min=0.0,
            max=1.0,
        )

        # ============================================================
        # Reward weights
        # ============================================================
        degradation_weight = float(
            self._exp_conf.reward_degradation_weight
        )

        high_degradation_weight = float(
            self._exp_conf.reward_high_degradation_weight
        )

        ternary_weight = float(
            self._exp_conf.reward_ternary_weight
        )

        strict_topology_weight = float(
            self._exp_conf.reward_strict_topology_weight
        )

        # Disable reward weights when their corresponding predictors are unavailable.
        if self.protac_stan_reward is None:
            degradation_weight = 0.0
            high_degradation_weight = 0.0

        if self.deepternary_reward is None:
            ternary_weight = 0.0

        # ============================================================
        # Strict-gated degradation rewards
        # ============================================================
        gated_degradation_reward = (
                raw_degradation_reward
                * strict_protac_reward
        )

        high_deg_threshold = float(
            self._exp_conf.degradation_threshold
        )

        high_degradation_reward = (
                (
                        raw_degradation_reward
                        >= high_deg_threshold
                ).float()
                * strict_protac_reward
        )

        # ============================================================
        # DeepTernary availability
        # ============================================================
        strict_bool = (
                strict_protac_reward > 0.5
        )

        dt_scored_bool = (
                deepternary_scored_mask > 0.5
        )

        dt_candidate_bool = (
                deepternary_valid_candidate_mask > 0.5
        )

        dt_attempted_bool = (
                deepternary_attempted_mask > 0.5
        )

        # Activate ternary rewards only for strict candidates with
        # eligible contexts and successful DeepTernary scoring.
        ternary_active_mask = (
                strict_bool
                & dt_candidate_bool
                & dt_scored_bool
        ).float()

        # Exclude unavailable ternary scores from the reward numerator.
        # Denominator adjustment is handled separately according to
        # DeepTernary context eligibility.
        ternary_reward = (
                ternary_reward
                * ternary_active_mask
        )

        # ============================================================
        # Reward numerator
        # ============================================================
        reward_numerator = (
                strict_topology_weight
                * strict_protac_reward

                + degradation_weight
                * gated_degradation_reward

                + high_degradation_weight
                * high_degradation_reward

                + ternary_weight
                * ternary_reward
        )

        # ============================================================
        # Stage-dependent partial reward normalization
        # ============================================================
        renormalize_missing_ternary = bool(self._exp_conf.renormalize_missing_ternary)

        if (
                renormalize_missing_ternary
                and ternary_weight > 0.0
        ):

            # Sum reward weights excluding the DeepTernary component.
            base_weight_sum = (
                    strict_topology_weight
                    + degradation_weight
                    + high_degradation_weight
            )

            configured_weight_sum = (
                    base_weight_sum
                    + ternary_weight
            )

            # Normalize using DeepTernary context eligibility.
            # Eligible contexts retain the ternary weight even if scoring fails.
            # Ineligible contexts use only the non-ternary base weights.
            dt_context_eligible_bool = (
                    ternary_info[
                        "deepternary_context_eligible_mask"
                    ].to(self._device) > 0.5
            )

            active_weight_sum = (
                    torch.full_like(
                        strict_protac_reward,
                        base_weight_sum,
                    )
                    + ternary_weight
                    * dt_context_eligible_bool.float()
            )

            reward = (
                    reward_numerator
                    / active_weight_sum.clamp(min=1e-8)
                    * configured_weight_sum
            )

        else:
            # Use the weighted reward sum when ternary renormalization is inactive.
            reward = reward_numerator

        reward = torch.nan_to_num(
            reward,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ============================================================
        # DeepTernary coverage / failure logging
        # ============================================================
        strict_count = (
            strict_protac_reward
            .sum()
            .clamp(min=1.0)
        )

        # Strict candidates eligible for DeepTernary evaluation.
        eligible_strict_mask = (
                strict_bool
                & dt_candidate_bool
        )

        # Strict candidates without an eligible DeepTernary context.
        ineligible_strict_mask = (
                strict_bool
                & (~dt_candidate_bool)
        )

        # Context-eligible strict candidates attempted but not scored.
        technical_failed_mask = (
                eligible_strict_mask
                & dt_attempted_bool
                & (~dt_scored_bool)
        )

        # Context-eligible strict candidates without a DeepTernary attempt.
        skipped_eligible_mask = (
                eligible_strict_mask
                & (~dt_attempted_bool)
        )
        # Unscored strict candidates include missing contexts,
        # failed DeepTernary inference, and skipped eligible candidates.
        unscored_strict_mask = (
                strict_bool
                & (~dt_scored_bool)
        )

        eligible_strict_count = (
            eligible_strict_mask
            .float()
            .sum()
            .clamp(min=1.0)
        )

        # ============================================================
        # Rates
        # ============================================================

        deepternary_scored_rate_among_strict = (
                (
                        dt_scored_bool.float()
                        * strict_protac_reward
                ).sum()
                / strict_count
        )

        deepternary_attempted_rate_among_strict = (
                (
                        dt_attempted_bool.float()
                        * strict_protac_reward
                ).sum()
                / strict_count
        )

        deepternary_unscored_strict_rate = (
                unscored_strict_mask
                .float()
                .sum()
                / strict_count
        )

        deepternary_technical_failed_strict_rate = (
                technical_failed_mask
                .float()
                .sum()
                / strict_count
        )

        deepternary_ineligible_strict_rate = (
                ineligible_strict_mask
                .float()
                .sum()
                / strict_count
        )

        deepternary_skipped_eligible_strict_rate = (
                skipped_eligible_mask
                .float()
                .sum()
                / eligible_strict_count
        )

        deepternary_scored_rate_among_eligible_strict = (
                (
                        dt_scored_bool
                        & eligible_strict_mask
                )
                .float()
                .sum()
                / eligible_strict_count
        )

        raw_degradation_strict_mean = (
                                              raw_degradation_reward
                                              * strict_protac_reward
                                      ).sum() / strict_count

        high_degradation_global = (
            high_degradation_reward.mean()
        )

        high_degradation_among_strict = (
                high_degradation_reward.sum()
                / strict_count
        )
        reward_info = {
            "reward": reward.detach(),
            "strict_protac_reward": strict_protac_reward.detach(),
            # Raw predictor scores; non-strict candidates are not submitted for scoring.
            "raw_degradation_reward": raw_degradation_reward.detach(),
            "gated_degradation_reward": gated_degradation_reward.detach(),
            "high_degradation_reward": high_degradation_reward.detach(),
            # Mean predicted degradation among strict-valid molecules.
            "raw_degradation_strict_mean": raw_degradation_strict_mean.detach(),
            "high_degradation_global": high_degradation_global.detach(),
            "high_degradation_among_strict": high_degradation_among_strict.detach(),
            "ternary_reward": ternary_reward.detach(),
            "deepternary_pred_p2_rmsd": ternary_info["deepternary_pred_p2_rmsd"].detach(),
            "deepternary_clash_ratio": ternary_info["deepternary_clash_ratio"].detach(),

            "deepternary_pred_p2_rmsd_raw":
                ternary_info[
                    "deepternary_pred_p2_rmsd_raw"
                ].detach(),

            "deepternary_clash_ratio_raw":
                ternary_info[
                    "deepternary_clash_ratio_raw"
                ].detach(),

            "deepternary_context_eligible_mask":
                ternary_info[
                    "deepternary_context_eligible_mask"
                ].detach(),

            "deepternary_num_scored": ternary_info["deepternary_num_scored"].detach(),
            "deepternary_scored_rate": ternary_info["deepternary_scored_rate"].detach(),
            "deepternary_scored_mask": deepternary_scored_mask.detach(),
            "deepternary_attempted_mask": deepternary_attempted_mask.detach(),
            "deepternary_valid_candidate_mask": deepternary_valid_candidate_mask.detach(),
            "deepternary_scored_rate_among_strict": deepternary_scored_rate_among_strict.detach(),
            "deepternary_attempted_rate_among_strict": deepternary_attempted_rate_among_strict.detach(),
            "deepternary_unscored_strict_rate": deepternary_unscored_strict_rate.detach(),
            "deepternary_technical_failed_strict_rate": deepternary_technical_failed_strict_rate.detach(),
            "deepternary_ineligible_strict_rate":
                deepternary_ineligible_strict_rate.detach(),
            "deepternary_skipped_eligible_strict_rate":
                deepternary_skipped_eligible_strict_rate.detach(),
            "deepternary_scored_rate_among_eligible_strict":
                deepternary_scored_rate_among_eligible_strict.detach(),
            "ternary_active_mask":
                ternary_active_mask.detach(),
            "deepternary_scored_rate_among_valid": ternary_info["deepternary_scored_rate_among_valid"].detach(),
            "deepternary_scored_rate_among_attempted": ternary_info["deepternary_scored_rate_among_attempted"].detach(),
            "deepternary_attempted_rate_among_valid": ternary_info["deepternary_attempted_rate_among_valid"].detach(),

            **{k: v.detach() for k, v in strict_info_tensors.items()},
        }

        return reward, reward_info

    def evaluate_traj_log_prob(self, rollout_out):
        """
        Recompute trajectory log probabilities under the current policy.

        Evaluate stored velocity actions using the Flow Matching policy
        and return both new and rollout-time log probabilities.
        """

        traj = rollout_out["traj"]
        batch = rollout_out["batch"]

        node_mask = batch["atom_mask"]
        linker_mask = batch["linker_mask"]

        anchor1_mask = batch["anchor1_mask"]
        anchor2_mask = batch["anchor2_mask"]
        fragment1_mask = batch["fragment1_mask"]
        fragment2_mask = batch["fragment2_mask"]

        B = batch["positions"].size(0)
        N = batch["positions"].size(1)

        edge_mask = batch["edge_mask"].view(B, N, N)

        action_std = float(self._exp_conf.rl_action_std)

        log_probs_new = []
        log_probs_old = []

        for step in traj:
            z_t = step["z_t"]
            action = step["action"]
            t = step["t"]

            v_mean, _ = self.flow_policy_params(
                model=self.model,
                z_t=z_t,
                t=t,
                node_mask=node_mask,
                linker_mask=linker_mask,
                anchor1_mask=anchor1_mask,
                anchor2_mask=anchor2_mask,
                fragment1_mask=fragment1_mask,
                fragment2_mask=fragment2_mask,
                edge_mask=edge_mask,
            )

            log_prob_new = masked_gaussian_log_prob(
                z=action,
                mu=v_mean,
                sigma=torch.full_like(action, action_std),
                mask=linker_mask,
            )

            log_probs_new.append(log_prob_new)
            log_probs_old.append(step["log_prob_old"])

        log_prob_new_total = torch.stack(log_probs_new, dim=0).sum(dim=0)
        log_prob_old_total = torch.stack(log_probs_old, dim=0).sum(dim=0)

        return log_prob_new_total, log_prob_old_total

    @torch.no_grad()
    def rollout_sample(
            self,
            old_model,
            x,
            h,
            node_mask,
            fragment_mask,
            linker_mask,
            anchor1_mask,
            anchor2_mask,
            fragment1_mask,
            fragment2_mask,
            edge_mask,
            keep_frames=None,
    ):
        """
        Flow Matching PPO rollout.

        state:
            z_t

        action:
            velocity a_t

        policy:
            a_t ~ Normal(v_theta(z_t, t), action_std)

        update:
            z_{t+dt} = z_t + dt * a_t
        """

        old_model.eval()

        B = x.size(0)
        N = x.size(1)

        # Normalize coordinates and atom features before rollout.
        x, h = Experiment.normalize(x, h)
        xh = torch.cat([x, h], dim=2)

        # Initialize linker states with Gaussian noise.
        z_noise = Experiment.sample_combined_position_feature_noise(
            n_samples=B,
            n_nodes=N,
            mask=linker_mask,
        )

        z = (
                xh * fragment_mask
                + z_noise * linker_mask
        )

        if keep_frames is None:
            keep_frames = self.T

        assert keep_frames <= self.T

        chain = torch.zeros(
            (keep_frames,) + z.size(),
            device=z.device,
        )

        t_end = float(self._exp_conf.flow_t_end)
        ts = torch.linspace(0.0, t_end, self.T + 1, device=z.device)

        action_std = float(self._exp_conf.rl_action_std)

        traj = []

        num_keep_traj = int(self._exp_conf.ppo_t_indices)
        num_keep_traj = max(1, min(num_keep_traj, self.T))

        traj_keep_indices = torch.linspace(
            0,
            self.T - 1,
            steps=num_keep_traj,
            device=z.device,
        ).long().detach().cpu().tolist()

        traj_keep_indices = set(int(v) for v in traj_keep_indices)

        # ------------------------------------------------
        # Flow ODE rollout
        # ------------------------------------------------
        for k in range(self.T):
            t_now = ts[k]
            t_next = ts[k + 1]
            dt = t_next - t_now

            t_batch = torch.full(
                (B, 1),
                float(t_now.item()),
                device=z.device,
            )

            z_t = z.detach()

            v_mean, _ = self.flow_policy_params(
                model=old_model,
                z_t=z_t,
                t=t_batch,
                node_mask=node_mask,
                linker_mask=linker_mask,
                anchor1_mask=anchor1_mask,
                anchor2_mask=anchor2_mask,
                fragment1_mask=fragment1_mask,
                fragment2_mask=fragment2_mask,
                edge_mask=edge_mask,
            )

            noise = Experiment.sample_combined_position_feature_noise(
                n_samples=B,
                n_nodes=N,
                mask=linker_mask,
            )

            action = v_mean + action_std * noise
            action = action * linker_mask

            z_next = z_t + dt * action

            # Keep fragment states fixed during stochastic rollout.
            z = (
                    xh * fragment_mask
                    + z_next * linker_mask
            )

            log_prob_old = masked_gaussian_log_prob(
                z=action,
                mu=v_mean,
                sigma=torch.full_like(action, action_std),
                mask=linker_mask,
            )

            if k in traj_keep_indices:
                traj.append({
                    "z_t": z_t.detach(),
                    "action": action.detach(),
                    "t": t_batch.detach(),
                    "dt": dt.detach(),
                    "log_prob_old": log_prob_old.detach(),
                })

            # Store intermediate rollout frames for diagnostics.
            write_index = keep_frames - 1 - int((k * keep_frames) / self.T)
            write_index = max(1, min(write_index, keep_frames - 1))
            chain[write_index] = Experiment.unnormalize_z(z)

        # Predict the final clean state and bond logits at t_end.
        t_final = torch.full(
            (B, 1),
            float(t_end),
            device=z.device,
        )

        v_final, xh_pred_final, bond_logits = self.flow_policy_params(
            model=old_model,
            z_t=z,
            t=t_final,
            node_mask=node_mask,
            linker_mask=linker_mask,
            anchor1_mask=anchor1_mask,
            anchor2_mask=anchor2_mask,
            fragment1_mask=fragment1_mask,
            fragment2_mask=fragment2_mask,
            edge_mask=edge_mask,
            return_bond=True,
        )

        final_xh_norm = (
                xh * fragment_mask
                + xh_pred_final * linker_mask
        )

        x_final = final_xh_norm[:, :, :3]
        h_final = final_xh_norm[:, :, 3:]

        x_final, h_final = Experiment.unnormalize(x_final, h_final)

        # Discretize linker atom types using the permitted atom vocabulary.
        allowed_linker_atom_ids = get_atom_type_ids_by_symbols(
            ("C", "N", "O", "S", "F")
        )

        type_allowed_mask = torch.zeros(
            const.NUMBER_OF_ATOM_TYPES,
            device=h_final.device,
            dtype=torch.bool,
        )
        type_allowed_mask[allowed_linker_atom_ids] = True

        if linker_mask.dim() == 3:
            linker_mask_2d = linker_mask.squeeze(-1).bool()
        else:
            linker_mask_2d = linker_mask.bool()

        h_final_masked = h_final.clone()
        h_final_masked[..., ~type_allowed_mask] = -1e9

        atom_type_default = torch.argmax(
            h_final,
            dim=2,
        )

        atom_type_linker_restricted = torch.argmax(
            h_final_masked,
            dim=2,
        )

        atom_type_final = torch.where(
            linker_mask_2d,
            atom_type_linker_restricted,
            atom_type_default,
        )

        h_final = F.one_hot(
            atom_type_final,
            const.NUMBER_OF_ATOM_TYPES,
        ).float()

        h_final = h_final * node_mask

        final_xh = torch.cat([x_final, h_final], dim=2)

        pred_bond_probs, pred_bond_types = self.bond_logits_to_pred_bonds(
            bond_logits=bond_logits,
            anchor1_mask=anchor1_mask,
            anchor2_mask=anchor2_mask,
            linker_mask=linker_mask,
        )

        chain[0] = final_xh

        return {
            "final_z": z.detach(),
            "final_xh": final_xh.detach(),
            "pred_bond_probs": pred_bond_probs.detach(),
            "pred_bond_types": pred_bond_types.detach(),
            "traj": traj,
            "chain": chain.detach(),
        }

    def get_anchor_com(self, x, node_mask, anchor_mask):
        """
        Compute the masked anchor center in the original coordinate frame.

        Args:
            x: Atomic coordinates [B, N, 3].
            node_mask: Valid atom mask [B, N, 1].
            anchor_mask: Anchor atom mask [B, N, 1].

        Returns:
            Anchor center coordinates [B, 1, 3].
        """
        center_mask = node_mask.float() * anchor_mask.float()
        N = center_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (x * center_mask).sum(dim=1, keepdim=True) / N
        return mean
    @torch.no_grad()
    def rollout_sample_from_batch(self, old_model, batch, keep_frames=None):
        batch = {
            k: v.to(self._device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        x_raw = batch["positions"]
        h = batch["one_hot"]

        node_mask = batch["atom_mask"]
        linker_mask = batch["linker_mask"]
        anchor_mask = batch["anchors"]
        anchor1_mask = batch["anchor1_mask"]
        anchor2_mask = batch["anchor2_mask"]
        fragment1_mask = batch["fragment1_mask"]
        fragment2_mask = batch["fragment2_mask"]

        fragment_mask = (
                fragment1_mask.bool() | fragment2_mask.bool()
        ).to(linker_mask.dtype)

        B, N, _ = x_raw.shape
        edge_mask = batch["edge_mask"].view(B, N, N)

        # Preserve the original anchor center for coordinate restoration.
        anchor_com = self.get_anchor_com(
            x=x_raw,
            node_mask=node_mask,
            anchor_mask=anchor_mask,
        )

        # Center coordinates using the anchor reference before rollout.
        x = utils.remove_partial_mean_with_mask(
            x_raw,
            node_mask,
            anchor_mask,
        )

        rollout_out = self.rollout_sample(
            old_model=old_model,
            x=x,
            h=h,
            node_mask=node_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            anchor1_mask=anchor1_mask,
            anchor2_mask=anchor2_mask,
            fragment1_mask=fragment1_mask,
            fragment2_mask=fragment2_mask,
            edge_mask=edge_mask,
            keep_frames=keep_frames,
        )

        rollout_out["batch"] = batch
        rollout_out["anchor_com"] = anchor_com.detach()

        return rollout_out

    @torch.no_grad()
    def sample_pretrain_style_from_batch(
            self,
            model,
            batch,
            keep_frames=None,
            linker_size_offset=0,
    ):
        """
        Generate molecules using the deterministic pretrained Flow ODE sampler.

        A zero linker_size_offset preserves the original linker size.
        Nonzero offsets adjust the linker size and rebuild the
        generation template before sampling.
        """

        batch = {
            k: v.to(self._device, non_blocking=True)
            if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        # Standard sampling path without linker-size modification.
        if int(linker_size_offset) == 0:
            x_raw = batch["positions"]
            h = batch["one_hot"]

            node_mask = batch["atom_mask"]
            linker_mask = batch["linker_mask"]

            anchor_mask = batch["anchors"]

            anchor1_mask = batch["anchor1_mask"]
            anchor2_mask = batch["anchor2_mask"]
            fragment1_mask = batch["fragment1_mask"]
            fragment2_mask = batch["fragment2_mask"]

            fragment_mask = (
                    fragment1_mask.bool()
                    | fragment2_mask.bool()
            ).to(linker_mask.dtype)

            B, N, _ = x_raw.shape
            edge_mask = batch["edge_mask"].view(B, N, N)

            anchor_com = self.get_anchor_com(
                x=x_raw,
                node_mask=node_mask,
                anchor_mask=anchor_mask,
            )

            x = utils.remove_partial_mean_with_mask(
                x_raw,
                node_mask,
                anchor_mask,
            )

            sample_out = self.sample_pretrain_style(
                model=model,
                x=x,
                h=h,
                node_mask=node_mask,
                fragment_mask=fragment_mask,
                linker_mask=linker_mask,
                anchor1_mask=anchor1_mask,
                anchor2_mask=anchor2_mask,
                fragment1_mask=fragment1_mask,
                fragment2_mask=fragment2_mask,
                edge_mask=edge_mask,
                keep_frames=keep_frames,
            )

            sample_out["batch"] = batch
            sample_out["anchor_com"] = anchor_com.detach()

            return sample_out

        # Linker-size sensitivity path for nonzero offsets.
        true_linker_sizes = (
            batch["linker_mask"]
            .sum(dim=1)
            .view(-1)
            .long()
        )

        conditioned_linker_sizes = (
                true_linker_sizes + int(linker_size_offset)
        )


        if torch.any(conditioned_linker_sizes <= 0):
            bad_sizes = conditioned_linker_sizes[
                conditioned_linker_sizes <= 0
                ].detach().cpu().tolist()

            raise ValueError(
                f"Invalid linker size for offset={linker_size_offset}: "
                f"{bad_sizes}"
            )

        # Rebuild the generation template for the conditioned linker size.
        template_data = datasets.create_templates_for_linker_generation(
            batch,
            conditioned_linker_sizes,
        )

        x_raw = template_data["positions"]
        h = template_data["one_hot"]

        node_mask = template_data["atom_mask"]
        edge_mask = template_data["edge_mask"]

        anchor1_mask = template_data["anchor1_mask"]
        anchor2_mask = template_data["anchor2_mask"]
        fragment1_mask = template_data["fragment1_mask"]
        fragment2_mask = template_data["fragment2_mask"]
        linker_mask = template_data["linker_mask"]

        fragment_mask = (
                fragment1_mask.bool()
                | fragment2_mask.bool()
        ).to(linker_mask.dtype)

        B, N, _ = x_raw.shape
        edge_mask = edge_mask.view(B, N, N)

        # Preserve reference fragment atom properties in the new template.
        # Do not copy reference linker atom properties.
        fragment_mask_2d = (
                batch["fragment1_mask"].bool()
                | batch["fragment2_mask"].bool()
        ).squeeze(-1)

        for key in [
            "formal_charges",
            "atom_is_aromatic",
            "num_explicit_hs",
            "no_implicit",
        ]:
            if key not in batch:
                continue

            src = batch[key]

            if src.dim() == 3 and src.size(-1) == 1:
                src = src.squeeze(-1)

            B_src, _ = src.shape

            dst = torch.zeros(
                (B_src, N),
                dtype=src.dtype,
                device=src.device,
            )

            for i in range(B_src):
                frag_n = int(
                    fragment_mask_2d[i].sum().item()
                )
                dst[i, :frag_n] = src[i, :frag_n]

            template_data[key] = dst

        anchor_mask = (
                anchor1_mask + anchor2_mask
        ).clamp(max=1.0)

        anchor_com = self.get_anchor_com(
            x=x_raw,
            node_mask=node_mask,
            anchor_mask=anchor_mask,
        )

        x = utils.remove_partial_mean_with_mask(
            x_raw,
            node_mask,
            anchor_mask,
        )

        sample_out = self.sample_pretrain_style(
            model=model,
            x=x,
            h=h,
            node_mask=node_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            anchor1_mask=anchor1_mask,
            anchor2_mask=anchor2_mask,
            fragment1_mask=fragment1_mask,
            fragment2_mask=fragment2_mask,
            edge_mask=edge_mask,
            keep_frames=keep_frames,
        )

        # Use the size-conditioned template for downstream reconstruction
        # and evaluation.
        sample_out["batch"] = template_data
        sample_out["anchor_com"] = anchor_com.detach()

        return sample_out

    @torch.no_grad()
    def sample_pretrain_style(
            self,
            model,
            x,
            h,
            node_mask,
            fragment_mask,
            linker_mask,
            anchor1_mask,
            anchor2_mask,
            fragment1_mask,
            fragment2_mask,
            edge_mask,
            keep_frames=None,
    ):
        """
        Generate molecules using deterministic Flow Matching ODE sampling.

        Unlike stochastic PPO rollouts, this sampler integrates the
        predicted velocity directly without additional action noise.
        """

        model.eval()

        B = x.size(0)
        N = x.size(1)

        # 1. normalize
        x, h = Experiment.normalize(x, h)
        xh = torch.cat([x, h], dim=2)

        # Initialize linker states with masked Gaussian noise.
        z_noise = Experiment.sample_combined_position_feature_noise(
            n_samples=B,
            n_nodes=N,
            mask=linker_mask,
        )

        z = (
                xh * fragment_mask
                + z_noise * linker_mask
        )

        if keep_frames is None:
            keep_frames = self.T

        assert keep_frames <= self.T

        chain = torch.zeros(
            (keep_frames,) + z.size(),
            device=z.device,
        )

        t_end = 0.95
        ts = torch.linspace(0.0, t_end, self.T + 1, device=z.device)

        # Integrate the deterministic Flow ODE without policy action noise.
        for k in range(self.T):
            t_now = ts[k]
            t_next = ts[k + 1]
            dt = t_next - t_now

            t_batch = torch.full(
                (B, 1),
                float(t_now.item()),
                device=z.device,
            )

            x_t = z[:, :, :3]
            h_t = z[:, :, 3:]
            h_pred, x_pred = model(
                h_t,
                x_t,
                anchor1_mask,
                anchor2_mask,
                fragment1_mask,
                fragment2_mask,
                linker_mask,
                node_mask,
                t_batch,
                edge_mask,
            )

            xh_pred = torch.cat([x_pred, h_pred], dim=2)

            # Derive the velocity from the predicted clean state.
            v = (xh_pred - z) / (1.0 - t_now + 1e-6)
            z_next = z + dt * v
            # Preserve fixed fragment states at every integration step.
            z = (
                    xh * fragment_mask
                    + z_next * linker_mask
            )

            write_index = keep_frames - 1 - int((k * keep_frames) / self.T)
            write_index = max(1, min(write_index, keep_frames - 1))
            chain[write_index] = Experiment.unnormalize_z(z)

        # Predict the final clean state and bond logits at t_end.
        x_t = z[:, :, :3]
        h_t = z[:, :, 3:]

        t_final = torch.full(
            (B, 1),
            float(t_end),
            device=z.device,
        )

        h_pred, x_pred, bond_logits = model(
            h_t,
            x_t,
            anchor1_mask,
            anchor2_mask,
            fragment1_mask,
            fragment2_mask,
            linker_mask,
            node_mask,
            t_final,
            edge_mask,
            return_bond=True,
        )

        xh_pred_final = torch.cat([x_pred, h_pred], dim=2)

        final_xh_norm = (
                xh * fragment_mask
                + xh_pred_final * linker_mask
        )

        x_final_norm = final_xh_norm[:, :, :3]
        h_final_norm = final_xh_norm[:, :, 3:]

        x_final, h_final_cont = Experiment.unnormalize(
            x_final_norm,
            h_final_norm,
        )

        # Restrict generated linker atom types to C, N, O, S, and F.
        allowed_linker_atom_ids = get_atom_type_ids_by_symbols(
            ("C", "N", "O", "S", "F")
        )

        type_allowed_mask = torch.zeros(
            const.NUMBER_OF_ATOM_TYPES,
            device=h_final_cont.device,
            dtype=torch.bool,
        )
        type_allowed_mask[allowed_linker_atom_ids] = True

        if linker_mask.dim() == 3:
            linker_mask_2d = linker_mask.squeeze(-1).bool()
        else:
            linker_mask_2d = linker_mask.bool()

        h_final_cont_masked = h_final_cont.clone()
        h_final_cont_masked[..., ~type_allowed_mask] = -1e9

        atom_type_default = torch.argmax(
            h_final_cont,
            dim=2,
        )

        atom_type_linker_restricted = torch.argmax(
            h_final_cont_masked,
            dim=2,
        )

        atom_type_final = torch.where(
            linker_mask_2d,
            atom_type_linker_restricted,
            atom_type_default,
        )

        h_final = F.one_hot(
            atom_type_final,
            const.NUMBER_OF_ATOM_TYPES,
        ).float()

        h_final = h_final * node_mask

        final_xh = torch.cat([x_final, h_final], dim=2)

        pred_bond_probs, pred_bond_types = self.bond_logits_to_pred_bonds(
            bond_logits=bond_logits,
            anchor1_mask=anchor1_mask,
            anchor2_mask=anchor2_mask,
            linker_mask=linker_mask,
        )

        chain[0] = final_xh

        return {
            "final_xh": final_xh.detach(),
            "pred_bond_probs": pred_bond_probs.detach(),
            "pred_bond_types": pred_bond_types.detach(),
            "chain": chain.detach(),
        }

    def flow_policy_params(
            self,
            model,
            z_t,
            t,
            node_mask,
            linker_mask,
            anchor1_mask,
            anchor2_mask,
            fragment1_mask,
            fragment2_mask,
            edge_mask,
            return_value=False,
            return_bond=False,
    ):
        """
        Compute the Flow Matching policy velocity from the predicted clean state.

        v_mean = (xh_pred - z_t) / clamp(1 - t, min=flow_denom_min).
        Only linker states are updated.
        """

        x_t = z_t[:, :, :3]
        h_t = z_t[:, :, 3:]

        out = model(
            h_t,
            x_t,
            anchor1_mask,
            anchor2_mask,
            fragment1_mask,
            fragment2_mask,
            linker_mask,
            node_mask,
            t,
            edge_mask,
            return_value=return_value,
            return_bond=return_bond,
        )

        value = None
        bond_logits = None

        if return_value and return_bond:
            h_pred, x_pred, value, bond_logits = out
        elif return_value:
            h_pred, x_pred, value = out
        elif return_bond:
            h_pred, x_pred, bond_logits = out
        else:
            h_pred, x_pred = out

        xh_pred = torch.cat([x_pred, h_pred], dim=2)
        t_expand = t[:, None, :]  # [B, 1, 1]
        denom = torch.clamp(
            1.0 - t_expand,
            min=float(self._exp_conf.flow_denom_min),
        )

        v_mean = (xh_pred - z_t) / denom

        v_mean = v_mean * linker_mask

        if return_value and return_bond:
            return v_mean, xh_pred, value, bond_logits

        if return_value:
            return v_mean, xh_pred, value

        if return_bond:
            return v_mean, xh_pred, bond_logits

        return v_mean, xh_pred

    def bond_logits_to_pred_bonds(
            self,
            bond_logits,
            anchor1_mask,
            anchor2_mask,
            linker_mask,
    ):
        """
        bond_logits: [B, N, N, 5]
            0 = no bond
            1 = single
            2 = double
            3 = triple
            4 = aromatic
        """

        bond_type_probs = torch.softmax(
            bond_logits,
            dim=-1,
        )

        anchor_mask = (
                anchor1_mask + anchor2_mask
        ).clamp(max=1.0)

        if anchor_mask.dim() == 3:
            anchor_mask_2d = anchor_mask.squeeze(-1).bool()
        else:
            anchor_mask_2d = anchor_mask.bool()

        if linker_mask.dim() == 3:
            linker_mask_2d = linker_mask.squeeze(-1).bool()
        else:
            linker_mask_2d = linker_mask.bool()

        anchor_linker_pair_mask = (
                (anchor_mask_2d[:, :, None] & linker_mask_2d[:, None, :])
                |
                (linker_mask_2d[:, :, None] & anchor_mask_2d[:, None, :])
        )

        # General pairs: P(bond) = 1 - P(no bond).
        pred_bond_probs_default = 1.0 - bond_type_probs[..., 0]

        # Exclude aromatic bonds from anchor-linker existence probabilities.
        pred_bond_probs_anchor_linker = (
                bond_type_probs[..., 1]
                + bond_type_probs[..., 2]
                + bond_type_probs[..., 3]
        )

        pred_bond_probs = torch.where(
            anchor_linker_pair_mask,
            pred_bond_probs_anchor_linker,
            pred_bond_probs_default,
        )

        # General pairs: select the highest-scoring nonzero bond type.
        bond_type_scores = bond_type_probs[..., 1:5].clone()

        pred_bond_types_default = torch.argmax(
            bond_type_scores,
            dim=-1,
        ) + 1

        # Anchor-linker pairs: select the highest-scoring non-aromatic bond type.
        bond_type_scores_no_aromatic = bond_type_scores[..., :3].clone()

        pred_bond_types_anchor_linker = torch.argmax(
            bond_type_scores_no_aromatic,
            dim=-1,
        ) + 1

        pred_bond_types = torch.where(
            anchor_linker_pair_mask,
            pred_bond_types_anchor_linker,
            pred_bond_types_default,
        )

        return pred_bond_probs, pred_bond_types


    def sample_normal(self, mu, sigma, node_mask):
        """Samples from a Normal distribution."""
        eps = Experiment.sample_combined_position_feature_noise(mu.size(0), mu.size(1), node_mask)
        return mu + sigma * eps


@hydra.main(version_base=None, config_path="../configs", config_name="rl_flow_stage1")
def run(conf: DictConfig) -> None:

    os.environ["WANDB_START_METHOD"] = "thread"
    OmegaConf.set_struct(conf, False)

    seed = int(conf.experiment.seed)
    seed_everything(seed)

    logging.getLogger(__name__).info(f"Using random seed: {seed}")

    rl_exp = RLExperiment(conf=conf)
    rl_exp.start_rl_training()


if __name__ == "__main__":
    run()