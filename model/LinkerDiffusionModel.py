import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rbf_distance(dist, D_min=0.0, D_max=4.0, D_count=16):
    """
    dist: [B, N, N]
    坐标已经除以 5，所以 D_max=4.0 约等于真实 20 Å。
    """
    device = dist.device
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view(1, 1, 1, D_count)

    D_sigma = (D_max - D_min) / D_count
    dist = dist.unsqueeze(-1)

    return torch.exp(-((dist - D_mu) / D_sigma) ** 2)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.mlp[0].in_features // 2
        device = t.device

        freqs = torch.exp(
            torch.arange(half, device=device) * -(math.log(10000) / (half - 1))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return self.mlp(emb)


class PairStateInitializer(nn.Module):
    """
    初始化显式 pair state z_ij。

    z_ij 会在后续 TopoPairBlock 中持续更新，作为 bond/topology 的内部表示，
    不再只依赖最后一个 bond head 从 h_i/h_j 临时判断是否成键。
    """

    def __init__(
            self,
            c_s,
            c_z=64,
            rbf_dim=16,
            local_rbf_dim=16,
            pair_type_dim=5,
    ):
        super().__init__()

        self.rbf_dim = rbf_dim
        self.local_rbf_dim = local_rbf_dim
        self.pair_type_dim = pair_type_dim

        in_dim = (
            c_s * 2
            + 1
            + rbf_dim
            + local_rbf_dim
            + pair_type_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, c_z),
            nn.SiLU(),
            nn.Linear(c_z, c_z),
            nn.SiLU(),
            nn.Linear(c_z, c_z),
        )

        self.norm = nn.LayerNorm(c_z)

    def forward(self, h, x, pair_type, node_mask):
        if node_mask.dim() == 3:
            node_mask = node_mask.squeeze(-1)
        node_mask = node_mask.float()

        B, N, C = h.shape

        hi = h[:, :, None, :].expand(B, N, N, C)
        hj = h[:, None, :, :].expand(B, N, N, C)

        vec = x[:, :, None, :] - x[:, None, :, :]
        d = torch.norm(vec, dim=-1, keepdim=True)
        d_scalar = d.squeeze(-1)

        d_rbf = rbf_distance(
            d_scalar,
            D_min=0.0,
            D_max=4.0,
            D_count=self.rbf_dim,
        )

        d_local_rbf = rbf_distance(
            d_scalar,
            D_min=0.0,
            D_max=0.8,
            D_count=self.local_rbf_dim,
        )

        pair_input = torch.cat(
            [
                hi,
                hj,
                d,
                d_rbf,
                d_local_rbf,
                pair_type.to(dtype=h.dtype),
            ],
            dim=-1,
        )

        z = self.norm(self.mlp(pair_input))

        mask_2d = node_mask[:, :, None] * node_mask[:, None, :]
        eye = torch.eye(N, device=h.device, dtype=mask_2d.dtype)[None, :, :]
        mask_2d = mask_2d * (1.0 - eye)

        z = z * mask_2d[:, :, :, None]
        return z


class TopoPairBlock(nn.Module):
    """
    Topology-aware pair-state block.

    三条流同时更新：
      1. node stream: h_i
      2. pair stream: z_ij
      3. coordinate stream: x_i，只更新 linker 坐标

    关键区别：triangle path update 让 z_ij 看到 i-k-j 路径信息，
    对 linker 内部连通性比普通 node-only message passing 更直接。
    """

    def __init__(
            self,
            c_s,
            c_z=64,
            n_heads=8,
            rbf_dim=16,
            local_rbf_dim=16,
            pair_type_dim=5,
            coord_scale=0.15,
            triangle_scale=0.25,
            use_triangle=True,
            use_pair_bias=True,
    ):
        super().__init__()

        if c_s % n_heads != 0:
            raise ValueError(f"c_s={c_s} must be divisible by n_heads={n_heads}")

        self.c_s = c_s
        self.c_z = c_z
        self.n_heads = n_heads
        self.rbf_dim = rbf_dim
        self.local_rbf_dim = local_rbf_dim
        self.pair_type_dim = pair_type_dim
        self.coord_scale = coord_scale
        self.triangle_scale = triangle_scale

        self.use_triangle = use_triangle
        self.use_pair_bias = use_pair_bias

        pair_update_in = (
            c_z
            + c_s * 2
            + 1
            + rbf_dim
            + local_rbf_dim
            + pair_type_dim
        )

        self.pair_update = nn.Sequential(
            nn.Linear(pair_update_in, c_z),
            nn.SiLU(),
            nn.Linear(c_z, c_z),
            nn.SiLU(),
            nn.Linear(c_z, c_z),
        )

        self.tri_left = nn.Linear(c_z, c_z)
        self.tri_right = nn.Linear(c_z, c_z)
        self.tri_out = nn.Sequential(
            nn.Linear(c_z, c_z),
            nn.SiLU(),
            nn.Linear(c_z, c_z),
        )

        self.pair_norm = nn.LayerNorm(c_z)

        self.node_q = nn.Linear(c_s, c_s)
        self.node_k = nn.Linear(c_s, c_s)
        self.node_v = nn.Linear(c_s, c_s)
        self.pair_bias = nn.Linear(c_z, n_heads)

        self.node_out = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.SiLU(),
            nn.Linear(c_s, c_s),
        )

        self.node_norm = nn.LayerNorm(c_s)

        self.coord_gate = nn.Sequential(
            nn.Linear(c_z, c_z),
            nn.SiLU(),
            nn.Linear(c_z, 1),
            nn.Sigmoid(),
        )

        self.coord_weight = nn.Sequential(
            nn.Linear(c_z, c_z),
            nn.SiLU(),
            nn.Linear(c_z, 1),
        )

        # 坐标分支零初始化，避免训练早期坐标被 pair state 大幅拉坏。
        nn.init.zeros_(self.coord_weight[-1].weight)
        nn.init.zeros_(self.coord_weight[-1].bias)

    def build_distance_features(self, h, x):
        B, N, C = h.shape

        hi = h[:, :, None, :].expand(B, N, N, C)
        hj = h[:, None, :, :].expand(B, N, N, C)

        vec = x[:, :, None, :] - x[:, None, :, :]
        d = torch.norm(vec, dim=-1, keepdim=True)
        d_scalar = d.squeeze(-1)

        d_rbf = rbf_distance(
            d_scalar,
            D_min=0.0,
            D_max=4.0,
            D_count=self.rbf_dim,
        )

        d_local_rbf = rbf_distance(
            d_scalar,
            D_min=0.0,
            D_max=0.8,
            D_count=self.local_rbf_dim,
        )

        direction = vec / (d + 1e-8)

        return hi, hj, d, d_rbf, d_local_rbf, direction

    def triangle_path_update(self, z, mask_2d):
        zl = self.tri_left(z) * mask_2d[:, :, :, None]
        zr = self.tri_right(z) * mask_2d[:, :, :, None]

        # z_ij 接收所有 i-k-j path 的信息。
        tri = torch.einsum("bikc,bkjc->bijc", zl, zr)

        denom = torch.einsum("bik,bkj->bij", mask_2d, mask_2d)
        tri = tri / denom.clamp(min=1.0)[:, :, :, None]

        tri = self.tri_out(tri)
        tri = tri * mask_2d[:, :, :, None]
        return tri

    def forward(
            self,
            h,
            x,
            z,
            node_mask,
            linker_mask,
            pair_type,
            edge_mask=None,
    ):
        B, N, C = h.shape

        if node_mask.dim() == 3:
            node_mask = node_mask.squeeze(-1)
        if linker_mask.dim() == 3:
            linker_mask = linker_mask.squeeze(-1)

        node_mask = node_mask.float()
        linker_mask = linker_mask.float()
        pair_type = pair_type.to(dtype=h.dtype)

        mask_2d = node_mask[:, :, None] * node_mask[:, None, :]

        if edge_mask is not None:
            if edge_mask.dim() == 4:
                edge_mask = edge_mask.squeeze(-1)
            if edge_mask.dim() != 3:
                edge_mask = edge_mask.view(B, N, N)
            mask_2d = mask_2d * edge_mask.float()

        eye = torch.eye(N, device=h.device, dtype=mask_2d.dtype)[None, :, :]
        mask_2d = mask_2d * (1.0 - eye)

        hi, hj, d, d_rbf, d_local_rbf, direction = self.build_distance_features(h, x)

        # 1. pair update
        pair_input = torch.cat(
            [
                z,
                hi,
                hj,
                d,
                d_rbf,
                d_local_rbf,
                pair_type,
            ],
            dim=-1,
        )

        dz = self.pair_update(pair_input)

        if self.use_triangle:
            dz_tri = self.triangle_path_update(z, mask_2d)
            z = self.pair_norm(
                z + dz + self.triangle_scale * dz_tri
            )
        else:
            z = self.pair_norm(z + dz)

        z = z * mask_2d[:, :, :, None]

        # 2. pair-biased node attention
        H = self.n_heads
        D = self.c_s // H

        q = self.node_q(h).view(B, N, H, D)
        k = self.node_k(h).view(B, N, H, D)
        v = self.node_v(h).view(B, N, H, D)

        att = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(D)

        if self.use_pair_bias:
            att = att + self.pair_bias(z).permute(0, 3, 1, 2)

        att = att.masked_fill(mask_2d[:, None] == 0, -1e9)
        att = torch.softmax(att, dim=-1)

        h_msg = torch.einsum("bhij,bjhd->bihd", att, v)
        h_msg = h_msg.reshape(B, N, self.c_s)

        h = self.node_norm(h + self.node_out(h_msg))
        h = h * node_mask[:, :, None]

        # 3. coordinate update from pair state
        gate = self.coord_gate(z) * mask_2d[:, :, :, None]
        weight = torch.tanh(self.coord_weight(z))

        delta = direction * gate * weight
        denom = gate.sum(dim=2).clamp(min=1.0)
        delta_x = delta.sum(dim=2) / denom

        delta_x = delta_x * linker_mask[:, :, None]
        x = x + self.coord_scale * delta_x
        x = x * node_mask[:, :, None]

        return h, x, z


class FeatureNoiseHead(nn.Module):
    def __init__(self, c_s, num_atom_types):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.SiLU(),
            nn.Linear(c_s, c_s),
            nn.SiLU(),
            nn.Linear(c_s, num_atom_types),
        )
        self.time_proj = nn.Linear(c_s, c_s)

    def forward(self, h, t_emb=None, linker_mask=None):
        if t_emb is not None:
            h = h + 0.5 * self.time_proj(t_emb)[:, None, :]

        logits = self.mlp(h)

        if linker_mask is not None:
            logits = logits * linker_mask[:, :, None]

        return logits

class LinkerLocalGeometryRefiner(nn.Module):
    """
    Linker joint geometry-feature-pair refiner.

    作用：
        1. 只更新 linker atom 坐标；
        2. 只更新 linker node feature h；
        3. 根据 refined 坐标同步更新 pair state z；
        4. 只允许 linker-linker / anchor-linker pair 参与；
        5. fragment / anchor 坐标不动；
        6. 不改变 reconstruction，不改变 loss。
    """

    def __init__(
            self,
            c_s,
            c_z=64,
            hidden_dim=128,
            rbf_dim=16,
            local_rbf_dim=16,
            coord_scale=0.06,
    ):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.rbf_dim = rbf_dim
        self.local_rbf_dim = local_rbf_dim
        self.coord_scale = coord_scale

        # 输入:
        # z_ij + h_i + h_j + d + global_rbf + local_rbf + pair_type(5)
        in_dim = (
            c_z
            + 2 * c_s
            + 1
            + rbf_dim
            + local_rbf_dim
            + 5
        )

        # ------------------------------------------------
        # edge message：同时服务于坐标、h、z 更新
        # ------------------------------------------------
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 坐标更新头：输出 gate 和 signed weight
        self.coord_head = nn.Linear(hidden_dim, 2)

        # linker node feature h 更新头
        self.h_update = nn.Sequential(
            nn.LayerNorm(c_s + hidden_dim),
            nn.Linear(c_s + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, c_s),
        )

        # refined x + edge message 后更新 pair state z
        z_update_in_dim = (
                c_z
                + hidden_dim
                + rbf_dim
                + local_rbf_dim
                + 5
                + 1
        )

        self.z_update = nn.Sequential(
            nn.LayerNorm(z_update_in_dim),
            nn.Linear(z_update_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, c_z),
        )

        self.h_norm = nn.LayerNorm(c_s)
        self.z_norm = nn.LayerNorm(c_z)


        # ------------------------------------------------
        # 零初始化所有更新头，保证初始时几乎等价于原网络
        # ------------------------------------------------
        nn.init.zeros_(self.coord_head.weight)
        nn.init.zeros_(self.coord_head.bias)

        nn.init.zeros_(self.h_update[-1].weight)
        nn.init.zeros_(self.h_update[-1].bias)

        nn.init.zeros_(self.z_update[-1].weight)
        nn.init.zeros_(self.z_update[-1].bias)

    def forward(
            self,
            h,
            x,
            z,
            node_mask,
            linker_mask,
            pair_type_feat,
            pair_type_mask,
            edge_mask=None,
    ):
        """
        h: [B, N, c_s]
        x: [B, N, 3] normalized coordinates
        z: [B, N, N, c_z]
        pair_type: [B, N, N, 5]
            0 = anchor1-linker
            1 = anchor2-linker
            2 = linker-linker
            3 = bad fragment1-linker
            4 = bad fragment2-linker
        """

        B, N, C = h.shape

        if node_mask.dim() == 3:
            node_mask = node_mask.squeeze(-1)
        if linker_mask.dim() == 3:
            linker_mask = linker_mask.squeeze(-1)

        node_mask = node_mask.float()
        linker_mask = linker_mask.float()

        hi = h[:, :, None, :].expand(B, N, N, C)
        hj = h[:, None, :, :].expand(B, N, N, C)

        vec = x[:, :, None, :] - x[:, None, :, :]
        d = torch.norm(vec, dim=-1, keepdim=True)
        d_scalar = d.squeeze(-1)

        direction = vec / (d + 1e-8)

        d_rbf = rbf_distance(
            d_scalar,
            D_min=0.0,
            D_max=4.0,
            D_count=self.rbf_dim,
        )

        d_local_rbf = rbf_distance(
            d_scalar,
            D_min=0.0,
            D_max=0.8,
            D_count=self.local_rbf_dim,
        )

        edge_input = torch.cat(
            [
                z,
                hi,
                hj,
                d,
                d_rbf,
                d_local_rbf,
                pair_type_feat,
            ],
            dim=-1,
        )

        edge_msg = self.edge_mlp(edge_input)

        coord_out = self.coord_head(edge_msg)

        gate = torch.sigmoid(coord_out[..., 0:1])
        weight = torch.tanh(coord_out[..., 1:2])

        # ------------------------------------------------
        # 只允许：
        #   anchor1-linker
        #   anchor2-linker
        #   linker-linker
        # ------------------------------------------------
        allowed_pair = (
            pair_type_mask[..., 0]
            + pair_type_mask[..., 1]
            + pair_type_mask[..., 2]
        ).clamp(max=1.0)

        mask_2d = node_mask[:, :, None] * node_mask[:, None, :]

        if edge_mask is not None:
            if edge_mask.dim() == 4:
                edge_mask = edge_mask.squeeze(-1)
            if edge_mask.dim() != 3:
                edge_mask = edge_mask.view(B, N, N)

            mask_2d = mask_2d * edge_mask.float()

        eye = torch.eye(
            N,
            device=x.device,
            dtype=mask_2d.dtype,
        )[None, :, :]

        mask_2d = mask_2d * (1.0 - eye)
        pair_mask = mask_2d * allowed_pair

        gate = gate * pair_mask[:, :, :, None]

        delta = direction * weight * gate

        denom = gate.sum(dim=2).clamp(min=1.0)
        delta_x = delta.sum(dim=2) / denom

        # ------------------------------------------------
        # 1. 只更新 linker 坐标
        # ------------------------------------------------
        delta_x = delta_x * linker_mask[:, :, None]

        x = x + self.coord_scale * delta_x
        x = x * node_mask[:, :, None]

        # ------------------------------------------------
        # 2. 用同一套 edge message 更新 linker node feature h
        # ------------------------------------------------
        msg = edge_msg * pair_mask[:, :, :, None]

        msg_denom = pair_mask.sum(dim=2).clamp(min=1.0)
        agg_msg = msg.sum(dim=2) / msg_denom[:, :, None]

        dh = self.h_update(
            torch.cat(
                [
                    h,
                    agg_msg,
                ],
                dim=-1,
            )
        )

        # 对 residual 做 norm，而不是对 h + dh 做 norm
        dh = self.h_norm(dh)

        # 只替换 linker atom 的 h；
        # fragment / anchor 的 h 完全保持原样
        linker_mask_3d = linker_mask[:, :, None]
        h_candidate = h + dh
        h = h * (1.0 - linker_mask_3d) + h_candidate * linker_mask_3d
        h = h * node_mask[:, :, None]

        # ------------------------------------------------
        # 3. refined x 之后，重新计算距离特征，并更新 z
        # ------------------------------------------------
        d_new = torch.cdist(x, x)  # [B, N, N]

        d_rbf_new = rbf_distance(
            d_new,
            D_min=0.0,
            D_max=4.0,
            D_count=self.rbf_dim,
        )

        d_local_rbf_new = rbf_distance(
            d_new,
            D_min=0.0,
            D_max=0.8,
            D_count=self.local_rbf_dim,
        )

        edge_msg_allowed = edge_msg * pair_mask[:, :, :, None]

        z_input = torch.cat(
            [
                z,
                edge_msg_allowed,
                d_rbf_new,
                d_local_rbf_new,
                pair_type_feat,
                allowed_pair.unsqueeze(-1),
            ],
            dim=-1,
        )

        dz = self.z_update(z_input)

        # 只允许合法 linker-related pair 更新 z：
        # anchor1-linker / anchor2-linker / linker-linker
        pair_mask_4d = pair_mask[:, :, :, None]
        dz = dz * pair_mask_4d

        # 保持对称
        dz = 0.5 * (dz + dz.transpose(1, 2))

        # 对 residual 做 norm，而不是对 z + dz 做 norm
        dz = self.z_norm(dz)

        # 只替换 allowed pair 的 z；
        # 其他 pair state 保持原样
        z_candidate = z + dz
        z = z * (1.0 - pair_mask_4d) + z_candidate * pair_mask_4d

        z = z * mask_2d[:, :, :, None]
        z = 0.5 * (z + z.transpose(1, 2))

        return h, x, z

class ValueHead(nn.Module):
    def __init__(self, c_s):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(c_s * 3, c_s),
            nn.SiLU(),
            nn.Linear(c_s, c_s),
            nn.SiLU(),
            nn.Linear(c_s, 1),
        )

    @staticmethod
    def masked_mean(h, mask):
        if mask.dim() == 3:
            mask = mask.squeeze(-1)

        mask = mask.float()
        h = h * mask[:, :, None]

        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = h.sum(dim=1) / denom

        return pooled

    def forward(self, h, node_mask, linker_mask=None, t_emb=None):
        global_h = self.masked_mean(h, node_mask)

        if linker_mask is not None:
            linker_h = self.masked_mean(h, linker_mask)
        else:
            linker_h = global_h

        if t_emb is None:
            t_emb = torch.zeros_like(global_h)

        graph_h = torch.cat([global_h, linker_h, t_emb], dim=-1)
        value = self.mlp(graph_h).squeeze(-1)
        return value


class PairStateBondHead(nn.Module):
    def __init__(
            self,
            c_s,
            c_z=64,
            hidden_dim=128,
            rbf_dim=16,
            local_rbf_dim=16,
            dist_norm_value=5.0,
    ):
        super().__init__()

        self.rbf_dim = rbf_dim
        self.local_rbf_dim = local_rbf_dim
        self.dist_norm_value = float(dist_norm_value)

        # distance feature:
        # d + global_rbf + local_rbf + inv_d + low_gate + high_gate + coarse_gate
        self.dist_feat_dim = (
            1
            + rbf_dim
            + local_rbf_dim
            + 4
        )

        self.mlp = nn.Sequential(
            nn.Linear(c_z + c_s * 2 + 5 + self.dist_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),
        )

        self.endpoint_mlp = nn.Sequential(
            nn.Linear(c_s, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.anchor_exist_scale = nn.Parameter(torch.tensor(1.0))
        self.anchor_linker_exist_boost = nn.Parameter(torch.tensor(0.4))

        # 新增：防止 linker-linker bond 过度保守
        self.linker_linker_exist_boost = nn.Parameter(torch.tensor(0.25))

        self.bad_fragment_scale = nn.Parameter(torch.tensor(3.0))
        self.anchor_aromatic_penalty = 10.0

    def build_final_distance_features(self, x):
        """
        x: [B, N, 3]，这里是模型内部 normalized 坐标。
        当前数据里一般是 Å / 5，所以：
            0.75 Å -> 0.15
            2.20 Å -> 0.44

        注意：
            0.75~2.2 只是粗筛特征，不是目标键长。
            真正的键长偏好仍然由 local RBF 让模型自己学习。
        """

        vec = x[:, :, None, :] - x[:, None, :, :]
        d = torch.norm(vec, dim=-1, keepdim=True)      # [B, N, N, 1]
        d_scalar = d.squeeze(-1)                       # [B, N, N]

        d_rbf = rbf_distance(
            d_scalar,
            D_min=0.0,
            D_max=4.0,
            D_count=self.rbf_dim,
        )

        d_local_rbf = rbf_distance(
            d_scalar,
            D_min=0.0,
            D_max=0.8,
            D_count=self.local_rbf_dim,
        )

        inv_d = 1.0 / (1.0 + d_scalar)

        low = 0.75 / self.dist_norm_value
        high = 2.20 / self.dist_norm_value
        sharpness = 12.0

        low_gate = torch.sigmoid(
            sharpness * (d_scalar - low)
        )

        high_gate = torch.sigmoid(
            sharpness * (high - d_scalar)
        )

        coarse_gate = low_gate * high_gate

        extra = torch.stack(
            [
                inv_d,
                low_gate,
                high_gate,
                coarse_gate,
            ],
            dim=-1,
        )

        dist_feat = torch.cat(
            [
                d,
                d_rbf,
                d_local_rbf,
                extra,
            ],
            dim=-1,
        )

        return dist_feat

    def forward(
            self,
            h,
            z,
            x,
            node_mask,
            linker_mask,
            anchor1_mask,
            anchor2_mask,
            fragment1_mask,
            fragment2_mask,
            pair_type_feat,
            pair_type_mask,
    ):
        B, N, C = h.shape

        def squeeze_float(mask):
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            return mask.float()

        node_mask = squeeze_float(node_mask)
        linker_mask = squeeze_float(linker_mask)
        anchor1_mask = squeeze_float(anchor1_mask)
        anchor2_mask = squeeze_float(anchor2_mask)
        fragment1_mask = squeeze_float(fragment1_mask)
        fragment2_mask = squeeze_float(fragment2_mask)

        hi = h[:, :, None, :].expand(B, N, N, C)
        hj = h[:, None, :, :].expand(B, N, N, C)

        dist_feat = self.build_final_distance_features(x)
        dist_feat = dist_feat.to(dtype=h.dtype)

        logits = self.mlp(
            torch.cat(
                [
                    z,
                    hi,
                    hj,
                    pair_type_feat,
                    dist_feat,
                ],
                dim=-1,
            )
        )

        anchor1_linker_edge = pair_type_mask[..., 0]
        anchor2_linker_edge = pair_type_mask[..., 1]
        linker_linker_edge = pair_type_mask[..., 2]
        fragment1_linker_edge = pair_type_mask[..., 3]
        fragment2_linker_edge = pair_type_mask[..., 4]

        anchor_linker_edge = (anchor1_linker_edge + anchor2_linker_edge).clamp(max=1.0)

        # endpoint competition: 每端 anchor 在所有 linker atom 中竞争一个 endpoint。
        endpoint_raw = self.endpoint_mlp(h)  # [B, N, 2]
        endpoint_raw = endpoint_raw.masked_fill(linker_mask[:, :, None] <= 0.5, -1e9)

        n_linker = linker_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        endpoint1_bias = F.log_softmax(endpoint_raw[..., 0], dim=1) + torch.log(n_linker)
        endpoint2_bias = F.log_softmax(endpoint_raw[..., 1], dim=1) + torch.log(n_linker)

        ep1_i = endpoint1_bias[:, :, None].expand(B, N, N)
        ep1_j = endpoint1_bias[:, None, :].expand(B, N, N)
        ep2_i = endpoint2_bias[:, :, None].expand(B, N, N)
        ep2_j = endpoint2_bias[:, None, :].expand(B, N, N)

        anchor1_endpoint_bias = (
            anchor1_mask[:, :, None] * linker_mask[:, None, :] * ep1_j
            + linker_mask[:, :, None] * anchor1_mask[:, None, :] * ep1_i
        )

        anchor2_endpoint_bias = (
            anchor2_mask[:, :, None] * linker_mask[:, None, :] * ep2_j
            + linker_mask[:, :, None] * anchor2_mask[:, None, :] * ep2_i
        )

        endpoint_bias = (anchor1_endpoint_bias + anchor2_endpoint_bias) * anchor_linker_edge

        logits = logits.clone()

        anchor_scale = torch.clamp(self.anchor_exist_scale, min=0.0, max=5.0)
        logits[..., 1:4] = logits[..., 1:4] + anchor_scale * endpoint_bias[..., None]
        logits[..., 0] = logits[..., 0] - anchor_scale * endpoint_bias

        anchor_boost = torch.clamp(self.anchor_linker_exist_boost, min=0.0, max=2.0)
        logits[..., 1:4] = logits[..., 1:4] + anchor_boost * anchor_linker_edge[..., None]
        logits[..., 0] = logits[..., 0] - anchor_boost * anchor_linker_edge

        # linker-linker existence boost：
        # single/double/triple/aromatic 都允许；
        # 不把 aromatic 转 single。
        ll_boost = torch.clamp(self.linker_linker_exist_boost, min=0.0, max=1.5)
        logits[..., 1:] = logits[..., 1:] + ll_boost * linker_linker_edge[..., None]
        logits[..., 0] = logits[..., 0] - ll_boost * linker_linker_edge

        # anchor-linker 不允许 aromatic。
        logits[..., 4] = logits[..., 4] - self.anchor_aromatic_penalty * anchor_linker_edge

        # 非 anchor fragment-linker 抑制。
        bad_fragment_linker_edge = (fragment1_linker_edge + fragment2_linker_edge).clamp(max=1.0)
        bad_scale = torch.clamp(self.bad_fragment_scale, min=0.0, max=10.0)
        logits[..., 1:] = logits[..., 1:] - bad_scale * bad_fragment_linker_edge[..., None]
        logits[..., 0] = logits[..., 0] + bad_scale * bad_fragment_linker_edge

        # 对称化。
        logits = 0.5 * (logits + logits.transpose(1, 2))

        # invalid pair / diagonal 强制 no-bond。
        pair_mask = node_mask[:, :, None] * node_mask[:, None, :]
        eye = torch.eye(N, device=h.device, dtype=pair_mask.dtype)[None, :, :]
        pair_mask = pair_mask * (1.0 - eye)
        invalid = pair_mask <= 0.5

        logits_no_bond = logits[..., 0].masked_fill(invalid, 0.0)
        logits_bond = logits[..., 1:].masked_fill(invalid[..., None], -1e9)
        logits = torch.cat([logits_no_bond[..., None], logits_bond], dim=-1)

        return logits


class LinkerDiffusionModel(nn.Module):
    def __init__(self, model_conf, exp_conf):
        super().__init__()
        self._model_conf = model_conf
        self._exp_conf = exp_conf

        self.c_s = int(self._model_conf.c_s)
        self.c_z = int(getattr(self._model_conf, "c_z", 64))
        self.n_pair_blocks = int(getattr(self._model_conf, "n_pair_blocks", 6))
        self.coord_scale = float(getattr(self._model_conf, "coord_scale", 0.15))
        self.triangle_scale = float(getattr(self._model_conf, "triangle_scale", 0.25))

        self.use_pair_type = bool(
            getattr(self._model_conf, "use_pair_type", True)
        )

        self.use_triangle = bool(
            getattr(self._model_conf, "use_triangle", True)
        )

        self.use_pair_bias = bool(
            getattr(self._model_conf, "use_pair_bias", True)
        )

        self.time_emb = TimeEmbedding(dim=self.c_s)
        self.node_embed = nn.Linear(self._model_conf.num_atom_types + 5, self.c_s)

        self.pair_init = PairStateInitializer(
            c_s=self.c_s,
            c_z=self.c_z,
            rbf_dim=16,
            local_rbf_dim=16,
            pair_type_dim=5,
        )

        self.blocks = nn.ModuleList([
            TopoPairBlock(
                c_s=self.c_s,
                c_z=self.c_z,
                n_heads=self._model_conf.no_heads,
                coord_scale=self.coord_scale,
                triangle_scale=self.triangle_scale,
                use_triangle=self.use_triangle,
                use_pair_bias=self.use_pair_bias,
            )
            for _ in range(self.n_pair_blocks)
        ])

        self.use_local_geometry_refiner = bool(
            getattr(self._model_conf, "use_local_geometry_refiner", False)
        )

        self.n_local_geom_blocks = int(
            getattr(self._model_conf, "n_local_geom_blocks", 0)
        )

        self.local_geom_hidden_dim = int(
            getattr(self._model_conf, "local_geom_hidden_dim", 128)
        )

        self.local_geom_coord_scale = float(
            getattr(self._model_conf, "local_geom_coord_scale", 0.06)
        )

        if self.use_local_geometry_refiner and self.n_local_geom_blocks > 0:
            self.local_geom_refiners = nn.ModuleList([
                LinkerLocalGeometryRefiner(
                    c_s=self.c_s,
                    c_z=self.c_z,
                    hidden_dim=self.local_geom_hidden_dim,
                    rbf_dim=16,
                    local_rbf_dim=16,
                    coord_scale=self.local_geom_coord_scale,
                )
                for _ in range(self.n_local_geom_blocks)
            ])
        else:
            self.local_geom_refiners = nn.ModuleList([])

        self.noise_head = FeatureNoiseHead(
            c_s=self.c_s,
            num_atom_types=self._model_conf.num_atom_types,
        )

        self.value_head = ValueHead(c_s=self.c_s)

        self.bond_head = PairStateBondHead(
            c_s=self.c_s,
            c_z=self.c_z,
            hidden_dim=self.c_s,
            rbf_dim=16,
            local_rbf_dim=16,
            dist_norm_value=5.0,
        )

    @staticmethod
    def squeeze_mask(mask):
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        return mask

    def build_pair_type(
            self,
            anchor1_mask,
            anchor2_mask,
            fragment1_mask,
            fragment2_mask,
            linker_mask,
    ):
        anchor1_mask = anchor1_mask.float()
        anchor2_mask = anchor2_mask.float()
        fragment1_mask = fragment1_mask.float()
        fragment2_mask = fragment2_mask.float()
        linker_mask = linker_mask.float()

        anchor1_linker_edge = (
            anchor1_mask[:, :, None] * linker_mask[:, None, :]
            + linker_mask[:, :, None] * anchor1_mask[:, None, :]
        ).clamp(max=1.0)

        anchor2_linker_edge = (
            anchor2_mask[:, :, None] * linker_mask[:, None, :]
            + linker_mask[:, :, None] * anchor2_mask[:, None, :]
        ).clamp(max=1.0)

        linker_linker_edge = (
            linker_mask[:, :, None] * linker_mask[:, None, :]
        ).clamp(max=1.0)

        non_anchor_fragment1_mask = fragment1_mask * (1.0 - anchor1_mask)
        non_anchor_fragment2_mask = fragment2_mask * (1.0 - anchor2_mask)

        fragment1_linker_edge = (
            non_anchor_fragment1_mask[:, :, None] * linker_mask[:, None, :]
            + linker_mask[:, :, None] * non_anchor_fragment1_mask[:, None, :]
        ).clamp(max=1.0)

        fragment2_linker_edge = (
            non_anchor_fragment2_mask[:, :, None] * linker_mask[:, None, :]
            + linker_mask[:, :, None] * non_anchor_fragment2_mask[:, None, :]
        ).clamp(max=1.0)

        pair_type = torch.stack(
            [
                anchor1_linker_edge,
                anchor2_linker_edge,
                linker_linker_edge,
                fragment1_linker_edge,
                fragment2_linker_edge,
            ],
            dim=-1,
        )

        return pair_type

    def forward(
            self,
            h,
            x,
            anchor1_mask,
            anchor2_mask,
            fragment1_mask,
            fragment2_mask,
            linker_mask,
            node_mask,
            t,
            edge_mask,
            return_value=False,
            return_bond=False,
    ):

        node_mask_2d = self.squeeze_mask(node_mask).float()
        linker_mask_2d = self.squeeze_mask(linker_mask).float()
        anchor1_mask_2d = self.squeeze_mask(anchor1_mask).float()
        anchor2_mask_2d = self.squeeze_mask(anchor2_mask).float()
        fragment1_mask_2d = self.squeeze_mask(fragment1_mask).float()
        fragment2_mask_2d = self.squeeze_mask(fragment2_mask).float()

        if t.dim() == 2:
            t = t.squeeze(-1)

        pair_type = self.build_pair_type(
            anchor1_mask=anchor1_mask_2d,
            anchor2_mask=anchor2_mask_2d,
            fragment1_mask=fragment1_mask_2d,
            fragment2_mask=fragment2_mask_2d,
            linker_mask=linker_mask_2d,
        )

        pair_type_mask = pair_type

        if self.use_pair_type:
            pair_type_feat = pair_type
        else:
            pair_type_feat = torch.zeros_like(pair_type)

        x_t = x.clone()

        h_in = torch.cat(
            [
                h,
                anchor1_mask,
                anchor2_mask,
                fragment1_mask,
                fragment2_mask,
                linker_mask,
            ],
            dim=-1,
        )

        h_new = self.node_embed(h_in)

        t_emb = self.time_emb(t)
        h_new = h_new + t_emb[:, None, :]
        h_new = h_new * node_mask_2d[:, :, None]

        # 显式 pair state。
        z = self.pair_init(
            h=h_new,
            x=x_t,
            pair_type=pair_type_feat,
            node_mask=node_mask_2d,
        )

        for block in self.blocks:
            h_new, x_t, z = block(
                h=h_new,
                x=x_t,
                z=z,
                node_mask=node_mask_2d,
                linker_mask=linker_mask_2d,
                pair_type=pair_type_feat,
                edge_mask=edge_mask,
            )

        for refiner in self.local_geom_refiners:
            h_new, x_t, z = refiner(
                h=h_new,
                x=x_t,
                z=z,
                node_mask=node_mask_2d,
                linker_mask=linker_mask_2d,
                pair_type_feat=pair_type_feat,
                pair_type_mask=pair_type_mask,
                edge_mask=edge_mask,
            )

        bond_logits = None
        if return_bond:
            bond_logits = self.bond_head(
                h=h_new,
                z=z,
                x=x_t,
                node_mask=node_mask_2d,
                linker_mask=linker_mask_2d,
                anchor1_mask=anchor1_mask_2d,
                anchor2_mask=anchor2_mask_2d,
                fragment1_mask=fragment1_mask_2d,
                fragment2_mask=fragment2_mask_2d,
                pair_type_feat=pair_type_feat,
                pair_type_mask=pair_type_mask,
            )

        feature_noise = self.noise_head(
            h_new,
            t_emb,
            linker_mask_2d,
        )

        value = None
        if return_value:
            value = self.value_head(
                h_new,
                node_mask=node_mask_2d,
                linker_mask=linker_mask_2d,
                t_emb=t_emb,
            )

        if return_value and return_bond:
            return feature_noise, x_t, value, bond_logits

        if return_value:
            return feature_noise, x_t, value

        if return_bond:
            return feature_noise, x_t, bond_logits

        return feature_noise, x_t
