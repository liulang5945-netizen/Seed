"""Taiji Native Non-Transformer Cognitive Backbone (`taiji-native-cognitive-backbone-v1`).

Implements the non-Transformer core algorithm of `TAIJI_CORE_ALGORITHM_V1_DRAFT`:
  * ZERO standard Transformer / scaled_dot_product_attention / GQA blocks;
  * Three-timescale neural plasticity: slow knowledge (`w_slow`, `Theta`), medium
    low-rank adaptation (`psi_u @ psi_v^T`, `Psi`), and fast associative weights `F_t`;
  * Exact rank-1 proximal-erase + independent-write associative fast weights (`§6.2-6.5`)
    with `p_allowed=0.0` strict freeze and `Frobenius <= C_F` norm cap;
  * Multi-timescale leaky neural state `H_t` with causal depthwise local encoding;
  * Native built-in neutral structural growth primitives (`grow_group_units_neutral_`,
    `grow_depth_neutral_`, `grow_fast_heads_neutral_`) + selective plasticity grad mask;
  * Bit-exact `SessionStore` capsule export/load preserving grown topology & lineage.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

SCHEMA = "taiji-native-cognitive-backbone-v1"

UNIT_IMMATURE = 0
UNIT_MATURE = 1
UNIT_DORMANT = 2
UNIT_RETIRED = 3


def smooth_unit_ball(x: torch.Tensor) -> torch.Tensor:
    """Smooth unit ball mapping B(x) = x / sqrt(1 + ||x||_2^2) (§6.2)."""
    norm_sq = (x * x).sum(dim=-1, keepdim=True)
    return x * torch.rsqrt(1.0 + norm_sq)


def scale_floor_norm(h: torch.Tensor) -> torch.Tensor:
    """Scale-floor normalization N(h) = h / sqrt(1 + mean(h^2)) (§5.2)."""
    return h * torch.rsqrt(1.0 + h.pow(2).mean(dim=-1, keepdim=True))


@dataclass(frozen=True)
class NativeBackboneConfig:
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 8
    group_units: int | tuple[int, ...] = 384
    n_fast_heads: int | tuple[int, ...] = 4
    fast_key_dim: int = 32
    fast_val_dim: int = 32
    psi_rank: int = 16
    tau_min: float = 0.5
    tau_max: float = 128.0
    fast_norm_cap: float = 16.0
    chunk_size: int = 16
    max_seq_len: int = 512

    def layer_group_units(self, idx: int) -> int:
        if isinstance(self.group_units, int):
            return self.group_units
        return self.group_units[idx]

    def layer_fast_heads(self, idx: int) -> int:
        if isinstance(self.n_fast_heads, int):
            return self.n_fast_heads
        return self.n_fast_heads[idx]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if isinstance(self.group_units, tuple):
            d["group_units"] = list(self.group_units)
        if isinstance(self.n_fast_heads, tuple):
            d["n_fast_heads"] = list(self.n_fast_heads)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NativeBackboneConfig:
        d = dict(data)
        if isinstance(d.get("group_units"), list):
            d["group_units"] = tuple(int(x) for x in d["group_units"])
        if isinstance(d.get("n_fast_heads"), list):
            d["n_fast_heads"] = tuple(int(x) for x in d["n_fast_heads"])
        return cls(**d)


NATIVE_PRESETS: dict[str, NativeBackboneConfig] = {
    "native_tiny_2m": NativeBackboneConfig(
        vocab_size=8192,
        d_model=128,
        n_layers=4,
        group_units=128,
        n_fast_heads=2,
        fast_key_dim=16,
        fast_val_dim=16,
        psi_rank=8,
    ),
    "native_small_8m": NativeBackboneConfig(
        vocab_size=8192,
        d_model=256,
        n_layers=6,
        group_units=256,
        n_fast_heads=4,
        fast_key_dim=24,
        fast_val_dim=24,
        psi_rank=16,
    ),
    "native_mid_18m": NativeBackboneConfig(
        vocab_size=8192,
        d_model=384,
        n_layers=10,
        group_units=1024,
        n_fast_heads=6,
        fast_key_dim=32,
        fast_val_dim=32,
        psi_rank=16,
    ),
}


@dataclass
class NativeCognitiveState:
    """Persistent cross-turn / cross-session state (H_t + associative fast weights F_t)."""

    h_states: list[torch.Tensor]
    fast_weights: list[torch.Tensor]


class NativeNeuralGroupBlock(nn.Module):
    """Single Taiji native neural group layer with 3-timescale plasticity & fast weights."""

    def __init__(
        self,
        d_model: int,
        group_units: int,
        n_fast_heads: int,
        fast_key_dim: int,
        fast_val_dim: int,
        psi_rank: int,
        tau_min: float = 0.5,
        tau_max: float = 128.0,
        fast_norm_cap: float = 16.0,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.group_units = int(group_units)
        self.n_fast_heads = int(n_fast_heads)
        self.fast_key_dim = int(fast_key_dim)
        self.fast_val_dim = int(fast_val_dim)
        self.psi_rank = int(psi_rank)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.fast_norm_cap = float(fast_norm_cap)
        self.chunk_size = max(1, int(chunk_size))

        # Causal depthwise local encoder (kernel=4, left-padded)
        self.local_conv = nn.Conv1d(
            self.d_model, self.d_model, kernel_size=4, padding=0, groups=self.d_model, bias=False
        )

        # Multi-timescale leaky recurrence rho initialized across tau in [1, 4, 16, 64]
        taus = torch.tensor([1.0, 4.0, 16.0, 64.0], dtype=torch.float32).repeat(
            (self.d_model + 3) // 4
        )[: self.d_model]
        frac = ((taus - self.tau_min) / max(1e-6, self.tau_max - self.tau_min)).clamp(
            1e-4, 1 - 1e-4
        )
        self.rho = nn.Parameter(torch.log(frac / (1.0 - frac)))

        # Slow knowledge weights (Theta)
        self.w_slow = nn.Linear(self.d_model, self.d_model, bias=False)

        # Medium low-rank adaptation Psi = U @ V^T (U zero-init, V non-zero init)
        self.psi_u = nn.Parameter(torch.zeros(self.d_model, self.psi_rank))
        self.psi_v = nn.Parameter(
            torch.randn(self.d_model, self.psi_rank) / math.sqrt(self.d_model)
        )

        # Growable neural group units: P @ tanh(R @ x)
        self.group_in = nn.Linear(self.d_model, self.group_units, bias=False)
        self.group_out = nn.Linear(self.group_units, self.d_model, bias=False)
        self.register_buffer(
            "unit_stage", torch.full((self.group_units,), UNIT_IMMATURE, dtype=torch.long)
        )
        self.unit_stage: torch.Tensor

        # Associative fast-weight projections (F_t: L x k x v)
        self.fast_q = nn.Linear(self.d_model, self.n_fast_heads * self.fast_key_dim, bias=False)
        self.fast_k = nn.Linear(self.d_model, self.n_fast_heads * self.fast_key_dim, bias=False)
        self.fast_v = nn.Linear(self.d_model, self.n_fast_heads * self.fast_val_dim, bias=False)
        self.fast_read_out = nn.Linear(
            self.n_fast_heads * self.fast_val_dim, self.d_model, bias=False
        )

        # Fast-weight gates: alpha (decay), b (erase dir), w (write), erase_strength, chi
        self.fast_alpha = nn.Linear(self.d_model, self.n_fast_heads * self.fast_key_dim, bias=True)
        self.fast_b = nn.Linear(self.d_model, self.n_fast_heads * self.fast_key_dim, bias=True)
        self.fast_w = nn.Linear(self.d_model, self.n_fast_heads * self.fast_val_dim, bias=True)
        self.fast_eta = nn.Linear(self.d_model, self.n_fast_heads, bias=True)
        self.fast_chi = nn.Linear(self.d_model, self.n_fast_heads, bias=True)

        # Initialize chi bias slightly negative for conservative fast-memory writes
        nn.init.constant_(self.fast_chi.bias, -1.0)
        nn.init.constant_(self.fast_alpha.bias, 2.0)

    def alive_unit_mask(self) -> torch.Tensor:
        alive = (self.unit_stage == UNIT_IMMATURE) | (self.unit_stage == UNIT_MATURE)
        return alive.to(dtype=self.group_in.weight.dtype)

    def _fast_step(
        self,
        u_t: torch.Tensor,
        f_state: torch.Tensor,
        p_allowed: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step read-before-write + rank-1 proximal-erase & independent-write (§6.2)."""
        bsz = u_t.shape[0]
        l_heads = self.n_fast_heads
        k_dim = self.fast_key_dim
        v_dim = self.fast_val_dim

        q = smooth_unit_ball(self.fast_q(u_t).view(bsz, l_heads, k_dim))
        # Read before write: v_read = F_t^T @ q
        v_read = torch.einsum("blkv,blk->blv", f_state, q).reshape(bsz, l_heads * v_dim)
        f_out = self.fast_read_out(v_read)

        if p_allowed == 0.0:
            return f_out, f_state

        k = smooth_unit_ball(self.fast_k(u_t).view(bsz, l_heads, k_dim))
        v = smooth_unit_ball(torch.tanh(self.fast_v(u_t).view(bsz, l_heads, v_dim)))

        alpha = torch.sigmoid(self.fast_alpha(u_t)).view(bsz, l_heads, k_dim)
        b = torch.sigmoid(self.fast_b(u_t)).view(bsz, l_heads, k_dim)
        w = torch.sigmoid(self.fast_w(u_t)).view(bsz, l_heads, v_dim)
        eta_e = 16.0 * torch.sigmoid(self.fast_eta(u_t)).view(bsz, l_heads, 1, 1)
        chi = p_allowed * torch.sigmoid(self.fast_chi(u_t)).view(bsz, l_heads, 1, 1)

        f_bar = alpha.unsqueeze(-1) * f_state
        e = b * k
        e_norm_sq = (e * e).sum(dim=-1, keepdim=True).unsqueeze(-1)
        e_dot_f = torch.einsum("blk,blkv->blv", e, f_bar).unsqueeze(-2)
        f_erase = f_bar - (eta_e * e.unsqueeze(-1) * e_dot_f) / (1.0 + eta_e * e_norm_sq)

        write_term = k.unsqueeze(-1) * (w * v).unsqueeze(-2)
        f_tilde = f_erase + write_term
        f_next = (1.0 - chi) * f_state + chi * f_tilde

        # Frobenius norm cap per head
        f_norm = torch.sqrt((f_next * f_next).sum(dim=(-2, -1), keepdim=True) + 1e-12)
        scale = torch.clamp(self.fast_norm_cap / f_norm, max=1.0)
        f_next = f_next * scale
        return f_out, f_next

    def forward(
        self,
        x: torch.Tensor,
        h_state: torch.Tensor,
        f_state: torch.Tensor,
        p_allowed: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: (B, T, d_model) -> (out, next_h_state, next_f_state)."""
        bsz, seq_len, _ = x.shape

        # 1. Causal local depthwise conv
        x_conv_in = F.pad(x.transpose(1, 2), (3, 0))
        local_feat = self.local_conv(x_conv_in).transpose(1, 2)
        u = scale_floor_norm(x + local_feat)

        # 2. Multi-timescale leaky state H_t (causal)
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.rho)
        ell = (1.0 - torch.exp(-1.0 / tau)).to(dtype=x.dtype).view(1, -1)

        # 3. Slow + Psi + Growable neural group units (vectorized over T)
        psi_out = (u @ self.psi_v) @ self.psi_u.t()
        group_h = torch.tanh(self.group_in(u)) * self.alive_unit_mask()
        group_out = self.group_out(group_h)
        drive = torch.tanh(self.w_slow(u) + psi_out + group_out)

        h_cur = h_state
        h_seq: list[torch.Tensor] = []
        f_cur = f_state
        f_seq: list[torch.Tensor] = []

        for t in range(seq_len):
            h_cur = (1.0 - ell) * h_cur + ell * drive[:, t, :]
            h_seq.append(h_cur)
            f_out_t, f_cur = self._fast_step(u[:, t, :], f_cur, p_allowed=p_allowed)
            f_seq.append(f_out_t)

        h_stack = torch.stack(h_seq, dim=1)
        f_stack = torch.stack(f_seq, dim=1)
        out = x + h_stack + f_stack
        return out, h_cur, f_cur


class TaijiNativeCognitiveBackbone(nn.Module):
    """Taiji native non-Transformer cognitive backbone with built-in neutral growth."""

    def __init__(self, cfg: NativeBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.growth_lineage: list[dict[str, Any]] = []

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                NativeNeuralGroupBlock(
                    d_model=cfg.d_model,
                    group_units=cfg.layer_group_units(i),
                    n_fast_heads=cfg.layer_fast_heads(i),
                    fast_key_dim=cfg.fast_key_dim,
                    fast_val_dim=cfg.fast_val_dim,
                    psi_rank=cfg.psi_rank,
                    tau_min=cfg.tau_min,
                    tau_max=cfg.tau_max,
                    fast_norm_cap=cfg.fast_norm_cap,
                    chunk_size=cfg.chunk_size,
                )
                for i in range(cfg.n_layers)
            ]
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied embeddings

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> NativeCognitiveState:
        dev = device or self.tok_emb.weight.device
        dt = dtype or self.tok_emb.weight.dtype
        h_states: list[torch.Tensor] = []
        fast_weights: list[torch.Tensor] = []
        for blk in self.blocks:
            b = cast(NativeNeuralGroupBlock, blk)
            h_states.append(torch.zeros(batch_size, b.d_model, device=dev, dtype=dt))
            fast_weights.append(
                torch.zeros(
                    batch_size,
                    b.n_fast_heads,
                    b.fast_key_dim,
                    b.fast_val_dim,
                    device=dev,
                    dtype=dt,
                )
            )
        return NativeCognitiveState(h_states=h_states, fast_weights=fast_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: NativeCognitiveState | None = None,
        labels: torch.Tensor | None = None,
        p_allowed: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, NativeCognitiveState]:
        bsz, _ = input_ids.shape
        x = self.tok_emb(input_ids)
        if state is None:
            state = self.init_state(bsz, device=x.device, dtype=x.dtype)

        next_h: list[torch.Tensor] = []
        next_f: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            b = cast(NativeNeuralGroupBlock, blk)
            x, h_i, f_i = b(x, state.h_states[i], state.fast_weights[i], p_allowed=p_allowed)
            next_h.append(h_i)
            next_f.append(f_i)

        x_norm = scale_floor_norm(x)
        logits = self.lm_head(x_norm)
        loss: torch.Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), labels.view(-1))
        return logits, loss, NativeCognitiveState(h_states=next_h, fast_weights=next_f)

    def _refresh_cfg(self) -> None:
        units = tuple(cast(NativeNeuralGroupBlock, b).group_units for b in self.blocks)
        heads = tuple(cast(NativeNeuralGroupBlock, b).n_fast_heads for b in self.blocks)
        self.cfg = NativeBackboneConfig(
            vocab_size=self.cfg.vocab_size,
            d_model=self.cfg.d_model,
            n_layers=len(self.blocks),
            group_units=units,
            n_fast_heads=heads,
            fast_key_dim=self.cfg.fast_key_dim,
            fast_val_dim=self.cfg.fast_val_dim,
            psi_rank=self.cfg.psi_rank,
            tau_min=self.cfg.tau_min,
            tau_max=self.cfg.tau_max,
            fast_norm_cap=self.cfg.fast_norm_cap,
            chunk_size=self.cfg.chunk_size,
            max_seq_len=self.cfg.max_seq_len,
        )

    @torch.no_grad()
    def grow_group_units_neutral_(
        self, layer_idx: int | None = None, delta_units: int = 32, seed: int = 0
    ) -> dict[str, Any]:
        """Expand neural group units neutrally (zero outgoing weights => 0.0 birth delta)."""
        if delta_units <= 0:
            raise ValueError("delta_units must be positive")
        targets = list(range(len(self.blocks))) if layer_idx is None else [int(layer_idx)]
        before_params = self.count_parameters()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            for idx in targets:
                blk = cast(NativeNeuralGroupBlock, self.blocks[idx])
                dev = blk.group_in.weight.device
                dt = blk.group_in.weight.dtype
                old_u = blk.group_units
                new_u = old_u + delta_units

                new_in = nn.Linear(blk.d_model, new_u, bias=False, device=dev, dtype=dt)
                new_in.weight[:old_u].copy_(blk.group_in.weight)
                nn.init.normal_(new_in.weight[old_u:], std=1.0 / math.sqrt(blk.d_model))

                new_out = nn.Linear(new_u, blk.d_model, bias=False, device=dev, dtype=dt)
                new_out.weight[:, :old_u].copy_(blk.group_out.weight)
                new_out.weight[:, old_u:].zero_()

                new_stage = torch.full((new_u,), UNIT_IMMATURE, dtype=torch.long, device=dev)
                new_stage[:old_u].copy_(blk.unit_stage)

                blk.group_in = new_in
                blk.group_out = new_out
                blk.unit_stage = new_stage
                blk.group_units = new_u

        self._refresh_cfg()
        after_params = self.count_parameters()
        event = {
            "op": "grow_group_units_neutral",
            "layers": targets,
            "delta_units": delta_units,
            "before_params": before_params,
            "after_params": after_params,
        }
        self.growth_lineage.append(event)
        return event

    @torch.no_grad()
    def grow_fast_heads_neutral_(
        self, layer_idx: int | None = None, delta_heads: int = 1, seed: int = 0
    ) -> dict[str, Any]:
        """Expand associative fast-weight heads neutrally (zero fast_read_out cols)."""
        if delta_heads <= 0:
            raise ValueError("delta_heads must be positive")
        targets = list(range(len(self.blocks))) if layer_idx is None else [int(layer_idx)]
        before_params = self.count_parameters()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            for idx in targets:
                blk = cast(NativeNeuralGroupBlock, self.blocks[idx])
                dev = blk.fast_q.weight.device
                dt = blk.fast_q.weight.dtype
                old_h = blk.n_fast_heads
                new_h = old_h + delta_heads
                k_dim = blk.fast_key_dim
                v_dim = blk.fast_val_dim

                def _expand_linear_out(
                    old_lin: nn.Linear,
                    old_out: int,
                    new_out: int,
                    init_bias: float = 0.0,
                    dev: torch.device = dev,
                    dt: torch.dtype = dt,
                ) -> nn.Linear:
                    lin = nn.Linear(
                        old_lin.in_features,
                        new_out,
                        bias=old_lin.bias is not None,
                        device=dev,
                        dtype=dt,
                    )
                    lin.weight[:old_out].copy_(old_lin.weight)
                    nn.init.normal_(lin.weight[old_out:], std=1.0 / math.sqrt(old_lin.in_features))
                    if lin.bias is not None and old_lin.bias is not None:
                        lin.bias[:old_out].copy_(old_lin.bias)
                        lin.bias[old_out:].fill_(init_bias)
                    return lin

                blk.fast_q = _expand_linear_out(blk.fast_q, old_h * k_dim, new_h * k_dim)
                blk.fast_k = _expand_linear_out(blk.fast_k, old_h * k_dim, new_h * k_dim)
                blk.fast_v = _expand_linear_out(blk.fast_v, old_h * v_dim, new_h * v_dim)
                blk.fast_alpha = _expand_linear_out(
                    blk.fast_alpha, old_h * k_dim, new_h * k_dim, init_bias=2.0
                )
                blk.fast_b = _expand_linear_out(blk.fast_b, old_h * k_dim, new_h * k_dim)
                blk.fast_w = _expand_linear_out(blk.fast_w, old_h * v_dim, new_h * v_dim)
                blk.fast_eta = _expand_linear_out(blk.fast_eta, old_h, new_h)
                blk.fast_chi = _expand_linear_out(blk.fast_chi, old_h, new_h, init_bias=-1.0)

                new_ro = nn.Linear(new_h * v_dim, blk.d_model, bias=False, device=dev, dtype=dt)
                new_ro.weight[:, : old_h * v_dim].copy_(blk.fast_read_out.weight)
                new_ro.weight[:, old_h * v_dim :].zero_()
                blk.fast_read_out = new_ro
                blk.n_fast_heads = new_h

        self._refresh_cfg()
        after_params = self.count_parameters()
        event = {
            "op": "grow_fast_heads_neutral",
            "layers": targets,
            "delta_heads": delta_heads,
            "before_params": before_params,
            "after_params": after_params,
        }
        self.growth_lineage.append(event)
        return event

    @torch.no_grad()
    def grow_depth_neutral_(self, insert_after: int | None = None, seed: int = 0) -> dict[str, Any]:
        """Insert a new NativeNeuralGroupBlock as an exact residual identity (x + 0 = x)."""
        before_params = self.count_parameters()
        pos = len(self.blocks) if insert_after is None else int(insert_after) + 1
        dev = self.tok_emb.weight.device
        dt = self.tok_emb.weight.dtype
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            new_blk = NativeNeuralGroupBlock(
                d_model=self.cfg.d_model,
                group_units=self.cfg.layer_group_units(0),
                n_fast_heads=self.cfg.layer_fast_heads(0),
                fast_key_dim=self.cfg.fast_key_dim,
                fast_val_dim=self.cfg.fast_val_dim,
                psi_rank=self.cfg.psi_rank,
                tau_min=self.cfg.tau_min,
                tau_max=self.cfg.tau_max,
                fast_norm_cap=self.cfg.fast_norm_cap,
                chunk_size=self.cfg.chunk_size,
            ).to(device=dev, dtype=dt)
            # Zero-init all output projections so block output == x at birth
            new_blk.w_slow.weight.zero_()
            new_blk.psi_u.zero_()
            new_blk.group_out.weight.zero_()
            new_blk.fast_read_out.weight.zero_()
            # Force ell -> 0 at birth by setting rho very negative so H_t starts at 0
            new_blk.rho.fill_(-20.0)

        blocks_list = list(self.blocks)
        blocks_list.insert(pos, new_blk)
        self.blocks = nn.ModuleList(blocks_list)
        self._refresh_cfg()
        after_params = self.count_parameters()
        event = {
            "op": "grow_depth_neutral",
            "insert_pos": pos,
            "before_params": before_params,
            "after_params": after_params,
        }
        self.growth_lineage.append(event)
        return event

    @torch.no_grad()
    def mark_all_mature_(self) -> int:
        """Promote all immature group units to mature."""
        promoted = 0
        for blk in self.blocks:
            b = cast(NativeNeuralGroupBlock, blk)
            mask = b.unit_stage == UNIT_IMMATURE
            promoted += int(mask.sum().item())
            b.unit_stage[mask] = UNIT_MATURE
        return promoted

    def apply_selective_plasticity_grad_mask(
        self, freeze_mature_slow: bool = True
    ) -> dict[str, int]:
        """Zero gradients on mature units (and w_slow) while keeping immature units & Psi plastic."""
        masked_units = 0
        masked_slow = 0
        for blk in self.blocks:
            b = cast(NativeNeuralGroupBlock, blk)
            mature_mask = b.unit_stage == UNIT_MATURE
            n_mat = int(mature_mask.sum().item())
            if n_mat > 0:
                masked_units += n_mat
                if b.group_in.weight.grad is not None:
                    b.group_in.weight.grad[mature_mask] = 0.0
                if b.group_out.weight.grad is not None:
                    b.group_out.weight.grad[:, mature_mask] = 0.0
            if freeze_mature_slow and b.w_slow.weight.grad is not None:
                b.w_slow.weight.grad.zero_()
                masked_slow += 1
        return {"masked_units": masked_units, "masked_slow_layers": masked_slow}

    def topology_summary(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "non_transformer": True,
            "params": self.count_parameters(),
            "d_model": self.cfg.d_model,
            "n_layers": len(self.blocks),
            "group_units_per_layer": [
                cast(NativeNeuralGroupBlock, b).group_units for b in self.blocks
            ],
            "fast_heads_per_layer": [
                cast(NativeNeuralGroupBlock, b).n_fast_heads for b in self.blocks
            ],
            "psi_rank": self.cfg.psi_rank,
            "growth_events": len(self.growth_lineage),
            "growth_lineage": list(self.growth_lineage),
        }

    def export_capsule(self, prefix: str = "native_zh") -> dict[str, Any]:
        """Export state_dict + grown topology metadata into a SessionStore-compatible dict."""
        sd = {f"{prefix}.w.{k}": v.detach().cpu().clone() for k, v in self.state_dict().items()}
        sd[f"{prefix}.cfg"] = self.cfg.to_dict()
        sd[f"{prefix}.lineage"] = list(self.growth_lineage)
        return sd

    @classmethod
    def from_capsule(
        cls, capsule: dict[str, Any], prefix: str = "native_zh"
    ) -> TaijiNativeCognitiveBackbone:
        cfg = NativeBackboneConfig.from_dict(capsule[f"{prefix}.cfg"])
        model = cls(cfg)
        model.load_capsule(capsule, prefix=prefix)
        return model

    def load_capsule(self, capsule: dict[str, Any], prefix: str = "native_zh") -> None:
        pfx = f"{prefix}.w."
        sd = {k[len(pfx) :]: v for k, v in capsule.items() if k.startswith(pfx)}
        self.load_state_dict(sd, strict=True)
        self.growth_lineage = list(capsule.get(f"{prefix}.lineage", []))
