"""态极 Native Byte-Group Cognitive Backbone (`taiji-native-byte-group-v2`).

Strictly aligned with `TAIJI_CORE_ALGORITHM_V1_DRAFT`, `SPEC-TAIJI-CLOSED_LOOP_U2`,
and `SPEC-TAIJI_U2_NEURAL_UNIT_LIFECYCLE`:
  * Pure UTF-8 raw byte input (`BYTE_VOCAB = 259`, `OUT_VOCAB = 257`) with 3-layer causal
    Conv1d (`k=3, d=1/2/4`) -> 8-byte patch encoder (ZERO BPE, ZERO Transformer);
  * Heterogeneous persistent neural groups (`tau_init` in `{1, 4, 16, 64}`) with
    synchronous Q/K/V inter-group messaging and additive residual groups;
  * Stateful neural units inside each group (`unit_id`, `z`, `a`, directed `W_rec`,
    `c`, `eta`, `b`, and 5-stage lifecycle `proposed/immature/mature/dormant/retired`);
  * Exact FP1 KKT constrained quadratic fast memory (`taiji.u2_operators.fp1`) with
    `allowed=False` bit-exact freeze and `use_fast_memory=False` no-F ablation switch;
  * SVD zero-in / non-zero-out neutral birth (`grow_stateful_units_svd_neutral_` using
    `u2_operators.credit_matrix / choose_left_directions / append_neutral_units`),
    inheritance proliferation (`proliferate_units_neutral_`), and additive residual group
    growth (`grow_group_neutral_`);
  * Strict UTF-8 legal byte decoding via `taiji.u2_operators.utf8_mask` + `utf8_state`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from taiji.dynamic_byte_chunker import DynamicByteVocab
from taiji.u2_operators import (
    CursorState,
    Emission,
    append_neutral_units,
    choose_left_directions,
    credit_matrix,
    fp1,
    utf8_mask,
)
from taiji.u2_operators import (
    accept as u2_accept,
)
from taiji.u2_operators import (
    emission as u2_emission,
)
from taiji.u2_operators import (
    record_inputs as u2_record_inputs,
)
from taiji.u2_operators import (
    revoke_records as u2_revoke_records,
)
from taiji.utf8_state import advance_utf8

SCHEMA = "taiji-native-byte-group-v2"

BYTE_VOCAB = 259  # 0..255 raw UTF-8 bytes, 256=EOS, 257=BOS, 258=PAD
OUT_VOCAB = 257  # 0..255 + 256=EOS (matches u2_operators.utf8_mask)
EOS_BYTE = 256
BOS_BYTE = 257
PAD_BYTE = 258

UNIT_STAGES = ("proposed", "immature", "mature", "dormant", "retired")
STAGE_ID = {name: i for i, name in enumerate(UNIT_STAGES)}


def smooth_unit_ball(x: torch.Tensor) -> torch.Tensor:
    """Smooth unit ball mapping B(x) = x / sqrt(1 + ||x||_2^2) (§6.2)."""
    norm_sq = (x * x).sum(dim=-1, keepdim=True)
    return x * torch.rsqrt(1.0 + norm_sq)


def scale_floor_norm(h: torch.Tensor) -> torch.Tensor:
    """Scale-floor normalization N(h) = h / sqrt(1 + mean(h^2)) (§5.2)."""
    return h * torch.rsqrt(1.0 + h.pow(2).mean(dim=-1, keepdim=True))


def bits_per_byte(loss_nats: float) -> float:
    """Convert natural-log cross-entropy (nats/byte) to bits-per-byte (BPB)."""
    return float(loss_nats) / math.log(2.0)


@dataclass(frozen=True)
class NativeBackboneConfig:
    d_model: int = 256
    byte_emb_dim: int = 64
    patch_size: int = 8
    n_groups: int = 4
    group_units: int | tuple[int, ...] = 256
    fast_key_dim: int = 32
    fast_val_dim: int = 32
    fast_radius: float = 4.0
    psi_rank: int = 16
    tau_min: float = 0.5
    tau_max: float = 128.0
    use_fast_memory: bool = True
    max_seq_bytes: int = 512

    def group_unit_count(self, idx: int) -> int:
        if isinstance(self.group_units, int):
            return self.group_units
        return self.group_units[idx]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if isinstance(self.group_units, tuple):
            d["group_units"] = list(self.group_units)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NativeBackboneConfig:
        d = dict(data)
        if isinstance(d.get("group_units"), list):
            d["group_units"] = tuple(int(x) for x in d["group_units"])
        return cls(**d)


NATIVE_PRESETS: dict[str, NativeBackboneConfig] = {
    "native_byte_tiny": NativeBackboneConfig(
        d_model=128,
        byte_emb_dim=64,
        patch_size=8,
        n_groups=4,
        group_units=96,
        fast_key_dim=16,
        fast_val_dim=16,
        fast_radius=4.0,
        psi_rank=8,
        max_seq_bytes=256,
    ),
    "native_byte_small": NativeBackboneConfig(
        d_model=256,
        byte_emb_dim=64,
        patch_size=8,
        n_groups=4,
        group_units=384,
        fast_key_dim=32,
        fast_val_dim=32,
        fast_radius=4.0,
        psi_rank=16,
        max_seq_bytes=512,
    ),
    "native_byte_mid": NativeBackboneConfig(
        d_model=384,
        byte_emb_dim=64,
        patch_size=8,
        n_groups=4,
        group_units=768,
        fast_key_dim=32,
        fast_val_dim=32,
        fast_radius=4.0,
        psi_rank=24,
        max_seq_bytes=512,
    ),
}


@dataclass
class NativeSubjectState:
    """Persistent cross-window / cross-session state (H, z, a, F, residual)."""

    h_states: list[torch.Tensor]
    unit_z: list[torch.Tensor]
    unit_a: list[torch.Tensor]
    fast_memories: list[torch.Tensor]
    residual_h: list[torch.Tensor]
    residual_z: list[torch.Tensor]
    residual_a: list[torch.Tensor]

    def detach(self) -> NativeSubjectState:
        return NativeSubjectState(
            h_states=[t.detach() for t in self.h_states],
            unit_z=[t.detach() for t in self.unit_z],
            unit_a=[t.detach() for t in self.unit_a],
            fast_memories=[t.detach() for t in self.fast_memories],
            residual_h=[t.detach() for t in self.residual_h],
            residual_z=[t.detach() for t in self.residual_z],
            residual_a=[t.detach() for t in self.residual_a],
        )

    def to_dict(self) -> dict[str, list[torch.Tensor]]:
        """Serialize state to plain CPU tensor lists safe for `torch.load(..., weights_only=True)`."""
        return {
            "h_states": [t.detach().cpu().clone() for t in self.h_states],
            "unit_z": [t.detach().cpu().clone() for t in self.unit_z],
            "unit_a": [t.detach().cpu().clone() for t in self.unit_a],
            "fast_memories": [t.detach().cpu().clone() for t in self.fast_memories],
            "residual_h": [t.detach().cpu().clone() for t in self.residual_h],
            "residual_z": [t.detach().cpu().clone() for t in self.residual_z],
            "residual_a": [t.detach().cpu().clone() for t in self.residual_a],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, list[torch.Tensor]],
        device: torch.device | None = None,
    ) -> NativeSubjectState:
        def _move(xs: list[torch.Tensor]) -> list[torch.Tensor]:
            return [x.to(device=device).clone() if device is not None else x.clone() for x in xs]

        return cls(
            h_states=_move(list(data["h_states"])),
            unit_z=_move(list(data["unit_z"])),
            unit_a=_move(list(data["unit_a"])),
            fast_memories=_move(list(data["fast_memories"])),
            residual_h=_move(list(data.get("residual_h", []))),
            residual_z=_move(list(data.get("residual_z", []))),
            residual_a=_move(list(data.get("residual_a", []))),
        )


class BytePatchEncoder(nn.Module):
    """§5.1 Pure UTF-8 byte embedding + 3-layer causal Conv1d (d=1,2,4) + 8-byte patch."""

    def __init__(self, d_model: int, byte_emb_dim: int = 64, patch_size: int = 8) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.byte_emb_dim = int(byte_emb_dim)
        self.patch_size = int(patch_size)

        self.byte_emb = nn.Embedding(BYTE_VOCAB, self.byte_emb_dim)
        nn.init.normal_(self.byte_emb.weight, std=0.02)

        # Lossless dynamic macro-byte vocabulary hook (zero residual delta at birth)
        self.dynamic_vocab = DynamicByteVocab()
        self.macro_delta = nn.Parameter(torch.zeros(0, self.byte_emb_dim))

        self.conv1 = nn.Conv1d(self.byte_emb_dim, self.d_model, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, dilation=2)
        self.conv3 = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, dilation=4)

    def _embed_symbols(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Fast path for pure base bytes (0..258); macro path preserves birth equivalence."""
        if self.dynamic_vocab.vocab_size == BYTE_VOCAB or bool((byte_ids < BYTE_VOCAB).all()):
            return cast(torch.Tensor, self.byte_emb(byte_ids))
        base_ids = byte_ids.clamp(max=BYTE_VOCAB - 1)
        emb: torch.Tensor = cast(torch.Tensor, self.byte_emb(base_ids)).clone()
        macro_mask = byte_ids >= BYTE_VOCAB
        positions = torch.nonzero(macro_mask, as_tuple=False)
        for pos in positions:
            b_idx, t_idx = int(pos[0].item()), int(pos[1].item())
            sid = int(byte_ids[b_idx, t_idx].item())
            raw_b = self.dynamic_vocab.decode_to_bytes([sid])
            raw_t = torch.tensor(list(raw_b), dtype=torch.long, device=byte_ids.device)
            pooled = self.byte_emb(raw_t).mean(dim=0) + self.macro_delta[sid - BYTE_VOCAB]
            emb[b_idx, t_idx] = pooled.to(dtype=emb.dtype)
        return emb

    @torch.no_grad()
    def grow_macro_symbol_neutral_(
        self, raw_bytes: bytes, *, birth_step: int = 0, lineage_tag: str = "chunk_growth"
    ) -> dict[str, Any]:
        """Register a multi-byte macro-symbol with zero residual delta (`0.0` birth shift)."""
        sym_id = self.dynamic_vocab.register_macro(
            raw_bytes, birth_step=birth_step, lineage_tag=lineage_tag
        )
        old = self.macro_delta.detach()
        new_row = torch.zeros(1, self.byte_emb_dim, device=old.device, dtype=old.dtype)
        self.macro_delta = nn.Parameter(torch.cat([old, new_row], dim=0))
        return {
            "op": "grow_macro_symbol_neutral",
            "symbol_id": sym_id,
            "n_bytes": len(raw_bytes),
            "vocab_size": self.dynamic_vocab.vocab_size,
        }

    def forward(self, byte_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return strictly causal per-byte features (B, T, D) and patch embeddings (B, P, D)."""
        x = self._embed_symbols(byte_ids).transpose(1, 2)  # (B, E, T)
        h1 = torch.tanh(self.conv1(F.pad(x, (2, 0))))
        h2 = h1 + torch.tanh(self.conv2(F.pad(h1, (4, 0))))
        h3 = h2 + torch.tanh(self.conv3(F.pad(h2, (8, 0))))
        a_byte = h3.transpose(1, 2)  # (B, T, D)

        bsz, seq_len, d_dim = a_byte.shape
        rem = seq_len % self.patch_size
        if rem != 0:
            pad_len = self.patch_size - rem
            padded = F.pad(a_byte, (0, 0, 0, pad_len))
        else:
            padded = a_byte
        n_patches = padded.shape[1] // self.patch_size
        e_patch = padded.view(bsz, n_patches, self.patch_size, d_dim).mean(dim=2)
        return a_byte, e_patch


class StatefulNeuralGroup(nn.Module):
    """Heterogeneous neural group with stateful units (z, a, W_rec) & FP1 KKT fast memory."""

    def __init__(
        self,
        group_idx: int,
        d_model: int,
        n_units: int,
        fast_key_dim: int,
        fast_val_dim: int,
        fast_radius: float,
        psi_rank: int,
        tau_init: float,
        tau_min: float = 0.5,
        tau_max: float = 128.0,
        use_fast_memory: bool = True,
        *,
        is_residual: bool = False,
    ) -> None:
        super().__init__()
        self.group_idx = int(group_idx)
        self.d_model = int(d_model)
        self.n_units = int(n_units)
        self.fast_key_dim = int(fast_key_dim)
        self.fast_val_dim = int(fast_val_dim)
        self.fast_radius = float(fast_radius)
        self.psi_rank = int(psi_rank)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.use_fast_memory = bool(use_fast_memory)
        self.is_residual = bool(is_residual)

        frac = max(
            1e-4,
            min(1.0 - 1e-4, (tau_init - self.tau_min) / max(1e-6, self.tau_max - self.tau_min)),
        )
        self.rho = nn.Parameter(torch.tensor(math.log(frac / (1.0 - frac)), dtype=torch.float32))

        in_dim = 4 * self.d_model  # [N(H_g), e_patch, m_g, r_F]
        self.w_slow = nn.Linear(in_dim, self.d_model, bias=False)
        nn.init.normal_(self.w_slow.weight, std=0.02)

        self.psi_u = nn.Parameter(torch.zeros(self.d_model, self.psi_rank))
        self.psi_v = nn.Parameter(torch.randn(in_dim, self.psi_rank) * 0.02)

        # Stateful neural units (U2_NEURAL_UNIT §2-3)
        self.unit_r = nn.Parameter(torch.randn(self.n_units, in_dim) * (0.5 / math.sqrt(in_dim)))
        self.unit_p = nn.Parameter(
            torch.randn(self.d_model, self.n_units) * (0.5 / math.sqrt(self.n_units))
        )
        self.unit_w_rec = nn.Parameter(torch.zeros(self.n_units, self.n_units))
        self.unit_c = nn.Parameter(torch.full((self.n_units,), 0.5))
        self.unit_eta = nn.Parameter(torch.full((self.n_units,), 0.1))
        self.unit_b = nn.Parameter(torch.zeros(self.n_units))

        self.register_buffer(
            "unit_stage", torch.full((self.n_units,), STAGE_ID["immature"], dtype=torch.long)
        )
        self.unit_stage: torch.Tensor
        edge_mask = torch.ones(self.n_units, self.n_units, dtype=torch.float32)
        edge_mask.fill_diagonal_(0.0)
        self.register_buffer("edge_mask", edge_mask)
        self.edge_mask: torch.Tensor

        self.unit_ids: list[str] = [f"g{self.group_idx}/u{i}" for i in range(self.n_units)]
        self.unit_branches: list[str] = ["base"] * self.n_units
        self.retired_unit_ids: list[str] = []

        # Inter-group Q/K/V projections
        self.msg_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.msg_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.msg_v = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.normal_(self.msg_q.weight, std=0.02)
        nn.init.normal_(self.msg_k.weight, std=0.02)
        nn.init.normal_(self.msg_v.weight, std=0.02)

        # FP1 KKT Fast Memory projections
        self.fast_q = nn.Linear(self.d_model, self.fast_key_dim, bias=False)
        self.fast_k = nn.Linear(self.d_model, self.fast_key_dim, bias=False)
        self.fast_v = nn.Linear(self.d_model, self.fast_val_dim, bias=False)
        self.fast_read_out = nn.Linear(self.fast_val_dim, self.d_model, bias=False)
        self.fast_alpha = nn.Linear(self.d_model, self.fast_key_dim, bias=True)
        self.fast_gain = nn.Linear(self.d_model, 1, bias=True)
        self.fast_chi = nn.Linear(self.d_model, 1, bias=True)
        nn.init.normal_(self.fast_read_out.weight, std=0.02)
        nn.init.constant_(self.fast_alpha.bias, 2.0)
        nn.init.constant_(self.fast_chi.bias, -1.0)

        if self.is_residual:
            # CORE_ALG §11.3: zero incoming mapping for residual group at birth
            self.w_slow.weight.data.zero_()
            self.psi_v.data.zero_()
            self.unit_r.data.zero_()

    def alive_mask(self, dtype: torch.dtype) -> torch.Tensor:
        alive = (self.unit_stage == STAGE_ID["immature"]) | (self.unit_stage == STAGE_ID["mature"])
        return alive.to(dtype=dtype)

    def read_fast(self, h_prev: torch.Tensor, f_prev: torch.Tensor) -> torch.Tensor:
        if not self.use_fast_memory:
            return torch.zeros_like(h_prev)
        q = smooth_unit_ball(self.fast_q(h_prev))
        v_read = torch.einsum("bkv,bk->bv", f_prev.to(dtype=q.dtype), q)
        return cast(torch.Tensor, self.fast_read_out(v_read))

    def step_tick(
        self,
        x_g: torch.Tensor,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact synchronous U2_NEURAL_UNIT §3 tick."""
        alive = self.alive_mask(x_g.dtype)
        c = self.unit_c.clamp(0.0, 1.0 - 1e-4).to(dtype=x_g.dtype)
        eta = self.unit_eta.clamp(1e-4, 1.0).to(dtype=x_g.dtype)
        b = self.unit_b.clamp(0.0, 2.0).to(dtype=x_g.dtype)

        z_alive = z_prev * alive
        w_eff = (self.unit_w_rec * self.edge_mask).to(dtype=x_g.dtype)
        v = F.linear(x_g, self.unit_r.to(dtype=x_g.dtype)) + F.linear(z_alive, w_eff) - b * a_prev
        z_next = (c * z_prev + (1.0 - c) * torch.tanh(v)) * alive
        a_next = ((1.0 - eta) * a_prev + eta * z_next) * alive

        psi_out = (x_g @ self.psi_v.to(dtype=x_g.dtype)) @ self.psi_u.to(dtype=x_g.dtype).t()
        unit_out = F.linear(z_next * alive, self.unit_p.to(dtype=x_g.dtype))
        pre_act = self.w_slow(x_g) + psi_out + unit_out

        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.rho)
        ell = (1.0 - torch.exp(-1.0 / tau)).to(dtype=x_g.dtype)
        h_next = (1.0 - ell) * h_prev + ell * torch.tanh(pre_act)
        return h_next, z_next, a_next

    def write_fast_fp1(
        self,
        f_prev: torch.Tensor,
        h_seq: torch.Tensor,
        p_allowed: float,
    ) -> torch.Tensor:
        """Update fast memory via exact KKT constrained quadratic solver `u2_operators.fp1`."""
        if not self.use_fast_memory or p_allowed <= 0.0:
            return f_prev
        bsz = h_seq.shape[0]
        with torch.amp.autocast("cpu", enabled=False):
            k_seq = smooth_unit_ball(self.fast_k(h_seq.float())).float()  # (B, P, dk)
            v_seq = smooth_unit_ball(torch.tanh(self.fast_v(h_seq.float()))).float()  # (B, P, dv)
            alpha = torch.sigmoid(self.fast_alpha(h_seq[:, -1, :].float())).float()  # (B, dk)
            gains = (4.0 * torch.sigmoid(self.fast_gain(h_seq.float()))).squeeze(-1).float()
            chi = (p_allowed * torch.sigmoid(self.fast_chi(h_seq[:, -1, :].float())).mean()).float()
            radius = torch.tensor(self.fast_radius, dtype=torch.float32, device=f_prev.device)

            out_list: list[torch.Tensor] = []
            for b_idx in range(bsz):
                f_b = f_prev[b_idx].float()
                norm_b = float(f_b.detach().norm())
                if norm_b > self.fast_radius - 1e-5:
                    f_b = f_b * ((self.fast_radius - 1e-5) / max(1e-12, norm_b))
                f_new = fp1(
                    f_b,
                    k_seq[b_idx].transpose(0, 1),
                    v_seq[b_idx].transpose(0, 1),
                    alpha[b_idx],
                    gains[b_idx],
                    chi,
                    radius,
                    allowed=True,
                )
                out_list.append(f_new.to(dtype=f_prev.dtype))
        return torch.stack(out_list, dim=0)


class NativeExperienceCopyHead(nn.Module):
    """U2 §7 Position-sensitive raw-experience copy head powered by `taiji.u2_operators`."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.rec_emb = nn.Embedding(BYTE_VOCAB, self.d_model)
        nn.init.normal_(self.rec_emb.weight, std=0.02)
        self.route_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.pos_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.null_route = nn.Parameter(torch.zeros(1))
        self.lam_head = nn.Linear(self.d_model, 1, bias=True)
        self.gamma_head = nn.Linear(self.d_model, 1, bias=True)
        nn.init.normal_(self.route_proj.weight, std=0.02)
        nn.init.normal_(self.pos_proj.weight, std=0.02)
        nn.init.constant_(self.lam_head.bias, 1.0)
        nn.init.constant_(self.gamma_head.bias, 0.0)

    def init_cursor(
        self, n_records: int, max_len: int, device: torch.device | None = None
    ) -> CursorState:
        """Create a normalized log-domain cursor initialized in the reset (`zeta=1`) state."""
        return {
            "log_m": torch.full(
                (1, n_records, max_len), -torch.inf, dtype=torch.float32, device=device
            ),
            "log_zeta": torch.zeros((1,), dtype=torch.float32, device=device),
            "log_eta": torch.zeros((1,), dtype=torch.float32, device=device),
            "coverage": torch.zeros((1, n_records, max_len), dtype=torch.float32, device=device),
        }

    def step_emission(
        self,
        ctx_vec: torch.Tensor,
        gen_logits: torch.Tensor,
        cursor: CursorState,
        record_bytes: torch.Tensor,
        record_valid: torch.Tensor,
        utf8_legal_mask: torch.Tensor,
    ) -> Emission:
        """Compute U2 log-domain joint generate/copy emission masked to legal UTF-8 bytes."""
        with torch.amp.autocast("cpu", enabled=False):
            ctx = ctx_vec.float()
            rec_emb = self.rec_emb(record_bytes.clamp(0, BYTE_VOCAB - 1)).float()  # (R, L, D)
            valid_f = record_valid.float().unsqueeze(-1)
            rec_mean = (rec_emb * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)

            scale = 1.0 / math.sqrt(self.d_model)
            rec_logits = torch.einsum("d,rd->r", self.route_proj(ctx), rec_mean)
            rec_logits = rec_logits * scale
            route_scores = torch.cat([rec_logits, self.null_route.float()], dim=0).unsqueeze(0)

            pos_scores = torch.einsum("d,rld->rl", self.pos_proj(ctx), rec_emb) * scale
            pos_scores = pos_scores.unsqueeze(0)

            log_alpha, log_beta = u2_record_inputs(route_scores, pos_scores, record_valid)
            lam = torch.sigmoid(self.lam_head(ctx)).reshape(1)
            gamma = torch.sigmoid(self.gamma_head(ctx)).reshape(1)
            log_gen = F.log_softmax(gen_logits.float(), dim=-1).unsqueeze(0)
            return u2_emission(
                cursor,
                log_alpha,
                log_beta,
                lam,
                gamma,
                log_gen,
                record_bytes,
                record_valid,
                utf8_legal_mask,
            )

    def step_accept(
        self,
        cursor: CursorState,
        em: Emission,
        symbol: int,
        record_bytes: torch.Tensor,
        record_valid: torch.Tensor,
    ) -> CursorState:
        return u2_accept(cursor, em, symbol, record_bytes, record_valid)

    def revoke(
        self, cursor: CursorState, indices: Sequence[int]
    ) -> tuple[CursorState, torch.Tensor]:
        return cast(tuple[CursorState, torch.Tensor], u2_revoke_records(cursor, indices))


class TaijiNativeCognitiveBackbone(nn.Module):
    """态极 Native Byte-Group Cognitive Backbone (Zero Transformer, Zero BPE)."""

    def __init__(self, cfg: NativeBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.growth_lineage: list[dict[str, Any]] = []

        self.encoder = BytePatchEncoder(cfg.d_model, cfg.byte_emb_dim, cfg.patch_size)
        tau_inits = [1.0, 4.0, 16.0, 64.0]
        self.groups = nn.ModuleList(
            [
                StatefulNeuralGroup(
                    group_idx=i,
                    d_model=cfg.d_model,
                    n_units=cfg.group_unit_count(i),
                    fast_key_dim=cfg.fast_key_dim,
                    fast_val_dim=cfg.fast_val_dim,
                    fast_radius=cfg.fast_radius,
                    psi_rank=cfg.psi_rank,
                    tau_init=tau_inits[i % len(tau_inits)],
                    tau_min=cfg.tau_min,
                    tau_max=cfg.tau_max,
                    use_fast_memory=cfg.use_fast_memory,
                )
                for i in range(cfg.n_groups)
            ]
        )
        self.residual_groups = nn.ModuleList()

        # Shared group readout + strictly causal byte output head (257 = 0..255 + EOS)
        self.group_read_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.byte_out = nn.Linear(2 * cfg.d_model, OUT_VOCAB, bias=True)
        nn.init.normal_(self.group_read_proj.weight, std=0.02)
        nn.init.normal_(self.byte_out.weight, std=0.02)
        nn.init.zeros_(self.byte_out.bias)

        # U2 §7 position-sensitive raw-experience copy head (used on experience-grounded rollout)
        self.copy_head = NativeExperienceCopyHead(cfg.d_model)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> NativeSubjectState:
        dev = device or self.encoder.byte_emb.weight.device
        dt = dtype or self.encoder.byte_emb.weight.dtype
        return NativeSubjectState(
            h_states=[
                torch.zeros(batch_size, self.cfg.d_model, device=dev, dtype=dt) for _ in self.groups
            ],
            unit_z=[
                torch.zeros(batch_size, cast(StatefulNeuralGroup, g).n_units, device=dev, dtype=dt)
                for g in self.groups
            ],
            unit_a=[
                torch.zeros(batch_size, cast(StatefulNeuralGroup, g).n_units, device=dev, dtype=dt)
                for g in self.groups
            ],
            fast_memories=[
                torch.zeros(
                    batch_size,
                    self.cfg.fast_key_dim,
                    self.cfg.fast_val_dim,
                    device=dev,
                    dtype=torch.float32,
                )
                for _ in self.groups
            ],
            residual_h=[
                torch.zeros(batch_size, self.cfg.d_model, device=dev, dtype=dt)
                for _ in self.residual_groups
            ],
            residual_z=[
                torch.zeros(batch_size, cast(StatefulNeuralGroup, g).n_units, device=dev, dtype=dt)
                for g in self.residual_groups
            ],
            residual_a=[
                torch.zeros(batch_size, cast(StatefulNeuralGroup, g).n_units, device=dev, dtype=dt)
                for g in self.residual_groups
            ],
        )

    def forward(
        self,
        byte_ids: torch.Tensor,
        state: NativeSubjectState | None = None,
        labels: torch.Tensor | None = None,
        p_allowed: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, NativeSubjectState]:
        bsz, seq_len = byte_ids.shape
        a_byte, e_patch = self.encoder(byte_ids)
        dt = a_byte.dtype
        if state is None:
            state = self.init_state(bsz, device=byte_ids.device, dtype=dt)

        n_patches = e_patch.shape[1]
        h_cur = [h.to(dtype=dt) for h in state.h_states]
        z_cur = [z.to(dtype=dt) for z in state.unit_z]
        a_cur = [a.to(dtype=dt) for a in state.unit_a]
        f_cur = list(state.fast_memories)
        rh_cur = [h.to(dtype=dt) for h in state.residual_h]
        rz_cur = [z.to(dtype=dt) for z in state.residual_z]
        ra_cur = [a.to(dtype=dt) for a in state.residual_a]

        patch_readouts: list[torch.Tensor] = []
        h_traces: list[list[torch.Tensor]] = [[] for _ in self.groups]
        scale = 1.0 / math.sqrt(self.cfg.d_model)

        for p_idx in range(n_patches):
            e_t = e_patch[:, p_idx, :]
            # Synchronous inter-group Q/K/V messaging from pre-step H
            qs = [cast(StatefulNeuralGroup, g).msg_q(h_cur[i]) for i, g in enumerate(self.groups)]
            ks = torch.stack(
                [cast(StatefulNeuralGroup, g).msg_k(h_cur[i]) for i, g in enumerate(self.groups)],
                dim=1,
            )  # (B, G, D)
            vs = torch.stack(
                [cast(StatefulNeuralGroup, g).msg_v(h_cur[i]) for i, g in enumerate(self.groups)],
                dim=1,
            )  # (B, G, D)

            next_h: list[torch.Tensor] = []
            next_z: list[torch.Tensor] = []
            next_a: list[torch.Tensor] = []
            for i, grp_m in enumerate(self.groups):
                grp = cast(StatefulNeuralGroup, grp_m)
                attn = torch.softmax(torch.einsum("bd,bgd->bg", qs[i], ks) * scale, dim=-1)
                m_g = torch.einsum("bg,bgd->bd", attn, vs)
                r_f = grp.read_fast(h_cur[i], f_cur[i])
                x_g = torch.cat([scale_floor_norm(h_cur[i]), e_t, m_g, r_f], dim=-1)
                h_n, z_n, a_n = grp.step_tick(x_g, h_cur[i], z_cur[i], a_cur[i])
                next_h.append(h_n)
                next_z.append(z_n)
                next_a.append(a_n)
                h_traces[i].append(h_n)

            h_cur, z_cur, a_cur = next_h, next_z, next_a

            # Mean group readout + additive residual groups
            z_p = self.group_read_proj(torch.stack(h_cur, dim=0).mean(dim=0))
            for r_i, r_grp_m in enumerate(self.residual_groups):
                r_grp = cast(StatefulNeuralGroup, r_grp_m)
                zero_msg = torch.zeros_like(z_p)
                rx_g = torch.cat([scale_floor_norm(rh_cur[r_i]), e_t, zero_msg, zero_msg], dim=-1)
                rh_n, rz_n, ra_n = r_grp.step_tick(rx_g, rh_cur[r_i], rz_cur[r_i], ra_cur[r_i])
                rh_cur[r_i], rz_cur[r_i], ra_cur[r_i] = rh_n, rz_n, ra_n
                z_p = z_p + rh_n
            patch_readouts.append(z_p)

        # Update FP1 KKT fast memories after patch window reads
        next_f: list[torch.Tensor] = []
        for i, grp_m in enumerate(self.groups):
            grp = cast(StatefulNeuralGroup, grp_m)
            h_seq = torch.stack(h_traces[i], dim=1)
            next_f.append(grp.write_fast_fp1(f_cur[i], h_seq, p_allowed=p_allowed))

        # Strictly causal byte readout: byte t in patch p uses z_{p-1} (0 for p=0) + a_byte[:, t]
        z_stack = torch.stack(patch_readouts, dim=1)  # (B, P, D)
        z_shifted = F.pad(z_stack[:, :-1, :], (0, 0, 1, 0))  # (B, P, D), strictly causal!
        z_per_byte = z_shifted.repeat_interleave(self.cfg.patch_size, dim=1)[:, :seq_len, :]
        logits = self.byte_out(torch.cat([a_byte, z_per_byte], dim=-1))

        loss: torch.Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, OUT_VOCAB), labels.reshape(-1), ignore_index=PAD_BYTE
            )

        next_state = NativeSubjectState(
            h_states=h_cur,
            unit_z=z_cur,
            unit_a=a_cur,
            fast_memories=next_f,
            residual_h=rh_cur,
            residual_z=rz_cur,
            residual_a=ra_cur,
        )
        return logits, loss, next_state

    @torch.no_grad()
    def generate_utf8(
        self,
        prompt: str,
        max_new_bytes: int = 64,
        state: NativeSubjectState | None = None,
        temperature: float = 0.8,
    ) -> tuple[str, NativeSubjectState]:
        """Autoregressively generate strictly valid UTF-8 bytes using `u2_operators.utf8_mask`."""
        self.eval()
        dev = self.encoder.byte_emb.weight.device
        raw = list(prompt.encode("utf-8"))
        if not raw:
            raw = [BOS_BYTE]
        x = torch.tensor([raw[-self.cfg.max_seq_bytes :]], dtype=torch.long, device=dev)
        logits, _, cur_state = self(x, state=state, p_allowed=0.0)

        generated: list[int] = []
        remaining, lead = 0, 0
        for step in range(max_new_bytes):
            budget = max_new_bytes - step
            mask = utf8_mask(remaining, lead, budget, device=dev)
            next_logits = logits[0, -1, :].float() / max(1e-4, temperature)
            next_logits = next_logits.masked_fill(~mask, -1e9)
            probs = torch.softmax(next_logits, dim=-1)
            sym = int(torch.multinomial(probs, num_samples=1).item())
            if sym == EOS_BYTE:
                break
            generated.append(sym)
            remaining, lead = advance_utf8(remaining, lead, sym)
            x = torch.cat([x, torch.tensor([[sym]], dtype=torch.long, device=dev)], dim=1)
            x_cond = x[:, -self.cfg.max_seq_bytes :]
            logits, _, cur_state = self(x_cond, state=state, p_allowed=0.0)

        text = bytes(generated).decode("utf-8")
        return text, cur_state

    @torch.no_grad()
    def generate_utf8_with_records(
        self,
        prompt: str,
        records: Sequence[bytes | str],
        max_new_bytes: int = 64,
        state: NativeSubjectState | None = None,
        temperature: float = 0.8,
        revoked_indices: Sequence[int] = (),
    ) -> tuple[str, NativeSubjectState, dict[str, Any]]:
        """Autoregressively generate valid UTF-8 with U2 §7 position-sensitive copy head."""
        self.eval()
        dev = self.encoder.byte_emb.weight.device
        raw = list(prompt.encode("utf-8"))
        if not raw:
            raw = [BOS_BYTE]

        rec_bytes_list: list[list[int]] = []
        for r in records:
            rb = list(r.encode("utf-8")) if isinstance(r, str) else list(r)
            rec_bytes_list.append(rb)

        n_records = len(rec_bytes_list)
        max_rec_len = max((len(rb) for rb in rec_bytes_list), default=1)
        record_bytes = torch.zeros((n_records, max_rec_len), dtype=torch.long, device=dev)
        record_valid = torch.zeros((n_records, max_rec_len), dtype=torch.bool, device=dev)
        for r_idx, rb in enumerate(rec_bytes_list):
            if rb:
                record_bytes[r_idx, : len(rb)] = torch.tensor(rb, dtype=torch.long, device=dev)
                record_valid[r_idx, : len(rb)] = True

        cursor = self.copy_head.init_cursor(n_records, max_rec_len, device=dev)
        archived_cov: torch.Tensor | None = None
        if revoked_indices:
            cursor, archived_cov = self.copy_head.revoke(cursor, revoked_indices)
            for idx in revoked_indices:
                r_i = int(idx)
                if 0 <= r_i < n_records:
                    record_valid[r_i] = False

        x = torch.tensor([raw[-self.cfg.max_seq_bytes :]], dtype=torch.long, device=dev)
        logits, _, cur_state = self(x, state=state, p_allowed=0.0)

        generated: list[int] = []
        remaining, lead = 0, 0
        for step in range(max_new_bytes):
            budget = max_new_bytes - step
            mask = utf8_mask(remaining, lead, budget, device=dev)
            ctx_vec = torch.stack(cur_state.h_states, dim=0).mean(dim=0)[0]
            gen_logits = logits[0, -1, :].float() / max(1e-4, temperature)
            em = self.copy_head.step_emission(
                ctx_vec,
                gen_logits,
                cursor,
                record_bytes,
                record_valid,
                mask,
            )
            probs = em["log_output"].exp()
            sym = int(torch.multinomial(probs, num_samples=1).item())
            if sym == EOS_BYTE:
                break
            cursor = self.copy_head.step_accept(cursor, em, sym, record_bytes, record_valid)
            generated.append(sym)
            remaining, lead = advance_utf8(remaining, lead, sym)
            x = torch.cat([x, torch.tensor([[sym]], dtype=torch.long, device=dev)], dim=1)
            x_cond = x[:, -self.cfg.max_seq_bytes :]
            logits, _, cur_state = self(x_cond, state=state, p_allowed=0.0)

        text = bytes(generated).decode("utf-8")
        meta = {
            "n_records": n_records,
            "revoked_indices": [int(i) for i in revoked_indices],
            "coverage_sum": float(cursor["coverage"].sum().item()),
            "archived_coverage_sum": (
                float(archived_cov.sum().item()) if archived_cov is not None else 0.0
            ),
        }
        return text, cur_state, meta

    def _refresh_cfg(self) -> None:
        units = tuple(cast(StatefulNeuralGroup, g).n_units for g in self.groups)
        self.cfg = NativeBackboneConfig(
            d_model=self.cfg.d_model,
            byte_emb_dim=self.cfg.byte_emb_dim,
            patch_size=self.cfg.patch_size,
            n_groups=len(self.groups),
            group_units=units,
            fast_key_dim=self.cfg.fast_key_dim,
            fast_val_dim=self.cfg.fast_val_dim,
            fast_radius=self.cfg.fast_radius,
            psi_rank=self.cfg.psi_rank,
            tau_min=self.cfg.tau_min,
            tau_max=self.cfg.tau_max,
            use_fast_memory=self.cfg.use_fast_memory,
            max_seq_bytes=self.cfg.max_seq_bytes,
        )

    @torch.no_grad()
    def grow_stateful_units_svd_neutral_(
        self,
        group_idx: int = 0,
        budget: int = 16,
        *,
        deltas: torch.Tensor | None = None,
        inputs: torch.Tensor | None = None,
        birth_id: str | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """CORE_ALG §11.2 + U2_NEURAL_UNIT §5: zero incoming r=0, non-zero outgoing p."""
        grp = cast(StatefulNeuralGroup, self.groups[group_idx])
        dev = grp.unit_r.device
        dt = grp.unit_r.dtype
        before_params = self.count_parameters()
        bid = birth_id or f"birth_{len(self.growth_lineage)}"
        while any(
            f"{bid}/{k}" in grp.retired_unit_ids or f"{bid}/{k}" in grp.unit_ids
            for k in range(max(1, budget))
        ):
            bid = f"{bid}_n"

        if deltas is not None and inputs is not None:
            gates = torch.ones((deltas.shape[0], 1), device=deltas.device, dtype=deltas.dtype)
            c_mat = credit_matrix(deltas, inputs, gates)[0]
            chosen = choose_left_directions(c_mat, budget)
            directions = chosen["directions"].to(device=dev, dtype=dt)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                raw_dir = torch.randn(grp.d_model, budget, device=dev, dtype=dt)
                directions, _ = torch.linalg.qr(raw_dir.float())
                directions = directions[:, :budget].to(dtype=dt)

        count = int(directions.shape[1])
        if count == 0:
            return {"op": "grow_stateful_units_svd_neutral", "count": 0, "skipped": True}

        res = append_neutral_units(
            grp.unit_r.detach(),
            grp.unit_p.detach(),
            directions,
            grp.unit_ids,
            grp.unit_branches,
            birth_id=bid,
        )
        old_n = grp.n_units
        new_n = int(res["units"])

        grp.unit_r = nn.Parameter(res["r"])
        grp.unit_p = nn.Parameter(res["p"])
        grp.unit_ids = list(res["unit_ids"])
        grp.unit_branches = list(res["branches"])

        new_w = torch.zeros(new_n, new_n, device=dev, dtype=dt)
        new_w[:old_n, :old_n].copy_(grp.unit_w_rec.detach())
        grp.unit_w_rec = nn.Parameter(new_w)

        new_mask = torch.ones(new_n, new_n, device=dev, dtype=torch.float32)
        new_mask.fill_diagonal_(0.0)
        new_mask[:old_n, :old_n].copy_(grp.edge_mask)
        grp.edge_mask = new_mask

        new_c = torch.full((new_n,), 0.5, device=dev, dtype=dt)
        new_c[:old_n].copy_(grp.unit_c.detach())
        grp.unit_c = nn.Parameter(new_c)

        new_eta = torch.full((new_n,), 0.1, device=dev, dtype=dt)
        new_eta[:old_n].copy_(grp.unit_eta.detach())
        grp.unit_eta = nn.Parameter(new_eta)

        new_b = torch.zeros(new_n, device=dev, dtype=dt)
        new_b[:old_n].copy_(grp.unit_b.detach())
        grp.unit_b = nn.Parameter(new_b)

        new_stage = torch.full((new_n,), STAGE_ID["immature"], device=dev, dtype=torch.long)
        new_stage[:old_n].copy_(grp.unit_stage)
        grp.unit_stage = new_stage
        grp.n_units = new_n

        self._refresh_cfg()
        event = {
            "op": "grow_stateful_units_svd_neutral",
            "group_idx": group_idx,
            "count": count,
            "before_params": before_params,
            "after_params": self.count_parameters(),
        }
        self.growth_lineage.append(event)
        return event

    @torch.no_grad()
    def proliferate_units_neutral_(
        self,
        group_idx: int = 0,
        parent_indices: list[int] | None = None,
        birth_id: str | None = None,
    ) -> dict[str, Any]:
        """U2_NEURAL_UNIT §6: exact inheritance proliferation preserving W_rec & p."""
        grp = cast(StatefulNeuralGroup, self.groups[group_idx])
        dev = grp.unit_r.device
        dt = grp.unit_r.dtype
        before_params = self.count_parameters()
        parents = list(parent_indices or [0])
        old_n = grp.n_units

        # Build mapping pi: new index -> parent index, and counts m_u
        pi = list(range(old_n)) + parents
        counts: dict[int, int] = {}
        for p_idx in pi:
            counts[p_idx] = counts.get(p_idx, 0) + 1
        new_n = len(pi)

        r_new = grp.unit_r.detach()[pi].clone()
        c_new = grp.unit_c.detach()[pi].clone()
        eta_new = grp.unit_eta.detach()[pi].clone()
        b_new = grp.unit_b.detach()[pi].clone()

        p_old = grp.unit_p.detach()
        p_new = torch.zeros(grp.d_model, new_n, device=dev, dtype=dt)
        w_old = grp.unit_w_rec.detach()
        w_new = torch.zeros(new_n, new_n, device=dev, dtype=dt)

        for i_new, p_i in enumerate(pi):
            p_new[:, i_new] = p_old[:, p_i] / float(counts[p_i])
            for j_new, p_j in enumerate(pi):
                w_new[i_new, j_new] = w_old[p_i, p_j] / float(counts[p_j])

        bid = birth_id or f"prolif_{len(self.growth_lineage)}"
        while any(
            f"{bid}/{k}" in grp.retired_unit_ids or f"{bid}/{k}" in grp.unit_ids
            for k in range(len(parents))
        ):
            bid = f"{bid}_n"
        new_ids = [f"{bid}/{k}" for k in range(len(parents))]

        grp.unit_r = nn.Parameter(r_new)
        grp.unit_p = nn.Parameter(p_new)
        grp.unit_w_rec = nn.Parameter(w_new)
        grp.unit_c = nn.Parameter(c_new)
        grp.unit_eta = nn.Parameter(eta_new)
        grp.unit_b = nn.Parameter(b_new)
        grp.unit_ids = grp.unit_ids + new_ids
        grp.unit_branches = grp.unit_branches + ["proliferated"] * len(parents)

        new_mask = torch.ones(new_n, new_n, device=dev, dtype=torch.float32)
        new_mask.fill_diagonal_(0.0)
        grp.edge_mask = new_mask

        new_stage = torch.full((new_n,), STAGE_ID["immature"], device=dev, dtype=torch.long)
        new_stage[:old_n].copy_(grp.unit_stage)
        grp.unit_stage = new_stage
        grp.n_units = new_n

        self._refresh_cfg()
        event = {
            "op": "proliferate_units_neutral",
            "group_idx": group_idx,
            "parents": parents,
            "before_params": before_params,
            "after_params": self.count_parameters(),
        }
        self.growth_lineage.append(event)
        return event

    @torch.no_grad()
    def grow_group_neutral_(self, tau_init: float = 32.0, seed: int = 0) -> dict[str, Any]:
        """CORE_ALG §11.3: append an additive residual group with zero incoming mapping."""
        before_params = self.count_parameters()
        dev = self.encoder.byte_emb.weight.device
        dt = self.encoder.byte_emb.weight.dtype
        idx = len(self.groups) + len(self.residual_groups)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            res_grp = StatefulNeuralGroup(
                group_idx=idx,
                d_model=self.cfg.d_model,
                n_units=64,
                fast_key_dim=self.cfg.fast_key_dim,
                fast_val_dim=self.cfg.fast_val_dim,
                fast_radius=self.cfg.fast_radius,
                psi_rank=self.cfg.psi_rank,
                tau_init=tau_init,
                tau_min=self.cfg.tau_min,
                tau_max=self.cfg.tau_max,
                use_fast_memory=False,
                is_residual=True,
            ).to(device=dev, dtype=dt)
        self.residual_groups.append(res_grp)
        event = {
            "op": "grow_group_neutral",
            "residual_idx": len(self.residual_groups) - 1,
            "before_params": before_params,
            "after_params": self.count_parameters(),
        }
        self.growth_lineage.append(event)
        return event

    @torch.no_grad()
    def mark_all_mature_(self) -> int:
        promoted = 0
        for g_m in list(self.groups) + list(self.residual_groups):
            g = cast(StatefulNeuralGroup, g_m)
            mask = g.unit_stage == STAGE_ID["immature"]
            promoted += int(mask.sum().item())
            g.unit_stage[mask] = STAGE_ID["mature"]
        return promoted

    @torch.no_grad()
    def set_dormant_(self, group_idx: int, unit_indices: Sequence[int]) -> dict[str, Any]:
        """Reversibly transition `immature`/`mature` units to `dormant` without weight loss."""
        grp = cast(StatefulNeuralGroup, self.groups[group_idx])
        changed: list[tuple[int, int]] = []
        for idx in unit_indices:
            i = int(idx)
            if not (0 <= i < grp.n_units):
                raise IndexError(f"unit_idx {i} out of range for group {group_idx}")
            prev = int(grp.unit_stage[i].item())
            if prev in (STAGE_ID["immature"], STAGE_ID["mature"]):
                grp.unit_stage[i] = STAGE_ID["dormant"]
                changed.append((i, prev))
        return {
            "op": "set_dormant",
            "group_idx": group_idx,
            "count": len(changed),
            "previous_stages": changed,
        }

    @torch.no_grad()
    def wake_(
        self,
        group_idx: int,
        unit_indices: Sequence[int],
        *,
        target_stage: str = "mature",
    ) -> dict[str, Any]:
        """Wake `dormant` units back into `mature` or `immature` active pool."""
        if target_stage not in ("mature", "immature"):
            raise ValueError(f"target_stage must be 'mature' or 'immature', got {target_stage!r}")
        grp = cast(StatefulNeuralGroup, self.groups[group_idx])
        tgt_id = STAGE_ID[target_stage]
        woken: list[int] = []
        for idx in unit_indices:
            i = int(idx)
            if not (0 <= i < grp.n_units):
                raise IndexError(f"unit_idx {i} out of range for group {group_idx}")
            if int(grp.unit_stage[i].item()) == STAGE_ID["dormant"]:
                grp.unit_stage[i] = tgt_id
                woken.append(i)
        return {
            "op": "wake",
            "group_idx": group_idx,
            "target_stage": target_stage,
            "woken_indices": woken,
        }

    @torch.no_grad()
    def retire_(self, group_idx: int, unit_indices: Sequence[int]) -> dict[str, Any]:
        """Mark units `retired` and zero their outgoing/incoming recurrent edges cleanly."""
        grp = cast(StatefulNeuralGroup, self.groups[group_idx])
        retired: list[int] = []
        for idx in unit_indices:
            i = int(idx)
            if not (0 <= i < grp.n_units):
                raise IndexError(f"unit_idx {i} out of range for group {group_idx}")
            grp.unit_stage[i] = STAGE_ID["retired"]
            grp.unit_p.data[:, i].zero_()
            grp.unit_w_rec.data[:, i].zero_()
            grp.unit_w_rec.data[i, :].zero_()
            retired.append(i)
        return {"op": "retire", "group_idx": group_idx, "retired_indices": retired}

    @torch.no_grad()
    def recycle_retired_(self, group_idx: int) -> dict[str, Any]:
        """U2_NEURAL_UNIT §4: verify zero outgoing weights, then compact retired slots."""
        grp = cast(StatefulNeuralGroup, self.groups[group_idx])
        retired_mask = grp.unit_stage == STAGE_ID["retired"]
        n_ret = int(retired_mask.sum().item())
        if n_ret == 0:
            return {"op": "recycle_retired", "group_idx": group_idx, "recycled": 0}

        if float(grp.unit_p[:, retired_mask].abs().max().item()) > 1e-12:
            raise RuntimeError("Cannot recycle retired units with non-zero outgoing unit_p")
        if float(grp.unit_w_rec[:, retired_mask].abs().max().item()) > 1e-12:
            raise RuntimeError("Cannot recycle retired units with non-zero outgoing unit_w_rec")

        keep_mask = ~retired_mask
        keep_idx = [i for i in range(grp.n_units) if bool(keep_mask[i].item())]
        removed_ids = [grp.unit_ids[i] for i in range(grp.n_units) if not bool(keep_mask[i].item())]
        grp.retired_unit_ids.extend(removed_ids)

        before_params = self.count_parameters()
        grp.unit_r = nn.Parameter(grp.unit_r.detach()[keep_mask].clone())
        grp.unit_p = nn.Parameter(grp.unit_p.detach()[:, keep_mask].clone())
        grp.unit_w_rec = nn.Parameter(grp.unit_w_rec.detach()[keep_mask][:, keep_mask].clone())
        grp.edge_mask = grp.edge_mask[keep_mask][:, keep_mask].clone()
        grp.unit_c = nn.Parameter(grp.unit_c.detach()[keep_mask].clone())
        grp.unit_eta = nn.Parameter(grp.unit_eta.detach()[keep_mask].clone())
        grp.unit_b = nn.Parameter(grp.unit_b.detach()[keep_mask].clone())
        grp.unit_stage = grp.unit_stage[keep_mask].clone()
        grp.unit_ids = [grp.unit_ids[i] for i in keep_idx]
        grp.unit_branches = [grp.unit_branches[i] for i in keep_idx]
        grp.n_units = len(keep_idx)

        self._refresh_cfg()
        event = {
            "op": "recycle_retired",
            "group_idx": group_idx,
            "recycled": n_ret,
            "retired_unit_ids": removed_ids,
            "before_params": before_params,
            "after_params": self.count_parameters(),
        }
        self.growth_lineage.append(event)
        return event

    @torch.no_grad()
    def grow_macro_symbol_neutral_(
        self, raw_bytes: bytes, *, birth_step: int = 0, lineage_tag: str = "chunk_growth"
    ) -> dict[str, Any]:
        """Grow a lossless multi-byte macro-symbol with `0.0` birth perturbation."""
        before_params = self.count_parameters()
        ev = self.encoder.grow_macro_symbol_neutral_(
            raw_bytes, birth_step=birth_step, lineage_tag=lineage_tag
        )
        ev["before_params"] = before_params
        ev["after_params"] = self.count_parameters()
        self.growth_lineage.append(dict(ev))
        return ev

    def snapshot_param_handles(self) -> dict[str, nn.Parameter]:
        """Capture `{param_name: nn.Parameter}` handles before structural topology mutation."""
        return dict(self.named_parameters())

    @torch.no_grad()
    def migrate_adamw_state_(
        self,
        optimizer: torch.optim.Optimizer,
        old_handles: dict[str, nn.Parameter],
    ) -> dict[str, int]:
        """U2_NEURAL_UNIT §6: migrate AdamW moments across replaced/grown parameters."""
        new_handles = dict(self.named_parameters())
        old_to_new: dict[nn.Parameter, nn.Parameter] = {}
        migrated = 0
        added = 0

        for name, new_p in new_handles.items():
            old_p = old_handles.get(name)
            if old_p is None:
                added += 1
                if optimizer.param_groups:
                    optimizer.param_groups[0]["params"].append(new_p)
                continue
            if old_p is new_p:
                continue
            old_to_new[old_p] = new_p
            old_state = optimizer.state.pop(old_p, None)
            if isinstance(old_state, dict) and old_state:
                new_state: dict[str, Any] = {}
                for k, v in old_state.items():
                    if isinstance(v, torch.Tensor) and v.shape == old_p.shape:
                        dst = torch.zeros_like(new_p, dtype=v.dtype, device=new_p.device)
                        slices = tuple(
                            slice(0, min(int(os_), int(ns_)))
                            for os_, ns_ in zip(v.shape, new_p.shape, strict=True)
                        )
                        dst[slices].copy_(v[slices].to(device=new_p.device))
                        new_state[k] = dst
                    elif isinstance(v, torch.Tensor):
                        new_state[k] = v.clone()
                    else:
                        new_state[k] = v
                optimizer.state[new_p] = new_state
            migrated += 1

        if old_to_new:
            for pg in optimizer.param_groups:
                pg["params"] = [old_to_new.get(p, p) for p in pg["params"]]

        return {"migrated": migrated, "added": added}

    def apply_selective_plasticity_grad_mask(
        self, freeze_mature_slow: bool = True
    ) -> dict[str, int]:
        """Zero gradients on mature unit params (r, p, w_rec[mature], c, eta, b) & w_slow."""
        masked_units = 0
        masked_slow = 0
        for g_m in list(self.groups) + list(self.residual_groups):
            g = cast(StatefulNeuralGroup, g_m)
            mature = g.unit_stage == STAGE_ID["mature"]
            n_mat = int(mature.sum().item())
            if n_mat > 0:
                masked_units += n_mat
                if g.unit_r.grad is not None:
                    g.unit_r.grad[mature] = 0.0
                if g.unit_p.grad is not None:
                    g.unit_p.grad[:, mature] = 0.0
                if g.unit_w_rec.grad is not None:
                    g.unit_w_rec.grad[mature, :] = 0.0
                if g.unit_c.grad is not None:
                    g.unit_c.grad[mature] = 0.0
                if g.unit_eta.grad is not None:
                    g.unit_eta.grad[mature] = 0.0
                if g.unit_b.grad is not None:
                    g.unit_b.grad[mature] = 0.0
            if freeze_mature_slow and g.w_slow.weight.grad is not None:
                g.w_slow.weight.grad.zero_()
                masked_slow += 1
        return {"masked_units": masked_units, "masked_slow_groups": masked_slow}

    def topology_summary(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "non_transformer": True,
            "input_encoding": "raw_utf8_bytes_259",
            "params": self.count_parameters(),
            "d_model": self.cfg.d_model,
            "n_groups": len(self.groups),
            "n_residual_groups": len(self.residual_groups),
            "group_units": [cast(StatefulNeuralGroup, g).n_units for g in self.groups],
            "dynamic_vocab_size": self.encoder.dynamic_vocab.vocab_size,
            "use_fast_memory": self.cfg.use_fast_memory,
            "growth_events": len(self.growth_lineage),
            "growth_lineage": list(self.growth_lineage),
        }

    def export_capsule(self, prefix: str = "native_byte") -> dict[str, Any]:
        sd: dict[str, Any] = {
            f"{prefix}.w.{k}": v.detach().cpu().clone() for k, v in self.state_dict().items()
        }
        sd[f"{prefix}.cfg"] = self.cfg.to_dict()
        sd[f"{prefix}.lineage"] = list(self.growth_lineage)
        sd[f"{prefix}.n_residual"] = len(self.residual_groups)
        sd[f"{prefix}.unit_ids"] = [cast(StatefulNeuralGroup, g).unit_ids for g in self.groups]
        sd[f"{prefix}.unit_branches"] = [
            cast(StatefulNeuralGroup, g).unit_branches for g in self.groups
        ]
        sd[f"{prefix}.retired_unit_ids"] = [
            cast(StatefulNeuralGroup, g).retired_unit_ids for g in self.groups
        ]
        sd[f"{prefix}.dynamic_vocab"] = self.encoder.dynamic_vocab.to_dict()
        return sd

    @classmethod
    def from_capsule(
        cls, capsule: dict[str, Any], prefix: str = "native_byte"
    ) -> TaijiNativeCognitiveBackbone:
        cfg = NativeBackboneConfig.from_dict(capsule[f"{prefix}.cfg"])
        model = cls(cfg)
        for _ in range(int(capsule.get(f"{prefix}.n_residual", 0))):
            model.grow_group_neutral_()
        model.load_capsule(capsule, prefix=prefix)
        return model

    def load_capsule(self, capsule: dict[str, Any], prefix: str = "native_byte") -> None:
        dv = capsule.get(f"{prefix}.dynamic_vocab")
        if isinstance(dv, dict):
            self.encoder.dynamic_vocab = DynamicByteVocab.from_dict(dv)
        pfx = f"{prefix}.w."
        sd = {k[len(pfx) :]: v for k, v in capsule.items() if k.startswith(pfx)}
        macro_w = sd.get("encoder.macro_delta")
        if isinstance(macro_w, torch.Tensor) and macro_w.shape != self.encoder.macro_delta.shape:
            self.encoder.macro_delta = nn.Parameter(
                torch.zeros_like(
                    macro_w,
                    device=self.encoder.byte_emb.weight.device,
                    dtype=self.encoder.byte_emb.weight.dtype,
                )
            )
        self.load_state_dict(sd, strict=True)
        self.growth_lineage = list(capsule.get(f"{prefix}.lineage", []))
        saved_ids = capsule.get(f"{prefix}.unit_ids")
        if isinstance(saved_ids, list):
            for g_m, ids in zip(self.groups, saved_ids, strict=True):
                cast(StatefulNeuralGroup, g_m).unit_ids = list(ids)
        saved_branches = capsule.get(f"{prefix}.unit_branches")
        if isinstance(saved_branches, list):
            for g_m, brs in zip(self.groups, saved_branches, strict=True):
                cast(StatefulNeuralGroup, g_m).unit_branches = list(brs)
        saved_retired = capsule.get(f"{prefix}.retired_unit_ids")
        if isinstance(saved_retired, list):
            for g_m, rids in zip(self.groups, saved_retired, strict=True):
                cast(StatefulNeuralGroup, g_m).retired_unit_ids = list(rids)
