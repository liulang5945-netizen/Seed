"""Slot-Attention Relational Multi-Hop GRU (``SlotHopGRU``) for entity binding & composition.

Motivation (N32.4 / N32.5.01):
  Plain single-vector ``GRU(48 -> 96) + final-tick read_gate`` reached only 0.7039
  (entity mean 0.4009) on two-hop relational composition ``Q3``, because:
    1. All unordered scene entities and question tokens must be compressed into a single
       fixed vector before answering, with no explicit slot-to-slot relational comparison.
    2. In question-first (QF) order, the final tick ``x[:, -1]`` is a random body tick
       rather than the query, so ``read_gate(x[:, -1])`` cannot condition readout on
       the question.

``SlotHopGRU`` fixes both issues cleanly while staying 100% task-agnostic (zero task IDs,
zero task-specific heads) and parameter-budget matched (~60-70k params vs GRU-u96 53.6k):
  1. Causal GRU encoder produces contextual token states ``h_t`` (holding state on zero
     padding ticks).
  2. Working-memory slots ``M_t = LayerNorm(Linear([x_t, h_t]))`` keep every valid tick
     addressable.
  3. One permutation-equivariant relational self-attention block lets entity slots and
     query tokens interact pairwise regardless of scene permutation order.
  4. ``K`` hops of query-driven cross-attention (initialized at ``h_last`` and updated via
     a hop ``GRUCell``) attend over ``[M, null_slot]``, letting hop 1 resolve the
     first-hop entity and hop 2 follow its second-hop relation (or select ``null_slot``
     when a relation does not exist).
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

ARCH_VERSION = "taiji-slot-hop-gru-v1"
INPUT_DIM = 48
READOUT_HIDDEN = 64


def valid_steps(ticks: torch.Tensor) -> torch.Tensor:
    """(B, T) bool mask; a tick whose features are all zero is padding."""
    return ticks.abs().sum(-1) > 0


class SlotHopGRU(nn.Module):
    """Task-agnostic GRU + relational self-attention + K-hop query-driven readout."""

    def __init__(
        self,
        out_dim: int,
        *,
        d_model: int = 48,
        n_heads: int = 4,
        n_hops: int = 2,
        ffn_mult: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be a positive multiple of n_heads ({n_heads})"
            )
        if n_hops < 1:
            raise ValueError(f"n_hops must be >= 1, got {n_hops}")

        self.out_dim = int(out_dim)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.n_hops = int(n_hops)
        self.ffn_mult = int(ffn_mult)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.gru = nn.GRU(INPUT_DIM, self.d_model, batch_first=True)
            self.slot_proj = nn.Linear(INPUT_DIM + self.d_model, self.d_model)
            self.slot_norm = nn.LayerNorm(self.d_model)

            # Permutation-equivariant relational self-attention over valid slots
            self.rel_norm1 = nn.LayerNorm(self.d_model)
            self.rel_qkv = nn.Linear(self.d_model, 3 * self.d_model)
            self.rel_out = nn.Linear(self.d_model, self.d_model)
            self.rel_norm2 = nn.LayerNorm(self.d_model)
            self.rel_ffn = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * self.ffn_mult),
                nn.GELU(),
                nn.Linear(self.d_model * self.ffn_mult, self.d_model),
            )

            # Iterative K-hop query-driven cross-attention with learned null slot
            self.null_slot = nn.Parameter(torch.empty(1, 1, self.d_model))
            nn.init.normal_(self.null_slot, std=0.02)
            self.hop_emb = nn.Parameter(torch.empty(self.n_hops, self.d_model))
            nn.init.normal_(self.hop_emb, std=0.02)

            self.hop_norm_q = nn.LayerNorm(self.d_model)
            self.hop_norm_kv = nn.LayerNorm(self.d_model)
            self.hop_q = nn.Linear(self.d_model, self.d_model)
            self.hop_kv = nn.Linear(self.d_model, 2 * self.d_model)
            self.hop_out = nn.Linear(self.d_model, self.d_model)
            self.hop_cell = nn.GRUCell(self.d_model, self.d_model)

            self.head = nn.Sequential(
                nn.Linear(self.d_model, READOUT_HIDDEN),
                nn.ReLU(),
                nn.Linear(READOUT_HIDDEN, self.out_dim),
            )

    def _gru_step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ih = F.linear(
            x_t,
            cast(torch.Tensor, self.gru.weight_ih_l0),
            cast(torch.Tensor, self.gru.bias_ih_l0),
        )
        hh = F.linear(
            h,
            cast(torch.Tensor, self.gru.weight_hh_l0),
            cast(torch.Tensor, self.gru.bias_hh_l0),
        )
        r_i, z_i, n_i = ih.chunk(3, dim=-1)
        r_h, z_h, n_h = hh.chunk(3, dim=-1)
        r = torch.sigmoid(r_i + r_h)
        z = torch.sigmoid(z_i + z_h)
        n = torch.tanh(n_i + r * n_h)
        return (1.0 - z) * n + z * h

    def _encode_gru(
        self, ticks: torch.Tensor, valid_bool: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, horizon, _ = ticks.shape
        if bool(valid_bool.all()):
            seq_h, h_n = self.gru(ticks)
            return seq_h, h_n.squeeze(0)

        h = ticks.new_zeros(batch, self.d_model)
        hs: list[torch.Tensor] = []
        valid_f = valid_bool.to(ticks.dtype)
        for t in range(horizon):
            v = valid_f[:, t : t + 1]
            h_cand = self._gru_step(ticks[:, t], h)
            h = v * h_cand + (1.0 - v) * h
            hs.append(h)
        seq_h = torch.stack(hs, dim=1)
        positions = torch.arange(horizon, device=ticks.device, dtype=ticks.dtype)
        last_idx = (valid_f * positions).argmax(dim=-1)
        h_last = seq_h[torch.arange(batch, device=ticks.device), last_idx]
        return seq_h, h_last

    def forward(self, ticks: torch.Tensor) -> torch.Tensor:
        if ticks.dim() != 3 or ticks.shape[-1] != INPUT_DIM:
            raise ValueError(f"ticks must be (B, T, {INPUT_DIM}); got {tuple(ticks.shape)}")

        batch, horizon, _ = ticks.shape
        valid_bool = valid_steps(ticks)
        valid_f = valid_bool.to(ticks.dtype)
        valid_3d = valid_f.unsqueeze(-1)
        scale = 1.0 / math.sqrt(self.head_dim)

        seq_h, h_last = self._encode_gru(ticks, valid_bool)

        # Build working-memory slots (valid ticks only)
        m = self.slot_norm(self.slot_proj(torch.cat([ticks, seq_h], dim=-1))) * valid_3d

        # 1) Relational self-attention block over memory slots
        m_n1 = self.rel_norm1(m)
        qkv = self.rel_qkv(m_n1).view(batch, horizon, 3, self.n_heads, self.head_dim)
        q_s = qkv[:, :, 0].transpose(1, 2)  # (B, nh, T, hd)
        k_s = qkv[:, :, 1].transpose(1, 2)
        v_s = qkv[:, :, 2].transpose(1, 2)

        scores = torch.matmul(q_s, k_s.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~valid_bool[:, None, None, :], -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = (
            torch.matmul(attn, v_s).transpose(1, 2).contiguous().view(batch, horizon, self.d_model)
        )
        m = (m + self.rel_out(ctx)) * valid_3d
        m = (m + self.rel_ffn(self.rel_norm2(m))) * valid_3d

        # 2) K-hop iterative query-driven cross-attention over [M, null_slot]
        kv_slots = torch.cat([m, self.null_slot.expand(batch, 1, self.d_model)], dim=1)
        kv_valid = torch.cat(
            [valid_bool, valid_bool.new_ones(batch, 1)], dim=1
        )  # (B, T + 1), null_slot is always valid
        kv = self.hop_kv(self.hop_norm_kv(kv_slots)).view(
            batch, horizon + 1, 2, self.n_heads, self.head_dim
        )
        k_h = kv[:, :, 0].transpose(1, 2)  # (B, nh, T+1, hd)
        v_h = kv[:, :, 1].transpose(1, 2)
        k_h_t = k_h.transpose(-2, -1)
        kv_mask = ~kv_valid[:, None, None, :]

        s = h_last
        for hop in range(self.n_hops):
            q_in = self.hop_norm_q(s + self.hop_emb[hop])
            q = self.hop_q(q_in).view(batch, self.n_heads, 1, self.head_dim)
            h_scores = torch.matmul(q, k_h_t) * scale
            h_scores = h_scores.masked_fill(kv_mask, -1e9)
            h_attn = torch.softmax(h_scores, dim=-1)
            retrieved = torch.matmul(h_attn, v_h).reshape(batch, self.d_model)
            s = self.hop_cell(self.hop_out(retrieved), s)

        return cast(torch.Tensor, self.head(s))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def spec(self) -> dict[str, Any]:
        return {
            "arch_version": ARCH_VERSION,
            "out_dim": self.out_dim,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_hops": self.n_hops,
            "ffn_mult": self.ffn_mult,
        }

    def capsule(self) -> dict[str, Any]:
        state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        return {"spec": self.spec(), "state": state}

    @classmethod
    def from_capsule(cls, capsule: dict[str, Any]) -> SlotHopGRU:
        spec = capsule["spec"]
        if spec.get("arch_version") != ARCH_VERSION:
            raise ValueError(f"capsule arch {spec.get('arch_version')!r} != {ARCH_VERSION}")
        model = cls(
            int(spec["out_dim"]),
            d_model=int(spec["d_model"]),
            n_heads=int(spec["n_heads"]),
            n_hops=int(spec["n_hops"]),
            ffn_mult=int(spec["ffn_mult"]),
        )
        model.load_state_dict(capsule["state"], strict=True)
        return model
