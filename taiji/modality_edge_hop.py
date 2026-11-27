"""Task-agnostic Modality-Split Pairwise Edge-Difference + Multi-Hop Architecture (``ModalityEdgeHop``).

Diagnostic findings (N32.5.01–N32.5.03):
  1. Single-vector ``GRU + read_gate`` saturates at ``0.7879`` (overfit gap ``+10.78%``)
     because one hidden vector cannot generalize two-hop relational pointers.
  2. Standard dot-product Transformer attention (``Q @ K^T``) struggles to learn signed
     coordinate comparisons (``pos_j - pos_i``) required for spatial relation grounding.
  3. Unordered scene slots need permutation-equivariant pairwise edge-difference message
     passing (``[o_i, o_j, o_j - o_i, q]``) combined with a direct query residual skip so
     the model avoids majority-class collapse.

``ModalityEdgeHop`` uses only modality boundary flags present in the raw observation ticks
(sensory scene tokens vs language query tokens), with zero ``task_id`` and zero hardcoded
entity counts or coordinates.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

ARCH_VERSION = "taiji-modality-edge-hop-v1"
INPUT_DIM = 48
READOUT_HIDDEN = 64
DEFAULT_SCENE_FLAG = 9
DEFAULT_QUERY_FLAG = 42


def extract_modality_masks(
    raw_ticks: torch.Tensor,
    *,
    scene_flag: int = DEFAULT_SCENE_FLAG,
    query_flag: int = DEFAULT_QUERY_FLAG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract boolean ``(scene_mask, query_mask)`` from unstandardized input ticks."""
    if raw_ticks.dim() != 3 or raw_ticks.shape[-1] != INPUT_DIM:
        raise ValueError(f"ticks must be (B, T, {INPUT_DIM}); got {tuple(raw_ticks.shape)}")
    scene_mask = raw_ticks[:, :, scene_flag] > 0.5
    query_mask = raw_ticks[:, :, query_flag] > 0.5
    return scene_mask, query_mask


class ModalityEdgeHop(nn.Module):
    """Language GRU + permutation-equivariant object edge-difference stack + K-hop readout."""

    def __init__(
        self,
        out_dim: int,
        *,
        d_model: int = 64,
        d_edge: int = 64,
        n_edge_layers: int = 2,
        n_hops: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_edge <= 0 or n_edge_layers < 1 or n_hops < 1:
            raise ValueError("dimensions, edge layers, and hops must be positive")

        self.out_dim = int(out_dim)
        self.d_model = int(d_model)
        self.d_edge = int(d_edge)
        self.n_edge_layers = int(n_edge_layers)
        self.n_hops = int(n_hops)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.q_gru = nn.GRU(INPUT_DIM, self.d_model, batch_first=True)
            self.obj_in = nn.Sequential(
                nn.Linear(INPUT_DIM, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.d_model),
            )
            self.edge_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(4 * self.d_model, self.d_edge),
                        nn.ReLU(),
                        nn.Linear(self.d_edge, self.d_model + 1),
                    )
                    for _ in range(self.n_edge_layers)
                ]
            )
            self.obj_lns = nn.ModuleList(
                [nn.LayerNorm(self.d_model) for _ in range(self.n_edge_layers)]
            )
            self.null_obj = nn.Parameter(torch.empty(1, 1, self.d_model))
            nn.init.normal_(self.null_obj, std=0.02)

            self.hop_q = nn.ModuleList(
                [nn.Linear(self.d_model, self.d_model) for _ in range(self.n_hops)]
            )
            self.hop_k = nn.ModuleList(
                [nn.Linear(self.d_model, self.d_model) for _ in range(self.n_hops)]
            )
            self.hop_v = nn.ModuleList(
                [nn.Linear(self.d_model, self.d_model) for _ in range(self.n_hops)]
            )
            self.hop_upd = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * self.d_model, self.d_model),
                        nn.LayerNorm(self.d_model),
                        nn.ReLU(),
                    )
                    for _ in range(self.n_hops)
                ]
            )
            self.head = nn.Sequential(
                nn.Linear(2 * self.d_model, READOUT_HIDDEN),
                nn.ReLU(),
                nn.Linear(READOUT_HIDDEN, self.out_dim),
            )

    def _gru_step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ih = F.linear(
            x_t,
            cast(torch.Tensor, self.q_gru.weight_ih_l0),
            cast(torch.Tensor, self.q_gru.bias_ih_l0),
        )
        hh = F.linear(
            h,
            cast(torch.Tensor, self.q_gru.weight_hh_l0),
            cast(torch.Tensor, self.q_gru.bias_hh_l0),
        )
        r_i, z_i, n_i = ih.chunk(3, dim=-1)
        r_h, z_h, n_h = hh.chunk(3, dim=-1)
        r = torch.sigmoid(r_i + r_h)
        z = torch.sigmoid(z_i + z_h)
        n = torch.tanh(n_i + r * n_h)
        return (1.0 - z) * n + z * h

    def _encode_query(self, ticks: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        batch, horizon, _ = ticks.shape
        if bool((query_mask == query_mask[0:1]).all()) and bool(query_mask[0].any()):
            idx = torch.nonzero(query_mask[0], as_tuple=False).squeeze(-1)
            q_lo = int(idx[0].item())
            q_hi = int(idx[-1].item()) + 1
            if idx.numel() == q_hi - q_lo:
                _, qh = self.q_gru(ticks[:, q_lo:q_hi])
                return cast(torch.Tensor, qh.squeeze(0))

        h = ticks.new_zeros(batch, self.d_model)
        qf = query_mask.to(ticks.dtype)
        for t in range(horizon):
            v = qf[:, t : t + 1]
            h = v * self._gru_step(ticks[:, t], h) + (1.0 - v) * h
        return h

    def _edge_pass(
        self,
        o: torch.Tensor,
        q_vec: torch.Tensor,
        edge_mlp: nn.Module,
        ln: nn.Module,
        omask: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_obj, d = o.shape
        oi = o.unsqueeze(2).expand(batch, n_obj, n_obj, d)
        oj = o.unsqueeze(1).expand(batch, n_obj, n_obj, d)
        qb = q_vec[:, None, None, :].expand(batch, n_obj, n_obj, d)
        out = edge_mlp(torch.cat([oi, oj, oj - oi, qb], dim=-1))
        msg, gate = out[..., :d], out[..., d]
        gate = gate.masked_fill(~omask.unsqueeze(1), -1e9)
        attn = torch.softmax(gate, dim=2)
        agg = (attn.unsqueeze(-1) * msg).sum(dim=2)
        return cast(torch.Tensor, ln(o + agg)) * omask.unsqueeze(-1).to(o.dtype)

    def forward(
        self,
        ticks: torch.Tensor,
        scene_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        if ticks.dim() != 3 or ticks.shape[-1] != INPUT_DIM:
            raise ValueError(f"ticks must be (B, T, {INPUT_DIM}); got {tuple(ticks.shape)}")
        if scene_mask.shape != ticks.shape[:2] or query_mask.shape != ticks.shape[:2]:
            raise ValueError("scene_mask and query_mask must match (B, T)")

        batch = ticks.shape[0]
        q_vec = self._encode_query(ticks, query_mask)

        max_objs = int(scene_mask.sum(dim=1).max().item())
        if max_objs == 0:
            o = ticks.new_zeros(batch, 0, self.d_model)
            omask = scene_mask[:, :0]
        else:
            idx = (
                scene_mask.to(torch.int64).topk(max_objs, dim=1, largest=True, sorted=False).indices
            )
            o_raw = ticks.gather(1, idx.unsqueeze(-1).expand(-1, -1, INPUT_DIM))
            omask = scene_mask.gather(1, idx)
            o = self.obj_in(o_raw) * omask.unsqueeze(-1).to(ticks.dtype)
            for edge_mlp, ln in zip(self.edge_mlps, self.obj_lns, strict=True):
                o = self._edge_pass(o, q_vec, edge_mlp, ln, omask)

        slots = torch.cat([o, self.null_obj.expand(batch, 1, self.d_model)], dim=1)
        smask = torch.cat([omask, scene_mask.new_ones(batch, 1)], dim=1)

        s = q_vec
        scale = 1.0 / math.sqrt(self.d_model)
        for hop in range(self.n_hops):
            scores = (self.hop_q[hop](s).unsqueeze(1) * self.hop_k[hop](slots)).sum(-1) * scale
            scores = scores.masked_fill(~smask, -1e9)
            attn = torch.softmax(scores, dim=-1)
            retrieved = (attn.unsqueeze(-1) * self.hop_v[hop](slots)).sum(dim=1)
            s = s + self.hop_upd[hop](torch.cat([s, retrieved], dim=-1))

        return cast(torch.Tensor, self.head(torch.cat([q_vec, s], dim=-1)))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def spec(self) -> dict[str, Any]:
        return {
            "arch_version": ARCH_VERSION,
            "out_dim": self.out_dim,
            "d_model": self.d_model,
            "d_edge": self.d_edge,
            "n_edge_layers": self.n_edge_layers,
            "n_hops": self.n_hops,
        }

    def capsule(self) -> dict[str, Any]:
        state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        return {"spec": self.spec(), "state": state}

    @classmethod
    def from_capsule(cls, capsule: dict[str, Any]) -> ModalityEdgeHop:
        spec = capsule["spec"]
        if spec.get("arch_version") != ARCH_VERSION:
            raise ValueError(f"capsule arch {spec.get('arch_version')!r} != {ARCH_VERSION}")
        model = cls(
            int(spec["out_dim"]),
            d_model=int(spec["d_model"]),
            d_edge=int(spec["d_edge"]),
            n_edge_layers=int(spec["n_edge_layers"]),
            n_hops=int(spec["n_hops"]),
        )
        model.load_state_dict(capsule["state"], strict=True)
        return model
