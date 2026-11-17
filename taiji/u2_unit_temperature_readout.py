"""Unit-core B position readout with a learnable logit temperature (22.5.U.1r2).

U.1r1 located the read failure in the attention scale: the trained logit range was
0.025-0.203 nats against a ln(16)=2.77 ceiling, so attention stayed uniform and the
readout collapsed to the same column-mean statistic as U.1's mean-pool.

The obvious fix -- a larger W_q init -- was measured and rejected. Focus needs a
query norm 6.6-24.5x the non-saturating scale, and at that norm a random birth
direction saturates the softmax (max attention 0.55/0.71/0.99) and corrupts the
query gradient direction (task-gradient cosine 0.30/0.35/-0.31 against the
unsaturated scale). A large W_q also freezes its own direction: Adam displaces each
coordinate by ~lr, so 384 updates at lr 1e-3 move the query by 0.05-0.11% of its
norm, three orders of magnitude short of the rotation alignment needs.

So the scale lives in ONE scalar outside the query. ``log_tau`` starts at 0, which
reproduces the parent bit-for-bit, and the query keeps its birth norm so its
direction stays trainable. The forward below delegates the entire chain to
PositionRouteBGraph; the only difference is that ``attn_scale`` is a differentiable
tensor for the duration of the forward instead of a constant float.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .u2_neural_units import UnitGraphSpec
from .u2_unit_position_readout import PositionRouteBGraph


class TemperaturePositionGraph(PositionRouteBGraph):
    """Parent chain, one extra scalar: logits are multiplied by exp(log_tau)."""

    def __init__(self, spec: UnitGraphSpec) -> None:
        super().__init__(spec)
        self.route = "B_position_temp"
        self.base_attn_scale = 1.0 / math.sqrt(spec.hidden_dim)
        self.log_tau = nn.Parameter(torch.zeros((), dtype=spec.torch_dtype))

    def forward(
        self, channels: dict[str, torch.Tensor], gates: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        self.attn_scale = self.base_attn_scale * torch.exp(self.log_tau)
        try:
            return super().forward(channels, gates, state)
        finally:
            # with_neutral_birth deepcopies the module, and a non-leaf tensor
            # attribute cannot be deepcopied: the scale must not outlive forward.
            self.attn_scale = self.base_attn_scale
