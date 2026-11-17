"""22.5.V.1 hard-selection readout: query-indexed column value, value side frozen.

The value-side review showed that the answer column's value is linearly readable
(3/3 seeds pass both gates) while the attention average is not (0/3), and the
model's own top-1 column is often wrong. This module replaces the soft average
with a hard top-1 column read:

  * ``selected = argmax(attention)``;
  * forward readout ``position_head(projected[selected])``;
  * training only: a straight-through soft-attention gradient for ``W_q`` so the
    task loss can move the selection, while the forward stays hard.

No new parameters. The runner freezes everything except ``W_q`` and
``position_head`` and inherits the parent U.1r2 capsule's weights explicitly
(the route changes to ``B_hard_select`` so a plain ``apply_capsule`` refuses the
parent instead of silently loading it).
"""

from __future__ import annotations

from typing import Any

import torch

from .u2_neural_units import UnitGraphSpec
from .u2_unit_temperature_readout import TemperaturePositionGraph


class HardSelectionGraph(TemperaturePositionGraph):
    """Parent chain with ``argmax(attention)`` and direct column readout."""

    def __init__(self, spec: UnitGraphSpec) -> None:
        super().__init__(spec)
        self.route = "B_hard_select"

    def forward(
        self, channels: dict[str, torch.Tensor], gates: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        inputs = self.encode_legal_inputs(channels)
        active = self.units.active_gate(gates)
        result = self.units(inputs, active, state)
        gated_a = result["a_trace"] * active
        state_view = self.state_mix[0] * result["activity"] + self.state_mix[1] * gated_a
        projected = torch.einsum("hn,tn->th", self.units.p, state_view)  # [T,H]
        question_vec = channels["question"][0].to(projected.dtype)
        attn_query = question_vec @ self.W_q
        self.attn_scale = self.base_attn_scale * torch.exp(self.log_tau)
        try:
            attn_logits = (projected @ attn_query) * self.attn_scale
            attention = torch.softmax(attn_logits, dim=0)
        finally:
            self.attn_scale = self.base_attn_scale
        selected = int(torch.argmax(attention).item())
        hard_value = projected[selected]
        soft_value = attention @ projected
        # Forward hard, backward through the soft selection: the only gradient
        # path from the task loss to W_q. position_head sees the hard column.
        value = hard_value + (soft_value - soft_value.detach())
        readout = self.position_head(value)
        return {
            "readout": readout,
            "attention": attention,
            "attended": soft_value,
            "selected": selected,
            "hard_value": hard_value,
            "projected": projected,
            "direct_query": value,
            "unit_result": result,
            "next_state": {
                **state,
                "z": result["z"],
                "a": result["a"],
                "H": state["H"],
                "internal_tick": state["internal_tick"] + inputs.shape[0],
            },
            "path": "direct_units_hard_select",
        }
