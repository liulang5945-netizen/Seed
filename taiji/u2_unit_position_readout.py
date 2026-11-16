"""Unit-core B position-aware readout (22.5.U.1 revision 1, ``unit-core-b-v1`` r1).

U.1 (mean-pool readout over T columns) failed the read prerequisite with a located
gap: the encoding is information-faithful (96/96 public oracle) but mean-pooling
the per-column readout washes out "which column carries the queried colour", and
the RGB input encoder is credit-starved (~30x weaker gradient than the query).

This revision replaces the mean-pool aggregation with a QUESTION-CONDITIONED
COLUMN ATTENTION while keeping the input encoding contract frozen:

  * column keys come from the unit activity (``p @ state_view``, RGB-driven), so
    gradient routes back through the units into the RGB content input;
  * the attention query is a learned projection of the question encoding
    (``question_vec @ W_q``), so attention is conditioned on WHICH entity is asked;
  * softmax attention over the T columns preserves the selected column's binding
    (its colour and its position axis x), and a position head reads the attended
    state into the 4-way position label.

Identity properties inherited from route B are re-verified by the U.1r1 contract:
H is scratch (cut_h bit-identical), neutral birth is exact (a new unit with p=0
does not change the column keys, hence attention/readout are unchanged). The
graph reads unit state directly; the question only conditions the attention query.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .u2_neural_units import RouteBGraph, UnitGraphSpec


class PositionRouteBGraph(RouteBGraph):
    """Route B with question-conditioned column attention instead of mean-pool."""

    def __init__(self, spec: UnitGraphSpec) -> None:
        super().__init__(spec)
        self.route = "B_position"
        dtype = spec.torch_dtype
        # The mean-pool readout head and the fixed query slots are replaced by the
        # attention mechanism; remove them so every parameter carries credit.
        del self.readout
        del self.query
        generator = torch.Generator().manual_seed(spec.seed + 61)
        self.W_q = nn.Parameter(
            (
                (
                    torch.rand(
                        spec.input_dim, spec.hidden_dim, generator=generator, dtype=torch.float64
                    )
                    * 2
                    - 1
                )
                * 0.3
            ).to(dtype)
        )
        self.position_head = nn.Linear(spec.hidden_dim, spec.output_dim, bias=False, dtype=dtype)
        self.attn_scale = 1.0 / math.sqrt(spec.hidden_dim)

    def forward(
        self, channels: dict[str, torch.Tensor], gates: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        inputs = self.encode_legal_inputs(channels)
        active = self.units.active_gate(gates)
        result = self.units(inputs, active, state)
        gated_a = result["a_trace"] * active
        state_view = self.state_mix[0] * result["activity"] + self.state_mix[1] * gated_a
        projected = torch.einsum("hn,tn->th", self.units.p, state_view)  # [T,H] column keys
        question_vec = channels["question"][0].to(projected.dtype)  # [input_dim]
        attn_query = question_vec @ self.W_q  # [H]
        attn_logits = (projected @ attn_query) * self.attn_scale  # [T]
        attention = torch.softmax(attn_logits, dim=0)  # [T]
        attended = attention @ projected  # [H]
        readout = self.position_head(attended)  # [output_dim]
        return {
            "readout": readout,
            "attention": attention,
            "attended": attended,
            "direct_query": attended,
            "unit_result": result,
            "next_state": {
                **state,
                "z": result["z"],
                "a": result["a"],
                "H": state["H"],
                "internal_tick": state["internal_tick"] + inputs.shape[0],
            },
            "path": "direct_units_position",
        }
