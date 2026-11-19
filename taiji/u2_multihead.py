"""Task-conditioned readout heads on top of the unit graph (architecture option ①).

Adds no change to ``taiji/u2_neural_units.py``: this subclass keeps the existing
RouteBGraph but exposes several readout heads, so a new task can use a fresh head
while the old head and the shared units stay frozen (parameter isolation through a
task-specific output path).  Used by the multi-head feasibility probe.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec


class MultiHeadGraph(RouteBGraph):
    """RouteBGraph with a ModuleList of readout heads and a ``head`` index."""

    def __init__(self, spec: UnitGraphSpec, heads: int = 2, head_hidden: int = 0) -> None:
        super().__init__(spec)
        first = self.readout
        extra_modules = []
        for _ in range(max(0, heads - 1)):
            if head_hidden > 0:
                extra_modules.append(
                    nn.Sequential(
                        nn.Linear(spec.query_slots, head_hidden, bias=True, dtype=spec.torch_dtype),
                        nn.Tanh(),
                        nn.Linear(head_hidden, spec.output_dim, bias=False, dtype=spec.torch_dtype),
                    )
                )
            else:
                extra_modules.append(
                    nn.Linear(spec.query_slots, spec.output_dim, bias=False, dtype=spec.torch_dtype)
                )
        self.heads = nn.ModuleList([first, *extra_modules])

    def forward(  # type: ignore[override]
        self,
        channels: dict[str, torch.Tensor],
        gates: torch.Tensor,
        state: dict[str, torch.Tensor],
        *,
        head: int = 0,
    ) -> dict[str, Any]:
        inputs = self.encode_legal_inputs(channels)
        active = self.units.active_gate(gates)
        result = self.units(inputs, active, state)
        gated_a = result["a_trace"] * active
        state_view = self.state_mix[0] * result["activity"] + self.state_mix[1] * gated_a
        projected = torch.einsum("hn,tn->th", self.units.p, state_view)
        direct = projected @ self.query.T
        return {
            "readout": self.heads[head](direct),
            "direct_query": direct,
            "unit_result": result,
            "next_state": {
                **state,
                "z": result["z"],
                "a": result["a"],
                "H": state["H"],
                "internal_tick": state["internal_tick"] + inputs.shape[0],
            },
            "path": "direct_units",
        }


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def copy_head_state(source: nn.Module, target: nn.Module) -> None:
    with torch.no_grad():
        target.weight.copy_(source.weight)
