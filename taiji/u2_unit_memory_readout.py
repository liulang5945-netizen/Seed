"""22.5.W.1 structural memory oracle: write / hold / read one sample slot.

The DMS feasibility probe showed that the parent unit-core B graph does not retain
the sample even at zero delay. This module adds a minimal hand-written memory
state ``m`` (4 attribute slots) on top of the same chain:

  * write: at the sample stage (question[:,0] == 1), m = rgb[:, :4].mean(0);
  * hold: delay / test / query leave m unchanged;
  * read: the probe reads [m, present, m * present] into a diagnostic head.

This is a structural oracle, not a learned memory: no new parameters and no
gradient through the write. The learned-gate variant is a later card.
"""

from __future__ import annotations

from typing import Any

import torch

from .u2_neural_units import RouteBGraph, UnitGraphSpec


class MemoryRouteBGraph(RouteBGraph):
    """Route B plus a fixed 4-slot sample memory."""

    def __init__(self, spec: UnitGraphSpec) -> None:
        super().__init__(spec)
        self.route = "B_memory"

    def initial_state(self) -> dict[str, torch.Tensor]:
        state = super().initial_state()
        state["m"] = torch.zeros(4, dtype=self.spec.torch_dtype)
        return state

    def forward(
        self, channels: dict[str, torch.Tensor], gates: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        out = super().forward(channels, gates, state)
        question = channels["question"][0]
        if float(question[0]) > 0.5:
            memory = channels["rgb"][:, :4].mean(dim=0).to(self.spec.torch_dtype)
        else:
            memory = state.get("m", torch.zeros(4, dtype=self.spec.torch_dtype))
        return {
            **out,
            "memory": memory,
            "next_state": {**out["next_state"], "m": memory},
            "path": "direct_units_memory",
        }
