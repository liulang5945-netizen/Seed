"""22.5.W.1 memory readout: a structural oracle and a learned-gate variant.

``MemoryRouteBGraph`` is the fixed write/hold/read oracle used by the W.1 probe.
``LearnedMemoryGraph`` replaces the fixed write with a gated update and adds a
small 2-way memory readout; at birth the gate is closed and the readout is zero,
so the base graph's behavior is unchanged. The runner freezes the base graph and
trains only the gate and the memory readout.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

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


class LearnedMemoryGraph(RouteBGraph):
    """Gated write / hold / read with a neutral birth and a 2-way readout."""

    def __init__(self, spec: UnitGraphSpec) -> None:
        super().__init__(spec)
        self.route = "B_learned_memory"
        dtype = spec.torch_dtype
        width = spec.groups * spec.input_dim + 8
        self.write_gate = nn.Linear(width, 4, dtype=dtype)
        self.write_candidate = nn.Linear(width, 4, dtype=dtype)
        self.memory_readout = nn.Linear(12, 2, bias=False, dtype=dtype)
        with torch.no_grad():
            self.write_gate.weight.zero_()
            self.write_gate.bias.fill_(-10.0)
            self.write_candidate.weight.zero_()
            self.write_candidate.bias.zero_()
            self.memory_readout.weight.zero_()

    def initial_state(self) -> dict[str, torch.Tensor]:
        state = super().initial_state()
        dtype = self.spec.torch_dtype
        state["m"] = torch.zeros(4, dtype=dtype)
        state["p"] = torch.zeros(4, dtype=dtype)
        return state

    def forward(
        self, channels: dict[str, torch.Tensor], gates: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        out = super().forward(channels, gates, state)
        inputs = self.encode_legal_inputs(channels)
        summary = inputs.mean(dim=0).reshape(-1)
        memory = state.get("m", torch.zeros(4, dtype=self.spec.torch_dtype))
        presence = state.get("p", torch.zeros(4, dtype=self.spec.torch_dtype))
        question = channels["question"][0]
        x = torch.cat([summary, memory, presence])
        gate = torch.sigmoid(self.write_gate(x))
        candidate = torch.tanh(self.write_candidate(x))
        updated = (1.0 - gate) * memory + gate * candidate
        if float(question[2]) > 0.5:  # test stage carries the test row presence
            presence = channels["rgb"][:, :4].max(dim=0).values.to(self.spec.torch_dtype)
        readout = self.memory_readout(torch.cat([updated, presence, updated * presence]))
        return {
            **out,
            "memory": updated,
            "presence": presence,
            "memory_readout": readout,
            "next_state": {**out["next_state"], "m": updated, "p": presence},
            "path": "direct_units_learned_memory",
        }
