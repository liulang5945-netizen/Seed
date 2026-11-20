"""Dedicated working-memory slots over the unit graph (new module; core untouched).

``MemorySlots`` keeps K slots of d dims, written each step from the current unit
activity with a learned soft gate; ``MemoryLoop`` couples it with ``UnitScan`` so that
cross-step information can be carried by the slots alone (the caller may zero the
persistent unit state z/a to test single-channel sufficiency).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from taiji.u2_scan import UnitScan


class MemorySlots(nn.Module):
    def __init__(self, units: int, slots: int = 2, dim: int = 8) -> None:
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.write_candidate = nn.Linear(units, slots * dim)
        self.write_gate = nn.Linear(units, slots * dim)
        self.reset()

    def reset(self) -> None:
        self.memory: torch.Tensor | None = None

    def write(self, activity: torch.Tensor) -> None:
        candidate = torch.tanh(self.write_candidate(activity))
        gate = torch.sigmoid(self.write_gate(activity))
        if self.memory is None:
            self.memory = candidate
        else:
            self.memory = (1.0 - gate) * self.memory + gate * candidate

    def read(self, batch: int) -> torch.Tensor:
        if self.memory is None:
            return torch.zeros(batch, self.slots * self.dim)
        return self.memory


class MemoryLoop(nn.Module):
    def __init__(self, graph: Any, slots: int = 2, dim: int = 8) -> None:
        super().__init__()
        self.graph = graph
        self.scan = UnitScan(graph)
        self.memory = MemorySlots(graph.units.n_units, slots, dim)
        self.head = nn.Linear(graph.spec.query_slots + slots * dim, graph.spec.output_dim)

    def step(
        self, channels_t: dict[str, torch.Tensor], state: dict | None
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        batch = next(iter(channels_t.values())).shape[0]
        gates = torch.ones(batch, 1, self.graph.units.n_units)
        result = self.scan.run(channels_t, gates, state)
        state = result["state"]
        activity = result["activity"][:, 0]
        state_view = self.graph.state_mix[0] * activity + self.graph.state_mix[1] * (
            state["a"] * gates[:, 0]
        )
        projected = state_view @ self.graph.units.p.T
        direct = projected @ self.graph.query.T
        self.memory.write(activity)
        logits = self.head(torch.cat([direct, self.memory.read(batch)], dim=-1))
        return logits, state, activity
