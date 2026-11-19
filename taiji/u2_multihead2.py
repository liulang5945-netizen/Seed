"""Per-task representation entries on top of frozen shared units (option ②).

Each task gets its own input encoders, query matrix and readout head, while the
shared unit graph (and the base input projection) stays frozen.  This lets a new
task form its own pathway without altering the pathway used by earlier tasks, so
old abilities can be retained exactly while new tasks get real representational
capacity (D-20261119-76/77 showed heads alone are capacity-limited and unfreezing
shared units breaks old abilities).

Core ``taiji/u2_neural_units.py`` is untouched.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from taiji.u2_neural_units import LEGAL_INPUT_CHANNELS, RouteBGraph, UnitGraphSpec


class MultiHeadGraph2(RouteBGraph):
    """RouteB with per-task input encoders/query/head over frozen shared units."""

    def __init__(self, spec: UnitGraphSpec, heads: int = 3, head_hidden: int = 16) -> None:
        super().__init__(spec)
        dtype = spec.torch_dtype
        extra_queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(spec.query_slots, spec.hidden_dim, dtype=dtype) * 0.3)
                for _ in range(max(0, heads - 1))
            ]
        )
        self.task_queries = extra_queries
        extra_heads = []
        for _ in range(max(0, heads - 1)):
            extra_heads.append(
                nn.Sequential(
                    nn.Linear(spec.query_slots, head_hidden, bias=True, dtype=dtype),
                    nn.Tanh(),
                    nn.Linear(head_hidden, spec.output_dim, bias=False, dtype=dtype),
                )
            )
        self.heads = nn.ModuleList([self.readout, *extra_heads])
        encoders = nn.ModuleDict()
        for index in range(max(0, heads - 1)):
            encoders[f"h{index + 1}"] = nn.ModuleDict(
                {
                    channel: nn.Linear(spec.input_dim, spec.input_dim, bias=False, dtype=dtype)
                    for channel in LEGAL_INPUT_CHANNELS
                }
            )
        with torch.no_grad():
            for index in range(max(0, heads - 1)):
                for channel in LEGAL_INPUT_CHANNELS:
                    encoders[f"h{index + 1}"][channel].weight.copy_(
                        self.input_encoders[channel].weight
                    )
        self.task_encoders = encoders

    def _encode(self, channels: dict[str, torch.Tensor], head: int) -> torch.Tensor:
        encoders = self.task_encoders[f"h{head}"] if head > 0 else self.input_encoders
        columns = []
        for channel in LEGAL_INPUT_CHANNELS:
            columns.append(encoders[channel](channels[channel]) @ self.input_projection)
        stacked = torch.stack(columns, dim=1)
        repeats = [
            stacked[:, group % len(LEGAL_INPUT_CHANNELS), :] for group in range(self.spec.groups)
        ]
        return torch.stack(repeats, dim=1)

    def forward(  # type: ignore[override]
        self,
        channels: dict[str, torch.Tensor],
        gates: torch.Tensor,
        state: dict[str, torch.Tensor],
        *,
        head: int = 0,
    ) -> dict[str, Any]:
        inputs = self._encode(channels, head)
        active = self.units.active_gate(gates)
        result = self.units(inputs, active, state)
        gated_a = result["a_trace"] * active
        state_view = self.state_mix[0] * result["activity"] + self.state_mix[1] * gated_a
        projected = torch.einsum("hn,tn->th", self.units.p, state_view)
        query = self.query if head == 0 else self.task_queries[head - 1]
        direct = projected @ query.T
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


def task_parameters(graph: MultiHeadGraph2, head: int) -> list[nn.Parameter]:
    """Parameters belonging to one task's pathway (head 0 = shared/base path)."""

    if head == 0:
        parameters = list(graph.heads[0].parameters()) + [graph.query]
        parameters += list(graph.input_encoders.parameters())
    else:
        parameters = list(graph.heads[head].parameters()) + [graph.task_queries[head - 1]]
        parameters += list(graph.task_encoders[f"h{head}"].parameters())
    return parameters
