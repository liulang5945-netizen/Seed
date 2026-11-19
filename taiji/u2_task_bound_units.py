"""Task-bound units: explicitly differentiated units gated per task (unit level).

Extends MultiHeadGraph2 with a binding table: task 0 uses the shared units; each
new task gets k newborn units bound to it and is gated on for that task only.  The
rest of the shared graph stays frozen, so adding a task cannot change earlier
tasks' readouts.  Bound units train only their own rows/columns in the unit state.
"""

from __future__ import annotations

from typing import Any

import torch

from taiji.u2_multihead2 import MultiHeadGraph2


class TaskBoundGraph(MultiHeadGraph2):
    """MultiHeadGraph2 with per-task unit sets (shared + task-bound newborns)."""

    def __init__(self, spec, heads: int = 3, head_hidden: int = 16) -> None:
        super().__init__(spec, heads=heads, head_hidden=head_hidden)
        self.task_units: dict[int, list[int]] = {index: [] for index in range(heads)}
        self.shared_units: list[int] = list(range(self.units.n_units))

    def gate_mask(self, head: int, device=None) -> torch.Tensor:
        mask = torch.zeros(self.units.n_units, device=device)
        mask[self.shared_units] = 1.0
        for unit in self.task_units.get(head, []):
            mask[unit] = 1.0
        return mask

    def forward(  # type: ignore[override]
        self,
        channels: dict[str, torch.Tensor],
        gates: torch.Tensor,
        state: dict[str, torch.Tensor],
        *,
        head: int = 0,
    ) -> dict[str, Any]:
        mask = self.gate_mask(head, gates.device)
        return super().forward(channels, gates * mask, state, head=head)

    def bind_born_units(self, head: int, born: list[int], shared: list[int]) -> None:
        self.task_units[head] = list(born)
        self.shared_units = list(shared)


def masked_unit_step(
    graph: TaskBoundGraph,
    head: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    bound: list[int],
    steps: int,
    lr: float,
    generator: torch.Generator,
    batch: int = 64,
) -> None:
    """Train the task pathway plus only the bound units' own state rows."""

    parameters = list(graph.heads[head].parameters()) + [graph.task_queries[head - 1]]
    parameters += list(graph.task_encoders[f"h{head}"].parameters())
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    units = graph.units
    classes = int(labels.max().item()) + 1
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        index = torch.randint(0, features.shape[0], (batch,), generator=generator)
        gates = torch.ones(batch, units.n_units)
        optimizer.zero_grad(set_to_none=True)
        if units.p.grad is not None:
            units.p.grad = None
        logits = graph(_channels(features[index]), gates, graph.initial_state(), head=head)[
            "readout"
        ]
        loss = loss_fn(logits[:, :classes], labels[index])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if units.p.grad is not None:
                units.p[:, bound] -= lr * units.p.grad[:, bound]
            if units.r.grad is not None:
                units.r[bound, :] -= lr * units.r.grad[bound, :]
            for name in ("c", "eta", "b"):
                parameter = getattr(units, name)
                if parameter.grad is not None:
                    parameter[bound] -= lr * parameter.grad[bound]
        units.zero_grad(set_to_none=True)
        units.project_parameters_()


def _channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    names = ("rgb", "audio", "proprio", "question", "goal", "feedback")
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(names)}
