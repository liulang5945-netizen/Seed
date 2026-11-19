"""Live-entry birth for the 22.7 growth line (no change to u2_neural_units.py).

``with_neutral_birth`` zeroes both r_new and p_new, which closes the input path
(r=0 -> drive=0 -> activity=0) and therefore the gradient path for p_new and r_new:
the newborn is a permanent dead unit (D-20261119-45).  This module adds a *live
entry* birth: keep ``p_new = 0`` so the output is exactly unchanged at birth, but
inherit the parent's input row ``r_new = r[parent]`` so ``activity_new != 0`` and
``p_new`` receives a non-zero gradient on the first step.

Dependency pins: this module imports but never edits ``taiji/u2_neural_units.py``.
"""

from __future__ import annotations

from typing import Any

import torch

from taiji.u2_neural_units import NeuralUnits


def live_entry_birth(
    units: NeuralUnits, group_id: int, parent_ids: list[str]
) -> tuple[NeuralUnits, dict[str, Any]]:
    """Return a child with one live-entry immature unit and birth bookkeeping."""

    parent_id = parent_ids[0]
    index = {unit["unit_id"]: position for position, unit in enumerate(units.identities["units"])}
    if parent_id not in index:
        raise ValueError("unknown_parent_id")
    parent_row = units.r[index[parent_id]].detach().clone()

    grown = units.with_neutral_birth(group_id, [parent_id])
    newborn = len(grown.identities["units"]) - 1
    with torch.no_grad():
        grown.r[newborn].copy_(parent_row)
        grown.r[newborn].clamp_(-0.3, 0.3)
    grown.project_parameters_()
    return grown, {
        "newborn_index": newborn,
        "parent_id": parent_id,
        "parent_r_norm": float(parent_row.norm()),
        "inherited_r_norm": float(grown.r[newborn].norm()),
        "p_new_norm": float(grown.p[:, newborn].norm()),
    }


def newborn_liveness(
    graph, batch_channels: dict[str, torch.Tensor], batch_labels: torch.Tensor
) -> dict[str, float]:
    """One probe optimisation step; report the newborn's gradient/weight signal."""

    newborn = graph.units.n_units - 1
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(graph.parameters(), lr=1e-3)
    gates = torch.ones(batch_labels.shape[0], graph.units.n_units)
    readout = graph(batch_channels, gates, graph.initial_state())["readout"]
    loss = loss_fn(readout, batch_labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    p_grad = float(graph.units.p.grad[:, newborn].norm())
    r_grad = float(graph.units.r.grad[newborn].norm())
    p_before = graph.units.p[:, newborn].detach().clone()
    r_before = graph.units.r[newborn].detach().clone()
    optimizer.step()
    graph.units.project_parameters_()
    p_change = float((graph.units.p[:, newborn].detach() - p_before).norm())
    r_change = float((graph.units.r[newborn].detach() - r_before).norm())
    with torch.no_grad():
        activity = graph.units(
            graph.encode_legal_inputs(batch_channels),
            torch.ones(batch_labels.shape[0], graph.units.n_units),
            graph.units.initial_state(),
        )["activity"]
    newborn_activity = (
        float(activity[:, newborn].abs().max())
        if activity.ndim == 2
        else float(activity[..., newborn].abs().max())
    )
    return {
        "p_grad_norm": p_grad,
        "r_grad_norm": r_grad,
        "p_change_norm": p_change,
        "r_change_norm": r_change,
        "newborn_activity_max": newborn_activity,
    }
