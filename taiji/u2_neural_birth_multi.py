"""Multi-unit live-entry birth + multi-newborn liveness probe (22.7f)."""

from __future__ import annotations

from typing import Any

import torch

from taiji.u2_neural_birth import live_entry_birth


def live_entry_birth_multi(
    units, group_id: int, parent_ids: list[str], count: int = 3
) -> tuple[Any, list[dict[str, Any]]]:
    """Birth ``count`` live-entry units one by one, rotating the inherited parents."""

    if not 1 <= count <= 3:
        raise ValueError("birth_count_out_of_range")
    current = units
    details: list[dict[str, Any]] = []
    for offset in range(count):
        parent = parent_ids[offset % len(parent_ids)]
        current, detail = live_entry_birth(current, group_id, [parent])
        details.append(detail)
    return current, details


def multi_newborn_liveness_steps(
    graph,
    batch_channels: dict[str, torch.Tensor],
    batch_labels: torch.Tensor,
    count: int = 3,
    steps: int = 5,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Probe the last ``count`` units; return per-newborn and aggregate liveness."""

    total = graph.units.n_units
    indices = list(range(total - count, total))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(graph.parameters(), lr=lr)
    r_before = {index: graph.units.r[index].detach().clone() for index in indices}
    p_before = {index: graph.units.p[:, index].detach().clone() for index in indices}
    p_grad_first = {index: 0.0 for index in indices}
    r_grad_max = {index: 0.0 for index in indices}
    gates = torch.ones(batch_labels.shape[0], total)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        readout = graph(batch_channels, gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, batch_labels)
        loss.backward()
        for index in indices:
            if step == 0:
                p_grad_first[index] = float(graph.units.p.grad[:, index].norm())
            else:
                r_grad_max[index] = max(r_grad_max[index], float(graph.units.r.grad[index].norm()))
        optimizer.step()
        graph.units.project_parameters_()
    with torch.no_grad():
        activity = graph.units(
            graph.encode_legal_inputs(batch_channels), gates, graph.units.initial_state()
        )["activity"]
    per_newborn = {}
    for index in indices:
        per_newborn[str(index)] = {
            "p_grad_first": p_grad_first[index],
            "r_grad_max": r_grad_max[index],
            "p_change_norm": float(
                graph.units.p[:, index].detach().norm() - p_before[index].norm()
            ),
            "r_change_norm": float(graph.units.r[index].detach().norm() - r_before[index].norm()),
            "activity_max": float(activity[..., index].abs().max()),
        }
    return {
        "steps": steps,
        "count": count,
        "per_newborn": per_newborn,
        "live": all(
            entry["p_grad_first"] > 0.0
            and entry["r_grad_max"] > 0.0
            and entry["p_change_norm"] > 0.0
            and entry["r_change_norm"] > 0.0
            and entry["activity_max"] > 0.0
            for entry in per_newborn.values()
        ),
    }
