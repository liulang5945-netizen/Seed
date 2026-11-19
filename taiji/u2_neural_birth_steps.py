"""Append a multi-step liveness probe for live-entry births (22.7e).

The single-step probe in ``newborn_liveness`` cannot see the r-path gradient: r's
gradient is proportional to p_new, which is still zero at step 1.  This adds a
k-step probe that records p's first-step gradient, the maximum r gradient over
steps 2..k, the weight changes after k steps and the newborn activity, without
changing the original function (so earlier receipts stay reproducible).
"""

from __future__ import annotations

import torch

from taiji.u2_neural_birth import newborn_liveness  # noqa: F401  (kept importable)


def newborn_liveness_steps(
    graph,
    batch_channels: dict[str, torch.Tensor],
    batch_labels: torch.Tensor,
    steps: int = 5,
    lr: float = 1e-3,
) -> dict[str, float]:
    """Run ``steps`` probe optimiser steps and report the newborn's live signals."""

    newborn = graph.units.n_units - 1
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(graph.parameters(), lr=lr)
    p_before = graph.units.p[:, newborn].detach().clone()
    r_before = graph.units.r[newborn].detach().clone()
    p_grad_first = 0.0
    r_grad_max = 0.0
    gates = torch.ones(batch_labels.shape[0], graph.units.n_units)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        readout = graph(batch_channels, gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, batch_labels)
        loss.backward()
        if step == 0:
            p_grad_first = float(graph.units.p.grad[:, newborn].norm())
        else:
            r_grad_max = max(r_grad_max, float(graph.units.r.grad[newborn].norm()))
        optimizer.step()
        graph.units.project_parameters_()
    with torch.no_grad():
        activity = graph.units(
            graph.encode_legal_inputs(batch_channels),
            gates,
            graph.units.initial_state(),
        )["activity"]
    return {
        "steps": float(steps),
        "p_grad_first": p_grad_first,
        "r_grad_max": r_grad_max,
        "p_change_norm": float((graph.units.p[:, newborn].detach() - p_before).norm()),
        "r_change_norm": float((graph.units.r[newborn].detach() - r_before).norm()),
        "newborn_activity_max": float(activity[..., newborn].abs().max()),
    }
