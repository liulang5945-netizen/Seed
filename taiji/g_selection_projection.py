"""Feasible-region projection solver for the P4.11 update mechanism.

Given a joint constraint system ``a_i · w >= b_i`` over the 16
extended-feature weight dimensions and an anchor vector (the trained
endpoint), the solver finds the feasible point closest to the anchor by
penalty continuation: for an increasing penalty schedule ``rho``, it
minimises ``rho * sum(max(0, b_i - a_i.w)) + 0.5 * ||w - anchor||^2``
with full-batch Adam and warm continuation.  As ``rho`` grows the
violations are driven to zero and the anchor term selects the nearest
feasible point - the projection of the anchor onto the feasible region.

The solver is deterministic (no random restarts; warm continuation from
the anchor) and is a pure function over explicit constraint matrices -
no Taiji learner state, no checkpoint.  Convergence criteria are frozen
in the P4.11 preregistration: per-constraint violation <= 1e-6 and total
violation <= 1e-5; anything else is ``projection_incomplete`` and the
experiment stops honestly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

RHO_SCHEDULE = (1.0, 10.0, 100.0, 1000.0)
STEPS_PER_PHASE = 6000
LR_START = 0.05
LR_END = 0.005
PER_CONSTRAINT_TOLERANCE = 1e-6
TOTAL_TOLERANCE = 1e-5


def _cosine_lr(step: int, total: int) -> float:
    progress = min(1.0, step / max(1, total - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_END + (LR_START - LR_END) * cosine


def project_to_joint_feasible_region(
    constraints: Sequence[tuple[Sequence[float], float, str]],
    anchor: Sequence[float],
    *,
    rho_schedule: Sequence[float] = RHO_SCHEDULE,
    steps_per_phase: int = STEPS_PER_PHASE,
    lr_start: float = LR_START,
    lr_end: float = LR_END,
    per_constraint_tolerance: float = PER_CONSTRAINT_TOLERANCE,
    total_tolerance: float = TOTAL_TOLERANCE,
) -> dict[str, Any]:
    """Project ``anchor`` onto ``{w : a_i.w >= b_i}`` by penalty continuation.

    Returns the projected weights, the violation audit (per constraint
    family and totals), the distance audit against the anchor, and the
    frozen convergence verdict.  Deterministic: the optimisation starts
    at the anchor and every phase warm-starts from the previous solution.
    """
    if not constraints:
        raise ValueError("projection requires a non-empty constraint system")
    dim = len(anchor)
    matrix = torch.tensor([item[0] for item in constraints], dtype=torch.float32)
    if matrix.shape[1] != dim:
        raise ValueError("projection constraint dimension must match the anchor")
    rhs = torch.tensor([item[1] for item in constraints], dtype=torch.float32)
    anchor_tensor = torch.tensor(anchor, dtype=torch.float32)

    variable = anchor_tensor.clone()
    phase_summaries: list[dict[str, Any]] = []
    for rho in rho_schedule:
        variable.requires_grad_(True)
        optimizer = torch.optim.Adam([variable], lr=lr_start)
        for step in range(steps_per_phase):
            optimizer.param_groups[0]["lr"] = _cosine_lr(step, steps_per_phase)
            optimizer.zero_grad()
            violations = torch.relu(rhs - matrix @ variable)
            loss = rho * violations.sum() + 0.5 * ((variable - anchor_tensor) ** 2).sum()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            violations = torch.relu(rhs - matrix @ variable)
            phase_summaries.append(
                {
                    "rho": float(rho),
                    "total_violation": float(violations.sum().item()),
                    "max_violation": float(violations.max().item()) if violations.numel() else 0.0,
                }
            )
        variable.requires_grad_(False)

    with torch.no_grad():
        final_violations = torch.relu(rhs - matrix @ variable)
        total_violation = float(final_violations.sum().item())
        max_violation = float(final_violations.max().item()) if final_violations.numel() else 0.0
        per_family: dict[str, float] = {}
        for (_a, _b, label), violation in zip(constraints, final_violations, strict=True):
            family = label.split(":")[0]
            per_family[family] = per_family.get(family, 0.0) + float(violation.item())
        displacement = variable - anchor_tensor
        distance = {
            "l1": float(displacement.abs().sum().item()),
            "l2": float((displacement**2).sum().sqrt().item()),
            "linf": float(displacement.abs().max().item()),
            "per_dim": [float(value) for value in displacement],
        }
    converged = max_violation <= per_constraint_tolerance and total_violation <= total_tolerance
    return {
        "weights": [float(value) for value in variable],
        "converged": converged,
        "max_violation": max_violation,
        "total_violation": total_violation,
        "violation_per_constraint_family": per_family,
        "distance": distance,
        "phases": phase_summaries,
        "solver": {
            "rho_schedule": [float(rho) for rho in rho_schedule],
            "steps_per_phase": steps_per_phase,
            "lr_start": lr_start,
            "lr_end": lr_end,
            "per_constraint_tolerance": per_constraint_tolerance,
            "total_tolerance": total_tolerance,
            "constraint_count": len(constraints),
        },
    }


__all__ = [
    "LR_END",
    "LR_START",
    "PER_CONSTRAINT_TOLERANCE",
    "RHO_SCHEDULE",
    "STEPS_PER_PHASE",
    "TOTAL_TOLERANCE",
    "project_to_joint_feasible_region",
]
