"""Algorithm §10.2 constrained feedback updates on the R021 reference graph.

The update solves the declared trust-region problem over the flattened parameter
vector: minimize ``1/2||Delta + eta g_new||^2`` subject to first-order old-loss
constraints and a norm ball. It is an engineering instrument for R029; no task
capability, growth benefit or training admission is implied.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from taiji.living_core import LivingCoreReference
from taiji.state_migration import CareerJournal, run_career

TOLERANCE = 1e-12
KKT_TOLERANCE = 1e-8


@dataclass(frozen=True)
class ConstraintDiagnostics:
    delta_norm: float
    objective: float
    kkt_residual: float
    constraint_values: list[float]
    constraint_slacks: list[float]
    active_constraints: tuple[int, ...]
    ball_active: bool
    mu: float
    multipliers: list[float]
    complementarity_residual: float
    feasible: bool


def career_nll(model: LivingCoreReference, journal: CareerJournal, provider=None) -> torch.Tensor:
    _, steps = run_career(model, journal, growth_gate_provider=provider, retain_outputs=False)
    terms = [probabilities[byte] for probabilities, byte in steps if byte is not None]
    return -torch.stack(terms).log().mean()


def flatten(tensors) -> torch.Tensor:
    return torch.cat([tensor.detach().reshape(-1) for tensor in tensors])


def gradient_vector(loss: torch.Tensor, parameters) -> torch.Tensor:
    values = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1) if value is None else value.detach().reshape(-1)
            for parameter, value in zip(parameters, values, strict=True)
        ]
    )


def apply_delta(parameters, delta: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.add_(delta[offset : offset + count].view_as(parameter))
            offset += count


def _feasible(delta, constraints, epsilons, rho):
    if delta.numel() == 0:
        return True
    norm = float(delta.norm())
    if norm > rho + 1e-10:
        return False
    if constraints.shape[0] == 0:
        return True
    values = constraints @ delta
    return bool((values <= epsilons + 1e-10).all())


def _objective(delta, c):
    return float(0.5 * (delta + c).square().sum())


def _kkt(delta, c, constraints, lambdas, mu):
    residual = delta + c
    if constraints.shape[0]:
        residual = residual + constraints.transpose(0, 1) @ lambdas
    if mu != 0.0:
        residual = residual + mu * delta
    return float(residual.norm())


def _ball_only(c, rho):
    norm_c = float(c.norm())
    if norm_c <= rho:
        delta = -c.clone()
        mu = 0.0
    else:
        delta = -rho * c / c.norm()
        mu = norm_c / rho - 1.0
    return delta, mu


def _quadratic_roots(a: float, b: float, c: float):
    if abs(a) <= 1e-14:
        if abs(b) <= 1e-14:
            return []
        root = -c / b
        return [root] if root >= 1.0 else []
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return []
    sqrt_d = math.sqrt(discriminant)
    roots = [(-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)]
    return [root for root in roots if root >= 1.0]


def solve_constrained_step(
    c: torch.Tensor,
    constraints: torch.Tensor,
    epsilons: torch.Tensor,
    rho: float,
) -> tuple[torch.Tensor, ConstraintDiagnostics]:
    m = constraints.shape[0]
    best = None

    def consider(delta, lambdas, mu, active):
        nonlocal best
        if not _feasible(delta, constraints, epsilons, rho):
            return
        kkt = _kkt(delta, c, constraints, lambdas, mu)
        if kkt > KKT_TOLERANCE:
            return
        objective = _objective(delta, c)
        candidate = (objective, delta, lambdas, mu, active, kkt)
        if best is None or candidate[0] < best[0] - TOLERANCE:
            best = candidate

    delta, mu = _ball_only(c, rho)
    zeros = torch.zeros(m, dtype=c.dtype)
    consider(delta, zeros, mu, ())

    if m:
        constraint_c = constraints @ c
        for mask in range(1, 1 << m):
            active = tuple(index for index in range(m) if mask & (1 << index))
            rows = constraints[list(active)]
            eps_active = epsilons[list(active)]
            gram = rows @ rows.transpose(0, 1)
            gram_inverse = torch.linalg.pinv(gram, rcond=1e-12)
            if float((gram_inverse @ gram @ gram_inverse - gram_inverse).abs().max()) > 1e-8:
                continue

            alpha = gram_inverse @ (eps_active + constraint_c[list(active)])
            lambdas_active = -alpha
            if float(lambdas_active.min()) < -1e-10:
                continue
            delta = -c + rows.transpose(0, 1) @ alpha
            lambdas = torch.zeros(m, dtype=c.dtype)
            lambdas[list(active)] = lambdas_active
            consider(delta.clone(), lambdas, 0.0, active)

            projected_eps = gram_inverse @ eps_active
            projected_c = gram_inverse @ constraint_c[list(active)]
            base = c - rows.transpose(0, 1) @ projected_c
            slope = -rows.transpose(0, 1) @ projected_eps
            roots = _quadratic_roots(
                float(slope.dot(slope)) - rho * rho,
                2.0 * float(base.dot(slope)),
                float(base.dot(base)),
            )
            for t in roots:
                lambdas_active = -t * projected_eps - projected_c
                if float(lambdas_active.min()) < -1e-10:
                    continue
                y = base + t * slope
                delta = -y / t
                lambdas = torch.zeros(m, dtype=c.dtype)
                lambdas[list(active)] = lambdas_active
                consider(delta.clone(), lambdas, t - 1.0, active)

    if best is None:
        delta = torch.zeros_like(c)
        values = (constraints @ delta).tolist() if m else []
        return delta, ConstraintDiagnostics(
            delta_norm=0.0,
            objective=_objective(delta, c),
            kkt_residual=float((delta + c).norm()),
            constraint_values=values,
            constraint_slacks=[e - v for e, v in zip(epsilons.tolist(), values, strict=True)],
            active_constraints=(),
            ball_active=False,
            mu=0.0,
            multipliers=[],
            complementarity_residual=0.0,
            feasible=_feasible(delta, constraints, epsilons, rho),
        )

    objective, delta, lambdas, mu, active, kkt = best
    values = (constraints @ delta).tolist() if m else []
    slacks = [e - v for e, v in zip(epsilons.tolist(), values, strict=True)]
    complementarity = max(
        [
            abs(lambda_i * (v - e))
            for lambda_i, v, e in zip(lambdas.tolist(), values, epsilons.tolist(), strict=True)
        ]
        + [abs(mu * (float(delta.norm()) - rho))]
    )
    return delta, ConstraintDiagnostics(
        delta_norm=float(delta.norm()),
        objective=objective,
        kkt_residual=kkt,
        constraint_values=values,
        constraint_slacks=slacks,
        active_constraints=active,
        ball_active=mu > TOLERANCE,
        mu=mu,
        multipliers=lambdas.tolist(),
        complementarity_residual=complementarity,
        feasible=True,
    )


def constrained_feedback_step(
    model: LivingCoreReference,
    fit_journal: CareerJournal,
    old_journals: tuple[CareerJournal, ...],
    *,
    eta: float = 1.0,
    epsilon: float = 2e-3,
    rho: float = 2e-2,
    provider=None,
):
    parameters = [p for p in model.parameters() if p.requires_grad]
    timings: dict[str, float] = {}
    mark = time.process_time()
    fit_loss = career_nll(model, fit_journal, provider)
    new_gradient = gradient_vector(fit_loss, parameters)
    timings["fit_gradient"] = time.process_time() - mark
    c = eta * new_gradient
    fit_loss_value = float(fit_loss.detach())
    del fit_loss, new_gradient

    rows = []
    old_losses = []
    mark = time.process_time()
    for journal in old_journals:
        loss = career_nll(model, journal, provider)
        rows.append(gradient_vector(loss, parameters))
        old_losses.append(float(loss.detach()))
        del loss
    timings["old_gradients"] = time.process_time() - mark
    constraints = torch.stack(rows) if rows else torch.zeros((0, c.numel()), dtype=c.dtype)
    del rows
    epsilons = torch.full((constraints.shape[0],), epsilon, dtype=c.dtype)
    keep = [index for index in range(constraints.shape[0]) if float(constraints[index].norm()) > 0]
    if keep:
        constraints = constraints[keep]
        epsilons = epsilons[keep]
    else:
        constraints = torch.zeros((0, c.numel()), dtype=c.dtype)
        epsilons = torch.zeros(0, dtype=c.dtype)

    mark = time.process_time()
    delta, diagnostics = solve_constrained_step(c, constraints, epsilons, rho)
    timings["solve"] = time.process_time() - mark
    mark = time.process_time()
    apply_delta(parameters, delta)
    timings["apply"] = time.process_time() - mark
    del delta, constraints, epsilons
    return {
        "fit_loss_before": fit_loss_value,
        "old_losses_before": old_losses,
        "eta": eta,
        "epsilon": epsilon,
        "rho": rho,
        "gradient_norm": float(c.norm() / eta if eta else 0.0),
        "diagnostics": diagnostics,
        "timings": timings,
    }


def clone_parameters(parameters):
    return [parameter.detach().clone() for parameter in parameters]


def restore_parameters(parameters, snapshots) -> None:
    with torch.no_grad():
        for parameter, snapshot in zip(parameters, snapshots, strict=True):
            parameter.copy_(snapshot)


def evaluate_old_losses(model, old_journals, provider=None) -> list[float]:
    with torch.no_grad():
        return [float(career_nll(model, journal, provider).detach()) for journal in old_journals]


def constrained_feedback_updates(
    model: LivingCoreReference,
    fit_journal: CareerJournal,
    old_journals: tuple[CareerJournal, ...],
    *,
    steps: int = 8,
    eta: float = 1.0,
    epsilon: float = 2e-3,
    rho: float = 2e-2,
    actual_epsilon: float = 3.75e-3,
    max_retries: int = 4,
    provider=None,
):
    """First-order constrained step plus the declared actual-probe rollback."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    current_rho = rho
    rows = []
    for _ in range(steps):
        accepted = False
        attempt = 0
        row: dict = {}
        while attempt <= max_retries and not accepted:
            snapshot = clone_parameters(parameters)
            mark = time.process_time()
            before = evaluate_old_losses(model, old_journals, provider)
            attempt_timings = {"evaluate_before": time.process_time() - mark}
            row = constrained_feedback_step(
                model,
                fit_journal,
                old_journals,
                eta=eta,
                epsilon=epsilon,
                rho=current_rho,
                provider=provider,
            )
            mark = time.process_time()
            after = evaluate_old_losses(model, old_journals, provider)
            attempt_timings["evaluate_after"] = time.process_time() - mark
            attempt_timings.update(row["timings"])
            actual = [abs(a - b) for a, b in zip(after, before, strict=True)]
            row.update(
                {
                    "old_losses_after": after,
                    "actual_old_abs_change": actual,
                    "actual_old_max_abs_change": max(actual) if actual else 0.0,
                    "actual_epsilon": actual_epsilon,
                    "attempts": attempt + 1,
                    "rho_used": current_rho,
                    "accepted": False,
                    "timings": attempt_timings,
                }
            )
            if not actual or max(actual) <= actual_epsilon:
                row["accepted"] = True
                accepted = True
            else:
                restore_parameters(parameters, snapshot)
                current_rho *= 0.5
                attempt += 1
        rows.append(row)
        if not accepted:
            break
    return rows
