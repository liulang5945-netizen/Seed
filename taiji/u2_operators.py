"""Pure tensor operators for the U2 reference subject.

The constrained quadratic follows the audited MGW1 KKT/VJP implementation.
The log-domain cursor follows the independent 22.1 reference. This module has
no dependency on verification scripts, datasets, optimizers or checkpoints.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .utf8_state import advance_utf8, utf8_allowed

CursorState = dict[str, torch.Tensor]
Emission = dict[str, torch.Tensor]


def _floating(value: torch.Tensor, name: str) -> None:
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _same_layout(reference: torch.Tensor, *values: torch.Tensor) -> None:
    if any(value.dtype != reference.dtype or value.device != reference.device for value in values):
        raise ValueError("Operator tensors must share dtype and device")


def _log_values(value: torch.Tensor, name: str) -> None:
    _floating(value, name)
    if bool(torch.isnan(value).any()) or bool(torch.isposinf(value).any()):
        raise ValueError(f"{name} contains an invalid log value")


def _tolerance(value: torch.Tensor) -> float:
    return 1e-5 if value.dtype == torch.float32 else 1e-10


def _safe_log_probability(probability: torch.Tensor) -> torch.Tensor:
    positive = probability > 0
    safe = torch.where(positive, probability, torch.ones_like(probability))
    return safe.log().masked_fill(~positive, -torch.inf)


def _safe_logsumexp(values: torch.Tensor, dim: int) -> torch.Tensor:
    """All-impossible support has zero derivative, rather than a NaN VJP."""
    supported = torch.isfinite(values).any(dim=dim, keepdim=True)
    safe = torch.where(supported, values, torch.zeros_like(values))
    result = torch.logsumexp(safe, dim=dim)
    return result.masked_fill(~supported.squeeze(dim), -torch.inf)


def _safe_logaddexp(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left, right = torch.broadcast_tensors(left, right)
    return _safe_logsumexp(torch.stack([left, right], dim=-1), -1)


class SphereQuadratic(torch.autograd.Function):
    """Solve an SPD quadratic on a Frobenius ball, including the active VJP.

    A zero radius has an explicit zero-value branch. Its radius derivative is
    not an assertion about the one-sided derivative of a learnable radius.
    """

    @staticmethod
    def forward(ctx: Any, hessian: torch.Tensor, rhs: torch.Tensor, radius: torch.Tensor):
        _floating(hessian, "hessian")
        _same_layout(hessian, rhs, radius)
        _finite(hessian, "hessian")
        _finite(rhs, "rhs")
        _finite(radius, "radius")
        if (
            hessian.ndim != 2
            or hessian.shape[0] != hessian.shape[1]
            or hessian.shape[0] == 0
            or rhs.ndim != 2
            or rhs.shape[0] != hessian.shape[0]
            or rhs.shape[1] == 0
            or radius.ndim != 0
            or float(radius) < 0
        ):
            raise ValueError("Invalid quadratic dimensions or radius")
        if not torch.allclose(hessian, hessian.T, atol=_tolerance(hessian), rtol=1e-5):
            raise ValueError("Quadratic hessian must be symmetric")
        if bool(torch.linalg.cholesky_ex(hessian).info != 0):
            raise ValueError("Quadratic hessian must be positive definite")
        ctx.zero = float(radius) == 0
        if ctx.zero:
            ctx.save_for_backward(hessian, rhs, radius)
            return torch.zeros_like(rhs)
        eye = torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device)
        result = torch.linalg.solve(hessian, rhs)
        ctx.active = bool(result.norm() > radius)
        multiplier = radius.new_zeros(())
        if ctx.active:
            low = radius.new_zeros(())
            high = torch.maximum(radius.new_tensor(1.0), rhs.norm() / radius)
            for _ in range(64):
                midpoint = (low + high) / 2
                trial = torch.linalg.solve(hessian + midpoint * eye, rhs)
                if bool(trial.norm() > radius):
                    low = midpoint
                else:
                    high = midpoint
            multiplier = high
            result = torch.linalg.solve(hessian + multiplier * eye, rhs)
        ctx.save_for_backward(hessian + multiplier * eye, result, radius)
        return result

    @staticmethod
    def backward(ctx: Any, upstream: torch.Tensor):
        if ctx.zero:
            hessian, rhs, radius = ctx.saved_tensors
            return torch.zeros_like(hessian), torch.zeros_like(rhs), torch.zeros_like(radius)
        matrix, result, radius = ctx.saved_tensors
        solved_upstream = torch.linalg.solve(matrix.T, upstream)
        radius_gradient = torch.zeros_like(radius)
        if ctx.active:
            solved_result = torch.linalg.solve(matrix, result)
            coefficient = (upstream * solved_result).sum() / (result * solved_result).sum()
            solved_upstream = solved_upstream - coefficient * solved_result
            radius_gradient = coefficient * radius
        return -solved_upstream @ result.T, solved_upstream, radius_gradient


def fp1(
    memory: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    alpha: torch.Tensor,
    gains: torch.Tensor,
    chi: torch.Tensor,
    radius: torch.Tensor,
    *,
    allowed: bool = True,
) -> torch.Tensor:
    """Edit one branch memory. Paired samples are columns of keys and values."""
    _floating(memory, "memory")
    _same_layout(memory, keys, values, alpha, gains, chi, radius)
    for name, value in (
        ("memory", memory),
        ("keys", keys),
        ("values", values),
        ("alpha", alpha),
        ("gains", gains),
        ("chi", chi),
        ("radius", radius),
    ):
        _finite(value, name)
    if (
        memory.ndim != 2
        or min(memory.shape, default=0) == 0
        or keys.ndim != 2
        or values.ndim != 2
        or keys.shape[0] != memory.shape[0]
        or values.shape[0] != memory.shape[1]
        or keys.shape[1] != values.shape[1]
        or alpha.shape != memory.shape[:1]
        or gains.shape != keys.shape[1:]
        or chi.ndim != 0
        or radius.ndim != 0
    ):
        raise ValueError("Invalid FP1 tensor dimensions")
    if (
        not 0 <= float(chi.detach()) <= 1
        or bool((alpha < 0).any())
        or bool((alpha > 1).any())
        or bool((gains < 0).any())
        or float(radius.detach()) < 0
    ):
        raise ValueError("Invalid FP1 gate or radius")
    bound = float(radius.detach())
    norm = float(memory.detach().norm())
    if (bound == 0 and norm != 0) or norm > bound + _tolerance(memory) * max(1.0, bound):
        raise ValueError("FP1 input memory lies outside its legal ball")
    if not allowed:
        return memory
    beta = chi * gains
    baseline = (1 - chi + chi * alpha)[:, None] * memory
    eye = torch.eye(memory.shape[0], dtype=memory.dtype, device=memory.device)
    hessian = eye + (keys * beta) @ keys.T
    rhs = baseline + (keys * beta) @ values.T
    return SphereQuadratic.apply(hessian, rhs, radius)


def initial_cursor(
    branches: int,
    records: int,
    positions: int,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
    log_eta: torch.Tensor | None = None,
) -> CursorState:
    if branches < 1 or records < 0 or positions < 0:
        raise ValueError("Invalid cursor dimensions")
    reference = torch.zeros((), dtype=dtype, device=device)
    _floating(reference, "cursor")
    if log_eta is None:
        eta = torch.full(
            (branches,), -torch.log(reference.new_tensor(branches)), device=device, dtype=dtype
        )
    else:
        _same_layout(reference, log_eta)
        _log_values(log_eta, "log_eta")
        if log_eta.shape != (branches,) or not torch.allclose(
            _safe_logsumexp(log_eta, 0), reference, atol=_tolerance(reference), rtol=0
        ):
            raise ValueError("Branch weights must be a normalized log distribution")
        eta = log_eta.clone()
    return {
        "log_m": reference.new_full((branches, records, positions), -torch.inf),
        "log_zeta": reference.new_zeros(branches),
        "log_eta": eta,
        "coverage": reference.new_zeros((branches, records, positions)),
    }


def _valid_positions(valid: torch.Tensor) -> None:
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("Position validity must be a two-dimensional boolean mask")
    if valid.shape[-1] > 1 and bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("Valid positions must be contiguous record prefixes")


def _validate_cursor(state: CursorState, valid: torch.Tensor) -> None:
    _valid_positions(valid)
    log_m, log_zeta, log_eta, coverage = (
        state[name] for name in ("log_m", "log_zeta", "log_eta", "coverage")
    )
    _same_layout(log_m, log_zeta, log_eta, coverage)
    if valid.device != log_m.device or log_m.ndim != 3 or log_m.shape[1:] != valid.shape:
        raise ValueError("Cursor and record dimensions differ")
    if (
        log_zeta.shape != log_m.shape[:1]
        or log_eta.shape != log_zeta.shape
        or coverage.shape != log_m.shape
    ):
        raise ValueError("Invalid cursor field dimensions")
    for name in ("log_m", "log_zeta", "log_eta"):
        _log_values(state[name], name)
    _finite(coverage, "coverage")
    if bool((coverage < 0).any()) or bool(torch.isfinite(log_m[:, ~valid]).any()):
        raise ValueError("Invalid coverage or cursor mass on padding")
    norm = _safe_logsumexp(torch.cat([log_m.flatten(1), log_zeta[:, None]], -1), -1)
    if not torch.allclose(norm, torch.zeros_like(norm), atol=_tolerance(log_m), rtol=0):
        raise ValueError("Cursor and reset masses must sum to one")
    if not torch.allclose(
        _safe_logsumexp(log_eta, 0), log_eta.new_zeros(()), atol=_tolerance(log_m), rtol=0
    ):
        raise ValueError("Branch weights must sum to one")


def record_inputs(record_scores: torch.Tensor, position_scores: torch.Tensor, valid: torch.Tensor):
    """Normalize routes only on nonempty records plus the always-available null."""
    _valid_positions(valid)
    _same_layout(record_scores, position_scores)
    _log_values(record_scores, "record_scores")
    _log_values(position_scores, "position_scores")
    if (
        record_scores.ndim != 2
        or position_scores.ndim != 3
        or position_scores.shape[0] != record_scores.shape[0]
        or position_scores.shape[1:] != valid.shape
        or record_scores.shape[1] != valid.shape[0] + 1
        or valid.device != record_scores.device
    ):
        raise ValueError("Invalid route or position score dimensions")
    record_valid = valid.any(-1)
    route_valid = torch.cat([record_valid, record_valid.new_ones(1)])
    masked_route = record_scores.masked_fill(~route_valid[None], -torch.inf)
    if not bool(torch.isfinite(_safe_logsumexp(masked_route, -1)).all()):
        raise ValueError("Every branch needs a supported route or null")
    log_alpha = masked_route.log_softmax(-1)
    rows = []
    for index in range(valid.shape[0]):
        if bool(record_valid[index]):
            masked = position_scores[:, index].masked_fill(~valid[index], -torch.inf)
            if not bool(torch.isfinite(_safe_logsumexp(masked, -1)).all()):
                raise ValueError("A supported record has no position probability")
            rows.append(masked.log_softmax(-1))
        else:
            rows.append(torch.full_like(position_scores[:, index], -torch.inf))
    log_beta = torch.stack(rows, 1) if rows else torch.full_like(position_scores, -torch.inf)
    return log_alpha, log_beta


def emission(
    state: CursorState,
    log_alpha: torch.Tensor,
    log_beta: torch.Tensor,
    lam: torch.Tensor,
    gamma: torch.Tensor,
    log_generator: torch.Tensor,
    byte_ids: torch.Tensor,
    valid: torch.Tensor,
    mask: torch.Tensor,
) -> Emission:
    """Compute the joint branch/generation/copy mixture with one global mask."""
    _validate_cursor(state, valid)
    reference = state["log_m"]
    _same_layout(reference, log_alpha, log_beta, lam, gamma, log_generator)
    branches, records, positions = reference.shape
    if (
        log_alpha.shape != (branches, records + 1)
        or log_beta.shape != reference.shape
        or lam.shape != (branches,)
        or gamma.shape != (branches,)
        or log_generator.ndim != 2
        or log_generator.shape[0] != branches
        or byte_ids.shape != (records, positions)
        or byte_ids.dtype != torch.long
        or mask.shape != log_generator.shape[1:]
        or mask.dtype != torch.bool
        or mask.device != reference.device
        or byte_ids.device != reference.device
    ):
        raise ValueError("Invalid emission dimensions or byte mask")
    for name, value in (
        ("log_alpha", log_alpha),
        ("log_beta", log_beta),
        ("log_generator", log_generator),
    ):
        _log_values(value, name)
    tolerance = _tolerance(reference)
    for name, distribution in (("log_alpha", log_alpha), ("log_generator", log_generator)):
        normalizer = _safe_logsumexp(distribution, -1)
        if not torch.allclose(normalizer, torch.zeros_like(normalizer), atol=tolerance, rtol=0):
            raise ValueError(f"{name} must be a normalized log distribution")
    record_valid = valid.any(-1)
    if bool(torch.isfinite(log_alpha[:, :-1][:, ~record_valid]).any()) or bool(
        torch.isfinite(log_beta[:, ~valid]).any()
    ):
        raise ValueError("Empty records and padding cannot carry route or position mass")
    position_normalizer = _safe_logsumexp(log_beta, -1)[:, record_valid]
    if not torch.allclose(
        position_normalizer, torch.zeros_like(position_normalizer), atol=tolerance, rtol=0
    ):
        raise ValueError("Each supported record needs a normalized position distribution")
    _finite(lam, "continuation gate")
    _finite(gamma, "generation gate")
    if bool(((lam < 0) | (lam > 1) | (gamma < 0) | (gamma > 1)).any()):
        raise ValueError("Emission gates must lie in [0,1]")
    alphabet = log_generator.shape[1]
    if bool(((byte_ids[valid] < 0) | (byte_ids[valid] >= alphabet)).any()):
        raise ValueError("Record symbol outside the declared alphabet")
    log_lam = _safe_log_probability(lam)
    log_restart = _safe_logaddexp(_safe_log_probability(1 - lam), log_lam + state["log_zeta"])
    log_anchor = log_alpha[:, :-1, None] + log_beta
    log_w = _safe_logaddexp(
        log_lam[:, None, None] + state["log_m"], log_restart[:, None, None] + log_anchor
    )
    log_q = log_alpha[:, -1] + log_restart
    log_one_minus_gamma = _safe_log_probability(1 - gamma)
    log_generate_mass = _safe_logaddexp(_safe_log_probability(gamma), log_one_minus_gamma + log_q)
    log_gen = log_generate_mass[:, None] + log_generator
    log_copy = log_one_minus_gamma[:, None, None] + log_w
    columns = []
    for symbol in range(alphabet):
        chosen = valid & (byte_ids == symbol)
        copied = _safe_logsumexp(log_copy.masked_fill(~chosen[None], -torch.inf).flatten(1), -1)
        columns.append(_safe_logaddexp(log_gen[:, symbol], copied))
    log_branch = torch.stack(columns, -1)
    log_total = _safe_logsumexp(state["log_eta"][:, None] + log_branch, 0)
    log_normalizer = _safe_logsumexp(log_total.masked_fill(~mask, -torch.inf), 0)
    if not bool(torch.isfinite(log_normalizer)):
        raise ValueError("unsupported_emit")
    return {
        "log_output": log_total.masked_fill(~mask, -torch.inf) - log_normalizer,
        "log_total": log_total,
        "log_branch": log_branch,
        "log_gen": log_gen,
        "log_copy": log_copy,
        "mask": mask,
    }


def accept(
    state: CursorState, output: Emission, symbol: int, byte_ids: torch.Tensor, valid: torch.Tensor
) -> CursorState:
    _validate_cursor(state, valid)
    if not 0 <= symbol < output["mask"].numel() or not bool(output["mask"][symbol]):
        raise ValueError("invalid_emit")
    if not bool(torch.isfinite(output["log_total"][symbol])):
        raise ValueError("unsupported_emit")
    branch = output["log_branch"][:, symbol]
    alive = torch.isfinite(branch)
    safe_branch = torch.where(alive, branch, torch.zeros_like(branch))
    chosen = valid & (byte_ids == symbol)
    log_r = (output["log_copy"] - safe_branch[:, None, None]).masked_fill(
        ~chosen[None] | ~alive[:, None, None], -torch.inf
    )
    log_gen = (output["log_gen"][:, symbol] - safe_branch).masked_fill(~alive, -torch.inf)
    log_m = torch.full_like(log_r, -torch.inf)
    if valid.shape[-1] > 1:
        log_m[:, :, 1:] = log_r[:, :, :-1].masked_fill(~valid[None, :, 1:], -torch.inf)
    terminal = valid.clone()
    if valid.shape[-1] > 1:
        terminal[:, :-1] &= ~valid[:, 1:]
    ended = _safe_logsumexp(log_r.masked_fill(~terminal[None], -torch.inf).flatten(1), -1)
    log_zeta = _safe_logaddexp(log_gen, ended)
    return {
        "log_m": log_m,
        "log_zeta": torch.where(alive, log_zeta, torch.zeros_like(log_zeta)),
        "log_eta": state["log_eta"] + branch - output["log_total"][symbol],
        "coverage": state["coverage"] + log_r.exp(),
    }


def revoke_records(state: CursorState, indices: Sequence[int]):
    """Remove cursor mass and archive coverage; caller also masks record routes."""
    records = state["log_m"].shape[1]
    selected = tuple(dict.fromkeys(int(index) for index in indices))
    if any(index < 0 or index >= records for index in selected):
        raise ValueError("Invalid revoked record index")
    log_m = state["log_m"].clone()
    removed = _safe_logsumexp(log_m[:, selected].flatten(1), -1)
    log_m[:, selected] = -torch.inf
    coverage = state["coverage"].clone()
    archived = coverage[:, selected].clone()
    coverage[:, selected] = 0
    return {
        **state,
        "log_m": log_m,
        "log_zeta": _safe_logaddexp(state["log_zeta"], removed),
        "coverage": coverage,
    }, archived


def utf8_mask(
    remaining: int, lead: int, budget: int, device: torch.device | str | None = None
) -> torch.Tensor:
    if not 0 <= remaining <= 3 or budget < 0 or not 0 <= lead <= 255:
        raise ValueError("Invalid UTF-8 state or budget")
    mask = torch.zeros(257, dtype=torch.bool, device=device)
    if budget > 0:
        for symbol in utf8_allowed(remaining, lead):
            after, _ = advance_utf8(remaining, lead, symbol)
            if after <= budget - 1:
                mask[symbol] = True
    mask[256] = remaining == 0
    return mask


def credit_matrix(deltas: torch.Tensor, inputs: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Task derivatives already include their target denominator exactly once."""
    _floating(deltas, "deltas")
    _same_layout(deltas, inputs, gates)
    for name, value in (("deltas", deltas), ("inputs", inputs), ("gates", gates)):
        _finite(value, name)
    if (
        deltas.ndim != 2
        or inputs.ndim != 2
        or gates.ndim != 2
        or inputs.shape[0] != deltas.shape[0]
        or gates.shape[0] != deltas.shape[0]
        or gates.shape[1] < 1
    ):
        raise ValueError("Credit inputs must have a shared sample axis")
    if bool((gates < 0).any()) or not torch.allclose(
        gates.sum(-1), torch.ones_like(gates[:, 0]), atol=_tolerance(gates), rtol=0
    ):
        raise ValueError("Conditional gates must form a convex partition")
    return torch.einsum("nc,nd,np->cdp", gates, deltas, inputs)


def choose_left_directions(credit: torch.Tensor, budget: int, noise: float = 0.0) -> dict[str, Any]:
    _floating(credit, "credit")
    _finite(credit, "credit")
    if credit.ndim != 2 or budget < 0 or not 0 <= noise < float("inf"):
        raise ValueError("Invalid credit matrix, budget or noise floor")
    left, singular, _ = torch.linalg.svd(credit, full_matrices=False)
    top = float(singular[0].detach()) if singular.numel() else 0.0
    threshold = max(1e-8, 1e-5 * top, noise)
    rank = int((singular > threshold).sum())
    count = min(budget, rank)
    return {
        "directions": left[:, :count],
        "singular_values": singular,
        "threshold": threshold,
        "rank": rank,
        "effective_budget": count,
    }


def append_neutral_units(
    r: torch.Tensor,
    p: torch.Tensor,
    directions: torch.Tensor,
    unit_ids: Sequence[str],
    branches: Sequence[str],
    *,
    birth_id: str,
    new_branch: str = "shared",
) -> dict[str, Any]:
    """Append neutral R/P units for one group, preserving all earlier lineage."""
    _floating(r, "R")
    _same_layout(r, p, directions)
    for name, value in (("R", r), ("P", p), ("directions", directions)):
        _finite(value, name)
    if (
        r.ndim != 2
        or p.ndim != 2
        or directions.ndim != 2
        or p.shape[1] != r.shape[0]
        or directions.shape[0] != p.shape[0]
        or len(unit_ids) != r.shape[0]
        or len(branches) != r.shape[0]
        or len(set(unit_ids)) != len(unit_ids)
    ):
        raise ValueError("Invalid parent structure or unit lineage")
    count = directions.shape[1]
    if not birth_id or not new_branch or bool((directions.norm(dim=0) == 0).any()):
        raise ValueError("Birth needs a unique identity and nonzero outgoing directions")
    newborn_ids = tuple(f"{birth_id}/{index}" for index in range(count))
    if set(newborn_ids).intersection(unit_ids):
        raise ValueError("Birth unit identity already exists")
    return {
        "r": torch.cat([r, r.new_zeros((count, r.shape[1]))], 0),
        "p": torch.cat([p, directions], 1),
        "unit_ids": tuple(unit_ids) + newborn_ids,
        "branches": tuple(branches) + (new_branch,) * count,
        "parent_units": r.shape[0],
        "units": r.shape[0] + count,
    }
