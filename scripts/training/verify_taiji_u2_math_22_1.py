"""U2 22.1 local operator checks and discriminating counterexamples.

Only synthetic tensors and event fixtures are evaluated. This is not a full
subject, optimizer run, dataset evaluation or production model replacement.
The inherited MGW1 constrained quadratic VJP and UTF-8 DFA are reused by path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

WALL_STARTED = time.monotonic()

import torch

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "plans/manifests/taiji_u2_22_1_contract_20261110_v2.json"


def canonical_sha(path):
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()


def load_source(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MGW = load_source("u2_inherited_mgw_math", "scripts/training/verify_taiji_mgw1_math.py")
UTF8 = load_source("u2_inherited_utf8", "taiji/utf8_state.py")


def tolerances(dtype):
    return (1e-10, 1e-8) if dtype == torch.float64 else (1e-5, 1e-4)


def close(actual, expected, dtype):
    atol, rtol = tolerances(dtype)
    return bool(
        torch.allclose(torch.as_tensor(actual), torch.as_tensor(expected), atol=atol, rtol=rtol)
    )


def fp1(memory, keys, values, alpha, gains, chi, radius, *, allowed=True):
    """Columns of keys/values are paired samples. Keep the allowed chi=0 graph."""
    if not allowed:
        return memory
    beta = chi * gains
    baseline = (1 - chi + chi * alpha)[:, None] * memory
    hessian = torch.eye(keys.shape[0], dtype=keys.dtype) + (keys * beta) @ keys.T
    rhs = baseline + (keys * beta) @ values.T
    return MGW.SphereQuadratic.apply(hessian, rhs, radius)


def fp1_objective(result, memory, keys, values, alpha, gains, chi):
    baseline = (1 - chi + chi * alpha)[:, None] * memory
    residual = result.T @ keys - values
    return (
        0.5 * (result - baseline).square().sum()
        + 0.5 * (residual.square() * (chi * gains)[None]).sum()
    )


def finite_differences(function, inputs):
    """Independent coordinate perturbations, fixed objective and three scales."""
    analytic = torch.autograd.grad(function(*inputs), inputs)
    dtype = inputs[0].dtype
    steps = (1e-4, 1e-5, 1e-6) if dtype == torch.float64 else (0.02, 0.01, 0.005)
    atol, rtol = tolerances(dtype)
    rows = []
    for tensor_index, value in enumerate(inputs):
        for coordinate in range(value.numel()):
            expected = float(analytic[tensor_index].flatten()[coordinate])
            estimates = []
            for step in steps:
                plus = [item.detach().clone() for item in inputs]
                minus = [item.detach().clone() for item in inputs]
                plus[tensor_index].flatten()[coordinate] += step
                minus[tensor_index].flatten()[coordinate] -= step
                estimates.append(float((function(*plus) - function(*minus)) / (2 * step)))
            errors = [abs(estimate - expected) for estimate in estimates]
            threshold = atol + rtol * abs(expected)
            rows.append(
                {
                    "input": tensor_index,
                    "coordinate": coordinate,
                    "analytic": expected,
                    "estimates": estimates,
                    "absolute_errors": errors,
                    "relative_errors": [error / max(abs(expected), atol) for error in errors],
                    "resolved": sum(error <= threshold for error in errors) >= 2,
                }
            )
    return {
        "steps": list(steps),
        "components": rows,
        "passed": all(row["resolved"] for row in rows),
    }


def log_probability(probability):
    return torch.log(probability)


def record_inputs(record_scores, position_scores, valid):
    """Empty records have no route support. Null always remains available."""
    record_valid = valid.any(-1)
    route_valid = torch.cat([record_valid, record_valid.new_ones(1)])
    log_alpha = record_scores.masked_fill(~route_valid[None], -torch.inf).log_softmax(-1)
    log_beta = torch.full_like(position_scores, -torch.inf)
    for index in range(valid.shape[0]):
        if bool(record_valid[index]):
            log_beta[:, index] = (
                position_scores[:, index].masked_fill(~valid[index], -torch.inf).log_softmax(-1)
            )
    return log_alpha, log_beta


def emission(state, log_alpha, log_beta, lam, gamma, log_generator, byte_ids, valid, mask):
    """Full latent branch/gen/copy model with one global mask normalizer."""
    log_lam = log_probability(lam)
    log_restart = torch.logaddexp(torch.log1p(-lam), log_lam + state["log_zeta"])
    log_anchor = log_alpha[:, :-1, None] + log_beta
    log_w = torch.logaddexp(
        log_lam[:, None, None] + state["log_m"], log_restart[:, None, None] + log_anchor
    )
    log_q = log_alpha[:, -1] + log_restart
    log_generate_mass = torch.logaddexp(log_probability(gamma), torch.log1p(-gamma) + log_q)
    log_gen = log_generate_mass[:, None] + log_generator
    log_copy = torch.log1p(-gamma)[:, None, None] + log_w
    by_symbol = []
    for symbol in range(log_generator.shape[-1]):
        chosen = valid & (byte_ids == symbol)
        copied = torch.logsumexp(log_copy.masked_fill(~chosen[None], -torch.inf).flatten(1), -1)
        by_symbol.append(torch.logaddexp(log_gen[:, symbol], copied))
    log_branch = torch.stack(by_symbol, -1)
    log_total = torch.logsumexp(state["log_eta"][:, None] + log_branch, 0)
    log_normalizer = torch.logsumexp(log_total.masked_fill(~mask, -torch.inf), 0)
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


def accept(state, output, symbol, byte_ids, valid):
    if not bool(output["mask"][symbol]):
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
    ended = torch.logsumexp(log_r.masked_fill(~terminal[None], -torch.inf).flatten(1), -1)
    log_zeta = torch.logaddexp(log_gen, ended)
    log_zeta = torch.where(alive, log_zeta, torch.zeros_like(log_zeta))
    return {
        "log_m": log_m,
        "log_zeta": log_zeta,
        "log_eta": state["log_eta"] + branch - output["log_total"][symbol],
        "coverage": state["coverage"] + log_r.exp(),
    }


def initial_cursor(branches, records, positions, dtype):
    return {
        "log_m": torch.full((branches, records, positions), -torch.inf, dtype=dtype),
        "log_zeta": torch.zeros(branches, dtype=dtype),
        "log_eta": torch.full((branches,), -math.log(branches), dtype=dtype),
        "coverage": torch.zeros((branches, records, positions), dtype=dtype),
    }


def revoke_records(state, indices):
    log_m = state["log_m"].clone()
    removed = torch.logsumexp(log_m[:, indices].flatten(1), -1)
    log_m[:, indices] = -torch.inf
    coverage = state["coverage"].clone()
    archived = coverage[:, indices].clone()
    coverage[:, indices] = 0
    return {
        **state,
        "log_m": log_m,
        "log_zeta": torch.logaddexp(state["log_zeta"], removed),
        "coverage": coverage,
    }, archived


def utf8_mask(remaining, lead, budget):
    mask = torch.zeros(257, dtype=torch.bool)
    if budget == 0:
        if remaining == 0:
            mask[256] = True
        return mask
    for symbol in UTF8.utf8_allowed(remaining, lead):
        after, _ = UTF8.advance_utf8(remaining, lead, symbol)
        if after <= budget - 1:
            mask[symbol] = True
    mask[256] = remaining == 0
    return mask


def packet_admission(packet, *, world, process, source, scope, seen):
    times = [packet[key] for key in ("start", "end", "arrival")] + [world, process]
    if not all(math.isfinite(value) for value in times) or packet["start"] > packet["end"]:
        return "invalid_time"
    if packet["id"] in seen:
        return "duplicate"
    if not packet["valid"]:
        return "invalid_observation"
    if packet["source"] != source or packet["scope"] != scope or not packet["allowed"]:
        return "unauthorized"
    if packet["end"] > world or packet["arrival"] > process:
        return "not_arrived"
    return "admitted"


def event_transition(state, kind, *, elapsed=0.0, evidence=None):
    """Small event-contract fixture, not a controller for a running subject."""
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid_elapsed")
    child = {**state, "journal": list(state["journal"])}
    if kind == "elapsed":
        child["world_time"] += elapsed
        child["position"] += elapsed * state["velocity"]
    elif kind == "late_observation":
        child["journal"].append({"history_shadow": evidence})
    elif kind in {"replay", "imagine", "self_text", "grow", "consolidate"}:
        child["journal"].append({"internal": kind})
    else:
        raise ValueError("unregistered_event")
    return child


def settle(receipt, outcome, values, settled):
    if outcome["receipt_id"] != receipt["id"]:
        raise ValueError("receipt_mismatch")
    if receipt["id"] in settled:
        return None
    if not outcome["known"]:
        return None
    old_goal = receipt["goal"]
    delta = (
        outcome["utility"]
        + receipt["discount"] * (1 - outcome["terminal"]) * values[old_goal]
        - receipt["old_value"]
    )
    settled.add(receipt["id"])
    return delta


def check_read_derivative(dtype):
    generator = torch.Generator().manual_seed(22101)
    d, rank = 2, 3
    x = torch.randn(8 * d, dtype=dtype, generator=generator) * 0.2
    w = torch.randn(d, 8 * d, dtype=dtype, generator=generator) * 0.2
    r = torch.randn(rank, 8 * d, dtype=dtype, generator=generator) * 0.2
    p = torch.randn(d, rank, dtype=dtype, generator=generator)
    block = slice(5 * d, 6 * d)
    w[:, block] = 0  # Only the nonlinear consumer can carry this fixture's read credit.
    incoming = torch.randn(d, dtype=dtype, generator=generator, requires_grad=True)
    upstream = torch.tensor([0.7, -0.4], dtype=dtype)
    h = torch.tensor([0.1, -0.2], dtype=dtype)
    leak = torch.tensor([0.4, 0.6], dtype=dtype)

    def objective(value):
        inputs = torch.cat([x[: block.start], value, x[block.stop :]])
        pre = w @ inputs + p @ torch.tanh(r @ inputs)
        return (((1 - leak) * h + leak * pre.tanh()) * upstream).sum()

    actual = torch.autograd.grad(objective(incoming), incoming, retain_graph=True)[0]
    inputs = torch.cat([x[: block.start], incoming, x[block.stop :]])
    pre = w @ inputs + p @ torch.tanh(r @ inputs)
    delta = leak * (1 - pre.tanh().square()) * upstream
    analytic = w[:, block].T @ delta + r[:, block].T @ (
        (1 - (r @ inputs).tanh().square()) * (p.T @ delta)
    )
    assert close(actual, analytic, dtype)
    assert not close(actual, w[:, block].T @ delta, dtype)
    fd = finite_differences(objective, [incoming])
    assert fd["passed"]
    # Zero output layer preserves the parent. Its upstream is initially inactive.
    z = torch.tensor([0.3, -0.5], dtype=dtype, requires_grad=True)
    last = torch.zeros((d, d), dtype=dtype, requires_grad=True)
    bias = torch.zeros(d, dtype=dtype, requires_grad=True)
    output_loss = objective(last @ z + bias)
    grad_last, grad_bias, grad_z = torch.autograd.grad(output_loss, [last, bias, z])
    assert float(grad_last.norm()) > 0 and float(grad_bias.norm()) > 0
    assert float(grad_z.norm()) == 0
    keep = torch.cat([torch.arange(block.start), torch.arange(block.stop, 8 * d)])
    old_pre = w[:, keep] @ x[keep] + p @ torch.tanh(r[:, keep] @ x[keep])
    parent_loss = (((1 - leak) * h + leak * old_pre.tanh()) * upstream).sum()
    assert close(objective(torch.zeros(d, dtype=dtype)), parent_loss, dtype)
    assert not close(objective(incoming), parent_loss, dtype)
    r[:, block] = 0
    dead = torch.autograd.grad(objective(incoming), incoming)[0]
    assert float(dead.norm()) == 0
    return {
        "full_derivative": actual.detach().tolist(),
        "finite_difference": fd,
        "nonlinear_omission_rejected": True,
        "double_zero_detected": True,
    }


def check_fp1(dtype):
    def scalar(value):
        return torch.tensor(value, dtype=dtype)

    chi_values = []
    for point in (0.0, 0.25, 0.5, 1.0):
        chi = scalar(point).requires_grad_()
        x = fp1(
            scalar([[0.2]]),
            scalar([[1.0]]),
            scalar([[1.0]]),
            scalar([0.8]),
            scalar([1.0]),
            chi,
            scalar(10.0),
        )
        gradient = torch.autograd.grad(x.sum(), chi)[0]
        expected = scalar(0.76 / (1 + point) ** 2)
        assert close(gradient, expected, dtype)
        chi_values.append({"chi": point, "value": float(x.detach()), "derivative": float(gradient)})
    zero = scalar(0.0).requires_grad_()
    identity = scalar([[0.2]])
    assert (
        fp1(
            identity,
            scalar([[1.0]]),
            scalar([[1.0]]),
            scalar([0.8]),
            scalar([1.0]),
            zero,
            scalar(1.0),
            allowed=False,
        )
        is identity
    )
    right = (
        fp1(
            identity,
            scalar([[1.0]]),
            scalar([[1.0]]),
            scalar([0.8]),
            scalar([1.0]),
            scalar(1e-4),
            scalar(10.0),
        )
        - identity
    ) / 1e-4
    assert abs(float(right) - 0.76) < (0.001 if dtype == torch.float32 else 1e-4)
    old_output = MGW.fp1(
        identity,
        scalar([[1.0]]),
        scalar([[1.0]]),
        zero.reshape(1),
        scalar([0.8]),
        zero,
        scalar(10.0),
    )
    old_gradient = (
        torch.autograd.grad(old_output.sum(), zero, allow_unused=True)[0]
        if old_output.requires_grad
        else None
    )
    old_value = 0.0 if old_gradient is None else float(old_gradient)
    assert not close(scalar(old_value), right, dtype)
    inputs = [
        scalar([[0.03, -0.02], [0.01, 0.04]]).requires_grad_(),
        scalar([[0.7, -0.2], [0.3, 0.8]]).requires_grad_(),
        scalar([[2.0, -0.5], [0.3, 1.5]]).requires_grad_(),
        scalar([0.6, 0.8]).requires_grad_(),
        scalar([1.1, 0.8]).requires_grad_(),
        scalar(0.6).requires_grad_(),
        scalar(0.18).requires_grad_(),
    ]
    read = scalar([[0.4, -0.3], [0.1, 0.7]])

    def objective(*args):
        return (fp1(*args) * read).sum()

    fd = finite_differences(objective, inputs)
    assert fd["passed"], fd
    memory, keys, values, alpha, gains, chi, radius = inputs
    result = fp1(*inputs)
    beta = chi * gains
    a = torch.eye(2, dtype=dtype) + (keys * beta) @ keys.T
    b = (1 - chi + chi * alpha)[:, None] * memory + (keys * beta) @ values.T
    mu = (result * (b - a @ result)).sum() / result.square().sum()
    assert float(mu.detach()) > 0 and close(a @ result + mu * result, b, dtype)
    detached_result = torch.linalg.solve(a + mu.detach() * torch.eye(2, dtype=dtype), b)
    assert close(detached_result, result, dtype)
    wrong_gradients = torch.autograd.grad((detached_result * read).sum(), inputs, allow_unused=True)
    wrong_radius = 0.0 if wrong_gradients[-1] is None else float(wrong_gradients[-1])
    correct_radius = next(row["analytic"] for row in fd["components"] if row["input"] == 6)
    assert not close(scalar(wrong_radius), scalar(correct_radius), dtype)
    assert close(result.norm(), radius, dtype)
    assert close(
        result, fp1(memory, keys.flip(1), values.flip(1), alpha, gains.flip(0), chi, radius), dtype
    )
    zeros = fp1(torch.zeros_like(memory), keys, values, alpha, gains, chi, scalar(0.0))
    assert float(zeros.norm().detach()) == 0
    k = torch.diag(scalar([1.0, 2.0]))
    v = scalar([[3.0, 3.0]])
    exact = fp1(
        scalar([[0.0], [0.0]]),
        k,
        v,
        scalar([1.0, 1.0]),
        scalar([1.0, 1.0]),
        scalar(1.0),
        scalar(math.sqrt(2)),
    )
    ridge = torch.linalg.solve(torch.eye(2, dtype=dtype) + k @ k.T, k @ v.T)
    radial = ridge * math.sqrt(2) / ridge.norm()
    exact_j = fp1_objective(
        exact, torch.zeros_like(exact), k, v, scalar([1.0, 1.0]), scalar([1.0, 1.0]), scalar(1.0)
    )
    radial_j = fp1_objective(
        radial, torch.zeros_like(exact), k, v, scalar([1.0, 1.0]), scalar([1.0, 1.0]), scalar(1.0)
    )
    assert close(exact_j, scalar(3.5), dtype) and float((radial_j - exact_j).detach()) > 0.05
    return {
        "chi_curve": chi_values,
        "active_multiplier": float(mu.detach()),
        "active_finite_difference": fd,
        "exact_objective": float(exact_j.detach()),
        "radial_objective": float(radial_j.detach()),
        "early_return_derivative": old_value,
        "early_return_derivative_mutant_rejected": not close(scalar(old_value), right, dtype),
        "detached_mu_radius_gradient": wrong_radius,
        "correct_radius_gradient": correct_radius,
    }


def check_residuals(dtype):
    generator = torch.Generator().manual_seed(22102)
    errors = []
    rtol = 1e-10 if dtype == torch.float64 else 1e-5
    for _ in range(24):
        k = torch.randn((3, 5), generator=generator, dtype=dtype)
        k[2] = k[0] + k[1]
        v = torch.randn((2, 5), generator=generator, dtype=dtype)
        u, sigma, vh = torch.linalg.svd(k, full_matrices=False)
        keep = sigma > rtol * sigma[0]
        truncated = (u[:, keep] * sigma[keep]) @ vh[keep]
        inverse = (vh[keep].T / sigma[keep]) @ u[:, keep].T
        x_min = inverse.T @ v.T
        comp = (v @ (torch.eye(5, dtype=dtype) - inverse @ truncated)).norm()
        x = torch.randn((3, 2), generator=generator, dtype=dtype)
        radius = float(x.norm())
        lhs = (x.T @ k - v).square().sum()
        truncated_lhs = (x.T @ truncated - v).square().sum()
        decomposition = ((x - x_min).T @ truncated).square().sum() + comp.square()
        assert close(truncated_lhs, decomposition, dtype)
        bound = (
            comp.square() + sigma[keep][-1].square() * max(0.0, float(x_min.norm()) - radius) ** 2
        )
        raw_bound = (
            bound.sqrt() - radius * torch.linalg.matrix_norm(k - truncated, ord=2)
        ).clamp_min(0)
        assert float(lhs.sqrt() - raw_bound) >= -tolerances(dtype)[0]
        errors.append(float(abs(truncated_lhs - decomposition)))
    k = torch.zeros((2, 3), dtype=dtype)
    v = torch.ones((1, 3), dtype=dtype)
    assert close((torch.ones((1, 2), dtype=dtype) @ k - v).norm(), v.norm(), dtype)
    conflict_k = torch.tensor([[1.0, 0.0]], dtype=dtype)
    conflict_v = torch.tensor([[2.0, 3.0]], dtype=dtype)
    before = float((conflict_v[:, :1].clamp(-1, 1) @ conflict_k - conflict_v).norm())
    after = float((conflict_v[:, :1].clamp(-2, 2) @ conflict_k - conflict_v).norm())
    assert after < before
    near_k = torch.diag(torch.tensor([1.0, 1e-7], dtype=dtype))
    near_v = torch.tensor([[0.0, 1.0]], dtype=dtype)
    rho_exact = torch.linalg.vector_norm(near_v / torch.diag(near_k))
    wrong_bound = float((rho_exact - 1).square())
    feasible_residual = float(near_v.square().sum())
    true_bound = float((torch.diag(near_k)[-1] * (rho_exact - 1)).square())
    assert wrong_bound > feasible_residual * 1e10
    assert true_bound <= feasible_residual + tolerances(dtype)[0]
    return {
        "rank_deficient_cases": len(errors),
        "max_decomposition_error": max(errors),
        "zero_key_checked": True,
        "capacity_with_structural_floor": [before, after],
        "mixed_cutoff_false_squared_bound": wrong_bound,
        "consistent_squared_bound": true_bound,
    }


def enumerate_latent_paths(joint, alpha, beta, lam, gamma, generators, byte_rows, lengths, symbol):
    """Independent probability path enumeration over (branch, cursor or reset)."""
    outputs = [0.0] * len(generators[0])
    children = {}

    def add(branch, cursor, emitted, mass):
        outputs[emitted] += mass
        if emitted == symbol:
            key = (branch, cursor)
            children[key] = children.get(key, 0.0) + mass

    for (branch, cursor), mass in joint.items():
        continuation = lam[branch] if cursor is not None else 0.0
        restart = 1 - continuation
        gen_mass = gamma[branch] + (1 - gamma[branch]) * restart * alpha[branch][-1]
        for emitted, probability in enumerate(generators[branch]):
            add(branch, None, emitted, mass * gen_mass * probability)
        if cursor is not None:
            record, position = cursor
            next_cursor = (record, position + 1) if position + 1 < lengths[record] else None
            add(
                branch,
                next_cursor,
                byte_rows[record][position],
                mass * (1 - gamma[branch]) * continuation,
            )
        for record, length in enumerate(lengths):
            for position in range(length):
                next_cursor = (record, position + 1) if position + 1 < length else None
                source_mass = (
                    mass
                    * (1 - gamma[branch])
                    * restart
                    * alpha[branch][record]
                    * beta[branch][record][position]
                )
                add(branch, next_cursor, byte_rows[record][position], source_mass)
    total = outputs[symbol]
    return outputs, {key: value / total for key, value in children.items()}


def check_sequence(dtype):
    generator = torch.Generator().manual_seed(22103)
    valid = torch.tensor([[True, True, True], [True, True, False]])
    ids = torch.tensor([[0, 0, 1], [0, 2, 0]])
    alpha, beta = record_inputs(
        torch.randn((2, 3), dtype=dtype, generator=generator),
        torch.randn((2, 2, 3), dtype=dtype, generator=generator),
        valid,
    )
    state = initial_cursor(2, 2, 3, dtype)
    omega = state["log_eta"].clone()
    lam = torch.tensor([0.7, 0.8], dtype=dtype)
    gamma = torch.tensor([0.3, 0.4], dtype=dtype)
    gen = torch.randn((2, 3), dtype=dtype, generator=generator).log_softmax(-1)
    errors = []
    joint = {(branch, None): 0.5 for branch in range(2)}
    enumeration_errors = []
    for index, symbol in enumerate([0, 0, 1, 2] * 16):
        out = emission(
            state, alpha, beta, lam, gamma, gen, ids, valid, torch.ones(3, dtype=torch.bool)
        )
        assert close(out["log_output"].exp().sum(), torch.tensor(1.0, dtype=dtype), dtype)
        if index < 8:
            independent, joint = enumerate_latent_paths(
                joint,
                alpha.exp().tolist(),
                beta.exp().tolist(),
                lam.tolist(),
                gamma.tolist(),
                gen.exp().tolist(),
                ids.tolist(),
                valid.sum(-1).tolist(),
                symbol,
            )
            assert close(out["log_output"].exp(), torch.tensor(independent, dtype=dtype), dtype)
        state = accept(state, out, symbol, ids, valid)
        if index < 8:
            branch_mass = [
                sum(value for (b, _), value in joint.items() if b == branch) for branch in range(2)
            ]
            difference = (
                (state["log_eta"].exp() - torch.tensor(branch_mass, dtype=dtype)).abs().max()
            )
            assert close(state["log_eta"].exp(), torch.tensor(branch_mass, dtype=dtype), dtype)
            for branch in range(2):
                for record in range(2):
                    for position in range(int(valid[record].sum())):
                        expected_mass = (
                            joint.get((branch, (record, position)), 0.0) / branch_mass[branch]
                        )
                        assert close(
                            state["log_m"][branch, record, position].exp(),
                            torch.tensor(expected_mass, dtype=dtype),
                            dtype,
                        )
                expected_reset = joint.get((branch, None), 0.0) / branch_mass[branch]
                assert close(
                    state["log_zeta"][branch].exp(),
                    torch.tensor(expected_reset, dtype=dtype),
                    dtype,
                )
            enumeration_errors.append(float(difference))
        norm = torch.logsumexp(
            torch.cat([state["log_m"].flatten(1), state["log_zeta"][:, None]], -1), -1
        )
        assert close(norm, torch.zeros_like(norm), dtype)
        errors.append(float(norm.abs().max()))
    assert torch.equal(omega, initial_cursor(2, 2, 3, dtype)["log_eta"])
    # A branch-local renormalization has a measurably different posterior.
    empty = torch.empty((0, 0), dtype=torch.bool)
    s = initial_cursor(2, 0, 0, dtype)
    a = torch.zeros((2, 1), dtype=dtype)
    b = torch.empty((2, 0, 0), dtype=dtype)
    out = emission(
        s,
        a,
        b,
        torch.ones(2, dtype=dtype),
        torch.ones(2, dtype=dtype),
        torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=dtype).log(),
        torch.empty((0, 0), dtype=torch.long),
        empty,
        torch.tensor([True, False]),
    )
    child = accept(s, out, 0, torch.empty((0, 0), dtype=torch.long), empty)
    assert close(child["log_eta"].exp(), torch.tensor([0.9, 0.1], dtype=dtype), dtype)
    raw_generators = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=dtype)
    wrongly_normalized = raw_generators[:, 0] / raw_generators[:, :1].sum(-1)
    wrong_eta = s["log_eta"].exp() * wrongly_normalized
    wrong_eta = wrong_eta / wrong_eta.sum()
    assert not close(child["log_eta"].exp(), wrong_eta, dtype)
    try:
        accept(s, out, 1, torch.empty((0, 0), dtype=torch.long), empty)
    except ValueError as error:
        assert str(error) == "invalid_emit"
    else:
        raise AssertionError("mask violation was accepted")
    # One record's end cannot shift into the following record.
    one = initial_cursor(1, 2, 3, dtype)
    one["log_m"][0, 0, 2] = 0.0
    one["log_zeta"][:] = -torch.inf
    out = emission(
        one,
        alpha[:1],
        beta[:1],
        torch.ones(1, dtype=dtype),
        torch.zeros(1, dtype=dtype),
        gen[:1],
        ids,
        valid,
        torch.ones(3, dtype=torch.bool),
    )
    child = accept(one, out, 1, ids, valid)
    assert not bool(torch.isfinite(child["log_m"]).any()) and float(child["log_zeta"][0]) == 0
    # Rare restart stays representable in log space after mass rounds to one.
    rare = initial_cursor(1, 1, 2, dtype)
    rare["log_m"][0, 0, 1] = 0.0
    rare["log_zeta"][0] = -1000.0
    rare_valid = torch.ones((1, 2), dtype=torch.bool)
    rare_ids = torch.tensor([[0, 1]])
    aa = torch.tensor([[0.0, -torch.inf]], dtype=dtype)
    bb = torch.tensor([[[0.0, -torch.inf]]], dtype=dtype)
    out = emission(
        rare,
        aa,
        bb,
        torch.ones(1, dtype=dtype),
        torch.zeros(1, dtype=dtype),
        torch.tensor([[-torch.inf, 0.0]], dtype=dtype),
        rare_ids,
        rare_valid,
        torch.ones(2, dtype=torch.bool),
    )
    assert abs(float(out["log_output"][0]) + 1000) < tolerances(dtype)[0]
    reconstructed = 1 - rare["log_m"].exp().sum()
    assert float(reconstructed) == 0
    wrong_restart = emission(
        {**rare, "log_zeta": reconstructed.reshape(1).log()},
        aa,
        bb,
        torch.ones(1, dtype=dtype),
        torch.zeros(1, dtype=dtype),
        torch.tensor([[-torch.inf, 0.0]], dtype=dtype),
        rare_ids,
        rare_valid,
        torch.ones(2, dtype=torch.bool),
    )
    assert not bool(torch.isfinite(wrong_restart["log_output"][0]))
    revoked, _ = revoke_records(rare, [0])
    assert not bool(torch.isfinite(revoked["log_m"]).any())
    assert close(revoked["log_zeta"], torch.zeros(1, dtype=dtype), dtype)
    empty_valid = torch.zeros((1, 2), dtype=torch.bool)
    empty_alpha, empty_beta = record_inputs(
        torch.zeros((1, 2), dtype=dtype), torch.zeros((1, 1, 2), dtype=dtype), empty_valid
    )
    assert float(empty_alpha[0, -1]) == 0 and not bool(torch.isfinite(empty_beta).any())
    # An old-positive branch can assign zero probability to the accepted byte.
    no_records = initial_cursor(2, 0, 0, dtype)
    dead_output = emission(
        no_records,
        a,
        b,
        torch.ones(2, dtype=dtype),
        torch.ones(2, dtype=dtype),
        torch.tensor([[0.0, -torch.inf], [-torch.inf, 0.0]], dtype=dtype),
        torch.empty((0, 0), dtype=torch.long),
        empty,
        torch.ones(2, dtype=torch.bool),
    )
    dead_child = accept(no_records, dead_output, 0, torch.empty((0, 0), dtype=torch.long), empty)
    assert not bool(torch.isfinite(dead_child["log_eta"][1]))
    assert float(dead_child["log_zeta"][1]) == 0
    return {
        "recursive_steps": len(errors),
        "max_log_mass_error": max(errors),
        "rare_restart_log_probability": float(out["log_output"][0]),
        "subtraction_mutant_rejected": True,
        "global_mask_mutant_rejected": True,
        "independent_latent_path_steps": len(enumeration_errors),
        "max_latent_branch_probability_error": max(enumeration_errors),
    }


def check_utf8(dtype):
    raw = "放玉".encode()
    valid = torch.ones((1, len(raw)), dtype=torch.bool)
    ids = torch.tensor([list(raw)])
    state = initial_cursor(2, 1, len(raw), dtype)
    alpha = torch.tensor([[0.0, -torch.inf], [0.0, -torch.inf]], dtype=dtype)
    beta = torch.full((2, 1, len(raw)), -torch.inf, dtype=dtype)
    beta[:, 0, 0] = 0
    remaining, lead = 0, 0
    generators = torch.full((2, 257), -math.log(257), dtype=dtype)
    for index, symbol in enumerate(raw):
        mask = utf8_mask(remaining, lead, len(raw) - index)
        assert bool(mask[symbol])
        out = emission(
            state,
            alpha,
            beta,
            torch.ones(2, dtype=dtype),
            torch.full((2,), 0.1, dtype=dtype),
            generators,
            ids,
            valid,
            mask,
        )
        state = accept(state, out, symbol, ids, valid)
        remaining, lead = UTF8.advance_utf8(remaining, lead, symbol)
    assert remaining == 0 and bool(utf8_mask(0, 0, 0)[256])
    assert not bool(utf8_mask(1, 0xE6, 0).any())
    assert not bool(utf8_mask(0, 0, 1)[0xE6])
    assert not bool(utf8_mask(2, 0xED, 3)[0xA0])
    assert not bool(utf8_mask(3, 0xF4, 4)[0x90])
    return {
        "alphabet": 257,
        "accepted_bytes": len(raw),
        "incomplete_character_eos_rejected": True,
        "remaining_budget_checked": True,
    }


def check_events():
    packet = {
        "id": "p0",
        "source": "sensor",
        "scope": "world0",
        "allowed": True,
        "valid": True,
        "start": 4.0,
        "end": 5.0,
        "arrival": 12.0,
    }
    kwargs = {"world": 10.0, "process": 10.0, "source": "sensor", "scope": "world0", "seen": set()}
    assert packet_admission(packet, **kwargs) == "not_arrived"
    assert packet_admission({**packet, "arrival": 8.0}, **kwargs) == "admitted"
    assert (
        packet_admission({**packet, "arrival": 8.0, "source": "wrong"}, **kwargs) == "unauthorized"
    )
    assert packet_admission({**packet, "start": float("nan")}, **kwargs) == "invalid_time"
    assert packet_admission({**packet, "start": 6.0}, **kwargs) == "invalid_time"
    assert packet_admission(packet, **{**kwargs, "world": float("nan")}) == "invalid_time"
    assert packet_admission({**packet, "valid": False}, **kwargs) == "invalid_observation"
    assert packet_admission({**packet, "arrival": 8.0}, **{**kwargs, "seen": {"p0"}}) == "duplicate"
    parent = {
        "world_time": 10.0,
        "position": 3.0,
        "velocity": 2.0,
        "world_weights": (0.4, 0.6),
        "journal": [],
    }
    for kind in ("replay", "imagine", "self_text", "grow", "consolidate", "late_observation"):
        child = event_transition(parent, kind, elapsed=50.0, evidence={"past_position": -4.0})
        assert child["world_time"] == 10.0 and child["position"] == 3.0
        assert child["world_weights"] == parent["world_weights"]
    child = event_transition(parent, "elapsed", elapsed=0.5)
    assert child["world_time"] == 10.5 and child["position"] == 4.0
    for invalid in (-1.0, float("nan"), float("inf")):
        try:
            event_transition(parent, "elapsed", elapsed=invalid)
        except ValueError as error:
            assert str(error) == "invalid_elapsed"
        else:
            raise AssertionError("invalid elapsed time was accepted")
    receipt = {"id": "r0", "goal": "old", "discount": 0.9, "old_value": 0.0}
    outcome = {"id": "o0", "receipt_id": "r0", "known": True, "utility": 0.0, "terminal": False}
    settled = set()
    result = settle(receipt, outcome, {"old": 1.0, "new": -1.0}, settled)
    assert result == 0.9 and result != -0.9
    assert settle(receipt, outcome, {"old": 1.0}, settled) is None
    assert settle(receipt, {**outcome, "id": "o1", "known": False}, {"old": 1.0}, settled) is None
    assert settle(receipt, {**outcome, "terminal": True}, {"old": 1.0}, set()) == 0.0
    assert settle(receipt, {**outcome, "id": "duplicate_other_id"}, {"old": 1.0}, settled) is None
    try:
        settle(receipt, {**outcome, "receipt_id": "wrong"}, {"old": 1.0}, set())
    except ValueError as error:
        assert str(error) == "receipt_mismatch"
    else:
        raise AssertionError("wrong receipt was accepted")
    return {
        "future_arrival_rejected": True,
        "internal_world_unchanged": True,
        "old_goal_delta": result,
        "duplicate_settlements": 0,
        "authority_is_not_truth": "admission does not assign truth status",
    }


def check_growth(dtype):
    credit = torch.tensor([[0.0, 0.0, 3.0], [0.0, 2.0, 0.0]], dtype=dtype)
    left, sigma, vh = torch.linalg.svd(credit, full_matrices=False)
    direction, mutant = left[:, 0], vh[:, 0]
    assert close(direction.norm(), torch.tensor(1.0, dtype=dtype), dtype)
    assert close((direction @ credit).norm(), sigma[0], dtype)
    assert float(mutant.norm()) == 0
    assert close(credit @ credit.T @ direction, sigma[0].square() * direction, dtype)
    c = torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype))
    b = (
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ],
            dtype=dtype,
        )
        / 2
    )
    gradients = b.T @ c
    assert int(torch.linalg.matrix_rank(gradients)) == 1
    assert int((gradients.norm(dim=1) > 1e-8).sum()) == 4
    # Rank-deficient top-left-SVD initialization has exact dead directions.
    top, _, _ = torch.linalg.svd(c)
    assert int(((top.T @ c).norm(dim=1) > 1e-8).sum()) == 1
    # Normalized soft gates can duplicate one sample across two branches.
    gate = torch.tensor([0.5, 0.5], dtype=dtype)
    assert close(gate.sum(), torch.tensor(1.0, dtype=dtype), dtype)
    soft_rank_sum = sum(int(torch.linalg.matrix_rank(weight.reshape(1, 1))) for weight in gate)
    assert soft_rank_sum == 2
    j = torch.eye(2, dtype=dtype)
    residual = torch.ones(2, dtype=dtype)
    rtol = 1e-10 if dtype == torch.float64 else 1e-5
    projected = residual - j @ torch.linalg.pinv(j, rtol=rtol) @ residual
    assert float(projected.norm()) == 0
    constrained_residual = residual * (1 - 0.1 / float(residual.norm()))
    assert float(constrained_residual.norm()) > 1
    # Legal replay can leave rank unchanged or reduce aggregate rank.
    duplicate = torch.ones((2, 2), dtype=dtype)
    assert (
        int(torch.linalg.matrix_rank(duplicate))
        == int(torch.linalg.matrix_rank(10 * duplicate))
        == 1
    )
    before = torch.eye(2, dtype=dtype)
    u = torch.ones(2, dtype=dtype) / math.sqrt(2)
    after = before - u[:, None] @ u[None]
    assert int(torch.linalg.matrix_rank(after, rtol=1e-5)) == 1
    x = torch.tensor(
        [[1.0, 0.0, 1.0], [-1.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.5, 0.5, 1.0]], dtype=dtype
    )
    delta = torch.tensor([[0.4, -0.2], [-0.1, 0.6], [0.7, 0.3], [-0.3, 0.2]], dtype=dtype)
    gates = torch.tensor([0.2, 0.8, 0.4, 0.7], dtype=dtype)
    outgoing = torch.tensor([[0.3, -0.2], [0.6, 0.4]], dtype=dtype)
    newborn = torch.zeros((2, 3), dtype=dtype, requires_grad=True)

    def loss(incoming):
        prediction = delta + gates[:, None] * (torch.tanh(x @ incoming.T) @ outgoing.T)
        return prediction.square().sum() / (2 * len(x))

    assert torch.equal(delta + gates[:, None] * (torch.tanh(x @ newborn.T) @ outgoing.T), delta)
    actual = torch.autograd.grad(loss(newborn), newborn)[0]
    expected = outgoing.T @ ((delta * gates[:, None]).T @ x / len(x))
    assert close(actual, expected, dtype)
    assert not close(-expected, actual, dtype)
    birth_fd = finite_differences(loss, [newborn])
    assert birth_fd["passed"], birth_fd
    outgoing_leaf = outgoing.detach().clone().requires_grad_()
    neutral_loss = (
        delta + gates[:, None] * (torch.tanh(x @ newborn.T) @ outgoing_leaf.T)
    ).square().sum() / (2 * len(x))
    assert float(torch.autograd.grad(neutral_loss, outgoing_leaf)[0].norm()) == 0
    dead_loss = (delta + torch.tanh(x @ newborn.T) @ torch.zeros_like(outgoing).T).square().sum()
    assert float(torch.autograd.grad(dead_loss, newborn)[0].norm()) == 0
    cancellation_x = torch.tensor(
        [[-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0]], dtype=dtype
    )
    cancellation_delta = -(cancellation_x[:, 0] * cancellation_x[:, 1])[:, None]
    aligned = (cancellation_x[:, 0] > 0).to(dtype)
    global_credit = cancellation_delta.T @ cancellation_x / 4
    branches = [
        ((cancellation_delta * g[:, None]).T @ cancellation_x / 4) for g in (aligned, 1 - aligned)
    ]
    assert float(global_credit.norm()) == 0 and all(float(c.norm()) > 0 for c in branches)
    # A content-blind random gate can still generate a large nonzero sample credit.
    blind = torch.tensor([0.1, 0.8, 0.4, 0.9], dtype=dtype)
    blind_credit = (cancellation_delta * blind[:, None]).T @ cancellation_x / 4
    assert float(blind_credit.norm()) > 0
    direction_fit = branches[0].flatten() / branches[0].norm()
    aligned_check = branches[0] * 1.3
    permuted_gate = aligned.roll(1)
    permuted_check = (cancellation_delta * permuted_gate[:, None]).T @ cancellation_x / 4
    aligned_cosine = float(direction_fit @ (aligned_check.flatten() / aligned_check.norm()))
    permuted_cosine = float(direction_fit @ (permuted_check.flatten() / permuted_check.norm()))
    assert aligned_cosine > 0.9 and permuted_cosine < 0.1
    return {
        "correct_startup_norm": float((direction @ credit).norm()),
        "legacy_vh_startup_norm": float((mutant @ credit).norm()),
        "rank_one_live_units": 4,
        "soft_rank_sum_one_sample": soft_rank_sum,
        "zero_tangent_residual_is_not_growth_veto": True,
        "startup_finite_difference": birth_fd,
        "content_blind_sample_credit": float(blind_credit.norm()),
        "aligned_check_cosine": aligned_cosine,
        "permuted_check_cosine": permuted_cosine,
    }


def run_checks():
    rows = []
    for dtype in (torch.float64, torch.float32):
        for name, check in (
            ("read_R", check_read_derivative),
            ("FP1", check_fp1),
            ("residuals", check_residuals),
            ("sequence", check_sequence),
            ("utf8", check_utf8),
            ("growth", check_growth),
        ):
            rows.append(
                {"name": name, "dtype": str(dtype), "passed": True, "evidence": check(dtype)}
            )
    rows.append({"name": "events", "passed": True, "evidence": check_events()})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    target = args.report.resolve() if args.report else None
    if target is not None and (target.parent != ROOT / "reports" or target.exists()):
        parser.error("A report must be a new file directly inside reports/.")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for path, expected in contract["source_pins"].items():
        actual = canonical_sha(ROOT / path)
        if actual != expected:
            raise RuntimeError(f"Contract source changed: {path}")
    torch.set_num_threads(1)
    cpu_started, wall_started = time.process_time(), time.monotonic()
    checks = run_checks()
    actual_groups = sorted((row["name"], row.get("dtype", "none")) for row in checks)
    expected_groups = sorted(tuple(row) for row in contract["expected_groups"])
    assert actual_groups == expected_groups, (actual_groups, expected_groups)
    check_cpu, check_wall = time.process_time() - cpu_started, time.monotonic() - wall_started
    cpu, wall = time.process_time(), time.monotonic() - WALL_STARTED
    assert cpu <= contract["budget"]["cpu_seconds"] and wall <= contract["budget"]["wall_seconds"]
    report = {
        "schema": "taiji-u2-22.1-local-math-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "contract_sha256": canonical_sha(CONTRACT),
        "verdict": "passed_local_math_only",
        "checks": checks,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "check_cpu_seconds": check_cpu,
        "check_wall_seconds": check_wall,
        "timer_scope": "process CPU from process start; wall before torch import",
        "python": sys.version,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "training_steps": 0,
        "optimizer_steps": 0,
        "project_dataset_reads": 0,
        "checkpoints_loaded": 0,
        "full_subject_verified": False,
        "coverage": contract["coverage"],
    }
    if target is not None:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "groups": len(checks),
                "cpu_seconds": cpu,
                "wall_seconds": wall,
                "report": str(target) if target else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
