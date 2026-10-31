"""Counterexamples and coupled synthetic validation for Taiji equations v1.1.

This is a numerical equation reference, not a living model or a training runner.
It opens no corpus/checkpoint/split and constructs no optimizer. One differentiable
inner-gradient expression is evaluated solely to validate the meta derivative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

DTYPE = torch.float64


def state_norm(value: torch.Tensor, revised: bool = True) -> torch.Tensor:
    floor = 1.0 if revised else 1e-6
    return value / torch.sqrt(value.square().mean(dim=-1, keepdim=True) + floor)


def unit_ball(value: torch.Tensor) -> torch.Tensor:
    return value / torch.sqrt(1.0 + value.square().sum(dim=-1, keepdim=True))


def memory_edit(
    memory: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    channel: torch.Tensor,
    write: torch.Tensor,
    decay: torch.Tensor,
    strength: torch.Tensor,
    *,
    revised: bool = True,
    allowed: bool = True,
) -> torch.Tensor:
    if not allowed:
        return memory
    decayed = decay[:, None] * memory
    erase_key = channel * key
    if revised:
        # Exact rank-one proximal erasure; no SVD or nearly-zero normalization.
        eta = 16.0 * torch.sigmoid(strength)
        coefficient = eta / (1.0 + eta * (erase_key @ erase_key))
        erased = decayed - coefficient * torch.outer(erase_key, erase_key @ decayed)
    else:
        erased = decayed - torch.outer(key, erase_key @ decayed)
    edited = erased + torch.outer(key, write * value)
    scale = (4.0 / torch.linalg.vector_norm(edited).clamp_min(1e-12)).clamp_max(1.0)
    return scale * edited


def make_case(seed: int = 20261031, steps: int = 16, dtype: torch.dtype = DTYPE) -> dict:
    generator = torch.Generator().manual_seed(seed)

    def draw(*shape: int, scale: float = 0.15, grad: bool = False) -> torch.Tensor:
        return (torch.randn(shape, generator=generator, dtype=dtype) * scale).requires_grad_(grad)

    n, d, m, k, v = 2, 4, 3, 3, 2
    return {
        "n": n,
        "d": d,
        "h0": draw(n, d),
        "f0": draw(n, k, v),
        "events": draw(steps, d),
        "targets": draw(steps, 2),
        "records": draw(3, d),
        "goal": draw(d),
        "drive": draw(d),
        "tau_logit": torch.logit(
            (torch.tensor([1.0, 4.0], dtype=dtype) - 0.5) / 127.5
        ).requires_grad_(True),
        "encoder": draw(d, d, grad=True),
        "w": draw(n, d, 7 * d, grad=True),
        "r": draw(n, m, 7 * d, grad=True),
        "p": draw(n, d, m, grad=True),
        "message": draw(d, d),
        "q": draw(k, d),
        "key": draw(k, d),
        "value": draw(v, d),
        "fout": draw(d, v),
        "out": draw(2, d),
        "erase_bias": draw(n, k, grad=True),
        "write_bias": draw(n, v, grad=True),
        "decay_bias": draw(n, k, grad=True),
        "strength": draw(n, grad=True),
    }


def roll(
    case: dict,
    *,
    revised: bool = True,
    newborn_in: torch.Tensor | None = None,
    newborn_out: torch.Tensor | None = None,
    adapter: torch.Tensor | None = None,
    initial: tuple[torch.Tensor, torch.Tensor] | None = None,
    all_writes_blocked: bool = False,
) -> dict:
    h, memory = (case["h0"], case["f0"]) if initial is None else initial
    encoded_records = case["records"] @ case["encoder"].T
    preactivations, inputs, outputs, states = [], [], [], []
    for index, event in enumerate(case["events"]):
        event_vector = case["encoder"] @ event
        normalized = state_norm(h, revised)
        attention = torch.softmax(normalized @ normalized.T / case["d"] ** 0.5, dim=-1)
        message = attention @ normalized @ case["message"].T
        query = unit_ball((h + event_vector) @ case["q"].T)
        fast = torch.einsum("nkv,nk->nv", memory, query) @ case["fout"].T
        record_attention = torch.softmax((h + case["goal"]) @ encoded_records.T, dim=-1)
        recall = record_attention @ encoded_records
        x = torch.cat(
            [
                normalized,
                event_vector.expand_as(h),
                message,
                fast,
                recall,
                case["goal"].expand_as(h),
                case["drive"].expand_as(h),
            ],
            dim=-1,
        )
        pre = torch.einsum("ndx,nx->nd", case["w"], x)
        middle = torch.tanh(torch.einsum("nmx,nx->nm", case["r"], x))
        pre = pre + torch.einsum("ndm,nm->nd", case["p"], middle)
        additions = torch.zeros_like(pre)
        if newborn_in is not None:
            assert newborn_out is not None
            addition = newborn_out @ torch.tanh(newborn_in @ x[0])
            additions = torch.stack([addition, torch.zeros_like(addition)])
        pre = pre + additions
        if adapter is not None:
            pre = pre + h @ adapter.T
        tau = 0.5 + 127.5 * torch.sigmoid(case["tau_logit"])
        leak = (1 - torch.exp(-1 / tau))[:, None]
        h = (1 - leak) * h + leak * torch.tanh(pre)
        output = case["out"] @ h.mean(dim=0)
        preactivations.append(pre)
        inputs.append(x)
        outputs.append(output)
        states.append(h)
        keys = unit_ball((h + event_vector) @ case["key"].T)
        values = unit_ball(torch.tanh((h + event_vector) @ case["value"].T))
        rows = []
        for group in range(case["n"]):
            rows.append(
                memory_edit(
                    memory[group],
                    keys[group],
                    values[group],
                    torch.sigmoid(case["erase_bias"][group]),
                    torch.sigmoid(case["write_bias"][group]),
                    0.90 + 0.09 * torch.sigmoid(case["decay_bias"][group]),
                    case["strength"][group],
                    revised=revised,
                    allowed=not all_writes_blocked and index % 5 != 3,
                )
            )
        memory = torch.stack(rows)
    outputs_tensor = torch.stack(outputs)
    weights = torch.arange(1, len(outputs) + 1, dtype=h.dtype)
    weights = weights / weights.sum()
    per_step_loss = (outputs_tensor - case["targets"]).square().mean(dim=-1)
    return {
        "loss": (weights * per_step_loss).sum(),
        "h": h,
        "memory": memory,
        "outputs": outputs_tensor,
        "states": torch.stack(states),
        "preactivations": preactivations,
        "inputs": inputs,
    }


def validate() -> dict:
    torch.set_num_threads(1)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    checks: list[dict] = []

    def check(name: str, passed: bool, **evidence) -> None:
        checks.append({"name": name, "passed": bool(passed), **evidence})

    # All old gates are legal; no write, yet switching oblique erasures amplifies.
    root = np.sqrt(0.99)
    keys = [np.array([0.1, root]), np.array([root, 0.1])]
    channels = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    original = [
        (np.eye(2) - np.outer(k, b * k)) * 0.99 for k, b in zip(keys, channels, strict=True)
    ]
    eigenvalues, eigenvectors = np.linalg.eig(original[1] @ original[0])
    initial = np.real(eigenvectors[:, np.argmax(np.abs(eigenvalues))])
    initial /= np.linalg.norm(initial)
    old, new = initial[:, None].copy(), initial[:, None].copy()
    clips = 0
    for step in range(200):
        which = step % 2
        old = original[which] @ old
        if np.linalg.norm(old) > 16:
            clips += 1
            old *= 16 / np.linalg.norm(old)
        erase = channels[which] * keys[which]
        safe = np.eye(2) - 16 * np.outer(erase, erase) / (1 + 16 * (erase @ erase))
        new = safe @ (0.99 * new)
    counterexamples = {
        "erase_only": {
            "write": 0,
            "decay": 0.99,
            "original_two_step_spectral_radius": float(max(abs(eigenvalues))),
            "original_final_norm": float(np.linalg.norm(old)),
            "original_clip_count": clips,
            "revised_final_norm": float(np.linalg.norm(new)),
        }
    }
    check("original_erase_counterexample_reproduced", max(abs(eigenvalues)) > 1 and clips > 0)
    check("revised_same_erase_trace_no_amplification", np.linalg.norm(new) <= 1)

    random = np.random.default_rng(20261031)
    largest = 0.0
    for _ in range(200):
        k = random.normal(size=5)
        k /= max(1, np.linalg.norm(k))
        erase = random.uniform(size=5) * k
        eta = random.uniform(0, 16)
        safe = np.eye(5) - eta * np.outer(erase, erase) / (1 + eta * (erase @ erase))
        operator = safe @ np.diag(random.uniform(0, 1, size=5))
        largest = max(largest, float(np.linalg.norm(operator, ord=2)))
    check("proximal_erase_random_operator_bound", largest <= 1 + 1e-12, largest_norm=largest)

    zero = torch.zeros(4, dtype=DTYPE)
    old_j = torch.autograd.functional.jacobian(lambda x: state_norm(x, False), zero)
    new_j = torch.autograd.functional.jacobian(state_norm, zero)
    old_norm = float(torch.linalg.matrix_norm(old_j, ord=2))
    new_norm = float(torch.linalg.matrix_norm(new_j, ord=2))
    counterexamples["zero_rms_reference"] = {
        "original_epsilon": 1e-6,
        "original_jacobian": old_norm,
        "revised_jacobian": new_norm,
    }
    check("state_normalizer_zero_jacobian_bound", new_norm <= 1 and old_norm > 100)
    zero_j = torch.autograd.functional.jacobian(unit_ball, zero)
    check(
        "query_key_zero_normalization_finite",
        torch.isfinite(zero_j).all() and torch.linalg.matrix_norm(zero_j, ord=2) <= 1,
    )

    jacobian_case = make_case(seed=318, steps=1)
    zero_state = torch.zeros_like(jacobian_case["h0"])
    old_full_j = torch.autograd.functional.jacobian(
        lambda h: roll(jacobian_case, revised=False, initial=(h, jacobian_case["f0"]))["h"],
        zero_state,
    ).reshape(8, 8)
    new_full_j = torch.autograd.functional.jacobian(
        lambda h: roll(jacobian_case, initial=(h, jacobian_case["f0"]))["h"],
        zero_state,
    ).reshape(8, 8)
    old_full_norm = float(torch.linalg.matrix_norm(old_full_j, ord=2))
    new_full_norm = float(torch.linalg.matrix_norm(new_full_j, ord=2))
    check(
        "coupled_zero_state_jacobian_diagnostic",
        torch.isfinite(new_full_j).all() and new_full_norm < old_full_norm,
        original_norm=old_full_norm,
        revised_norm=new_full_norm,
        scope="One fixed synthetic case; not a whole-model contraction theorem.",
    )

    case = make_case()
    parent = roll(case)
    check(
        "coupled_window_finite_and_bounded",
        torch.isfinite(parent["loss"]) and parent["states"].abs().max() <= 1,
    )
    groups = [
        "encoder",
        "w",
        "r",
        "p",
        "tau_logit",
        "erase_bias",
        "write_bias",
        "decay_bias",
        "strength",
    ]
    gradients = torch.autograd.grad(
        parent["loss"], [case[name] for name in groups], retain_graph=True
    )
    norms = {name: float(g.norm()) for name, g in zip(groups, gradients, strict=True)}
    check(
        "task_credit_through_encoder_state_and_memory",
        all(torch.isfinite(g).all() and g.norm() > 1e-12 for g in gradients),
        gradient_norms=norms,
    )

    def shifted_loss(name: str, sign: float, step: float = 1e-5) -> float:
        changed = dict(case)
        value = case[name].detach().clone()
        value.reshape(-1)[0] += sign * step
        changed[name] = value
        return float(roll(changed)["loss"].detach())

    finite_errors = {}
    for name in ["encoder", "tau_logit", "erase_bias", "write_bias", "decay_bias", "strength"]:
        numeric = (shifted_loss(name, 1) - shifted_loss(name, -1)) / 2e-5
        analytic = float(gradients[groups.index(name)].reshape(-1)[0])
        finite_errors[name] = abs(numeric - analytic)
    check(
        "coupled_gradients_finite_difference",
        max(finite_errors.values()) < 1e-7,
        absolute_errors=finite_errors,
    )

    # Global weighted loss derivatives already contain averaging/weights.
    pre_grads = torch.autograd.grad(parent["loss"], parent["preactivations"], retain_graph=True)
    covariance = sum(
        torch.outer(g[0], x[0]) for g, x in zip(pre_grads, parent["inputs"], strict=True)
    ).detach()
    left, singular, _ = torch.linalg.svd(covariance.detach(), full_matrices=False)
    newborn_out = 0.1 * left[:, :2]
    newborn_in = torch.zeros((2, 7 * case["d"]), dtype=DTYPE, requires_grad=True)
    child = roll(case, newborn_in=newborn_in, newborn_out=newborn_out)
    check(
        "coupled_birth_neutral_outputs_and_states",
        torch.equal(parent["outputs"], child["outputs"])
        and torch.equal(parent["h"], child["h"])
        and torch.equal(parent["memory"], child["memory"]),
    )
    incoming = torch.autograd.grad(child["loss"], newborn_in)[0]
    expected = newborn_out.T @ covariance
    difference = float((incoming - expected).abs().max())
    check(
        "coupled_birth_true_aggregate_credit",
        incoming.norm() > 1e-12 and difference < 1e-10,
        error=difference,
        norm=float(incoming.norm()),
        singular_values=singular.tolist(),
    )
    wrong = expected / len(case["events"])
    counterexamples["birth_extra_averaging"] = {
        "weighted_loss_steps": len(case["events"]),
        "wrong_to_true_norm_ratio": float(wrong.norm() / expected.norm()),
    }
    check(
        "extra_mean_negative_control_detected",
        not torch.allclose(incoming, wrong, rtol=1e-2, atol=1e-12),
    )
    direction = torch.tensor(random.normal(size=newborn_in.shape), dtype=DTYPE)
    direction /= direction.norm()
    step = 1e-5
    plus = roll(case, newborn_in=step * direction, newborn_out=newborn_out)["loss"]
    minus = roll(case, newborn_in=-step * direction, newborn_out=newborn_out)["loss"]
    directional_error = abs(
        float(((plus - minus) / (2 * step)).detach()) - float((incoming * direction).sum())
    )
    check("coupled_birth_finite_difference", directional_error < 1e-7, error=directional_error)

    first_in = torch.zeros_like(newborn_in, requires_grad=True)
    first_out = newborn_out.detach().clone().requires_grad_(True)
    first_born = roll(case, newborn_in=first_in, newborn_out=first_out)
    in_grad, out_grad = torch.autograd.grad(first_born["loss"], [first_in, first_out])
    later = roll(case, newborn_in=-0.01 * in_grad.detach(), newborn_out=first_out)
    later_out_grad = torch.autograd.grad(later["loss"], first_out)[0]
    check(
        "birth_credit_transitions_from_incoming_to_outgoing",
        in_grad.norm() > 1e-12
        and torch.equal(out_grad, torch.zeros_like(out_grad))
        and later_out_grad.norm() > 1e-14,
        incoming_norm=float(in_grad.norm()),
        outgoing_at_birth=float(out_grad.norm()),
        outgoing_after_one_synthetic_incoming_expression=float(later_out_grad.norm()),
    )

    # A genuine second derivative, compared with differences of first derivatives.
    first = torch.autograd.grad(parent["loss"], case["strength"], create_graph=True)[0][0]
    second = torch.autograd.grad(first, case["strength"])[0][0]
    first_at = []
    for sign in [1, -1]:
        changed = dict(case)
        value = case["strength"].detach().clone()
        value[0] += sign * step
        value.requires_grad_(True)
        changed["strength"] = value
        first_at.append(torch.autograd.grad(roll(changed)["loss"], value)[0][0])
    second_error = abs(float(second - (first_at[0] - first_at[1]) / (2 * step)))
    check(
        "memory_gate_second_order_credit",
        torch.isfinite(second) and second_error < 1e-6,
        error=second_error,
    )

    # Differentiate through one mathematical support-gradient transformation.
    support = make_case(seed=104, steps=4)
    query_case = make_case(seed=105, steps=4)
    # Reuse parameters, change only observations/targets for the subsequent window.
    query = dict(support, events=query_case["events"], targets=query_case["targets"])

    def outer(logit: torch.Tensor) -> torch.Tensor:
        adapter = torch.zeros((4, 4), dtype=DTYPE, requires_grad=True)
        inner = roll(support, adapter=adapter)
        grad = torch.autograd.grad(inner["loss"], adapter, create_graph=True)[0]
        adapted = adapter - 0.1 * torch.sigmoid(logit) * grad
        return roll(query, adapter=adapted, initial=(inner["h"], inner["memory"]))["loss"]

    logit = torch.tensor(-0.4, dtype=DTYPE, requires_grad=True)
    meta_gradient = torch.autograd.grad(outer(logit), logit)[0]
    numeric_meta = (outer(logit.detach() + step) - outer(logit.detach() - step)) / (2 * step)
    meta_error = abs(float((meta_gradient - numeric_meta).detach()))
    check(
        "support_update_to_query_meta_credit",
        torch.isfinite(meta_gradient) and meta_gradient.abs() > 1e-12 and meta_error < 1e-7,
        gradient=float(meta_gradient),
        error=meta_error,
    )

    blocked = roll(case, all_writes_blocked=True)
    blocked_grads = torch.autograd.grad(
        blocked["loss"],
        [case["erase_bias"], case["write_bias"], case["strength"]],
        allow_unused=True,
    )
    oversized = case["f0"][0].detach() * 100.0
    key_zero = torch.zeros(3, dtype=DTYPE)
    value_zero = torch.zeros(2, dtype=DTYPE)
    capped = memory_edit(
        oversized,
        key_zero,
        value_zero,
        key_zero,
        value_zero,
        torch.ones(3, dtype=DTYPE),
        torch.tensor(0.0, dtype=DTYPE),
    )
    skipped = memory_edit(
        oversized,
        key_zero,
        value_zero,
        key_zero,
        value_zero,
        torch.ones(3, dtype=DTYPE),
        torch.tensor(0.0, dtype=DTYPE),
        allowed=False,
    )
    check(
        "all_blocked_writes_preserve_memory_and_forbid_gate_credit",
        torch.equal(blocked["memory"], case["f0"])
        and all(g is None or torch.equal(g, torch.zeros_like(g)) for g in blocked_grads)
        and torch.linalg.vector_norm(capped) <= 4.0 + 1e-12
        and torch.equal(skipped, oversized),
        allowed_projection_norm=float(torch.linalg.vector_norm(capped)),
        scope="The oversized input is a branch-control fixture, not a valid restored live state.",
    )

    stress = make_case(seed=119, steps=64)
    # No incoming event in the latter half; goals/drive/state remain present.
    stress["events"] = torch.cat([stress["events"][:32], torch.zeros_like(stress["events"][32:])])
    trace = roll(stress)
    stress_grad = torch.autograd.grad(trace["loss"], stress["w"])[0]
    check(
        "longer_coupled_and_no_new_input_trace_finite",
        torch.isfinite(trace["loss"])
        and torch.isfinite(stress_grad).all()
        and trace["states"].abs().max() <= 1,
        steps=64,
        silent_steps=32,
        max_abs_state=float(trace["states"].detach().abs().max()),
        gradient_norm=float(stress_grad.norm()),
    )

    single_precision = make_case(seed=120, steps=64, dtype=torch.float32)
    single_trace = roll(single_precision)
    single_grad = torch.autograd.grad(single_trace["loss"], single_precision["w"])[0]
    check(
        "float32_coupled_trace_and_credit_finite",
        single_trace["h"].dtype == torch.float32
        and torch.isfinite(single_trace["loss"])
        and torch.isfinite(single_grad).all()
        and single_trace["states"].abs().max() <= 1,
        dtype=str(single_trace["h"].dtype),
        gradient_norm=float(single_grad.norm()),
    )

    gamma, lam, duration = 0.99, 0.95, 32
    reward_events = [(0, -0.1), (7, -0.2), (31, 1.0)]
    reward = sum(gamma**offset * value for offset, value in reward_events)
    expected_td = reward + gamma**duration * 0.6 - 0.4
    revised_td = reward + np.exp(duration * np.log(gamma)) * 0.6 - 0.4
    original_td = reward + gamma * 0.6 - 0.4
    counterexamples["variable_duration_td"] = {
        "duration": duration,
        "old_bootstrap_discount": gamma,
        "correct_bootstrap_discount": gamma**duration,
        "old_td": original_td,
        "revised_td": revised_td,
    }
    check(
        "semi_markov_td_matches_reward_timeline",
        abs(revised_td - expected_td) < 1e-12 and abs(original_td - expected_td) > 1e-3,
    )
    previous = random.normal(size=4)
    score = random.normal(size=4)
    dense = previous.copy()
    for _ in range(duration):
        dense *= gamma * lam
    dense += score
    compact = (gamma * lam) ** duration * previous + score
    old_trace = gamma * lam * previous + score
    check(
        "duration_eligibility_matches_micro_decay",
        np.allclose(dense, compact) and not np.allclose(dense, old_trace),
    )
    terminal_td = reward + gamma**duration * 0.0 * 0.6 - 0.4
    check("terminal_bootstrap_is_zero", abs(terminal_td - (reward - 0.4)) < 1e-12)

    return {
        "schema": "taiji-core-equation-v11-coupled-validation",
        "date": "2026-10-31",
        "baseline_design": "a760ce06e",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "dtype": "float64",
        "device": "cpu",
        "original_counterexamples": counterexamples,
        "checks": checks,
        "status": (
            "passed_equation_properties_only" if all(c["passed"] for c in checks) else "failed"
        ),
        "optimizer_steps": 0,
        "synthetic_inner_update_expressions": 2,
        "data_splits_read": [],
        "cpu_seconds": time.process_time() - start_cpu,
        "wall_seconds": time.perf_counter() - start_wall,
        "limits": "No full subject, semantic/task training, lifetime behavior, actual structural benefit or checkpoint recovery is validated here.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate()
    if args.output:
        if args.output.exists():
            raise FileExistsError("equation-validation report is write-once")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": len(result["checks"]),
                "failed": [c["name"] for c in result["checks"] if not c["passed"]],
                "wall_seconds": result["wall_seconds"],
            }
        )
    )
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
