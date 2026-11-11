"""R022 credit and numerical discrimination oracles on the R021 formal reference graph.

Synthetic fixture only. Declared responsible targets, finite-difference agreement with
an eps sweep, second directional derivatives, zero/sign/circuit-break negative controls,
float64/float32 comparison and a declared two-step synthetic Adam update expression.
No corpus partition, task evaluation, pricing, growth or capability claim. Contract:
plans/reference/SPEC-TAIJI-CORE_CREDIT_R022_20261031.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import torch

from taiji.living_core import EOS, LivingCoreConfig, LivingCoreReference, RawSource

CONFIG = LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)
PROMPT = b"prompt"
GOAL = b"goal"
TARGET = b"ab"
RECORDS = (
    RawSource("s1", b"abc target ab", version=1, observed_at=0),
    RawSource("s2", b"secondary ab", version=2, observed_at=4),
)
RAW_NU = (0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3)
KNOWN_NU = (True, True, False, True, True, True, False, True)
NOW = 10
WORLD_SEED = 20261031
WORLD_REWARD_TARGET = 0.3
SUBJECT = "r022-credit"

ACTIVITY_FLOOR = 1e-10
EPS_LIST = (1e-4, 1e-5, 1e-6)
FD_ATOL = 1e-8
FD_RTOL = 1e-3
DIRECTIONAL_H = 1e-5
SECOND_H = 3e-4
SECOND_ATOL = 1e-6
SECOND_RTOL = 5e-2
CONTROL_FLOOR = 1e-10
CONTROL_RELATIVE = 1e-8
COORD_LIMIT = 12
FP32_RESOLVED = 1e-6
FP32_TOLERANCE = 0.1
UPDATE_LR = 1e-2
UPDATE_STEPS = 2
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8

OWNER_PREFIXES = (
    ("I1", ("encoder",)),
    ("I2", ("nu_projection",)),
    ("I3", ("episodic",)),
    ("I4", ("message_q", "message_k", "message_v", "message_goal")),
    ("F_READ", ("fast_queries", "fast_outputs")),
    ("I5", ("xi", "readout")),
    ("I6", ("expression",)),
    ("DYNAMICS", ("W", "U", "V", "R", "P", "tau_logit")),
    ("F_WRITE", ("write_heads",)),
    ("WORLD_FORWARD", ("world",)),
)
TARGETS = ("T_EXPR", "T_WRITE2", "T_WORLD_PROXY")
CREDITED_STATUSES = ("credited", "proxy_credited", "birth_zero", "inert_shift_invariant")
# episodic.score.bias (spec I3 b_3) is provably inert: the positional softmax is
# invariant to a shared shift of all scores, so this parameter cannot affect any
# output for any value. Declared zero class in the R022 contract; flagged for a
# future R021 interface revision.
INERT_PARAMETERS = ("episodic.score.bias",)
INERT_FLOOR = 1e-14

BACKWARD_CALLS = 0
_CACHE: dict | None = None


def fixture(dtype=torch.float64):
    torch.set_num_threads(1)
    model = LivingCoreReference(CONFIG).to(dtype)
    state = model.initial_state(SUBJECT)
    state = replace(
        state,
        h=torch.linspace(-0.4, 0.6, 16, dtype=dtype).reshape(2, 8),
        fast=torch.linspace(-0.2, 0.3, 64, dtype=dtype).reshape(2, 2, 4, 4),
    )
    kwargs = {
        "goal": GOAL,
        "records": RECORDS,
        "raw_nu": torch.tensor(RAW_NU, dtype=dtype),
        "known_nu": torch.tensor(KNOWN_NU),
        "now": NOW,
        "allowed": torch.ones((CONFIG.groups, CONFIG.fast_heads), dtype=torch.bool),
    }
    return model, state, kwargs


def expr_loss(model, state, kwargs):
    distributions, final = model.score_targets(state, prompt=PROMPT, target=TARGET, **kwargs)
    tokens = (*TARGET, EOS)
    total = -torch.stack([distributions[i, b] for i, b in enumerate(tokens)]).log().sum()
    return total, distributions, final


def write2_loss(model, state, kwargs):
    event = model.encoder.patches(b"seed")[0]
    first = model.step(state, event=event, previous_byte=EOS, **kwargs)
    event = model.encoder.patches(b"next")[0]
    second = model.step(first.state, event=event, previous_byte=TARGET[0], **kwargs)
    return -second.probabilities[TARGET[1]].log(), first, second


def world_targets(dtype):
    generator = torch.Generator().manual_seed(WORLD_SEED)
    return (torch.randn(CONFIG.width, generator=generator) * 0.1).to(dtype), WORLD_REWARD_TARGET


def world_loss(model, state, kwargs):
    event = model.encoder.patches(PROMPT)[0]
    output = model.step(state, event=event, previous_byte=EOS, **kwargs)
    state_target, reward_target = world_targets(model.W.dtype)
    state_error = (output.prediction - state_target).square().sum(-1)
    reward_error = (output.predicted_reward - reward_target).square()
    return (state_error + reward_error).sum(), output


def loss_evaluator(model, state, kwargs, target):
    def evaluate():
        with torch.no_grad():
            if target == "T_EXPR":
                return expr_loss(model, state, kwargs)[0]
            if target == "T_WRITE2":
                return write2_loss(model, state, kwargs)[0]
            return world_loss(model, state, kwargs)[0]

    return evaluate


def collect_grads(loss, named):
    global BACKWARD_CALLS
    BACKWARD_CALLS += 1
    names = [name for name, _ in named]
    collected = torch.autograd.grad(loss, [param for _, param in named], allow_unused=True)
    return {
        name: torch.zeros_like(param) if grad is None else grad.detach().clone()
        for name, param, grad in zip(names, [param for _, param in named], collected, strict=True)
    }


def owner_target(name: str) -> str:
    family = name.split(".")[0]
    if family == "world":
        return "T_WORLD_PROXY"
    if family == "write_heads":
        return "T_WRITE2"
    return "T_EXPR"


def forward_owner(name: str) -> str:
    for identity, prefixes in OWNER_PREFIXES:
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            return identity
    raise AssertionError(f"Parameter {name} has no forward owner")


def target_losses_and_grads(model, state, kwargs):
    named = list(model.named_parameters())
    world_named = [(n, p) for n, p in named if n.startswith("world.")]
    expr_value, distributions, final = expr_loss(model, state, kwargs)
    write_value, first, second = write2_loss(model, state, kwargs)
    world_value, _ = world_loss(model, state, kwargs)
    losses = {"T_EXPR": expr_value, "T_WRITE2": write_value, "T_WORLD_PROXY": world_value}
    grads = {
        "T_EXPR": collect_grads(expr_value, named),
        "T_WRITE2": collect_grads(write_value, named),
        "T_WORLD_PROXY": collect_grads(world_value, world_named),
    }
    grads["T_WORLD_PROXY"] = {
        name: grads["T_WORLD_PROXY"].get(name, torch.zeros_like(param)) for name, param in named
    }
    limit = CONFIG.fast_limit
    margins = [
        limit - float(first.state.fast.detach().norm()),
        limit - float(second.state.fast.detach().norm()),
        limit - float(final.fast.detach().norm()),
    ]
    return losses, grads, distributions, min(margins)


def coordinate_indices(numel: int, gradient: torch.Tensor) -> list[int]:
    if numel <= COORD_LIMIT:
        return list(range(numel))
    picks = torch.linspace(0, numel - 1, COORD_LIMIT).round().long().tolist()
    return sorted({int(index) for index in picks} | {int(gradient.abs().argmax())})


def finite_difference_sweep(model, state, kwargs, named, gradients, target):
    evaluate = loss_evaluator(model, state, kwargs, target)
    base = {name: param.detach().clone() for name, param in named}
    summary = {}
    for name, param in named:
        flat = param.view(-1)
        saved = flat.detach().clone()
        gradient = gradients[name].reshape(-1)
        per_eps = {}
        for eps in EPS_LIST:
            worst, failures, checked = 0.0, 0, 0
            for index in coordinate_indices(param.numel(), gradient):
                automatic = float(gradient[index])
                step = eps * max(1.0, abs(float(saved[index])))
                with torch.no_grad():
                    flat.copy_(saved)
                    flat[index] += step
                plus = float(evaluate())
                with torch.no_grad():
                    flat.copy_(saved)
                    flat[index] -= step
                minus = float(evaluate())
                with torch.no_grad():
                    flat.copy_(saved)
                finite = (plus - minus) / (2 * step)
                difference = abs(automatic - finite)
                tolerance = FD_ATOL + FD_RTOL * (abs(automatic) + abs(finite))
                checked += 1
                failures += int(difference > tolerance)
                worst = max(worst, difference)
            per_eps[f"{eps:g}"] = {"ok": failures == 0, "max_mismatch": worst, "coords": checked}
        summary[name] = {
            "per_eps": per_eps,
            "best_mismatch": min(row["max_mismatch"] for row in per_eps.values()),
            "gate_ok": sum(row["ok"] for row in per_eps.values()) >= 2,
        }
    with torch.no_grad():
        for name, param in named:
            param.copy_(base[name])
    return summary


def directional_sweep(model, state, kwargs, named, gradients, target):
    evaluate = loss_evaluator(model, state, kwargs, target)
    generator = torch.Generator().manual_seed(WORLD_SEED)
    base = {name: param.detach().clone() for name, param in named}
    results = {}
    for name, param in named:
        gradient = gradients[name].reshape(-1)
        if float(gradient.norm()) == 0.0:
            results[name] = {"status": "skipped_exact_zero"}
            continue
        direction = torch.randn(param.numel(), generator=generator, dtype=param.dtype)
        direction = direction / direction.norm()
        flat = param.view(-1)
        saved = flat.detach().clone()
        shifted = direction * DIRECTIONAL_H
        with torch.no_grad():
            flat.copy_(saved)
            flat.add_(shifted)
        plus = float(evaluate())
        with torch.no_grad():
            flat.copy_(saved)
            flat.sub_(shifted)
        minus = float(evaluate())
        with torch.no_grad():
            flat.copy_(saved)
        finite = (plus - minus) / (2 * DIRECTIONAL_H)
        automatic = float((gradient * direction).sum())
        difference = abs(automatic - finite)
        tolerance = FD_ATOL + FD_RTOL * (abs(automatic) + abs(finite))
        results[name] = {
            "status": "checked",
            "automatic": automatic,
            "finite": finite,
            "difference": difference,
            "ok": bool(difference <= tolerance),
        }
    with torch.no_grad():
        for name, param in named:
            param.copy_(base[name])
    return results


def second_derivative_sweep(model, state, kwargs, named, target):
    evaluate = loss_evaluator(model, state, kwargs, target)
    if target == "T_EXPR":
        graphed = expr_loss(model, state, kwargs)[0]
    elif target == "T_WRITE2":
        graphed = write2_loss(model, state, kwargs)[0]
    else:
        graphed = world_loss(model, state, kwargs)[0]
    global BACKWARD_CALLS
    params = [param for _, param in named]
    first = torch.autograd.grad(graphed, params, create_graph=True, allow_unused=True)
    BACKWARD_CALLS += 1
    automatics, directions = {}, {}
    for (name, param), grad in zip(named, first, strict=True):
        if grad is None or float(grad.detach().abs().max()) == 0.0:
            automatics[name] = None
            continue
        direction = (grad.detach() / grad.detach().norm()).reshape(-1)
        dot = (grad.reshape(-1) * direction).sum()
        row = torch.autograd.grad(dot, param, retain_graph=True)[0]
        BACKWARD_CALLS += 1
        automatics[name] = float((row.reshape(-1) * direction).sum())
        directions[name] = direction
    del first, graphed, params
    base = {name: param.detach().clone() for name, param in named}
    base_value = float(evaluate())
    results = {}
    for name, param in named:
        if name not in directions:
            results[name] = {"status": "skipped_exact_zero"}
            continue
        direction = directions[name]
        flat = param.view(-1)
        saved = flat.detach().clone()
        shifted = direction * SECOND_H
        with torch.no_grad():
            flat.copy_(saved)
            flat.add_(shifted)
        plus = float(evaluate())
        with torch.no_grad():
            flat.copy_(saved)
            flat.sub_(shifted)
        minus = float(evaluate())
        with torch.no_grad():
            flat.copy_(saved)
        finite = (plus - 2 * base_value + minus) / SECOND_H**2
        automatic = automatics[name]
        difference = abs(automatic - finite)
        tolerance = SECOND_ATOL + SECOND_RTOL * (abs(automatic) + abs(finite))
        results[name] = {
            "status": "checked",
            "automatic": automatic,
            "finite": finite,
            "difference": difference,
            "ok": bool(difference <= tolerance),
        }
    with torch.no_grad():
        for name, param in named:
            param.copy_(base[name])
    return results


def blocking_probes(model, state, kwargs, named):
    write_names = [name for name, _ in named if name.startswith("write_heads.")]
    # Probe events are data inputs reused across separate backward passes; detach so a
    # freed probe graph is never extended. Encoder credit is still probed through the
    # goal/source paths inside each step.
    event = model.encoder.patches(b"probe")[0].detach()
    single_loss = (
        -model.step(state, event=event, previous_byte=EOS, **kwargs).probabilities[TARGET[0]].log()
    )
    single = collect_grads(single_loss, named)
    ordering = {
        "write_heads_exact_zero": all(float(single[n].abs().max()) == 0.0 for n in write_names),
    }
    blocked_kwargs = dict(
        kwargs, allowed=torch.zeros((CONFIG.groups, CONFIG.fast_heads), dtype=torch.bool)
    )
    blocked = collect_grads(write2_loss(model, state, blocked_kwargs)[0], named)
    positive = collect_grads(write2_loss(model, state, kwargs)[0], named)
    permission = {
        "write_heads_exact_zero": all(float(blocked[n].abs().max()) == 0.0 for n in write_names),
        "positive_control_nonzero": all(
            float(positive[n].abs().max()) > ACTIVITY_FLOOR for n in write_names
        ),
    }
    empty_kwargs = dict(kwargs, records=())
    empty_loss = (
        -model.step(state, event=event, previous_byte=EOS, **empty_kwargs)
        .probabilities[TARGET[0]]
        .log()
    )
    empty = collect_grads(empty_loss, named)
    # episodic.null_value stays live through the x-block even with an empty view; the
    # declared cut covers the singleton-softmax score path and the discarded copy path.
    cut = [
        "episodic.a.weight",
        "episodic.h.weight",
        "episodic.g.weight",
        "episodic.meta.weight",
        "episodic.meta.bias",
        "episodic.second.weight",
        "episodic.second.bias",
        "episodic.score.weight",
        "episodic.score.bias",
        "episodic.value.weight",
        "episodic.value.bias",
        "episodic.value_meta.weight",
        "episodic.null_key",
        "expression.a.weight",
        "expression.meta.weight",
        "expression.position.weight",
        "expression.coverage",
        "expression.score.weight",
        "expression.mix.weight",
        "expression.mix.bias",
    ]
    null_view = {"cut_exact_zero": all(float(empty[n].abs().max()) == 0.0 for n in cut)}
    first_step = model.step(state, event=event, previous_byte=EOS, **kwargs)
    conditioned_first = collect_grads(-first_step.probabilities[TARGET[0]].log(), named)
    consumed = model.consume_output(first_step, TARGET[0])
    # The coverage/prior recursion is differentiable inside score_targets; probe steps
    # backward between rounds, so the conditioning state is detached before reuse.
    detached = replace(
        consumed,
        h=consumed.h.detach(),
        fast=consumed.fast.detach(),
        nu_ema=consumed.nu_ema.detach(),
        coverage=consumed.coverage.detach(),
        prior=consumed.prior.detach(),
    )
    second_step = model.step(
        detached,
        event=model.encoder.patches(bytes([TARGET[0]]))[0],
        previous_byte=TARGET[0],
        **kwargs,
    )
    conditioned_second = collect_grads(-second_step.probabilities[TARGET[1]].log(), named)
    state_conditioned = {
        "coverage_step1_exact_zero": float(conditioned_first["expression.coverage"].abs().max())
        == 0.0,
        "continuation_step1_exact_zero": all(
            float(conditioned_first[f"expression.continuation.{part}"].abs().max()) == 0.0
            for part in ("weight", "bias")
        ),
        "coverage_step2_nonzero": float(conditioned_second["expression.coverage"].abs().max())
        > ACTIVITY_FLOOR,
        "continuation_step2_nonzero": all(
            float(conditioned_second[f"expression.continuation.{part}"].abs().max())
            > ACTIVITY_FLOOR
            for part in ("weight", "bias")
        ),
    }
    return {
        "ordering": ordering,
        "permission": permission,
        "null_view": null_view,
        "state_conditioned": state_conditioned,
    }


def inert_parameter_probe(model, state, kwargs, named, gradients):
    name = INERT_PARAMETERS[0]
    param = dict(named)[name]
    saved = param.detach().clone()
    evaluator = loss_evaluator(model, state, kwargs, "T_EXPR")
    base = float(evaluator())
    generator = torch.Generator().manual_seed(WORLD_SEED)
    replacement = torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.5
    with torch.no_grad():
        param.copy_(replacement)
    replaced = float(evaluator())
    with torch.no_grad():
        param.copy_(saved)
    threshold = max(CONTROL_FLOOR, CONTROL_RELATIVE * abs(base))
    return {
        "name": name,
        "gradient_max": float(gradients["T_EXPR"][name].abs().max()),
        "gradient_below_inert_floor": float(gradients["T_EXPR"][name].abs().max()) <= INERT_FLOOR,
        "value_replacement_change": abs(base - replaced),
        "value_replacement_no_effect": abs(base - replaced) <= threshold,
    }


def participation_controls(model, state, kwargs, named, gradients):
    evaluators = {target: loss_evaluator(model, state, kwargs, target) for target in TARGETS}
    base_values = {target: float(evaluators[target]()) for target in TARGETS}
    families: dict[str, list[str]] = {}
    for name, _ in named:
        families.setdefault(name.split(".")[0], []).append(name)
    picks = []
    for family, names in families.items():
        target = owner_target(names[0])
        pick = max(names, key=lambda n: float(gradients[target][n].abs().max()))
        picks.append((family, target, pick))
    for extra in ("W", "U", "tau_logit", "V"):
        if extra not in {pick[2] for pick in picks}:
            picks.append(("DYNAMICS", "T_EXPR", extra))
    lookup = dict(named)
    rows = []
    for family, target, name in sorted(picks):
        param = lookup[name]
        saved = param.detach().clone()
        base = base_values[target]
        threshold = max(CONTROL_FLOOR, CONTROL_RELATIVE * abs(base))
        all_zero = float(saved.abs().max()) == 0.0
        if all_zero:
            # U is zero-initialized: zeroing/sign-flipping it is vacuous, so the
            # participation control becomes an activation control with a seeded
            # nonzero replacement and its sign flip.
            generator = torch.Generator().manual_seed(WORLD_SEED)
            replacement = torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.25
            with torch.no_grad():
                param.copy_(replacement)
            zero_value = float(evaluators[target]())
            with torch.no_grad():
                param.copy_(-replacement)
            sign_value = float(evaluators[target]())
            with torch.no_grad():
                param.copy_(saved)
            control = "activation"
        else:
            with torch.no_grad():
                param.zero_()
            zero_value = float(evaluators[target]())
            with torch.no_grad():
                param.copy_(saved)
                param.mul_(-1.0)
            sign_value = float(evaluators[target]())
            with torch.no_grad():
                param.copy_(saved)
            control = "zero_sign"
        zero_change, sign_change = abs(base - zero_value), abs(base - sign_value)
        predicted = (
            0.0
            if control == "activation"
            else abs(float((gradients[target][name].reshape(-1) * saved.reshape(-1)).sum()))
        )
        # The control verifies causal connectivity, not effect size; effect size is
        # owned by the FD-confirmed gradient. A below-noise first-order prediction
        # makes a below-noise change acceptable.
        connected = zero_change > threshold or predicted <= threshold
        row = {
            "family": family,
            "name": name,
            "target": target,
            "control_type": control,
            "base_loss": base,
            "first_order_prediction": predicted,
            "zero_change": zero_change,
            "sign_change": sign_change,
            "expected_no_effect": name == "V",
            "zero_ok": connected,
            "sign_ok": sign_change > threshold or predicted <= threshold,
            "distinct_ok": abs(sign_change - zero_change) > threshold,
        }
        if name == "V":
            row["zero_ok"] = row["sign_ok"] = zero_change <= threshold and sign_change <= threshold
        rows.append(row)
    return rows


def fp32_comparison():
    model, state, kwargs = fixture(torch.float32)
    named = list(model.named_parameters())
    world_named = [(n, p) for n, p in named if n.startswith("world.")]
    expr_value, _, _ = expr_loss(model, state, kwargs)
    write_value, _, _ = write2_loss(model, state, kwargs)
    world_value, _ = world_loss(model, state, kwargs)
    world_grads = collect_grads(world_value, world_named)
    grads = {
        "T_EXPR": collect_grads(expr_value, named),
        "T_WRITE2": collect_grads(write_value, named),
        "T_WORLD_PROXY": {
            name: world_grads.get(name, torch.zeros_like(param)) for name, param in named
        },
    }
    losses = {
        "T_EXPR": float(expr_value.detach()),
        "T_WRITE2": float(write_value.detach()),
        "T_WORLD_PROXY": float(world_value.detach()),
    }
    return losses, grads


def parameter_hash(named) -> str:
    digest = hashlib.sha256()
    for name, param in named:
        digest.update(name.encode("utf-8"))
        digest.update(param.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_declared_update(model, state, kwargs, named):
    global BACKWARD_CALLS
    optimizer = torch.optim.Adam(
        [param for _, param in named], lr=UPDATE_LR, betas=ADAM_BETAS, eps=ADAM_EPS
    )
    finfo = torch.finfo(model.W.dtype)
    pre_distribution = expr_loss(model, state, kwargs)[1].detach().clone()
    pre_hash = parameter_hash(named)
    steps = []
    for _ in range(UPDATE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = expr_loss(model, state, kwargs)[0] + write2_loss(model, state, kwargs)[0]
        loss.backward()
        BACKWARD_CALLS += 1
        grads = {name: None if p.grad is None else p.grad.detach() for name, p in named}
        before = {name: p.detach().clone() for name, p in named}
        optimizer.step()
        rows = {}
        for name, param in named:
            delta = (param.detach() - before[name]).abs()
            moment = optimizer.state.get(param, {})
            exp_avg, exp_avg_sq = moment.get("exp_avg"), moment.get("exp_avg_sq")
            scale = finfo.eps * max(
                float(before[name].abs().max()), float(param.detach().abs().max()), 1e-12
            )
            rows[name] = {
                "grad_max": None if grads[name] is None else float(grads[name].abs().max()),
                "delta_max": float(delta.max()),
                "delta_norm": float(delta.norm()),
                "ulp_scale": scale,
                "ulp_ratio": float(delta.max()) / scale,
                "moment_max": None if exp_avg is None else float(exp_avg.abs().max()),
                "moment_sq_max": None if exp_avg_sq is None else float(exp_avg_sq.max()),
            }
        steps.append(rows)
    post_distribution = expr_loss(model, state, kwargs)[1].detach()
    post_named = list(model.named_parameters())
    post_expr_grads = collect_grads(expr_loss(model, state, kwargs)[0], post_named)
    post_write_grads = collect_grads(write2_loss(model, state, kwargs)[0], post_named)
    return {
        "adam": {"lr": UPDATE_LR, "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0},
        "objective": "T_EXPR + T_WRITE2 (T_WORLD_PROXY excluded by contract)",
        "steps": steps,
        "pre_parameter_sha256": pre_hash,
        "post_parameter_sha256": parameter_hash(named),
        "output_change_max": float((post_distribution - pre_distribution).abs().max()),
        "v_step1_delta_exact_zero": steps[0]["V"]["delta_max"] == 0.0,
        "v_unblocked": float(post_expr_grads["V"].abs().max()) >= ACTIVITY_FLOOR,
        "v_post_gradient_max": float(post_expr_grads["V"].abs().max()),
        "u_step1_delta_max": steps[0]["U"]["delta_max"],
        "world_deltas_exact_zero": all(
            row["delta_max"] == 0.0
            for step in steps
            for name, row in step.items()
            if name.startswith("world.")
        ),
        "inert_update_max": max(
            row["delta_max"]
            for step in steps
            for name, row in step.items()
            if name in INERT_PARAMETERS
        ),
        "inert_post_gradient_max": float(post_expr_grads[INERT_PARAMETERS[0]].abs().max()),
        "write_chain_post_nonzero": float(post_write_grads["write_heads.0.0.weight"].abs().max())
        > ACTIVITY_FLOOR,
    }


def _run_all() -> dict:
    model, state, kwargs = fixture()
    named = list(model.named_parameters())
    losses64, grads64, _, clamp_margin = target_losses_and_grads(model, state, kwargs)

    write_named = [(n, p) for n, p in named if n.startswith("write_heads.")]
    world_named = [(n, p) for n, p in named if n.startswith("world.")]
    finite_difference = {
        "T_EXPR": finite_difference_sweep(model, state, kwargs, named, grads64["T_EXPR"], "T_EXPR"),
        "T_WRITE2": finite_difference_sweep(
            model, state, kwargs, write_named, grads64["T_WRITE2"], "T_WRITE2"
        ),
        "T_WORLD_PROXY": finite_difference_sweep(
            model, state, kwargs, world_named, grads64["T_WORLD_PROXY"], "T_WORLD_PROXY"
        ),
    }
    directional, second_derivative = {}, {}
    # Later sweeps overwrite earlier entries per name, leaving each tensor recorded on
    # exactly its declared owner target (write_heads under T_WRITE2, world under proxy).
    for target, subset in (
        ("T_EXPR", [(n, p) for n, p in named if not n.startswith("world.")]),
        ("T_WRITE2", write_named),
        ("T_WORLD_PROXY", world_named),
    ):
        directional.update(directional_sweep(model, state, kwargs, subset, grads64[target], target))
        second_derivative.update(second_derivative_sweep(model, state, kwargs, subset, target))
    probes = blocking_probes(model, state, kwargs, named)
    probes["inert"] = inert_parameter_probe(model, state, kwargs, named, grads64)
    controls = participation_controls(model, state, kwargs, named, grads64)
    losses32, grads32 = fp32_comparison()
    update = run_declared_update(model, state, kwargs, named)

    tensors, unexpected, below_noise = [], [], []
    fd_disagreements, directional_disagreements, second_disagreements = [], [], []
    for name, param in named:
        family = name.split(".")[0]
        target = owner_target(name)
        maxima = {t: float(grads64[t][name].abs().max()) for t in TARGETS}
        fd_targets = ["T_EXPR", "T_WRITE2"] if family == "write_heads" else [target]
        fd_ok = all(finite_difference[t][name]["gate_ok"] for t in fd_targets)
        direction = directional[name]
        second = second_derivative[name]
        direction_ok = bool(direction.get("ok", False))
        second_ok = bool(second.get("ok", False)) or second["status"] == "skipped_exact_zero"
        active = maxima[target] >= ACTIVITY_FLOOR
        if family == "world":
            if active and fd_ok and direction_ok and second_ok:
                status = "proxy_credited"
            elif 0.0 < maxima[target] < ACTIVITY_FLOOR:
                status = "below_noise_unresolved"
            else:
                status = "unexpected_broken"
        elif name == "V":
            status = "birth_zero"
        elif name in INERT_PARAMETERS:
            if maxima[target] <= INERT_FLOOR:
                status = "inert_shift_invariant"
            elif maxima[target] < ACTIVITY_FLOOR:
                status = "below_noise_unresolved"
            else:
                status = "unexpected_broken"
        elif not active:
            status = "below_noise_unresolved" if maxima[target] > 0.0 else "unexpected_broken"
        elif not (fd_ok and direction_ok and second_ok):
            status = "fd_disagreement"
        else:
            status = "credited"
        if family == "write_heads" and status == "credited" and maxima["T_EXPR"] <= ACTIVITY_FLOOR:
            status = "unexpected_broken"
        if status == "unexpected_broken":
            unexpected.append(name)
        elif status == "below_noise_unresolved":
            below_noise.append(name)
        if active and not fd_ok:
            fd_disagreements.append(name)
        if active and direction["status"] == "checked" and not direction_ok:
            directional_disagreements.append(name)
        if active and second["status"] == "checked" and not second_ok:
            second_disagreements.append(name)
        tensors.append(
            {
                "name": name,
                "family": family,
                "shape": list(param.shape),
                "numel": param.numel(),
                "forward_owner": forward_owner(name),
                "update_owner": "Psi" if name in {"U", "V"} else "Theta",
                "owner_target": target,
                "gradient_max": maxima,
                "status": status,
                "finite_difference": {t: finite_difference[t][name] for t in fd_targets},
                "directional": direction,
                "second_derivative": second,
            }
        )

    fp32_rows, fp32_violations = [], []
    all_finite = True
    for name, _ in named:
        target = owner_target(name)
        g64, g32 = grads64[target][name], grads32[target][name]
        finite = bool(torch.isfinite(g32).all())
        all_finite = all_finite and finite
        resolved = float(g64.abs().max()) >= FP32_RESOLVED
        deviation = float((g32 - g64).abs().max() / max(float(g64.abs().max()), 1e-8))
        ok = (not resolved) or (finite and deviation <= FP32_TOLERANCE)
        if resolved and not ok:
            fp32_violations.append(name)
        fp32_rows.append(
            {
                "name": name,
                "target": target,
                "float64_max": float(g64.abs().max()),
                "relative_deviation": deviation,
                "resolved": resolved,
                "finite": finite,
                "ok": ok,
            }
        )

    probe_failures = [
        *(
            f"{probe}.{key}"
            for probe, results in probes.items()
            if probe != "inert"
            for key, value in results.items()
            if not value
        ),
        *(
            f"inert.{key}"
            for key, value in probes["inert"].items()
            if key not in ("name", "gradient_max", "value_replacement_change") and not value
        ),
    ]
    control_failures = [
        row["name"]
        for row in controls
        if not (
            row["expected_no_effect"] or (row["zero_ok"] and row["sign_ok"] and row["distinct_ok"])
        )
    ]
    credited = [row["name"] for row in tensors if row["status"] in CREDITED_STATUSES]
    statuses = {row["name"]: row["status"] for row in tensors}
    moved = all(
        update["steps"][0][name]["delta_max"] > 0.0
        for name in credited
        if name != "V"
        and not name.startswith("world.")
        and statuses[name] != "inert_shift_invariant"
    )
    gate = {
        "unexpected": unexpected,
        "below_noise": below_noise,
        "fd_disagreements": fd_disagreements,
        "directional_disagreements": directional_disagreements,
        "second_disagreements": second_disagreements,
        "fp32_violations": fp32_violations,
        "probe_failures": probe_failures,
        "control_failures": control_failures,
        "clamp_margin": clamp_margin,
        "credited_count": len(credited),
        "update_credited_deltas_nonzero": moved,
        "inert_updates_negligible": (
            update["inert_update_max"] <= UPDATE_LR * 1e-9
            and update["inert_post_gradient_max"] <= INERT_FLOOR
        ),
    }
    gate["passed"] = not (
        unexpected
        or below_noise
        or fd_disagreements
        or directional_disagreements
        or second_disagreements
        or fp32_violations
        or probe_failures
        or control_failures
        or clamp_margin <= 1.0
        or not moved
        or not gate["inert_updates_negligible"]
    )
    return {
        "losses_float64": {t: float(v.detach()) for t, v in losses64.items()},
        "losses_float32": losses32,
        "tensors": tensors,
        "gate": gate,
        "probes": probes,
        "controls": controls,
        "finite_difference": finite_difference,
        "directional": directional,
        "second_derivative": second_derivative,
        "fp32": {
            "all_finite": all_finite,
            "violations": fp32_violations,
            "tolerance": FP32_TOLERANCE,
            "resolved_floor": FP32_RESOLVED,
            "rows": fp32_rows,
        },
        "update": update,
        "fixture_parameter_sha256_pre_update": update["pre_parameter_sha256"],
        "fixture_parameter_sha256_post_update": update["post_parameter_sha256"],
    }


def compute_all() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = _run_all()
    return _CACHE


def test_r022_declared_credit_gate():
    payload = compute_all()
    gate = payload["gate"]
    assert gate["unexpected"] == []
    assert gate["below_noise"] == []
    assert gate["fd_disagreements"] == []
    assert gate["directional_disagreements"] == []
    assert gate["second_disagreements"] == []
    assert gate["passed"] is True, gate
    assert len(payload["tensors"]) == 101
    assert gate["credited_count"] == 101
    assert gate["inert_updates_negligible"] is True
    inert = payload["probes"]["inert"]
    assert inert["gradient_below_inert_floor"] is True
    assert inert["value_replacement_no_effect"] is True
    for row in payload["tensors"]:
        assert row["status"] in CREDITED_STATUSES


def test_r022_birth_zero_and_post_update_unblocking():
    payload = compute_all()
    row = {entry["name"]: entry for entry in payload["tensors"]}["V"]
    assert row["gradient_max"]["T_EXPR"] == 0.0
    assert row["gradient_max"]["T_WRITE2"] == 0.0
    assert row["status"] == "birth_zero"
    update = payload["update"]
    assert update["v_step1_delta_exact_zero"] is True
    assert update["v_unblocked"] is True
    assert update["v_post_gradient_max"] > ACTIVITY_FLOOR
    assert update["u_step1_delta_max"] > 0.0
    assert update["world_deltas_exact_zero"] is True


def test_r022_declared_blocking_probes():
    probes = compute_all()["probes"]
    assert probes["ordering"]["write_heads_exact_zero"] is True
    assert probes["permission"]["write_heads_exact_zero"] is True
    assert probes["permission"]["positive_control_nonzero"] is True
    assert probes["null_view"]["cut_exact_zero"] is True
    conditioned = probes["state_conditioned"]
    assert conditioned["coverage_step1_exact_zero"] is True
    assert conditioned["continuation_step1_exact_zero"] is True
    assert conditioned["coverage_step2_nonzero"] is True
    assert conditioned["continuation_step2_nonzero"] is True


def test_r022_finite_difference_eps_sweep():
    finite_difference = compute_all()["finite_difference"]
    assert set(finite_difference) == set(TARGETS)
    for per_tensor in finite_difference.values():
        for name, summary in per_tensor.items():
            assert summary["gate_ok"], f"{name} finite-difference sweep failed"
            assert len(summary["per_eps"]) == len(EPS_LIST)


def test_r022_directional_and_second_derivative():
    payload = compute_all()
    for name, row in payload["directional"].items():
        assert row["status"] in {"checked", "skipped_exact_zero"}
        if row["status"] == "checked":
            assert row["ok"], f"{name} directional derivative disagreement"
    for name, row in payload["second_derivative"].items():
        assert row["status"] in {"checked", "skipped_exact_zero"}
        if row["status"] == "checked":
            assert row["ok"], f"{name} second directional derivative disagreement"
    assert payload["gate"]["clamp_margin"] > 1.0


def test_r022_zero_and_signflip_controls():
    for row in compute_all()["controls"]:
        if row["expected_no_effect"]:
            assert row["zero_ok"] and row["sign_ok"]
        else:
            assert row["zero_ok"], f"{row['name']} zeroing control failed"
            assert row["sign_ok"], f"{row['name']} sign-flip control failed"
            assert row["distinct_ok"], f"{row['name']} responses degenerate"


def test_r022_float32_float64_group_comparison():
    comparison = compute_all()["fp32"]
    assert comparison["all_finite"] is True
    assert comparison["violations"] == []
    resolved = [row for row in comparison["rows"] if row["resolved"]]
    assert resolved
    deviations = sorted(row["relative_deviation"] for row in resolved)
    assert deviations[len(deviations) // 2] < comparison["tolerance"]


def test_r022_actual_update_delta_and_moments():
    payload = compute_all()
    update = payload["update"]
    assert len(update["steps"]) == UPDATE_STEPS
    assert update["output_change_max"] > 0.0
    assert update["write_chain_post_nonzero"] is True
    for step in update["steps"]:
        for row in step.values():
            assert row["ulp_ratio"] >= 0.0
            if row["moment_max"] is not None:
                assert row["moment_sq_max"] >= 0.0
    assert payload["gate"]["update_credited_deltas_nonzero"] is True
