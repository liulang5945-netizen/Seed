"""Recipe-owned gradients and verified checkpoint I/O for the 22.4.1 u1 update."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from scripts.training.u2_update_support_22_3_4_3 import (
    CheckedRuntime,
    fingerprint,
)
from taiji.u2_history import assert_values_equal
from taiji.u2_recovery import ReferenceExecutor
from taiji.u2_subject import U2Config, U2Packet, U2Subject, synthetic_fixture

UPDATE_ID = "u1-candidate-float32"
ARM = "u1-candidate-float32"
ADAM_FLOAT32_TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}


def normalized_sha256(path: Path) -> str:
    """UTF8 without BOM, universal newline to LF (frozen hash semantics)."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8-sig").encode()).hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


class U1Runtime(CheckedRuntime):
    """Every canonical checkpoint is create-once and read back by a fresh load."""

    CREATE_ONCE = {"parent.bin", "post.bin"}

    def save(self, path) -> Path:
        path = Path(path)
        if path.name in self.CREATE_ONCE and path.exists():
            raise FileExistsError(path)
        return super().save(path)


def optimizer_for(model, manifest):
    options = dict(manifest["optimizer"])
    if options.pop("type") != "Adam":
        raise ValueError("Only the frozen Adam recipe is supported")
    options.pop("basis", None)
    options["betas"] = tuple(options["betas"])
    return torch.optim.Adam(model.parameters(), **options)


def parent_digest(dtype: torch.dtype) -> str:
    from taiji.u2_history import value_digest

    base = U2Subject(U2Config.small()).to(dtype=dtype)
    return value_digest(base.state_dict())


def prepare_parent(manifest: dict[str, Any], directory: Path) -> U1Runtime:
    recipe = manifest["recipe"]
    dtype = getattr(torch, manifest["arms"][ARM]["dtype"])
    base = U2Subject(U2Config.small()).to(dtype=dtype)
    model = base.metadata_candidate()
    runtime = U1Runtime(model, executor=ReferenceExecutor(), subject_id="engineering-parent-" + ARM)
    fixture = synthetic_fixture(model.c, dtype)
    runtime.set_goal(recipe["u1_shadow"]["goal"])
    runtime.set_records(fixture["records"])
    mask = torch.zeros(9, dtype=torch.bool)
    mask[0] = True
    identity = runtime.issue(runtime.choose(mask), dt=recipe["dt"])
    runtime.execute_pending(checkpoint=directory / "_preparation-intent.bin")
    runtime.set_process_time(recipe["dt"])
    event = fixture["events"][0]
    runtime.receive(
        U2Packet(
            "engineering-parent-observation",
            "sensor",
            "world",
            0.0,
            recipe["dt"],
            recipe["dt"],
            event["observations"],
            event["coordinates"],
            event["outcome"],
            identity,
            feedback_record_ids=tuple(recipe["feedback_records"]),
        )
    )
    runtime.begin_new_window()
    runtime.optimizer = optimizer_for(model, manifest)
    return runtime


def _tensor_digests(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(
                value.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
        }
    raise TypeError(type(value))


def recipe_digest(manifest: dict[str, Any]) -> str:
    return canonical_sha256(manifest["recipe"])


def slice_descriptor(manifest: dict[str, Any], runtime_model) -> dict[str, Any]:
    recipe = manifest["recipe"]
    fixture = synthetic_fixture(runtime_model.c, runtime_model.native_w.dtype)
    observable = fixture["events"][: recipe["observations_per_shadow_episode"]]
    return {
        "config": "U2Config.small",
        "seed": runtime_model.c.seed,
        "dtype": manifest["arms"][ARM]["dtype"],
        "dt": recipe["dt"],
        "goal": recipe["u1_shadow"]["goal"],
        "records": [
            {
                "record_id": record.record_id,
                "raw_hex": record.raw.hex(),
                "version": record.version,
                "source": record.source,
                "scope": record.scope,
                "observed_at": record.observed_at,
                "writable": record.writable,
            }
            for record in fixture["records"]
        ],
        "observations": [
            {
                "observations": {
                    name: _tensor_digests(value) for name, value in event["observations"].items()
                },
                "coordinates": {
                    name: _tensor_digests(value) for name, value in event["coordinates"].items()
                },
                "outcome": _tensor_digests(event["outcome"]),
            }
            for event in observable
        ],
        "feedback_records": list(recipe["feedback_records"]),
    }


def slice_digest(manifest: dict[str, Any], runtime_model) -> str:
    return canonical_sha256(slice_descriptor(manifest, runtime_model))


def active_parameters(model: U2Subject, manifest: dict[str, Any]) -> list[str]:
    names = list(manifest["arms"][ARM]["active_parameters"])
    if model.c.group_units is not None:
        for key in model.group_extensions:
            names.extend([f"group_extensions.{key}.incoming", f"group_extensions.{key}.outgoing"])
    return names


def shadow_terms(model: U2Subject, manifest: dict[str, Any]):
    """The frozen u1 shadow episode: original records, act-only, no target."""
    recipe = manifest["recipe"]
    dt = recipe["dt"]
    fixture = synthetic_fixture(model.c, model.native_w.dtype)
    state = model.initial("engineering-shadow-" + ARM)
    state = model.set_goal(state, recipe["u1_shadow"]["goal"])
    state = model.set_records(state, fixture["records"])
    receipts = []
    for index, event in enumerate(fixture["events"][: recipe["observations_per_shadow_episode"]]):
        mask = torch.zeros(9, dtype=torch.bool)
        mask[0] = True
        state, decision = model.choose_operation(state, mask)
        state, identity = model.issue(state, decision, dt=dt)
        end = (index + 1) * dt
        state["process_time"] = end
        state, result = model.receive(
            state,
            U2Packet(
                f"shadow-{ARM}-{index}",
                "sensor",
                "world",
                index * dt,
                end,
                end,
                event["observations"],
                event["coordinates"],
                event["outcome"],
                identity,
                feedback_record_ids=tuple(recipe["feedback_records"]),
            ),
        )
        receipts.append(result)
    weights = manifest["arms"][ARM]["weights"]
    pieces = {
        "predictive": -torch.stack(
            [receipt["predictive_logp"].mean() for receipt in receipts]
        ).mean(),
        "actor": torch.stack([receipt["actor"] for receipt in receipts]).mean(),
        "critic": torch.stack([receipt["critic"] for receipt in receipts]).mean(),
        "verbatim": model.native_w.new_zeros(()),
    }
    return sum(weights[name] * value for name, value in pieces.items()), pieces


def gradients_for(runtime: U1Runtime, manifest: dict[str, Any]):
    model = runtime.model
    previous = fingerprint(runtime)
    loss, pieces = shadow_terms(model, manifest)
    named = dict(model.named_parameters())
    measured = torch.autograd.grad(loss, list(named.values()), allow_unused=True)
    values = dict(
        zip(
            named,
            (None if gradient is None else gradient.detach().clone() for gradient in measured),
            strict=True,
        )
    )
    if any(
        value is not None and not bool(torch.isfinite(value).all()) for value in values.values()
    ):
        raise ValueError("Nonfinite engineering gradient")
    floor = manifest["activity_floors"][manifest["arms"][ARM]["dtype"]]
    names = active_parameters(model, manifest)
    if any(
        values[name] is None or float(values[name].norm()) < floor["gradient_norm"]
        for name in names
    ):
        from scripts.training.u2_update_executor_22_4_1 import ActivityGateError

        rows = [
            {
                "parameter": name,
                "gradient_norm": None if values[name] is None else float(values[name].norm()),
                "change": 0.0,
                "change_floor": floor["max_parameter_change"],
                "margin": 0.0,
                "barely_live": True,
                "gradient_ok": values[name] is not None
                and float(values[name].norm()) >= floor["gradient_norm"],
                "change_ok": False,
                "note": "no optimizer step was taken; the gradient floor failed first",
            }
            for name in names
        ]
        error = ActivityGateError(
            "Frozen active-gradient floor failed: "
            + str([row["parameter"] for row in rows if not row["gradient_ok"]])
        )
        error.receipt = {
            "parameters": rows,
            "min_margin": 0.0,
            "barely_live_parameters": [row["parameter"] for row in rows],
            "floors": floor,
        }
        raise error
    assert_values_equal(previous, fingerprint(runtime), "shadow_training_did_not_change_parent")
    return values, {
        "total": float(loss.detach()),
        **{name: float(value.detach()) for name, value in pieces.items()},
    }


def fork_candidate(runtime: U1Runtime, gradients, manifest: dict[str, Any]):
    from scripts.training.u2_update_support_22_3_4_3 import fork_candidate as fork

    return fork(runtime, gradients, manifest=manifest)


def check_adam_equations(parent: U1Runtime, post: U1Runtime, gradients, manifest) -> list:
    """Recompute Adam's update for every parameter; tolerance is the frozen 22.4.1 one."""
    options = manifest["optimizer"]
    beta1, beta2 = options["betas"]
    old = dict(parent.model.named_parameters())
    new = dict(post.model.named_parameters())
    results = []
    with torch.no_grad():
        for name, parameter in old.items():
            gradient = gradients[name]
            previous = parent.optimizer.state.get(parameter, {})
            actual = post.optimizer.state.get(new[name], {})
            if gradient is None:
                assert_values_equal(parameter, new[name], name + ".unused_parameter")
                assert_values_equal(previous, actual, name + ".unused_moment")
                results.append({"parameter": name, "status": "unused_by_loss"})
                continue
            mean = previous.get("exp_avg", torch.zeros_like(parameter)) * beta1 + gradient * (
                1 - beta1
            )
            variance = previous.get(
                "exp_avg_sq", torch.zeros_like(parameter)
            ) * beta2 + gradient.square() * (1 - beta2)
            step = float(previous.get("step", 0)) + 1
            expected = parameter - options["lr"] * (mean / (1 - beta1**step)) / (
                (variance / (1 - beta2**step)).sqrt() + options["eps"]
            )
            assert float(actual["step"]) == step, name
            assert torch.allclose(actual["exp_avg"], mean, **ADAM_FLOAT32_TOLERANCE), name + ".mean"
            assert torch.allclose(actual["exp_avg_sq"], variance, **ADAM_FLOAT32_TOLERANCE), (
                name + ".variance"
            )
            assert torch.allclose(new[name], expected, **ADAM_FLOAT32_TOLERANCE), name + ".update"
            results.append(
                {
                    "parameter": name,
                    "status": "adam_equation_matched",
                    "max_abs_delta": float((new[name] - expected).abs().max()),
                }
            )
    return results


def parameter_rows(parent: U1Runtime, post: U1Runtime, gradients) -> list[dict[str, Any]]:
    before = parent.model.state_dict()
    after = post.model.state_dict()
    return [
        {
            "parameter": name,
            "gradient_norm": None if gradient is None else float(gradient.norm()),
            "max_parameter_change": float((before[name] - after[name]).abs().max()),
        }
        for name, gradient in gradients.items()
    ]


def activity_receipt(
    model: U2Subject,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    gradients,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    floor = manifest["activity_floors"][manifest["arms"][ARM]["dtype"]]
    rows = []
    for name in active_parameters(model, manifest):
        change = float((before_state[name] - after_state[name]).abs().max())
        margin = change / floor["max_parameter_change"]
        rows.append(
            {
                "parameter": name,
                "gradient_norm": float(gradients[name].norm()),
                "change": change,
                "change_floor": floor["max_parameter_change"],
                "margin": margin,
                "barely_live": margin < 10,
                "gradient_ok": float(gradients[name].norm()) >= floor["gradient_norm"],
                "change_ok": change >= floor["max_parameter_change"],
            }
        )
    failed = [row["parameter"] for row in rows if not (row["gradient_ok"] and row["change_ok"])]
    receipt = {
        "parameters": rows,
        "min_margin": min(row["margin"] for row in rows),
        "barely_live_parameters": [row["parameter"] for row in rows if row["barely_live"]],
        "floors": floor,
    }
    if failed:
        from scripts.training.u2_update_executor_22_4_1 import ActivityGateError

        error = ActivityGateError(f"Frozen activity floor failed: {failed}")
        error.receipt = receipt
        raise error
    return receipt


def output_displacement(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    return {
        "next_output_l1": float(
            (torch.tensor(before["next_output"]) - torch.tensor(after["next_output"])).abs().sum()
        ),
        "policy_l1": float(
            (torch.tensor(before["policy"]) - torch.tensor(after["policy"])).abs().sum()
        ),
    }
