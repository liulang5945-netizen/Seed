"""Test fixed first-answer objectives using isolated real inherited-Adam steps."""

# ruff: noqa: E402 -- startup belongs to the engineering budget
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.evidence_a_learner_22_5 import fresh_optimizer
from scripts.training.first_binding_objective_22_5 import add_gradients, quad_terms
from scripts.training.paired_compute_22_5 import shadow_optimizer
from scripts.training.run_task_goal_learning_22_5 import sources as learning_sources
from scripts.training.task_goal_learner_22_5 import load_learner
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import assert_values_equal, atomic_load, detached_tree, value_digest

CONTRACT = ROOT / "plans/manifests/taiji_u2_first_binding_update_contract_20261113.json"
REPORT = ROOT / "reports/taiji_u2_first_binding_update_20261113.json"


def sources():
    result = learning_sources()
    for filename in (
        "scripts/training/probe_first_binding_22_5.py",
        "scripts/training/first_binding_objective_22_5.py",
        "scripts/training/verify_first_binding_update_22_5.py",
    ):
        result[filename] = hashlib.sha256(
            (ROOT / filename).read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
    return result


def cast_partition(partition, dtype):
    result = copy.deepcopy(partition)
    for world in result["worlds"].values():
        public = world["public"]
        for key in ("observations", "coordinates"):
            public[key] = {name: value.to(dtype=dtype) for name, value in public[key].items()}
        for key in ("action", "outcome"):
            public[key] = public[key].to(dtype=dtype)
    return result


def measure(model, partition, contract, guard, *, gradients):
    named = list(model.named_parameters())
    totals = {name: [None] * len(named) for name in ("ce", "mixed", "delta")}
    rows = []
    count = len(contract["positions"]) * len(contract["color_pairs"])
    with StepGuard() as blocked, torch.set_grad_enabled(gradients):
        for position in contract["positions"]:
            for colors in contract["color_pairs"]:
                guard()
                terms = quad_terms(model, partition, position, colors)
                assert all(bool(torch.isfinite(terms[key])) for key in totals)
                for index, name in enumerate(totals):
                    if gradients:
                        values = torch.autograd.grad(
                            terms[name] / count,
                            [parameter for _, parameter in named],
                            allow_unused=True,
                            retain_graph=index < len(totals) - 1,
                        )
                        totals[name] = add_gradients(totals[name], values)
                rows.append(
                    {
                        "position": position,
                        "colors": colors,
                        **{name: float(terms[name].detach()) for name in totals},
                        "odds": terms["odds"].detach().tolist(),
                        "histories": terms["histories"],
                    }
                )
        assert blocked.calls == 0
    return {
        "means": {key: sum(row[key] for row in rows) / count for key in totals},
        "rows": rows,
    }, totals


def dot(first, second):
    return sum(
        float((left.detach().double() * right.detach().double()).sum())
        for left, right in zip(first, second, strict=True)
        if left is not None and right is not None
    )


def cosine(first, second):
    denominator = (dot(first, first) * dot(second, second)) ** 0.5
    return None if denominator == 0 else dot(first, second) / denominator


def inspect_step(
    model, optimizer, gradient, moment_mode, contract, before, partition, guard, counter
):
    guard()
    temporary, shadow = shadow_optimizer(model, optimizer)
    if moment_mode == "zero_first_moment":
        for state in shadow.state.values():
            state["exp_avg"].zero_()
    before_clocks = {
        index: int(state["step"]) for index, state in shadow.state_dict()["state"].items()
    }
    for parameter, value in zip(temporary, gradient, strict=True):
        parameter.grad = None if value is None else value.detach().clone()
    shadow.step()
    counter["shadow_steps"] += 1
    after_state = shadow.state_dict()["state"]
    for index, age in before_clocks.items():
        assert int(after_state[index]["step"]) == age + (gradient[index] is not None)
    displacement = [
        new.detach() - old.detach() for new, old in zip(temporary, model.parameters(), strict=True)
    ]
    candidate = copy.deepcopy(model)
    with torch.no_grad():
        for target, source in zip(candidate.parameters(), temporary, strict=True):
            assert target.data_ptr() != source.data_ptr()
            target.copy_(source)
    candidate.parameter_revision += 1
    after, _ = measure(candidate, partition, contract, guard, gradients=False)
    tolerance = contract["delta_tolerance"][str(next(model.parameters()).dtype).split(".")[-1]]
    improvements = [
        later["delta"] - earlier["delta"]
        for earlier, later in zip(before["rows"], after["rows"], strict=True)
    ]
    effective = (
        after["means"]["delta"] - before["means"]["delta"] > tolerance
        and sum(value > tolerance for value in improvements) >= contract["minimum_improving_quads"]
        and after["means"]["ce"] - before["means"]["ce"] <= contract["ce_increase_tolerance"]
    )
    named = list(model.named_parameters())
    return {
        "after": after,
        "mean_changes": {
            key: after["means"][key] - before["means"][key] for key in before["means"]
        },
        "delta_improvements": improvements,
        "effective_local_update": effective,
        "first_histories_changed": sum(
            a != b
            for earlier, later in zip(before["rows"], after["rows"], strict=True)
            for a, b in zip(earlier["histories"], later["histories"], strict=True)
        ),
        "parameter_changes": {
            key: {
                "gradient_norm": None if value is None else float(value.norm()),
                "step_norm": float(change.norm()),
                "step_max_abs": float(change.abs().max()),
            }
            for (key, _), value, change in zip(named, gradient, displacement, strict=True)
        },
    }, displacement


def run():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["source_pins"] == sources()
    evidence_path = ROOT / "reports/taiji_u2_task_goal_learning_20261113.json"
    assert (
        hashlib.sha256(evidence_path.read_bytes()).hexdigest() == contract["learning_report_sha256"]
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    data = atomic_load(contract["data_path"])
    assert value_digest(data) == contract["data_digest"]
    counter = {"shadow_steps": 0}
    profiles = {}

    def guard():
        if (
            time.process_time() > contract["cpu_cap"]
            or time.monotonic() - STARTED > contract["wall_cap"]
        ):
            raise RuntimeError("first_binding_update_budget_stop")

    try:
        for name, precision in contract["profiles"]:
            guard()
            checkpoint = evidence["results"][name]["checkpoint"]
            assert (
                hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                == contract["checkpoint_sha256"][name]
            )
            model, inherited, _ = load_learner(checkpoint)
            assert value_digest(model.state_dict()) == evidence["results"][name]["parameter_digest"]
            model = model.to(dtype=getattr(torch, precision))
            optimizer = fresh_optimizer(model)
            optimizer.load_state_dict(detached_tree(inherited.state_dict()))
            if precision == "float32":
                assert_values_equal(
                    optimizer.state_dict(), inherited.state_dict(), "unchanged_float32_moments"
                )
            parameters_before = value_digest(model.state_dict())
            moments_before = value_digest(optimizer.state_dict())
            partition = cast_partition(data["train"], getattr(torch, precision))
            before, gradients = measure(model, partition, contract, guard, gradients=True)
            assert all(
                value is None or bool(torch.isfinite(value).all())
                for values in gradients.values()
                for value in values
            )
            objectives = {
                "ce": gradients["ce"],
                "mixed": gradients["mixed"],
                "combined": add_gradients(gradients["ce"], gradients["mixed"]),
            }
            mixed_squared_norm = dot(gradients["mixed"], gradients["mixed"])
            record = {
                "before": before,
                "ce_mixed_cosine": cosine(gradients["ce"], gradients["mixed"]),
                "ce_mixed_norm_ratio": (
                    None
                    if mixed_squared_norm == 0
                    else (dot(gradients["ce"], gradients["ce"]) / mixed_squared_norm) ** 0.5
                ),
                "steps": {},
            }
            for objective, gradient in objectives.items():
                for moment_mode in contract["moment_modes"]:
                    result, displacement = inspect_step(
                        model,
                        optimizer,
                        gradient,
                        moment_mode,
                        contract,
                        before,
                        partition,
                        guard,
                        counter,
                    )
                    result["linear_prediction"] = {
                        key: dot(value, displacement) for key, value in gradients.items()
                    }
                    record["steps"][f"{objective}-{moment_mode}"] = result
                    assert value_digest(model.state_dict()) == parameters_before
                    assert value_digest(optimizer.state_dict()) == moments_before
            profiles[f"{name}-{precision}"] = record
            print(
                json.dumps(
                    {
                        "profile": f"{name}-{precision}",
                        "ce_mixed_cosine": record["ce_mixed_cosine"],
                        "changes": {
                            key: value["mean_changes"] for key, value in record["steps"].items()
                        },
                        "cpu_seconds": time.process_time(),
                    }
                ),
                flush=True,
            )
        guard()
        assert counter["shadow_steps"] == contract["shadow_optimizer_steps"]
        supported = all(
            profiles[f"D-{precision}"]["steps"]["combined-inherited"]["effective_local_update"]
            for precision in ("float32", "float64")
        )
        verdict = (
            "fixed_joint_objective_effective_local_update"
            if supported
            else "fixed_joint_objective_not_supported"
        )
        complete, error = True, None
    except Exception:
        complete, error = False, traceback.format_exc()
        verdict = "budget_stop" if "budget_stop" in error else "engineering_failure"
    report = {
        "schema": "taiji-u2-first-binding-update-v1",
        "verdict": verdict,
        "complete": complete,
        "error": error,
        "profiles": profiles,
        "learner_optimizer_steps": 0,
        "shadow_optimizer_steps": counter["shadow_steps"],
        "learning_admitted": False,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
    }
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("verdict", "complete", "shadow_optimizer_steps", "cpu_seconds")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
