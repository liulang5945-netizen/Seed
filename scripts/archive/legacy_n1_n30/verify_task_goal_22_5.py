"""Audit neutral task conditioning, conditional history credit and typed recovery."""

# ruff: noqa: E402 -- include startup in resource accounting
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.evidence_a_learner_22_5 import fresh_optimizer
from scripts.training.paired_compute_22_5 import shadow_optimizer
from scripts.training.task_goal_learner_22_5 import (
    fork_with_optimizer,
    load_verified_parent,
    save_learner,
)
from scripts.training.task_history_loss_22_5 import history_loss, score_answer
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from seed_platform.paths import get_external_path
from taiji.u2_history import assert_values_equal, atomic_load, atomic_save, value_digest
from taiji.u2_recovery import _source_pins
from taiji.u2_task_goal import (
    ActiveQueryRuntime,
    ActiveQuerySubject,
    PersistentGoalRuntime,
    PersistentGoalSubject,
)

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_6_interface_contract_20261113.json"
REPORT = ROOT / "reports/taiji_u2_task_goal_engineering_20261113.json"
ACCEPTED = ROOT / "plans/manifests/taiji_u2_task_goal_engineering_20261113.json"


def derivatives(model, partition, episode, source, common_names):
    trace = []
    original = model.output

    def observe(state, *args, **kwargs):
        result = original(state, *args, **kwargs)
        trace.append((result[0].detach().clone(), state["H"].detach().clone()))
        return result

    named = dict(model.named_parameters())
    with patch.object(model, "output", observe):
        result = history_loss(model, partition, episode, source)
        gradients = torch.autograd.grad(
            result["loss"], list(named.values()), create_graph=True, allow_unused=True
        )
        indexed = dict(zip(named, gradients, strict=True))
        scale = math.sqrt(sum(named[name].numel() for name in common_names))
        direction = torch.stack(
            [
                indexed[name].sum() / scale
                for name in common_names
                if indexed[name] is not None and indexed[name].requires_grad
            ]
        ).sum()
        hvp = torch.autograd.grad(direction, list(named.values()), allow_unused=True)
    return {
        "loss": result["loss"].detach(),
        "trace": trace,
        "generated": result["first_history_bytes"],
        "gradients": {
            name: None if value is None else value.detach()
            for name, value in zip(named, gradients, strict=True)
        },
        "hvp": {
            name: None if value is None else value.detach()
            for name, value in zip(named, hvp, strict=True)
        },
    }


def compare(old, new, tolerance):
    assert_values_equal(old["trace"], new["trace"], "neutral.logp_H")
    assert old["generated"] == new["generated"]
    torch.testing.assert_close(old["loss"], new["loss"], **tolerance)
    errors = {}
    for kind in ("gradients", "hvp"):
        maximum = 0.0
        for name, left in old[kind].items():
            right = new[kind][name]
            assert (left is None) == (right is None)
            if left is not None:
                torch.testing.assert_close(left, right, **tolerance)
                maximum = max(maximum, float((left - right).abs().max()))
        errors[kind + "_old_max_abs"] = maximum
    return errors


def recovery_measure(runtime):
    state, model = runtime.state, runtime.model
    loss, _ = score_answer(model, model.begin_utterance(state), "未知".encode())
    named = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss, list(named.values()), allow_unused=True)
    return {
        "state_digest": value_digest(state),
        "loss": loss.detach(),
        "gradients": {
            name: None if value is None else value.detach()
            for name, value in zip(named, gradients, strict=True)
        },
    }


def worker(jobs_path, destination):
    torch.set_num_threads(1)
    factories = {"active-query": ActiveQueryRuntime, "persistent-control": PersistentGoalRuntime}
    with StepGuard() as guard:
        rows = [
            recovery_measure(factories[job["mode"]].load(job["checkpoint"]))
            for job in atomic_load(jobs_path)
        ]
    atomic_save(
        {
            "pid": os.getpid(),
            "rows": rows,
            "optimizer_steps": guard.calls,
            "cpu_seconds": time.process_time(),
        },
        destination,
    )


def verify():
    if REPORT.exists() or ACCEPTED.exists():
        raise FileExistsError("task_goal_engineering_already_frozen")
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    task = json.loads(
        (ROOT / "plans/manifests/taiji_u2_paired_task_20261112.json").read_text(encoding="utf-8")
    )
    data = atomic_load(task["data_path"])
    assert value_digest(data) == task["data_digest"]
    root = Path(
        get_external_path(f"data/experiments/taiji_u2/task_goal_engineering/{time.time_ns()}")
    )
    cases, jobs, expected, parents = [], [], [], {}
    shadow_steps = 0
    for dtype in (torch.float32, torch.float64):
        parent, optimizer, original = load_verified_parent()
        parent = parent.to(dtype=dtype)
        optimizer = fresh_optimizer(parent)
        optimizer.load_state_dict(original["optimizer"])
        partition = {
            "worlds": {
                identity: {
                    **world,
                    "public": {
                        **world["public"],
                        "observations": {
                            k: v.to(dtype=dtype) for k, v in world["public"]["observations"].items()
                        },
                        "coordinates": {
                            k: v.to(dtype=dtype) for k, v in world["public"]["coordinates"].items()
                        },
                        "outcome": world["public"]["outcome"].to(dtype=dtype),
                    },
                }
                for identity, world in data["train"]["worlds"].items()
            },
            "episodes": data["train"]["episodes"],
        }
        common = list(dict(parent.named_parameters()))
        reference = {
            source: derivatives(parent, partition, partition["episodes"][0], source, common)
            for source in ("teacher", "actual-self")
        }
        for cls, runtime_cls, architecture in (
            (ActiveQuerySubject, ActiveQueryRuntime, "native-goal-query"),
            (PersistentGoalSubject, PersistentGoalRuntime, "native-goal-control"),
        ):
            model, child_optimizer = fork_with_optimizer(parent, optimizer, cls)
            before = value_digest(model.state_dict()), value_digest(child_optimizer.state_dict())
            for source in ("teacher", "actual-self"):
                result = derivatives(model, partition, partition["episodes"][0], source, common)
                tolerance = contract["equivalence"][str(dtype).split(".")[-1]]
                errors = compare(reference[source], result, tolerance)
                gradient = result["gradients"]["native_task_projection.weight"]
                assert (
                    gradient is not None
                    and bool(torch.isfinite(gradient).all())
                    and bool(gradient.any())
                )
                temporary, shadow = shadow_optimizer(model, child_optimizer)
                for target, name in zip(temporary, dict(model.named_parameters()), strict=True):
                    value = result["gradients"][name]
                    target.grad = None if value is None else value.clone()
                shadow.step()
                shadow_steps += 1
                moved = float(temporary[-1].detach().abs().max())
                assert moved > 0
                singular = torch.linalg.svdvals(gradient.double())
                cases.append(
                    {
                        "dtype": str(dtype),
                        "mode": model.projection_mode,
                        "history_source": source,
                        **errors,
                        "new_projection_gradient_norm": float(gradient.norm()),
                        "new_projection_credit_singular_values": singular.tolist(),
                        "shadow_projection_max_move": moved,
                        "neutral_logp_and_H_exact": True,
                        "old_policy_eligibility_unchanged": list(
                            dict(model.policy_named_parameters())
                        )
                        == list(dict(parent.policy_named_parameters())),
                    }
                )
            assert before == (
                value_digest(model.state_dict()),
                value_digest(child_optimizer.state_dict()),
            )
            directory = root / f"{model.projection_mode}-{str(dtype).split('.')[-1]}"
            directory.mkdir()
            if dtype == torch.float32:
                checkpoint = directory / "research-parent.bin"
                torch.set_rng_state(original["torch_rng"])
                save_learner(
                    checkpoint,
                    model,
                    child_optimizer,
                    architecture,
                    {
                        **original["lineage"],
                        "task_projection_parent_sha256": contract["parent"]["sha256"],
                        "projection_mode": model.projection_mode,
                    },
                    update=original["update"],
                    cursor=original["cursor"],
                    order=original["order"],
                    data_digest=original["data_digest"],
                )
                parents[model.projection_mode] = {
                    "checkpoint": str(checkpoint),
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "parameter_digest": value_digest(model.state_dict()),
                    "optimizer_digest": value_digest(child_optimizer.state_dict()),
                    "architecture": architecture,
                }
            runtime = runtime_cls(model, subject_id="task-goal-protocol")
            runtime.set_goal("保持持续目标。")
            runtime.set_request("first", "红色圆片现在在哪里？")
            runtime.begin_utterance()
            for symbol in [*"右边".encode(), 256]:
                runtime.accept_byte(symbol)
            runtime.set_request("second", "蓝色圆片现在在哪里？")
            snapshot = directory / "second-request.bin"
            runtime.save(snapshot)
            jobs.append({"mode": model.projection_mode, "checkpoint": str(snapshot)})
            expected.append(recovery_measure(runtime))
    atomic_save(jobs, root / "jobs.bin")
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__)),
            "--worker",
            str(root / "jobs.bin"),
            "--output",
            str(root / "worker.bin"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr[-6000:])
    restored = atomic_load(root / "worker.bin")
    assert restored["pid"] != os.getpid() and restored["optimizer_steps"] == 0
    assert_values_equal(expected, restored["rows"], "task-goal-independent-recovery")
    cpu = time.process_time() + restored["cpu_seconds"]
    assert cpu < contract["cpu_cap"] and time.monotonic() - STARTED < contract["wall_cap"]
    manifest = {
        "schema": "taiji-u2-task-goal-engineering-v1",
        "status": "neutral_candidate_and_conditional_history_checked",
        "parents": parents,
        "source_pins": _source_pins(),
        "data_manifest": "plans/manifests/taiji_u2_paired_task_20261112.json",
        "data_digest": task["data_digest"],
        "learning_admitted": False,
        "learner_optimizer_steps": 0,
        "capability_qualifications": [],
    }
    ACCEPTED.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {
        **manifest,
        "schema": "taiji-u2-task-goal-engineering-report-v1",
        "verdict": "passed_neutral_goal_and_conditional_history_engineering",
        "cases": cases,
        "independent_recovery_cases": 4,
        "recovery_gradient_comparisons": sum(len(item["gradients"]) for item in expected),
        "recovery_gradient_max_difference": 0.0,
        "optimizer_steps": shadow_steps,
        "shadow_optimizer_steps": shadow_steps,
        "learner_optimizer_steps": 0,
        "check_reads": 0,
        "final_reads": 0,
        "cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
        "verifier_sha256": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8-sig").encode()
        ).hexdigest(),
    }
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "cpu_seconds": cpu,
                "cases": len(cases),
                "shadow_steps": shadow_steps,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    worker(args.worker, args.output) if args.worker else verify()
