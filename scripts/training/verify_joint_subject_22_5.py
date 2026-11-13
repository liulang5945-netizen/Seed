"""Verify normal shared fusion, real engineering updates and independent recovery."""

# ruff: noqa: E402 -- startup is included in the engineering price
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
from scripts.training.joint_token_learner_22_5 import (
    fork_with_optimizer,
    load_learner,
    load_parent,
    save_learner,
)
from scripts.training.paired_episode_22_5 import start_episode
from scripts.training.paired_world_22_5 import public_example
from scripts.training.task_history_loss_22_5 import history_loss, score_answer
from scripts.training.verify_first_binding_update_22_5 import cast_partition
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from scripts.training.verify_task_goal_22_5 import compare, derivatives
from seed_platform.paths import get_external_path
from taiji.u2_history import (
    assert_values_equal,
    atomic_load,
    atomic_save,
    detached_tree,
    value_digest,
)
from taiji.u2_joint_subject import (
    JointGoalRuntime,
    JointGoalSubject,
    JointQueryRuntime,
    JointQuerySubject,
)
from taiji.u2_recovery import _source_pins
from taiji.u2_subject import U2Packet

CONTRACT = ROOT / "plans/manifests/taiji_u2_joint_subject_contract_20261113.json"
REPORT = ROOT / "reports/taiji_u2_joint_subject_20261113.json"
ACCEPTED = ROOT / "plans/manifests/taiji_u2_joint_subject_accepted_20261113.json"
STEPS = 0
RUN_ROOT = None
FACTORIES = {
    "query": (JointQuerySubject, JointQueryRuntime, "native-joint-query"),
    "goal-control": (JointGoalSubject, JointGoalRuntime, "native-joint-control"),
}


def measure_runtime(runtime, target):
    model, state = runtime.model, runtime.state
    loss, _ = score_answer(model, model.begin_utterance(state), target)
    gradients = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    return {
        "state_digest": value_digest(state),
        "optimizer_digest": value_digest(runtime.optimizer.state_dict()),
        "history_reference": runtime.history.reference(),
        "loss": loss.detach(),
        "gradients": {
            name: None if value is None else value.detach()
            for (name, _), value in zip(model.named_parameters(), gradients, strict=True)
        },
    }


def make_runtime(model, optimizer, runtime_cls, partition, episode, directory):
    public = public_example(partition, episode)["public"]
    runtime = runtime_cls(model, optimizer=optimizer, subject_id="joint-token-protocol")
    runtime.set_goal(public["goal"])
    runtime.begin_observation("recorded-observation", 0.05)
    runtime.execute_pending(directory / "intent.bin")
    packet = U2Packet(
        "recorded-public-frame",
        "sensor",
        "world",
        public["t_start"],
        public["t_end"],
        0.0,
        public["observations"],
        public["coordinates"],
        public["outcome"],
        "recorded-observation",
    )
    runtime.receive(packet)
    runtime.set_request("first", episode["requests"][0])
    runtime.begin_utterance()
    # A scripted wrong prefix verifies factual-history replay and carries no capability claim.
    for symbol in [*"右边".encode(), 256]:
        runtime.accept_byte(symbol)
    runtime.set_request("second", episode["requests"][1])
    return runtime


def worker(jobs_path, output):
    torch.set_num_threads(1)
    rows = []
    with StepGuard() as guard:
        for job in atomic_load(jobs_path):
            runtime = FACTORIES[job["mode"]][1].load(job["snapshot"])
            rows.append(measure_runtime(runtime, job["target"]))
    atomic_save(
        {
            "pid": os.getpid(),
            "rows": rows,
            "optimizer_steps": guard.calls,
            "cpu_seconds": time.process_time(),
        },
        output,
    )


def verify():
    global STEPS, RUN_ROOT
    if REPORT.exists() or ACCEPTED.exists():
        raise FileExistsError("joint_subject_engineering_already_run")
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name, pin in contract["source_pins"].items():
        assert (
            hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").encode()).hexdigest()
            == pin
        )
    data = atomic_load(contract["data_path"])
    assert value_digest(data) == contract["data_digest"]
    root = Path(get_external_path(f"data/experiments/taiji_u2/joint_subject/{time.time_ns()}"))
    RUN_ROOT = str(root)
    jobs, expected, cases, parents = [], [], [], {}

    def guard():
        if (
            time.process_time() > contract["cpu_cap"]
            or time.monotonic() - STARTED > contract["wall_cap"]
        ):
            raise RuntimeError("joint_subject_budget_stop")

    for mode, precision in contract["profiles"]:
        guard()
        cls, runtime_cls, architecture = FACTORIES[mode]
        parent, inherited, original = load_parent("D")
        parent = parent.to(dtype=getattr(torch, precision))
        inherited = fresh_optimizer(parent)
        inherited.load_state_dict(detached_tree(original["optimizer"]))
        partition = cast_partition(data["train"], getattr(torch, precision))
        episode = partition["episodes"][0]
        model, optimizer = fork_with_optimizer(parent, inherited, cls)
        before = value_digest(model.state_dict()), value_digest(optimizer.state_dict())
        common = list(dict(parent.named_parameters()))
        model._verification_cuts = ("joint_tokens",)
        with StepGuard() as blocked:
            reference = derivatives(parent, partition, episode, "actual-self", common)
            closed = derivatives(model, partition, episode, "actual-self", common)
            equivalence = compare(reference, closed, contract["equivalence"][precision])
            assert all(
                closed["gradients"][name] is None
                for name in dict(model.named_parameters())
                if name.startswith("joint_fusion.")
            )
            assert blocked.calls == 0
        del model._verification_cuts
        credit = {}
        for cut in ("joint_native", "joint_expression", "none"):
            model._verification_cuts = () if cut == "none" else (cut,)
            state = start_episode(model, partition, episode)
            loss, _ = score_answer(model, state, episode["targets"][0])
            values = torch.autograd.grad(loss, list(model.joint_fusion.parameters()))
            assert all(
                bool(torch.isfinite(value).all()) and bool(value.norm() > 0) for value in values
            )
            credit[cut] = {
                name: float(value.norm())
                for (name, _), value in zip(
                    model.joint_fusion.named_parameters(), values, strict=True
                )
            }
        del model._verification_cuts
        assert before == (value_digest(model.state_dict()), value_digest(optimizer.state_dict()))
        assert list(dict(model.policy_named_parameters())) == list(
            dict(parent.policy_named_parameters())
        )
        directory = root / f"{mode}-{precision}"
        directory.mkdir()
        pre = directory / "pre-learner.bin"
        lineage = {
            **original["lineage"],
            "joint_parent_sha256": contract["parents"]["D"]["sha256"],
            "joint_mode": mode,
        }
        metadata = {
            "update": 0,
            "cursor": 0,
            "order": original["order"],
            "data_digest": contract["data_digest"],
        }
        save_learner(pre, model, optimizer, architecture, lineage, **metadata)
        reloaded, reloaded_opt, _ = load_learner(pre)
        assert_values_equal(model.state_dict(), reloaded.state_dict(), "pre.parameters")
        assert_values_equal(optimizer.state_dict(), reloaded_opt.state_dict(), "pre.optimizer")
        runtime = make_runtime(model, optimizer, runtime_cls, partition, episode, directory)
        snapshot = directory / "pre-runtime.bin"
        runtime.save(snapshot)
        jobs.append({"mode": mode, "snapshot": str(snapshot), "target": episode["targets"][1]})
        expected.append(measure_runtime(runtime, episode["targets"][1]))
        # Engineering changes happen only after durable learner and runtime readback checks.
        changed, changed_opt, _ = load_learner(pre)
        changed_opt.zero_grad(set_to_none=True)
        task = history_loss(changed, partition, episode, "actual-self")
        task["loss"].backward()
        gradients = {
            name: float(parameter.grad.norm())
            for name, parameter in changed.named_parameters()
            if name.startswith("joint_fusion.") and parameter.grad is not None
        }
        assert len(gradients) == 12 and all(value > 0 for value in gradients.values())
        changed_opt.step()
        STEPS += 1
        changed.parameter_revision += 1
        moves = {
            name: float((changed.state_dict()[name] - model.state_dict()[name]).abs().max())
            for name in gradients
        }
        assert all(value > 0 for value in moves.values())
        post = directory / "post-learner.bin"
        save_learner(
            post,
            changed,
            changed_opt,
            architecture,
            lineage,
            **{**metadata, "update": 1, "cursor": 1},
        )
        migrated = runtime.migrate_parameters(
            changed, optimizer=changed_opt, acceptance=lambda *_: True, cpu_budget_seconds=30
        )
        assert migrated["status"] == "committed"
        snapshot = directory / "post-runtime.bin"
        runtime.save(snapshot)
        jobs.append({"mode": mode, "snapshot": str(snapshot), "target": episode["targets"][1]})
        expected.append(measure_runtime(runtime, episode["targets"][1]))
        if precision == "float32":
            parents[mode] = {
                "checkpoint": str(pre),
                "sha256": hashlib.sha256(pre.read_bytes()).hexdigest(),
                "parameter_digest": before[0],
                "optimizer_digest": before[1],
                "architecture": architecture,
            }
        cases.append(
            {
                "mode": mode,
                "precision": precision,
                **equivalence,
                "credit_by_cut": credit,
                "new_parameter_gradients": gradients,
                "new_parameter_max_moves": moves,
                "migration": migrated,
                "new_adam_ages": {
                    name: int(changed_opt.state[parameter]["step"])
                    for name, parameter in changed.named_parameters()
                    if name.startswith("joint_fusion.")
                },
            }
        )
        print(
            json.dumps(
                {
                    "mode": mode,
                    "precision": precision,
                    "engineering_steps": STEPS,
                    "cpu_seconds": time.process_time(),
                }
            ),
            flush=True,
        )
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
    assert_values_equal(expected, restored["rows"], "joint.independent_recovery")
    cpu = time.process_time() + restored["cpu_seconds"]
    assert cpu < contract["cpu_cap"] and time.monotonic() - STARTED < contract["wall_cap"]
    assert contract["engineering_optimizer_steps"] == STEPS
    manifest = {
        "schema": "taiji-u2-joint-subject-accepted-v1",
        "status": "shared_task_credit_updates_and_recovery_verified",
        "parents": parents,
        "source_pins": _source_pins(),
        "learning_admitted": False,
        "engineering_optimizer_steps": STEPS,
        "capability_qualifications": [],
        "run_root": str(root),
    }
    report = {
        **manifest,
        "schema": "taiji-u2-joint-subject-report-v1",
        "verdict": "passed_shared_joint_subject_engineering",
        "cases": cases,
        "independent_recoveries": len(jobs),
        "recovery_gradient_comparisons": sum(len(item["gradients"]) for item in expected),
        "recovery_gradient_max_difference": 0.0,
        "optimizer_steps": STEPS,
        "check_reads": 0,
        "final_reads": 0,
        "cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
    }
    write_report(REPORT, report)
    ACCEPTED.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "engineering_steps": STEPS,
                "recoveries": len(jobs),
                "cpu_seconds": cpu,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
    else:
        try:
            verify()
        except Exception:
            error = traceback.format_exc()
            failure = {
                "schema": "taiji-u2-joint-subject-report-v1",
                "verdict": "engineering_failure",
                "error": error,
                "engineering_optimizer_steps": STEPS,
                "optimizer_steps": STEPS,
                "run_root": RUN_ROOT,
                "learning_admitted": False,
                "capability_qualifications": [],
                "check_reads": 0,
                "final_reads": 0,
                "cpu_seconds": time.process_time(),
                "wall_seconds": time.monotonic() - STARTED,
                "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
            }
            if "budget_stop" in error:
                failure["verdict"] = "budget_stop"
            if not REPORT.exists():
                write_report(REPORT, failure)
            raise
