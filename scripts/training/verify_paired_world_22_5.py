"""Freeze paired-task data, lawful-history ceilings and scripted protocol recovery."""

# ruff: noqa: E402 -- account for imports in the verifier CPU budget
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.evidence_a_learner_22_5 import load_learner
from scripts.training.paired_world_22_5 import audit_partition, build_partition
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from seed_platform.paths import get_external_path
from taiji.u2_evidence_runtime import EvidenceRuntime
from taiji.u2_history import (
    assert_values_equal,
    atomic_load,
    atomic_save,
    detached_tree,
    value_digest,
)
from taiji.u2_recovery import _source_pins
from taiji.u2_task_executor import PhysicalExecutor
from taiji.u2_task_world import PhysicalWorld

CONTRACT = ROOT / "plans/manifests/taiji_u2_p3_5_1_paired_contract_20261112.json"
MANIFEST = ROOT / "plans/manifests/taiji_u2_paired_task_20261112.json"
REPORT = ROOT / "reports/taiji_u2_paired_task_20261112.json"


def measure(runtime, remaining):
    state, model = runtime.state, runtime.model
    terms = []
    probability, _, _ = runtime.output()
    first = probability.detach().clone()
    for symbol in remaining:
        logp, _, details = model.output(
            state, model.language_goal(state), state["prediction_action"]
        )
        terms.append(-logp[symbol])
        state = model.accept_byte(state, symbol, details)
    named = dict(model.named_parameters())
    gradients = torch.autograd.grad(
        torch.stack(terms).mean(), list(named.values()), allow_unused=True
    )
    assert all(value is None or bool(torch.isfinite(value).all()) for value in gradients)
    return {
        "state_digest": value_digest(runtime.state),
        "next_logp": first,
        "gradients": dict(zip(named, detached_tree(gradients), strict=True)),
    }


def worker(jobs, output):
    torch.set_num_threads(1)
    with StepGuard() as guard:
        rows = [
            measure(EvidenceRuntime.load(job["checkpoint"]), job["remaining"])
            for job in atomic_load(jobs)
        ]
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
    if MANIFEST.exists() or REPORT.exists():
        raise FileExistsError("paired_task_already_frozen")
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    root = Path(get_external_path(f"data/experiments/taiji_u2/paired_task/{time.time_ns()}"))
    with StepGuard() as guard:
        data = {name: build_partition(name) for name in contract["partitions"]}
        audits = {name: audit_partition(partition) for name, partition in data.items()}
        for audit in audits.values():
            assert audit["worlds"] == 18 and audit["episodes"] == 180
            assert audit["rounds"] == audit["oracle_correct"] == 360
            assert all(row["accuracy"] == 0.5 for row in audit["partial_input_ceilings"].values())
        assert set(data["train"]["worlds"]).isdisjoint(data["dev"]["worlds"])
        assert {w["audit"]["seed"] for w in data["train"]["worlds"].values()}.isdisjoint(
            {w["audit"]["seed"] for w in data["dev"]["worlds"].values()}
        )
        retired = {
            **data["train"],
            "episodes": [
                episode
                for episode in data["train"]["episodes"]
                if episode["audit"]["absent"]
                == (episode["audit"]["visible"] + episode["audit"]["position"] + 1) % 6
            ],
        }
        retired_audit = audit_partition(retired)
        assert (
            retired_audit["partial_input_ceilings"]["public_without_rgb_and_question_history"][
                "accuracy"
            ]
            == 2 / 3
        )
        atomic_save(data, root / "data.bin")
        assert value_digest(atomic_load(root / "data.bin")) == value_digest(data)
        old = json.loads(
            (ROOT / "reports/taiji_u2_evidence_A_learning_20261112_v2.json").read_text(
                encoding="utf-8"
            )
        )
        parent_path = old["arms"]["native-query"]["checkpoint"]
        jobs, expected = [], []
        for dtype in (torch.float32, torch.float64):
            for index in (0, 1):
                directory = root / f"{str(dtype).split('.')[-1]}-{index}"
                directory.mkdir()
                model, _, _ = load_learner(parent_path)
                model = model.to(dtype=dtype)
                episode = data["train"]["episodes"][index]
                fixture = data["train"]["worlds"][episode["world_id"]]
                executor = PhysicalExecutor.create(
                    PhysicalWorld.restore(fixture["audit"]["genesis"]), directory / "world.bin"
                )
                runtime = EvidenceRuntime(
                    model, executor=executor, subject_id="paired-protocol-subject"
                )
                runtime.set_goal(fixture["public"]["goal"])
                runtime.begin_observation("paired-observation")
                runtime.execute_pending(directory / "intent.bin")
                packet = executor.packet(runtime)
                assert_values_equal(
                    {k: v.to(dtype=dtype) for k, v in fixture["public"]["observations"].items()},
                    packet.observations,
                )
                runtime.receive(packet)
                goal_version = runtime.state["goal_version"]
                runtime.set_request("first", episode["requests"][0])
                runtime.begin_utterance()
                # Deliberately wrong literal output witnesses protocol continuity only.
                for symbol in [*"右边".encode(), 256]:
                    runtime.accept_byte(symbol)
                runtime.set_request("second", episode["requests"][1])
                assert runtime.state["goal_version"] == goal_version
                assert len(runtime.state["seen_packets"]) == 1
                runtime.begin_utterance()
                target = episode["targets"][1]
                runtime.accept_byte(target[0])
                checkpoint = directory / "second-request.bin"
                runtime.save(checkpoint)
                remaining = [*target[1:], 256]
                jobs.append({"checkpoint": str(checkpoint), "remaining": remaining})
                expected.append(measure(runtime, remaining))
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
            timeout=120,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-6000:])
        restored = atomic_load(root / "worker.bin")
        assert restored["pid"] != os.getpid() and restored["optimizer_steps"] == 0
        assert_values_equal(expected, restored["rows"], "paired-second-request-recovery")
    cpu = time.process_time() + restored["cpu_seconds"]
    assert cpu <= contract["cpu_cap"] and time.monotonic() - STARTED <= contract["wall_cap"]
    pins = {f"taiji/{name}": pin for name, pin in _source_pins().items()}
    for name in (
        "scripts/training/paired_world_22_5.py",
        "scripts/training/paired_episode_22_5.py",
        "scripts/training/first_a_support_22_5.py",
        "scripts/training/evidence_a_learner_22_5.py",
        "scripts/training/verify_paired_world_22_5.py",
    ):
        pins[name] = hashlib.sha256(
            (ROOT / name).read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
    manifest = {
        "schema": "taiji-u2-paired-task-data-v1",
        "status": "fixture_and_protocol_accepted",
        "data_path": str(root / "data.bin"),
        "data_digest": value_digest(data),
        "partitions": audits,
        "source_pins": pins,
        "contract_sha256": hashlib.sha256(
            CONTRACT.read_text(encoding="utf-8-sig").encode()
        ).hexdigest(),
        "research_parent_checkpoint_sha256": hashlib.sha256(
            Path(parent_path).read_bytes()
        ).hexdigest(),
        "model_task_evaluations": 0,
        "optimizer_steps": guard.calls,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {
        **manifest,
        "schema": "taiji-u2-paired-task-verification-v1",
        "verdict": "passed_fixture_and_protocol_only",
        "retired_cyclic_design": retired_audit,
        "independent_protocol_cases": len(jobs),
        "parameter_gradient_comparisons": sum(len(row["gradients"]) for row in expected),
        "max_gradient_difference": 0.0,
        "witness_scope": "Scripted wrong first answer, partial second answer and recovered credit graph. This is not model task accuracy.",
        "cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
        "run_root": str(root),
    }
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "cpu_seconds": cpu,
                "rounds_per_partition": 360,
                "recovery_cases": len(jobs),
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
