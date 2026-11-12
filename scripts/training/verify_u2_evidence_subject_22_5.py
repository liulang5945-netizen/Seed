"""Verify opt-in evidence subject migration and independent-process credit recovery."""

# ruff: noqa: E402 -- include startup in measured costs
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
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from seed_platform.paths import get_external_path
from taiji.u2_evidence_runtime import EvidenceRuntime, load_legacy_learner
from taiji.u2_evidence_subject import EvidenceSubject
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


def measure(runtime):
    model, state = runtime.model, runtime.state
    terms = []
    probability, _, _ = model.output(state, model.language_goal(state), state["prediction_action"])
    first = probability.detach().clone()
    for symbol in (0xB7, 0xA6, 256):
        probability, _, details = model.output(
            state, model.language_goal(state), state["prediction_action"]
        )
        terms.append(-probability[symbol])
        state = model.accept_byte(state, symbol, details)
    loss = torch.stack(terms).mean()
    named = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss, list(named.values()), allow_unused=True)
    return {
        "state_digest": value_digest(runtime.state),
        "probability": first,
        "loss": loss.detach(),
        "gradients": dict(zip(named, detached_tree(gradients), strict=True)),
    }


def worker(jobs, destination):
    torch.set_num_threads(1)
    with StepGuard() as guard:
        rows = [measure(EvidenceRuntime.load(job["checkpoint"])) for job in atomic_load(jobs)]
    atomic_save(
        {
            "pid": os.getpid(),
            "rows": rows,
            "cpu_seconds": time.process_time(),
            "optimizer_steps": guard.calls,
        },
        destination,
    )


def verify():
    torch.set_num_threads(1)
    report_path = ROOT / "reports/taiji_u2_evidence_subject_20261112.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    root = Path(get_external_path(f"data/experiments/taiji_u2/evidence_subject/{time.time_ns()}"))
    root.mkdir(parents=True, exist_ok=True)
    p2 = json.loads(
        (ROOT / "reports/taiji_u2_conditional_A_20261112.json").read_text(encoding="utf-8")
    )
    parent, _, migration = load_legacy_learner(p2["arms"]["native-conditional"]["checkpoint"])
    assert value_digest(parent.state_dict()) == migration["parameter_digest"]
    jobs, expected, profiles = [], [], []
    with StepGuard() as guard:
        for dtype in (torch.float32, torch.float64):
            directory = root / str(dtype).replace("torch.", "")
            directory.mkdir()
            model = EvidenceSubject.from_parent(parent.to(dtype=dtype))
            for name, value in parent.state_dict().items():
                assert torch.equal(value, model.state_dict()[name])
            executor = PhysicalExecutor.create(
                PhysicalWorld.reset(34, {"body_count": 2}), directory / "world.bin"
            )
            runtime = EvidenceRuntime(model, executor=executor, subject_id=f"p3-{dtype}")
            runtime.set_goal("依据公开观测用中文回答请求。")
            runtime.begin_observation("public-frame")
            runtime.execute_pending(directory / "intent.bin")
            runtime.receive(executor.packet(runtime))
            assert runtime.model.evidence_tokens(runtime.state)[0].shape[0] == 24
            for version in range(3):
                if version:
                    candidate, _ = EvidenceSubject.from_bundle(
                        runtime.model.export_bundle(runtime.state)
                    )
                    with torch.no_grad():
                        candidate.encoder["rgb"].weight.add_(0.001)
                    candidate.mark_parameter_update()
                    accepted = runtime.migrate_parameters(candidate, acceptance=lambda *_: True)
                    assert accepted["status"] == "committed"
                runtime.set_request(f"request-{version}", "红色圆片现在在哪里？")
                runtime.begin_utterance()
                runtime.accept_byte(0xE5)
                checkpoint = directory / f"version-{version}.bin"
                runtime.save(checkpoint)
                jobs.append({"checkpoint": str(checkpoint)})
                measurement = measure(runtime)
                assert all(
                    g is None or bool(torch.isfinite(g).all())
                    for g in measurement["gradients"].values()
                )
                for name in runtime.model.evidence_reader.state_dict():
                    gradient = measurement["gradients"][f"evidence_reader.{name}"]
                    assert gradient is not None and bool(gradient.any()), name
                gradient = measurement["gradients"]["encoder.rgb.weight"]
                assert gradient is not None and bool(gradient.any())
                expected.append(measurement)
                for symbol in (0xB7, 0xA6, 256):
                    runtime.accept_byte(symbol)
            profiles.append(
                {
                    "dtype": str(dtype),
                    "parameters": sum(p.numel() for p in runtime.model.parameters()),
                    "parameter_tensors": len(list(runtime.model.parameters())),
                    "state_fields": len(runtime.model.state_registry()),
                    "events": len(runtime.state["evidence_directory"]),
                    "raw_history_bytes": runtime.history.total_bytes,
                }
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
            timeout=240,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-6000:])
        restored = atomic_load(root / "worker.bin")
        assert restored["pid"] != os.getpid() and restored["optimizer_steps"] == 0
        assert_values_equal(expected, restored["rows"], "independent_process_credit")
    cpu = time.process_time() + restored["cpu_seconds"]
    accounted_cpu = cpu + 30
    assert accounted_cpu <= 240 and time.monotonic() - STARTED <= 480
    report = {
        "schema": "taiji-u2-evidence-subject-verification-v1",
        "verdict": "passed_engineering_only",
        "scope": "Normal physical observation, request and partial native byte expression; six exact independent-process continuations, including two encoder migrations per precision. Literal supervised bytes are a credit fixture, not world-QA capability.",
        "legacy_parent": migration,
        "source_pins": _source_pins(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profiles": profiles,
        "independent_process_cases": len(jobs),
        "parameter_gradient_comparisons": sum(len(row["gradients"]) for row in expected),
        "max_gradient_difference": 0.0,
        "optimizer_steps": guard.calls,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": cpu,
        "preparation_cpu_reserve": 30,
        "accounted_cpu_seconds": accounted_cpu,
        "preparation_failures": [
            "Verifier mkdir duplicated get_external_path directory creation",
            "Verifier expected accepted but migration API returns committed",
        ],
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
        "run_root": str(root),
    }
    write_report(report_path, report)
    print(
        json.dumps({"verdict": report["verdict"], "cpu_seconds": cpu, "cases": len(jobs)}),
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
        verify()
