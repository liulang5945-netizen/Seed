"""Final-schema U2 S01-S12 recovery closure with zero additional optimizer steps."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest
import torch

from scripts.training.u2_update_executor_22_3_4_3 import write_once
from scripts.training.u2_update_support_22_3_4_3 import distributions, fingerprint
from scripts.training.verify_taiji_u2_cross_process_22_3 import (
    forbidden_step,
    packet_for,
    prepare_action,
    snapshot,
)
from scripts.training.verify_taiji_u2_runtime_22_3 import digest
from taiji.u2_history import HISTORY_SCHEMA, atomic_load, atomic_save, value_digest
from taiji.u2_recovery import ENVELOPE_SCHEMA, ReferenceExecutor, U2Runtime, run_conditions
from taiji.u2_subject import U2Config, U2Subject, synthetic_fixture

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_3_closeout_contract_20261111.json"


def isolated_executor(envelope, directory, label):
    path = directory / (label + "-executor.bin")
    atomic_save(envelope["executor_snapshot"]["ledger"], path)
    return ReferenceExecutor(path)


def endpoint(runtime, fixture, directory, identity):
    initial = {"fingerprint": fingerprint(runtime), **distributions(runtime)}
    if identity == "S01":
        runtime.set_goal(fixture["goal"])
        runtime.set_records(fixture["records"])
    if runtime.state["utterance_active"]:
        for symbol in [0xBA, 0xA2, 256]:
            runtime.accept_byte(symbol)
    prepare_action(runtime, directory)
    packet = packet_for(runtime, fixture, identity + "-next")
    receipt = runtime.receive(packet)
    # The expression probe has its own explicit request and uniquely visible text.
    runtime.set_goal("逐字输出当前唯一记录的完整原文。")
    runtime.set_records((fixture["records"][0],))
    utterance = runtime.model.begin_utterance(runtime.state)
    goal = runtime.model.goal(utterance["goal_raw"])
    terms = []
    for symbol in [*fixture["records"][0].raw, 256]:
        probability, _, output = runtime.model.output(
            utterance, goal, utterance["prediction_action"]
        )
        terms.append(-probability[symbol])
        utterance = runtime.model.accept_byte(utterance, symbol, output)
    loss = (
        torch.stack(terms).mean()
        - receipt["predictive_logp"].mean()
        + receipt["actor"]
        + receipt["critic"]
    )
    named = dict(runtime.model.named_parameters())
    gradients = torch.autograd.grad(loss, list(named.values()), allow_unused=True)
    return {
        "initial": initial,
        "state_digest": value_digest(runtime.state),
        "parameter_digest": value_digest(runtime.model.state_dict()),
        "history_reference": runtime.history.reference(),
        **distributions(runtime),
        "gradients": dict(
            zip(
                named,
                [None if gradient is None else gradient.detach() for gradient in gradients],
                strict=True,
            )
        ),
        "settlement_count": runtime.state["settled"].count(packet.receipt_id),
        "effect_count": runtime.executor.effect_count,
        "run_conditions": run_conditions(runtime.model),
    }


def worker(input_path, output_path):
    torch.set_num_threads(1)
    jobs = atomic_load(input_path)
    rows = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", forbidden_step)
        patch.setattr(torch.optim.SGD, "step", forbidden_step)
        for job in jobs:
            envelope = atomic_load(job["checkpoint"])
            runtime = U2Runtime.from_envelope(
                envelope, executor=isolated_executor(envelope, Path(job["directory"]), "worker")
            )
            measured = endpoint(runtime, job["fixture"], Path(job["directory"]), job["id"])
            rows.append({"id": job["id"], "measured": measured})
    atomic_save(
        {
            "rows": rows,
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "cpu_seconds": time.process_time(),
            "wall_seconds": time.monotonic() - STARTED,
        },
        output_path,
    )
    return 0


def compare(left, right, tolerance, parameters):
    for key in [
        "initial",
        "state_digest",
        "parameter_digest",
        "history_reference",
        "next_output",
        "policy",
        "settlement_count",
        "effect_count",
        "run_conditions",
    ]:
        assert left[key] == right[key], key
    assert left["settlement_count"] == 1
    maximum = 0.0
    for name, gradient in left["gradients"].items():
        restored = right["gradients"][name]
        row = parameters.setdefault(
            name,
            {
                "compared_snapshots": 0,
                "nonzero_snapshots": 0,
                "unused_snapshots": 0,
                "max_abs_delta": 0.0,
            },
        )
        row["compared_snapshots"] += 1
        if gradient is None:
            assert restored is None, name
            row["unused_snapshots"] += 1
            continue
        assert (
            restored is not None
            and bool(torch.isfinite(gradient).all())
            and bool(torch.isfinite(restored).all())
        ), name
        assert torch.allclose(gradient, restored, **tolerance), name
        delta = float((gradient - restored).abs().max())
        maximum = max(maximum, delta)
        row["max_abs_delta"] = max(row["max_abs_delta"], delta)
        row["nonzero_snapshots"] += int(bool(gradient.any()))
    return maximum


def verify(target):
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name, expected in contract["source_pins"].items():
        assert digest(ROOT / name) == expected, name
    summary = json.loads((ROOT / contract["update_summary"]).read_text(encoding="utf-8"))
    assert (
        summary["attempt_open"] == summary["attempt_close"] == 8
        and summary["optimizer_step_calls"] == 6
    )
    assert summary["checkpoint_coverage"] == 8
    torch.set_num_threads(1)
    parameter_rows, checks, workers = {}, {}, []
    with tempfile.TemporaryDirectory(
        prefix="u2-final-recovery-", dir=ROOT / ".seed_test_tmp"
    ) as temp:
        root = Path(temp)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.optim.Adam, "step", forbidden_step)
            patch.setattr(torch.optim.SGD, "step", forbidden_step)
            for precision in ["float64", "float32"]:
                jobs, controls = [], []
                for identity in contract["base_snapshots"]:
                    directory = root / (precision + "-" + identity)
                    runtime, fixture = snapshot(identity, getattr(torch, precision), directory)
                    path = runtime.save(directory / "snapshot.bin")
                    jobs.append(
                        {
                            "id": identity,
                            "directory": directory,
                            "checkpoint": path,
                            "fixture": fixture,
                        }
                    )
                    envelope = atomic_load(path)
                    control = U2Runtime.from_envelope(
                        envelope,
                        executor=isolated_executor(envelope, directory, "control"),
                    )
                    controls.append(endpoint(control, fixture, directory, identity))
                for item in contract["s12_snapshots"]:
                    if item["precision"] != precision:
                        continue
                    s12 = json.loads((ROOT / item["path"]).read_text(encoding="utf-8"))
                    worker_path = ROOT / item["worker"]
                    independent = json.loads(worker_path.read_text(encoding="utf-8"))
                    assert (
                        independent["verdict"] == "pass"
                        and independent["optimizer_step_calls"] == 0
                    )
                    path = ROOT / (s12["post_checkpoint"] or s12["parent_checkpoint"])
                    directory = root / (precision + "-" + item["id"])
                    directory.mkdir()
                    envelope = atomic_load(path)
                    control = U2Runtime.from_envelope(
                        envelope,
                        executor=isolated_executor(envelope, directory, "control"),
                    )
                    fixture = synthetic_fixture(control.model.c, getattr(torch, precision))
                    jobs.append(
                        {
                            "id": item["id"],
                            "directory": directory,
                            "checkpoint": path,
                            "fixture": fixture,
                        }
                    )
                    controls.append(endpoint(control, fixture, directory, item["id"]))
                input_path, output_path = root / (precision + "-jobs.bin"), root / (
                    precision + "-result.bin"
                )
                atomic_save(jobs, input_path)
                process = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=150,
                )
                if process.returncode:
                    raise RuntimeError(process.stdout + process.stderr)
                resumed = atomic_load(output_path)
                assert resumed["parent_pid"] == os.getpid() and resumed["pid"] != os.getpid()
                for job, control, row in zip(jobs, controls, resumed["rows"], strict=True):
                    assert row["id"] == job["id"]
                    maximum = compare(
                        control, row["measured"], contract["tolerances"][precision], parameter_rows
                    )
                    checks[precision + "-" + job["id"]] = {"passed": True, "max_abs_delta": maximum}
                workers.append({key: value for key, value in resumed.items() if key != "rows"})
        cpu = time.process_time() + sum(worker["cpu_seconds"] for worker in workers)
        wall = time.monotonic() - STARTED
        assert len(checks) == 34 and all(
            row["compared_snapshots"] == 34 for row in parameter_rows.values()
        )
        assert (
            cpu <= contract["budget"]["cpu_seconds"] and wall <= contract["budget"]["wall_seconds"]
        )
        for name, expected in contract["source_pins"].items():
            assert digest(ROOT / name) == expected, name
        initial = {}
        for precision in ["float64", "float32"]:
            model = U2Subject(U2Config.small()).to(dtype=getattr(torch, precision))
            initial[precision] = value_digest(model.state_dict())
        report = {
            "schema": "taiji-u2-22.3-final-closure-v1",
            "verdict": "passed_engineering_recovery_only",
            "contract_sha256": digest(CONTRACT),
            "source_pins": contract["source_pins"],
            "checks": checks,
            "parameter_comparisons": parameter_rows,
            "workers": workers,
            "accepted_parent": {
                "status": "accepted_engineering_initialization",
                "algorithm": "U2",
                "profile": "U2Config.small",
                "seed": 20261110,
                "device": "cpu",
                "precisions": ["float64", "float32"],
                "initial_parameter_sha256": initial,
                "runtime_schema": ENVELOPE_SCHEMA,
                "history_schema": HISTORY_SCHEMA,
                "subject_schema": "taiji-u2-subject-v2",
                "state_fields": 40,
                "parameter_tensors": 191,
                "capability_qualifications": [],
                "source_pins": contract["core_source_pins"],
            },
            "actual_optimizer_step_calls_in_22_3_4_3": 6,
            "additional_optimizer_step_calls": 0,
            "cpu_seconds_including_workers": cpu,
            "wall_seconds": wall,
            "whole_22_3_passed": True,
            "task_capability_evaluated": False,
            "project_dataset_reads": 0,
            "confirm_reads": 0,
            "run_conditions": run_conditions(model),
        }
        write_once(target, report)
    print(json.dumps({"verdict": report["verdict"], "snapshots": len(checks), "cpu_seconds": cpu}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        return worker(args.worker, args.output)
    if args.report is None:
        parser.error("A new report is required")
    target = args.report.resolve()
    if target.parent != ROOT / "reports" or target.exists():
        parser.error("Report must be a new file directly inside reports/")
    try:
        return verify(target)
    except Exception as error:
        failure = {
            "schema": "taiji-u2-closure-failure-v1",
            "verdict": "engineering_failure",
            "error": f"{type(error).__name__}: {error}",
            "threads": torch.get_num_threads(),
            "parent_cpu_seconds": time.process_time(),
            "wall_seconds": time.monotonic() - STARTED,
            "whole_22_3_passed": False,
        }
        if not target.exists():
            write_once(target, failure)
        print(json.dumps(failure, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
