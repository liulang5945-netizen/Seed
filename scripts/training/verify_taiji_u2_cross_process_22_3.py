"""U2 22.3.3 independent-process checks for the eleven zero-update families.

Usage: python scripts/training/verify_taiji_u2_cross_process_22_3.py --report reports/<new>.json
Each precision uses a fresh child process that restores every snapshot separately.
S12 is supplied by the later engineering-update card. Artifacts are disposable
synthetic checkpoints in .seed_test_tmp, and optimizer steps are prohibited.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest
import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.verify_taiji_u2_runtime_22_3 import digest
from taiji.u2_recovery import (
    ReferenceExecutor,
    U2Runtime,
    atomic_load,
    atomic_save,
    detached_tree,
    run_conditions,
    value_digest,
)
from taiji.u2_subject import U2Config, U2Packet, U2Subject, synthetic_fixture

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_3_cross_process_contract_20261111.json"


def legal_operation(index):
    mask = torch.zeros(9, dtype=torch.bool)
    mask[index] = True
    return mask


def packet_for(runtime, fixture, identity):
    start = float(runtime.state["world_time"])
    event = fixture["events"][min(len(runtime.state["history"]), len(fixture["events"]) - 1)]
    return U2Packet(
        identity,
        "sensor",
        "world",
        start,
        start + 0.05,
        start + 0.05,
        event["observations"],
        event["coordinates"],
        event["outcome"],
        runtime.state["pending"]["id"],
        feedback_record_ids=("red",),
    )


def prepare_action(runtime, directory):
    if runtime.state["pending"] is None:
        if runtime.last_decision is None:
            runtime.choose(legal_operation(0))
        runtime.issue(dt=0.05)
    runtime.execute_pending(checkpoint=directory / "action.bin")
    runtime.set_process_time(max(runtime.state["process_time"], runtime.state["world_time"] + 0.05))


def observe(runtime, fixture, directory, identity):
    prepare_action(runtime, directory)
    return runtime.receive(packet_for(runtime, fixture, identity))


def emit_actual(runtime, symbols):
    if not runtime.state["utterance_active"]:
        runtime.begin_utterance()
    for symbol in symbols:
        runtime.accept_byte(symbol)


def snapshot(case, dtype, directory):
    directory.mkdir()
    model = U2Subject(U2Config.small()).to(dtype=dtype)
    executor = ReferenceExecutor(directory / "executor.bin")
    runtime = U2Runtime(model, executor=executor, subject_id=case)
    fixture = synthetic_fixture(model.c, dtype)
    if case != "S01":
        runtime.set_goal(fixture["goal"])
        runtime.set_records(fixture["records"])
    if case not in {"S01", "S02", "S03"}:
        observe(runtime, fixture, directory, case + "-prefix")
    if case == "S03":
        runtime.choose(legal_operation(0))
    elif case == "S04":
        runtime.issue(runtime.choose(legal_operation(0)))
        try:
            runtime.execute_pending(
                checkpoint=directory / "intent.bin", fault="after_execute_before_ack"
            )
        except RuntimeError as error:
            assert "injected" in str(error)
        assert runtime.executor.query(runtime.state["pending"]["id"])["status"] == "applied"
    elif case in {"S05", "S06"}:
        prepare_action(runtime, directory)
        if case == "S05":
            runtime.set_goal(fixture["replacement_goal"])
        else:
            packet = packet_for(runtime, fixture, case + "-unknown")
            unknown = packet.outcome.clone()
            unknown[4:] = 0
            assert runtime.receive(replace(packet, outcome=unknown))["status"] == "awaiting_outcome"
    elif case == "S07":
        assert float(runtime.state["F"].detach().norm()) > 0
    elif case == "S08":
        emit_actual(runtime, [0xE7])
        assert runtime.state["utf8_remaining"] == 2
    elif case == "S09":
        emit_actual(runtime, [*fixture["target"], 256])
    elif case.startswith("S10"):
        runtime.internal(runtime.choose(legal_operation(4)))
        if case == "S10-wait":
            runtime.internal(runtime.choose(legal_operation(8)))
    elif case.startswith("S11"):
        for generation in range(1, (2 if case == "S11-second" else 1) + 1):
            credit = torch.zeros((model.c.groups, model.c.d, 8 * model.c.d), dtype=dtype)
            credit[:, 0, 0], credit[:, 1, 1] = 3, 2
            runtime.birth_transaction(credit, branch=generation - 1, birth_id=f"g{generation}")
    return runtime, fixture


def continue_and_measure(runtime, fixture, directory, case):
    if case == "S01":
        runtime.set_goal(fixture["goal"])
        runtime.set_records(fixture["records"])
    if runtime.state["utterance_active"]:
        emit_actual(runtime, [0xBA, 0xA2, 256])
    prepare_action(runtime, directory)
    packet = packet_for(runtime, fixture, case + "-continuation")
    receipt = runtime.receive(packet)
    utterance = runtime.model.begin_utterance(runtime.state)
    goal = runtime.model.goal(utterance["goal_raw"])
    terms = []
    for symbol in [*fixture["target"], 256]:
        log_probability, _, output = runtime.model.output(
            utterance, goal, utterance["prediction_action"]
        )
        terms.append(-log_probability[symbol])
        utterance = runtime.model.accept_byte(utterance, symbol, output)
    teacher_loss = torch.stack(terms).mean()
    loss = teacher_loss - receipt["predictive_logp"].sum() + receipt["critic"] + receipt["actor"]
    named = dict(runtime.model.named_parameters())
    gradients = torch.autograd.grad(loss, list(named.values()), allow_unused=True)
    return {
        "state_digest": value_digest(runtime.state),
        "parameter_digest": value_digest(runtime.model.state_dict()),
        "log_output": runtime.output()[0].detach(),
        "policy": runtime.model.policy(runtime.state)["log_policy"].detach(),
        "gradients": dict(zip(named, (detached_tree(value) for value in gradients), strict=True)),
        "settlements": runtime.state["settled"].count(packet.receipt_id),
        "effect_count": runtime.executor.effect_count,
        "run_conditions": run_conditions(runtime.model),
    }


def worker(jobs_path, output_path, threads):
    torch.set_num_threads(threads)
    jobs = atomic_load(jobs_path)
    rows = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", forbidden_step)
        patch.setattr(torch.optim.SGD, "step", forbidden_step)
        for job in jobs:
            try:
                runtime = U2Runtime.load(
                    job["checkpoint"], executor=ReferenceExecutor(job["executor"])
                )
                measured = continue_and_measure(
                    runtime, job["fixture"], Path(job["directory"]), job["id"]
                )
                rows.append({"id": job["id"], "passed": True, "measured": measured})
            except Exception as error:
                rows.append(
                    {"id": job["id"], "passed": False, "error": f"{type(error).__name__}: {error}"}
                )
    atomic_save(
        {
            "rows": rows,
            "cpu_seconds": time.process_time(),
            "wall_seconds": time.monotonic() - STARTED,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "snapshot_count": len(jobs),
        },
        output_path,
    )
    return 0


def forbidden_step(*args, **kwargs):
    raise AssertionError("Optimizer attempts belong to 22.3.4")


def launch(jobs, root, name, threads):
    input_path, output_path = root / (name + "-jobs.bin"), root / (name + "-result.bin")
    atomic_save(jobs, input_path)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(input_path),
            "--output",
            str(output_path),
            "--threads",
            str(threads),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return atomic_load(output_path)


def compare(before, after, tolerance, parameter_rows):
    for field in (
        "state_digest",
        "parameter_digest",
        "settlements",
        "effect_count",
        "run_conditions",
    ):
        assert before[field] == after[field], field
    for field in ("log_output", "policy"):
        assert torch.equal(before[field], after[field]), field
    assert before["settlements"] == 1
    maximum = 0.0
    for name, left in before["gradients"].items():
        right = after["gradients"][name]
        row = parameter_rows.setdefault(
            name,
            {
                "compared_snapshots": 0,
                "nonzero_snapshots": 0,
                "unused_snapshots": 0,
                "max_abs_delta": 0.0,
            },
        )
        row["compared_snapshots"] += 1
        if left is None:
            assert right is None, name
            row["unused_snapshots"] += 1
            continue
        assert right is not None
        assert bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()), name
        assert torch.allclose(left, right, **tolerance), name
        difference = float((left - right).abs().max())
        maximum = max(maximum, difference)
        row["max_abs_delta"] = max(row["max_abs_delta"], difference)
        row["nonzero_snapshots"] += int(bool(left.any()))
    return maximum


def verify(target):
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name, expected in contract["source_pins"].items():
        assert digest(ROOT / name) == expected, name
    torch.set_num_threads(1)
    rows, workers, parameter_rows = [], [], {}
    with tempfile.TemporaryDirectory(
        prefix="u2-cross-process-", dir=ROOT / ".seed_test_tmp"
    ) as work:
        root = Path(work)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.optim.Adam, "step", forbidden_step)
            patch.setattr(torch.optim.SGD, "step", forbidden_step)
            for precision in ("float64", "float32"):
                jobs, controls = [], []
                for case in contract["snapshots"]:
                    directory = root / (precision + "-" + case)
                    runtime, fixture = snapshot(case, getattr(torch, precision), directory)
                    checkpoint = runtime.save(directory / "subject.bin")
                    jobs.append(
                        {
                            "id": case,
                            "checkpoint": checkpoint,
                            "executor": runtime.executor.path,
                            "directory": directory,
                            "fixture": fixture,
                        }
                    )
                    controls.append(continue_and_measure(runtime, fixture, directory, case))
                resumed = launch(jobs, root, precision, 1)
                assert resumed["pid"] != os.getpid() and resumed["parent_pid"] == os.getpid()
                workers.append({key: value for key, value in resumed.items() if key != "rows"})
                for job, before, result in zip(jobs, controls, resumed["rows"], strict=True):
                    assert result["id"] == job["id"] and result["passed"], result
                    difference = compare(
                        before,
                        result["measured"],
                        contract["tolerances"][precision],
                        parameter_rows,
                    )
                    rows.append(
                        {
                            "precision": precision,
                            "snapshot": job["id"],
                            "passed": True,
                            "max_abs_delta": difference,
                        }
                    )
                wrong = launch(jobs[:1], root, precision + "-wrong-threads", 2)
                workers.append({key: value for key, value in wrong.items() if key != "rows"})
                assert len(wrong["rows"]) == 1 and not wrong["rows"][0]["passed"]
                assert "conditions mismatch" in wrong["rows"][0]["error"]
        cpu = time.process_time() + sum(item["cpu_seconds"] for item in workers)
        wall = time.monotonic() - STARTED
        assert (
            cpu <= contract["budget"]["cpu_seconds"] and wall <= contract["budget"]["wall_seconds"]
        )
        for name, expected in contract["source_pins"].items():
            assert digest(ROOT / name) == expected, name
        report = {
            "schema": "taiji-u2-22.3.3-cross-process-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "verdict": "passed_S01_S11_cross_process_only",
            "contract_sha256": digest(CONTRACT),
            "source_pins": contract["source_pins"],
            "run_conditions": run_conditions(U2Subject(U2Config.small()).double()),
            "parent_pid": os.getpid(),
            "checks": rows,
            "parameter_comparisons": parameter_rows,
            "workers": workers,
            "wrong_thread_negative_controls": 2,
            "cpu_seconds_including_workers": cpu,
            "wall_seconds": wall,
            "optimizer_attempts": 0,
            "training_steps": 0,
            "project_dataset_reads": 0,
            "existing_checkpoints_loaded": 0,
            "S12_verified": False,
            "post_update_semantic_rebuild_verified": False,
            "whole_22_3_passed": False,
            "task_capability_evaluated": False,
        }
        staging = write_report(root / "report.writing", report)
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(staging.read_text(encoding="utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"verdict": report["verdict"], "snapshots": len(rows), "cpu_seconds": cpu}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.worker is not None:
        if arguments.output is None:
            parser.error("A worker requires an output file")
        return worker(arguments.worker, arguments.output, arguments.threads)
    if arguments.report is None:
        parser.error("A new report path is required")
    target = arguments.report.resolve()
    if target.parent != ROOT / "reports" or target.exists():
        parser.error("Report must be a new file directly inside reports/.")
    try:
        return verify(target)
    except Exception as error:
        failure = {
            "schema": "taiji-u2-22.3.3-cross-process-failure-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "verdict": "engineering_failure",
            "error": f"{type(error).__name__}: {error}",
            "contract_sha256": digest(CONTRACT),
            "threads": torch.get_num_threads(),
            "parent_cpu_seconds": time.process_time(),
            "wall_seconds": time.monotonic() - STARTED,
            "worker_resource_accounting": "incomplete_on_failure",
            "whole_22_3_passed": False,
        }
        if not target.exists():
            write_report(target, failure)
        print(json.dumps(failure, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
