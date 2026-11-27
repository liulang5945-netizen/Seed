"""Bind physical effects, paged history and per-loss credit across processes."""

# ruff: noqa: E402 -- measure startup before importing the tensor runtime
from __future__ import annotations

import argparse
import hashlib
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

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import atomic_load, atomic_save, detached_tree, value_digest
from taiji.u2_recovery import U2Runtime
from taiji.u2_subject import U2Config, U2Subject
from taiji.u2_task_executor import PhysicalExecutor
from taiji.u2_task_world import PhysicalWorld

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_4_2_runtime_contract_20261112.json"
REPORT = ROOT / "reports/taiji_u2_physical_runtime_22_4_2_20261112.json"


def measure(runtime, receipt):
    state = runtime.model.begin_utterance(runtime.state)
    goal = runtime.model.goal(state["goal_raw"])
    terms = []
    for symbol in [*"观察".encode(), 256]:
        probability, _, output = runtime.model.output(state, goal, state["prediction_action"])
        terms.append(-probability[symbol])
        state = runtime.model.accept_byte(state, symbol, output)
    losses = {
        "expression": torch.stack(terms).mean(),
        "predictive": -receipt["predictive_logp"].sum(),
        "actor": receipt["actor"],
        "critic": receipt["critic"],
    }
    named = dict(runtime.model.named_parameters())
    gradients = {}
    for name, loss in losses.items():
        gradients[name] = dict(
            zip(
                named,
                detached_tree(
                    torch.autograd.grad(
                        loss, list(named.values()), allow_unused=True, retain_graph=True
                    )
                ),
                strict=True,
            )
        )
    return {
        "state": value_digest(runtime.state),
        "history": runtime.history.reference(),
        "parameters": value_digest(runtime.model.state_dict()),
        "world": value_digest(runtime.executor.world.snapshot()),
        "log_output": runtime.output()[0].detach(),
        "policy": runtime.model.policy(runtime.state)["log_policy"].detach(),
        "gradients": gradients,
        "losses": {k: float(v.detach()) for k, v in losses.items()},
    }


def act(runtime, directory):
    mask = torch.zeros(9, dtype=torch.bool)
    mask[0] = True
    runtime.issue(runtime.choose(mask), dt=0.05)
    runtime.execute_pending(directory / "intent.bin")


def worker(jobs_path, output):
    torch.set_num_threads(1)
    rows = []
    with StepGuard() as guard:
        for job in atomic_load(jobs_path):
            runtime = U2Runtime.load(job["checkpoint"])
            restored = value_digest(runtime.state)
            act(runtime, Path(job["directory"]))
            runtime.set_process_time(job["packet"].t_arrival)
            receipt = runtime.receive(job["packet"])
            rows.append(
                {"id": job["id"], "restored": restored, "measured": measure(runtime, receipt)}
            )
    atomic_save(
        {
            "rows": rows,
            "pid": os.getpid(),
            "optimizer_steps": guard.calls,
            "cpu_seconds": time.process_time(),
        },
        output,
    )


def compare(left, right):
    rows = []
    for name in left.keys() - {"gradients"}:
        if isinstance(left[name], torch.Tensor):
            assert torch.equal(left[name], right[name]), name
        else:
            assert left[name] == right[name], name
    for loss, values in left["gradients"].items():
        for name, value in values.items():
            peer = right["gradients"][loss][name]
            if value is None:
                assert peer is None
                status = "unused"
            else:
                assert (
                    peer is not None
                    and torch.equal(value, peer)
                    and bool(torch.isfinite(value).all())
                ), (loss, name)
                status = "nonzero" if bool(value.any()) else "zero"
            rows.append({"loss": loss, "parameter": name, "status": status, "max_delta": 0.0})
    return rows


def verify():
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for path, expected in contract["source_pins"].items():
        assert (
            hashlib.sha256((ROOT / path).read_text(encoding="utf-8-sig").encode()).hexdigest()
            == expected
        ), path
    jobs, original, points = [], {}, {}
    with (
        tempfile.TemporaryDirectory(
            dir=ROOT / ".seed_test_tmp", prefix="u2-physical-subject-"
        ) as directory,
        StepGuard() as guard,
    ):
        root = Path(directory)
        for precision in ("float32", "float64"):
            case = root / precision
            case.mkdir()
            base = U2Subject(U2Config.small()).to(dtype=getattr(torch, precision))
            assert value_digest(base.state_dict()) == contract["base_initial_parameters"][precision]
            executor = PhysicalExecutor.create(
                PhysicalWorld.reset(20261112, {"body_count": 1, "dtype": precision}),
                case / "world.bin",
            )
            runtime = U2Runtime(
                base.metadata_candidate(), executor=executor, subject_id="physical-" + precision
            )
            runtime.set_goal("逐字输出观察。")
            for index in range(25):
                if runtime.operation_count > 24:
                    runtime.begin_new_window()
                runtime.begin_observation(f"observe-{index}", 0 if index == 0 else 0.05)
                runtime.execute_pending(case / "intent.bin")
                runtime.receive(executor.packet(runtime))
            runtime.begin_new_window()
            runtime.save(case / "subject.bin")
            points[precision] = {
                "state": value_digest(runtime.state),
                "entries": runtime.history.entry_count,
                "pages": len(runtime.history.export()["pages"]),
            }
            assert runtime.history.entry_count > 96
            act(runtime, case)
            packet = executor.packet(runtime)
            receipt = runtime.receive(packet)
            original[precision] = measure(runtime, receipt)
            jobs.append(
                {
                    "id": precision,
                    "checkpoint": str(case / "subject.bin"),
                    "directory": str(case),
                    "packet": packet,
                }
            )
        atomic_save(jobs, root / "jobs.bin")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.training.verify_taiji_u2_physical_runtime_22_4_2",
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
        assert result.returncode == 0, result.stderr
        recovered = atomic_load(root / "worker.bin")
        assert recovered["pid"] != os.getpid() and recovered["optimizer_steps"] == 0
        comparisons = {}
        for row in recovered["rows"]:
            assert row["restored"] == points[row["id"]]["state"]
            comparisons[row["id"]] = compare(original[row["id"]], row["measured"])
    report = {
        "schema": "taiji-u2-physical-subject-verification-v1",
        "verdict": "passed_physical_subject_recovery_credit",
        "contract_sha256": hashlib.sha256(
            CONTRACT.read_text(encoding="utf-8-sig").encode()
        ).hexdigest(),
        "checkpoints": points,
        "gradient_rows": comparisons,
        "max_abs_gradient_delta": 0.0,
        "optimizer_steps": guard.calls,
        "project_data_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": time.process_time() + recovered["cpu_seconds"],
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {
            "device": "cpu",
            "threads": 1,
            "torch": torch.__version__,
            "precisions": ["float32", "float64"],
        },
        "scope": "Shared canonical physical store, controlled redelivery of the same actual sensor packet, same sampled actor action and four per-parameter loss maps. Expression target is the explicit literal instruction. Task oracle and capability remain pending.",
    }
    assert (
        report["cpu_seconds"] <= contract["cpu_cap"]
        and report["wall_seconds"] <= contract["wall_cap"]
    )
    if REPORT.exists():
        raise FileExistsError(REPORT)
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "gradient_rows": sum(map(len, comparisons.values())),
                "cpu_seconds": report["cpu_seconds"],
                "wall_seconds": report["wall_seconds"],
            }
        )
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
