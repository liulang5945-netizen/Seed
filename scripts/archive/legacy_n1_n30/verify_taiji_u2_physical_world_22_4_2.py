"""Frozen pure-world implementation checks with a fresh recovery process."""

# ruff: noqa: E402 -- capture startup time and support direct repository execution

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
from taiji.u2_history import atomic_load, atomic_save, value_digest
from taiji.u2_task_world import PhysicalWorld

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_4_2_world_implementation_20261112_v2.json"
REPORT = ROOT / "reports/taiji_u2_physical_world_22_4_2_20261112.json"


def measure(world):
    results = []
    for index in range(3):
        action = torch.zeros(15, dtype=torch.float64)
        action[2 if index == 0 else 0] = 1
        action[9:11] = torch.tensor(world.snapshot()["bodies"][0]["position"])
        action[13:15] = torch.tensor([0.7, 0.03])
        if index:
            action[11] = -0.5
        result = world.step(f"continuation-{index}", action)
        results.append(value_digest(result))
    return {"results": results, "snapshot": value_digest(world.snapshot())}


def worker(jobs_path, output):
    torch.set_num_threads(1)
    rows = []
    for job in atomic_load(jobs_path):
        world = PhysicalWorld.restore(job["snapshot"])
        rows.append({"id": job["id"], "measured": measure(world)})
    atomic_save({"rows": rows, "pid": os.getpid(), "cpu_seconds": time.process_time()}, output)


def verify():
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name, expected in contract["source_pins"].items():
        assert (
            hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").encode()).hexdigest()
            == expected
        ), name
    with tempfile.TemporaryDirectory(
        dir=ROOT / ".seed_test_tmp", prefix="u2-physical-world-"
    ) as directory:
        base = Path(directory)
        jobs, original = [], {}
        for dtype in ("float32", "float64"):
            for count in (1, 2, 3):
                identity = f"{dtype}-{count}"
                world = PhysicalWorld.reset(20261112, {"dtype": dtype, "body_count": count})
                world.initial_observation("initial")
                world.elapse("prefix")
                jobs.append({"id": identity, "snapshot": world.snapshot()})
                original[identity] = measure(world)
        atomic_save(jobs, base / "jobs.bin")
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.training.verify_taiji_u2_physical_world_22_4_2",
                "--worker",
                str(base / "jobs.bin"),
                "--output",
                str(base / "result.bin"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert process.returncode == 0, process.stderr
        recovered = atomic_load(base / "result.bin")
        assert recovered["pid"] != os.getpid()
        for row in recovered["rows"]:
            assert row["measured"] == original[row["id"]], row["id"]
        started = time.monotonic()
        test = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/taiji_native/test_u2_task_world_22_4_2.py",
                "-q",
                "--basetemp=" + str(base / "pytest"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert test.returncode == 0, test.stdout + test.stderr
        assert "24 passed" in test.stdout
        report = {
            "schema": "taiji-u2-physical-world-implementation-v1",
            "verdict": "passed_pure_world_implementation",
            "contract_sha256": hashlib.sha256(
                CONTRACT.read_text(encoding="utf-8-sig").encode()
            ).hexdigest(),
            "cross_process_cases": len(jobs),
            "matching_continuations": original,
            "tests": {
                "passed": 24,
                "stdout": test.stdout,
                "wall_seconds": time.monotonic() - started,
            },
            "reachability": {
                "fixtures": 9,
                "rounds_each": 16,
                "controller": "privileged physical-state PD witness",
                "qualification": "mechanical reachability; legal-history oracle awaits 22.4.2.4",
            },
            "model_forwards": 0,
            "optimizer_steps": 0,
            "project_data_reads": 0,
            "cpu_seconds_parent_and_recovery": time.process_time() + recovered["cpu_seconds"],
            "wall_seconds": time.monotonic() - STARTED,
            "run_conditions": {
                "torch": torch.__version__,
                "device": "cpu",
                "threads": 1,
                "codec_precisions": ["float32", "float64"],
                "world_precision": "float64",
            },
            "pending": [
                "durable physical executor and full subject integration",
                "exogenous observation intents",
                "segmented subject history",
                "legal-history task oracle",
                "learning and behavior",
            ],
        }
        assert report["wall_seconds"] <= 200
        if REPORT.exists():
            raise FileExistsError(REPORT)
        write_report(REPORT, report)
        print(json.dumps({"verdict": report["verdict"], "cross_process_cases": 6, "tests": 24}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
    else:
        verify()
