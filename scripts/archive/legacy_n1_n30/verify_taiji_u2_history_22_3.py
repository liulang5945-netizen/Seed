"""Verify cross-window U2 raw history before slow-parameter migration.

Usage: python scripts/training/verify_taiji_u2_history_22_3.py --report reports/<new>.json
All binary artifacts are freshly generated in an isolated test directory. The
verification includes normal-entry regressions and float32/64 child processes,
with optimizer updates prohibited in both parent and workers.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest
import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.verify_taiji_u2_cross_process_22_3 import (
    compare,
    continue_and_measure,
    forbidden_step,
    snapshot,
)
from scripts.training.verify_taiji_u2_runtime_22_3 import cases, digest, execute_checks, load_checks
from taiji import u2_history
from taiji.u2_recovery import ReferenceExecutor, U2Runtime, atomic_load, atomic_save, run_conditions
from taiji.u2_subject import U2Config, U2Subject

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_3_history_contract_20261111.json"
HISTORY_TESTS = ROOT / "tests/taiji_native/test_u2_history_22_3.py"


def load_history_checks():
    spec = importlib.util.spec_from_file_location("u2_history_checks", HISTORY_TESTS)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    return checks


def measure(runtime, fixture, directory, identity):
    measured = continue_and_measure(runtime, fixture, directory, identity)
    measured.update(
        history_reference=runtime.history.reference(),
        history_bytes=runtime.history.total_bytes,
        history_complete=runtime.history.complete,
    )
    return measured


def worker(input_path, output_path):
    torch.set_num_threads(1)
    job = atomic_load(input_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", forbidden_step)
        patch.setattr(torch.optim.SGD, "step", forbidden_step)
        runtime = U2Runtime.load(job["checkpoint"], executor=ReferenceExecutor(job["executor"]))
        assert runtime.history.reference() == job["prefix"]
        result = measure(runtime, job["fixture"], Path(job["directory"]), job["id"])
    atomic_save(
        {
            "measured": result,
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "cpu_seconds": time.process_time(),
            "wall_seconds": time.monotonic() - STARTED,
            "run_conditions": run_conditions(runtime.model),
        },
        output_path,
    )
    return 0


def cross_process_checks(root, contract):
    results, workers, parameter_rows = [], [], {}
    for precision, case in (("float64", "S11-second"), ("float32", "S08")):
        directory = root / precision
        runtime, fixture = snapshot(case, getattr(torch, precision), directory)
        if precision == "float64":
            runtime.begin_new_window()
        checkpoint = runtime.save(directory / "raw-history.bin")
        prefix = runtime.history.reference()
        job = {
            "id": "history-" + precision,
            "checkpoint": checkpoint,
            "executor": runtime.executor.path,
            "directory": directory,
            "fixture": fixture,
            "prefix": prefix,
        }
        original = measure(runtime, fixture, directory, job["id"])
        input_path, output_path = directory / "job.bin", directory / "result.bin"
        atomic_save(job, input_path)
        child = subprocess.run(
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
            timeout=60,
        )
        if child.returncode:
            raise RuntimeError(child.stdout + child.stderr)
        resumed = atomic_load(output_path)
        assert resumed["pid"] != os.getpid() and resumed["parent_pid"] == os.getpid()
        after = resumed.pop("measured")
        difference = compare(original, after, contract["tolerances"][precision], parameter_rows)
        for field in ("history_reference", "history_bytes", "history_complete"):
            assert original[field] == after[field], field
        assert after["history_complete"]
        runtime.history.validate_reference(prefix)
        workers.append(resumed)
        results.append(
            {
                "precision": precision,
                "snapshot": case,
                "saved_prefix": prefix,
                "resumed_history": after["history_reference"],
                "raw_payload_bytes": after["history_bytes"],
                "max_abs_delta": difference,
                "passed": True,
            }
        )
    return results, workers, parameter_rows


def persistence_helper_equivalence():
    previous = subprocess.run(
        ["git", "show", "cc55752d7:taiji/u2_recovery.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    functions = {
        node.name: node for node in ast.parse(previous).body if isinstance(node, ast.FunctionDef)
    }
    names = (
        "detached_tree",
        "assert_values_equal",
        "_hash_value",
        "value_digest",
        "atomic_save",
        "atomic_load",
    )
    for name in names:
        current = ast.parse(inspect.getsource(getattr(u2_history, name))).body[0]
        assert ast.dump(functions[name]) == ast.dump(current), name
    return {"historical_revision": "cc55752d7", "functions": list(names), "ast_equal": True}


def verify(target):
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name, expected in contract["source_pins"].items():
        assert digest(ROOT / name) == expected, name
    torch.set_num_threads(1)
    check_modules = {"runtime": load_checks(), "history": load_history_checks()}
    assert {
        name: [identity for identity, _, _ in cases(module)]
        for name, module in check_modules.items()
    } == contract["expected_checks"]
    with tempfile.TemporaryDirectory(
        prefix="u2-history-22-3-", dir=ROOT / ".seed_test_tmp"
    ) as work:
        root = Path(work)
        results = []
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.optim.Adam, "step", forbidden_step)
            patch.setattr(torch.optim.SGD, "step", forbidden_step)
            for family, module in check_modules.items():
                directory = root / family
                directory.mkdir()
                results.extend(
                    {"family": family, **row} for row in execute_checks(module, directory)
                )
            assert all(row["passed"] for row in results), [
                row for row in results if not row["passed"]
            ]
            processes, workers, parameters = cross_process_checks(root, contract)
            equivalent = persistence_helper_equivalence()
        parent_cpu = time.process_time()
        cpu = parent_cpu + sum(row["cpu_seconds"] for row in workers)
        wall = time.monotonic() - STARTED
        assert (
            cpu <= contract["budget"]["cpu_seconds"] and wall <= contract["budget"]["wall_seconds"]
        )
        for name, expected in contract["source_pins"].items():
            assert digest(ROOT / name) == expected, name
        report = {
            "schema": "taiji-u2-22.3.4.1-raw-history-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "verdict": "passed_raw_history_only",
            "plan_revision_before_implementation": contract["plan_revision"],
            "contract_sha256": digest(CONTRACT),
            "source_pins": contract["source_pins"],
            "run_conditions": run_conditions(U2Subject(U2Config.small()).double()),
            "checks": results,
            "cross_process": processes,
            "parameter_comparisons": parameters,
            "workers": workers,
            "persistence_helper_equivalence": equivalent,
            "parent_cpu_seconds": parent_cpu,
            "cpu_seconds_including_workers": cpu,
            "wall_seconds": wall,
            "optimizer_attempts": 0,
            "training_steps": 0,
            "project_dataset_reads": 0,
            "existing_checkpoints_loaded": 0,
            "post_update_semantic_rebuild_verified": False,
            "S12_verified": False,
            "whole_22_3_passed": False,
            "task_capability_evaluated": False,
        }
        staging = write_report(root / "report.writing", report)
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(staging.read_text(encoding="utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"verdict": report["verdict"], "checks": len(results), "cpu_seconds": cpu}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.worker is not None:
        if arguments.output is None:
            parser.error("A worker requires output")
        return worker(arguments.worker, arguments.output)
    if arguments.report is None:
        parser.error("A new report is required")
    target = arguments.report.resolve()
    if target.parent != ROOT / "reports" or target.exists():
        parser.error("Report must be a new file directly inside reports/.")
    try:
        return verify(target)
    except Exception as error:
        failure = {
            "schema": "taiji-u2-history-failure-v1",
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
