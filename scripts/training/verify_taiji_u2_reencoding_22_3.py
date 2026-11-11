"""Verify U2 fixed-history reencoding with zero optimizer updates.

Usage: python scripts/training/verify_taiji_u2_reencoding_22_3.py --report reports/<new>.json
Inputs are freshly generated engineering fixtures. Workers verify post-migration
continuation, while historical source verifies the receive refactor's numerics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import types
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
)
from scripts.training.verify_taiji_u2_history_22_3 import load_history_checks
from scripts.training.verify_taiji_u2_runtime_22_3 import cases, digest, execute_checks, load_checks
from taiji.u2_history import atomic_load, atomic_save, value_digest
from taiji.u2_recovery import ReferenceExecutor, U2Runtime, run_conditions
from taiji.u2_subject import U2Config, U2Subject, run_reference_sequence, synthetic_fixture

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_3_reencoding_contract_20261111.json"
TESTS = ROOT / "tests/taiji_native/test_u2_reencoding_22_3.py"
PROVENANCE_FIELDS = (
    "source_packet_id",
    "source_observation_packet_id",
    "source_receipt_id",
    "source_subject_version",
)


def load_reencoding_checks():
    spec = importlib.util.spec_from_file_location("u2_reencoding_checks", TESTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receive_equivalence():
    source = subprocess.run(
        ["git", "show", "d8993ecc8:taiji/u2_subject.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    legacy = types.ModuleType("u2_before_assimilation_extraction")
    sys.modules[legacy.__name__] = legacy
    rows = []
    try:
        exec(compile(source, "u2_subject.py@d8993ecc8", "exec"), legacy.__dict__)
        for precision in ("float64", "float32"):
            dtype = getattr(torch, precision)
            before = legacy.U2Subject(legacy.U2Config.small()).to(dtype=dtype)
            after = U2Subject(U2Config.small()).to(dtype=dtype)
            old_loss, old_state, _, _ = legacy.run_reference_sequence(
                before, legacy.synthetic_fixture(before.c, dtype)
            )
            new_loss, new_state, _, _ = run_reference_sequence(
                after, synthetic_fixture(after.c, dtype)
            )
            assert all(set(PROVENANCE_FIELDS).issubset(row) for row in new_state["history"])
            new_state["history"] = [
                {name: value for name, value in row.items() if name not in PROVENANCE_FIELDS}
                for row in new_state["history"]
            ]
            assert value_digest(old_state) == value_digest(new_state)
            assert value_digest(old_loss) == value_digest(new_loss)
            original_gradients = torch.autograd.grad(
                old_loss["total"], list(before.parameters()), allow_unused=True
            )
            current_gradients = torch.autograd.grad(
                new_loss["total"], list(after.parameters()), allow_unused=True
            )
            parameters = []
            for (name, _), left, right in zip(
                after.named_parameters(), original_gradients, current_gradients, strict=True
            ):
                if left is None:
                    assert right is None, name
                    parameters.append({"parameter": name, "status": "unused_by_endpoint"})
                else:
                    assert right is not None and torch.equal(left, right), name
                    parameters.append(
                        {"parameter": name, "max_abs_delta": float((left - right).abs().max())}
                    )
            rows.append(
                {
                    "precision": precision,
                    "historical_revision": "d8993ecc8",
                    "losses_equal": True,
                    "state_values_equal_except_additive_provenance": True,
                    "parameter_gradients": parameters,
                }
            )
    finally:
        del sys.modules[legacy.__name__]
    return rows


def worker(input_path, output_path):
    torch.set_num_threads(1)
    job = atomic_load(input_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", forbidden_step)
        patch.setattr(torch.optim.SGD, "step", forbidden_step)
        runtime = U2Runtime.load(job["checkpoint"], executor=ReferenceExecutor(job["executor"]))
        assert runtime.history.reference() == job["prefix"]
        measured = continue_and_measure(runtime, job["fixture"], Path(job["directory"]), job["id"])
        measured["history_reference"] = runtime.history.reference()
    atomic_save(
        {
            "measured": measured,
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "cpu_seconds": time.process_time(),
            "wall_seconds": time.monotonic() - STARTED,
            "run_conditions": run_conditions(runtime.model),
        },
        output_path,
    )
    return 0


def migration_metrics(root, checks, contract):
    rows, workers, parameters = [], [], {}
    for precision in ("float64", "float32"):
        directory = root / precision
        directory.mkdir()
        runtime, fixture = checks.settled(directory, precision, rich=True)
        old_state, old_prefix = runtime.state, runtime.history.reference()
        effects = runtime.executor.effect_count
        first = runtime.migrate_parameters(
            checks.candidate_from(runtime), acceptance=lambda model, state, opt: True
        )
        changed = {
            "H_max_abs_change": float((old_state["H"] - runtime.state["H"]).abs().max()),
            "F_max_abs_change": float((old_state["F"] - runtime.state["F"]).abs().max()),
            "keys_max_abs_change": float(
                (old_state["history"][0]["keys"] - runtime.state["history"][0]["keys"]).abs().max()
            ),
        }
        assert all(value > 0 for value in changed.values())
        assert runtime.executor.effect_count == effects
        runtime.history.validate_reference(old_prefix)
        second = runtime.migrate_parameters(
            checks.candidate_from(runtime), acceptance=lambda model, state, opt: True
        )
        checkpoint = runtime.save(directory / "accepted-migration.bin")
        prefix = runtime.history.reference()
        job = {
            "id": "migration-" + precision,
            "checkpoint": checkpoint,
            "executor": runtime.executor.path,
            "directory": directory,
            "fixture": fixture,
            "prefix": prefix,
        }
        control = continue_and_measure(runtime, fixture, directory, job["id"])
        control["history_reference"] = runtime.history.reference()
        input_path, output_path = directory / "job.bin", directory / "worker-result.bin"
        atomic_save(job, input_path)
        result = subprocess.run(
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
            timeout=90,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        resumed = atomic_load(output_path)
        assert resumed["pid"] != os.getpid() and resumed["parent_pid"] == os.getpid()
        measured = resumed.pop("measured")
        maximum = compare(control, measured, contract["tolerances"][precision], parameters)
        assert control["history_reference"] == measured["history_reference"]
        workers.append(resumed)
        rows.append(
            {
                "precision": precision,
                "first_migration": first,
                "second_migration": second,
                "representation_changes": changed,
                "prior_prefix": old_prefix,
                "post_migration_prefix": prefix,
                "max_abs_delta": maximum,
                "historical_executor_effects_preserved": effects,
                "passed": True,
            }
        )
    return rows, workers, parameters


def verify(target):
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name, expected in contract["source_pins"].items():
        assert digest(ROOT / name) == expected, name
    torch.set_num_threads(1)
    modules = {
        "runtime": load_checks(),
        "history": load_history_checks(),
        "reencoding": load_reencoding_checks(),
    }
    assert {
        name: [identity for identity, _, _ in cases(module)] for name, module in modules.items()
    } == contract["expected_checks"]
    with tempfile.TemporaryDirectory(prefix="u2-reencoding-", dir=ROOT / ".seed_test_tmp") as work:
        root = Path(work)
        checks = []
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.optim.Adam, "step", forbidden_step)
            patch.setattr(torch.optim.SGD, "step", forbidden_step)
            for family, module in modules.items():
                directory = root / family
                directory.mkdir()
                checks.extend(
                    {"family": family, **row} for row in execute_checks(module, directory)
                )
            assert all(row["passed"] for row in checks), [
                row for row in checks if not row["passed"]
            ]
            equivalence = receive_equivalence()
            processes, workers, parameters = migration_metrics(
                root, modules["reencoding"], contract
            )
        cpu = time.process_time() + sum(worker["cpu_seconds"] for worker in workers)
        wall = time.monotonic() - STARTED
        assert (
            cpu <= contract["budget"]["cpu_seconds"] and wall <= contract["budget"]["wall_seconds"]
        )
        for name, expected in contract["source_pins"].items():
            assert digest(ROOT / name) == expected, name
        report = {
            "schema": "taiji-u2-22.3.4.2-reencoding-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "verdict": "passed_zero_optimizer_reencoding_only",
            "contract_sha256": digest(CONTRACT),
            "design_sha256": contract["design_sha256"],
            "source_pins": contract["source_pins"],
            "run_conditions": run_conditions(U2Subject(U2Config.small()).double()),
            "checks": checks,
            "receive_equivalence": equivalence,
            "migration_processes": processes,
            "workers": workers,
            "parameter_comparisons": parameters,
            "cpu_seconds_including_workers": cpu,
            "wall_seconds": wall,
            "optimizer_attempts": 0,
            "training_steps": 0,
            "project_dataset_reads": 0,
            "existing_checkpoints_loaded": 0,
            "S12_actual_optimizer_verified": False,
            "whole_22_3_passed": False,
            "task_capability_evaluated": False,
        }
        staging = write_report(root / "report.writing", report)
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(staging.read_text(encoding="utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"verdict": report["verdict"], "checks": len(checks), "cpu_seconds": cpu}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.worker is not None:
        if arguments.output is None:
            parser.error("Worker output is required")
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
            "schema": "taiji-u2-reencoding-failure-v1",
            "verdict": "engineering_failure",
            "error": f"{type(error).__name__}: {error}",
            "threads": torch.get_num_threads(),
            "contract_sha256": digest(CONTRACT),
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
