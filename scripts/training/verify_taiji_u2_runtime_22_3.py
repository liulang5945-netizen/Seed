"""Verify U2 22.3.2 finite-window recovery without optimizer updates.

Usage: python scripts/training/verify_taiji_u2_runtime_22_3.py --report reports/<new>.json
The frozen contract pins the implementation and every regression case. All binary
artifacts are generated inside an isolated repository test directory and removed
on completion. Existing reports, weights and project datasets are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import itertools
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
from taiji.u2_recovery import U2Runtime, atomic_load, run_conditions, value_digest
from taiji.u2_subject import U2Config, U2Subject, run_reference_sequence, synthetic_fixture

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_3_window_contract_20261111.json"
TESTS = ROOT / "tests/taiji_native/test_u2_runtime_recovery_22_3.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def load_checks():
    spec = importlib.util.spec_from_file_location("u2_runtime_checks", TESTS)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    return checks


def cases(checks):
    for name in sorted(key for key in vars(checks) if key.startswith("test_")):
        function = getattr(checks, name)
        marks = [mark for mark in getattr(function, "pytestmark", []) if mark.name == "parametrize"]
        combinations = itertools.product(*(mark.args[1] for mark in marks))
        for values in combinations:
            parameters = dict(zip((mark.args[0] for mark in marks), values, strict=True))
            suffix = f"[{'-'.join(map(str, values))}]" if values else ""
            yield name + suffix, function, parameters


def execute_checks(checks, root):
    results = []
    for index, (identity, function, parameters) in enumerate(cases(checks)):
        directory = root / f"case-{index:02d}"
        directory.mkdir()
        cpu, wall = time.process_time(), time.monotonic()
        with pytest.MonkeyPatch.context() as patch:
            arguments = {"tmp_path": directory, **parameters}
            if "monkeypatch" in inspect.signature(function).parameters:
                arguments["monkeypatch"] = patch
            try:
                function(**arguments)
                result = {"id": identity, "passed": True}
            except Exception as error:
                result = {
                    "id": identity,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        result.update(
            source_line=inspect.getsourcelines(function)[1],
            cpu_seconds=time.process_time() - cpu,
            wall_seconds=time.monotonic() - wall,
        )
        results.append(result)
    return results


def replay_metrics(checks, root, tolerance):
    root.mkdir()
    runtime, _, packet = checks.pending_after_one_observation(root)
    checkpoint = root / "pending.bin"
    runtime.save(checkpoint)
    effects = runtime.executor.effect_count
    envelope = atomic_load(checkpoint)
    restored = U2Runtime.from_envelope(envelope, executor=runtime.executor)
    assert runtime.executor.effect_count == effects
    assert restored.state["process_time"] == runtime.state["process_time"]
    assert restored.host_clock["epoch"] != runtime.host_clock["epoch"]
    live_receipt, restored_receipt = runtime.receive(packet), restored.receive(packet)
    assert value_digest(runtime.state) == value_digest(restored.state)
    names, parameters = zip(*runtime.model.named_parameters(), strict=True)
    restored_parameters = dict(restored.model.named_parameters())
    left_loss = (
        -live_receipt["predictive_logp"].sum() + live_receipt["critic"] + live_receipt["actor"]
    )
    right_loss = (
        -restored_receipt["predictive_logp"].sum()
        + restored_receipt["critic"]
        + restored_receipt["actor"]
    )
    before = torch.autograd.grad(left_loss, parameters, allow_unused=True, retain_graph=True)
    after = torch.autograd.grad(
        right_loss,
        [restored_parameters[name] for name in names],
        allow_unused=True,
        retain_graph=True,
    )
    gradients = []
    for name, left, right in zip(names, before, after, strict=True):
        if left is None:
            assert right is None, name
            gradients.append({"parameter": name, "status": "unused_by_this_endpoint"})
            continue
        assert right is not None and bool(torch.isfinite(left).all())
        assert bool(torch.isfinite(right).all())
        assert torch.allclose(left, right, **tolerance), name
        gradients.append(
            {
                "parameter": name,
                "status": "matched",
                "norm_before": float(left.norm()),
                "norm_after": float(right.norm()),
                "max_abs_delta": float((left - right).abs().max()),
            }
        )
    values_model, values_state = U2Subject.from_bundle(envelope["subject_bundle"])
    _, values_receipt = values_model.receive(values_state, packet)
    values_motion = torch.autograd.grad(
        values_receipt["predictive_logp"].sum(), values_model.motion.first.weight, allow_unused=True
    )[0]
    live_motion = torch.autograd.grad(
        live_receipt["predictive_logp"].sum(), runtime.model.motion.first.weight, allow_unused=True
    )[0]
    assert values_motion is None and live_motion is not None and float(live_motion.norm()) > 0
    assert runtime.executor.effect_count == effects
    assert restored.receive(packet) == {}
    assert restored.state["settled"].count(packet.receipt_id) == 1
    return {
        "run_conditions": run_conditions(runtime.model),
        "serialized_file_bytes": checkpoint.stat().st_size,
        "window_operations_at_save": len(envelope["window"]["operations"]),
        "state_digest": value_digest(runtime.state),
        "all_parameter_gradient_comparisons": gradients,
        "compared_parameter_count": len(gradients),
        "nonzero_gradient_parameters": sum(row.get("norm_before", 0) > 0 for row in gradients),
        "max_abs_delta": max(row.get("max_abs_delta", 0) for row in gradients),
        "motion_gradient_norm": float(live_motion.norm()),
        "values_only_motion_gradient_missing": values_motion is None,
        "executor_effects_before_restore": effects,
        "executor_effects_after_replay_and_settlement": runtime.executor.effect_count,
        "duplicate_outcome_settlements": restored.state["settled"].count(packet.receipt_id),
        "new_host_epoch_same_logical_time": True,
        "credit_scope": "same_frozen_parameters_complete_declared_finite_window",
    }


def historical_type_equivalence(revision, expected_sha):
    source = subprocess.run(
        ["git", "show", f"{revision}:taiji/u2_subject.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    assert hashlib.sha256(source.encode()).hexdigest() == expected_sha
    legacy = types.ModuleType("u2_subject_before_runtime_type_cleanup")
    sys.modules[legacy.__name__] = legacy
    try:
        exec(compile(source, "taiji/u2_subject.py@" + revision, "exec"), legacy.__dict__)
        original = legacy.U2Subject(legacy.U2Config.small()).double()
        current = U2Subject(U2Config.small()).double()
        assert value_digest(original.state_dict()) == value_digest(current.state_dict())
        old_loss, old_state, _, _ = legacy.run_reference_sequence(
            original, legacy.synthetic_fixture(original.c, torch.float64)
        )
        new_loss, new_state, _, _ = run_reference_sequence(
            current, synthetic_fixture(current.c, torch.float64)
        )
        assert value_digest(old_state) == value_digest(new_state)
        assert value_digest(old_loss) == value_digest(new_loss)
        return {
            "historical_revision": revision,
            "historical_source_sha256": expected_sha,
            "parameter_tensors": len(list(current.parameters())),
            "initial_parameter_values_equal": True,
            "same_synthetic_sequence_state_equal": True,
            "same_synthetic_sequence_losses_equal": True,
            "scope": "small CPU float64 forward identity after typing changes",
        }
    finally:
        del sys.modules[legacy.__name__]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    target = arguments.report.resolve()
    if target.parent != ROOT / "reports" or target.exists():
        parser.error("Report must be a new file directly inside reports/.")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name, expected in contract["source_pins"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    torch.set_num_threads(contract["conditions"]["threads"])
    checks = load_checks()
    assert [identity for identity, _, _ in cases(checks)] == contract["expected_checks"]
    optimizer_attempts = 0

    def reject_optimizer_step(*args, **kwargs):
        nonlocal optimizer_attempts
        optimizer_attempts += 1
        raise AssertionError("Optimizer updates are outside 22.3.2")

    test_root = ROOT / ".seed_test_tmp"
    test_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="u2-window-22-3-", dir=test_root) as directory:
        root = Path(directory)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.optim.Adam, "step", reject_optimizer_step)
            patch.setattr(torch.optim.SGD, "step", reject_optimizer_step)
            results = execute_checks(checks, root)
            checks_passed = all(row["passed"] for row in results)
            metrics = (
                replay_metrics(checks, root / "metrics", contract["tolerances"]["float64"])
                if checks_passed
                else None
            )
            equivalence = (
                historical_type_equivalence(**contract["type_equivalence_parent"])
                if checks_passed
                else None
            )
        cpu, wall = time.process_time(), time.monotonic() - STARTED
        within_budget = (
            cpu <= contract["budget"]["cpu_seconds"]
            and wall <= contract["budget"]["wall_seconds"]
            and optimizer_attempts == 0
        )
        for name, expected in contract["source_pins"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Frozen source changed during verification: {name}")
        passed = checks_passed and within_budget
        report = {
            "schema": "taiji-u2-22.3.2-window-verification-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "verdict": (
                "passed_finite_window_runtime_only" if passed else "failed_window_runtime_check"
            ),
            "run_conditions": run_conditions(U2Subject(U2Config.small()).double()),
            "contract_sha256": digest(CONTRACT),
            "source_pins": contract["source_pins"],
            "checks": results,
            "metrics": metrics,
            "type_cleanup_equivalence": equivalence,
            "cpu_seconds": cpu,
            "wall_seconds": wall,
            "within_budget": within_budget,
            "optimizer_attempts": optimizer_attempts,
            "training_steps": 0,
            "project_dataset_reads": 0,
            "existing_checkpoints_loaded": 0,
            "whole_22.3_passed": False,
            "independent_process_restore_verified": False,
            "post_update_semantic_rebuild_verified": False,
            "task_capability_evaluated": False,
        }
        staging = write_report(root / "report.writing", report)
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(staging.read_text(encoding="utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"verdict": report["verdict"], "checks": len(results), "cpu_seconds": cpu}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
