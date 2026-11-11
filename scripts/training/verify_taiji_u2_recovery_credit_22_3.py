"""U2 22.3.0/1 state-contract and conditional pending-credit receipt.

Only synthetic inputs and fresh in-memory bundles are used. The full Theta
learning-window graph is deliberately tested as an unresolved negative control.
No optimizer step, sealed data or existing checkpoint is accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from taiji.u2_subject import U2Config, U2Subject

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_3_recovery_contract_20261110.json"
TESTS = ROOT / "tests/taiji_native/test_u2_recovery_credit_22_3.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def load_checks():
    spec = importlib.util.spec_from_file_location("u2_pending_recovery_checks", TESTS)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    return checks


def execute_checks(checks, contract):
    results = []
    for name in sorted(item for item in vars(checks) if item.startswith("test_")):
        function = getattr(checks, name)
        cases = (
            contract["mutations"]
            if "mutation" in inspect.signature(function).parameters
            else [None]
        )
        for mutation in cases:
            identity = name if mutation is None else f"{name}[{mutation}]"
            model = U2Subject(U2Config.small()).double()
            try:
                function(model) if mutation is None else function(model, mutation)
                result = {"id": identity, "passed": True}
            except Exception as error:
                result = {
                    "id": identity,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            result["source_line"] = inspect.getsourcelines(function)[1]
            results.append(result)
    assert sorted(row["id"] for row in results) == sorted(contract["expected_checks"])
    return results


def gradient_metrics(checks):
    model = U2Subject(U2Config.small()).double()
    state, packet = checks.pending_fixture(model)
    bundle = model.export_bundle(state)
    restored, resumed_state = U2Subject.from_bundle(checks.roundtrip(bundle))
    _, live = model.receive(state, packet)
    _, resumed = restored.receive(resumed_state, packet)
    before = checks.critic_gradients(model, live["critic"])
    after = checks.critic_gradients(restored, resumed["critic"])
    differences = [
        float((left - right).abs().max()) for left, right in zip(before, after, strict=True)
    ]
    original_motion = torch.autograd.grad(
        live["predictive_logp"].sum(), model.motion.first.weight, allow_unused=True
    )[0]
    resumed_motion = torch.autograd.grad(
        resumed["predictive_logp"].sum(), restored.motion.first.weight, allow_unused=True
    )[0]
    return {
        "schema": bundle["schema"],
        "state_fields": len(state),
        "parameter_tensors": len(list(model.parameters())),
        "max_abs_delta": max(differences),
        "critic_gradient_norm_before": sum(float(value.norm()) for value in before),
        "critic_gradient_norm_after": sum(float(value.norm()) for value in after),
        "cached_value_requires_grad_after_load": resumed_state["pending"]["decision"][
            "value"
        ].requires_grad,
        "full_window_negative_control": {
            "live_motion_credit_norm": float(original_motion.norm()),
            "value_restored_motion_credit_missing": resumed_motion is None,
            "status": "requires_22.3.2_causal_window_replay",
        },
        "credit_contract": bundle["credit_contract"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    target = args.report.resolve()
    if target.parent != ROOT / "reports" or target.exists():
        parser.error("Report must be a new file directly inside reports/.")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name, expected in contract["source_pins"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    torch.set_num_threads(1)
    checks = load_checks()
    results = execute_checks(checks, contract)
    passed = all(row["passed"] for row in results)
    metrics = gradient_metrics(checks) if passed else None
    cpu, wall = time.process_time(), time.monotonic() - STARTED
    assert cpu <= contract["subcard_budget"]["cpu_seconds"]
    assert wall <= contract["subcard_budget"]["wall_seconds"]
    report = {
        "schema": "taiji-u2-22.3.1-pending-credit-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "verdict": (
            "passed_pending_conditional_credit_only" if passed else "failed_pending_credit_check"
        ),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "device": "cpu",
            "dtype": "torch.float64",
            "python": sys.version,
            "torch": torch.__version__,
            "environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
        "contract_sha256": digest(CONTRACT),
        "checks": results,
        "metrics": metrics,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "optimizer_attempts": 0,
        "training_steps": 0,
        "project_dataset_reads": 0,
        "existing_checkpoints_loaded": 0,
        "whole_22.3_passed": False,
        "independent_process_restore_verified": False,
    }
    staging = ROOT / ".seed_test_tmp/u2_22_3_credit_report.writing"
    staging.parent.mkdir(exist_ok=True)
    write_report(staging, report)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(staging.read_text(encoding="utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    staging.unlink()
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "checks": len(results),
                "cpu_seconds": cpu,
                "optimizer_attempts": 0,
            }
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
