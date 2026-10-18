"""Offline revalidation of PLAN-N1-118 lossless training receipts."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import _taiji_n1_plan118_lossless_receipt as receipt

SOURCE_REPORT = ROOT / "reports/taiji_n1_normalized_coupled_rwl_length_balanced_20261018.json"
SOURCE_REPORT_SHA256 = "5ceabffadc7d62e2aa7be3996847c92713fe9cce604c0f10e463ad97fbe8217e"
OUTPUT = (
    ROOT
    / "reports/taiji_n1_normalized_coupled_rwl_length_balanced_20261018_receipt_revalidation.json"
)
RECEIPT_PATH = ROOT / "scripts/training/_taiji_n1_plan118_lossless_receipt.py"
TRAINING_RECEIPT_SHA256 = "b235dd722de90bab69994e854b98e1628740c2dcff4c2fac7c92d9d0f31049a9"
CPU_LIMIT_SECONDS = 60
RWL_GROUPS = ("reader", "writer", "local")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def audit() -> dict:
    if sha256(SOURCE_REPORT) != SOURCE_REPORT_SHA256:
        raise ValueError("frozen PLAN-N1-118 training report changed")
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    source_guards = source.get("source_guards", {})
    if (
        source.get("status") != "stopped"
        or source.get("error")
        != "ValueError: finite bounded actual Adam proposal evidence required"
        or source.get("completed_updates") != 221184
        or len(source.get("models", [])) != 9
        or not source.get("sources_unchanged")
        or not source.get("code_unchanged")
        or not source.get("cpu_accounting_complete")
        or any(
            source.get(key) != 0
            for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
        or source_guards.get(str(RECEIPT_PATH)) != TRAINING_RECEIPT_SHA256
    ):
        raise ValueError(
            "expected completed PLAN-N1-118 run with isolated receipt validation failure"
        )
    if OUTPUT.exists():
        raise FileExistsError(f"write-once receipt revalidation exists: {OUTPUT}")

    started = time.process_time()
    train = receipt.train
    original_protocol = train.PROTOCOL
    model_results = []
    shared_measurements = 0
    independent_measurements = 0
    single_cap_overshoots = dict.fromkeys(RWL_GROUPS, 0)
    max_shared_actual_radius = 0.0
    position_gates = []
    restore_checks = 0
    try:
        train.PROTOCOL = source["protocol"]
        _, _, views = train.known_views()
        seen = set()
        for model in sorted(source["models"], key=lambda value: (value["seed"], value["arm"])):
            address = (model["seed"], model["arm"])
            if address in seen:
                raise ValueError(f"duplicate model receipt: {address}")
            seen.add(address)
            record = receipt.read_model(model)
            sufficient = receipt.validate_model(model, record, views=views)
            restore_checks += len(record["restore_checks"])
            for measurement in record["measurements"]:
                groups = measurement["groups"]
                ratios = []
                for group_name in RWL_GROUPS:
                    group = groups[group_name]
                    ratio = group["actual_l2"] / group["cap"]
                    ratios.append(ratio)
                    if group["actual_l2"] > group["cap"] * 1.0001 + 1e-6:
                        single_cap_overshoots[group_name] += 1
                radius = math.sqrt(sum(value**2 for value in ratios))
                max_shared_actual_radius = max(max_shared_actual_radius, radius)
                shared_measurements += len(RWL_GROUPS)
                independent_measurements += 2
            final = record["milestones"][-1]
            result = {
                "seed": model["seed"],
                "arm": model["arm"],
                "completed_updates": record["completed_updates"],
                "training_sufficient": sufficient,
                "independent_restore_checks": len(record["restore_checks"]),
                "all_restore_checks_equal": all(
                    check["equal"] for check in record["restore_checks"]
                ),
            }
            if model["arm"] == "position":
                gate = {
                    "original_exact": final["free_exact"],
                    "original_pair_exact": final["pair_exact"],
                    "renamed_exact": final["rename_free_exact"],
                    "renamed_pair_exact": final["rename_pair_exact"],
                    "original_by_kind": final["free_by_kind"],
                    "renamed_by_kind": final["rename_free_by_kind"],
                }
                result["position_gate_readout"] = gate
                position_gates.append(sufficient)
            model_results.append(result)
    finally:
        train.PROTOCOL = original_protocol

    cpu_seconds = round(time.process_time() - started, 6)
    expected_models = {
        (seed, arm) for seed in (11, 12, 13) for arm in ("position", "terminal", "cut")
    }
    complete = (
        set(seen) == expected_models
        and len(model_results) == 9
        and all(model["completed_updates"] == 24576 for model in model_results)
        and restore_checks == 18
        and all(model["all_restore_checks_equal"] for model in model_results)
        and shared_measurements == 9 * 24576 * 3
        and independent_measurements == 9 * 24576 * 2
        and max_shared_actual_radius <= math.sqrt(3.0) * 1.0001 + 1e-6
        and len(position_gates) == 3
        and cpu_seconds <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan118-lossless-receipt-revalidation-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "source_training_report": str(SOURCE_REPORT.relative_to(ROOT)),
        "source_training_report_sha256": SOURCE_REPORT_SHA256,
        "source_training_status": source["status"],
        "source_training_error": source["error"],
        "source_training_receipt_validator_sha256": TRAINING_RECEIPT_SHA256,
        "revalidation_validator_sha256": sha256(RECEIPT_PATH),
        "plan_sha256": source["protocol"]["coupled_update_plan_sha256"],
        "schedule_sha256": source["schedule_sha256"],
        "models": model_results,
        "model_receipts_validated": len(model_results),
        "completed_updates": sum(model["completed_updates"] for model in model_results),
        "independent_restore_checks": restore_checks,
        "all_restore_checks_equal": all(
            model["all_restore_checks_equal"] for model in model_results
        ),
        "normalized_rwl_group_measurements_checked": shared_measurements,
        "normalized_rwl_single_group_cap_overshoots": single_cap_overshoots,
        "maximum_joint_actual_rwl_radius": max_shared_actual_radius,
        "joint_actual_radius_limit": math.sqrt(3.0),
        "independent_context_relation_measurements_checked": independent_measurements,
        "position_training_sufficient_by_seed": dict(
            zip((11, 12, 13), position_gates, strict=True)
        ),
        "training_sufficient": len(position_gates) == 3 and all(position_gates),
        "cpu_seconds": cpu_seconds,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "checkpoint_loads": 0,
        "model_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "scope": "read-only validation of existing lossless train shards; original stopped receipt remains unchanged",
    }
    report["complete"] = report["complete"] and complete
    report["status"] = "complete" if report["complete"] else "stopped"
    write_new(OUTPUT, report)
    return report


if __name__ == "__main__":
    result = audit()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "complete",
                    "status",
                    "model_receipts_validated",
                    "completed_updates",
                    "independent_restore_checks",
                    "training_sufficient",
                    "cpu_seconds",
                    "can_promote",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
