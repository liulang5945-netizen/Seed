"""Read-only reconciliation of PLAN-N1-132 execution and recovery receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-132_trainable_question_binding_supervision_calibration_draft_20261019.md"
)
PLAN_SHA256 = "4177e1fa9da00ef1911a3c23f8598a67efd2e5fd5e54b2d4bd96b02fd60dd10f"
RUNNER = ROOT / "scripts/training/run_taiji_n1_plan132_binding_supervision_calibration.py"
RESULT = ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_result_20261019.json"
ATTEMPT = ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_attempt_20261019.json"
PROGRESS = ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_progress_20261019.jsonl"
READINESS = (
    ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_readiness_20261019.json"
)
READINESS_REVALIDATION = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_readiness_revalidation_20261019.json"
)
READINESS_REVALIDATION_V2 = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_readiness_revalidation_v2_20261019.json"
)
SOURCE_BINDING_V2 = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_source_binding_revalidation_v2_20261019.json"
)
FAILED_PREFLIGHT = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_20261019.json"
)
FAILED_PREFLIGHT_BINDING = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_binding_20261019.json"
)
PREFLIGHT = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_revalidation_20261019.json"
)
PREFLIGHT_BINDING = (
    ROOT
    / "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_revalidation_binding_20261019.json"
)
PREFLIGHT_ATTEMPTS = tuple(
    ROOT / path
    for path in (
        "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_attempt_20261019.json",
        "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_repair_attempt_20261019.json",
        "reports/taiji_n1_plan132_binding_supervision_calibration_checkpoint_preflight_repair2_attempt_20261019.json",
    )
)
REPORT = ROOT / "reports/taiji_n1_plan132_provenance_reconciliation_20261019.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def run() -> dict:
    if REPORT.exists():
        raise FileExistsError("PLAN-N1-132 provenance reconciliation is write-once")
    for path in (
        PLAN,
        RUNNER,
        RESULT,
        ATTEMPT,
        PROGRESS,
        READINESS,
        READINESS_REVALIDATION,
        READINESS_REVALIDATION_V2,
        SOURCE_BINDING_V2,
        FAILED_PREFLIGHT,
        FAILED_PREFLIGHT_BINDING,
        PREFLIGHT,
        PREFLIGHT_BINDING,
        *PREFLIGHT_ATTEMPTS,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required N1-132 receipt missing: {path}")
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("frozen N1-132 plan changed")

    started_hashes = {
        str(Path(path).resolve()): sha256(Path(path))
        for path in (
            PLAN,
            RUNNER,
            RESULT,
            ATTEMPT,
            PROGRESS,
            READINESS,
            READINESS_REVALIDATION,
            READINESS_REVALIDATION_V2,
            SOURCE_BINDING_V2,
            FAILED_PREFLIGHT,
            FAILED_PREFLIGHT_BINDING,
            PREFLIGHT,
            PREFLIGHT_BINDING,
            *PREFLIGHT_ATTEMPTS,
        )
    }
    result = read_json(RESULT)
    attempt = read_json(ATTEMPT)
    readiness_v2 = read_json(READINESS_REVALIDATION_V2)
    source_binding_v2 = read_json(SOURCE_BINDING_V2)
    failed_preflight = read_json(FAILED_PREFLIGHT)
    failed_binding = read_json(FAILED_PREFLIGHT_BINDING)
    preflight = read_json(PREFLIGHT)
    preflight_binding = read_json(PREFLIGHT_BINDING)

    runner_key = str(RUNNER.resolve())
    readiness_key = str(READINESS_REVALIDATION_V2.resolve())
    preflight_key = str(PREFLIGHT.resolve())
    source_map = result.get("source_hashes", {})
    runner_bound = source_map.get(runner_key) == started_hashes[runner_key]
    active_ready_bound_by_revalidation = source_binding_v2.get(
        "readiness_revalidation_sha256"
    ) == started_hashes[readiness_key] and source_binding_v2.get(
        "source_hashes"
    ) == readiness_v2.get(
        "source_hashes"
    )
    active_preflight_bound = (
        preflight.get("complete") is True
        and preflight.get("status") == "complete"
        and preflight_binding.get("preflight_sha256") == started_hashes[preflight_key]
        and preflight_binding.get("zero_checkpoint_hashes")
        == {row["path"]: row["sha256"] for row in preflight.get("zero_checkpoints", [])}
    )

    attempt_readiness_matches_initial = (
        attempt.get("readiness_sha256") == started_hashes[str(READINESS.resolve())]
    )
    attempt_preflight_points_to_stopped_receipt = (
        attempt.get("checkpoint_preflight_sha256")
        == started_hashes[str(FAILED_PREFLIGHT.resolve())]
    )
    result_readiness_points_to_initial = (
        result.get("readiness_sha256") == started_hashes[str(READINESS.resolve())]
    )
    result_preflight_points_to_stopped = (
        result.get("checkpoint_preflight_sha256") == started_hashes[str(FAILED_PREFLIGHT.resolve())]
    )
    attempts_zero_load = all(
        read_json(path).get("model_loads") == 0 and read_json(path).get("parameter_updates") == 0
        for path in PREFLIGHT_ATTEMPTS
    )
    failures_stopped_before_load = (
        failed_preflight.get("complete") is False
        and failed_preflight.get("model_loads") == 0
        and failed_preflight.get("checkpoint_loads") == 0
        and failed_preflight.get("parameter_updates") == 0
        and failed_preflight.get("zero_checkpoints") == []
        and failed_binding.get("preflight_sha256")
        == started_hashes[str(FAILED_PREFLIGHT.resolve())]
    )

    final_checkpoint_checks = []
    for model in result.get("models", []):
        path = Path(model["final_checkpoint_path"])
        digest = model["final_checkpoint_sha256"]
        final_checkpoint_checks.append(
            {
                "seed": model["seed"],
                "path": str(path.resolve()),
                "sha256_expected": digest,
                "sha256_current": sha256(path) if path.is_file() else None,
                "independent_restore_equal": model.get("final_independent_restore", {}).get(
                    "equal"
                ),
                "restore_signature_matches": model.get("final_independent_restore", {}).get(
                    "signature"
                )
                == model.get("final_bundle_fingerprint"),
            }
        )
    zero_checkpoint_checks = [
        {
            "seed": row["seed"],
            "path": row["path"],
            "sha256_expected": row["sha256"],
            "sha256_current": sha256(Path(row["path"])) if Path(row["path"]).is_file() else None,
            "independent_restore_equal": row.get("independent_restore", {}).get("equal"),
        }
        for row in preflight.get("zero_checkpoints", [])
    ]
    progress_lines = sum(1 for _ in PROGRESS.open("r", encoding="utf-8"))
    verified_source_hashes = {
        **source_map,
        **readiness_v2.get("source_hashes", {}),
        **preflight.get("source_hashes", {}),
        **readiness_v2.get("parent_checkpoint_hashes", {}),
        **preflight.get("parent_checkpoint_hashes", {}),
    }
    for receipt in preflight_binding.get("attempt_receipts", []):
        verified_source_hashes[receipt["path"]] = receipt["sha256"]
    source_hashes_current = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in verified_source_hashes.items()
    )
    complete = (
        result.get("complete") is True
        and result.get("training_updates") == 27_648
        and result.get("train_score_queries") == 2_880
        and result.get("parameter_updates") == 27_648
        and result.get("dev_queries") == 0
        and result.get("v13_queries") == 0
        and result.get("v14_queries") == 0
        and result.get("confirmation_queries") == 0
        and result.get("can_promote") is False
        and runner_bound
        and active_ready_bound_by_revalidation
        and active_preflight_bound
        and attempts_zero_load
        and failures_stopped_before_load
        and result_readiness_points_to_initial
        and result_preflight_points_to_stopped
        and readiness_v2.get("parent_checkpoint_hashes") == result.get("parent_checkpoint_hashes")
        and preflight.get("parent_checkpoint_hashes") == result.get("parent_checkpoint_hashes")
        and progress_lines == 27_648
        and source_hashes_current
        and all(
            row["independent_restore_equal"] and row["restore_signature_matches"]
            for row in final_checkpoint_checks
        )
        and all(row["independent_restore_equal"] for row in zero_checkpoint_checks)
        and len(final_checkpoint_checks) == 3
        and len(zero_checkpoint_checks) == 3
    )
    report = {
        "schema": "n1-plan132-provenance-reconciliation-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "training_result_path": str(RESULT.resolve()),
        "training_result_sha256": started_hashes[str(RESULT.resolve())],
        "training_attempt_sha256": started_hashes[str(ATTEMPT.resolve())],
        "initial_readiness_sha256": started_hashes[str(READINESS.resolve())],
        "effective_readiness_revalidation_path": str(READINESS_REVALIDATION_V2.resolve()),
        "effective_readiness_revalidation_sha256": started_hashes[readiness_key],
        "effective_preflight_path": str(PREFLIGHT.resolve()),
        "effective_preflight_sha256": started_hashes[preflight_key],
        "runner_sha256_bound_by_training_result": source_map.get(runner_key),
        "runner_sha256_current": started_hashes[runner_key],
        "training_result_top_level_readiness_field_points_to_initial_receipt": attempt_readiness_matches_initial,
        "training_attempt_preflight_field_points_to_stopped_preflight_receipt": attempt_preflight_points_to_stopped_receipt,
        "result_readiness_field_points_to_initial_receipt": result_readiness_points_to_initial,
        "result_preflight_field_points_to_stopped_receipt": result_preflight_points_to_stopped,
        "attempt_receipts_model_loads_zero_updates": attempts_zero_load,
        "failed_preflight_receipts_preserved_before_load": failures_stopped_before_load,
        "effective_readiness_and_preflight_reconciled": active_ready_bound_by_revalidation
        and active_preflight_bound,
        "final_checkpoints": final_checkpoint_checks,
        "zero_copy_checkpoints": zero_checkpoint_checks,
        "progress_lines": progress_lines,
        "source_hashes_current": source_hashes_current,
        "verified_source_hashes": verified_source_hashes,
        "training_updates": result.get("training_updates"),
        "train_score_queries": result.get("train_score_queries"),
        "dev_v13_v14_confirmation_queries": {
            "dev": result.get("dev_queries"),
            "v13": result.get("v13_queries"),
            "v14": result.get("v14_queries"),
            "confirmation": result.get("confirmation_queries"),
        },
        "training_sufficient_all_seeds": result.get("training_sufficient_all_seeds"),
        "can_promote": False,
    }
    write_new(REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    result = run()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "effective_readiness_and_preflight_reconciled",
                    "source_hashes_current",
                    "training_updates",
                    "can_promote",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
