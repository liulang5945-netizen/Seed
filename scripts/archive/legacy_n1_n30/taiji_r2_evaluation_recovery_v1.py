"""Explicit, initially disabled recovery qualification; never reopens CC-1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.training import check_taiji_r2_evaluation_recovery as audit
from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as frozen
from scripts.training import taiji_r2_continuation_evaluation_adapter as adapter

ROOT = Path(__file__).resolve().parents[2]
RECOVERY_ID = "taiji-r2-evaluation-engineering-recovery-v1-20261025"
CONTRACT = ROOT / "reports" / "taiji_r2_evaluation_recovery_v1_contract_20261025.json"
REPORT_PREFIX = "taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1_paired_eval_recovery_v1"
CPU_LIMIT = 10500.0
WALL_LIMIT = 13000.0
REQUIRED_SOURCES = (
    "scripts/training/taiji_r2_evaluation_recovery_v1.py",
    "scripts/training/evaluate_taiji_r2_engineering_recovery_v1.py",
    "scripts/training/evaluate_taiji_r2_daily_dialogue_continuation_extension_worker_v1.py",
)


def paths() -> dict[str, Path]:
    return {
        key: ROOT / "reports" / f"{REPORT_PREFIX}_{suffix}.{extension}"
        for key, suffix, extension in (
            ("readiness", "readiness", "json"),
            ("attempt", "attempt", "json"),
            ("ledger", "ledger", "jsonl"),
            ("progress", "progress", "jsonl"),
            ("face", "face", "json"),
            ("generations", "generations", "json"),
            ("result", "result", "json"),
            ("cost_receipt", "cost_receipt", "json"),
            ("review", "review", "md"),
            ("failure", "failure", "json"),
            ("endpoint_manifest", "endpoints", "json"),
        )
    }


def require_authorization(contract: dict[str, Any]) -> None:
    if contract.get("recovery_id") != RECOVERY_ID:
        raise ValueError("unknown recovery contract identity")
    for name in (
        "authorizes_new_attempt",
        "same_dev_face_reuse_approved",
        "historical_tail_reserve_accepted",
        "old_attempt_consumed",
        "preserve_old_failure",
    ):
        if contract.get(name) is not True:
            raise ValueError(f"recovery requires explicit protocol decision: {name}")
    if (
        contract.get("cpu_limit_seconds") != CPU_LIMIT
        or contract.get("wall_limit_seconds") != WALL_LIMIT
    ):
        raise ValueError("recovery limits differ from the versioned contract")
    if (
        contract.get("historical_cpu_reserve_seconds") != 3400
        or contract.get("preparation_cpu_reserve_seconds") != 240
    ):
        raise ValueError("historical/preparation CPU reserves differ")
    if 3400 + 240 + CPU_LIMIT > frozen.EVALUATION_CPU_LIMIT_SECONDS:
        raise ValueError("combined recovery CPU exceeds the original evaluation window")
    if (
        contract.get("historical_wall_reserve_seconds") != 180
        or contract.get("preparation_wall_reserve_seconds") != 300
    ):
        raise ValueError("historical/preparation wall reserves differ")
    if 180 + 300 + WALL_LIMIT > frozen.EVALUATION_WALL_LIMIT_SECONDS:
        raise ValueError("combined recovery wall exceeds the original evaluation window")
    if contract.get("generation_outputs") != 156 or contract.get("review_answers") != 48:
        raise ValueError("recovery denominators differ")
    sources = contract.get("approved_source_sha256", {})
    if not isinstance(sources, dict) or not set(REQUIRED_SOURCES).issubset(sources):
        raise ValueError("recovery source snapshot is incomplete")
    if any(not isinstance(value, str) or len(value) != 64 for value in sources.values()):
        raise ValueError("recovery source snapshot contains invalid hashes")


def preflight() -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    require_authorization(contract)  # Before checkpoint hashes, model, or selector.
    old = audit.audit_failed_attempt(ROOT / "reports" / audit.PREFIX)
    if (
        old["attempt_id"] != contract["old_attempt_id"]
        or old["raw_sha256"] != contract["old_raw_sha256"]
    ):
        raise ValueError("original failed evidence changed")
    if (
        old["ledger_sha256"] != contract["old_ledger_sha256"]
        or not old["stopped_before_selection_in_this_attempt"]
    ):
        raise ValueError("old attempt exposure/ledger differs from the authorized recovery")
    for name, expected in contract["approved_source_sha256"].items():
        source = (ROOT / name).resolve()
        if (
            not source.is_relative_to(ROOT)
            or not source.is_file()
            or audit.sha256(source) != expected
        ):
            raise ValueError(f"recovery source changed: {name}")
    result = adapter.extension_preflight()
    observed = {row["endpoint_id"]: row["sha256"] for row in result["endpoints"]}
    if observed != contract["endpoint_checkpoint_sha256"]:
        raise ValueError("recovery endpoint identity differs from the trained candidate")
    result["source_hashes"].update(
        {
            str((ROOT / name).resolve()): expected
            for name, expected in contract["approved_source_sha256"].items()
        }
    )
    result["source_hashes"][str(CONTRACT)] = audit.sha256(CONTRACT)
    return result


def budget_factory(**kwargs: Any) -> frozen.EvaluationBudget:
    # Formal preflight costs enter the frozen execute's kwargs; historical and
    # preparation reserves are separately included in the same 14,400 window.
    return frozen.EvaluationBudget(
        cpu_limit_seconds=CPU_LIMIT, wall_limit_seconds=WALL_LIMIT, **kwargs
    )
