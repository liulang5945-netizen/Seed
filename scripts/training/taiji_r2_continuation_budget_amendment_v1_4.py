"""Activate the user-directed additive R2 CPU ceiling without rewriting v1.3."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from scripts.training import taiji_r2_dialogue_continuation_extension as budget

PER_ARM_CPU_LIMIT = 1_550_000.0
AGGREGATE_CPU_LIMIT = 4_400_000.0
PREVIOUS_BUDGET = {"per_arm": 1_500_000.0, "aggregate": 4_300_000.0}
SOURCE_REVISION_SHA256 = "2f6938afb910a7bca2c9d5bd5c3d4f79e5a67aa2cd19da08c0a9577589640bf7"
CONTRACT_FILE_SHA256 = "df70a17b9b7aa3b40c1f24967926559c259f3b19140adfa3f1cc122a3b9ceb3c"
CONTRACT_OBJECT_SHA256 = "7685c69a1f1ea2cd80990f04e06241757d9f1360d661c896545693805e01ad5a"
REVISION_PATH = budget.REPORTS_DIR / f"{budget.EXTENSION_PREFIX}_budget_amendment_v1_4.json"
SOURCE_REVISION_PATH = budget.REPORTS_DIR / f"{budget.EXTENSION_PREFIX}_source_revision_v1_3.json"
SOURCES = (
    "scripts/training/taiji_r2_continuation_budget_amendment_v1_4.py",
    "tests/taiji_native/test_r2_continuation_budget_amendment_v1_4_contract.py",
)
APPROVAL_BASIS = (
    "User directed to run one arm first and avoid repeated stops caused by an overly tight ceiling."
)


class BudgetAmendmentRefuse(ValueError):
    """The additive budget amendment does not match this immutable run."""


def _canonical(record: dict[str, Any]) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def build_amendment(anchor: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(anchor, dict) or any(
        key not in anchor for key in ("sequence", "event_sha256", "charges", "reservations")
    ):
        raise BudgetAmendmentRefuse("budget amendment needs a verified current ledger anchor")
    record = {
        "schema": "taiji-r2-continuation-budget-amendment-v1.4",
        "extension_id": budget.EXTENSION_PREFIX,
        "source_revision_sha256": SOURCE_REVISION_SHA256,
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "contract_object_sha256": CONTRACT_OBJECT_SHA256,
        "previous_budget": dict(PREVIOUS_BUDGET),
        "approved_budget": {"per_arm": PER_ARM_CPU_LIMIT, "aggregate": AGGREGATE_CPU_LIMIT},
        "approval_basis": APPROVAL_BASIS,
        "reason": (
            "A verified host-CPU tail pushed the selected warm arm just over the previous"
            " aggregate ceiling during resume. The warm arm projects about 30,000 CPU seconds"
            " remaining; the additive cap leaves about 70,000 seconds after it completes."
            " Other arms stay deferred for a later review."
        ),
        "ledger_anchor": {
            key: anchor[key] for key in ("sequence", "event_sha256", "charges", "reservations")
        },
    }
    record["amendment_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return record


def validate_amendment(record: dict[str, Any], *, source_revision_sha256: str) -> str:
    if not isinstance(record, dict):
        raise BudgetAmendmentRefuse("budget amendment must be an object")
    expected = {
        "schema": "taiji-r2-continuation-budget-amendment-v1.4",
        "extension_id": budget.EXTENSION_PREFIX,
        "source_revision_sha256": SOURCE_REVISION_SHA256,
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "contract_object_sha256": CONTRACT_OBJECT_SHA256,
        "previous_budget": PREVIOUS_BUDGET,
        "approved_budget": {"per_arm": PER_ARM_CPU_LIMIT, "aggregate": AGGREGATE_CPU_LIMIT},
        "approval_basis": APPROVAL_BASIS,
    }
    if (
        any(record.get(key) != value for key, value in expected.items())
        or source_revision_sha256 != SOURCE_REVISION_SHA256
        or not isinstance(record.get("ledger_anchor"), dict)
        or record.get("amendment_sha256")
        != hashlib.sha256(
            _canonical({key: value for key, value in record.items() if key != "amendment_sha256"})
        ).hexdigest()
    ):
        raise BudgetAmendmentRefuse("budget amendment schema, lineage, cap or digest differs")
    return record["amendment_sha256"]


def activate_budget_amendment() -> dict[str, Any]:
    """Write-once bind the new ceilings to the current verified ledger tip and activate them."""
    if SOURCE_REVISION_PATH.is_symlink() or not SOURCE_REVISION_PATH.is_file():
        raise BudgetAmendmentRefuse("approved v1.3 source revision is missing")
    source_sha = hashlib.sha256(SOURCE_REVISION_PATH.read_bytes()).hexdigest()
    if source_sha != SOURCE_REVISION_SHA256:
        raise BudgetAmendmentRefuse("approved v1.3 source revision changed")
    contract_path = budget.CONTRACT_PATH
    if (
        contract_path.is_symlink()
        or not contract_path.is_file()
        or hashlib.sha256(contract_path.read_bytes()).hexdigest() != CONTRACT_FILE_SHA256
    ):
        raise BudgetAmendmentRefuse("immutable base budget contract changed")
    with budget._aggregate_lock(budget.LOCK_PATH):
        contract, charges, reservations, tip = budget._read_summary_fast()
        if contract.get("contract_sha256") != CONTRACT_OBJECT_SHA256:
            raise BudgetAmendmentRefuse("immutable budget contract identity differs")
        summary = budget._read_json(budget.AGGREGATE_LEDGER_SUMMARY)
        anchor = {
            "sequence": summary["sequence"],
            "event_sha256": tip,
            "charges": charges,
            "reservations": reservations,
        }
        if REVISION_PATH.exists():
            if REVISION_PATH.is_symlink() or not REVISION_PATH.is_file():
                raise BudgetAmendmentRefuse("budget amendment path is not a regular file")
            record = budget._read_json(REVISION_PATH)
        else:
            if budget.effective_budget_limits() != (
                PREVIOUS_BUDGET["per_arm"],
                PREVIOUS_BUDGET["aggregate"],
            ):
                raise BudgetAmendmentRefuse("current approved budget differs from v1.3")
            record = build_amendment(anchor)
            raw = _canonical(record) + b"\n"
            budget.durable_publish(REVISION_PATH, lambda stream: stream.write(raw))
        validate_amendment(record, source_revision_sha256=source_sha)

    budget.PER_ARM_CPU_LIMIT = PER_ARM_CPU_LIMIT
    budget.AGGREGATE_CPU_LIMIT = AGGREGATE_CPU_LIMIT
    budget.effective_budget_limits = lambda: (PER_ARM_CPU_LIMIT, AGGREGATE_CPU_LIMIT)
    return record
