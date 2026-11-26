"""Replay only the post-audit ledger tail when resuming long-running arms."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from scripts.training import taiji_r2_dialogue_continuation_extension as budget

ARMS = budget.ARMS
BASELINE = {
    "sequence": 59_476,
    "event_sha256": "542c61ddfe993976d5bd16ce1a777d7753732526fc973c3b72d7becc73939279",
    "charges": {
        "full_turn_1m": 1_381_259.2983159989,
        "warm_start": 1_381_751.7480909992,
        "fresh_same_capacity": 1_378_957.423197999,
    },
    "reservations": {
        "fresh_same_capacity": {
            "reservation_id": "341643633c1da4e3dab3ffb691b33532",
            "reserved_cpu_seconds": 252.0,
        }
    },
}
BASE_CONTRACT_SHA256 = "7685c69a1f1ea2cd80990f04e06241757d9f1360d661c896545693805e01ad5a"
_TOKEN = re.compile(r"^[0-9a-f]{32}$")


class LedgerAnchorRefuse(ValueError):
    """The ledger tail is not descended from the last fully audited tip."""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _digest_event(event: dict[str, Any]) -> str:
    return budget._digest(event, omit="event_sha256")


def _check_totals(charges, reservations, *, per_arm_limit, aggregate_limit):
    if set(charges) != set(ARMS) or any(
        not _finite(charges[arm]) or not 0 <= charges[arm] <= per_arm_limit for arm in ARMS
    ):
        raise LedgerAnchorRefuse("ledger tail has an invalid arm charge")
    reserved = 0.0
    for arm, reservation in reservations.items():
        upper = reservation.get("reserved_cpu_seconds")
        token = reservation.get("reservation_id")
        if (
            arm not in ARMS
            or not isinstance(token, str)
            or not _TOKEN.fullmatch(token)
            or not _finite(upper)
            or upper <= 0
            or charges[arm] + upper >= per_arm_limit
        ):
            raise LedgerAnchorRefuse("ledger tail has an invalid active reservation")
        reserved += upper
    if sum(charges.values()) + reserved > aggregate_limit:
        raise LedgerAnchorRefuse("ledger tail exceeds the aggregate CPU ceiling")


def replay_ledger_tail(
    anchor: dict[str, Any],
    events: list[dict[str, Any]],
    contract: dict[str, Any],
    *,
    per_arm_limit: float,
    aggregate_limit: float,
) -> dict[str, Any]:
    """Replay contiguous reservation/settlement events from a pinned full-audit tip."""
    sequence = anchor["sequence"]
    previous_sha = anchor["event_sha256"]
    charges = dict(anchor["charges"])
    reservations = {arm: dict(value) for arm, value in anchor["reservations"].items()}
    _check_totals(
        charges,
        reservations,
        per_arm_limit=per_arm_limit,
        aggregate_limit=aggregate_limit,
    )
    for event in events:
        sequence += 1
        arm = event.get("arm")
        if (
            event.get("schema") != budget.EVENT_SCHEMA
            or event.get("sequence") != sequence
            or event.get("extension_id") != budget.EXTENSION_PREFIX
            or arm not in ARMS
            or event.get("previous_event_sha256") != previous_sha
            or event.get("event_sha256") != _digest_event(event)
            or not re.fullmatch(r"[0-9a-f]{64}", str(event.get("event_sha256", "")))
        ):
            raise LedgerAnchorRefuse("ledger tail has a broken sequence or SHA chain")
        parent = contract["parent_records"][arm]
        if event.get("parent_attempt_sha256") != parent.get("parent_attempt_sha256") or event.get(
            "parent_receipt_sha256"
        ) != parent.get("parent_receipt_sha256"):
            raise LedgerAnchorRefuse("ledger tail changed the arm's parent lineage")
        charge = event.get("charged_cpu_seconds")
        if not _finite(charge):
            raise LedgerAnchorRefuse("ledger tail charge is not finite")
        current = reservations.get(arm)
        if event.get("event") == "work_unit_reserved":
            reserve = event.get("reserved_cpu_seconds")
            token = event.get("reservation_id")
            if (
                current is not None
                or not isinstance(token, str)
                or not _TOKEN.fullmatch(token)
                or not _finite(reserve)
                or reserve <= 0
                or charge < charges[arm]
                or charge > per_arm_limit
            ):
                raise LedgerAnchorRefuse("ledger tail reservation is malformed")
            charges[arm] = float(charge)
            reservations[arm] = {"reservation_id": token, "reserved_cpu_seconds": float(reserve)}
        elif event.get("event") in {
            "continuation_started",
            "continuation_resumed",
            "continuation_stopped",
            "continuation_completed",
            "training_resource_preflight_passed",
            "parent_checkpoint_restored",
            "episode_completed",
            "durable_training_commit",
            "work_unit_settled",
            "final_stage_restored",
            "crash_recovery_upper_settlement",
        }:
            if (
                current is None
                or event.get("reservation_id") != current["reservation_id"]
                or event.get("reserved_cpu_seconds") != current["reserved_cpu_seconds"]
                or event.get("details", {}).get("reservation_id") != current["reservation_id"]
                or charge < charges[arm]
                or charge > per_arm_limit
            ):
                raise LedgerAnchorRefuse("ledger tail settlement is malformed")
            charges[arm] = float(charge)
            del reservations[arm]
        else:
            raise LedgerAnchorRefuse("ledger tail contains an event requiring full-history audit")
        _check_totals(
            charges,
            reservations,
            per_arm_limit=per_arm_limit,
            aggregate_limit=aggregate_limit,
        )
        previous_sha = event["event_sha256"]
    return {
        "sequence": sequence,
        "event_sha256": previous_sha,
        "charges": charges,
        "reservations": reservations,
    }


def _read_event(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LedgerAnchorRefuse("ledger event file is missing or linked")
    try:
        event = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LedgerAnchorRefuse("ledger event file is unreadable") from exc
    if not isinstance(event, dict):
        raise LedgerAnchorRefuse("ledger event is not an object")
    return event


def verified_snapshot(*, sequence: int | None = None) -> dict[str, Any]:
    """Verify the checkpointed audit anchor and append-only tail to a current tip."""
    with budget._aggregate_lock(budget.LOCK_PATH):
        contract, live_charges, live_reservations, live_tip = budget._read_summary_fast()
        if contract.get("contract_sha256") != BASE_CONTRACT_SHA256:
            raise LedgerAnchorRefuse("aggregate ledger contract is not the audited R2 window")
        current_summary = budget._read_json(budget.AGGREGATE_LEDGER_SUMMARY)
        current_sequence = current_summary["sequence"]
        target = current_sequence if sequence is None else sequence
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or target < BASELINE["sequence"]
            or target > current_sequence
        ):
            raise LedgerAnchorRefuse("requested ledger sequence is outside the audited tail")
        compare_live_tip = target == current_sequence
    # Event files are immutable and append-only; replaying a captured prefix
    # outside the lock keeps the lock hold short enough for another arm's
    # settlement.  New events after the captured target do not affect this
    # snapshot, and the live-tip comparison uses the values captured above.
    anchor_event = _read_event(
        budget._event_path(budget.AGGREGATE_LEDGER_PREFIX, BASELINE["sequence"])
    )
    if (
        anchor_event.get("event_sha256") != BASELINE["event_sha256"]
        or anchor_event.get("event_sha256") != _digest_event(anchor_event)
        or anchor_event.get("event") != "work_unit_reserved"
        or anchor_event.get("arm") != "fresh_same_capacity"
        or anchor_event.get("charged_cpu_seconds") != BASELINE["charges"]["fresh_same_capacity"]
        or anchor_event.get("reservation_id")
        != BASELINE["reservations"]["fresh_same_capacity"]["reservation_id"]
        or anchor_event.get("reserved_cpu_seconds")
        != BASELINE["reservations"]["fresh_same_capacity"]["reserved_cpu_seconds"]
    ):
        raise LedgerAnchorRefuse("pinned full-audit ledger anchor differs")
    tail = [
        _read_event(budget._event_path(budget.AGGREGATE_LEDGER_PREFIX, index))
        for index in range(BASELINE["sequence"] + 1, target + 1)
    ]
    snapshot = replay_ledger_tail(
        BASELINE,
        tail,
        contract,
        per_arm_limit=budget.PER_ARM_CPU_LIMIT,
        aggregate_limit=budget.AGGREGATE_CPU_LIMIT,
    )
    if compare_live_tip and (
        snapshot["event_sha256"] != live_tip
        or snapshot["charges"] != live_charges
        or snapshot["reservations"] != live_reservations
    ):
        raise LedgerAnchorRefuse("replayed ledger tail differs from verified summary tip")
    snapshot["effective_charges"] = {
        arm: snapshot["charges"][arm]
        + float(snapshot["reservations"].get(arm, {}).get("reserved_cpu_seconds", 0.0))
        for arm in ARMS
    }
    return snapshot


def current_fast_charges() -> dict[str, float]:
    with budget._aggregate_lock(budget.LOCK_PATH):
        contract, charges, reservations, _ = budget._read_summary_fast()
        if contract.get("contract_sha256") != BASE_CONTRACT_SHA256:
            raise LedgerAnchorRefuse("aggregate ledger contract changed")
    return {
        arm: charges[arm] + float(reservations.get(arm, {}).get("reserved_cpu_seconds", 0.0))
        for arm in ARMS
    }


def current_fast_reservations() -> dict[str, dict[str, Any]]:
    with budget._aggregate_lock(budget.LOCK_PATH):
        contract, _, reservations, _ = budget._read_summary_fast()
        if contract.get("contract_sha256") != BASE_CONTRACT_SHA256:
            raise LedgerAnchorRefuse("aggregate ledger contract changed")
    return {arm: dict(record) for arm, record in reservations.items()}
