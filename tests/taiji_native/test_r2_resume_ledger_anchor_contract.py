import copy

import pytest

from scripts.training import taiji_r2_dialogue_continuation_extension as budget
from scripts.training.taiji_r2_resume_ledger_anchor import LedgerAnchorRefuse, replay_ledger_tail

ARMS = budget.ARMS


def fixture():
    contract = {
        "parent_records": {
            arm: {
                "parent_attempt_sha256": str(index) * 64,
                "parent_receipt_sha256": str(index + 3) * 64,
            }
            for index, arm in enumerate(ARMS, start=1)
        }
    }
    anchor = {
        "sequence": 100,
        "event_sha256": "a" * 64,
        "charges": {arm: 100.0 * (index + 1) for index, arm in enumerate(ARMS)},
        "reservations": {
            "full_turn_1m": {"reservation_id": "b" * 32, "reserved_cpu_seconds": 25.0}
        },
    }

    def event(sequence, previous, arm, kind, charge, token, reserve, details=None):
        record = {
            "schema": budget.EVENT_SCHEMA,
            "sequence": sequence,
            "event": kind,
            "extension_id": budget.EXTENSION_PREFIX,
            "arm": arm,
            "charged_cpu_seconds": charge,
            "parent_attempt_sha256": contract["parent_records"][arm]["parent_attempt_sha256"],
            "parent_receipt_sha256": contract["parent_records"][arm]["parent_receipt_sha256"],
            "reservation_id": token,
            "reserved_cpu_seconds": reserve,
            "details": details or {},
            "previous_event_sha256": previous,
        }
        record["event_sha256"] = budget._digest(record, omit="event_sha256")
        return record

    first = event(
        101,
        anchor["event_sha256"],
        "full_turn_1m",
        "episode_completed",
        125.0,
        "b" * 32,
        25.0,
        {"reservation_id": "b" * 32},
    )
    second = event(
        102,
        first["event_sha256"],
        "warm_start",
        "work_unit_reserved",
        200.0,
        "c" * 32,
        50.0,
    )
    third = event(
        103,
        second["event_sha256"],
        "warm_start",
        "work_unit_settled",
        240.0,
        "c" * 32,
        50.0,
        {"reservation_id": "c" * 32},
    )
    fourth = event(
        104,
        third["event_sha256"],
        "fresh_same_capacity",
        "work_unit_reserved",
        300.0,
        "d" * 32,
        50.0,
    )
    fifth = event(
        105,
        fourth["event_sha256"],
        "fresh_same_capacity",
        "continuation_stopped",
        330.0,
        "d" * 32,
        50.0,
        {"reservation_id": "d" * 32, "unknown_tail": True},
    )
    return anchor, [first, second, third, fourth, fifth], contract


def test_replays_reservations_and_settlements_from_pinned_tip():
    anchor, events, contract = fixture()

    result = replay_ledger_tail(
        anchor,
        events,
        contract,
        per_arm_limit=1_500_000.0,
        aggregate_limit=4_300_000.0,
    )

    assert result["sequence"] == 105
    assert result["charges"] == {
        "full_turn_1m": 125.0,
        "warm_start": 240.0,
        "fresh_same_capacity": 330.0,
    }
    assert result["reservations"] == {}


def test_replays_final_stage_restored_as_a_settlement():
    anchor, events, contract = fixture()
    fifth = events[-1]
    reserved = copy.deepcopy(fifth)
    reserved.update(
        {
            "sequence": 106,
            "event": "work_unit_reserved",
            "charged_cpu_seconds": 330.0,
            "reservation_id": "e" * 32,
            "reserved_cpu_seconds": 10.0,
            "details": {},
            "previous_event_sha256": fifth["event_sha256"],
        }
    )
    reserved["event_sha256"] = budget._digest(reserved, omit="event_sha256")
    restored = copy.deepcopy(reserved)
    restored.update(
        {
            "sequence": 107,
            "event": "final_stage_restored",
            "charged_cpu_seconds": 340.0,
            "details": {"reservation_id": "e" * 32, "stage": "pretraining"},
            "previous_event_sha256": reserved["event_sha256"],
        }
    )
    restored["event_sha256"] = budget._digest(restored, omit="event_sha256")

    result = replay_ledger_tail(
        anchor,
        [*events, reserved, restored],
        contract,
        per_arm_limit=1_500_000.0,
        aggregate_limit=4_300_000.0,
    )

    assert result["sequence"] == 107
    assert result["charges"]["fresh_same_capacity"] == 340.0
    assert result["reservations"] == {}


def test_replays_crash_recovery_settlement_as_a_settlement():
    anchor, events, contract = fixture()
    fifth = events[-1]
    reserved = copy.deepcopy(fifth)
    reserved.update(
        {
            "sequence": 106,
            "event": "work_unit_reserved",
            "charged_cpu_seconds": 330.0,
            "reservation_id": "e" * 32,
            "reserved_cpu_seconds": 10.0,
            "details": {},
            "previous_event_sha256": fifth["event_sha256"],
        }
    )
    reserved["event_sha256"] = budget._digest(reserved, omit="event_sha256")
    settled = copy.deepcopy(reserved)
    settled.update(
        {
            "sequence": 107,
            "event": "crash_recovery_upper_settlement",
            "charged_cpu_seconds": 340.0,
            "details": {"reservation_id": "e" * 32, "reason": "orphan"},
            "previous_event_sha256": reserved["event_sha256"],
        }
    )
    settled["event_sha256"] = budget._digest(settled, omit="event_sha256")

    result = replay_ledger_tail(
        anchor,
        [*events, reserved, settled],
        contract,
        per_arm_limit=1_500_000.0,
        aggregate_limit=4_300_000.0,
    )

    assert result["sequence"] == 107
    assert result["charges"]["fresh_same_capacity"] == 340.0
    assert result["reservations"] == {}


@pytest.mark.parametrize(
    "mutation",
    [
        "sequence",
        "previous_sha",
        "event_sha",
        "parent",
        "unknown_event",
        "charge_decrease",
        "double_reserve",
    ],
)
def test_refuses_invalid_ledger_tail(mutation):
    anchor, events, contract = fixture()
    events = copy.deepcopy(events)
    if mutation == "sequence":
        events[0]["sequence"] += 1
    elif mutation == "previous_sha":
        events[0]["previous_event_sha256"] = "0" * 64
    elif mutation == "event_sha":
        events[0]["event_sha256"] = "0" * 64
    elif mutation == "parent":
        events[0]["parent_attempt_sha256"] = "0" * 64
    elif mutation == "unknown_event":
        events[0]["event"] = "boot_accounting_upper_reconciled"
        events[0]["event_sha256"] = budget._digest(events[0], omit="event_sha256")
    elif mutation == "charge_decrease":
        events[0]["charged_cpu_seconds"] = 99.0
        events[0]["event_sha256"] = budget._digest(events[0], omit="event_sha256")
    else:
        duplicate = copy.deepcopy(events[1])
        duplicate["sequence"] = events[1]["sequence"] + 1
        duplicate["previous_event_sha256"] = events[1]["event_sha256"]
        duplicate["event_sha256"] = budget._digest(duplicate, omit="event_sha256")
        events.insert(2, duplicate)

    with pytest.raises(LedgerAnchorRefuse):
        replay_ledger_tail(
            anchor,
            events,
            contract,
            per_arm_limit=1_500_000.0,
            aggregate_limit=4_300_000.0,
        )
