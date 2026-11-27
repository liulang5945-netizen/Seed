import copy

import pytest

from scripts.training.taiji_r2_continuation_budget_amendment_v1_4 import (
    AGGREGATE_CPU_LIMIT,
    PER_ARM_CPU_LIMIT,
    SOURCE_REVISION_SHA256,
    BudgetAmendmentRefuse,
    build_amendment,
    validate_amendment,
)
from scripts.training.taiji_r2_dialogue_continuation_extension import _digest


def anchor():
    return {
        "sequence": 61_114,
        "event_sha256": "a" * 64,
        "charges": {"full_turn_1m": 1.4e6, "warm_start": 1.48e6, "fresh_same_capacity": 1.38e6},
        "reservations": {"warm_start": {"reservation_id": "b" * 32, "reserved_cpu_seconds": 2.0}},
    }


def test_additive_amendment_binds_previous_ceiling_and_current_ledger_tip():
    record = build_amendment(anchor())

    assert record["previous_budget"] == {"per_arm": 1_500_000.0, "aggregate": 4_300_000.0}
    assert record["approved_budget"] == {
        "per_arm": PER_ARM_CPU_LIMIT,
        "aggregate": AGGREGATE_CPU_LIMIT,
    }
    assert record["ledger_anchor"] == anchor()
    assert (
        validate_amendment(record, source_revision_sha256=SOURCE_REVISION_SHA256)
        == record["amendment_sha256"]
    )


@pytest.mark.parametrize("drift", ["budget", "lineage", "digest"])
def test_amendment_refuses_cap_or_lineage_drift(drift):
    record = copy.deepcopy(build_amendment(anchor()))
    if drift == "budget":
        record["approved_budget"]["aggregate"] += 1
        record["amendment_sha256"] = _digest(record, omit="amendment_sha256")
    elif drift == "lineage":
        record["source_revision_sha256"] = "0" * 64
        record["amendment_sha256"] = _digest(record, omit="amendment_sha256")
    else:
        record["amendment_sha256"] = "0" * 64

    with pytest.raises(BudgetAmendmentRefuse):
        validate_amendment(record, source_revision_sha256=SOURCE_REVISION_SHA256)
