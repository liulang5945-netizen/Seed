import copy

import pytest

from scripts.training.taiji_r2_continuation_budget_amendment_v1_5 import (
    AGGREGATE_CPU_LIMIT,
    PER_ARM_CPU_LIMIT,
    SOURCE_REVISION_SHA256,
    BudgetAmendmentRefuse,
    build_amendment,
    validate_amendment,
)
from scripts.training.taiji_r2_dialogue_continuation_extension import _digest

PREVIOUS_AMENDMENT_SHA256 = "c" * 64


def anchor():
    return {
        "sequence": 66_699,
        "event_sha256": "a" * 64,
        "charges": {
            "full_turn_1m": 1_436_869.0326909989,
            "warm_start": 1_549_763.9824659985,
            "fresh_same_capacity": 1_383_166.235697999,
        },
        "reservations": {},
    }


def test_additive_amendment_binds_v1_4_and_current_ledger_tip():
    record = build_amendment(anchor(), previous_amendment_sha256=PREVIOUS_AMENDMENT_SHA256)

    assert record["previous_budget"] == {"per_arm": 1_550_000.0, "aggregate": 4_400_000.0}
    assert record["previous_budget_amendment_sha256"] == PREVIOUS_AMENDMENT_SHA256
    assert record["approved_budget"] == {
        "per_arm": PER_ARM_CPU_LIMIT,
        "aggregate": AGGREGATE_CPU_LIMIT,
    }
    assert record["ledger_anchor"] == anchor()
    assert (
        validate_amendment(
            record,
            source_revision_sha256=SOURCE_REVISION_SHA256,
            previous_amendment_sha256=PREVIOUS_AMENDMENT_SHA256,
        )
        == record["amendment_sha256"]
    )


@pytest.mark.parametrize("drift", ["budget", "lineage", "previous", "digest"])
def test_amendment_refuses_cap_or_lineage_drift(drift):
    record = copy.deepcopy(
        build_amendment(anchor(), previous_amendment_sha256=PREVIOUS_AMENDMENT_SHA256)
    )
    if drift == "budget":
        record["approved_budget"]["aggregate"] += 1
        record["amendment_sha256"] = _digest(record, omit="amendment_sha256")
    elif drift == "lineage":
        record["source_revision_sha256"] = "0" * 64
        record["amendment_sha256"] = _digest(record, omit="amendment_sha256")
    elif drift == "previous":
        record["previous_budget_amendment_sha256"] = "0" * 64
        record["amendment_sha256"] = _digest(record, omit="amendment_sha256")
    else:
        record["amendment_sha256"] = "0" * 64

    with pytest.raises(BudgetAmendmentRefuse):
        validate_amendment(
            record,
            source_revision_sha256=SOURCE_REVISION_SHA256,
            previous_amendment_sha256=PREVIOUS_AMENDMENT_SHA256,
        )
