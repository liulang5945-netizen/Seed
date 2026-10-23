import pytest

from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as runner
from scripts.training import run_taiji_r2_fast_ledger_resume_v4 as v4
from scripts.training.run_taiji_r2_fast_ledger_resume import SOURCE_REVISION_SHA256
from scripts.training.run_taiji_r2_fast_ledger_resume_v4 import (
    APPROVED_BUDGET,
    SOURCES,
    build_manifest_v4,
)
from scripts.training.taiji_r2_resume_ledger_anchor import BASELINE


def inputs(arm="warm_start"):
    forecast = {
        "arm": arm,
        "feasible": True,
        "global_feasible": True,
        "selected_arm_feasible": True,
        "projection_scope": "selected_arm",
        "deferred_arms": [name for name in runner.ARMS if name != arm],
        "projection_unit": "target_byte_updates",
        "safety_factor": 1.15,
        "projected_cpu_by_arm": {
            "full_turn_1m": 81_023.0,
            "warm_start": 11_396.0,
            "fresh_same_capacity": 25_884.0,
        },
        "projected_cpu_aggregate": 118_303.0,
        "aggregate_headroom_cpu_seconds": 190_200.8,
        "arm_headroom_by_arm_cpu_seconds": {
            "full_turn_1m": 153_131.0,
            "warm_start": 40_236.0,
            "fresh_same_capacity": 206_833.8,
        },
        "arm_headroom_cpu_seconds": 40_236.0,
        "per_arm_feasible": {name: True for name in runner.ARMS},
    }
    return {
        "arm": arm,
        "source_hashes": {path: "b" * 64 for path in SOURCES},
        "budget_amendment": {
            "schema": "taiji-r2-continuation-budget-amendment-v1.5",
            "amendment_sha256": "e" * 64,
        },
        "source_revision_sha256": SOURCE_REVISION_SHA256,
        "extension_attempt_sha256": "c" * 64,
        "latest_commit": {"bundle": "commit.pt", "sha256": "d" * 64},
        "forecast": forecast,
        "snapshot": {
            "baseline_sequence": BASELINE["sequence"],
            "baseline_event_sha256": BASELINE["event_sha256"],
            "sequence": 66_699,
            "event_sha256": "a" * 64,
            "charges": {name: 1_400_000.0 for name in runner.ARMS},
            "reservations": {},
        },
        "approved_budget": dict(APPROVED_BUDGET),
    }


@pytest.mark.parametrize("arm", ["full_turn_1m", "warm_start", "fresh_same_capacity"])
def test_v4_manifest_binds_v1_5_and_per_arm_memory_floor(arm):
    manifest = build_manifest_v4(**inputs(arm))

    assert manifest["schema"] == "taiji-r2-fast-ledger-resume-v4"
    assert manifest["approved_budget"] == {"per_arm": 1_590_000.0, "aggregate": 4_560_000.0}
    assert manifest["budget_amendment_sha256"] == "e" * 64
    expected_floor = (
        {"memory_floor_gib": 7.0, "scope": "warm_start_only"}
        if arm == "warm_start"
        else {"memory_floor_gib": 8.0, "scope": "default"}
    )
    assert manifest["resource_policy"] == expected_floor


def test_safe_start_charge_uses_ledger_after_a_settled_stop(monkeypatch):
    charges = {"full_turn_1m": 10.0, "warm_start": 123.0, "fresh_same_capacity": 20.0}
    monkeypatch.setattr(v4.r, "latest_charges", lambda: charges)
    monkeypatch.setattr(v4.r, "latest_reservations", lambda: {})
    assert v4.safe_start_charge({"arm": "warm_start"}) == 123.0

    monkeypatch.setattr(v4, "_ORIGINAL_START_CHARGE", lambda context: 999.0)
    monkeypatch.setattr(
        v4.r,
        "latest_reservations",
        lambda: {"warm_start": {"reservation_id": "e" * 32, "reserved_cpu_seconds": 10.0}},
    )
    assert v4.safe_start_charge({"arm": "warm_start"}) == 999.0


@pytest.mark.parametrize("change", ["arm", "amendment", "budget", "sources", "forecast"])
def test_v4_manifest_refuses_scope_or_budget_drift(change):
    data = inputs()
    if change == "arm":
        data["arm"] = "unknown"
    elif change == "amendment":
        data["budget_amendment"] = {"schema": "taiji-r2-continuation-budget-amendment-v1.4"}
    elif change == "budget":
        data["approved_budget"]["aggregate"] = 4_400_000.0
    elif change == "sources":
        data["source_hashes"].pop(next(iter(SOURCES)))
    else:
        data["forecast"]["selected_arm_feasible"] = False

    with pytest.raises(runner.Refuse):
        build_manifest_v4(**data)
