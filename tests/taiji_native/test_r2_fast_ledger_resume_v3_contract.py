import pytest

from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as runner
from scripts.training.run_taiji_r2_fast_ledger_resume import SOURCE_REVISION_SHA256
from scripts.training.run_taiji_r2_fast_ledger_resume_v3 import SOURCES, build_manifest_v3
from scripts.training.taiji_r2_resume_ledger_anchor import BASELINE


def inputs():
    forecast = {
        "arm": "warm_start",
        "feasible": True,
        "global_feasible": False,
        "selected_arm_feasible": True,
        "projection_scope": "selected_arm",
        "deferred_arms": ["full_turn_1m", "fresh_same_capacity"],
        "projection_unit": "target_byte_updates",
        "safety_factor": 1.15,
        "projected_cpu_by_arm": {
            "full_turn_1m": 80_820.9,
            "warm_start": 30_003.1,
            "fresh_same_capacity": 25_832.7,
        },
        "projected_cpu_aggregate": 136_656.7,
        "aggregate_headroom_cpu_seconds": 103_046.9,
        "arm_headroom_by_arm_cpu_seconds": {
            "full_turn_1m": 72_463.4,
            "warm_start": 113_749.7,
            "fresh_same_capacity": 116_833.7,
        },
        "arm_headroom_cpu_seconds": 113_749.7,
        "per_arm_feasible": {
            "full_turn_1m": False,
            "warm_start": True,
            "fresh_same_capacity": True,
        },
    }
    return {
        "arm": "warm_start",
        "source_hashes": {path: "b" * 64 for path in SOURCES},
        "budget_amendment": {"amendment_sha256": "e" * 64},
        "source_revision_sha256": SOURCE_REVISION_SHA256,
        "extension_attempt_sha256": "c" * 64,
        "latest_commit": {"bundle": "commit.pt", "sha256": "d" * 64},
        "forecast": forecast,
        "snapshot": {
            "baseline_sequence": BASELINE["sequence"],
            "baseline_event_sha256": BASELINE["event_sha256"],
            "sequence": 61_114,
            "event_sha256": "a" * 64,
            "charges": {arm: 1_000_000.0 for arm in runner.ARMS},
            "reservations": {},
        },
        "approved_budget": {"per_arm": 1_550_000.0, "aggregate": 4_400_000.0},
    }


def test_v3_manifest_binds_budget_amendment_and_warm_only_floor():
    manifest = build_manifest_v3(**inputs())

    assert manifest["schema"] == "taiji-r2-fast-ledger-resume-v3"
    assert manifest["approved_budget"] == {"per_arm": 1_550_000.0, "aggregate": 4_400_000.0}
    assert manifest["budget_amendment_sha256"] == "e" * 64
    assert manifest["resource_policy"] == {"memory_floor_gib": 7.0, "scope": "warm_start_only"}
    assert manifest["forecast"]["global_feasible"] is False


@pytest.mark.parametrize("change", ["arm", "amendment", "budget"])
def test_v3_manifest_refuses_scope_or_budget_drift(change):
    data = inputs()
    if change == "arm":
        data["arm"] = "full_turn_1m"
    elif change == "amendment":
        data["budget_amendment"] = {}
    else:
        data["approved_budget"]["aggregate"] = 4_300_000.0

    with pytest.raises(runner.Refuse):
        build_manifest_v3(**data)
