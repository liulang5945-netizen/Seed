from types import SimpleNamespace

import pytest

from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as runner
from scripts.training.run_taiji_r2_fast_ledger_resume import SOURCES as V1_SOURCES
from scripts.training.run_taiji_r2_fast_ledger_resume_v2 import (
    SOURCES,
    build_manifest_v2,
    sample_resources,
)
from scripts.training.taiji_r2_resume_ledger_anchor import BASELINE


def manifest_inputs():
    snapshot = {
        "baseline_sequence": BASELINE["sequence"],
        "baseline_event_sha256": BASELINE["event_sha256"],
        "sequence": 61_114,
        "event_sha256": "a" * 64,
        "charges": {arm: 1_000_000.0 for arm in runner.ARMS},
        "reservations": {},
    }
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
        "memory_policy": {"memory_floor_gib": 7.0, "scope": "warm_start_only"},
        "source_revision_sha256": "2f6938afb910a7bca2c9d5bd5c3d4f79e5a67aa2cd19da08c0a9577589640bf7",
        "extension_attempt_sha256": "c" * 64,
        "latest_commit": {"bundle": "commit.pt", "sha256": "d" * 64},
        "forecast": forecast,
        "snapshot": snapshot,
        "approved_budget": {"per_arm": 1_500_000.0, "aggregate": 4_300_000.0},
    }


def test_v2_records_scoped_memory_floor_and_deferred_arms():
    manifest = build_manifest_v2(**manifest_inputs())

    assert manifest["schema"] == "taiji-r2-fast-ledger-resume-v2"
    assert manifest["resource_policy"] == {"memory_floor_gib": 7.0, "scope": "warm_start_only"}
    assert manifest["forecast"]["global_feasible"] is False
    assert len(manifest["sources"]) == len(SOURCES)
    assert all(path in SOURCES for path in V1_SOURCES)


def test_memory_floor_override_applies_only_to_warm_start():
    fake = SimpleNamespace(
        cpu_percent=lambda interval: 30.0,
        virtual_memory=lambda: SimpleNamespace(available=7.5 * 1024**3),
    )

    result = sample_resources(fake, arm="warm_start")

    assert result["memory_floor_gib"] == 7.0
    with pytest.raises(runner.Refuse, match="below selected-arm floor 8 GiB"):
        sample_resources(fake, arm="fresh_same_capacity")


@pytest.mark.parametrize("change", ["wrong_arm", "wrong_floor", "infeasible_arm"])
def test_v2_refuses_scope_or_budget_drift(change):
    data = manifest_inputs()
    if change == "wrong_arm":
        data["arm"] = "fresh_same_capacity"
    elif change == "wrong_floor":
        data["memory_policy"]["memory_floor_gib"] = 6.0
    else:
        data["forecast"]["selected_arm_feasible"] = False

    with pytest.raises(runner.Refuse):
        build_manifest_v2(**data)
