import pytest

from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as runner
from scripts.training.run_taiji_r2_fast_ledger_resume import (
    SOURCE_REVISION_SHA256,
    SOURCES,
    build_manifest,
)
from scripts.training.taiji_r2_resume_ledger_anchor import BASELINE


def inputs():
    snapshot = {
        "baseline_sequence": 59_476,
        "baseline_event_sha256": BASELINE["event_sha256"],
        "sequence": 59_800,
        "event_sha256": "c" * 64,
        "charges": {arm: 1_000_000.0 for arm in runner.ARMS},
        "reservations": {},
    }
    return {
        "arm": "warm_start",
        "source_revision_sha256": SOURCE_REVISION_SHA256,
        "extension_attempt_sha256": "d" * 64,
        "source_hashes": {path: "e" * 64 for path in SOURCES},
        "latest_commit": {"bundle": "commit.pt", "sha256": "f" * 64},
        "forecast": {
            "arm": "warm_start",
            "feasible": True,
            "global_feasible": False,
            "selected_arm_feasible": True,
            "projection_scope": "selected_arm",
            "projection_unit": "target_byte_updates",
            "safety_factor": 1.15,
            "projected_cpu_aggregate": 136_656.7,
            "aggregate_headroom_cpu_seconds": 103_046.9,
            "arm_headroom_cpu_seconds": 113_749.7,
            "deferred_arms": ["full_turn_1m", "fresh_same_capacity"],
            "projected_cpu_by_arm": {
                "full_turn_1m": 80_820.9,
                "warm_start": 30_003.1,
                "fresh_same_capacity": 25_832.7,
            },
            "arm_headroom_by_arm_cpu_seconds": {
                "full_turn_1m": 72_463.4,
                "warm_start": 113_749.7,
                "fresh_same_capacity": 116_833.7,
            },
            "per_arm_feasible": {
                "full_turn_1m": False,
                "warm_start": True,
                "fresh_same_capacity": True,
            },
        },
        "snapshot": snapshot,
        "approved_budget": {"per_arm": 1_500_000.0, "aggregate": 4_300_000.0},
    }


def test_manifest_binds_fast_ledger_anchor_sources_and_resume_checkpoint(tmp_path):
    data = inputs()

    manifest = build_manifest(**data)

    assert manifest["schema"] == "taiji-r2-fast-ledger-resume-v1"
    assert manifest["ledger_anchor"]["sequence"] == data["snapshot"]["sequence"]
    assert manifest["latest_commit"] == data["latest_commit"]
    assert set(manifest["sources"].values()) == {"e" * 64}
    assert manifest["forecast"]["global_feasible"] is False
    assert manifest["forecast"]["feasible"] is True
    assert manifest["projection_scope"] == "selected_arm"


@pytest.mark.parametrize(
    "change",
    [
        "source_revision_sha256",
        "attempt_sha",
        "missing_source",
        "forecast",
        "budget",
        "ledger_anchor",
    ],
)
def test_manifest_refuses_unbound_or_infeasible_resume(change):
    data = inputs()
    if change == "source_revision_sha256":
        data["source_revision_sha256"] = "0" * 64
    elif change == "attempt_sha":
        data["extension_attempt_sha256"] = "invalid"
    elif change == "missing_source":
        data["source_hashes"].pop(SOURCES[0])
    elif change == "forecast":
        data["forecast"]["selected_arm_feasible"] = False
    elif change == "budget":
        data["approved_budget"]["aggregate"] = 4_200_000.0
    else:
        data["snapshot"]["baseline_sequence"] = 1

    with pytest.raises(runner.Refuse):
        build_manifest(**data)
