from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_pilot_probe_20260910.json"
)


def test_b3_sg_probe_is_real_inherited_learning_but_not_promotion() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert report["gates"]["replay_captured_real_events"] is True
    assert report["gates"]["replay_consolidated_and_cleared_fast"] is True
    assert report["gates"]["fresh_restore_matches"] is True
    assert report["gates"]["fresh_restore_is_read_only"] is True
    assert report["gates"]["rollback_matches_parent"] is True
    assert report["parent"]["checkpoint_digest"] == (
        "3e1b39b68be25de672535939401f3c2cf0fb9229c99458237c888be9a50f6675"
    )


def test_b3_replay_arm_beats_both_fixed_capacity_sg_arms() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    arms = report["arms"]
    replay = arms["fast_replay"]["scores"]
    slow = arms["slow_only"]["scores"]
    fast = arms["fast_only"]["scores"]

    for domain in ("S", "G"):
        assert replay[domain] < slow[domain]
        assert replay[domain] < fast[domain]
    assert arms["fast_replay"]["resources"]["replay_events_before_sleep"] == 424
