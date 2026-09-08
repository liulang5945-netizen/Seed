from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4r6_cycle_retention_attribution_20260908.json"
)


def test_m4r6_isolates_half_scale_failure_without_promotion() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["technical_gate_all_passed"] is True
    assert report["can_promote"] is False
    assert report["diagnosis"]["half_scale_c_cycle2_failure_count"] == 1
    assert report["diagnosis"]["half_scale_c_cycle2_failure_seeds"] == [47]
    assert report["diagnosis"]["half_scale_c3_failure_count"] == 0
    assert report["diagnosis"]["legacy_scale_c_cycle2_failure_count"] == 2
    assert report["diagnosis"]["half_scale_c_cycle2_failure_isolated"] is True
    assert report["diagnosis"]["seed47_owner_update_outlier"] is False
    assert report["decision_boundary"]["formal_candidate"] is True
    assert report["decision_boundary"]["promotion_allowed"] is False

    matrix = report["matrix"]
    assert len(matrix) == 9
    assert all(item["frozen_difficulty"]["c_cycle1_bpb"] > 0.0 for item in matrix)
