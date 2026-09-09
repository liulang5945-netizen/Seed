from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_r6_matched_control_aggregate_20260910.json"
)


def test_m4v2_r6_matched_control_aggregate_closes_causal_resource_gate() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    gates = report["gates"]

    assert report["status"] == "passed"
    assert report["report_format"] == "taiji-m4v2-r6-matched-control-aggregate-v1"
    assert report["control_revision"]["format"] == "taiji-m4v2-r6-matched-control-v2"
    assert len(report["cells"]) == 9
    assert report["blocking_failures"] == []
    assert gates["all_cells_executed_passed"] is True
    assert gates["candidate_holdout_floor"] is True
    assert gates["k3_lesion_breaks_gain"] is True
    assert gates["candidate_over_matched_wall_budget"] is True
    assert gates["candidate_over_matched_peak_budget"] is True
    assert gates["paired_frozen_parent_delta_available"] is True
    assert gates["paired_matched_fixed_capacity_delta_available"] is True
    assert gates["paired_frozen_parent_delta_floor"] is True
    assert gates["paired_matched_fixed_capacity_delta_floor"] is True
    assert report["metrics"][
        "candidate_minus_frozen_parent_task_success_rate"
    ]["one_sided_95_student_t_lower_bound"] >= 0.25
    assert report["metrics"][
        "candidate_minus_matched_fixed_capacity_task_success_rate"
    ]["one_sided_95_student_t_lower_bound"] >= 0.25
    assert report["can_start_r6_formal"] is False
    assert report["can_promote"] is False
