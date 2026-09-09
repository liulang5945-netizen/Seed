from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_single_step_20260910.json"
)


def test_b3_k_single_step_is_real_but_not_promotion() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["run_kind"] == "learning-pilot"
    assert report["training_performed"] is True
    assert report["research_course_executed"] is True
    assert report["training_update_steps"] == 2
    assert report["can_promote"] is False
    assert report["checks"]["candidate_checkpoint_fresh_restore"] is True
    assert report["checks"]["parent_checkpoint_unchanged"] is True
    assert report["checks"]["adapter_rollback_restored"] is True


def test_b3_k_single_step_keeps_k3_frozen_and_reports_saturated_holdout() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["updated_workers"] == ["k1.semantic", "k2.transition"]
    assert report["frozen_workers"] == ["k3.outcome_projection"]
    assert report["checks"]["k3_owner_unchanged"] is True
    assert report["optimizer_state_present"] is False
    assert report["holdout_before"]["combined_accuracy"] == 1.0
    assert report["holdout_after"]["combined_accuracy"] == 1.0
    assert report["holdout_improved"] is False
    assert report["can_start_r6_formal"] is False
