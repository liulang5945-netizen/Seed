from __future__ import annotations

import json
from pathlib import Path

REPORT = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "taiji_m4v2_b3_k_loss_diagnostic_20260910.json"
)


def test_b3_k_loss_diagnostic_is_real_and_not_promotion() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "passed"
    assert report["run_kind"] == "learning-diagnostic"
    assert report["training_performed"] is True
    assert report["research_course_executed"] is True
    assert report["holdout_count"] == 3
    assert report["holdout_loss_improved"] is True
    assert report["can_promote"] is False
    assert report["checks"]["candidate_checkpoint_fresh_restore"] is True
    assert report["checks"]["parent_checkpoint_unchanged"] is True
    assert report["checks"]["adapter_rollback_restored"] is True


def test_b3_k_loss_diagnostic_improves_all_structured_components() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    delta = report["holdout_structured_loss_delta"]

    assert delta["combined_mse"] < 0.0
    assert all(value < 0.0 for key, value in delta.items() if key != "combined_mse")
    assert report["updated_workers"] == ["k1.semantic", "k2.transition"]
    assert report["frozen_workers"] == ["k3.outcome_projection"]
    assert report["optimizer_state_present"] is False
    assert report["can_start_r6_formal"] is False
