from __future__ import annotations

from scripts.training.eval_taiji_p5_replan_calibration import evaluate


def test_p5_replan_execution_and_calibration_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-p5-replan-calibration-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["first_selected_rollout"] == "safe-rollout"
    assert report["metrics"]["first_replan_required"] is True
    assert report["metrics"]["second_selected_rollout"] == "risky-rollout"
    assert report["metrics"]["second_replan_required"] is False
    assert report["metrics"]["checkpoint_risky_confidence"] == 1.0
