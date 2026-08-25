from __future__ import annotations

from scripts.training.eval_taiji_p5_goal_planning import evaluate


def test_p5_goal_planning_runtime_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-p5-goal-planning-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["selected_candidate"] == "safe-route"
    assert report["metrics"]["uncertainty_aware_lesion_candidate"] == "risky-route"
    assert report["metrics"]["runtime_goal_progress"] > 0.0
    assert report["metrics"]["checkpoint_plan_id"] == "safe-route"
