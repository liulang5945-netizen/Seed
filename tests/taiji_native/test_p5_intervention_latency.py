from __future__ import annotations

from scripts.training.eval_taiji_p5_intervention_latency import evaluate


def test_p5_intervention_latency_and_lesion_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-p5-intervention-latency-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["planner_rollout"] == "delayed-safe"
    assert report["metrics"]["reactive_rollout"] == "immediate-risky"
    assert report["metrics"]["intervention_replan_required"] is True
    assert report["metrics"]["recovery_rollout"] == "recovery"
    assert report["metrics"]["final_goal_progress"] > 0.0
