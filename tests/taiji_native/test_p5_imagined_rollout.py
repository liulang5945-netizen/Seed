from __future__ import annotations

from scripts.training.eval_taiji_p5_imagined_rollout import evaluate


def test_p5_imagined_rollout_replan_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-p5-imagined-rollout-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["selected_rollout"] == "safe-rollout"
    assert report["metrics"]["selected_rollout_steps"] == 2
    assert report["metrics"]["selected_provenance"] == "imagined"
    assert report["metrics"]["replan_required_after_error"] is True
    assert report["metrics"]["checkpoint_replan_required"] is True
