from __future__ import annotations

from scripts.training.eval_taiji_concept_suffix import evaluate


def test_state_conditioned_concept_suffix_gate() -> None:
    report = evaluate()
    metrics = report["metrics"]
    runtime = metrics["runtime"]

    assert report["gate"]["passed"] is True
    assert metrics["concept_count"] == 1
    assert metrics["full_affinity"] > metrics["reversed_affinity"]
    assert metrics["suffix_affinity"] > metrics["wrong_state_affinity"]
    assert metrics["reordered_affinity"] == 0.0
    assert max(metrics["trace_step_credit"]) > min(metrics["trace_step_credit"])
    assert metrics["checkpoint_recovery"] is True
    assert runtime["selected_full_rollout"] == "suffix-good"
    assert runtime["selected_after_checkpoint"] == "suffix-good"
    assert runtime["remaining_action_kinds"] == ("confirm", "archive")
    assert runtime["suffix_affinity_after_execution"] > 0.0
    assert runtime["suffix_affinity_after_checkpoint"] == runtime["suffix_affinity_after_execution"]
