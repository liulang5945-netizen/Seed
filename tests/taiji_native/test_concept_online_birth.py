from __future__ import annotations

from scripts.training.eval_taiji_concept_online_birth import evaluate


def test_online_concept_branch_birth_gate() -> None:
    report = evaluate()
    metrics = report["metrics"]

    assert report["format"] == "taiji-concept-online-birth-v1"
    assert report["gate"]["passed"] is True
    assert metrics["trace_count_before_birth"] == 1
    assert metrics["trace_count_after_birth"] == 2
    assert metrics["novel_trace_id"] is not None
    assert metrics["duplicate_birth"] is None
    assert metrics["feedback_updates"] == 1
    assert metrics["credit_after_feedback"] < metrics["credit_before_feedback"]
    assert metrics["checkpoint_trace_id"] == metrics["novel_trace_id"]
    assert metrics["novel_trace_id"] in metrics["runtime_checkpoint_trace_ids"]
