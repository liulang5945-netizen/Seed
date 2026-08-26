from __future__ import annotations

from scripts.training.eval_taiji_concept_trace_capacity import evaluate


def test_concept_trace_capacity_and_selective_lesion_gate() -> None:
    report = evaluate()
    metrics = report["metrics"]
    curve = metrics["capacity_curve"]

    assert report["format"] == "taiji-concept-trace-capacity-v1"
    assert report["gate"]["passed"] is True
    assert curve["1"]["trace_count"] == 1
    assert curve["2"]["trace_count"] == 2
    assert curve["4"]["trace_count"] == 2
    assert all(item["selected_branch"] == "branch-good" for item in curve.values())
    assert metrics["branch_count_before_add"] == 1
    assert metrics["branch_count_after_add"] == 2
    assert len(metrics["remaining_trace_ids"]) == 1
    assert metrics["remaining_trace_ids"] == metrics["checkpoint_remaining_trace_ids"]
