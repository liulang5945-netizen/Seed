from __future__ import annotations

from scripts.training.eval_taiji_structural_growth import evaluate


def test_structural_growth_budget_validation_and_rollback_gate() -> None:
    report = evaluate()

    assert report["gate"]["passed"] is True
    metrics = report["metrics"]
    assert metrics["accepted_validation_score"] == 1.0
    assert metrics["budget_after_accept"] == 0
    assert metrics["checkpoint_request_status"] == "accepted"
    assert metrics["rollback"] is True
    assert metrics["budget_after_rollback"] == 1
    assert metrics["growth_count_after_rollback"] == 0
    assert metrics["trace_count_after_rollback"] == metrics["before_trace_count"]
    assert metrics["rejected_request_status"] == "rejected"
    assert metrics["budget_after_rejection"] == 0
