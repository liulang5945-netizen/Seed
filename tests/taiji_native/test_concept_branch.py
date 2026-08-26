from __future__ import annotations

from scripts.training.eval_taiji_concept_branch import evaluate


def test_variable_horizon_concept_branch_gate() -> None:
    report = evaluate()
    metrics = report["metrics"]

    assert report["format"] == "taiji-concept-branch-v1"
    assert report["gate"]["passed"] is True
    assert metrics["good_suffix_affinity"] > metrics["alt_suffix_affinity"]
    assert metrics["selected_branch"] == "branch-good"
    assert metrics["trace_updates"] > 0
    assert metrics["trace_visits_after"] > metrics["trace_visits_before"]
    assert metrics["step_credit_after"] > metrics["step_credit_before"]
    assert metrics["lesioned_affinity"] == 0.0
    assert metrics["checkpoint_recovery"] is True
