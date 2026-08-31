from __future__ import annotations

from scripts.training.eval_taiji_cross_domain_structural_gain import evaluate


def test_cross_domain_structural_gain_gate() -> None:
    report = evaluate()

    assert report["gate"]["passed"] is True
    assert report["metrics"]["training_contains_actual_editor_and_mcp_records"] is True
    assert report["metrics"]["cross_domain_structural_gain"] is True
    assert report["metrics"]["old_workspace_capability_retention"] is True
    assert report["metrics"]["lesion_removes_cross_domain_gain"] is True
    assert report["metrics"]["rollback_restores_topology_and_budget"] is True
