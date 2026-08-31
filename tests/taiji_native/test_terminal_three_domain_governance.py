from __future__ import annotations

from scripts.training.eval_taiji_terminal_three_domain_governance import evaluate


def test_terminal_three_domain_governance_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p4-12-terminal-three-domain-governance-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["training_contains_actual_terminal_records"] is True
    assert report["metrics"]["three_domain_structural_gain"] is True
    assert report["metrics"]["terminal_requires_approval_and_respects_resources"] is True
    assert report["metrics"]["terminal_failure_stops_and_fresh_recovery_succeeds"] is True
    assert report["metrics"]["rollback_restores_topology_and_budget"] is True
