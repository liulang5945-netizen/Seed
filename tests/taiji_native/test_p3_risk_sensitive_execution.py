from scripts.training.eval_taiji_p3_risk_sensitive_execution import evaluate


def test_risk_sensitive_execution_gate() -> None:
    report = evaluate(seeds=(11,))

    assert report["gate"]["passed"] is True
    run = report["metrics"]["runs"][0]
    assert run["risk_mode_before"] == "stochastic"
    assert run["conflicted_mode_before"] == "conflicted"
    assert run["conflicted_replan"] is True
    assert run["ambiguity_replan"] is True
    assert run["failure_replan"] is True
    assert run["checkpoint_no_replay"] is True
    assert run["checkpoint_branch_preserved"] is True
    assert run["recovery_branch_filters_rejected_action"] is True
    assert run["recovery_branch_selects_low_risk_counterfactual"] is True
    assert run["recovery_candidates_generated_from_affordances"] is True
    assert run["trace_complete"] is True
    assert run["checkpoint_trace_complete"] is True
    assert run["recovery_reader_contributions_recorded"] is True
    assert run["recovery_reader_interactions_recorded"] is True
    assert run["recovery_reader_interactions_checkpoint_preserved"] is True
    assert run["recovery_reader_contribution_revoke_is_exact"] is True
    assert run["recovery_reader_interaction_revoke_is_exact"] is True
