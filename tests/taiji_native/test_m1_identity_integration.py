from scripts.training.eval_taiji_m1_identity_integration import run_integration_diagnostics


def test_m1_identity_integration_has_fallback_and_no_execution() -> None:
    result = run_integration_diagnostics(
        train_count=4,
        holdout_count=4,
        retention_count=4,
        seeds=(11,),
    )

    assert result["promotable"]
    record = result["records"]["identity_route_fallback"][0]
    assert record["identity_bound_count"] == 4
    assert record["fallback_count"] == 4
    assert record["no_action_intent"]
    assert record["checkpoint"]["bundle_digest_matches"]
