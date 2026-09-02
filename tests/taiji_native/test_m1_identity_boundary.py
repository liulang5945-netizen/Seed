from scripts.training.eval_taiji_m1_identity_boundary import run_boundary_diagnostics


def test_m1_identity_boundary_is_explicit_and_recoverable() -> None:
    result = run_boundary_diagnostics(
        train_count=4,
        holdout_count=4,
        seeds=(11,),
        capacity=16,
    )

    record = result["records"][0]
    assert record["unseen"]["unbound_rate"] == 1.0
    assert record["unseen"]["route_digest_unchanged"]
    assert record["capacity_stress"]["replacement_is_explicit"]
    assert record["checkpoint"]["bundle_digest_matches"]
