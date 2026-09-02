from scripts.training.eval_taiji_m1_identity_route import run_identity_diagnostics


def test_m1_identity_route_is_physical_and_checkpoint_safe() -> None:
    result = run_identity_diagnostics(
        train_count=4,
        holdout_count=4,
        retention_count=4,
        seeds=(11,),
        modes=("identity_route",),
    )

    record = result["records"]["identity_route"][0]
    assert result["answer_table"] is False
    assert record["identity"]["cross_phase_slot_collisions"] == 0
    assert record["identity"]["replacement_count"] == 0
    assert record["repeated_replay"]["same_slot_rate"] == 1.0
    assert record["no_change"]["route_digest_unchanged"]
    assert record["checkpoint"]["restore_bundle_digest_matches"]
