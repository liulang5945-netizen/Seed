from scripts.training.eval_taiji_m1_support_mask import run_support_diagnostics


def test_m1_support_audit_isolated_and_checkpoint_safe() -> None:
    result = run_support_diagnostics(
        train_count=4,
        holdout_count=4,
        retention_count=4,
        seeds=(11,),
        modes=("shared", "cue_mask"),
        mask_fraction=0.5,
    )

    assert result["replay_disabled"] is True
    assert result["promotable_modes"] == []
    assert set(result["records"]) == {"shared", "cue_mask"}
    for records in result["records"].values():
        record = records[0]
        assert record["support"]["phase_a"]["min_effective_fan_in"] > 0
        assert record["support"]["phase_b"]["min_effective_fan_in"] > 0
        assert record["checkpoint"]["restore_digest_matches"]
        assert record["checkpoint"]["read_only_persistent_state"]
