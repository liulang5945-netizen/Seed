from scripts.training.eval_taiji_m1_support_alignment import run_alignment_diagnostics


def test_m1_support_alignment_isolated_and_checkpoint_safe() -> None:
    result = run_alignment_diagnostics(
        train_count=4,
        holdout_count=4,
        retention_count=4,
        seeds=(11,),
        modes=("shared", "write_mask_only", "read_mask_only", "aligned_mask"),
        mask_fraction=0.5,
    )

    assert result["replay_disabled"] is True
    assert set(result["records"]) == {
        "shared",
        "write_mask_only",
        "read_mask_only",
        "aligned_mask",
    }
    for records in result["records"].values():
        record = records[0]
        assert record["support"]["phase_a"]["read_mean_effective_fan_in"] > 0
        assert record["support"]["phase_b"]["write_mean_effective_fan_in"] > 0
        assert record["checkpoint"]["restore_digest_matches"]
        assert record["checkpoint"]["read_only_persistent_state"]
