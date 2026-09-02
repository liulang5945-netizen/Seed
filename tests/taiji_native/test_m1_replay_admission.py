from scripts.training.eval_taiji_m1_replay_admission import (
    run_replay_admission_diagnostics,
)


def test_m1_replay_admission_is_native_and_checkpoint_safe() -> None:
    result = run_replay_admission_diagnostics(
        train_count=4,
        holdout_count=4,
        retention_count=4,
        seeds=(11,),
    )

    assert result["sample_counts"]["phase_a_train"] == 4
    assert result["thresholds"] == {
        "familiarity": 0.90,
        "phase_b_conflict": 0.82,
    }
    assert result["promotable_policies"] == []
    for records in result["records"].values():
        record = records[0]
        assert record["replay"]["replay_considered"] == 4
        assert record["checkpoint"]["restore_digest_matches"]
        assert record["checkpoint"]["read_only_persistent_state"]
        assert record["restored_scores_match"]
