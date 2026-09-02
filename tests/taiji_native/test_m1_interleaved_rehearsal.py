from scripts.training.eval_taiji_m1_interleaved_rehearsal import (
    run_interleaved_diagnostics,
)


def test_m1_interleaved_rehearsal_is_diagnostic_only() -> None:
    result = run_interleaved_diagnostics(
        train_count=4,
        holdout_count=4,
        retention_count=4,
        seeds=(11,),
        schedules=("no_replay", "interleave_every_1"),
    )

    assert result["sample_counts"]["phase_a_train"] == 4
    assert result["schedule_replay_counts"] == {
        "no_replay": [0],
        "interleave_every_1": [4],
    }
    assert result["promotable_schedules"] == []
    assert all(
        record["checkpoint"]["restore_digest_matches"]
        for records in result["records"].values()
        for record in records
    )
