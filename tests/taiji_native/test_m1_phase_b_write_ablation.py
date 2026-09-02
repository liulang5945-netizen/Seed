from scripts.training.eval_taiji_m1_phase_b_write_ablation import (
    run_phase_b_write_diagnostics,
)


def test_m1_phase_b_write_ablation_keeps_no_write_control_and_roundtrip() -> None:
    result = run_phase_b_write_diagnostics(
        train_count=4,
        holdout_count=4,
        retention_count=4,
        seeds=(11,),
        targets=("all",),
        scales=(0.25,),
    )

    assert result["replay_disabled"] is True
    assert result["promotable_candidates"] == []
    assert result["records"]["no_write"][0]["phase_b_memory_writes"] == 0
    record = result["records"]["all@0.25"][0]
    assert record["phase_b_memory_writes"] == 4
    assert record["checkpoint"]["restore_digest_matches"]
    assert record["checkpoint"]["read_only_persistent_state"]
