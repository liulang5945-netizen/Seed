from scripts.training.eval_taiji_m1_signal_diagnostics import run_b5_diagnostics


def test_m1_signal_diagnostic_keeps_shared_decoder_and_reports_b5_contract() -> None:
    result = run_b5_diagnostics(
        train_count=4,
        holdout_count=4,
        retention_count=4,
        seeds=(11,),
        replay_scales=(0.25,),
        replay_targets=("all",),
    )

    assert result["sample_counts"]["phase_a_train"] == 4
    assert len(result["records"]) == 1
    assert result["records"][0]["measurement"]["ability_id"] == "b5_continual_learning"
    assert result["candidates_passing_b5"] == []
