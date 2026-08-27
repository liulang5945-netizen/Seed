from __future__ import annotations

from scripts.training.eval_taiji_p3_probability_calibration import evaluate


def test_p3_probability_calibration_gate_passes() -> None:
    report = evaluate(seeds=(11,))

    assert report["aggregate"]["passed"] is True
    assert report["aggregate"]["brier_max"] <= 0.25
    assert report["aggregate"]["nll_max"] <= 0.70
    assert report["aggregate"]["coverage_min"] == 1.0
    assert report["aggregate"]["known_stochastic_prediction_min"] == 1.0
    assert report["aggregate"]["unseen_relation_prediction_min"] == 1.0
    assert report["aggregate"]["holdout_feedback_isolated_min"] == 1.0
    assert report["aggregate"]["checkpoint_continuation_min"] == 1.0
    assert report["aggregate"]["world_model_lesion_min"] == 1.0
