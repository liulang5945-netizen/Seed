from __future__ import annotations

from scripts.training.eval_taiji_p4_semantic_scale import evaluate_scale


def test_p4_semantic_scale_handles_noisy_multi_factor_holdout() -> None:
    report = evaluate_scale()

    assert report["format"] == "taiji-p4-semantic-scale-v1"
    assert report["gate"]["passed"] is True
    assert report["train_records"] == 60
    assert report["metrics"]["semantic_consolidated_error"] < 0.2
    assert (
        report["metrics"]["semantic_consolidated_error"]
        < report["metrics"]["episodic_nearest_error"]
    )
    assert report["metrics"]["replay_lesion_error"] > 2.0
    assert report["metrics"]["checkpoint_continuation_error"] < 0.2
