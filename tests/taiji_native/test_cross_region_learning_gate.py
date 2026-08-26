from __future__ import annotations

from scripts.training.eval_taiji_cross_region_learning import evaluate


def test_cross_region_learning_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-cross-region-learning-v1"
    assert report["gate"]["passed"] is True
    assert (
        report["metrics"]["selected_holdout_transfer"]
        > report["metrics"]["fixed_full_holdout_transfer"]
    )
    assert (
        report["metrics"]["selected_holdout_transfer"]
        > report["metrics"]["random_holdout_transfer"]
    )
    assert report["metrics"]["checkpoint_continuation"] is True
    assert report["metrics"]["connection_lesion_excludes_selected"] is True
    assert report["metrics"]["region_lesion_excludes_routes"] is True
    assert report["metrics"]["resource_constrained_selection"] is True
