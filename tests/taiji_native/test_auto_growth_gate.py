from __future__ import annotations

from scripts.training.eval_taiji_auto_growth import evaluate


def test_auto_growth_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-auto-growth-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["proposal_emitted_after_steps"] == 3
    assert report["metrics"]["holdout_after_local_learning"] > report["metrics"]["holdout_before"]
    assert report["metrics"]["checkpoint_continuation"] is True
    assert report["metrics"]["rollback"] is True
    assert report["metrics"]["rejected_without_budget"] is True
