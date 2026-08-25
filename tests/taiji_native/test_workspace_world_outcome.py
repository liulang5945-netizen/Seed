from __future__ import annotations

from scripts.training.eval_taiji_a3_workspace import build_corpus
from scripts.training.eval_taiji_a3_world_workspace import evaluate_world_workspace


def test_workspace_selection_changes_a_contiguous_world_outcome() -> None:
    train, holdout = build_corpus(seed=20260825, train_count=48, holdout_count=24)
    report = evaluate_world_workspace(
        train,
        holdout,
        seeds=(11, 29, 47),
        capacity=2,
        epochs=100,
        learning_rate=0.2,
    )

    assert report["format"] == "taiji-a3-world-workspace-v1"
    assert report["aggregate"]["history_length"] == 2
    assert report["gate"]["passed"] is True
    assert report["aggregate"]["learned_final_success_accuracy"] == 1.0
    assert report["aggregate"]["learned_gain_vs_strongest_single_min"] >= 0.2
