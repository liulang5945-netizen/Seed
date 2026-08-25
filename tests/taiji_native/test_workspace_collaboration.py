from __future__ import annotations

from scripts.training.eval_taiji_a3_workspace import build_corpus
from taiji import WorkspaceCollaborationEvaluator


def test_a3_workspace_composition_beats_single_and_dense_controls() -> None:
    train, holdout = build_corpus(seed=20260825, train_count=48, holdout_count=24)
    report = WorkspaceCollaborationEvaluator(content_dim=2, seeds=(11, 29, 47)).evaluate(
        train,
        holdout,
        capacity=2,
        epochs=100,
        learning_rate=0.2,
    )

    assert report["format"] == "taiji-a3-workspace-composition-v1"
    assert report["gate"]["passed"] is True
    assert report["aggregate"]["learned_gain_vs_strongest_single_min"] > 0.05
    assert report["aggregate"]["learned_gain_vs_dense"] > 0.05
    assert report["aggregate"]["exact_route_rate_min"] >= 0.9
