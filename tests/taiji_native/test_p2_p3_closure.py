from __future__ import annotations

from scripts.training.eval_taiji_a3_workspace import build_corpus
from scripts.training.eval_taiji_p2_p3_closure import evaluate_closure


def test_perception_workspace_world_closure_gate_keeps_lineage_and_lesion() -> None:
    train, holdout = build_corpus(seed=20260827, train_count=4, holdout_count=2)

    report = evaluate_closure(train, holdout, seeds=(11,), epochs=100)

    assert report["aggregate"]["passed"] is True
    assert report["aggregate"]["lineage_rate_min"] == 1.0
    assert report["aggregate"]["none_route_success_rate_max"] == 0.0
