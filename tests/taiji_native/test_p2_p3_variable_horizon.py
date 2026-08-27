from __future__ import annotations

from scripts.training.eval_taiji_a3_workspace import build_corpus
from scripts.training.eval_taiji_p2_p3_variable_horizon import (
    build_manifest,
    evaluate_variable_horizon,
)


def test_variable_horizon_closure_keeps_two_transition_contract() -> None:
    train, holdout = build_corpus(seed=20260827, train_count=8, holdout_count=2)
    report = evaluate_variable_horizon(
        train,
        holdout,
        seeds=(11,),
        horizons=(3,),
        epochs=20,
        learning_rate=0.2,
    )
    assert report["aggregate"]["passed"] is True
    assert report["aggregate"]["learned_route_success_rate_min"] == 1.0
    assert report["aggregate"]["learned_world_transition_success_rate_min"] == 1.0
    assert report["aggregate"]["none_route_success_rate_max"] == 0.0
    assert report["aggregate"]["closed_assemblies_min_per_episode"] == 3.0


def test_variable_horizon_manifest_declares_runtime_boundary() -> None:
    manifest = build_manifest(train_count=8, holdout_count=2, seeds=(11,), horizons=(3, 4, 5))
    assert manifest["assembly_horizons"] == [3, 4, 5]
    assert "TaijiWorldState two-transition ownership and roundtrip" in manifest["controls"]
