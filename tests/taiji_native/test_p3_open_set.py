from __future__ import annotations

from scripts.training.eval_taiji_a3_workspace import build_corpus
from scripts.training.eval_taiji_p3_open_set import build_manifest, evaluate


def test_open_set_world_schema_gate_keeps_growth_and_feedback_contract() -> None:
    train, holdout = build_corpus(seed=20260827, train_count=8, holdout_count=2)
    report = evaluate(train, holdout, seeds=(11,))

    assert report["aggregate"]["passed"] is True
    assert report["aggregate"]["learned_relation_progression_min"] == 1.0
    assert report["aggregate"]["learned_schema_object_min"] == 1.0
    assert report["aggregate"]["learned_schema_relations_min"] == 1.0
    assert report["aggregate"]["learned_schema_actions_min"] == 1.0
    assert report["aggregate"]["learned_schema_checkpoint_min"] == 1.0
    assert report["aggregate"]["learned_cross_episode_min"] == 1.0
    assert report["aggregate"]["learned_history_min"] == 1.0
    assert report["aggregate"]["learned_roundtrip_min"] == 1.0
    assert report["aggregate"]["learned_calibration_min"] == 1.0
    assert report["aggregate"]["none_route_success_max"] == 0.0


def test_open_set_manifest_declares_runtime_schema_boundary() -> None:
    manifest = build_manifest()

    assert manifest["format"] == "taiji-p3-open-set-manifest-v1"
    assert "open object registration" in manifest["controls"]
    assert "open relation registration" in manifest["controls"]
    assert "open action-kind registration" in manifest["controls"]
