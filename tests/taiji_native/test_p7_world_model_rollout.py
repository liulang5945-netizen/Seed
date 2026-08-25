from __future__ import annotations

from scripts.training.eval_taiji_p7_world_model_rollout import evaluate


def test_p7_world_model_rollout_cross_seed_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-p7-world-model-rollout-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["cross_seed_rates"]["horizon_gate"] == 1.0
