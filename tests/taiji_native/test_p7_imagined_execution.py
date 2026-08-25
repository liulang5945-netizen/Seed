from __future__ import annotations

from scripts.training.eval_taiji_p7_imagined_execution import evaluate


def test_p7_imagined_rollout_real_execution_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-p7-imagined-execution-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["cross_seed_rates"]["execution_gate"] == 1.0
