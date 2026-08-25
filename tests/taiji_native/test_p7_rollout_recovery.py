from __future__ import annotations

from scripts.training.eval_taiji_p7_rollout_recovery import evaluate


def test_p7_rollout_recovery_cross_seed_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-p7-rollout-recovery-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["cross_seed_rates"]["recovery_gate"] == 1.0
