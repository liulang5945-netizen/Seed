from __future__ import annotations

from scripts.training.eval_taiji_p7_rollout_recovery_transfer import evaluate


def test_p7_rollout_recovery_transfer_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-p7-rollout-recovery-transfer-v1"
    assert report["metrics"]["case_count"] == 27
    assert report["gate"]["passed"] is True
    assert report["metrics"]["cross_seed_rates"]["recovery_gate"] == 1.0
    assert report["metrics"]["calibration_policy_lesion"]["detected"] is True
