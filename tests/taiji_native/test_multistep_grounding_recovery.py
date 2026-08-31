"""P2-10 Taiji-owned multi-step grounding and recovery Gate."""

from scripts.training.eval_taiji_multistep_grounding_recovery import evaluate


def test_multistep_grounding_recovery_gate_passes() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p2-10-multistep-grounding-recovery-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
