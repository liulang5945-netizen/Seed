from __future__ import annotations

from scripts.training.eval_taiji_p4_procedural_robustness import evaluate


def test_p4_procedural_sequence_robustness_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-p4-procedural-robustness-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["baseline_transfer_accuracy"] == 1.0
    assert report["metrics"]["checkpoint_continuation_accuracy"] == 1.0
    assert report["metrics"]["similar_interference_accuracy"] >= 0.75
    assert report["metrics"]["budgeted_accuracy_after_forgetting"] < 1.0
