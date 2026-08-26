from __future__ import annotations

from scripts.training.eval_taiji_concept_sequence import evaluate


def test_concept_sequence_rollout_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-concept-sequence-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["baseline_without_sequence_prior"] == "sequence-reversed"
    assert report["metrics"]["sequence_prior_selection"] == "sequence-good"
    assert report["metrics"]["concept_lesion_selection"] == "sequence-reversed"
    assert report["metrics"]["native_runtime"]["feedback_replan"] is True
