from __future__ import annotations

from scripts.training.eval_taiji_concept_capacity import evaluate


def test_concept_capacity_checkpoint_and_lesion_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-concept-capacity-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["capacity_curve_passed"] is True
    assert report["metrics"]["checkpoint_continuation"] is True
    assert report["metrics"]["lesion_removed_concept"] is True
    assert all(item["retained_concepts"] == item["capacity"] for item in report["capacity_curve"])
