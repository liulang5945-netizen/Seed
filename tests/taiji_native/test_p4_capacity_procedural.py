from __future__ import annotations

from scripts.training.eval_taiji_p4_capacity_procedural import evaluate


def test_p4_capacity_and_procedural_skill_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-p4-capacity-procedural-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["procedural_accuracy"] >= 0.95
    assert report["metrics"]["skill_lesion_accuracy"] < report["metrics"]["procedural_accuracy"]
    assert report["metrics"]["episode_id_lesion_accuracy"] >= 0.95
    assert report["metrics"]["checkpoint_continuation_accuracy"] >= 0.95
    assert all(item["retained_records"] == item["capacity"] for item in report["capacity_curve"])
