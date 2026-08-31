from __future__ import annotations

from scripts.training.eval_taiji_interaction_group_multifamily import evaluate


def test_interaction_group_multifamily_leave_one_out_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p4-5-interaction-group-multifamily-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
    assert len(report["held_out_families"]) == 3
    assert len(report["learner_seeds"]) == 3
