from __future__ import annotations

from scripts.training.eval_taiji_p4_homeostasis import evaluate


def test_p4_homeostasis_and_sleep_play_lesion_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-p4-homeostasis-v1"
    assert report["gate"]["passed"] is True
    assert report["metrics"]["adaptive_mode"] == "sleep"
    assert report["metrics"]["sleep_fatigue"] < report["metrics"]["sleep_lesion_fatigue"]
    assert report["metrics"]["play_stress"] < report["metrics"]["play_lesion_stress"]
    assert report["metrics"]["runtime_checkpoint_restored"] is True
