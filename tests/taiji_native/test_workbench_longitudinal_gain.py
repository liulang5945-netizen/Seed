from __future__ import annotations

from scripts.training.eval_taiji_workbench_longitudinal_gain import evaluate


def test_workbench_longitudinal_gain_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p4-3-workbench-longitudinal-gain-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
    assert report["holdout_complementary"]["grouped_gain_vs_strongest_single"] >= 0.2
    assert report["train_conflicting_negative_control"]["grouped_pair_reward"] < report[
        "train_conflicting_negative_control"
    ]["strongest_single_reward"]
