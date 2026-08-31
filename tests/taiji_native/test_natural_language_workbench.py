"""P2-8 Taiji-owned natural-language Workbench execution Gate."""

from scripts.training.eval_taiji_natural_language_workbench import evaluate


def test_natural_language_workbench_gate_passes() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p2-8-natural-language-workbench-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
