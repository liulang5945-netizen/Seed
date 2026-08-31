"""P2-12 Taiji-owned natural-language controlled write Gate."""

from scripts.training.eval_taiji_natural_language_write import evaluate


def test_natural_language_write_gate_passes() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p2-12-natural-language-write-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
