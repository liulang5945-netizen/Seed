from __future__ import annotations

from scripts.training.eval_taiji_continuous_structural_growth import evaluate


def test_continuous_structural_growth_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p4-10b-continuous-structural-growth-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
