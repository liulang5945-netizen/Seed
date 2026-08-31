from __future__ import annotations

from scripts.training.eval_taiji_small_simulation_gate import evaluate


def test_small_simulation_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p4-2-small-simulation-gate-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
    assert all(gate["passed"] for gate in report["component_gates"].values())
