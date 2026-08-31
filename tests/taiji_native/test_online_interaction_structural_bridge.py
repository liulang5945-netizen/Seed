from __future__ import annotations

from scripts.training.eval_taiji_online_interaction_structural_bridge import evaluate


def test_online_interaction_structural_bridge_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p4-9-online-interaction-structural-bridge-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
