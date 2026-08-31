from __future__ import annotations

from scripts.training.eval_taiji_structural_workspace_net_gain import evaluate


def test_structural_workspace_net_gain_gate() -> None:
    report = evaluate()

    assert report["format"] == "taiji-w7-p4-10-structural-workspace-net-gain-v1"
    assert report["gate"]["passed"] is True
    assert all(report["metrics"].values())
