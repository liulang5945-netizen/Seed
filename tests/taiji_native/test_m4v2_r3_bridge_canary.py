from __future__ import annotations

from scripts.training.eval_taiji_m4v2_r3_bridge import run_canary


def test_r3_bridge_canary_closes_without_promotion() -> None:
    report = run_canary()

    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert all(report["gates"].values())
    assert report["active_probe"]["residual_activity_l1"] > 0.0
    assert report["active_probe"]["probability_max_delta"] > 0.0
