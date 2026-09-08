from __future__ import annotations

from scripts.training.eval_taiji_m4v2_r2_canary import run_canary


def test_r2_sg_canary_passes_contract_gates_without_promoting() -> None:
    report = run_canary()

    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert all(report["gates"].values())
    assert set(report["arms"]) == {"slow_only", "fast_only", "fast_replay"}
    assert report["arms"]["fast_replay"]["replay"]["events"] > 0
