from __future__ import annotations

from scripts.training.eval_taiji_m4v2_r4_shadow import run_canary


def test_r4_shadow_canary_closes_technical_gates_without_promotion() -> None:
    report = run_canary()

    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert all(report["technical_gates"].values())
    assert set(report["arms"]) == {
        "frozen-parent",
        "r3-fixed-capacity",
        "pressure-driven-growth",
        "random-growth",
        "fixed-large",
    }
    assert report["interpretation"]["matched_capacity_is_required_before_growth_claim"] is True
