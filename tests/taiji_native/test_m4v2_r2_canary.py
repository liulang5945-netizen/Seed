from __future__ import annotations

from scripts.training.eval_taiji_m4v2_r2_canary import (
    _calibrate_epsilon_from_scores,
    run_canary,
    run_formal,
)


def test_r2_sg_canary_passes_contract_gates_without_promoting() -> None:
    report = run_canary()

    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert all(report["gates"].values())
    assert set(report["arms"]) == {"slow_only", "fast_only", "fast_replay"}
    assert report["arms"]["fast_replay"]["replay"]["events"] > 0


def test_r2_formal_fails_closed_when_parent_variation_exceeds_epsilon_cap() -> None:
    calibration = _calibrate_epsilon_from_scores({"S": [0.0, 1.0], "G": [2.0, 2.0]})

    assert calibration["max_deviation"] > calibration["upper_bound"]
    assert calibration["calibration_valid"] is False
    assert calibration["epsilon"] == calibration["upper_bound"]


def test_r2_formal_three_course_gate_passes_after_calibration() -> None:
    report = run_formal()

    assert report["status"] == "passed"
    assert report["r2_candidate_accepted"] is True
    assert report["can_promote"] is False
    assert report["calibration"]["calibration_valid"] is True
    assert all(report["gates"].values())
