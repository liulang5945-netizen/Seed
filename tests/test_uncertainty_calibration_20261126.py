"""Tests for the uncertainty calibration facility (zero training)."""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")
import pytest

from taiji.uncertainty_calibration import (
    ECE_BINS,
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_power_temperature,
    verbalize_confidence,
)


def test_perfect_calibration_zero_ece() -> None:
    conf: list[float] = []
    correct: list[int] = []
    for q in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        n = 1000
        conf.extend([q] * n)
        correct.extend([1] * int(q * n) + [0] * (n - int(q * n)))
    ece = expected_calibration_error(conf, correct)["ece"]
    assert ece <= 0.01


def test_overconfident_detected() -> None:
    conf = [0.9] * 100
    correct = [1] * 50 + [0] * 50
    out = expected_calibration_error(conf, correct)
    assert out["ece"] == pytest.approx(0.4, abs=0.01)
    assert out["n"] == 100
    assert len(out["bins"]) == ECE_BINS


def test_brier_score() -> None:
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_temperature_fit_reduces_ece() -> None:
    conf = [0.9] * 100
    correct = [1] * 50 + [0] * 50
    fit = fit_power_temperature(conf, correct)
    assert fit["temperature"] != 1.0
    assert fit["calibrated_ece"] <= expected_calibration_error(conf, correct)["ece"]
    cal = apply_temperature(conf, fit["temperature"])
    assert all(0.0 <= q <= 1.0 for q in cal)


def test_verbalisation_levels() -> None:
    pairs = [(0.95, "我很确定"), (0.70, "比较确定"), (0.50, "不太确定"), (0.10, "不确定")]
    for q, phrase in pairs:
        assert verbalize_confidence(q) == phrase


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [1, 0])
    with pytest.raises(ValueError):
        expected_calibration_error([1.5], [1])
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [2])
    with pytest.raises(ValueError):
        verbalize_confidence(1.2)
