"""Tests for the learned uncertainty estimation facility (zero heavy training)."""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")
import torch

from taiji.uncertainty_learning import (
    BaseScore,
    LearnedConfidence,
    TemperatureCalibrator,
    brier,
    ece,
    evaluate,
    make_data,
)


def test_data_deterministic() -> None:
    a = make_data(5, 200)
    b = make_data(5, 200)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
    assert torch.equal(a[2], b[2])


def test_learned_beats_constant_brier() -> None:
    x, y, _ = make_data(11, 2000)
    lc = LearnedConfidence.train(x, y, seed=11, steps=600)
    static = torch.full_like(y, 0.5)
    assert brier(lc.predict(x), y) <= brier(static, y) - 0.03


def test_learned_is_well_calibrated() -> None:
    x, y, _ = make_data(3, 3000)
    lc = LearnedConfidence.train(x, y, seed=3, steps=800)
    assert ece(lc.predict(x), y) <= 0.05


def test_temperature_improves_base_score() -> None:
    x, y, _ = make_data(7, 2000)
    base = BaseScore()
    before = evaluate(base.predict(x), y)
    cal = TemperatureCalibrator()
    cal.fit(x, y)
    after = evaluate(cal.predict(x), y)
    assert after["ece"] <= before["ece"]
    assert after["brier"] <= before["brier"] + 1e-9


def test_ece_extremes() -> None:
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    p_perfect = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert ece(p_perfect, y) == 0.0
    p_bad = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert ece(p_bad, y) >= 0.5
