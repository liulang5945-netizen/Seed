"""Tests for the online source-credibility learning facility (zero training)."""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")

from taiji.credibility_learning import (
    SHIFT_ACC,
    SHIFT_INDEX,
    SOURCES,
    TRUE_ACC,
    BetaCredibility,
    ClaimStream,
    OracleAccuracy,
    StaticPrior,
    evaluate,
)


def test_stream_deterministic_and_shift() -> None:
    a = ClaimStream(7).generate(1200)
    b = ClaimStream(7).generate(1200)
    assert a == b
    s = ClaimStream(7)
    assert s.accuracy("forum", SHIFT_INDEX - 1) == TRUE_ACC["forum"]
    assert s.accuracy("forum", SHIFT_INDEX) == SHIFT_ACC
    assert s.accuracy("owner", SHIFT_INDEX) == TRUE_ACC["owner"]


def test_stream_coverage() -> None:
    items = ClaimStream(11).generate(1500)
    assert {src for src, _ in items} == set(SOURCES)
    assert all(c in (0, 1) for _, c in items)


def test_beta_beats_static_on_shift() -> None:
    items = ClaimStream(21).generate(3000)
    static = evaluate(StaticPrior(), items)
    beta = evaluate(BetaCredibility(), items)
    assert beta["brier"] <= static["brier"] - 0.02
    assert beta["post_abs_error"] <= static["post_abs_error"] - 0.05


def test_beta_converges_to_true_accuracy_without_shift() -> None:
    stream = ClaimStream(3, shift=False)
    items = stream.generate(4000)
    beta = BetaCredibility()
    for _i, (src, correct) in enumerate(items):
        beta.update(src, correct)
    for src in SOURCES:
        assert abs(beta.predict(src, 0) - TRUE_ACC[src]) <= 0.10, src


def test_oracle_tracks_shift() -> None:
    stream = ClaimStream(5)
    items = stream.generate(2000)
    oracle = evaluate(OracleAccuracy(stream), items)
    assert oracle["post_abs_error"] <= 0.30
    assert oracle["brier"] < 0.25


def test_update_is_online() -> None:
    beta = BetaCredibility()
    before = beta.predict("sensor", 0)
    beta.update("sensor", 1)
    after = beta.predict("sensor", 0)
    assert after > before
