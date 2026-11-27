"""Uncertainty calibration measurement facility (`taiji-uncertainty-calibration-v1`).

Zero-training facility for the "表达并处理不确定性" clause (coverage gap H). It measures
expected calibration error (ECE) and Brier score of confidence/correctness pairs, fits a
post-hoc power calibration `q' = q**t` on a calibration split, and maps confidence to a short
Chinese verbalisation. The power fit and priors are engineering prototypes, not learned
capabilities; uncertainty *learning* would be a later training line.
"""

from __future__ import annotations

import math
from typing import Any

SCHEMA = "taiji-uncertainty-calibration-v1"
ECE_BINS = 10
T_GRID: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0)
VERBAL_LEVELS: tuple[tuple[float, str], ...] = (
    (0.85, "我很确定"),
    (0.65, "比较确定"),
    (0.45, "不太确定"),
    (0.0, "不确定"),
)


def _validate(conf: list[float], correct: list[int]) -> None:
    if len(conf) != len(correct):
        raise ValueError("confidence and correctness must have equal length")
    if not conf:
        raise ValueError("empty input")
    for q in conf:
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"confidence out of range: {q}")
    for c in correct:
        if c not in (0, 1):
            raise ValueError(f"correctness must be 0/1, got {c}")


def expected_calibration_error(
    conf: list[float], correct: list[int], n_bins: int = ECE_BINS
) -> dict[str, Any]:
    _validate(conf, correct)
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for q, c in zip(conf, correct, strict=True):
        idx = min(n_bins - 1, int(q * n_bins))
        bins[idx].append((q, c))
    total = len(conf)
    ece = 0.0
    details: list[dict[str, Any]] = []
    for i, entries in enumerate(bins):
        if not entries:
            details.append({"bin": i, "n": 0})
            continue
        avg_conf = sum(q for q, _ in entries) / len(entries)
        accuracy = sum(c for _, c in entries) / len(entries)
        gap = abs(avg_conf - accuracy)
        ece += gap * len(entries) / total
        details.append(
            {
                "bin": i,
                "n": len(entries),
                "avg_conf": round(avg_conf, 4),
                "accuracy": round(accuracy, 4),
                "gap": round(gap, 4),
            }
        )
    return {"ece": round(ece, 4), "n": total, "bins": details}


def brier_score(conf: list[float], correct: list[int]) -> float:
    _validate(conf, correct)
    return round(sum((q - c) ** 2 for q, c in zip(conf, correct, strict=True)) / len(conf), 4)


def fit_power_temperature(conf: list[float], correct: list[int]) -> dict[str, float]:
    """Grid-search `q' = q**t` minimising negative log-likelihood; engineering prototype."""
    _validate(conf, correct)
    best_t, best_nll = 1.0, float("inf")
    for t in T_GRID:
        nll = 0.0
        for q, c in zip(conf, correct, strict=True):
            qq = min(1.0 - 1e-6, max(1e-6, q**t))
            nll -= c * math.log(qq) + (1 - c) * math.log(1 - qq)
        if nll < best_nll:
            best_t, best_nll = t, nll
    cal = [min(1.0 - 1e-6, max(1e-6, q**best_t)) for q in conf]
    return {
        "temperature": best_t,
        "nll": round(best_nll, 4),
        "calibrated_ece": expected_calibration_error(cal, correct)["ece"],
    }


def apply_temperature(conf: list[float], temperature: float) -> list[float]:
    return [min(1.0 - 1e-6, max(1e-6, q**temperature)) for q in conf]


def verbalize_confidence(conf: float) -> str:
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"confidence out of range: {conf}")
    for threshold, phrase in VERBAL_LEVELS:
        if conf >= threshold:
            return phrase
    return VERBAL_LEVELS[-1][1]
