"""Fixed-denominator scoring for the R1 course and the R5 gates.

The old F1/F1b scorer mixed denominators, skipped unparsed queries and treated
"output changed" as cut successfully without checking correctness.  This module
fixes the scoring semantics once:

* the denominator is the declared answer-turn list, not a filtered subset;
* a missing prediction, parse failure or abstention is still scored and wrong;
* cut is successful only when a previously correct answer becomes wrong;
* rescue is successful only when the cut answer is restored to correct;
* relevant/unrelated swaps are judged against the expected answer, not activity.

It has no model dependency and is unit-testable with synthetic predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .internalization import content_digest

SCORE_FORMAT = "taiji-native-core-fixed-denominator-score-v1"


@dataclass(frozen=True)
class ExpectedTurnV1:
    episode_id: str
    turn_index: int
    category: str
    expected: bytes

    @property
    def key(self) -> tuple[str, int]:
        return (self.episode_id, self.turn_index)


@dataclass(frozen=True)
class TurnPredictionV1:
    episode_id: str
    turn_index: int
    produced: bytes
    parse_ok: bool = True
    selected_value: str | None = None

    @property
    def key(self) -> tuple[str, int]:
        return (self.episode_id, self.turn_index)


@dataclass(frozen=True)
class MechanismCaseV1:
    """One correctness probe around a baseline prediction."""

    episode_id: str
    turn_index: int
    expected: bytes
    baseline: TurnPredictionV1
    cut: TurnPredictionV1 | None = None
    rescue: TurnPredictionV1 | None = None
    relevant_swap: TurnPredictionV1 | None = None
    relevant_expected: bytes | None = None
    unrelated_swap: TurnPredictionV1 | None = None

    @property
    def key(self) -> tuple[str, int]:
        return (self.episode_id, self.turn_index)


def _ratio(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _exact(prediction: TurnPredictionV1 | None, expected: bytes) -> bool:
    return prediction is not None and prediction.parse_ok and prediction.produced == expected


def score_run(
    expected: Sequence[ExpectedTurnV1],
    predictions: Mapping[tuple[str, int], TurnPredictionV1],
    mechanisms: Sequence[MechanismCaseV1] = (),
) -> dict[str, Any]:
    """Score every declared answer turn; missing/parse-failure counts as wrong."""

    scored = len(expected)
    exact = parse_failures = missing = abstentions = 0
    by_category: dict[str, dict[str, int]] = {}
    state_turns = state_exact = 0
    for turn in expected:
        bucket = by_category.setdefault(
            turn.category, {"turns": 0, "exact": 0, "parse_failures": 0}
        )
        bucket["turns"] += 1
        prediction = predictions.get(turn.key)
        if prediction is None:
            missing += 1
            continue
        if not prediction.parse_ok:
            parse_failures += 1
            bucket["parse_failures"] += 1
            continue
        if not prediction.produced:
            abstentions += 1
            continue
        if prediction.produced == turn.expected:
            exact += 1
            bucket["exact"] += 1
        if prediction.selected_value is not None:
            state_turns += 1
            expected_value = turn.expected.decode("utf-8")
            if prediction.selected_value in expected_value:
                state_exact += 1

    cut_cases = rescue_cases = cut_drop = rescue_gain = 0
    relevant_cases = relevant_exact = 0
    unrelated_cases = unrelated_keep = 0
    for case in mechanisms:
        baseline_correct = _exact(case.baseline, case.expected)
        if case.cut is not None and baseline_correct:
            cut_cases += 1
            if not _exact(case.cut, case.expected):
                cut_drop += 1
                if case.rescue is not None:
                    rescue_cases += 1
                    if _exact(case.rescue, case.expected):
                        rescue_gain += 1
        if case.relevant_swap is not None:
            relevant_cases += 1
            relevant_target = (
                case.expected if case.relevant_expected is None else case.relevant_expected
            )
            if _exact(case.relevant_swap, relevant_target):
                relevant_exact += 1
        if case.unrelated_swap is not None:
            unrelated_cases += 1
            if _exact(case.unrelated_swap, case.expected):
                unrelated_keep += 1

    payload = {
        "format": SCORE_FORMAT,
        "version": 1,
        "scored_turns": scored,
        "exact": exact,
        "answer_exact": _ratio(exact, scored),
        "parse_failures": parse_failures,
        "missing_predictions": missing,
        "abstentions": abstentions,
        "by_category": by_category,
        "state_turns": state_turns,
        "state_exact": _ratio(state_exact, state_turns),
        "cut_cases": cut_cases,
        "cut_drop": _ratio(cut_drop, cut_cases),
        "rescue_cases": rescue_cases,
        "rescue_gain": _ratio(rescue_gain, rescue_cases),
        "relevant_cases": relevant_cases,
        "relevant_swap_exact": _ratio(relevant_exact, relevant_cases),
        "unrelated_cases": unrelated_cases,
        "unrelated_swap_keep": _ratio(unrelated_keep, unrelated_cases),
    }
    payload["digest"] = content_digest(payload)
    return payload


def failed_gates(score: Mapping[str, Any], *, required: Mapping[str, float]) -> list[str]:
    """Return the gates whose value is below the declared floor."""

    failures: list[str] = []
    for name, floor in required.items():
        value = score.get(name)
        if value is None or value < floor:
            failures.append(f"{name}: {value} < {floor}")
    return failures
