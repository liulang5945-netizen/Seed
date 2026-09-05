"""Versioned capability verdicts separating absolute, retention, incremental.

R0.5: the legacy ``FoundationEvaluation`` gate conflates three distinct
questions into one pass/fail (F03): whether a model has an ability at all
(absolute), whether a continued child kept what its parent had (retention),
and whether a course produced a pre-registered gain (incremental).  This
module defines a versioned contract that judges each question separately,
leaving the legacy gate, JSON and manifest semantics untouched so old reports
are never rewritten green in place.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MEASUREMENT_VERDICT_FORMAT = "taiji-measurement-verdict-v1"
MEASUREMENT_VERDICT_VERSION = 1
VERDICT_KINDS = ("absolute", "retention", "incremental")
VERDICT_JUDGEMENTS = ("passed", "failed", "not_applicable")
VERDICT_COMPARATORS = ("<=", ">=", "<", ">")


@dataclass(frozen=True)
class CapabilityVerdict:
    """One judgement for one ability on one capability question."""

    ability_id: str
    kind: str
    judgement: str
    metric: str
    value: float | None
    threshold: float | None
    comparator: str
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.ability_id.strip():
            raise ValueError("verdict ability_id cannot be empty")
        if self.kind not in VERDICT_KINDS:
            raise ValueError(f"unsupported verdict kind: {self.kind}")
        if self.judgement not in VERDICT_JUDGEMENTS:
            raise ValueError(f"unsupported verdict judgement: {self.judgement}")
        if self.comparator not in VERDICT_COMPARATORS:
            raise ValueError(f"unsupported verdict comparator: {self.comparator}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": MEASUREMENT_VERDICT_FORMAT,
            "version": MEASUREMENT_VERDICT_VERSION,
            "ability_id": self.ability_id,
            "kind": self.kind,
            "judgement": self.judgement,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "comparator": self.comparator,
            "detail": self.detail,
        }


def _compare(value: float | None, threshold: float | None, comparator: str) -> bool:
    if value is None or threshold is None:
        return False
    if comparator == "<=":
        return value <= threshold
    if comparator == ">=":
        return value >= threshold
    if comparator == "<":
        return value < threshold
    return value > threshold


def _pass_verdict(
    ability_id: str,
    kind: str,
    metric: str,
    value: float | None,
    threshold: float | None,
    comparator: str,
    detail: str = "",
) -> CapabilityVerdict:
    return CapabilityVerdict(
        ability_id=ability_id,
        kind=kind,
        judgement="passed",
        metric=metric,
        value=value,
        threshold=threshold,
        comparator=comparator,
        detail=detail,
    )


def judge_absolute(
    *,
    ability_id: str,
    metric: str,
    value: float | None,
    threshold: float,
    direction: str,
    detail: str = "",
) -> CapabilityVerdict:
    """Absolute capability against a fixed, pre-registered threshold.

    ``direction`` is the metric direction: ``higher_is_better`` requires
    ``value >= threshold``, ``lower_is_better`` requires ``value <=
    threshold``.  This judges existence of an ability on the model alone;
    it never compares against a parent or a fresh sibling.
    """

    comparator = ">=" if direction == "higher_is_better" else "<="
    passed = _compare(value, threshold, comparator)
    return CapabilityVerdict(
        ability_id=ability_id,
        kind="absolute",
        judgement="passed" if passed else "failed",
        metric=metric,
        value=value,
        threshold=float(threshold),
        comparator=comparator,
        detail=detail,
    )


def judge_retention(
    *,
    ability_id: str,
    metric: str,
    child_value: float | None,
    parent_value: float | None,
    tolerance: float,
    direction: str,
    detail: str = "",
) -> CapabilityVerdict:
    """Retention: the continued child must not be significantly worse than
    its parent on the same metric.

    ``tolerance`` is an absolute slack in metric units.  A saturated
    child that matches its parent passes retention; this is deliberately not
    an incremental question.
    """

    if parent_value is None or child_value is None:
        return CapabilityVerdict(
            ability_id=ability_id,
            kind="retention",
            judgement="not_applicable",
            metric=metric,
            value=child_value,
            threshold=None,
            comparator=">=",
            detail="parent or child value missing",
        )
    if direction == "higher_is_better":
        threshold = float(parent_value) - float(tolerance)
        passed = child_value >= threshold
        comparator = ">="
    else:
        threshold = float(parent_value) + float(tolerance)
        passed = child_value <= threshold
        comparator = "<="
    return CapabilityVerdict(
        ability_id=ability_id,
        kind="retention",
        judgement="passed" if passed else "failed",
        metric=metric,
        value=float(child_value),
        threshold=threshold,
        comparator=comparator,
        detail=detail,
    )


def judge_incremental(
    *,
    ability_id: str,
    metric: str,
    child_value: float | None,
    baseline_value: float | None,
    target_delta: float,
    direction: str,
    require_delta: bool,
    detail: str = "",
) -> CapabilityVerdict:
    """Incremental gain against a pre-registered baseline and target.

    ``baseline_value`` is the pre-registered comparison (fresh sibling or
    parent).  When ``require_delta`` is false, a child that simply holds the
    baseline is not a failure: this is the R0.5 contract that a saturated
    ability must not be counted as new learning.  When ``require_delta`` is
    true, the pre-registered gain target must actually be met.
    """

    if baseline_value is None or child_value is None:
        return CapabilityVerdict(
            ability_id=ability_id,
            kind="incremental",
            judgement="not_applicable",
            metric=metric,
            value=child_value,
            threshold=None,
            comparator=">=",
            detail="baseline or child value missing",
        )
    if not require_delta:
        return _pass_verdict(
            ability_id,
            "incremental",
            metric,
            float(child_value),
            None,
            ">=",
            detail="no gain required for this course",
        )
    if direction == "higher_is_better":
        threshold = float(baseline_value) + float(target_delta)
        passed = child_value >= threshold
        comparator = ">="
    else:
        threshold = float(baseline_value) - float(target_delta)
        passed = child_value <= threshold
        comparator = "<="
    return CapabilityVerdict(
        ability_id=ability_id,
        kind="incremental",
        judgement="passed" if passed else "failed",
        metric=metric,
        value=float(child_value),
        threshold=threshold,
        comparator=comparator,
        detail=detail,
    )


def verdict_report(
    verdicts: Mapping[str, CapabilityVerdict],
) -> dict[str, Any]:
    """Assemble a versioned, sorted verdict report for one re-evaluation."""

    ordered = sorted(verdicts.items())
    return {
        "format": MEASUREMENT_VERDICT_FORMAT,
        "version": MEASUREMENT_VERDICT_VERSION,
        "verdicts": [verdict.to_payload() for _, verdict in ordered],
    }
