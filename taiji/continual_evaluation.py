"""Versioned measurement contracts for Taiji's continual-growth experiments.

The contract keeps three questions separate:

* what capability is present at a checkpoint (an absolute measurement);
* what changed relative to the parent checkpoint (a parent delta); and
* what changed relative to a declared comparison arm (a comparison delta).

It is intentionally independent from the model and from PyTorch.  An
evaluator may construct these objects from any checkpoint implementation, but
the model never receives course phase labels or evaluator-only metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CONTINUAL_EVALUATION_FORMAT = "taiji-continual-evaluation-v1"
CONTINUAL_EVALUATION_VERSION = 1
METRIC_DIRECTIONS = ("higher", "lower")
METRIC_BASELINE_KINDS = ("absolute", "parent", "comparison")


class ContinualEvaluationContractError(ValueError):
    """Base error for malformed or semantically unsafe evaluation contracts."""


class MissingParentBaselineError(ContinualEvaluationContractError):
    """Raised when a parent-relative aggregate has no parent baseline."""


class CrossDomainAggregationError(ContinualEvaluationContractError):
    """Raised when raw values from multiple domains are aggregated together."""


def _content_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContinualEvaluationContractError(f"{field_name} must not be empty")


def _require_finite(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ContinualEvaluationContractError(f"{field_name} must be finite")


def _verify_envelope(payload: Mapping[str, Any], expected_format: str) -> None:
    if payload.get("format") != expected_format:
        raise ContinualEvaluationContractError(
            f"unsupported contract format: {payload.get('format')!r}"
        )
    if payload.get("version") != CONTINUAL_EVALUATION_VERSION:
        raise ContinualEvaluationContractError(
            f"unsupported contract version: {payload.get('version')!r}"
        )


def _verify_digest(payload: Mapping[str, Any], digest_field: str) -> None:
    actual = payload.get(digest_field)
    if not isinstance(actual, str) or not actual:
        raise ContinualEvaluationContractError(f"{digest_field} must be present")
    unsigned = dict(payload)
    unsigned.pop(digest_field, None)
    expected = _content_digest(unsigned)
    if actual != expected:
        raise ContinualEvaluationContractError(f"{digest_field} does not match payload")


@dataclass(frozen=True)
class MetricSpec:
    """The semantic contract for one measured capability."""

    name: str
    direction: str
    unit: str
    baseline_kind: str = "parent"
    domain_id: str = "*"
    critical: bool = False
    catastrophic_forgetting_threshold: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "metric name")
        _require_text(self.unit, "metric unit")
        _require_text(self.domain_id, "metric domain_id")
        if self.direction not in METRIC_DIRECTIONS:
            raise ContinualEvaluationContractError(
                f"unsupported metric direction: {self.direction!r}"
            )
        if self.baseline_kind not in METRIC_BASELINE_KINDS:
            raise ContinualEvaluationContractError(
                f"unsupported metric baseline_kind: {self.baseline_kind!r}"
            )
        if self.catastrophic_forgetting_threshold is not None:
            _require_finite(
                self.catastrophic_forgetting_threshold,
                "catastrophic_forgetting_threshold",
            )
            if self.catastrophic_forgetting_threshold <= 0:
                raise ContinualEvaluationContractError(
                    "catastrophic_forgetting_threshold must be positive"
                )
        if self.critical and self.catastrophic_forgetting_threshold is None:
            raise ContinualEvaluationContractError(
                "critical metrics must declare catastrophic_forgetting_threshold"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": CONTINUAL_EVALUATION_FORMAT,
            "version": CONTINUAL_EVALUATION_VERSION,
            "type": "metric_spec",
            "name": self.name,
            "direction": self.direction,
            "unit": self.unit,
            "baseline_kind": self.baseline_kind,
            "domain_id": self.domain_id,
            "critical": self.critical,
            "catastrophic_forgetting_threshold": self.catastrophic_forgetting_threshold,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MetricSpec:
        _verify_envelope(payload, CONTINUAL_EVALUATION_FORMAT)
        if payload.get("type") != "metric_spec":
            raise ContinualEvaluationContractError("payload is not a metric_spec")
        return cls(
            name=str(payload["name"]),
            direction=str(payload["direction"]),
            unit=str(payload["unit"]),
            baseline_kind=str(payload.get("baseline_kind", "parent")),
            domain_id=str(payload.get("domain_id", "*")),
            critical=bool(payload.get("critical", False)),
            catastrophic_forgetting_threshold=payload.get(
                "catastrophic_forgetting_threshold"
            ),
        )


@dataclass(frozen=True)
class CoursePhase:
    """Evaluator-only description of one course phase."""

    phase_id: str
    source_digest: str
    dataset_digest: str
    train_budget_bytes: int
    holdout_budget_bytes: int
    course_seed: int
    order: int

    def __post_init__(self) -> None:
        for field_name in ("phase_id", "source_digest", "dataset_digest"):
            _require_text(getattr(self, field_name), f"phase {field_name}")
        if self.train_budget_bytes <= 0 or self.holdout_budget_bytes <= 0:
            raise ContinualEvaluationContractError(
                "phase train and holdout budgets must be positive"
            )
        if self.order <= 0:
            raise ContinualEvaluationContractError("phase order must be positive")

    def to_payload(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "source_digest": self.source_digest,
            "dataset_digest": self.dataset_digest,
            "train_budget_bytes": self.train_budget_bytes,
            "holdout_budget_bytes": self.holdout_budget_bytes,
            "course_seed": self.course_seed,
            "order": self.order,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CoursePhase:
        return cls(
            phase_id=str(payload["phase_id"]),
            source_digest=str(payload["source_digest"]),
            dataset_digest=str(payload["dataset_digest"]),
            train_budget_bytes=int(payload["train_budget_bytes"]),
            holdout_budget_bytes=int(payload["holdout_budget_bytes"]),
            course_seed=int(payload["course_seed"]),
            order=int(payload["order"]),
        )


@dataclass(frozen=True)
class ContinualCourseManifest:
    """Content-addressed course contract with no model-visible phase label."""

    course_id: str
    phases: tuple[CoursePhase, ...]
    manifest_revision: int = 1

    def __post_init__(self) -> None:
        _require_text(self.course_id, "course_id")
        if self.manifest_revision != 1:
            raise ContinualEvaluationContractError("unsupported manifest revision")
        if not self.phases:
            raise ContinualEvaluationContractError("course must contain at least one phase")
        phase_ids = [phase.phase_id for phase in self.phases]
        if len(set(phase_ids)) != len(phase_ids):
            raise ContinualEvaluationContractError("course phase_id values must be unique")
        orders = [phase.order for phase in self.phases]
        if len(set(orders)) != len(orders):
            raise ContinualEvaluationContractError("course phase order values must be unique")

    def phase(self, phase_id: str) -> CoursePhase:
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        raise ContinualEvaluationContractError(f"unknown course phase: {phase_id}")

    def model_context(self, phase_id: str) -> dict[str, Any]:
        """Return the model-facing context; evaluator phase labels never leak in."""

        self.phase(phase_id)
        return {}

    def to_payload(self) -> dict[str, Any]:
        unsigned = {
            "format": CONTINUAL_EVALUATION_FORMAT,
            "version": CONTINUAL_EVALUATION_VERSION,
            "type": "course_manifest",
            "manifest_revision": self.manifest_revision,
            "course_id": self.course_id,
            "phases": [phase.to_payload() for phase in self.phases],
        }
        return {**unsigned, "manifest_digest": _content_digest(unsigned)}

    @property
    def content_digest(self) -> str:
        return str(self.to_payload()["manifest_digest"])

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ContinualCourseManifest:
        _verify_envelope(payload, CONTINUAL_EVALUATION_FORMAT)
        if payload.get("type") != "course_manifest":
            raise ContinualEvaluationContractError("payload is not a course_manifest")
        _verify_digest(payload, "manifest_digest")
        return cls(
            course_id=str(payload["course_id"]),
            phases=tuple(CoursePhase.from_payload(item) for item in payload["phases"]),
            manifest_revision=int(payload.get("manifest_revision", 1)),
        )


@dataclass(frozen=True)
class MetricObservation:
    """One absolute measurement plus explicitly declared relative deltas."""

    metric_name: str
    domain_id: str
    checkpoint_digest: str
    absolute_value: float
    owner_id: str
    read_only_input_digest: str
    parent_delta: float | None = None
    parent_checkpoint_digest: str | None = None
    comparison_delta: float | None = None
    comparison_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "metric_name",
            "domain_id",
            "checkpoint_digest",
            "owner_id",
            "read_only_input_digest",
        ):
            _require_text(getattr(self, field_name), f"observation {field_name}")
        _require_finite(self.absolute_value, "absolute_value")
        if self.parent_delta is not None:
            _require_finite(self.parent_delta, "parent_delta")
            if not self.parent_checkpoint_digest:
                raise ContinualEvaluationContractError(
                    "parent_delta requires parent_checkpoint_digest"
                )
        if self.parent_checkpoint_digest is not None:
            _require_text(self.parent_checkpoint_digest, "parent_checkpoint_digest")
        if self.comparison_delta is not None:
            _require_finite(self.comparison_delta, "comparison_delta")
            if not self.comparison_id:
                raise ContinualEvaluationContractError(
                    "comparison_delta requires comparison_id"
                )
        if self.comparison_id is not None:
            _require_text(self.comparison_id, "comparison_id")

    def to_payload(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "domain_id": self.domain_id,
            "checkpoint_digest": self.checkpoint_digest,
            "absolute_value": self.absolute_value,
            "owner_id": self.owner_id,
            "read_only_input_digest": self.read_only_input_digest,
            "parent_delta": self.parent_delta,
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "comparison_delta": self.comparison_delta,
            "comparison_id": self.comparison_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MetricObservation:
        return cls(
            metric_name=str(payload["metric_name"]),
            domain_id=str(payload["domain_id"]),
            checkpoint_digest=str(payload["checkpoint_digest"]),
            absolute_value=float(payload["absolute_value"]),
            owner_id=str(payload["owner_id"]),
            read_only_input_digest=str(payload["read_only_input_digest"]),
            parent_delta=(
                None if payload.get("parent_delta") is None else float(payload["parent_delta"])
            ),
            parent_checkpoint_digest=(
                None
                if payload.get("parent_checkpoint_digest") is None
                else str(payload["parent_checkpoint_digest"])
            ),
            comparison_delta=(
                None
                if payload.get("comparison_delta") is None
                else float(payload["comparison_delta"])
            ),
            comparison_id=(
                None if payload.get("comparison_id") is None else str(payload["comparison_id"])
            ),
        )


@dataclass(frozen=True)
class ContinualEvaluationSnapshot:
    """All capability measurements captured at one checkpoint."""

    checkpoint_digest: str
    phase_id: str
    owner_graph_digest: str
    read_only_input_digest: str
    observations: tuple[MetricObservation, ...]
    parent_checkpoint_digest: str | None = None
    resource_cost: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "checkpoint_digest",
            "phase_id",
            "owner_graph_digest",
            "read_only_input_digest",
        ):
            _require_text(getattr(self, field_name), f"snapshot {field_name}")
        if not self.observations:
            raise ContinualEvaluationContractError(
                "evaluation snapshot must contain at least one observation"
            )
        if self.parent_checkpoint_digest is not None:
            _require_text(self.parent_checkpoint_digest, "snapshot parent_checkpoint_digest")
        _require_finite(self.resource_cost, "resource_cost")
        if self.resource_cost <= 0:
            raise ContinualEvaluationContractError("resource_cost must be positive")
        keys = [(item.metric_name, item.domain_id) for item in self.observations]
        if len(set(keys)) != len(keys):
            raise ContinualEvaluationContractError(
                "snapshot cannot contain duplicate metric/domain observations"
            )
        for item in self.observations:
            if item.checkpoint_digest != self.checkpoint_digest:
                raise ContinualEvaluationContractError(
                    "observation checkpoint_digest does not match snapshot"
                )
            if item.parent_delta is not None and self.parent_checkpoint_digest is None:
                raise ContinualEvaluationContractError(
                    "parent_delta requires snapshot parent_checkpoint_digest"
                )
            if (
                item.parent_checkpoint_digest is not None
                and self.parent_checkpoint_digest is not None
                and item.parent_checkpoint_digest != self.parent_checkpoint_digest
            ):
                raise ContinualEvaluationContractError(
                    "observation parent checkpoint does not match snapshot"
                )

    def to_payload(self) -> dict[str, Any]:
        unsigned = {
            "format": CONTINUAL_EVALUATION_FORMAT,
            "version": CONTINUAL_EVALUATION_VERSION,
            "type": "evaluation_snapshot",
            "checkpoint_digest": self.checkpoint_digest,
            "phase_id": self.phase_id,
            "owner_graph_digest": self.owner_graph_digest,
            "read_only_input_digest": self.read_only_input_digest,
            "observations": [item.to_payload() for item in self.observations],
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "resource_cost": self.resource_cost,
        }
        return {**unsigned, "snapshot_digest": _content_digest(unsigned)}

    @property
    def content_digest(self) -> str:
        return str(self.to_payload()["snapshot_digest"])

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ContinualEvaluationSnapshot:
        _verify_envelope(payload, CONTINUAL_EVALUATION_FORMAT)
        if payload.get("type") != "evaluation_snapshot":
            raise ContinualEvaluationContractError("payload is not an evaluation_snapshot")
        _verify_digest(payload, "snapshot_digest")
        return cls(
            checkpoint_digest=str(payload["checkpoint_digest"]),
            phase_id=str(payload["phase_id"]),
            owner_graph_digest=str(payload["owner_graph_digest"]),
            read_only_input_digest=str(payload["read_only_input_digest"]),
            observations=tuple(
                MetricObservation.from_payload(item) for item in payload["observations"]
            ),
            parent_checkpoint_digest=(
                None
                if payload.get("parent_checkpoint_digest") is None
                else str(payload["parent_checkpoint_digest"])
            ),
            resource_cost=float(payload.get("resource_cost", 1.0)),
        )


@dataclass(frozen=True)
class ContinualScorecard:
    """Aggregate contract that refuses ambiguous retention claims."""

    metric_specs: tuple[MetricSpec, ...]
    snapshots: tuple[ContinualEvaluationSnapshot, ...]
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        if not self.metric_specs:
            raise ContinualEvaluationContractError("scorecard needs metric specs")
        if not self.snapshots:
            raise ContinualEvaluationContractError("scorecard needs snapshots")
        _require_finite(self.epsilon, "epsilon")
        if self.epsilon < 0:
            raise ContinualEvaluationContractError("epsilon cannot be negative")
        spec_names = [spec.name for spec in self.metric_specs]
        if len(set(spec_names)) != len(spec_names):
            raise ContinualEvaluationContractError("metric spec names must be unique")
        checkpoint_ids = [snapshot.checkpoint_digest for snapshot in self.snapshots]
        if len(set(checkpoint_ids)) != len(checkpoint_ids):
            raise ContinualEvaluationContractError(
                "scorecard snapshots must use unique checkpoint digests"
            )
        specs = {spec.name: spec for spec in self.metric_specs}
        for snapshot in self.snapshots:
            for observation in snapshot.observations:
                spec = specs.get(observation.metric_name)
                if spec is None:
                    raise ContinualEvaluationContractError(
                        f"missing metric spec: {observation.metric_name}"
                    )
                if spec.domain_id != "*" and spec.domain_id != observation.domain_id:
                    raise ContinualEvaluationContractError(
                        f"metric {spec.name} is not declared for domain {observation.domain_id}"
                    )

    def metric_spec(self, metric_name: str) -> MetricSpec:
        for spec in self.metric_specs:
            if spec.name == metric_name:
                return spec
        raise ContinualEvaluationContractError(f"unknown metric: {metric_name}")

    def _observations(
        self,
        metric_name: str,
        domain_id: str | None = None,
    ) -> tuple[MetricObservation, ...]:
        observations = tuple(
            observation
            for snapshot in self.snapshots
            for observation in snapshot.observations
            if observation.metric_name == metric_name
            and (domain_id is None or observation.domain_id == domain_id)
        )
        if not observations:
            raise ContinualEvaluationContractError(
                f"no observations for metric={metric_name!r}, domain={domain_id!r}"
            )
        if domain_id is None:
            domains = {observation.domain_id for observation in observations}
            if len(domains) > 1:
                raise CrossDomainAggregationError(
                    f"metric {metric_name!r} spans domains {sorted(domains)}; choose one domain"
                )
        return observations

    @staticmethod
    def _gain(spec: MetricSpec, raw_delta: float) -> float:
        return raw_delta if spec.direction == "higher" else -raw_delta

    def parent_retention(
        self,
        metric_name: str,
        domain_id: str | None = None,
    ) -> dict[str, Any]:
        """Return parent-relative retention, failing closed on absent baselines."""

        spec = self.metric_spec(metric_name)
        observations = self._observations(metric_name, domain_id)
        valid = tuple(item for item in observations if item.parent_delta is not None)
        missing_count = len(observations) - len(valid)
        parent_deltas = tuple(
            item.parent_delta for item in valid if item.parent_delta is not None
        )
        gains = tuple(self._gain(spec, float(delta)) for delta in parent_deltas)
        forgetting = tuple(max(0.0, -gain) for gain in gains)
        non_inferiority = (
            missing_count == 0 and all(gain >= -self.epsilon for gain in gains)
        )
        catastrophic = False
        if spec.catastrophic_forgetting_threshold is not None:
            catastrophic = any(
                value > spec.catastrophic_forgetting_threshold for value in forgetting
            )
        return {
            "metric_name": metric_name,
            "domain_id": domain_id or observations[0].domain_id,
            "sample_count": len(observations),
            "valid_parent_delta_count": len(valid),
            "missing_parent_baseline_count": missing_count,
            "average_parent_gain": (sum(gains) / len(gains)) if gains else None,
            "average_forgetting": (sum(forgetting) / len(forgetting)) if forgetting else None,
            "worst_forgetting": max(forgetting) if forgetting else None,
            "exact_zero_non_degradation": (
                missing_count == 0 and all(gain >= 0.0 for gain in gains)
            ),
            "non_inferiority": non_inferiority,
            "catastrophic_forgetting": catastrophic,
            "epsilon": self.epsilon,
        }

    def comparison_transfer(
        self,
        metric_name: str,
        domain_id: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate declared arm-vs-arm deltas, never parent-retention deltas."""

        spec = self.metric_spec(metric_name)
        observations = self._observations(metric_name, domain_id)
        valid = tuple(item for item in observations if item.comparison_delta is not None)
        missing_count = len(observations) - len(valid)
        comparison_deltas = tuple(
            item.comparison_delta for item in valid if item.comparison_delta is not None
        )
        gains = tuple(self._gain(spec, float(delta)) for delta in comparison_deltas)
        return {
            "metric_name": metric_name,
            "domain_id": domain_id or observations[0].domain_id,
            "sample_count": len(observations),
            "valid_comparison_delta_count": len(valid),
            "missing_comparison_baseline_count": missing_count,
            "average_comparison_gain": (sum(gains) / len(gains)) if gains else None,
            "non_inferiority": (
                missing_count == 0 and all(gain >= -self.epsilon for gain in gains)
            ),
            "epsilon": self.epsilon,
        }

    def resource_normalized_parent_gain(
        self,
        metric_name: str,
        domain_id: str | None = None,
    ) -> float | None:
        spec = self.metric_spec(metric_name)
        observations = self._observations(metric_name, domain_id)
        snapshot_costs = {
            snapshot.checkpoint_digest: snapshot.resource_cost for snapshot in self.snapshots
        }
        weighted_gain = 0.0
        total_cost = 0.0
        for observation in observations:
            if observation.parent_delta is None:
                continue
            cost = snapshot_costs[observation.checkpoint_digest]
            weighted_gain += self._gain(spec, float(observation.parent_delta))
            total_cost += cost
        if total_cost == 0.0:
            return None
        return weighted_gain / total_cost

    def can_promote(self) -> bool:
        """Require declared retention/comparison baselines and no critical loss."""

        evaluated = False
        for spec in self.metric_specs:
            domains = {
                observation.domain_id
                for snapshot in self.snapshots
                for observation in snapshot.observations
                if observation.metric_name == spec.name
            }
            if spec.baseline_kind == "absolute":
                continue
            evaluated = True
            for domain_id in domains:
                result = (
                    self.comparison_transfer(spec.name, domain_id)
                    if spec.baseline_kind == "comparison"
                    else self.parent_retention(spec.name, domain_id)
                )
                if not result["non_inferiority"]:
                    return False
                if result.get("catastrophic_forgetting"):
                    return False
        return evaluated

    def to_payload(self) -> dict[str, Any]:
        unsigned = {
            "format": CONTINUAL_EVALUATION_FORMAT,
            "version": CONTINUAL_EVALUATION_VERSION,
            "type": "scorecard",
            "epsilon": self.epsilon,
            "metric_specs": [spec.to_payload() for spec in self.metric_specs],
            "snapshots": [snapshot.to_payload() for snapshot in self.snapshots],
        }
        return {**unsigned, "scorecard_digest": _content_digest(unsigned)}

    @property
    def content_digest(self) -> str:
        return str(self.to_payload()["scorecard_digest"])

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ContinualScorecard:
        _verify_envelope(payload, CONTINUAL_EVALUATION_FORMAT)
        if payload.get("type") != "scorecard":
            raise ContinualEvaluationContractError("payload is not a scorecard")
        _verify_digest(payload, "scorecard_digest")
        return cls(
            metric_specs=tuple(MetricSpec.from_payload(item) for item in payload["metric_specs"]),
            snapshots=tuple(
                ContinualEvaluationSnapshot.from_payload(item)
                for item in payload["snapshots"]
            ),
            epsilon=float(payload.get("epsilon", 0.0)),
        )
