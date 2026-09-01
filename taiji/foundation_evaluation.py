"""Contract-first evaluation for Taiji's five foundation abilities.

This module deliberately does not train a model or call a provider.  It owns
the shared vocabulary for the M0 benchmark so that task runners can be added
without creating five incompatible notions of "passed".
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FOUNDATION_MANIFEST_FORMAT = "taiji-foundation-baseline-v1"
FOUNDATION_EVALUATION_FORMAT = "taiji-foundation-evaluation-v1"
FOUNDATION_REQUIRED_ABILITIES = (
    "b1_sequence_prediction",
    "b2_delayed_memory",
    "b3_world_transition",
    "b4_goal_action",
    "b5_continual_learning",
)
FOUNDATION_PARTITIONS = ("train", "holdout", "retention", "control")
FOUNDATION_BASELINE_KINDS = (
    "random",
    "frozen_parent",
    "simple_rule",
    "hash_only",
)
FOUNDATION_SAMPLE_FIELDS = (
    "sample_id",
    "source",
    "license",
    "objective",
    "partition",
    "payload",
    "target_or_outcome",
    "provenance",
    "taint",
    "content_digest",
)
FOUNDATION_REPORT_FIELDS = (
    "manifest_digest",
    "checkpoint_gate_status",
    "status",
    "can_promote",
    "failure_reasons",
    "measurements",
)
FOUNDATION_MEASUREMENT_FIELDS = (
    "ability_id",
    "status",
    "primary_metric",
    "metric_direction",
    "metric_value",
    "baseline_metrics",
    "sample_counts",
    "holdout_updates",
    "evidence",
)
_MEASUREMENT_STATUSES = frozenset({"passed", "failed", "not_evaluated"})
_METRIC_DIRECTIONS = frozenset({"higher_is_better", "lower_is_better"})


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _float(value: Any, field_name: str, *, required: bool) -> float | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be an array")
    return tuple(_text(item, f"{field_name}[]") for item in value)


@dataclass(frozen=True)
class FoundationTaskSpec:
    """One immutable task definition loaded from the versioned manifest."""

    ability_id: str
    name: str
    input_kind: str
    primary_metric: str
    metric_direction: str
    minimum_train_units: int
    minimum_holdout_units: int
    minimum_retention_units: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FoundationTaskSpec:
        data = _mapping(payload, "task")
        direction = _text(data.get("metric_direction"), "task.metric_direction")
        if direction not in _METRIC_DIRECTIONS:
            raise ValueError("task.metric_direction is unsupported")
        return cls(
            ability_id=_text(data.get("ability_id"), "task.ability_id"),
            name=_text(data.get("name"), "task.name"),
            input_kind=_text(data.get("input_kind"), "task.input_kind"),
            primary_metric=_text(data.get("primary_metric"), "task.primary_metric"),
            metric_direction=direction,
            minimum_train_units=_positive_int(
                data.get("minimum_train_units"), "task.minimum_train_units"
            ),
            minimum_holdout_units=_positive_int(
                data.get("minimum_holdout_units"), "task.minimum_holdout_units"
            ),
            minimum_retention_units=_positive_int(
                data.get("minimum_retention_units"), "task.minimum_retention_units"
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "ability_id": self.ability_id,
            "name": self.name,
            "input_kind": self.input_kind,
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
            "minimum_train_units": self.minimum_train_units,
            "minimum_holdout_units": self.minimum_holdout_units,
            "minimum_retention_units": self.minimum_retention_units,
        }


@dataclass(frozen=True)
class FoundationManifest:
    """Versioned data/evaluation contract shared by all M0 task runners."""

    format: str
    version: int
    split_policy: str
    partitions: tuple[str, ...]
    seeds: tuple[int, ...]
    baselines: tuple[str, ...]
    checkpoint_gate: str
    tasks: tuple[FoundationTaskSpec, ...]
    data_contract: Mapping[str, Any]
    baseline_protocol: Mapping[str, Any]
    report_schema: Mapping[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FoundationManifest:
        data = _mapping(payload, "manifest")
        manifest_format = _text(data.get("format"), "manifest.format")
        if manifest_format != FOUNDATION_MANIFEST_FORMAT:
            raise ValueError(f"unsupported manifest format: {manifest_format}")
        version = _positive_int(data.get("version"), "manifest.version")
        partitions = _string_tuple(data.get("partitions"), "manifest.partitions")
        if partitions != FOUNDATION_PARTITIONS:
            raise ValueError(
                "manifest.partitions must be exactly " + ", ".join(FOUNDATION_PARTITIONS)
            )
        seeds_value = data.get("seeds")
        if not isinstance(seeds_value, Sequence) or isinstance(
            seeds_value, (str, bytes, bytearray)
        ):
            raise ValueError("manifest.seeds must be an array")
        seeds = tuple(_positive_int(seed, "manifest.seeds[]") for seed in seeds_value)
        if len(seeds) < 3 or len(set(seeds)) != len(seeds):
            raise ValueError("manifest.seeds must contain at least three unique seeds")
        baselines = _string_tuple(data.get("baselines"), "manifest.baselines")
        if not set(FOUNDATION_BASELINE_KINDS).issubset(baselines):
            raise ValueError("manifest.baselines must include all required controls")
        data_contract = _mapping(data.get("data_contract"), "manifest.data_contract")
        sample_fields = _string_tuple(
            data_contract.get("required_fields"), "manifest.data_contract.required_fields"
        )
        if sample_fields != FOUNDATION_SAMPLE_FIELDS:
            raise ValueError("manifest.data_contract.required_fields do not match the contract")
        forbidden_sources = _string_tuple(
            data_contract.get("forbidden_sources"),
            "manifest.data_contract.forbidden_sources",
        )
        if not forbidden_sources:
            raise ValueError("manifest.data_contract.forbidden_sources cannot be empty")
        _text(data_contract.get("partition_rule"), "manifest.data_contract.partition_rule")
        _text(data_contract.get("holdout_rule"), "manifest.data_contract.holdout_rule")
        baseline_protocol = _mapping(
            data.get("baseline_protocol"), "manifest.baseline_protocol"
        )
        protocol_controls = _string_tuple(
            baseline_protocol.get("required_controls"),
            "manifest.baseline_protocol.required_controls",
        )
        if protocol_controls != FOUNDATION_BASELINE_KINDS:
            raise ValueError("manifest.baseline_protocol.required_controls do not match controls")
        for field_name in (
            "seeds_are_shared",
            "report_mean_std_worst_seed",
            "strictly_beats_strongest_control",
        ):
            if baseline_protocol.get(field_name) is not True:
                raise ValueError(f"manifest.baseline_protocol.{field_name} must be true")
        report_schema = _mapping(data.get("report_schema"), "manifest.report_schema")
        report_fields = _string_tuple(
            report_schema.get("required_fields"), "manifest.report_schema.required_fields"
        )
        measurement_fields = _string_tuple(
            report_schema.get("measurement_required_fields"),
            "manifest.report_schema.measurement_required_fields",
        )
        if report_fields != FOUNDATION_REPORT_FIELDS:
            raise ValueError("manifest.report_schema.required_fields do not match the contract")
        if measurement_fields != FOUNDATION_MEASUREMENT_FIELDS:
            raise ValueError(
                "manifest.report_schema.measurement_required_fields do not match the contract"
            )
        task_values = data.get("tasks")
        if not isinstance(task_values, Sequence) or isinstance(
            task_values, (str, bytes, bytearray)
        ):
            raise ValueError("manifest.tasks must be an array")
        tasks = tuple(FoundationTaskSpec.from_payload(item) for item in task_values)
        task_ids = tuple(task.ability_id for task in tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("manifest.tasks contains duplicate ability ids")
        if set(task_ids) != set(FOUNDATION_REQUIRED_ABILITIES):
            raise ValueError(
                "manifest.tasks abilities must be exactly "
                + ", ".join(FOUNDATION_REQUIRED_ABILITIES)
            )
        return cls(
            format=manifest_format,
            version=version,
            split_policy=_text(data.get("split_policy"), "manifest.split_policy"),
            partitions=partitions,
            seeds=seeds,
            baselines=baselines,
            checkpoint_gate=_text(data.get("checkpoint_gate"), "manifest.checkpoint_gate"),
            tasks=tasks,
            data_contract=dict(data_contract),
            baseline_protocol=dict(baseline_protocol),
            report_schema=dict(report_schema),
        )

    @classmethod
    def load(cls, path: str | Path) -> FoundationManifest:
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls.from_payload(payload)

    @property
    def digest(self) -> str:
        return _canonical_digest(self.to_payload())

    def task(self, ability_id: str) -> FoundationTaskSpec:
        for task in self.tasks:
            if task.ability_id == ability_id:
                return task
        raise KeyError(ability_id)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "split_policy": self.split_policy,
            "partitions": list(self.partitions),
            "seeds": list(self.seeds),
            "baselines": list(self.baselines),
            "checkpoint_gate": self.checkpoint_gate,
            "data_contract": dict(self.data_contract),
            "baseline_protocol": dict(self.baseline_protocol),
            "tasks": [task.to_payload() for task in self.tasks],
            "report_schema": dict(self.report_schema),
        }


@dataclass(frozen=True)
class FoundationMeasurement:
    """One task runner's measured result; no result is inferred by this DTO."""

    ability_id: str
    status: str
    primary_metric: str
    metric_direction: str
    metric_value: float | None
    baseline_metrics: Mapping[str, float]
    sample_counts: Mapping[str, int]
    holdout_updates: int
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.ability_id.strip():
            raise ValueError("measurement.ability_id must be non-empty")
        if self.status not in _MEASUREMENT_STATUSES:
            raise ValueError("measurement.status is unsupported")
        if not self.primary_metric.strip():
            raise ValueError("measurement.primary_metric must be non-empty")
        if self.metric_direction not in _METRIC_DIRECTIONS:
            raise ValueError("measurement.metric_direction is unsupported")
        _float(self.metric_value, "measurement.metric_value", required=self.status == "passed")
        for name, value in self.baseline_metrics.items():
            _float(value, f"measurement.baseline_metrics.{name}", required=True)
        for partition in FOUNDATION_PARTITIONS:
            if partition in self.sample_counts:
                _non_negative_int(
                    self.sample_counts[partition], f"measurement.sample_counts.{partition}"
                )
        _non_negative_int(self.holdout_updates, "measurement.holdout_updates")
        if any(not isinstance(item, str) or not item.strip() for item in self.evidence):
            raise ValueError("measurement.evidence must contain non-empty strings")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FoundationMeasurement:
        data = _mapping(payload, "measurement")
        baseline_values = _mapping(data.get("baseline_metrics", {}), "measurement.baseline_metrics")
        sample_values = _mapping(data.get("sample_counts", {}), "measurement.sample_counts")
        return cls(
            ability_id=_text(data.get("ability_id"), "measurement.ability_id"),
            status=_text(data.get("status"), "measurement.status"),
            primary_metric=_text(data.get("primary_metric"), "measurement.primary_metric"),
            metric_direction=_text(
                data.get("metric_direction"), "measurement.metric_direction"
            ),
            metric_value=_float(
                data.get("metric_value"),
                "measurement.metric_value",
                required=False,
            ),
            baseline_metrics={
                _text(key, "measurement.baseline_metrics.key"): _float(
                    value,
                    f"measurement.baseline_metrics.{key}",
                    required=True,
                )
                for key, value in baseline_values.items()
            },
            sample_counts={
                _text(key, "measurement.sample_counts.key"): _non_negative_int(
                    value, f"measurement.sample_counts.{key}"
                )
                for key, value in sample_values.items()
            },
            holdout_updates=_non_negative_int(
                data.get("holdout_updates", 0), "measurement.holdout_updates"
            ),
            evidence=_string_tuple(data.get("evidence", ()), "measurement.evidence"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "ability_id": self.ability_id,
            "status": self.status,
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
            "metric_value": self.metric_value,
            "baseline_metrics": dict(sorted(self.baseline_metrics.items())),
            "sample_counts": dict(sorted(self.sample_counts.items())),
            "holdout_updates": self.holdout_updates,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class FoundationEvaluation:
    """Deterministic aggregation and promotion gate for M0 measurements."""

    manifest_digest: str
    checkpoint_gate_status: str
    status: str
    can_promote: bool
    failure_reasons: tuple[str, ...]
    measurements: tuple[FoundationMeasurement, ...]

    @classmethod
    def evaluate(
        cls,
        manifest: FoundationManifest,
        measurements: Mapping[str, FoundationMeasurement],
        *,
        checkpoint_gate_status: str,
    ) -> FoundationEvaluation:
        required_ids = set(FOUNDATION_REQUIRED_ABILITIES)
        measured_ids = set(measurements)
        missing = sorted(required_ids - measured_ids)
        unknown = sorted(measured_ids - required_ids)
        if missing:
            raise ValueError("missing foundation measurements: " + ", ".join(missing))
        if unknown:
            raise ValueError("unknown foundation measurements: " + ", ".join(unknown))

        reasons: list[str] = []
        ordered = tuple(measurements[ability_id] for ability_id in FOUNDATION_REQUIRED_ABILITIES)
        for measurement in ordered:
            task = manifest.task(measurement.ability_id)
            if measurement.primary_metric != task.primary_metric:
                reasons.append(f"{measurement.ability_id}:metric_mismatch")
            if measurement.metric_direction != task.metric_direction:
                reasons.append(f"{measurement.ability_id}:metric_direction_mismatch")
            if measurement.holdout_updates > 0:
                reasons.append("holdout_side_effect")
            if measurement.status == "failed":
                reasons.append(f"{measurement.ability_id}:measurement_failed")
            elif measurement.status == "not_evaluated":
                reasons.append(f"{measurement.ability_id}:not_evaluated")
            else:
                for partition, minimum in (
                    ("train", task.minimum_train_units),
                    ("holdout", task.minimum_holdout_units),
                    ("retention", task.minimum_retention_units),
                ):
                    if measurement.sample_counts.get(partition, 0) < minimum:
                        reasons.append(f"{measurement.ability_id}:insufficient_{partition}")
                if not _beats_all_baselines(measurement):
                    reasons.append(f"{measurement.ability_id}:baseline_not_beaten")

        if checkpoint_gate_status != "passed":
            reasons.append("checkpoint_gate_not_passed")

        if any(reason.endswith(":not_evaluated") for reason in reasons):
            status = "not_evaluated"
        elif reasons and any(
            reason != "checkpoint_gate_not_passed"
            and not reason.endswith(":not_evaluated")
            for reason in reasons
        ):
            status = "failed"
        else:
            status = "passed"
        return cls(
            manifest_digest=manifest.digest,
            checkpoint_gate_status=checkpoint_gate_status,
            status=status,
            can_promote=status == "passed" and not reasons,
            failure_reasons=tuple(dict.fromkeys(reasons)),
            measurements=ordered,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": FOUNDATION_EVALUATION_FORMAT,
            "manifest_digest": self.manifest_digest,
            "checkpoint_gate_status": self.checkpoint_gate_status,
            "status": self.status,
            "can_promote": self.can_promote,
            "failure_reasons": list(self.failure_reasons),
            "measurements": [measurement.to_payload() for measurement in self.measurements],
        }


def _beats_all_baselines(measurement: FoundationMeasurement) -> bool:
    if measurement.status != "passed" or measurement.metric_value is None:
        return False
    if not all(kind in measurement.baseline_metrics for kind in FOUNDATION_BASELINE_KINDS):
        return False
    baseline_values = [measurement.baseline_metrics[kind] for kind in FOUNDATION_BASELINE_KINDS]
    if measurement.metric_direction == "higher_is_better":
        return measurement.metric_value > max(baseline_values)
    return measurement.metric_value < min(baseline_values)
