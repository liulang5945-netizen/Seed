"""Deterministic measurement owner for replay-backed structural validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

STRUCTURAL_VALIDATION_MEASUREMENT_FORMAT = (
    "taiji-workbench-structural-validation-measurement-v1"
)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "type": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "bytes": bytes(tensor.view(torch.uint8).reshape(-1).tolist()).hex(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_tensor_sequence(value: Any, name: str) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = tuple(value)
    else:
        raise TypeError(f"{name} must be a tensor or tensor sequence")
    if not values or any(not isinstance(item, torch.Tensor) for item in values):
        raise TypeError(f"{name} must contain at least one tensor")
    return tuple(item.detach().float() for item in values)


def _mean_absolute_error(
    observed: Any,
    target: Any,
    *,
    name: str,
) -> float:
    observed_values = _as_tensor_sequence(observed, f"{name}.observed")
    target_values = _as_tensor_sequence(target, f"{name}.target")
    if len(observed_values) != len(target_values):
        raise ValueError(f"{name} observed and target sequence lengths differ")
    errors: list[float] = []
    for observed_tensor, target_tensor in zip(observed_values, target_values):
        if observed_tensor.shape != target_tensor.shape:
            raise ValueError(f"{name} observed and target tensor shapes differ")
        errors.append(float(torch.mean(torch.abs(observed_tensor - target_tensor)).item()))
    return sum(errors) / len(errors)


def _bounded_gain(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _resource_pressure(resource_measurement: Any) -> float:
    if isinstance(resource_measurement, Mapping):
        value = resource_measurement.get("pressure")
    else:
        value = getattr(resource_measurement, "pressure", None)
    if value is None:
        raise ValueError("resource measurement must expose pressure")
    pressure = float(value)
    if not 0.0 <= pressure <= 1.0:
        raise ValueError("resource measurement pressure must be in [0, 1]")
    return pressure


@dataclass(frozen=True)
class StructuralValidationMeasurements:
    """Measured validation metrics and digests produced by replay probes."""

    holdout_gain: float
    retention_regression: float
    lesion_effect: float
    resource_state: float
    resource_cost: int
    holdout_baseline_digest: str
    holdout_candidate_digest: str
    holdout_target_digest: str
    retention_baseline_digest: str
    retention_candidate_digest: str
    retention_target_digest: str
    lesion_full_digest: str
    lesion_lesioned_digest: str
    lesion_target_digest: str
    resource_measurement_digest: str
    measurement_digest: str

    def __post_init__(self) -> None:
        for name in (
            "holdout_gain",
            "retention_regression",
            "lesion_effect",
            "resource_state",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if int(self.resource_cost) <= 0:
            raise ValueError("resource_cost must be positive")
        for name in (
            "holdout_baseline_digest",
            "holdout_candidate_digest",
            "holdout_target_digest",
            "retention_baseline_digest",
            "retention_candidate_digest",
            "retention_target_digest",
            "lesion_full_digest",
            "lesion_lesioned_digest",
            "lesion_target_digest",
            "resource_measurement_digest",
            "measurement_digest",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, str(getattr(self, name)))
        for name in (
            "holdout_gain",
            "retention_regression",
            "lesion_effect",
            "resource_state",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "resource_cost", int(self.resource_cost))
        if self.measurement_digest != self._expected_measurement_digest():
            raise ValueError("structural validation measurement digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "holdout_gain": self.holdout_gain,
            "retention_regression": self.retention_regression,
            "lesion_effect": self.lesion_effect,
            "resource_state": self.resource_state,
            "resource_cost": self.resource_cost,
            "holdout_baseline_digest": self.holdout_baseline_digest,
            "holdout_candidate_digest": self.holdout_candidate_digest,
            "holdout_target_digest": self.holdout_target_digest,
            "retention_baseline_digest": self.retention_baseline_digest,
            "retention_candidate_digest": self.retention_candidate_digest,
            "retention_target_digest": self.retention_target_digest,
            "lesion_full_digest": self.lesion_full_digest,
            "lesion_lesioned_digest": self.lesion_lesioned_digest,
            "lesion_target_digest": self.lesion_target_digest,
            "resource_measurement_digest": self.resource_measurement_digest,
        }

    def _expected_measurement_digest(self) -> str:
        return _digest(
            {
                "format": STRUCTURAL_VALIDATION_MEASUREMENT_FORMAT,
                **self._payload_without_digest(),
            }
        )

    @classmethod
    def from_replay_probes(
        cls,
        *,
        holdout_baseline_outputs: Any,
        holdout_candidate_outputs: Any,
        holdout_target_outputs: Any,
        retention_baseline_outputs: Any,
        retention_candidate_outputs: Any,
        retention_target_outputs: Any,
        lesion_full_outputs: Any,
        lesion_lesioned_outputs: Any,
        lesion_target_outputs: Any,
        resource_measurement: Any,
        resource_cost: int,
    ) -> StructuralValidationMeasurements:
        holdout_baseline_error = _mean_absolute_error(
            holdout_baseline_outputs,
            holdout_target_outputs,
            name="holdout",
        )
        holdout_candidate_error = _mean_absolute_error(
            holdout_candidate_outputs,
            holdout_target_outputs,
            name="holdout",
        )
        retention_baseline_error = _mean_absolute_error(
            retention_baseline_outputs,
            retention_target_outputs,
            name="retention",
        )
        retention_candidate_error = _mean_absolute_error(
            retention_candidate_outputs,
            retention_target_outputs,
            name="retention",
        )
        lesion_full_error = _mean_absolute_error(
            lesion_full_outputs,
            lesion_target_outputs,
            name="lesion",
        )
        lesion_lesioned_error = _mean_absolute_error(
            lesion_lesioned_outputs,
            lesion_target_outputs,
            name="lesion",
        )
        pressure = _resource_pressure(resource_measurement)
        values = {
            "holdout_gain": _bounded_gain(holdout_baseline_error - holdout_candidate_error),
            "retention_regression": _bounded_gain(
                retention_candidate_error - retention_baseline_error
            ),
            "lesion_effect": _bounded_gain(lesion_lesioned_error - lesion_full_error),
            "resource_state": _bounded_gain(1.0 - pressure),
            "resource_cost": int(resource_cost),
            "holdout_baseline_digest": _digest(holdout_baseline_outputs),
            "holdout_candidate_digest": _digest(holdout_candidate_outputs),
            "holdout_target_digest": _digest(holdout_target_outputs),
            "retention_baseline_digest": _digest(retention_baseline_outputs),
            "retention_candidate_digest": _digest(retention_candidate_outputs),
            "retention_target_digest": _digest(retention_target_outputs),
            "lesion_full_digest": _digest(lesion_full_outputs),
            "lesion_lesioned_digest": _digest(lesion_lesioned_outputs),
            "lesion_target_digest": _digest(lesion_target_outputs),
            "resource_measurement_digest": _digest(resource_measurement),
        }
        values["measurement_digest"] = _digest(
            {
                "format": STRUCTURAL_VALIDATION_MEASUREMENT_FORMAT,
                **values,
            }
        )
        return cls(**values)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": STRUCTURAL_VALIDATION_MEASUREMENT_FORMAT,
            **self._payload_without_digest(),
            "measurement_digest": self.measurement_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StructuralValidationMeasurements:
        if payload.get("format") != STRUCTURAL_VALIDATION_MEASUREMENT_FORMAT:
            raise ValueError("unsupported structural validation measurement format")
        values = {
            "holdout_gain": float(payload["holdout_gain"]),
            "retention_regression": float(payload["retention_regression"]),
            "lesion_effect": float(payload["lesion_effect"]),
            "resource_state": float(payload["resource_state"]),
            "resource_cost": int(payload["resource_cost"]),
            "holdout_baseline_digest": str(payload["holdout_baseline_digest"]),
            "holdout_candidate_digest": str(payload["holdout_candidate_digest"]),
            "holdout_target_digest": str(payload["holdout_target_digest"]),
            "retention_baseline_digest": str(payload["retention_baseline_digest"]),
            "retention_candidate_digest": str(payload["retention_candidate_digest"]),
            "retention_target_digest": str(payload["retention_target_digest"]),
            "lesion_full_digest": str(payload["lesion_full_digest"]),
            "lesion_lesioned_digest": str(payload["lesion_lesioned_digest"]),
            "lesion_target_digest": str(payload["lesion_target_digest"]),
            "resource_measurement_digest": str(payload["resource_measurement_digest"]),
            "measurement_digest": str(payload["measurement_digest"]),
        }
        expected_digest = _digest(
            {
                "format": STRUCTURAL_VALIDATION_MEASUREMENT_FORMAT,
                **{key: value for key, value in values.items() if key != "measurement_digest"},
            }
        )
        if values["measurement_digest"] != expected_digest:
            raise ValueError("structural validation measurement digest mismatch")
        return cls(
            **values,
        )
