"""Content-addressed validation facts produced from real Workbench replays."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

STRUCTURAL_VALIDATION_ARTIFACT_FORMAT = "taiji-workbench-structural-validation-artifact-v1"


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


def _digest_value(value: Any) -> str:
    encoded = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class WorkbenchStructuralValidationArtifact:
    """Immutable validation facts tied to one real Workbench candidate replay."""

    candidate_id: str
    network_id: str
    region_id: str
    task_slice_id: str
    outcome_digests: tuple[str, ...]
    parent_checkpoint_digest: str
    trial_checkpoint_digest: str
    holdout_input_digest: str
    holdout_output_digest: str
    retention_baseline_digest: str
    retention_candidate_digest: str
    lesion_baseline_digest: str
    lesion_candidate_digest: str
    resource_measurement_digest: str
    holdout_gain: float
    retention_regression: float
    lesion_effect: float
    resource_state: float
    resource_cost: int
    evidence_ids: tuple[str, ...]
    artifact_digest: str
    measurement_digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "network_id",
            "region_id",
            "task_slice_id",
            "parent_checkpoint_digest",
            "trial_checkpoint_digest",
            "holdout_input_digest",
            "holdout_output_digest",
            "retention_baseline_digest",
            "retention_candidate_digest",
            "lesion_baseline_digest",
            "lesion_candidate_digest",
            "resource_measurement_digest",
            "artifact_digest",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"Workbench validation artifact {name} must not be empty")
        for name in (
            "holdout_gain",
            "retention_regression",
            "lesion_effect",
            "resource_state",
        ):
            _unit(getattr(self, name), f"Workbench validation artifact {name}")
        if int(self.resource_cost) <= 0:
            raise ValueError("Workbench validation artifact resource_cost must be positive")
        outcome_digests = tuple(str(item) for item in self.outcome_digests)
        evidence_ids = tuple(str(item) for item in self.evidence_ids)
        if not outcome_digests or any(not item for item in outcome_digests):
            raise ValueError("Workbench validation artifact outcome_digests must not be empty")
        if len(set(outcome_digests)) != len(outcome_digests):
            raise ValueError("Workbench validation artifact outcome_digests must be unique")
        if not evidence_ids or any(not item for item in evidence_ids):
            raise ValueError("Workbench validation artifact evidence_ids must not be empty")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Workbench validation artifact evidence_ids must be unique")
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "network_id", str(self.network_id))
        object.__setattr__(self, "region_id", str(self.region_id))
        object.__setattr__(self, "task_slice_id", str(self.task_slice_id))
        object.__setattr__(self, "outcome_digests", outcome_digests)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        for name in (
            "parent_checkpoint_digest",
            "trial_checkpoint_digest",
            "holdout_input_digest",
            "holdout_output_digest",
            "retention_baseline_digest",
            "retention_candidate_digest",
            "lesion_baseline_digest",
            "lesion_candidate_digest",
            "resource_measurement_digest",
            "artifact_digest",
        ):
            object.__setattr__(self, name, str(getattr(self, name)))
        for name in (
            "holdout_gain",
            "retention_regression",
            "lesion_effect",
            "resource_state",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "resource_cost", int(self.resource_cost))
        object.__setattr__(self, "measurement_digest", str(self.measurement_digest))
        if self.artifact_digest != _digest_value(self._payload_without_digest()):
            raise ValueError("Workbench validation artifact digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        payload = {
            "format": STRUCTURAL_VALIDATION_ARTIFACT_FORMAT,
            "candidate_id": self.candidate_id,
            "network_id": self.network_id,
            "region_id": self.region_id,
            "task_slice_id": self.task_slice_id,
            "outcome_digests": list(self.outcome_digests),
            "parent_checkpoint_digest": self.parent_checkpoint_digest,
            "trial_checkpoint_digest": self.trial_checkpoint_digest,
            "holdout_input_digest": self.holdout_input_digest,
            "holdout_output_digest": self.holdout_output_digest,
            "retention_baseline_digest": self.retention_baseline_digest,
            "retention_candidate_digest": self.retention_candidate_digest,
            "lesion_baseline_digest": self.lesion_baseline_digest,
            "lesion_candidate_digest": self.lesion_candidate_digest,
            "resource_measurement_digest": self.resource_measurement_digest,
            "holdout_gain": self.holdout_gain,
            "retention_regression": self.retention_regression,
            "lesion_effect": self.lesion_effect,
            "resource_state": self.resource_state,
            "resource_cost": self.resource_cost,
            "evidence_ids": list(self.evidence_ids),
        }
        if self.measurement_digest:
            payload["measurement_digest"] = self.measurement_digest
        return payload

    @classmethod
    def from_measurements(
        cls,
        *,
        candidate_id: str,
        network_id: str,
        region_id: str,
        task_slice_id: str,
        outcome_digests: tuple[str, ...],
        parent_checkpoint_digest: str,
        trial_checkpoint_digest: str,
        holdout_inputs: Any,
        holdout_outputs: Any,
        retention_baseline: Any,
        retention_candidate: Any,
        lesion_baseline: Any,
        lesion_candidate: Any,
        resource_measurement: Any,
        holdout_gain: float,
        retention_regression: float,
        lesion_effect: float,
        resource_state: float,
        resource_cost: int,
        evidence_ids: tuple[str, ...],
        measurement_digest: str = "",
    ) -> WorkbenchStructuralValidationArtifact:
        values = {
            "candidate_id": str(candidate_id),
            "network_id": str(network_id),
            "region_id": str(region_id),
            "task_slice_id": str(task_slice_id),
            "outcome_digests": tuple(str(item) for item in outcome_digests),
            "parent_checkpoint_digest": str(parent_checkpoint_digest),
            "trial_checkpoint_digest": str(trial_checkpoint_digest),
            "holdout_input_digest": _digest_value(holdout_inputs),
            "holdout_output_digest": _digest_value(holdout_outputs),
            "retention_baseline_digest": _digest_value(retention_baseline),
            "retention_candidate_digest": _digest_value(retention_candidate),
            "lesion_baseline_digest": _digest_value(lesion_baseline),
            "lesion_candidate_digest": _digest_value(lesion_candidate),
            "resource_measurement_digest": _digest_value(resource_measurement),
            "holdout_gain": float(holdout_gain),
            "retention_regression": float(retention_regression),
            "lesion_effect": float(lesion_effect),
            "resource_state": float(resource_state),
            "resource_cost": int(resource_cost),
            "evidence_ids": tuple(str(item) for item in evidence_ids),
        }
        if measurement_digest:
            values["measurement_digest"] = str(measurement_digest)
        values["artifact_digest"] = _digest_value(
            {
                **values,
                "outcome_digests": list(values["outcome_digests"]),
                "evidence_ids": list(values["evidence_ids"]),
                "format": STRUCTURAL_VALIDATION_ARTIFACT_FORMAT,
            }
        )
        return cls(**values)

    def matches_holdout_replay(self, holdout_inputs: Any, holdout_outputs: Any) -> bool:
        """Verify that the replay payload is the payload bound by this artifact."""

        return (
            self.holdout_input_digest == _digest_value(holdout_inputs)
            and self.holdout_output_digest == _digest_value(holdout_outputs)
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "artifact_digest": self.artifact_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> WorkbenchStructuralValidationArtifact:
        if payload.get("format") != STRUCTURAL_VALIDATION_ARTIFACT_FORMAT:
            raise ValueError("unsupported Workbench validation artifact format")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            network_id=str(payload["network_id"]),
            region_id=str(payload["region_id"]),
            task_slice_id=str(payload["task_slice_id"]),
            outcome_digests=tuple(str(item) for item in payload.get("outcome_digests", ())),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            trial_checkpoint_digest=str(payload["trial_checkpoint_digest"]),
            holdout_input_digest=str(payload["holdout_input_digest"]),
            holdout_output_digest=str(payload["holdout_output_digest"]),
            retention_baseline_digest=str(payload["retention_baseline_digest"]),
            retention_candidate_digest=str(payload["retention_candidate_digest"]),
            lesion_baseline_digest=str(payload["lesion_baseline_digest"]),
            lesion_candidate_digest=str(payload["lesion_candidate_digest"]),
            resource_measurement_digest=str(payload["resource_measurement_digest"]),
            holdout_gain=float(payload["holdout_gain"]),
            retention_regression=float(payload["retention_regression"]),
            lesion_effect=float(payload["lesion_effect"]),
            resource_state=float(payload["resource_state"]),
            resource_cost=int(payload["resource_cost"]),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", ())),
            artifact_digest=str(payload["artifact_digest"]),
            measurement_digest=str(payload.get("measurement_digest", "")),
        )
