"""Bridge admitted online interaction evidence to structural pressure only.

This module deliberately stops before candidate creation.  It converts
already-admitted native Outcome feedback into a sealed structural evidence
projection, while requiring independent holdout and retention observations.
The existing structural controller remains the only owner allowed to turn that
projection into a topology candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .interaction_group_online import (
    InteractionGroupOnlineAdmission,
    InteractionGroupOutcomeFeedback,
)
from .structural_evidence import StructuralEvidenceLedger
from .structural_growth import StructuralRuntimeObservation
from .structural_pressure import (
    StructuralGrowthEvidenceProjection,
    project_structural_growth_pressure,
)

INTERACTION_STRUCTURAL_BRIDGE_FORMAT = "taiji-interaction-structural-bridge-v1"
INTERACTION_STRUCTURAL_BRIDGE_REVISION = 1


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=repr,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: str, name: str) -> str:
    value = str(value)
    if not value:
        raise ValueError(f"{name} cannot be empty")
    return value


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return value


@dataclass(frozen=True)
class InteractionStructuralBridgeConfig:
    """Content-addressed thresholds for online-to-structure projection."""

    network_id: str
    region_id: str
    minimum_feedbacks: int = 2
    minimum_holdout_transfer: float = 0.5
    resource_normalizer: float = 2.0
    version: int = 1

    def __post_init__(self) -> None:
        _text(self.network_id, "interaction structural network_id")
        _text(self.region_id, "interaction structural region_id")
        if int(self.minimum_feedbacks) <= 0:
            raise ValueError("interaction structural minimum_feedbacks must be positive")
        _unit(self.minimum_holdout_transfer, "interaction structural holdout threshold")
        if not math.isfinite(float(self.resource_normalizer)) or float(
            self.resource_normalizer
        ) <= 0.0:
            raise ValueError("interaction structural resource_normalizer must be positive")
        if int(self.version) != 1:
            raise ValueError(f"unsupported interaction structural config version: {self.version}")

    @property
    def digest(self) -> str:
        return _digest(self.to_payload(include_digest=False))

    def to_payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": INTERACTION_STRUCTURAL_BRIDGE_FORMAT,
            "revision": INTERACTION_STRUCTURAL_BRIDGE_REVISION,
            "version": self.version,
            "network_id": self.network_id,
            "region_id": self.region_id,
            "minimum_feedbacks": self.minimum_feedbacks,
            "minimum_holdout_transfer": self.minimum_holdout_transfer,
            "resource_normalizer": self.resource_normalizer,
        }
        if include_digest:
            payload["config_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> InteractionStructuralBridgeConfig:
        if payload.get("format") != INTERACTION_STRUCTURAL_BRIDGE_FORMAT:
            raise ValueError("unsupported interaction structural config format")
        if int(payload.get("revision", -1)) != INTERACTION_STRUCTURAL_BRIDGE_REVISION:
            raise ValueError("unsupported interaction structural bridge revision")
        config = cls(
            version=int(payload.get("version", 1)),
            network_id=str(payload["network_id"]),
            region_id=str(payload["region_id"]),
            minimum_feedbacks=int(payload["minimum_feedbacks"]),
            minimum_holdout_transfer=float(payload["minimum_holdout_transfer"]),
            resource_normalizer=float(payload["resource_normalizer"]),
        )
        if str(payload.get("config_digest")) != config.digest:
            raise ValueError("interaction structural config digest mismatch")
        return config


@dataclass(frozen=True)
class InteractionStructuralPressure:
    """A sealed structural projection bound to admitted online feedback."""

    projection: StructuralGrowthEvidenceProjection
    feedback_ids: tuple[str, ...]
    outcome_ids: tuple[str, ...]
    mean_interaction: float
    bridge_digest: str
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.projection, StructuralGrowthEvidenceProjection):
            raise TypeError("interaction structural pressure requires a pressure projection")
        feedback_ids = tuple(str(item) for item in self.feedback_ids)
        outcome_ids = tuple(str(item) for item in self.outcome_ids)
        if len(feedback_ids) == 0 or len(set(feedback_ids)) != len(feedback_ids):
            raise ValueError("interaction structural feedback_ids must be unique and non-empty")
        if len(outcome_ids) != len(feedback_ids) or len(set(outcome_ids)) != len(outcome_ids):
            raise ValueError("interaction structural outcome_ids must align and be unique")
        if not math.isfinite(float(self.mean_interaction)):
            raise ValueError("interaction structural mean_interaction must be finite")
        _text(self.bridge_digest, "interaction structural bridge_digest")
        if int(self.version) != 1:
            raise ValueError(f"unsupported interaction structural pressure version: {self.version}")
        object.__setattr__(self, "feedback_ids", feedback_ids)
        object.__setattr__(self, "outcome_ids", outcome_ids)
        object.__setattr__(self, "mean_interaction", float(self.mean_interaction))
        expected = _digest(self._payload_without_digest())
        if self.bridge_digest != expected:
            raise ValueError("interaction structural bridge digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": INTERACTION_STRUCTURAL_BRIDGE_FORMAT,
            "revision": INTERACTION_STRUCTURAL_BRIDGE_REVISION,
            "version": self.version,
            "projection": self.projection.to_payload(),
            "feedback_ids": list(self.feedback_ids),
            "outcome_ids": list(self.outcome_ids),
            "mean_interaction": self.mean_interaction,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "bridge_digest": self.bridge_digest}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> InteractionStructuralPressure:
        if payload.get("format") != INTERACTION_STRUCTURAL_BRIDGE_FORMAT:
            raise ValueError("unsupported interaction structural pressure format")
        if int(payload.get("revision", -1)) != INTERACTION_STRUCTURAL_BRIDGE_REVISION:
            raise ValueError("unsupported interaction structural bridge revision")
        return cls(
            version=int(payload.get("version", 1)),
            projection=StructuralGrowthEvidenceProjection.from_payload(payload["projection"]),
            feedback_ids=tuple(str(item) for item in payload.get("feedback_ids", ())),
            outcome_ids=tuple(str(item) for item in payload.get("outcome_ids", ())),
            mean_interaction=float(payload["mean_interaction"]),
            bridge_digest=str(payload["bridge_digest"]),
        )


class InteractionStructuralBridge:
    """Produce structural pressure from admitted online interaction evidence."""

    def __init__(self, config: InteractionStructuralBridgeConfig) -> None:
        if not isinstance(config, InteractionStructuralBridgeConfig):
            raise TypeError("interaction structural bridge requires its config")
        self.config = config

    def project(
        self,
        feedbacks: Sequence[InteractionGroupOutcomeFeedback],
        admissions: Sequence[InteractionGroupOnlineAdmission],
        independent_observations: Sequence[StructuralRuntimeObservation],
    ) -> InteractionStructuralPressure:
        if not feedbacks:
            raise ValueError("interaction structural bridge requires online feedbacks")
        if len(feedbacks) < int(self.config.minimum_feedbacks):
            raise ValueError("interaction structural bridge lacks repeated online feedback")
        if any(not isinstance(item, InteractionGroupOutcomeFeedback) for item in feedbacks):
            raise TypeError("interaction structural bridge feedbacks must be online feedback values")
        if any(not isinstance(item, InteractionGroupOnlineAdmission) for item in admissions):
            raise TypeError("interaction structural bridge admissions must be online admissions")
        feedback_ids = tuple(item.feedback_id for item in feedbacks)
        if len(set(feedback_ids)) != len(feedback_ids):
            raise ValueError("interaction structural bridge feedbacks must be unique")
        admission_by_feedback = {item.feedback_id: item for item in admissions}
        if any(item.feedback_id not in admission_by_feedback for item in feedbacks):
            raise ValueError("interaction structural bridge feedback admission is missing")
        if any(
            admission_by_feedback[item.feedback_id].status != "applied"
            for item in feedbacks
        ):
            raise ValueError("interaction structural bridge accepts applied feedback only")
        if len({item.member_ids for item in feedbacks}) < 2:
            raise ValueError("interaction structural bridge requires distinct interaction contexts")
        if any(item.source_split != "online" for item in feedbacks):
            raise ValueError("interaction structural bridge rejects non-online feedback")
        if any(not item.outcome.terminal or item.outcome.success is not True for item in feedbacks):
            raise ValueError("interaction structural bridge accepts successful terminal Outcomes only")
        independent = tuple(independent_observations)
        if not independent:
            raise ValueError("interaction structural bridge requires independent observations")
        if any(not isinstance(item, StructuralRuntimeObservation) for item in independent):
            raise TypeError("interaction structural observations must be runtime observations")
        if any(
            item.network_id != self.config.network_id or item.region_id != self.config.region_id
            for item in independent
        ):
            raise ValueError("interaction structural observations cross network or region")
        if any(item.partition not in {"holdout", "retention"} for item in independent):
            raise ValueError("interaction structural observations must be holdout or retention")

        online_observations = tuple(
            StructuralRuntimeObservation(
                network_id=self.config.network_id,
                region_id=self.config.region_id,
                tick=index,
                usage=min(
                    1.0,
                    max(0.0, float(item.resource_cost) / self.config.resource_normalizer),
                ),
                resource_pressure=min(
                    1.0, max(0.0, float(item.resource_cost) / self.config.resource_normalizer)
                ),
                prediction_error=min(
                    1.0, max(0.0, 1.0 - float(item.realized_interaction))
                ),
                learning_gain=min(1.0, max(0.0, float(item.realized_interaction))),
                holdout_transfer=0.0,
                evidence_id=f"online-feedback:{item.feedback_id}",
                task_slice_id=f"online-interaction:{item.candidate_id}",
                partition="train",
            )
            for index, item in enumerate(feedbacks, start=1)
        )
        normalized_independent = tuple(
            replace(item, tick=len(online_observations) + index)
            for index, item in enumerate(independent, start=1)
        )
        ledger = StructuralEvidenceLedger(window_capacity=1)
        for observation in (*online_observations, *normalized_independent):
            ledger.append(observation)
        projection = project_structural_growth_pressure(
            ledger.sealed_summaries,
            minimum_train_task_slices=int(self.config.minimum_feedbacks),
            minimum_train_windows=int(self.config.minimum_feedbacks),
            require_holdout=True,
            require_retention=True,
        )
        if projection.mean_holdout_transfer is None or (
            projection.mean_holdout_transfer < self.config.minimum_holdout_transfer
        ):
            raise ValueError("interaction structural holdout transfer is below bridge threshold")
        online_evidence_ids = {
            f"online-feedback:{item.feedback_id}" for item in feedbacks
        }
        if not online_evidence_ids.issubset(set(projection.evidence_ids)):
            raise ValueError("interaction structural projection lost online evidence binding")
        payload = {
            "format": INTERACTION_STRUCTURAL_BRIDGE_FORMAT,
            "revision": INTERACTION_STRUCTURAL_BRIDGE_REVISION,
            "version": 1,
            "projection": projection.to_payload(),
            "feedback_ids": list(feedback_ids),
            "outcome_ids": [item.outcome_id for item in feedbacks],
            "mean_interaction": sum(item.realized_interaction for item in feedbacks)
            / len(feedbacks),
        }
        return InteractionStructuralPressure(
            projection=projection,
            feedback_ids=feedback_ids,
            outcome_ids=tuple(item.outcome_id for item in feedbacks),
            mean_interaction=float(payload["mean_interaction"]),
            bridge_digest=_digest(payload),
        )


__all__ = [
    "INTERACTION_STRUCTURAL_BRIDGE_FORMAT",
    "INTERACTION_STRUCTURAL_BRIDGE_REVISION",
    "InteractionStructuralBridge",
    "InteractionStructuralBridgeConfig",
    "InteractionStructuralPressure",
]
