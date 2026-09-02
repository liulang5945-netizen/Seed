"""Auditable data and objective contract for native episodic training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .internalization import content_digest

MEMORY_OBJECTIVE_FORMAT = "taiji-native-memory-objective-v1"
MEMORY_OBJECTIVE_VERSION = 1
MEMORY_OBJECTIVE_COMPONENTS = (
    "cue_identity",
    "action",
    "outcome",
    "reward",
    "time",
    "episode",
    "provenance",
)
MEMORY_OBJECTIVE_CREDIT_AXES = ("association", "action_readout", "outcome_readout")


@dataclass(frozen=True)
class EpisodicObjectiveContract:
    """Define what an episodic example may teach and what it may not inspect."""

    source_partitions: tuple[str, ...] = ("phase_a_train", "phase_b_train", "replay_train")
    protected_partition: str = "phase_b_train"
    prohibited_partitions: tuple[str, ...] = (
        "phase_a_holdout",
        "phase_a_retention",
        "phase_b_holdout",
        "phase_b_retention",
    )
    positive_binding: str = "cue_identity_to_event"
    negative_competition: str = "cross_cue_event_exclusion"
    credit_axes: tuple[str, ...] = MEMORY_OBJECTIVE_CREDIT_AXES
    replay_provenance: str = "replayed"
    default_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.source_partitions:
            raise ValueError("episodic objective needs at least one source partition")
        if self.protected_partition not in self.source_partitions:
            raise ValueError("protected partition must be an objective source partition")
        if set(self.source_partitions) & set(self.prohibited_partitions):
            raise ValueError("objective source and prohibited partitions must be disjoint")
        if not self.credit_axes:
            raise ValueError("episodic objective needs at least one credit axis")
        unknown = set(self.credit_axes) - set(MEMORY_OBJECTIVE_CREDIT_AXES)
        if unknown:
            raise ValueError(f"unsupported episodic objective credit axes: {sorted(unknown)}")
        if not self.replay_provenance.strip():
            raise ValueError("episodic objective replay provenance cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MEMORY_OBJECTIVE_FORMAT,
            "version": MEMORY_OBJECTIVE_VERSION,
            "components": list(MEMORY_OBJECTIVE_COMPONENTS),
            "source_partitions": list(self.source_partitions),
            "protected_partition": self.protected_partition,
            "prohibited_partitions": list(self.prohibited_partitions),
            "positive_binding": self.positive_binding,
            "negative_competition": self.negative_competition,
            "credit_axes": list(self.credit_axes),
            "replay_provenance": self.replay_provenance,
            "default_enabled": self.default_enabled,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EpisodicObjectiveContract:
        if payload.get("format") != MEMORY_OBJECTIVE_FORMAT:
            raise ValueError("unsupported episodic objective contract format")
        if int(payload.get("version", -1)) != MEMORY_OBJECTIVE_VERSION:
            raise ValueError("unsupported episodic objective contract version")
        components = tuple(str(item) for item in payload.get("components", ()))
        if components != MEMORY_OBJECTIVE_COMPONENTS:
            raise ValueError("episodic objective components do not match the contract")
        return cls(
            source_partitions=tuple(str(item) for item in payload["source_partitions"]),
            protected_partition=str(payload["protected_partition"]),
            prohibited_partitions=tuple(
                str(item) for item in payload["prohibited_partitions"]
            ),
            positive_binding=str(payload["positive_binding"]),
            negative_competition=str(payload["negative_competition"]),
            credit_axes=tuple(str(item) for item in payload["credit_axes"]),
            replay_provenance=str(payload["replay_provenance"]),
            default_enabled=bool(payload.get("default_enabled", False)),
        )
