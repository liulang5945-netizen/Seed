"""Taiji-owned brain/client credit selection.

The selector decides whether a runtime experience should be answered by
learning inside Taiji or by asking the Seed client for a new capability.  It
never installs, connects or executes anything: the only artefact it can emit
is one content-addressed candidate.  This module is intentionally free of
Seed, Workbench and provider imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .evolution_experience import EvolutionExperience
from .internalization import content_digest

EVOLUTION_CREDIT_FORMAT = "taiji-evolution-credit-v1"
EVOLUTION_CREDIT_ATTRIBUTION_FORMAT = "taiji-evolution-credit-attribution-v1"
EVOLUTION_CREDIT_VERSION = 1

EVOLUTION_CREDIT_CANDIDATE_KINDS = (
    "weight_update",
    "memory_consolidation",
    "route_update",
    "structure_candidate",
    "client_capability_candidate",
    "clarify_or_stop",
)

EVOLUTION_CREDIT_ARMS = (
    "brain_only",
    "client_plugin_only",
    "unattributed",
)

_ARM_BY_CANDIDATE_KIND = {
    "weight_update": "brain_only",
    "memory_consolidation": "brain_only",
    "route_update": "brain_only",
    "structure_candidate": "brain_only",
    "client_capability_candidate": "client_plugin_only",
    "clarify_or_stop": "unattributed",
}

_LANGUAGE_SOURCE_KINDS = ("provider",)


@dataclass(frozen=True)
class BrainClientCreditDecision:
    """Exactly one candidate, with the evidence that produced it."""

    experience_id: str
    experience_digest: str
    candidate_kind: str
    reasons: tuple[str, ...]
    decision_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "experience_id", str(self.experience_id).strip())
        if not self.experience_id:
            raise ValueError("credit decision experience_id must not be empty")
        object.__setattr__(
            self, "experience_digest", str(self.experience_digest).strip()
        )
        if not self.experience_digest:
            raise ValueError("credit decision experience_digest must not be empty")
        if self.candidate_kind not in EVOLUTION_CREDIT_CANDIDATE_KINDS:
            raise ValueError("unsupported evolution credit candidate_kind")
        object.__setattr__(self, "reasons", tuple(str(item) for item in self.reasons))
        if not self.reasons or any(not item for item in self.reasons):
            raise ValueError("credit decision reasons must not be empty")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("credit decision reasons must be unique")
        if not str(self.decision_digest):
            raise ValueError("credit decision decision_digest must not be empty")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": EVOLUTION_CREDIT_FORMAT,
            "version": EVOLUTION_CREDIT_VERSION,
            "experience_id": self.experience_id,
            "experience_digest": self.experience_digest,
            "candidate_kind": self.candidate_kind,
            "reasons": list(self.reasons),
            "decision_digest": self.decision_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> BrainClientCreditDecision:
        if payload.get("format") != EVOLUTION_CREDIT_FORMAT:
            raise ValueError("unsupported evolution credit decision format")
        identity = {
            key: value for key, value in payload.items() if key != "decision_digest"
        }
        expected = content_digest(identity)
        if str(payload.get("decision_digest", "")) != expected:
            raise ValueError("evolution credit decision digest mismatch")
        return cls(
            experience_id=str(payload["experience_id"]),
            experience_digest=str(payload["experience_digest"]),
            candidate_kind=str(payload["candidate_kind"]),
            reasons=tuple(str(item) for item in payload.get("reasons", ())),
            decision_digest=expected,
        )


class BrainClientCreditSelector:
    """Route one experience to a single learning or capability candidate."""

    def __init__(self, *, registered_capability_ids: tuple[str, ...] = ()) -> None:
        ids = tuple(str(item).strip() for item in registered_capability_ids)
        if any(not item for item in ids):
            raise ValueError("registered_capability_ids must not contain empty values")
        self.registered_capability_ids = tuple(sorted(set(ids)))

    def select(
        self,
        experience: EvolutionExperience,
        *,
        growth_permitted: bool = False,
        resources_exhausted: bool = False,
    ) -> BrainClientCreditDecision:
        if not isinstance(experience, EvolutionExperience):
            raise TypeError("credit selector accepts EvolutionExperience values")

        candidate_kind, reasons = self._classify(
            experience,
            growth_permitted=bool(growth_permitted),
            resources_exhausted=bool(resources_exhausted),
        )
        identity = {
            "format": EVOLUTION_CREDIT_FORMAT,
            "version": EVOLUTION_CREDIT_VERSION,
            "experience_id": experience.experience_id,
            "experience_digest": experience.experience_digest,
            "candidate_kind": candidate_kind,
            "reasons": list(reasons),
        }
        return BrainClientCreditDecision(
            experience_id=experience.experience_id,
            experience_digest=experience.experience_digest,
            candidate_kind=candidate_kind,
            reasons=tuple(reasons),
            decision_digest=content_digest(identity),
        )

    def _classify(
        self,
        experience: EvolutionExperience,
        *,
        growth_permitted: bool,
        resources_exhausted: bool,
    ) -> tuple[str, tuple[str, ...]]:
        if resources_exhausted:
            return "clarify_or_stop", ("resource_budget_exhausted",)

        capability_id = str(experience.capability_id).strip()
        if not capability_id:
            return "clarify_or_stop", ("experience_identifies_no_capability",)
        if capability_id not in self.registered_capability_ids:
            return "client_capability_candidate", ("capability_not_registered",)

        registered = ("capability_already_registered",)
        if not experience.success:
            if experience.source_kind in _LANGUAGE_SOURCE_KINDS:
                return "clarify_or_stop", registered + (
                    "language_failure_is_not_structural_evidence",
                )
            return "route_update", registered + (
                "registered_capability_failed_on_an_existing_route",
            )
        if growth_permitted:
            return "structure_candidate", registered + (
                "capacity_gate_permitted_a_structural_proposal",
            )
        if experience.partition == "retention":
            return "memory_consolidation", registered + (
                "retention_partition_consolidates_existing_memory",
            )
        return "weight_update", registered + (
            "registered_capability_can_learn_from_this_outcome",
        )


@dataclass(frozen=True)
class BrainClientAblationAttribution:
    """Partition one experience set across the brain and client-plugin arms."""

    brain_only: tuple[str, ...]
    client_plugin_only: tuple[str, ...]
    unattributed: tuple[str, ...]
    attribution_digest: str

    def __post_init__(self) -> None:
        for arm in EVOLUTION_CREDIT_ARMS:
            values = tuple(str(item).strip() for item in getattr(self, arm))
            if any(not item for item in values):
                raise ValueError(f"attribution {arm} must not contain empty values")
            object.__setattr__(self, arm, values)
        booked = self.brain_only + self.client_plugin_only + self.unattributed
        if len(set(booked)) != len(booked):
            raise ValueError("attribution must not book a duplicate experience_id")
        if not str(self.attribution_digest):
            raise ValueError("attribution attribution_digest must not be empty")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": EVOLUTION_CREDIT_ATTRIBUTION_FORMAT,
            "version": EVOLUTION_CREDIT_VERSION,
        }
        for arm in EVOLUTION_CREDIT_ARMS:
            payload[arm] = list(getattr(self, arm))
        payload["attribution_digest"] = self.attribution_digest
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> BrainClientAblationAttribution:
        if payload.get("format") != EVOLUTION_CREDIT_ATTRIBUTION_FORMAT:
            raise ValueError("unsupported evolution credit attribution format")
        identity = {
            key: value for key, value in payload.items() if key != "attribution_digest"
        }
        expected = content_digest(identity)
        if str(payload.get("attribution_digest", "")) != expected:
            raise ValueError("evolution credit attribution digest mismatch")
        return cls(
            brain_only=tuple(str(item) for item in payload.get("brain_only", ())),
            client_plugin_only=tuple(
                str(item) for item in payload.get("client_plugin_only", ())
            ),
            unattributed=tuple(str(item) for item in payload.get("unattributed", ())),
            attribution_digest=expected,
        )


def attribute_brain_client_ablation(
    decisions: Iterable[BrainClientCreditDecision],
) -> BrainClientAblationAttribution:
    """Book every decision into exactly one ablation arm, preserving input order."""

    arms: dict[str, list[str]] = {arm: [] for arm in EVOLUTION_CREDIT_ARMS}
    seen: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, BrainClientCreditDecision):
            raise TypeError(
                "ablation attribution accepts BrainClientCreditDecision values"
            )
        if decision.experience_id in seen:
            raise ValueError(
                "ablation attribution rejects a duplicate experience_id: "
                f"{decision.experience_id}"
            )
        seen.add(decision.experience_id)
        arms[_ARM_BY_CANDIDATE_KIND[decision.candidate_kind]].append(
            decision.experience_id
        )

    identity: dict[str, Any] = {
        "format": EVOLUTION_CREDIT_ATTRIBUTION_FORMAT,
        "version": EVOLUTION_CREDIT_VERSION,
    }
    for arm in EVOLUTION_CREDIT_ARMS:
        identity[arm] = list(arms[arm])
    return BrainClientAblationAttribution(
        brain_only=tuple(arms["brain_only"]),
        client_plugin_only=tuple(arms["client_plugin_only"]),
        unattributed=tuple(arms["unattributed"]),
        attribution_digest=content_digest(identity),
    )


__all__ = [
    "EVOLUTION_CREDIT_ARMS",
    "EVOLUTION_CREDIT_ATTRIBUTION_FORMAT",
    "EVOLUTION_CREDIT_CANDIDATE_KINDS",
    "EVOLUTION_CREDIT_FORMAT",
    "EVOLUTION_CREDIT_VERSION",
    "BrainClientAblationAttribution",
    "BrainClientCreditDecision",
    "BrainClientCreditSelector",
    "attribute_brain_client_ablation",
]
