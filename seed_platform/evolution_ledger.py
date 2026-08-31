"""Seed-owned append-only ledger for Taiji corpus and runtime experience.

The platform owns collection, redaction, provenance and checkpoint integrity.
Taiji receives only the DTOs from :mod:`taiji.evolution_experience`; the
ledger never mutates a learner or executes a capability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from taiji.evolution_experience import (
    EVOLUTION_PARTITIONS,
    EVOLUTION_REDACTION_REVISION,
    REDACTION_PLACEHOLDER,
    EvolutionCorpusArtifact,
    EvolutionExperience,
)
from taiji.internalization import content_digest

EVOLUTION_LEDGER_CHECKPOINT_FORMAT = "seed-evolution-ledger-checkpoint-v1"
EVOLUTION_LEDGER_VERSION = 1


def redact_sensitive_payload(value: Any) -> tuple[Any, tuple[str, ...]]:
    """Redact scalar values under common credential keys before admission."""

    sensitive_keys = {
        "access_key",
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "password",
        "private_key",
        "secret",
        "token",
    }
    flags: set[str] = set()

    def visit(item: Any, path: str) -> Any:
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text.strip().lower() in sensitive_keys and not isinstance(
                    child, (Mapping, list, tuple)
                ):
                    flags.add(child_path)
                    result[key_text] = REDACTION_PLACEHOLDER
                else:
                    result[key_text] = visit(child, child_path)
            return {key: result[key] for key in sorted(result)}
        if isinstance(item, (list, tuple)):
            return [visit(child, f"{path}[{index}]") for index, child in enumerate(item)]
        return item

    return visit(value, ""), tuple(sorted(flags))


@dataclass(frozen=True)
class EvolutionAppendResult:
    experience: EvolutionExperience
    accepted: bool
    duplicate: bool
    reason: str = ""


class EvolutionExperienceLedger:
    """Content-addressed corpus store plus a chained runtime event log."""

    def __init__(self) -> None:
        self._corpus: dict[str, EvolutionCorpusArtifact] = {}
        self._experiences: dict[str, EvolutionExperience] = {}
        self._experience_by_digest: dict[str, str] = {}
        self.revision = 0
        self._tail_event_digest = content_digest(
            {"format": EVOLUTION_LEDGER_CHECKPOINT_FORMAT, "genesis": True}
        )

    @property
    def tail_event_digest(self) -> str:
        return self._tail_event_digest

    @property
    def corpus(self) -> tuple[EvolutionCorpusArtifact, ...]:
        return tuple(self._corpus[key] for key in sorted(self._corpus))

    @property
    def experiences(self) -> tuple[EvolutionExperience, ...]:
        return tuple(
            self._experiences[key]
            for key in sorted(self._experiences, key=lambda item: self._experiences[item].event_sequence)
        )

    def add_corpus(self, artifact: EvolutionCorpusArtifact) -> EvolutionCorpusArtifact:
        if not isinstance(artifact, EvolutionCorpusArtifact):
            raise TypeError("evolution ledger corpus entry must be EvolutionCorpusArtifact")
        if artifact.status == "admitted":
            raise ValueError("admitted corpus must enter through admit_corpus")
        existing = self._corpus.get(artifact.artifact_digest)
        if existing is not None:
            if existing.to_payload() != artifact.to_payload():
                raise ValueError("corpus artifact digest collision")
            return existing
        self._corpus[artifact.artifact_digest] = artifact
        self.revision += 1
        return artifact

    def admit_corpus(self, artifact_digest: str, *, admission_revision: str) -> EvolutionCorpusArtifact:
        digest = str(artifact_digest).strip()
        if digest not in self._corpus:
            raise KeyError(f"unknown corpus artifact: {artifact_digest}")
        revision = str(admission_revision).strip()
        if not revision:
            raise ValueError("corpus admission_revision cannot be empty")
        current = self._corpus[digest]
        if current.status == "admitted":
            if current.admission_revision == revision:
                return current
            raise ValueError("corpus artifact is already admitted with another revision")
        if current.status != "candidate":
            raise ValueError("only candidate corpus artifacts can be admitted")
        admitted = current.with_status("admitted", admission_revision=revision)
        self._corpus[digest] = admitted
        self.revision += 1
        return admitted

    def append(self, experience: EvolutionExperience) -> EvolutionAppendResult:
        if not isinstance(experience, EvolutionExperience):
            raise TypeError("evolution ledger entry must be EvolutionExperience")
        existing = self._experiences.get(experience.experience_id)
        if existing is not None:
            if existing.experience_digest != experience.experience_digest:
                raise ValueError("experience_id content conflict")
            return EvolutionAppendResult(existing, accepted=False, duplicate=True, reason="idempotent")
        prior_id = self._experience_by_digest.get(experience.experience_digest)
        if prior_id is not None:
            prior = self._experiences[prior_id]
            raise ValueError(
                "experience content already has another identity: "
                f"{prior.experience_id} vs {experience.experience_id}"
            )
        expected_sequence = len(self._experiences) + 1
        if experience.event_sequence not in (0, expected_sequence):
            raise ValueError("experience event sequence is not contiguous")
        if experience.event_sequence == 0:
            bound = experience.bind_to_chain(expected_sequence, self._tail_event_digest)
        else:
            if experience.previous_event_digest != self._tail_event_digest:
                raise ValueError("experience previous_event_digest does not match ledger tail")
            bound = experience
        if bound.event_sequence != expected_sequence or bound.previous_event_digest != self._tail_event_digest:
            raise ValueError("experience chain binding failed")
        self._experiences[bound.experience_id] = bound
        self._experience_by_digest[bound.experience_digest] = bound.experience_id
        self._tail_event_digest = bound.event_digest
        self.revision += 1
        return EvolutionAppendResult(bound, accepted=True, duplicate=False)

    def records(self, *, partition: str | None = None) -> tuple[EvolutionExperience, ...]:
        if partition is not None and str(partition) not in EVOLUTION_PARTITIONS:
            raise ValueError(f"unsupported evolution partition: {partition}")
        return tuple(
            item
            for item in self.experiences
            if partition is None or item.partition == str(partition)
        )

    def training_view(self) -> tuple[tuple[EvolutionCorpusArtifact, ...], tuple[EvolutionExperience, ...]]:
        """Return only admitted train corpus and train experiences."""

        return (
            tuple(
                item
                for item in self.corpus
                if item.status == "admitted" and item.partition == "train"
            ),
            self.records(partition="train"),
        )

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": EVOLUTION_LEDGER_CHECKPOINT_FORMAT,
            "version": EVOLUTION_LEDGER_VERSION,
            "revision": self.revision,
            "tail_event_digest": self._tail_event_digest,
            "corpus": [item.to_payload() for item in self.corpus],
            "experiences": [item.to_payload() for item in self.experiences],
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> EvolutionExperienceLedger:
        if payload.get("format") != EVOLUTION_LEDGER_CHECKPOINT_FORMAT:
            raise ValueError("unsupported evolution ledger checkpoint format")
        if int(payload.get("version", -1)) != EVOLUTION_LEDGER_VERSION:
            raise ValueError("unsupported evolution ledger checkpoint version")
        expected_checkpoint_digest = content_digest(
            {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        )
        if str(payload.get("checkpoint_digest", "")) != expected_checkpoint_digest:
            raise ValueError("evolution ledger checkpoint digest mismatch")
        ledger = cls()
        raw_corpus = payload.get("corpus", ())
        raw_experiences = payload.get("experiences", ())
        if isinstance(raw_corpus, (str, bytes)) or not isinstance(raw_corpus, Sequence):
            raise ValueError("evolution ledger corpus must be a sequence")
        if isinstance(raw_experiences, (str, bytes)) or not isinstance(raw_experiences, Sequence):
            raise ValueError("evolution ledger experiences must be a sequence")
        for item in raw_corpus:
            artifact = EvolutionCorpusArtifact.from_payload(item)
            if artifact.status == "admitted":
                candidate = artifact.with_status("candidate")
                ledger.add_corpus(candidate)
                ledger.admit_corpus(
                    candidate.artifact_digest,
                    admission_revision=artifact.admission_revision,
                )
            else:
                ledger.add_corpus(artifact)
        for item in raw_experiences:
            result = ledger.append(EvolutionExperience.from_payload(item))
            if not result.accepted:
                raise ValueError("duplicate experience in checkpoint")
        if int(payload.get("revision", -1)) != ledger.revision:
            raise ValueError("evolution ledger checkpoint revision mismatch")
        if str(payload.get("tail_event_digest", "")) != ledger.tail_event_digest:
            raise ValueError("evolution ledger checkpoint tail mismatch")
        return ledger


def workbench_outcome_to_experience(
    outcome: Any,
    *,
    parent_checkpoint_digest: str,
    partition: str = "train",
    input_digest: str = "",
    episode_id: str = "",
    user_correction_digest: str = "",
    resource_usage: Mapping[str, Any] | None = None,
) -> EvolutionExperience:
    """Project a WorkbenchOutcome without copying raw result data into training."""

    from seed_platform.workbench import WorkbenchOutcome

    if not isinstance(outcome, WorkbenchOutcome):
        raise TypeError("workbench projection requires WorkbenchOutcome")
    outcome_payload = outcome.to_payload()
    source_digest = content_digest(outcome_payload)
    derived_input_digest = input_digest or content_digest(
        {
            "request_id": outcome.request_id,
            "intent_id": outcome.intent_id,
            "capability_id": outcome.capability_id,
            "snapshot_id": outcome.snapshot_id,
        }
    )
    outcome_id = f"{outcome.request_id}:{outcome.call_id or outcome.status}:{outcome.tick}"
    return EvolutionExperience(
        experience_id=f"workbench:{outcome_id}",
        source_kind="workbench",
        source_id=outcome.capability_id,
        source_version=f"workbench-contract-v{outcome.version}",
        source_digest=source_digest,
        parent_checkpoint_digest=parent_checkpoint_digest,
        partition=partition,
        status=outcome.status,
        success=bool(outcome.success),
        request_id=outcome.request_id,
        intent_id=outcome.intent_id,
        call_id=outcome.call_id,
        outcome_id=outcome_id,
        episode_id=str(episode_id).strip(),
        tick=int(outcome.tick),
        input_digest=derived_input_digest,
        capability_id=outcome.capability_id,
        capability_snapshot_id=outcome.snapshot_id,
        result_digest=content_digest(outcome.result),
        error_code=outcome.error_code,
        resource_usage={} if resource_usage is None else resource_usage,
        user_correction_digest=user_correction_digest,
        mcp_schema_digest=outcome.mcp_registry_snapshot_id,
        redaction_revision=EVOLUTION_REDACTION_REVISION,
        metadata={"projection": "workbench-outcome-digest-only-v1"},
    )


__all__ = [
    "EVOLUTION_LEDGER_CHECKPOINT_FORMAT",
    "EVOLUTION_LEDGER_VERSION",
    "EvolutionAppendResult",
    "EvolutionExperienceLedger",
    "redact_sensitive_payload",
    "workbench_outcome_to_experience",
]
