from __future__ import annotations

from copy import deepcopy

import pytest

from seed_platform.evolution_ledger import (
    EvolutionExperienceLedger,
    redact_sensitive_payload,
    workbench_outcome_to_experience,
)
from seed_platform.workbench import WorkbenchOutcome
from taiji.evolution_experience import EvolutionCorpusArtifact, EvolutionExperience
from taiji.internalization import content_digest


def _corpus(*, partition: str = "train", status: str = "candidate") -> EvolutionCorpusArtifact:
    content = {"title": "filesystem read", "procedure": ["validate", "read"]}
    return EvolutionCorpusArtifact(
        corpus_id="skill:filesystem-read",
        source_kind="skill_artifact",
        source_id="skill.filesystem.read",
        source_version="1",
        source_digest=content_digest({"source": "skill.filesystem.read", "version": "1"}),
        unit_kind="procedure",
        content=content,
        partition=partition,
        status=status,
    )


def _experience(
    *,
    experience_id: str = "episode-1",
    partition: str = "train",
    tick: int = 1,
) -> EvolutionExperience:
    return EvolutionExperience(
        experience_id=experience_id,
        source_kind="provider",
        source_id="fixture-provider",
        source_version="1",
        source_digest="a" * 64,
        parent_checkpoint_digest="b" * 64,
        partition=partition,
        status="success",
        success=True,
        episode_id="episode",
        tick=tick,
        result_digest="c" * 64,
        reward_components={"task": 1.0},
        resource_usage={"latency_ms": 2.0},
        metadata={"fixture": "e1"},
    )


def test_corpus_admission_roundtrip_and_sensitive_redaction() -> None:
    ledger = EvolutionExperienceLedger()
    candidate = _corpus()
    ledger.add_corpus(candidate)
    admitted = ledger.admit_corpus(candidate.artifact_digest, admission_revision="admission-1")

    restored = EvolutionCorpusArtifact.from_payload(admitted.to_payload())
    assert restored == admitted
    assert restored.artifact_digest == candidate.artifact_digest

    redacted, flags = redact_sensitive_payload({"api_key": "secret", "nested": {"token": "x"}})
    assert redacted == {
        "api_key": "<redacted>",
        "nested": {"token": "<redacted>"},
    }
    assert flags == ("api_key", "nested.token")
    with pytest.raises(ValueError, match="unredacted sensitive"):
        EvolutionCorpusArtifact(
            corpus_id="unsafe",
            source_kind="mcp_artifact",
            source_id="mcp.unsafe",
            source_version="1",
            source_digest="d" * 64,
            unit_kind="knowledge",
            content={"api_key": "secret"},
        )


def test_workbench_projection_keeps_result_as_digest_only() -> None:
    outcome = WorkbenchOutcome(
        request_id="request-1",
        intent_id="intent-1",
        call_id="call-1",
        capability_id="seed.workbench.read",
        snapshot_id="snapshot-1",
        status="success",
        success=True,
        result={"path": "README.md", "text": "private output"},
        tick=7,
        mcp_registry_snapshot_id="e" * 64,
    )
    projected = workbench_outcome_to_experience(
        outcome,
        parent_checkpoint_digest="f" * 64,
        partition="holdout",
    )
    assert projected.result_digest == content_digest(outcome.result)
    assert "private output" not in projected.to_payload().__repr__()
    assert projected.source_digest == content_digest(outcome.to_payload())
    assert projected.partition == "holdout"


def test_append_is_idempotent_and_rejects_identity_conflicts() -> None:
    ledger = EvolutionExperienceLedger()
    first = ledger.append(_experience()).experience
    duplicate = ledger.append(first)
    assert duplicate.duplicate is True
    assert duplicate.accepted is False
    assert ledger.records()[0].event_sequence == 1

    with pytest.raises(ValueError, match="experience_id content conflict"):
        ledger.append(_experience(tick=2))

    with pytest.raises(ValueError, match="another identity"):
        ledger.append(_experience(experience_id="episode-2"))


def test_checkpoint_replays_admission_chain_and_rejects_tampering() -> None:
    ledger = EvolutionExperienceLedger()
    corpus = _corpus()
    ledger.add_corpus(corpus)
    ledger.admit_corpus(corpus.artifact_digest, admission_revision="admission-1")
    first = ledger.append(_experience()).experience
    checkpoint = ledger.checkpoint()

    restored = EvolutionExperienceLedger.from_checkpoint(checkpoint)
    assert restored.revision == ledger.revision
    assert restored.tail_event_digest == ledger.tail_event_digest
    second = restored.append(_experience(experience_id="episode-2", tick=2)).experience
    assert second.event_sequence == 2
    assert second.previous_event_digest == first.event_digest

    tampered = deepcopy(checkpoint)
    tampered["experiences"][0]["success"] = False
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        EvolutionExperienceLedger.from_checkpoint(tampered)


def test_training_view_excludes_holdout_and_unadmitted_corpus() -> None:
    ledger = EvolutionExperienceLedger()
    train_candidate = _corpus()
    holdout_candidate = _corpus(partition="holdout")
    ledger.add_corpus(train_candidate)
    ledger.add_corpus(holdout_candidate)
    ledger.admit_corpus(train_candidate.artifact_digest, admission_revision="admission-1")
    ledger.append(_experience(partition="train"))
    ledger.append(_experience(experience_id="holdout-1", partition="holdout"))

    train_corpus, train_experiences = ledger.training_view()
    assert [item.artifact_digest for item in train_corpus] == [train_candidate.artifact_digest]
    assert [item.experience_id for item in train_experiences] == ["episode-1"]
