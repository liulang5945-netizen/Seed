"""R5A-S0 red/green contract for grounded DTO conversion and replay."""

from __future__ import annotations

import pytest
import torch

from taiji import (
    BoundedReplayBuffer,
    GroundedOutcomeEvidence,
    InternalizationCausalGate,
    InternalizationConverter,
    InternalizationInput,
    InternalizationLedger,
    Outcome,
    WorldAffordance,
    content_digest,
)


def _source(
    *,
    evidence_id: str = "evidence:1",
    outcome_id: str = "outcome:1",
    grounded: bool = True,
    reward: float = 0.75,
    reward_terms: dict[str, float] | None = None,
    missing_reward_terms: bool = False,
    metadata: dict[str, str] | None = None,
) -> GroundedOutcomeEvidence:
    affordance = WorldAffordance(
        affordance_id="affordance:read",
        action_kind="read",
        actor_id="workspace",
        target_id="file:README.md",
        features=torch.tensor((0.2, 0.4, 0.8), dtype=torch.float32),
        feature_provenance="world-state-grounding" if grounded else "manual",
        grounding_lineage=("world-state:12", "object:file:README.md") if grounded else (),
    )
    return GroundedOutcomeEvidence(
        evidence_id=evidence_id,
        outcome_id=outcome_id,
        outcome=Outcome(intent_id="intent:read", reward=reward, success=True, tick=12),
        affordance=affordance,
        capability_snapshot_digest="capability-sha256:1",
        parent_checkpoint_id="checkpoint:parent:1",
        owner_id="taiji:workbench-outcome",
        reward_terms=(
            None
            if missing_reward_terms
            else (reward_terms if reward_terms is not None else {"outcome": reward})
        ),
        percept_digest="percept-sha256:1",
        world_digest="world-sha256:1",
        recovery_digest="recovery-sha256:1",
        metadata=metadata or {},
    )


def test_grounded_conversion_is_content_addressed_and_order_independent() -> None:
    converter = InternalizationConverter(seed=7, replay_budget=8)
    first = converter.convert(_source(reward_terms={"outcome": 0.75, "confidence": 0.8}))
    second = converter.convert(_source(reward_terms={"confidence": 0.8, "outcome": 0.75}))

    assert first.accepted is True
    assert first.status == "external"
    assert first.example is not None
    assert second.example is not None
    assert first.example.example_id == second.example.example_id
    assert first.example.content_digest == second.example.content_digest
    assert first.lifecycle.events == ("outcome_bound", "grounding_verified", "example_created")
    assert first.example.provenance
    assert "provider_text" not in first.example.content_digest


@pytest.mark.parametrize(
    ("source", "reason"),
    (
        (_source(grounded=False), "missing_grounding"),
        (_source(missing_reward_terms=True), "missing_reward_terms"),
        (_source(reward_terms={"outcome": 1.5}), "invalid_reward_terms"),
        (_source(reward=1.5, reward_terms={"outcome": 0.9}), "outcome_reward_out_of_bounds"),
    ),
)
def test_conversion_fails_closed_for_unbound_or_invalid_learning_input(
    source: GroundedOutcomeEvidence, reason: str
) -> None:
    result = InternalizationConverter().convert(source)

    assert result.accepted is False
    assert result.status == "rejected"
    assert result.reason == reason
    assert result.example is None
    assert "conversion_rejected" in result.lifecycle.events


def test_provider_and_capability_text_cannot_become_a_learning_source() -> None:
    with pytest.raises(ValueError, match="forbidden input"):
        _source(metadata={"provider_text": "invented answer"})

    source = _source()
    assert "capability_id" not in source.binding_payload()


def test_replay_is_train_only_bounded_and_deduplicated() -> None:
    converter = InternalizationConverter(replay_budget=1)
    first = converter.convert(_source()).example
    assert first is not None
    buffer = BoundedReplayBuffer(capacity=1)

    assert buffer.add(first) is True
    assert buffer.add(first) is False
    assert buffer.deduplicated_count == 1
    assert len(buffer.replay()) == 1
    with pytest.raises(ValueError, match="holdout"):
        buffer.add(first, partition="holdout")
    with pytest.raises(BufferError, match="exhausted"):
        other = converter.convert(_source(evidence_id="evidence:2")).example
        assert other is not None
        buffer.add(other)


def test_lifecycle_requires_causal_gate_and_never_resurrects_tombstones() -> None:
    ledger = InternalizationLedger(converter=InternalizationConverter(replay_budget=4))
    result = ledger.ingest(_source())
    assert result.example is not None
    example_id = result.example.example_id

    shadow = ledger.advance_status(example_id, "shadow")
    assert "shadow_learned" in shadow.events
    with pytest.raises(ValueError, match="causal gate"):
        ledger.advance_status(example_id, "internalized")

    gate = InternalizationCausalGate(True, True, True, True, True)
    admitted = ledger.advance_status(example_id, "internalized", causal_gate=gate)
    assert admitted.status == "internalized"
    assert "holdout_checked" in admitted.events
    assert "affordance_lesion_checked" in admitted.events
    tombstoned = ledger.advance_status(
        example_id,
        "tombstoned",
        causal_gate=gate,
        reason="five causal checks passed",
    )
    assert tombstoned.status == "tombstoned"
    assert len(ledger.replay_buffer.examples) == 0
    with pytest.raises(ValueError, match="resurrected"):
        ledger.advance_status(example_id, "shadow")


def test_checkpoint_roundtrip_preserves_replay_digest_lineage_and_terminal_state() -> None:
    ledger = InternalizationLedger(converter=InternalizationConverter(seed=11, replay_budget=4))
    result = ledger.ingest(_source())
    assert result.example is not None
    checkpoint = ledger.checkpoint()
    restored = InternalizationLedger.from_checkpoint(checkpoint)

    assert restored.replay_digest == ledger.replay_digest
    assert restored.replay_update_count == 1
    assert restored.lifecycle(result.example.example_id).status == "external"
    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)

    duplicate = restored.ingest(_source())
    assert duplicate.accepted is True
    assert duplicate.reason == "replay_deduplicated"
    assert restored.replay_update_count == 1
    assert restored.replay_buffer.deduplicated_count == 1

    conflict = restored.ingest(_source(reward=0.5))
    assert conflict.accepted is False
    assert conflict.reason == "evidence_id_content_conflict"
    assert restored.replay_update_count == 1
