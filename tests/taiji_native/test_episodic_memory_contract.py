from __future__ import annotations

import torch

from taiji import (
    ActionIntent,
    EpisodicMemoryRecord,
    EpisodicMemoryStore,
    Outcome,
    SemanticMemoryLearner,
    TSKV8Adapter,
)


def _record(memory_id: str, cue: torch.Tensor, *, tick: int = 1) -> EpisodicMemoryRecord:
    intent = ActionIntent(
        intent_id=f"{memory_id}:intent",
        kind="test-action",
        tick=max(0, tick - 1),
    )
    return EpisodicMemoryRecord(
        memory_id=memory_id,
        episode_id="episode-memory",
        tick=tick,
        cue=cue,
        action_intent=intent,
        outcome=Outcome(intent.intent_id, reward=1.0, success=True, tick=tick),
    )


def test_episodic_memory_retrieval_is_content_addressed_and_checkpointable() -> None:
    store = EpisodicMemoryStore(capacity=2)
    first = _record("first", torch.tensor([1.0, 0.0]))
    second = _record("second", torch.tensor([0.0, 1.0]))
    store.write(first)
    store.write(second)

    hits = store.retrieve(torch.tensor([0.9, 0.1]), limit=2)
    assert hits[0].record.memory_id == "first"
    assert hits[0].score > hits[1].score

    restored = EpisodicMemoryStore.from_checkpoint(store.checkpoint())
    assert restored.count == 2
    assert restored.retrieve(torch.tensor([0.9, 0.1]))[0].record.memory_id == "first"


def test_episodic_memory_capacity_keeps_latest_real_records() -> None:
    store = EpisodicMemoryStore(capacity=2, cue_dim=2)
    store.write(_record("one", torch.tensor([1.0, 0.0])))
    store.write(_record("two", torch.tensor([0.0, 1.0])))
    store.write(_record("three", torch.tensor([1.0, 1.0])))

    assert tuple(record.memory_id for record in store.records) == ("two", "three")


def test_adapter_writes_real_outcome_and_restores_episodic_memory() -> None:
    model = TSKV8Adapter()
    model.attach_episodic_memory(EpisodicMemoryStore(capacity=8))
    model.attach_semantic_memory(SemanticMemoryLearner(model.perception.feature_dim))
    model.observe(97, learn=False)
    model.act((97, 98), sample=False)
    model.settle_action(1.0, learn=False)
    model.consolidate_semantic_memory(epochs=100, learning_rate=0.1)

    state = model.cognitive_snapshot()
    assert model._episodic_memory is not None
    assert model._episodic_memory.count == 1
    assert len(state.memory.episodic_ids) == 1
    assert len(state.memory.working_items) == 1
    assert state.memory.working_items[0].item_id == state.memory.working_ids[0]
    record = model._episodic_memory.records[0]
    assert record.outcome is not None
    assert record.outcome.reward == 1.0
    assert record.episode_id == state.episode_id

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored._episodic_memory is not None
    assert restored._episodic_memory.count == 1
    assert restored._semantic_memory is not None
    assert restored._semantic_memory.consolidation_count == 1
    assert restored.cognitive_snapshot().memory.episodic_ids == state.memory.episodic_ids
