from __future__ import annotations

import torch

from taiji import ConceptFormationOrgan, EpisodicMemoryRecord, Outcome


def _record(
    memory_id: str,
    episode_id: str,
    cue: tuple[float, float],
    relation: str,
    object_id: str,
    tick: int,
) -> EpisodicMemoryRecord:
    return EpisodicMemoryRecord(
        memory_id=memory_id,
        episode_id=episode_id,
        tick=tick,
        cue=torch.tensor(cue),
        outcome=Outcome(
            intent_id=f"intent-{memory_id}",
            reward=1.0,
            success=True,
            tick=tick,
        ),
        event_ids=(f"event-{memory_id}",),
        assembly_ids=(f"assembly-{memory_id}",),
        object_ids=(object_id,),
        relation_ids=(f"agent:{relation}:{object_id}",),
    )


def test_concept_formation_capacity_checkpoint_and_lesion() -> None:
    organ = ConceptFormationOrgan(capacity=1, prune_threshold=0.0)
    records = (
        _record("near-0", "episode-0", (1.0, 0.0), "near", "object-0", 1),
        _record("near-1", "episode-1", (0.99, 0.1), "near", "object-1", 1),
        _record("far-0", "episode-2", (-1.0, 0.0), "far", "object-2", 1),
        _record("far-1", "episode-3", (-0.99, -0.1), "far", "object-3", 1),
    )

    concepts = organ.consolidate(records, tick=1)

    assert len(concepts) == 1
    retrieval_organ = ConceptFormationOrgan(capacity=4, prune_threshold=0.0)
    retrieval_concepts = retrieval_organ.consolidate(records[:2], tick=1)
    match = retrieval_organ.retrieve(
        records[0].cue,
        object_ids=("unseen-object",),
        relation_ids=("agent:near:unseen-object",),
    )
    assert match and match[0].concept.concept_id == retrieval_concepts[0].concept_id
    assert match[0].score >= retrieval_organ.similarity_threshold
    checkpoint = organ.checkpoint()
    restored = ConceptFormationOrgan.from_checkpoint(checkpoint)
    assert restored.capacity == 1
    assert restored.concepts[0].concept_id == concepts[0].concept_id
    assert restored.concepts[0].support_event_ids == concepts[0].support_event_ids
    assert organ.lesion((concepts[0].concept_id, "missing")) == (concepts[0].concept_id,)
    assert organ.concepts == ()


def test_concept_formation_plasticity_keeps_identity_while_absorbing_new_cues() -> None:
    organ = ConceptFormationOrgan(capacity=4, plasticity_rate=0.25, prune_threshold=0.0)
    initial = organ.consolidate(
        (
            _record("initial-0", "episode-0", (1.0, 0.0), "near", "object-0", 1),
            _record("initial-1", "episode-1", (0.99, 0.1), "near", "object-1", 1),
        ),
        tick=1,
    )[0]
    updated = organ.consolidate(
        (
            _record("updated-0", "episode-2", (0.9, 0.435), "near", "object-2", 2),
            _record("updated-1", "episode-3", (0.89, 0.456), "near", "object-3", 2),
        ),
        tick=2,
    )[0]

    assert updated.concept_id == initial.concept_id
    assert updated.update_count == 2
    assert updated.support_event_ids == (
        "event-initial-0",
        "event-initial-1",
        "event-updated-0",
        "event-updated-1",
    )
    assert float(updated.prototype[1]) > float(initial.prototype[1])
