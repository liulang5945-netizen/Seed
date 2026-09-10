"""Targeted tests for the fast/slow K continuation instance (B3 pilot FS arm).

Frozen semantics under test: wake writes the continuation local delta into
``fast`` while ``slow`` stays bit-identical (so the wake trajectory equals
direct continuation); sleep replay applies deltas to ``slow``;
consolidation moves ``fast`` into ``slow`` without changing the effective
weights; the checkpoint restores the effective state bit-for-bit.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from taiji import (
    ContentPlan,
    Goal,
    PerceptEvent,
    StructuredSemanticCorpus,
    StructuredSemanticExample,
    StructuredSemanticTransitionCorpus,
    StructuredSemanticTransitionExample,
    WorldState,
    content_digest,
)
from taiji.k_fast_slow import (
    FastSlowKInstance,
    replay_sample_indices,
)

PARENT_DIGEST = "a" * 64
BUNDLE_DIGEST = "b" * 64
SOURCE_DIGEST = "c" * 64
DIGEST_A = "d" * 64
DIGEST_B = "e" * 64


def _percept(name: str, features: tuple[float, ...]) -> PerceptEvent:
    return PerceptEvent(
        event_id=f"percept-{name}",
        observation_tick=1,
        modality="fast-slow-test",
        features=torch.tensor(features, dtype=torch.float32),
        assembly_id=f"assembly-{name}",
        confidence=1.0,
    )


def _world(tick: int) -> WorldState:
    relations = (
        ("workbench", "read", "pending") if tick == 0 else ("workbench", "read", "success"),
    )
    return WorldState(tick=tick, entities=("workbench",), relations=relations, uncertainty=0.0)


def _semantic_example(name: str, features: tuple[float, ...]) -> StructuredSemanticExample:
    percept = _percept(name, features)
    goal = Goal(goal_id="goal:g1", description="inspect", priority=0.9)
    content = ContentPlan(
        content_id="content:c1",
        intent_id="intent:c1",
        intent_kind="report",
        semantic_slots={},
        source_goal_id=goal.goal_id,
        expected_outcome="verified",
        tick=1,
    )
    return StructuredSemanticExample(
        example_id=f"semantic:{name}",
        family_id=f"semantic-family:{name}",
        percept=percept,
        world=_world(1),
        goal=goal,
        content=content,
    )


def _transition_example(name: str, features: tuple[float, ...]) -> StructuredSemanticTransitionExample:
    event = _percept(name, features)
    goal = Goal(goal_id="goal:g1", description="inspect", priority=0.9)
    content = ContentPlan(
        content_id="content:c1",
        intent_id="intent:c1",
        intent_kind="report",
        semantic_slots={},
        source_goal_id=goal.goal_id,
        expected_outcome="verified",
        tick=1,
    )
    return StructuredSemanticTransitionExample(
        example_id=f"transition:{name}",
        family_id=f"transition-family:{name}",
        before=_world(0),
        event=event,
        after=_world(1),
        goal=goal,
        content=content,
    )


def _tiny_setup():
    features = {
        "s1": (1.0, 0.0, 0.0),
        "s2": (0.0, 1.0, 0.0),
        "s3": (0.0, 0.0, 1.0),
        "s4": (0.5, 0.5, 0.0),
    }
    semantic_corpus = StructuredSemanticCorpus.from_splits(
        train=(
            _semantic_example("s1", features["s1"]),
            _semantic_example("s2", features["s2"]),
        ),
        dev=(_semantic_example("s3", features["s3"]),),
        test=(_semantic_example("s4", features["s4"]),),
    )
    transition_corpus = StructuredSemanticTransitionCorpus.from_splits(
        train=(
            _transition_example("s1", features["s1"]),
            _transition_example("s2", features["s2"]),
        ),
        dev=(_transition_example("s3", features["s3"]),),
        test=(_transition_example("s4", features["s4"]),),
    )
    semantic = StructuredSemanticLearner(semantic_corpus)
    transition = StructuredSemanticTransitionLearner(transition_corpus)
    experiences = []
    for index, name in enumerate(("s1", "s2")):
        experiences.append(
            SimpleNamespace(
                semantic_example=_semantic_example(name, features[name]),
                transition_example=_transition_example(name, features[name]),
                experience_digest=content_digest({"experience": name, "pad": index}),
            )
        )
    return semantic, transition, semantic_corpus, transition_corpus, experiences


from taiji import StructuredSemanticLearner, StructuredSemanticTransitionLearner  # noqa: E402


def _fresh_instance(semantic, transition):
    return FastSlowKInstance(
        semantic_parent_checkpoint=semantic.checkpoint(),
        transition_parent_checkpoint=transition.checkpoint(),
        parent_worker_bundle_digest=BUNDLE_DIGEST,
        source_manifest_digest=SOURCE_DIGEST,
        parent_checkpoint_digest=PARENT_DIGEST,
    )


def _continuation_arm(semantic, transition, semantic_corpus, transition_corpus, experiences):
    """Direct continuation over the same stream (the C-arm twin)."""

    c_semantic = StructuredSemanticLearner.from_checkpoint(
        semantic.checkpoint(), semantic_corpus
    )
    c_transition = StructuredSemanticTransitionLearner.from_checkpoint(
        transition.checkpoint(), transition_corpus
    )
    for experience in experiences:
        c_semantic.fit((experience.semantic_example,), epochs=1, learning_rate=2.0)
        c_transition.fit((experience.transition_example,), epochs=1, learning_rate=0.2)
    return c_semantic, c_transition


def test_birth_state_is_parent_with_zero_fast():
    semantic, transition, _corpus, _tcorpus, _experiences = _tiny_setup()
    instance = _fresh_instance(semantic, transition)
    for worker in ("k1.semantic", "k2.transition"):
        assert instance.is_fast_zero(worker)
        assert instance.effective_state_digest(worker) == content_digest(
            dict(semantic.checkpoint()["state_dict"] if worker == "k1.semantic" else transition.checkpoint()["state_dict"])
        )


def test_wake_matches_continuation_and_leaves_slow_unchanged():
    semantic, transition, semantic_corpus, transition_corpus, experiences = _tiny_setup()
    instance = _fresh_instance(semantic, transition)
    slow_before = {
        worker: instance.slow_state_digest(worker) for worker in instance.slow
    }
    for experience in experiences:
        instance.wake_experience(
            experience,
            semantic_epochs=1,
            semantic_lr=2.0,
            transition_epochs=1,
            transition_lr=0.2,
        )
    for worker in instance.slow:
        assert instance.slow_state_digest(worker) == slow_before[worker]
        assert not instance.is_fast_zero(worker)
    c_semantic, c_transition = _continuation_arm(
        semantic, transition, semantic_corpus, transition_corpus, experiences
    )
    assert instance.effective_state_digest("k1.semantic") == content_digest(
        c_semantic.state_dict()
    )
    assert instance.effective_state_digest("k2.transition") == content_digest(
        c_transition.state_dict()
    )


def test_replay_updates_slow_and_consolidation_preserves_effective():
    semantic, transition, _corpus, _tcorpus, experiences = _tiny_setup()
    instance = _fresh_instance(semantic, transition)
    for experience in experiences:
        instance.wake_experience(
            experience,
            semantic_epochs=1,
            semantic_lr=2.0,
            transition_epochs=1,
            transition_lr=0.2,
        )
    effective_after_wake = {
        worker: instance.effective_state_digest(worker) for worker in instance.slow
    }
    instance.replay_experience(
        experiences[0],
        semantic_epochs=1,
        semantic_lr=2.0,
        transition_epochs=1,
        transition_lr=0.2,
    )
    # Replay touched slow, left fast alone, and moved the effective state.
    for worker in instance.slow:
        assert not instance.is_fast_zero(worker)
    assert any(
        instance.effective_state_digest(worker) != effective_after_wake[worker]
        for worker in instance.slow
    )
    effective_before_consolidation = {
        worker: instance.effective_state_digest(worker) for worker in instance.slow
    }
    instance.consolidate()
    for worker in instance.slow:
        assert instance.is_fast_zero(worker)
        # Consolidation moves fast into slow without changing effective weights.
        assert instance.effective_state_digest(worker) == effective_before_consolidation[worker]


def test_checkpoint_roundtrip_restores_effective_state():
    semantic, transition, _corpus, _tcorpus, experiences = _tiny_setup()
    instance = _fresh_instance(semantic, transition)
    for experience in experiences:
        instance.wake_experience(
            experience,
            semantic_epochs=1,
            semantic_lr=2.0,
            transition_epochs=1,
            transition_lr=0.2,
        )
    payload = instance.checkpoint()
    restored = FastSlowKInstance.from_checkpoint(payload)
    for worker in instance.slow:
        assert restored.effective_state_digest(worker) == instance.effective_state_digest(worker)
        assert restored.slow_state_digest(worker) == instance.slow_state_digest(worker)
        assert restored.is_fast_zero(worker) == instance.is_fast_zero(worker)
    assert restored.wake_steps == instance.wake_steps
    assert restored.replay_steps == instance.replay_steps
    assert restored.consolidations == instance.consolidations
    assert restored.replay_digests == instance.replay_digests


def test_replay_sample_indices_are_deterministic_and_without_replacement():
    first = replay_sample_indices(buffer_size=150, sample_count=50, digest=DIGEST_A)
    second = replay_sample_indices(buffer_size=150, sample_count=50, digest=DIGEST_A)
    assert first == second
    assert len(first) == 50
    assert len(set(first)) == 50
    assert all(0 <= index < 150 for index in first)
    other = replay_sample_indices(buffer_size=150, sample_count=50, digest=DIGEST_B)
    assert other != first
