from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from scripts.training.eval_taiji_a2_world import build_corpus
from taiji import WorldInterventionCorpus, WorldSchema
from taiji.contracts import WorldTransition
from taiji.internalization import content_digest
from taiji.world_evolution import NativeWorldPredictionTrainer, transition_to_case


def _transition(case, suffix: str) -> WorldTransition:
    action = replace(case.action, action_id=f"{case.action.action_id}-{suffix}")
    outcome = replace(case.expected_outcome, intent_id=action.action_id)
    return WorldTransition(
        before=case.initial,
        action=action,
        after=case.expected_state,
        outcome=outcome,
    )


def _partitions() -> tuple[tuple[WorldTransition, ...], ...]:
    corpus = build_corpus()
    train_cases = corpus.train[2:]
    retention = tuple(_transition(case, "retention") for case in corpus.train[:2])
    train = tuple(_transition(case, "train") for case in train_cases)
    holdout = tuple(_transition(case, "holdout") for case in corpus.holdout)
    return train, holdout, retention


def test_world_transition_intake_improves_holdout_and_roundtrips() -> None:
    train, holdout, retention = _partitions()
    schema = WorldSchema.from_corpus(
        WorldInterventionCorpus(
            train=tuple(transition_to_case(item) for item in train),
        )
    )
    trainer = NativeWorldPredictionTrainer(schema, hidden_dim=32, epochs=350, seed=11)
    report = trainer.consolidate(
        train,
        holdout_transitions=holdout,
        retention_transitions=retention,
    )

    assert report.admitted is True
    assert report.native_holdout_error < report.frozen_holdout_error
    assert report.replay_only_holdout_error == pytest.approx(report.frozen_holdout_error)
    assert report.native_retention_error <= report.frozen_retention_error + 0.05
    assert trainer.consumed_transition_ids == tuple(sorted(item.action.action_id for item in train))
    restored = NativeWorldPredictionTrainer.from_checkpoint(trainer.checkpoint())
    assert content_digest(restored.checkpoint()) == content_digest(trainer.checkpoint())


def test_world_transition_boundary_rejects_digest_only_input() -> None:
    train, holdout, retention = _partitions()
    schema = WorldSchema.from_corpus(
        WorldInterventionCorpus(train=tuple(transition_to_case(item) for item in train))
    )
    trainer = NativeWorldPredictionTrainer(schema, epochs=2)
    with pytest.raises(TypeError, match="WorldTransition"):
        trainer.consolidate(
            ("world-state-digest-only",),
            holdout_transitions=holdout,
            retention_transitions=retention,
        )


def test_world_transition_partitions_and_checkpoint_fail_closed() -> None:
    train, holdout, retention = _partitions()
    schema = WorldSchema.from_corpus(
        WorldInterventionCorpus(train=tuple(transition_to_case(item) for item in train))
    )
    trainer = NativeWorldPredictionTrainer(schema, epochs=2)
    with pytest.raises(ValueError, match="partitions must be disjoint"):
        trainer.consolidate(
            train,
            holdout_transitions=(train[0],),
            retention_transitions=retention,
        )
    tampered = deepcopy(trainer.checkpoint())
    tampered["revision"] = 9
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        NativeWorldPredictionTrainer.from_checkpoint(tampered)
