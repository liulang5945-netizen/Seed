from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from scripts.training.eval_taiji_a2_world import build_corpus
from taiji import (
    CapacityGrowthTrigger,
    CapacityGrowthTriggerDecision,
    CapacityGrowthTriggerPolicy,
    NativeFixedCapacityPreflight,
    WorldInterventionCorpus,
    WorldSchema,
)
from taiji.contracts import WorldTransition
from taiji.internalization import content_digest
from taiji.world_evolution import transition_to_case


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
    train = tuple(_transition(case, "train") for case in corpus.train[2:])
    holdout = tuple(_transition(case, "holdout") for case in corpus.holdout)
    retention = tuple(_transition(case, "retention") for case in corpus.train[:2])
    return train, holdout, retention


def _preflight() -> NativeFixedCapacityPreflight:
    train, _, _ = _partitions()
    schema = WorldSchema.from_corpus(
        WorldInterventionCorpus(train=tuple(transition_to_case(item) for item in train))
    )
    return NativeFixedCapacityPreflight(
        schema,
        hidden_dim=32,
        capacity_limit=32,
        seeds=(11, 29, 47),
        epochs=350,
    )


def test_fixed_capacity_preflight_is_multiseed_and_fail_closed() -> None:
    train, holdout, retention = _partitions()
    preflight = _preflight()
    report = preflight.compare(
        train,
        holdout_transitions=holdout,
        retention_transitions=retention,
    )

    assert len(report.seed_results) == 3
    assert all(item.admitted for item in report.seed_results)
    assert report.mean_native_holdout_error < report.mean_frozen_holdout_error
    assert report.mean_replay_only_holdout_error == pytest.approx(
        report.mean_frozen_holdout_error
    )
    assert report.maximum_retention_regression <= 0.05
    assert report.holdout_error_std <= 0.2
    assert report.capacity_pressure.pressure == pytest.approx(1.0)
    assert report.fixed_capacity_adequate is True
    assert report.trigger_decision.should_propose is False
    assert report.trigger_decision.consecutive_failure_steps == 0
    assert "fixed_capacity_not_persistently_failing" in report.trigger_decision.reasons


def test_growth_trigger_requires_persistent_independent_failure() -> None:
    trigger = CapacityGrowthTrigger(
        region_id="world.local",
        policy=CapacityGrowthTriggerPolicy(
            ema_rate=1.0,
            maximum_holdout_error=0.5,
            minimum_capacity_pressure=0.8,
            required_failure_steps=2,
        ),
    )
    first = trigger.observe(
        residual_error=0.9,
        retention_regression=0.0,
        capacity_pressure=1.0,
        structural_budget=1,
        evidence_ids=("round:1",),
    )
    second = trigger.observe(
        residual_error=0.9,
        retention_regression=0.0,
        capacity_pressure=1.0,
        structural_budget=1,
        evidence_ids=("round:2",),
    )

    assert first.should_propose is False
    assert "failure_persistence_below_threshold" in first.reasons
    assert second.should_propose is True
    assert second.reasons == ()
    assert CapacityGrowthTriggerDecision.from_payload(second.to_payload()) == second


def test_capacity_preflight_checkpoint_rejects_tampering_and_roundtrips() -> None:
    preflight = _preflight()
    before = preflight.checkpoint()
    restored = NativeFixedCapacityPreflight.from_checkpoint(before)
    assert content_digest(restored.checkpoint()) == content_digest(before)

    tampered = deepcopy(before)
    tampered["structural_budget"] = 99
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        NativeFixedCapacityPreflight.from_checkpoint(tampered)
