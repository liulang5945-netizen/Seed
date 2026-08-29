"""S0 deterministic Gate for trace-grounded interaction groups."""

from __future__ import annotations

from dataclasses import replace

import pytest

from taiji import (
    InteractionGroupEvaluator,
    InteractionGroupEvaluatorConfig,
    InteractionTraceCorpus,
    InteractionTraceEpisode,
    InteractionTraceEvent,
)


def _episode(
    episode_id: str,
    checkpoint_revision: int,
    owner_ids: tuple[str, ...],
    outcome: float,
    recovery_effect: float,
    *,
    split: str,
    context_id: str,
) -> InteractionTraceEpisode:
    outcome_id = f"{split}-outcome-{episode_id}"
    return InteractionTraceEpisode(
        episode_id=f"{split}-{episode_id}",
        checkpoint_revision=checkpoint_revision,
        outcome_id=outcome_id,
        events=tuple(
            InteractionTraceEvent(
                event_id=f"{split}-{episode_id}-event-{index}",
                owner_id=owner_id,
                episode_id=f"{split}-{episode_id}",
                checkpoint_revision=checkpoint_revision,
                outcome_id=outcome_id,
                resource_cost=0.4 + 0.1 * index,
            )
            for index, owner_id in enumerate(owner_ids)
        ),
        outcome=outcome,
        recovery_effect=recovery_effect,
        context_id=context_id,
    )


def _corpus() -> InteractionTraceCorpus:
    train = (
        _episode("none", 7, (), 0.0, 0.0, split="train", context_id="task-ab"),
        _episode("a", 7, ("surface-a",), 0.2, 0.1, split="train", context_id="task-ab"),
        _episode("b", 7, ("surface-b",), 0.3, 0.1, split="train", context_id="task-ab"),
        _episode(
            "ab", 7, ("surface-a", "surface-b"), 1.0, 0.7, split="train", context_id="task-ab"
        ),
        _episode("c", 7, ("surface-c",), 0.6, 0.2, split="train", context_id="task-cd"),
        _episode("d", 7, ("surface-d",), 0.2, 0.1, split="train", context_id="task-cd"),
        _episode(
            "cd", 7, ("surface-c", "surface-d"), -0.2, 0.05, split="train", context_id="task-cd"
        ),
        _episode("none-cd", 7, (), 0.0, 0.0, split="train", context_id="task-cd"),
    )
    holdout = (
        _episode("none", 7, (), 0.05, 0.0, split="holdout", context_id="task-ab-holdout"),
        _episode("a", 7, ("surface-a",), 0.25, 0.1, split="holdout", context_id="task-ab-holdout"),
        _episode("b", 7, ("surface-b",), 0.35, 0.1, split="holdout", context_id="task-ab-holdout"),
        _episode(
            "ab",
            7,
            ("surface-a", "surface-b"),
            1.1,
            0.8,
            split="holdout",
            context_id="task-ab-holdout",
        ),
        _episode("c", 7, ("surface-c",), 0.55, 0.2, split="holdout", context_id="task-cd-holdout"),
        _episode("d", 7, ("surface-d",), 0.15, 0.1, split="holdout", context_id="task-cd-holdout"),
        _episode(
            "cd",
            7,
            ("surface-c", "surface-d"),
            -0.3,
            0.05,
            split="holdout",
            context_id="task-cd-holdout",
        ),
        _episode("none-cd", 7, (), 0.05, 0.0, split="holdout", context_id="task-cd-holdout"),
    )
    return InteractionTraceCorpus(train=train, holdout=holdout)


def test_interaction_group_s0_recovers_complementarity_conflict_and_lesion() -> None:
    result = InteractionGroupEvaluator().evaluate(_corpus())

    assert result.passed is True
    assert result.metrics["role_label_input_count"] == 0
    assert result.metrics["holdout_direction_preserved"] is True
    assert result.metrics["lesion_effects_observed"] is True
    assert result.metrics["checkpoint_roundtrip"] is True
    assert result.metrics["checkpoint_owner_lineage_preserved"] is True
    assert any(item.interaction_kind == "complementary" for item in result.state.groups)
    assert any(item.interaction_kind == "conflicting" for item in result.state.groups)
    assert any(item["event"] == "counterfactual_evaluated" for item in result.events)
    assert any(item["event"] == "lesion_evaluated" for item in result.events)


def test_holdout_outcome_change_cannot_become_positive_attribution() -> None:
    corpus = _corpus()
    corrupted_holdout = tuple(
        replace(
            episode,
            outcome={"holdout-ab": -1.0, "holdout-cd": 2.0}.get(
                episode.episode_id, episode.outcome
            ),
        )
        for episode in corpus.holdout
    )
    result = InteractionGroupEvaluator().evaluate(
        InteractionTraceCorpus(train=corpus.train, holdout=corrupted_holdout)
    )

    assert result.passed is False
    assert result.metrics["holdout_direction_preserved"] is False
    assert result.state.groups == ()
    assert (
        result.source_trace_digest
        == InteractionGroupEvaluator().evaluate(corpus).source_trace_digest
    )


def test_mixed_checkpoint_revision_fails_closed_without_cross_revision_group() -> None:
    corpus = _corpus()
    drifted_events = tuple(
        replace(event, checkpoint_revision=8, episode_id=corpus.train[3].episode_id)
        for event in corpus.train[3].events
    )
    drifted = replace(corpus.train[3], checkpoint_revision=8, events=drifted_events)
    result = InteractionGroupEvaluator().evaluate(
        InteractionTraceCorpus(
            train=(*corpus.train[:3], drifted, *corpus.train[4:]), holdout=corpus.holdout
        )
    )

    assert result.passed is False
    assert result.metrics["failure_reason"] == "mixed_checkpoint_revision"
    assert result.state.groups == ()


def test_resource_budget_rejects_evaluation_without_mutating_parent_state() -> None:
    corpus = _corpus()
    evaluator = InteractionGroupEvaluator(
        InteractionGroupEvaluatorConfig(maximum_resource_cost=0.1)
    )
    result = evaluator.evaluate(corpus)

    assert result.passed is False
    assert result.state.groups == ()
    assert any(
        item["event"] == "group_rejected" and item["reason"] == "resource_pressure"
        for item in result.events
    )


def test_checkpoint_rejects_group_bound_to_a_different_trace_digest() -> None:
    result = InteractionGroupEvaluator().evaluate(_corpus())
    payload = result.state.checkpoint()
    payload["groups"][0]["source_trace_digest"] = "f" * 64

    with pytest.raises(ValueError, match="different trace revision"):
        result.state.from_checkpoint(payload)
