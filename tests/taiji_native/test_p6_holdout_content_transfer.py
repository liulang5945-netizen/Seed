from __future__ import annotations

from taiji import (
    ContentCandidate,
    ContentSelectionContext,
    ContentSelector,
    ContentTrainingExample,
)


def _train_selector() -> ContentSelector:
    answer = ContentCandidate(
        candidate_id="train-answer",
        intent_id="train:intent",
        intent_kind="report_status",
        semantic_slots={"mode": "answer"},
        goal_alignment=0.9,
        world_relevance=0.3,
        information_gain=0.2,
        confidence=0.9,
        uncertainty=0.1,
        resource_cost=0.1,
    )
    ask = ContentCandidate(
        candidate_id="train-ask",
        intent_id="train:intent",
        intent_kind="request_information",
        semantic_slots={"mode": "clarify"},
        goal_alignment=0.6,
        world_relevance=0.9,
        information_gain=0.9,
        confidence=0.7,
        uncertainty=0.2,
        resource_cost=0.2,
    )
    certain = ContentSelectionContext(0.9, 0.1, novelty=0.1, resource_budget=0.8)
    uncertain = ContentSelectionContext(0.9, 0.9, novelty=0.8, resource_budget=0.8)
    selector = ContentSelector()
    selector.fit(
        (
            ContentTrainingExample(answer, certain, 1.0),
            ContentTrainingExample(ask, certain, 0.0),
            ContentTrainingExample(answer, uncertain, 0.0),
            ContentTrainingExample(ask, uncertain, 1.0),
        ),
        epochs=400,
    )
    return selector


def test_holdout_content_uses_learned_context_utility_not_candidate_names() -> None:
    holdout = ContentCandidate(
        candidate_id="holdout-forecast-42",
        intent_id="holdout:intent:forecast",
        intent_kind="forecast_digest",
        semantic_slots={"format": "digest", "regions": ["east", "south"]},
        goal_id="stay-informed",
        goal_alignment=0.55,
        world_relevance=0.88,
        information_gain=0.92,
        confidence=0.68,
        uncertainty=0.2,
        resource_cost=0.2,
    )
    answer = ContentCandidate(
        candidate_id="holdout-answer-7",
        intent_id="holdout:intent:forecast",
        intent_kind="static_summary",
        semantic_slots={"format": "summary"},
        goal_id="stay-informed",
        goal_alignment=0.9,
        world_relevance=0.3,
        information_gain=0.2,
        confidence=0.9,
        uncertainty=0.1,
        resource_cost=0.1,
    )
    context = ContentSelectionContext(0.9, 0.9, novelty=0.8, resource_budget=0.8)
    selector = _train_selector()
    decision = selector.select((holdout, answer), context)

    assert decision.selected.candidate_id == "holdout-forecast-42"
    assert decision.selected.intent_kind == "forecast_digest"
    assert decision.selected.semantic_slots == {"format": "digest", "regions": ["east", "south"]}

