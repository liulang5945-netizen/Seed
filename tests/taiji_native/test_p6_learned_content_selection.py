from __future__ import annotations

from taiji import (
    ContentCandidate,
    ContentSelectionContext,
    ContentSelector,
    ContentTrainingExample,
)


def _candidates() -> tuple[ContentCandidate, ContentCandidate]:
    return (
        ContentCandidate(
            candidate_id="answer",
            intent_id="intent:content",
            intent_kind="report_status",
            semantic_slots={"mode": "answer"},
            goal_id="stay-informed",
            goal_alignment=0.9,
            world_relevance=0.3,
            information_gain=0.2,
            confidence=0.9,
            uncertainty=0.1,
            resource_cost=0.1,
        ),
        ContentCandidate(
            candidate_id="ask",
            intent_id="intent:content",
            intent_kind="request_information",
            semantic_slots={"mode": "clarify"},
            goal_id="stay-informed",
            goal_alignment=0.6,
            world_relevance=0.9,
            information_gain=0.9,
            confidence=0.7,
            uncertainty=0.2,
            resource_cost=0.2,
        ),
    )


def test_content_selection_learns_context_conditioned_content() -> None:
    answer, ask = _candidates()
    certain = ContentSelectionContext(
        goal_residual=0.9,
        world_uncertainty=0.1,
        novelty=0.1,
        resource_budget=0.8,
    )
    uncertain = ContentSelectionContext(
        goal_residual=0.9,
        world_uncertainty=0.9,
        novelty=0.8,
        resource_budget=0.8,
    )
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

    certain_decision = selector.select((answer, ask), certain)
    uncertain_decision = selector.select((answer, ask), uncertain)

    assert certain_decision.selected.candidate_id == "answer"
    assert uncertain_decision.selected.candidate_id == "ask"
    selected_content = uncertain_decision.selected.to_content_plan()
    assert selected_content.intent_kind == "request_information"
    assert selected_content.semantic_slots == {"mode": "clarify"}
    assert selected_content.source_goal_id == "stay-informed"

    restored = ContentSelector.from_checkpoint(selector.checkpoint())
    assert restored.select((answer, ask), uncertain).selected == uncertain_decision.selected
