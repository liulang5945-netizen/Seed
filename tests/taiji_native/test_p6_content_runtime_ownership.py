from __future__ import annotations

from taiji import (
    ContentCandidate,
    ContentSelectionContext,
    ContentSelector,
    ContentTrainingExample,
    GenerationController,
    Goal,
    TSKV8Adapter,
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


def _selector() -> ContentSelector:
    answer, ask = _candidates()
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


def test_adapter_owns_content_selection_and_expression_checkpoint() -> None:
    candidates = _candidates()
    adapter = TSKV8Adapter()
    adapter.attach_content_selector(_selector())
    adapter.attach_generation_controller(GenerationController())
    adapter.set_goals((Goal("stay-informed", "get current information", priority=1.0),))
    adapter.observe(97, learn=False)

    decision = adapter.select_content(candidates, novelty=0.8, resource_budget=0.8)
    expression = adapter.express_selected_content(modality="text", channel="message")

    assert decision.selected.candidate_id == "ask"
    assert decision.context.goal_residual == 1.0
    assert expression.content_id == adapter.selected_content_plan().content_id
    assert expression.fields["semantic_slots"] == {"mode": "clarify"}

    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    assert restored.last_content_selection == decision
    assert restored.express_selected_content(modality="text", channel="message") == expression
