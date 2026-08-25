from __future__ import annotations

from taiji import ContentCandidate, ContentSelector, TSKV8Adapter


def _candidates() -> tuple[ContentCandidate, ContentCandidate]:
    return (
        ContentCandidate(
            candidate_id="answer",
            intent_id="intent:content",
            intent_kind="report_status",
            semantic_slots={"mode": "answer"},
            goal_id="stay-informed",
            goal_alignment=0.8,
            world_relevance=0.2,
            information_gain=0.1,
            confidence=0.8,
            uncertainty=0.1,
            resource_cost=0.1,
        ),
        ContentCandidate(
            candidate_id="ask",
            intent_id="intent:content",
            intent_kind="request_information",
            semantic_slots={"mode": "clarify"},
            goal_id="stay-informed",
            goal_alignment=0.5,
            world_relevance=0.8,
            information_gain=0.9,
            confidence=0.7,
            uncertainty=0.2,
            resource_cost=0.2,
        ),
    )


def test_real_outcomes_update_selected_content_utility_once() -> None:
    candidates = _candidates()
    adapter = TSKV8Adapter()
    selector = ContentSelector()
    adapter.attach_content_selector(selector)
    adapter.observe(97, learn=False)

    first = adapter.select_content(candidates, novelty=0.8, resource_budget=0.8)
    assert first.selected.candidate_id == "answer"
    adapter.act((10,), sample=False)
    adapter.settle_action(-1.0, learn=False, success=False)
    first_error = adapter.last_content_prediction_error
    after_failure = selector.scores(candidates, first.context)
    adapter.observe(98, learn=False)

    second = adapter.select_content(candidates, novelty=0.8, resource_budget=0.8)
    assert second.selected.candidate_id == "ask"
    adapter.act((10,), sample=False)
    adapter.settle_action(1.0, learn=False, success=True)
    second_error = adapter.last_content_prediction_error
    after_success = selector.scores(candidates, second.context)

    assert first_error is not None and first_error > 0.0
    assert second_error is not None and second_error < 0.0
    assert after_failure[0] < after_failure[1]
    assert after_success[1] > after_success[0]
    assert selector.training_steps == 2
    assert adapter._content_feedback_applied is True

    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    assert restored.last_content_selection == second
    assert restored.last_content_prediction_error == second_error
    assert restored._content_feedback_applied is True
