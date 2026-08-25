from __future__ import annotations

from taiji import ActionIntent, ContentPlan, GenerationController


def test_one_content_plan_is_shared_by_tool_and_text_expression_organs() -> None:
    intent = ActionIntent(
        intent_id="episode:intent:cross-organ",
        kind="report_status",
        parameters={"topic": "weather", "detail": {"level": "summary"}},
        source_goal_id="stay-informed",
        expected_outcome="user receives status",
        confidence=0.85,
        tick=4,
    )
    controller = GenerationController()
    content = controller.plan_content(intent)
    tool_expression = controller.plan_expression(content, modality="tool", channel="rpc")
    text_expression = controller.plan_expression(content, modality="text", channel="message")

    assert isinstance(content, ContentPlan)
    assert tool_expression.content_id == text_expression.content_id == content.content_id
    assert tool_expression.modality == "tool"
    assert text_expression.modality == "text"
    assert tool_expression.channel == "rpc"
    assert text_expression.channel == "message"
    assert tool_expression.fields == text_expression.fields
    assert tool_expression.fields["semantic_slots"] == intent.parameters
    assert tool_expression.confidence == text_expression.confidence == intent.confidence


def test_expression_organ_cannot_change_content_identity_or_goal_provenance() -> None:
    intent = ActionIntent(
        intent_id="episode:intent:provenance",
        kind="notify",
        parameters={"message": "done"},
        source_goal_id="finish-task",
        confidence=0.7,
    )
    controller = GenerationController()
    content = controller.plan_content(intent)
    expression = controller.plan_expression(content, modality="text", channel="message")

    assert expression.content_id == content.content_id
    assert expression.fields["intent_kind"] == content.intent_kind
    assert content.source_goal_id == "finish-task"
    assert "source_goal_id" not in expression.fields

