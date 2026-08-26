from __future__ import annotations

from taiji import (
    TEXT_EXPRESSION_CODEC_FORMAT,
    ContentPlan,
    ExpressionPlan,
    GenerationController,
    TextExpressionCodec,
)


def test_text_organ_codec_preserves_structured_expression_semantics() -> None:
    content = ContentPlan(
        content_id="holdout:intent:content",
        intent_id="holdout:intent",
        intent_kind="forecast_digest",
        semantic_slots={"format": "digest", "regions": ["east", "south"]},
        source_goal_id="stay-informed",
        expected_outcome="user receives forecast",
        confidence=0.68,
        provenance="selected",
        tick=4,
    )
    expression = GenerationController().plan_expression(
        content,
        modality="text",
        channel="message",
    )
    encoded = TextExpressionCodec.encode(expression)
    restored = TextExpressionCodec.decode(encoded)

    assert isinstance(expression, ExpressionPlan)
    assert expression.source_goal_id == "stay-informed"
    assert restored == expression
    assert restored.fields["semantic_slots"] == content.semantic_slots
    assert restored.confidence == content.confidence
    assert restored.source_goal_id == content.source_goal_id
    assert TEXT_EXPRESSION_CODEC_FORMAT.encode("utf-8") in encoded


def test_generation_checkpoint_accepts_text_codec_boundary() -> None:
    controller = GenerationController()
    restored = GenerationController.from_checkpoint(controller.checkpoint())
    assert restored.checkpoint() == controller.checkpoint()
