from __future__ import annotations

import pytest

from taiji import (
    ContentPlan,
    GenerationController,
    LanguageEmission,
    StructuredTextLanguageOrgan,
    TextExpressionCodec,
    TSKV8Adapter,
)


def _expression():
    content = ContentPlan(
        content_id="language:intent:content",
        intent_id="language:intent",
        intent_kind="render_message",
        semantic_slots={"topic": "status", "format": "concise"},
        source_goal_id="language-goal",
        expected_outcome="user receives a message",
        confidence=0.8,
        provenance="selected",
        tick=0,
    )
    return GenerationController().plan_expression(
        content,
        modality="text",
        channel="message",
    )


def test_structured_language_organ_is_a_terminal_replaceable_stub() -> None:
    expression = _expression()
    organ = StructuredTextLanguageOrgan()

    emission = organ.emit(expression)

    assert isinstance(emission, LanguageEmission)
    assert emission.backend == "structured-stub"
    assert TextExpressionCodec.decode(emission.text_bytes) == expression
    assert organ.checkpoint()["backend"] == "structured-stub"


def test_language_organ_lesion_and_native_checkpoint_preserve_boundary() -> None:
    expression = _expression()
    adapter = TSKV8Adapter()
    adapter.attach_generation_controller(GenerationController())
    with pytest.raises(RuntimeError, match="language organ is not attached"):
        adapter.emit_language(expression)

    parameters_before = adapter.parameter_tensors()
    adapter.attach_language_organ(StructuredTextLanguageOrgan())
    action_before = adapter.cognitive_snapshot().action_intent
    emission = adapter.emit_language(expression)
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())

    assert action_before is None
    assert adapter.cognitive_snapshot().action_intent is None
    assert all(
        current.data_ptr() == before.data_ptr()
        for current, before in zip(adapter.parameter_tensors(), parameters_before, strict=True)
    )
    assert restored.last_language_emission == emission
    assert restored.last_language_emission is not None
    assert restored.last_language_emission.backend == "structured-stub"
    assert TextExpressionCodec.decode(restored.last_language_emission.text_bytes) == expression
