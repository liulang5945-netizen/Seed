from __future__ import annotations

import pytest

from taiji import (
    ContentPlan,
    GenerationController,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageEmission,
    LanguageTrainingExample,
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


def test_language_backend_registry_and_training_contract_are_model_agnostic() -> None:
    expression = _expression()
    example = LanguageTrainingExample(
        example_id="language-example-1",
        expression=expression,
        target_text="当前状态稳定。",
        split="holdout",
        weight=0.75,
        provenance="human-reviewed",
    )
    restored_example = LanguageTrainingExample.from_payload(example.to_payload())
    registry = LanguageBackendRegistry.default()
    registry.register(
        LanguageBackendSpec(
            backend_id="mature-decoder-v1",
            family="external-decoder",
            training_contract="expression-to-text-v1",
        )
    )
    restored_registry = LanguageBackendRegistry.from_checkpoint(registry.checkpoint())

    assert restored_example == example
    assert example.target_bytes == "当前状态稳定。".encode()
    assert restored_registry.get("mature-decoder-v1").training_contract == "expression-to-text-v1"
    assert restored_registry.validate(StructuredTextLanguageOrgan()).backend_id == "structured-stub"
    with pytest.raises(ValueError, match="cannot own Taiji cognition"):
        LanguageBackendSpec(
            backend_id="invalid-cognitive-backend",
            family="external-decoder",
            training_contract="expression-to-text-v1",
            owns_cognition=True,
        )
