from __future__ import annotations

import pytest

from taiji import (
    ContentPlan,
    ExternalTextDecoderLanguageOrgan,
    GenerationController,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageEmission,
    LanguageRealizationValidator,
    LanguageTrainingExample,
    StructuredTextLanguageOrgan,
    TextExpressionCodec,
    TSKV8Adapter,
    ValidatedLanguageOrgan,
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


def test_external_decoder_realization_and_lesion_stay_outside_taiji_core() -> None:
    class _ExternalDecoder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, float]] = []

        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            self.calls.append((prompt, max_tokens, temperature))
            return "外部 decoder 已完成表达。"

    decoder = _ExternalDecoder()
    organ = ExternalTextDecoderLanguageOrgan(
        decoder,
        prompt_builder=lambda expression: f"render:{expression.channel}:{expression.content_id}",
        max_tokens=32,
        temperature=0.1,
    )
    registry = LanguageBackendRegistry.default()
    registry.register(
        LanguageBackendSpec(
            backend_id="mature-decoder-v1",
            family="external-decoder",
            training_contract="expression-to-text-v1",
        )
    )
    adapter = TSKV8Adapter()
    adapter.attach_language_backend_registry(registry)
    adapter.attach_language_organ(organ)
    emission = adapter.emit_language(_expression())
    restored_organ = ExternalTextDecoderLanguageOrgan.from_checkpoint(
        organ.checkpoint(),
        decoder,
        prompt_builder=organ.prompt_builder,
    )
    restored_emission = restored_organ.emit(_expression())
    adapter.attach_language_organ(None)

    assert emission.text_bytes == "外部 decoder 已完成表达。".encode()
    assert restored_emission.text_bytes == emission.text_bytes
    assert decoder.calls[0][0].startswith("render:message:")
    assert decoder.calls[0][1:] == (32, 0.1)
    assert adapter.cognitive_snapshot().action_intent is None
    with pytest.raises(RuntimeError, match="language organ is not attached"):
        adapter.emit_language(_expression())


def test_realization_validator_falls_back_without_losing_expression_semantics() -> None:
    class _ExternalDecoder:
        def __init__(self, text: str) -> None:
            self.text = text

        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del prompt, max_tokens, temperature
            return self.text

    expression = _expression()
    registry = LanguageBackendRegistry.default()
    registry.register(
        LanguageBackendSpec(
            backend_id="mature-decoder-v1",
            family="external-decoder",
            training_contract="expression-to-text-v1",
        )
    )
    validator = LanguageRealizationValidator(minimum_coverage=1.0)
    guarded = ValidatedLanguageOrgan(
        ExternalTextDecoderLanguageOrgan(
            _ExternalDecoder("操作员收到一条消息。"),
            prompt_builder=lambda item: item.content_id,
            backend_id="mature-decoder-v1",
        ),
        validator=validator,
        required_terms_builder=lambda item: ("稳定",),
    )
    adapter = TSKV8Adapter()
    adapter.attach_language_backend_registry(registry)
    adapter.attach_language_organ(guarded)
    emission = adapter.emit_language(expression)
    decoded_fallback = TextExpressionCodec.decode(emission.text_bytes)

    assert emission.fallback_used is True
    assert emission.validation is not None
    assert emission.validation.accepted is False
    assert emission.validation.missing_terms == ("稳定",)
    assert emission.backend == "structured-stub"
    assert decoded_fallback == expression
    assert adapter.cognitive_snapshot().action_intent is None


def test_realization_validator_accepts_semantically_complete_external_text() -> None:
    expression = _expression()
    organ = ValidatedLanguageOrgan(
        ExternalTextDecoderLanguageOrgan(
            type(
                "Decoder",
                (),
                {"generate": lambda self, prompt, *, max_tokens, temperature: "当前状态稳定。"},
            )(),
            prompt_builder=lambda item: item.content_id,
            backend_id="mature-decoder-v1",
        ),
        required_terms_builder=lambda item: ("稳定",),
    )

    emission = organ.emit(expression)

    assert emission.fallback_used is False
    assert emission.validation is not None
    assert emission.validation.accepted is True
    assert emission.validation.coverage == 1.0
