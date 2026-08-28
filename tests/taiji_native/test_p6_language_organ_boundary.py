from __future__ import annotations

import pytest

from taiji import (
    ContentCandidate,
    ContentPlan,
    ContentSelector,
    ExpressionPlan,
    ExternalTextDecoderLanguageOrgan,
    GenerationController,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageEmission,
    LanguageProviderArtifact,
    LanguageProviderArtifactRegistry,
    LanguageProviderCanaryGate,
    LanguageRealizationGate,
    LanguageRealizationValidator,
    LanguageTrainingCorpus,
    LanguageTrainingExample,
    NativeReadableTextLanguageOrgan,
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
        required_terms=("稳定",),
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


def test_native_readable_language_organ_forms_candidate_or_truthful_fallback() -> None:
    expression = _expression()
    organ = NativeReadableTextLanguageOrgan()

    candidate = organ.emit(
        expression.__class__(
            **{
                **expression.to_payload(),
                "fields": {
                    **expression.fields,
                    "surface_text": "当前状态稳定。",
                },
            }
        )
    )
    fallback = organ.emit(expression)

    assert candidate.backend == "native-readable"
    assert candidate.text_bytes == "当前状态稳定。".encode()
    assert fallback.backend == "native-readable"
    assert "稳定" in fallback.text_bytes.decode()
    restored = NativeReadableTextLanguageOrgan.from_checkpoint(organ.checkpoint())
    assert restored.checkpoint() == organ.checkpoint()


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


def test_language_training_corpus_keeps_train_and_holdout_disjoint() -> None:
    expression = _expression()
    train = LanguageTrainingExample(
        example_id="language-train-1",
        expression=expression,
        target_text="当前状态稳定。",
        split="train",
    )
    holdout_expression = GenerationController().plan_expression(
        ContentPlan(
            content_id="language:holdout:content",
            intent_id="language:holdout",
            intent_kind="render_alert",
            semantic_slots={"topic": "incident"},
            required_terms=("警告",),
        ),
        modality="text",
        channel="message",
    )
    holdout = LanguageTrainingExample(
        example_id="language-holdout-1",
        expression=holdout_expression,
        target_text="出现警告。",
        split="holdout",
    )
    corpus = LanguageTrainingCorpus(train=(train,), holdout=(holdout,))
    restored = LanguageTrainingCorpus.from_payload(corpus.to_payload())

    assert corpus.size == 2
    assert restored == corpus
    duplicate_holdout = LanguageTrainingExample(
        example_id=train.example_id,
        expression=holdout_expression,
        target_text=holdout.target_text,
        split="holdout",
    )
    with pytest.raises(ValueError, match="example IDs must be unique"):
        LanguageTrainingCorpus(train=(train,), holdout=(duplicate_holdout,))


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
    )
    adapter = TSKV8Adapter()
    adapter.attach_language_backend_registry(registry)
    adapter.attach_language_organ(guarded)
    emission = adapter.emit_language(expression)
    fallback_text = emission.text_bytes.decode("utf-8")

    assert emission.fallback_used is True
    assert emission.validation is not None
    assert emission.validation.accepted is False
    assert emission.validation.missing_terms == ("稳定",)
    assert emission.backend == "native-readable"
    assert "稳定" in fallback_text
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
    )

    emission = organ.emit(expression)

    assert emission.fallback_used is False
    assert emission.validation is not None
    assert emission.validation.accepted is True
    assert emission.validation.coverage == 1.0


def test_language_fallback_assigns_content_credit_and_replan_signal() -> None:
    class _ExternalDecoder:
        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del prompt, max_tokens, temperature
            return "操作员收到一条消息。"

    candidate = ContentCandidate(
        candidate_id="status",
        intent_id="language:intent",
        intent_kind="render_message",
        semantic_slots={"topic": "status"},
        required_terms=("稳定",),
        confidence=0.8,
    )
    registry = LanguageBackendRegistry.default()
    registry.register(
        LanguageBackendSpec(
            backend_id="mature-decoder-v1",
            family="external-decoder",
            training_contract="expression-to-text-v1",
        )
    )
    guarded = ValidatedLanguageOrgan(
        ExternalTextDecoderLanguageOrgan(
            _ExternalDecoder(),
            prompt_builder=lambda item: item.content_id,
            backend_id="mature-decoder-v1",
        )
    )
    adapter = TSKV8Adapter()
    adapter.attach_generation_controller(GenerationController())
    adapter.attach_content_selector(ContentSelector())
    adapter.attach_language_backend_registry(registry)
    adapter.attach_language_organ(guarded)
    adapter.observe(97, learn=False)
    adapter.select_content((candidate,))

    emission = adapter.emit_language()

    assert emission.fallback_used is True
    assert adapter.language_fallback_count == 1
    assert adapter.replan_required is True
    assert adapter.last_content_prediction_error is not None
    assert adapter.last_content_prediction_error > 0.0
    assert adapter._content_feedback_applied is True
    adapter.attach_language_organ(None)
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    assert restored.language_fallback_count == 1
    assert restored.replan_required is True
    assert restored.last_content_prediction_error == adapter.last_content_prediction_error


def test_language_fallback_replans_to_an_alternative_expression() -> None:
    class _ExternalDecoder:
        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del max_tokens, temperature
            return "当前状态稳定。" if "recovery" in prompt else "操作员收到一条消息。"

    candidates = (
        ContentCandidate(
            candidate_id="status",
            intent_id="language:intent",
            intent_kind="render_message",
            semantic_slots={"topic": "status"},
            required_terms=("稳定",),
            confidence=0.8,
        ),
        ContentCandidate(
            candidate_id="recovery",
            intent_id="language:intent",
            intent_kind="render_message",
            semantic_slots={"topic": "recovery"},
            required_terms=("稳定",),
            confidence=0.8,
        ),
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
    adapter.attach_generation_controller(GenerationController())
    adapter.attach_content_selector(ContentSelector())
    adapter.attach_language_backend_registry(registry)
    adapter.attach_language_organ(
        ValidatedLanguageOrgan(
            ExternalTextDecoderLanguageOrgan(
                _ExternalDecoder(),
                prompt_builder=lambda item: (
                    item.fields["semantic_slots"]["topic"]
                    if "semantic_slots" in item.fields
                    else item.content_id
                ),
                backend_id="mature-decoder-v1",
            )
        )
    )
    adapter.observe(97, learn=False)
    first = adapter.select_content(candidates)
    first_emission = adapter.emit_language()
    replanned = adapter.replan_content_after_language_fallback(candidates)
    second_expression = adapter.express_selected_content(modality="text", channel="message")
    second_emission = adapter.emit_language(second_expression)

    assert first.selected.candidate_id == "status"
    assert first_emission.fallback_used is True
    assert replanned.selected.candidate_id == "recovery"
    assert second_expression.fields["semantic_slots"] == {"topic": "recovery"}
    assert second_emission.fallback_used is False
    assert adapter.language_fallback_count == 1
    assert adapter.replan_required is False


def test_reset_dynamics_clears_language_replan_state() -> None:
    class _ExternalDecoder:
        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del prompt, max_tokens, temperature
            return "操作员收到一条消息。"

    expression = _expression()
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
    adapter.attach_language_organ(
        ValidatedLanguageOrgan(
            ExternalTextDecoderLanguageOrgan(
                _ExternalDecoder(),
                prompt_builder=lambda item: item.content_id,
                backend_id="mature-decoder-v1",
            )
        )
    )

    adapter.emit_language(expression)
    assert adapter.replan_required is True

    adapter.reset_dynamics()

    assert adapter.replan_required is False
    assert adapter.language_fallback_count == 0


def test_provider_artifact_modes_and_checkpoint_are_explicit() -> None:
    guarded = LanguageProviderArtifact(
        artifact_id="qwen-lora-v1",
        backend_id="mature-decoder-v1",
        mode="guarded",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path="providers/qwen-lora-v1",
        training_corpus="reports/train-holdout.json",
        training_report="reports/trainer.json",
        safety_report="reports/safety.json",
    )
    restored_manifest = LanguageProviderArtifact.from_payload(guarded.to_payload())
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
    adapter.attach_language_provider_artifact(guarded)
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())

    assert restored_manifest == guarded
    assert restored.language_provider_artifact == guarded
    with pytest.raises(ValueError, match="guarded provider artifacts must remain opt-in"):
        LanguageProviderArtifact(
            artifact_id="invalid",
            backend_id="mature-decoder-v1",
            mode="guarded",
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            adapter_path="providers/invalid",
            default_enabled=True,
        )


def test_provider_artifact_content_address_and_first_chat_canary_are_deterministic() -> None:
    from pathlib import Path

    from taiji import (
        language_provider_artifact_digest,
        language_provider_content_digest,
    )

    anchor = Path(__file__).resolve()
    digest = language_provider_content_digest(anchor)
    artifact = LanguageProviderArtifact(
        artifact_id="qwen-guarded-addressed-v1",
        backend_id="mature-decoder-v1",
        mode="guarded",
        base_model=str(anchor),
        adapter_path=str(anchor),
        training_corpus=str(anchor),
        training_report=str(anchor),
        safety_report=str(anchor),
        content_digests=(
            ("base_model", digest),
            ("adapter", digest),
            ("training_corpus", digest),
            ("training_report", digest),
            ("safety_report", digest),
        ),
        expires_at=4_000_000_000.0,
    )
    assert artifact.artifact_digest == language_provider_artifact_digest(artifact)
    assert LanguageProviderArtifact.from_payload(artifact.to_payload()) == artifact

    class _CanaryDecoder:
        def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
            del max_tokens, temperature
            if "database-status" in prompt:
                return "数据库运行正常。"
            return "接口已经恢复。"

    organ = ExternalTextDecoderLanguageOrgan(
        _CanaryDecoder(),
        prompt_builder=lambda expression: expression.content_id,
        backend_id="mature-decoder-v1",
    )
    report = LanguageProviderCanaryGate().evaluate(organ)
    assert report["format"] == "taiji-language-provider-canary-v1"
    assert report["metrics"]["required_term_coverage"] == 1.0
    assert report["gate"]["passed"] is True


def test_provider_artifact_registry_tracks_rotation_and_checkpoint() -> None:
    def artifact(artifact_id: str) -> LanguageProviderArtifact:
        return LanguageProviderArtifact(
            artifact_id=artifact_id,
            backend_id="mature-decoder-v1",
            mode="guarded",
            base_model=f"models/{artifact_id}",
            adapter_path=f"adapters/{artifact_id}",
            training_corpus=f"reports/{artifact_id}-corpus.json",
            training_report=f"reports/{artifact_id}-training.json",
            safety_report=f"reports/{artifact_id}-safety.json",
        )

    first = artifact("provider-v1")
    second = artifact("provider-v2")
    registry = (
        LanguageProviderArtifactRegistry()
        .with_artifact(first, allow=True)
        .with_artifact(second, allow=True)
        .activate(first.artifact_id)
    )
    relocated = LanguageProviderArtifact(
        artifact_id=first.artifact_id,
        backend_id=first.backend_id,
        mode=first.mode,
        base_model="relocated/model",
        adapter_path="relocated/adapter",
        training_corpus="relocated/corpus.json",
        training_report="relocated/training.json",
        safety_report="relocated/safety.json",
    )
    registry.require_allowed(relocated)
    rotated = registry.activate(second.artifact_id)

    assert rotated.active_artifact_id == second.artifact_id
    assert rotated.previous_artifact_id == first.artifact_id
    assert rotated.active_artifact == second
    assert rotated.previous_artifact == first
    assert LanguageProviderArtifactRegistry.from_checkpoint(rotated.checkpoint()) == rotated
    assert rotated.rollback().active_artifact_id == first.artifact_id

    not_allowlisted = LanguageProviderArtifactRegistry().with_artifact(first)
    with pytest.raises(PermissionError, match="not allowlisted"):
        not_allowlisted.activate(first.artifact_id)


def test_expression_to_text_gate_requires_quality_rollback_and_checkpoint() -> None:
    def with_surface(expression: ExpressionPlan, text: str) -> ExpressionPlan:
        payload = expression.to_payload()
        payload["fields"] = {**expression.fields, "surface_text": text}
        return ExpressionPlan.from_payload(payload)

    train_expression = with_surface(_expression(), "当前状态稳定。")
    holdout_expression = GenerationController().plan_expression(
        ContentPlan(
            content_id="language:gate:holdout",
            intent_id="language:gate:holdout",
            intent_kind="render_alert",
            semantic_slots={"topic": "incident"},
            required_terms=("警告",),
        ),
        modality="text",
        channel="message",
    )
    holdout_expression = with_surface(holdout_expression, "检测到警告。")
    corpus = LanguageTrainingCorpus(
        train=(
            LanguageTrainingExample(
                example_id="language:gate:train",
                expression=train_expression,
                target_text="当前状态稳定。",
                split="train",
            ),
        ),
        holdout=(
            LanguageTrainingExample(
                example_id="language:gate:holdout",
                expression=holdout_expression,
                target_text="检测到警告。",
                split="holdout",
            ),
        ),
    )
    organ = NativeReadableTextLanguageOrgan()
    gate = LanguageRealizationGate()
    report = gate.evaluate(
        organ,
        corpus,
        rollback_organ=NativeReadableTextLanguageOrgan(),
        rollback_reference_organ=NativeReadableTextLanguageOrgan(),
        checkpoint_loader=NativeReadableTextLanguageOrgan.from_checkpoint,
    )

    assert report["gate"]["passed"] is True
    assert report["holdout"]["required_term_coverage"] == 1.0
    assert report["rollback"]["outputs_match_reference"] is True
    assert report["checkpoint"]["outputs_match"] is True

    structured_report = gate.evaluate(
        StructuredTextLanguageOrgan(),
        corpus,
        rollback_organ=StructuredTextLanguageOrgan(),
        rollback_reference_organ=StructuredTextLanguageOrgan(),
        checkpoint_loader=StructuredTextLanguageOrgan.from_checkpoint,
    )
    assert structured_report["gate"]["passed"] is False
