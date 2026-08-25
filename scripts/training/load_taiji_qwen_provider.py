"""Config-driven Qwen provider loader at the external integration edge."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from taiji import (
    ExternalTextDecoderLanguageOrgan,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageProviderArtifact,
    LanguageRealizationValidator,
    TSKV8Adapter,
    ValidatedLanguageOrgan,
)


def load_qwen_language_organ(
    artifact: LanguageProviderArtifact,
    *,
    model_dir: Path,
    prompt_builder: Callable[[Any], str],
    artifact_root: Path | None = None,
) -> tuple[Any, object]:
    """Load raw/LoRA/guarded Qwen from one audited provider artifact.

    Transformers and PEFT stay in this integration-edge module.  The returned
    organ still receives only a Taiji-owned ``ExpressionPlan``.
    """

    if not isinstance(artifact, LanguageProviderArtifact):
        raise TypeError("artifact must be a LanguageProviderArtifact")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Qwen model directory not found: {model_dir}")
    from scripts.training.eval_taiji_p6_qwen_provider import QwenTextDecoder

    decoder = QwenTextDecoder(model_dir)
    if artifact.mode in {"lora", "guarded"}:
        from peft import PeftModel

        adapter_path = Path(artifact.adapter_path or "")
        if not adapter_path.is_absolute():
            adapter_path = (artifact_root or Path.cwd()) / adapter_path
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"provider adapter directory not found: {adapter_path}")
        decoder.model = PeftModel.from_pretrained(
            decoder.model,
            adapter_path,
            is_trainable=False,
        )
        decoder.model.eval()
    primary = ExternalTextDecoderLanguageOrgan(
        decoder,
        prompt_builder=prompt_builder,
        backend_id=artifact.backend_id,
        max_tokens=24,
        temperature=0.0,
    )
    if artifact.mode == "guarded":
        if artifact.safety_report is None:
            raise ValueError("guarded provider artifact requires a safety_report")
        return decoder, ValidatedLanguageOrgan(
            primary,
            validator=LanguageRealizationValidator(minimum_coverage=1.0),
        )
    return decoder, primary


def attach_qwen_language_provider(
    adapter: TSKV8Adapter,
    artifact: LanguageProviderArtifact,
    *,
    model_dir: Path,
    prompt_builder: Callable[[Any], str],
    artifact_root: Path | None = None,
) -> Any:
    """Load one manifest-selected organ and attach its auditable metadata."""

    if not isinstance(adapter, TSKV8Adapter):
        raise TypeError("adapter must be a TSKV8Adapter")
    registry = LanguageBackendRegistry.default()
    registry.register(
        LanguageBackendSpec(
            backend_id=artifact.backend_id,
            family=f"external-causal-decoder-{artifact.mode}",
            training_contract="expression-to-text-v1",
            supports_training=artifact.mode != "raw",
        )
    )
    adapter.attach_language_backend_registry(registry)
    adapter.attach_language_provider_artifact(artifact)
    decoder, organ = load_qwen_language_organ(
        artifact,
        model_dir=model_dir,
        prompt_builder=prompt_builder,
        artifact_root=artifact_root,
    )
    adapter.attach_language_organ(organ)
    return decoder
