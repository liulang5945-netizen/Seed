"""Seed-side runtime assembly for Taiji's replaceable language organ.

This module is deliberately outside ``taiji/``.  It may lazily import an
optional mature decoder, but the decoder only receives an expression emitted
by Taiji and can never become the cognitive owner.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from taiji import (
    ExternalTextDecoderLanguageOrgan,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageProviderArtifact,
    LanguageRealizationValidator,
    StructuredTextLanguageOrgan,
    TSKV8Adapter,
    ValidatedLanguageOrgan,
)

from .config import LanguageProviderConfig

logger = logging.getLogger("Seed.LanguageProvider")


@dataclass(frozen=True)
class LanguageProviderStatus:
    """Observable provider state returned by Seed runtime health/status APIs."""

    mode: str
    state: str
    provider: str
    backend_id: str
    artifact_id: str
    reason_code: str = ""
    reason: str = ""
    rollback: str = "structured-stub"

    def to_dict(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "state": self.state,
            "provider": self.provider,
            "backend_id": self.backend_id,
            "artifact_id": self.artifact_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "rollback": self.rollback,
        }


class _QwenTextDecoder:
    """Optional local decoder kept at the Seed integration edge."""

    def __init__(self, model_dir: Path) -> None:
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "Qwen provider requires the optional transformers integration dependency"
            ) from exc
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.float32,
        )
        self.model.eval()

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        del temperature
        formatted_prompt = prompt
        if getattr(self.tokenizer, "chat_template", None):
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = self.tokenizer(formatted_prompt, return_tensors="pt")
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        return self.tokenizer.decode(
            generated[0, prompt_length:], skip_special_tokens=True
        ).strip()


def _prompt(expression: Any) -> str:
    payload = {
        "intent_kind": expression.fields.get("intent_kind", ""),
        "channel": expression.channel,
        "semantic_slots": expression.fields.get("semantic_slots", {}),
        "expected_outcome": expression.fields.get("expected_outcome", ""),
    }
    return (
        "请把下面的结构化表达实现为简洁、自然的中文消息。"
        "必须保留语义槽位中的全部关键值，不要添加输入中没有的事实。"
        "不要输出字段名或 JSON，只输出最终消息：\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _resolve(path_value: str, root: Path | None) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (root or Path.cwd()) / path


def build_provider_artifact(config: LanguageProviderConfig) -> LanguageProviderArtifact:
    if config.mode == "structured":
        raise ValueError("structured mode does not have an external provider artifact")
    adapter_path = None if config.mode == "raw" else config.adapter_dir
    return LanguageProviderArtifact(
        artifact_id=config.artifact_id,
        backend_id=config.backend_id,
        mode=config.mode,
        base_model=config.model_dir,
        adapter_path=adapter_path,
        training_corpus=config.training_corpus or None,
        training_report=config.training_report or None,
        safety_report=config.safety_report or None,
        default_enabled=False,
    )


def load_qwen_language_provider(
    adapter: TSKV8Adapter,
    artifact: LanguageProviderArtifact,
    *,
    model_dir: Path,
    artifact_root: Path | None = None,
    max_tokens: int = 24,
    temperature: float = 0.0,
) -> _QwenTextDecoder:
    """Load one manifest-selected Qwen provider and attach its organ."""

    if not model_dir.is_dir():
        raise FileNotFoundError(f"provider model directory not found: {model_dir}")
    decoder = _QwenTextDecoder(model_dir)
    if artifact.mode in {"lora", "guarded"}:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "LoRA provider requires the optional peft integration dependency"
            ) from exc
        adapter_path = _resolve(str(artifact.adapter_path or ""), artifact_root)
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"provider adapter directory not found: {adapter_path}")
        decoder.model = PeftModel.from_pretrained(
            decoder.model,
            adapter_path,
            is_trainable=False,
        )
        decoder.model.eval()
    organ: object = ExternalTextDecoderLanguageOrgan(
        decoder,
        prompt_builder=_prompt,
        backend_id=artifact.backend_id,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if artifact.mode == "guarded":
        if artifact.safety_report is None:
            raise ValueError("guarded provider artifact requires a safety_report")
        safety_report = _resolve(artifact.safety_report, artifact_root)
        if not safety_report.is_file():
            raise FileNotFoundError(f"guarded provider safety report not found: {safety_report}")
        organ = ValidatedLanguageOrgan(
            organ,
            validator=LanguageRealizationValidator(minimum_coverage=1.0),
        )
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
    adapter.attach_language_organ(organ)
    return decoder


def attach_structured_language_provider(adapter: TSKV8Adapter) -> None:
    """Attach the deterministic terminal fallback and clear external metadata."""

    adapter.attach_language_organ(None)
    adapter.attach_language_backend_registry(LanguageBackendRegistry.default())
    adapter.attach_language_provider_artifact(None)
    adapter.attach_language_organ(StructuredTextLanguageOrgan())


def activate_language_provider(
    adapter: TSKV8Adapter,
    config: LanguageProviderConfig,
) -> tuple[LanguageProviderStatus, Any | None]:
    """Activate an explicit provider, or safely roll back to structured stub."""

    if not isinstance(adapter, TSKV8Adapter):
        raise TypeError("language provider activation requires a TSKV8Adapter")
    if not isinstance(config, LanguageProviderConfig):
        raise TypeError("language provider config must be a LanguageProviderConfig")
    if config.mode == "structured":
        attach_structured_language_provider(adapter)
        return (
            LanguageProviderStatus(
                mode="structured",
                state="active",
                provider="structured",
                backend_id=StructuredTextLanguageOrgan.BACKEND_ID,
                artifact_id="structured-stub",
            ),
            None,
        )

    artifact = build_provider_artifact(config)
    try:
        if config.provider != "qwen":
            raise ValueError(f"unsupported Seed language provider: {config.provider}")
        root = Path(config.artifact_root) if config.artifact_root else None
        model_dir = _resolve(config.model_dir, root)
        decoder = load_qwen_language_provider(
            adapter,
            artifact,
            model_dir=model_dir,
            artifact_root=root,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        return (
            LanguageProviderStatus(
                mode=config.mode,
                state="active",
                provider=config.provider,
                backend_id=artifact.backend_id,
                artifact_id=artifact.artifact_id,
            ),
            decoder,
        )
    except FileNotFoundError as exc:
        reason_code = "provider_missing"
        reason = str(exc)
    except (ImportError, ModuleNotFoundError) as exc:
        reason_code = "provider_dependency_missing"
        reason = str(exc)
    except (KeyError, ValueError) as exc:
        reason_code = "provider_manifest_mismatch"
        reason = str(exc)
    except Exception as exc:  # noqa: BLE001 - boundary must preserve fallback
        reason_code = "provider_load_failed"
        reason = str(exc)
    logger.warning("language provider rolled back to structured stub: %s", reason)
    attach_structured_language_provider(adapter)
    return (
        LanguageProviderStatus(
            mode=config.mode,
            state="fallback",
            provider=config.provider,
            backend_id=config.backend_id,
            artifact_id=config.artifact_id,
            reason_code=reason_code,
            reason=reason,
        ),
        None,
    )
