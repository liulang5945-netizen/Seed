"""Seed-side runtime assembly for Taiji's replaceable language organ.

This module is deliberately outside ``taiji/``.  It may lazily import an
optional mature decoder, but the decoder only receives an expression emitted
by Taiji and can never become the cognitive owner.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from taiji import (
    LANGUAGE_PROVIDER_CONTENT_ADDRESS_FORMAT,
    LANGUAGE_REALIZATION_GATE_FORMAT,
    ExternalTextDecoderLanguageOrgan,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageOrgan,
    LanguageProviderArtifact,
    LanguageProviderArtifactRegistry,
    LanguageProviderCanaryGate,
    LanguageRealizationValidator,
    NativeReadableTextLanguageOrgan,
    StructuredTextLanguageOrgan,
    TSKV8Adapter,
    ValidatedLanguageOrgan,
    language_provider_artifact_digest,
    language_provider_content_digest,
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
    rollback: str = NativeReadableTextLanguageOrgan.BACKEND_ID
    chat_enabled: bool = False

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
            "chat_enabled": "true" if self.chat_enabled else "false",
        }


class _ProductChatGateError(ValueError):
    """Raised when an external provider is not admitted to product chat."""

    def __init__(self, message: str, *, reason_code: str = "chat_gate_failed") -> None:
        super().__init__(message)
        self.reason_code = reason_code


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
        return cast(
            str,
            self.tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip(),
        )


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


def _load_product_chat_report(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise _ProductChatGateError(f"product chat {label} report not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _ProductChatGateError(f"product chat {label} report is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise _ProductChatGateError(f"product chat {label} report must be a JSON object")
    return payload


def _realization_gate_passed(report: Mapping[str, Any]) -> bool:
    if report.get("format") != LANGUAGE_REALIZATION_GATE_FORMAT:
        return False
    corpus = report.get("corpus")
    train = report.get("train")
    holdout = report.get("holdout")
    rollback = report.get("rollback")
    checkpoint = report.get("checkpoint")
    gate = report.get("gate")
    if not isinstance(corpus, Mapping):
        return False
    if not isinstance(train, Mapping):
        return False
    if not isinstance(holdout, Mapping):
        return False
    if not isinstance(rollback, Mapping):
        return False
    if not isinstance(checkpoint, Mapping):
        return False
    if not isinstance(gate, Mapping):
        return False
    for split in (train, holdout):
        if not (
            split.get("passed") is True
            and split.get("output_nonempty_rate") == 1.0
            and split.get("readable_rate") == 1.0
            and float(split.get("required_term_coverage", 0.0) or 0.0) >= 1.0
            and split.get("structured_leakage_free_rate") == 1.0
            and split.get("fallback_rate") == 0.0
        ):
            return False
    return bool(
        corpus.get("round_trip") is True
        and corpus.get("split_disjoint") is True
        and rollback.get("checked") is True
        and rollback.get("outputs_match_reference") is True
        and checkpoint.get("checked") is True
        and checkpoint.get("outputs_match") is True
        and gate.get("passed") is True
    )


def _validate_product_chat_gate(
    artifact: LanguageProviderArtifact,
    *,
    artifact_root: Path | None,
) -> None:
    if artifact.mode != "guarded":
        raise _ProductChatGateError("product chat requires guarded provider mode")
    _verify_product_chat_artifact(artifact, artifact_root=artifact_root)
    if artifact.training_report is None:
        raise _ProductChatGateError("product chat requires a passing realization gate report")
    training_report = _load_product_chat_report(
        _resolve(artifact.training_report, artifact_root), "training"
    )
    training = training_report.get("training")
    realization_report = training_report.get("expression_to_text_gate")
    selected_report: Mapping[str, Any]
    if isinstance(realization_report, Mapping):
        selected_report = realization_report
    else:
        selected_report = training_report
    if not isinstance(training, Mapping) or training.get("training_applied") is not True:
        raise _ProductChatGateError("product chat training report does not prove applied training")
    if not _realization_gate_passed(selected_report):
        raise _ProductChatGateError(
            "product chat realization gate must pass semantic, readable, rollback, and checkpoint checks"
        )
    if artifact.safety_report is None:
        raise _ProductChatGateError("guarded product chat requires a safety report")
    safety_report = _load_product_chat_report(
        _resolve(artifact.safety_report, artifact_root), "safety"
    )
    adapted = safety_report.get("adapted")
    rollback = safety_report.get("rollback")
    gate = safety_report.get("gate")
    if not (
        isinstance(adapted, Mapping)
        and isinstance(rollback, Mapping)
        and isinstance(gate, Mapping)
        and gate.get("passed") is True
        and adapted.get("safe_realization_rate") == 1.0
        and rollback.get("outputs_match_raw") is True
    ):
        raise _ProductChatGateError("product chat safety report has not passed")


def _verify_product_chat_artifact(
    artifact: LanguageProviderArtifact,
    *,
    artifact_root: Path | None,
    now: float | None = None,
) -> dict[str, Any]:
    """Verify immutable provider content before any external model is loaded."""

    if artifact.content_address_format != LANGUAGE_PROVIDER_CONTENT_ADDRESS_FORMAT:
        raise _ProductChatGateError(
            "product chat requires a content-addressed provider artifact",
            reason_code="chat_artifact_invalid",
        )
    if artifact.canary_id != LanguageProviderCanaryGate.CANARY_ID:
        raise _ProductChatGateError(
            "product chat artifact canary contract is not supported",
            reason_code="chat_artifact_invalid",
        )
    if not artifact.content_digests or not artifact.artifact_digest:
        raise _ProductChatGateError(
            "product chat requires provider content digests and an artifact digest",
            reason_code="chat_artifact_missing",
        )
    if artifact.expires_at is None:
        raise _ProductChatGateError(
            "product chat requires an artifact expiry time",
            reason_code="chat_artifact_missing",
        )
    current_time = time.time() if now is None else float(now)
    if not math.isfinite(current_time) or current_time >= artifact.expires_at:
        raise _ProductChatGateError(
            "product chat provider artifact has expired",
            reason_code="chat_artifact_expired",
        )

    expected = dict(artifact.content_digests)
    paths = {
        "base_model": artifact.base_model,
        "adapter": artifact.adapter_path,
        "training_corpus": artifact.training_corpus,
        "training_report": artifact.training_report,
        "safety_report": artifact.safety_report,
    }
    required_roles = (
        "base_model",
        "adapter",
        "training_corpus",
        "training_report",
        "safety_report",
    )
    actual: dict[str, str] = {}
    for role in required_roles:
        path_value = paths[role]
        if not path_value:
            raise _ProductChatGateError(
                f"product chat artifact is missing the {role} path",
                reason_code="chat_artifact_invalid",
            )
        path = _resolve(path_value, artifact_root)
        try:
            actual[role] = language_provider_content_digest(path)
        except (OSError, ValueError) as exc:
            raise _ProductChatGateError(
                f"product chat artifact {role} content cannot be addressed: {path}",
                reason_code="chat_artifact_missing",
            ) from exc
        if expected.get(role) != actual[role]:
            raise _ProductChatGateError(
                f"product chat artifact {role} content digest drifted: {path}",
                reason_code="chat_artifact_drift",
            )

    if language_provider_artifact_digest(artifact) != artifact.artifact_digest:
        raise _ProductChatGateError(
            "product chat artifact manifest digest does not match its content manifest",
            reason_code="chat_artifact_drift",
        )
    return {
        "artifact_id": artifact.artifact_id,
        "expires_at": artifact.expires_at,
        "content_digests": actual,
        "artifact_digest": artifact.artifact_digest,
    }


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
        content_digests=config.content_digests,
        artifact_digest=config.artifact_digest,
        expires_at=config.expires_at,
        canary_id=config.canary_id,
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
    organ: LanguageOrgan = ExternalTextDecoderLanguageOrgan(
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
    """Attach the lossless structured codec for explicit debug use."""

    adapter.attach_language_organ(None)
    adapter.attach_language_backend_registry(LanguageBackendRegistry.default())
    adapter.attach_language_provider_artifact(None)
    adapter.attach_language_provider_artifact_registry(None)
    adapter.attach_language_organ(StructuredTextLanguageOrgan())


def attach_native_language_provider(adapter: TSKV8Adapter) -> None:
    """Attach the dependency-free readable product surface."""

    adapter.attach_language_organ(None)
    adapter.attach_language_backend_registry(LanguageBackendRegistry.default())
    adapter.attach_language_provider_artifact(None)
    adapter.attach_language_provider_artifact_registry(None)
    adapter.attach_language_organ(NativeReadableTextLanguageOrgan())


@dataclass(frozen=True)
class LanguageProviderRotationResult:
    """Outcome of an isolated provider rotation attempt."""

    status: LanguageProviderStatus
    runtime: Any | None
    registry: LanguageProviderArtifactRegistry
    committed: bool


def _unchanged_provider_status(
    adapter: TSKV8Adapter,
    *,
    reason_code: str,
    reason: str,
) -> LanguageProviderStatus:
    artifact = adapter.language_provider_artifact
    if artifact is None:
        return LanguageProviderStatus(
            mode="native",
            state="unchanged",
            provider="native",
            backend_id=NativeReadableTextLanguageOrgan.BACKEND_ID,
            artifact_id=NativeReadableTextLanguageOrgan.BACKEND_ID,
            reason_code=reason_code,
            reason=reason,
            rollback=NativeReadableTextLanguageOrgan.BACKEND_ID,
            chat_enabled=False,
        )
    return LanguageProviderStatus(
        mode=artifact.mode,
        state="unchanged",
        provider="existing",
        backend_id=artifact.backend_id,
        artifact_id=artifact.artifact_id,
        reason_code=reason_code,
        reason=reason,
        rollback=artifact.artifact_id,
        chat_enabled=adapter.language_organ is not None,
    )


def rotate_language_provider(
    adapter: TSKV8Adapter,
    registry: LanguageProviderArtifactRegistry,
    config: LanguageProviderConfig,
    *,
    current_runtime: Any | None = None,
) -> LanguageProviderRotationResult:
    """Stage, canary-check, and atomically publish one allowlisted version.

    The live adapter is never used to load the candidate.  A checkpoint clone
    with its terminal organ detached is used as staging; only after the
    existing artifact checks and first-chat canary pass does the adapter publish
    the new organ, manifest, and registry snapshot together.
    """

    if not isinstance(adapter, TSKV8Adapter):
        raise TypeError("language provider rotation requires a TSKV8Adapter")
    if not isinstance(registry, LanguageProviderArtifactRegistry):
        raise TypeError("language provider rotation requires a provider artifact registry")
    if not isinstance(config, LanguageProviderConfig):
        raise TypeError("language provider rotation requires a LanguageProviderConfig")

    candidate = build_provider_artifact(config)
    try:
        registry.require_allowed(candidate)
        if config.mode != "guarded" or not config.chat_enabled:
            raise _ProductChatGateError(
                "provider rotation requires an explicitly guarded chat-enabled candidate",
                reason_code="rotation_requires_chat_canary",
            )
        if registry.active_artifact_id == candidate.artifact_id:
            status = _unchanged_provider_status(
                adapter,
                reason_code="provider_version_already_active",
                reason="provider artifact version is already active",
            )
            return LanguageProviderRotationResult(
                status=status,
                runtime=current_runtime,
                registry=registry,
                committed=False,
            )

        staging_payload = adapter.native_checkpoint()
        staging_components = dict(staging_payload["components"])
        staging_components["language_organ"] = None
        staging_components["language_provider_artifact"] = None
        staging_components["language_provider_artifact_registry"] = None
        staging_payload["components"] = staging_components
        staging_adapter = TSKV8Adapter.from_native_checkpoint(staging_payload)
        staged_status, staged_runtime = activate_language_provider(staging_adapter, config)
        if staged_status.state != "active":
            status = _unchanged_provider_status(
                adapter,
                reason_code=staged_status.reason_code or "provider_rotation_rejected",
                reason=staged_status.reason or "provider rotation candidate failed activation",
            )
            return LanguageProviderRotationResult(
                status=status,
                runtime=current_runtime,
                registry=registry,
                committed=False,
            )
        staged_artifact = staging_adapter.language_provider_artifact
        staged_organ = staging_adapter.language_organ
        if staged_artifact is None or staged_organ is None:
            raise _ProductChatGateError(
                "provider rotation candidate did not produce a complete staged provider",
                reason_code="provider_rotation_rejected",
            )
        committed_registry = registry.activate(candidate.artifact_id)
        adapter.commit_language_provider_state(
            backend_registry=staging_adapter.language_backend_registry,
            artifact_registry=committed_registry,
            artifact=staged_artifact,
            organ=staged_organ,
        )
        return LanguageProviderRotationResult(
            status=staged_status,
            runtime=staged_runtime,
            registry=committed_registry,
            committed=True,
        )
    except _ProductChatGateError as exc:
        status = _unchanged_provider_status(
            adapter,
            reason_code=exc.reason_code,
            reason=str(exc),
        )
    except PermissionError as exc:
        status = _unchanged_provider_status(
            adapter,
            reason_code="provider_version_not_allowlisted",
            reason=str(exc),
        )
    except (KeyError, ValueError, TypeError) as exc:
        status = _unchanged_provider_status(
            adapter,
            reason_code="provider_rotation_manifest_mismatch",
            reason=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - rotation must preserve live provider
        status = _unchanged_provider_status(
            adapter,
            reason_code="provider_rotation_failed",
            reason=str(exc),
        )
    return LanguageProviderRotationResult(
        status=status,
        runtime=current_runtime,
        registry=registry,
        committed=False,
    )


def activate_language_provider(
    adapter: TSKV8Adapter,
    config: LanguageProviderConfig,
) -> tuple[LanguageProviderStatus, Any | None]:
    """Activate an explicit provider, or safely roll back to readable native text."""

    if not isinstance(adapter, TSKV8Adapter):
        raise TypeError("language provider activation requires a TSKV8Adapter")
    if not isinstance(config, LanguageProviderConfig):
        raise TypeError("language provider config must be a LanguageProviderConfig")
    if config.mode == "native":
        attach_native_language_provider(adapter)
        return (
            LanguageProviderStatus(
                mode="native",
                state="active",
                provider="native",
                backend_id=NativeReadableTextLanguageOrgan.BACKEND_ID,
                artifact_id=NativeReadableTextLanguageOrgan.BACKEND_ID,
                chat_enabled=False,
            ),
            None,
        )
    if config.mode == "structured":
        attach_structured_language_provider(adapter)
        return (
            LanguageProviderStatus(
                mode="structured",
                state="active",
                provider="structured",
                backend_id=StructuredTextLanguageOrgan.BACKEND_ID,
                artifact_id="structured-stub",
                chat_enabled=False,
            ),
            None,
        )

    artifact = build_provider_artifact(config)
    try:
        root = Path(config.artifact_root) if config.artifact_root else None
        if config.chat_enabled:
            _validate_product_chat_gate(artifact, artifact_root=root)
        if config.provider != "qwen":
            raise ValueError(f"unsupported Seed language provider: {config.provider}")
        model_dir = _resolve(config.model_dir, root)
        decoder = load_qwen_language_provider(
            adapter,
            artifact,
            model_dir=model_dir,
            artifact_root=root,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        if config.chat_enabled:
            organ = adapter.language_organ
            if organ is None:
                raise _ProductChatGateError(
                    "product chat provider did not attach a language organ",
                    reason_code="chat_canary_failed",
                )
            canary_report = LanguageProviderCanaryGate().evaluate(organ)
            if not canary_report["gate"]["passed"]:
                raise _ProductChatGateError(
                    "product chat first-chat canary did not pass semantic coverage",
                    reason_code="chat_canary_failed",
                )
        return (
            LanguageProviderStatus(
                mode=config.mode,
                state="active",
                provider=config.provider,
                backend_id=artifact.backend_id,
                artifact_id=artifact.artifact_id,
                chat_enabled=config.chat_enabled,
            ),
            decoder,
        )
    except _ProductChatGateError as exc:
        reason_code = exc.reason_code
        reason = str(exc)
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
    logger.warning("language provider rolled back to readable native text: %s", reason)
    attach_native_language_provider(adapter)
    return (
        LanguageProviderStatus(
            mode=config.mode,
            state="fallback",
            provider=config.provider,
            backend_id=NativeReadableTextLanguageOrgan.BACKEND_ID,
            artifact_id=NativeReadableTextLanguageOrgan.BACKEND_ID,
            reason_code=reason_code,
            reason=reason,
            rollback=NativeReadableTextLanguageOrgan.BACKEND_ID,
            chat_enabled=False,
        ),
        None,
    )
