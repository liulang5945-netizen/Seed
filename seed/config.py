"""Top-level configuration for the Seed product/runtime wrapper."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from taiji import (
    LANGUAGE_PROVIDER_CANARY_FORMAT,
    LanguageProviderHealthPolicy,
    TaijiConfig,
)

LANGUAGE_PROVIDER_CONFIG_VERSION = 2


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"cannot parse boolean language provider value: {value}")


@dataclass(frozen=True)
class LanguageProviderConfig:
    """Product-side selection for Taiji's replaceable terminal language organ."""

    config_version: int = LANGUAGE_PROVIDER_CONFIG_VERSION
    mode: str = "native"
    provider: str = "qwen"
    backend_id: str = "qwen2.5-0.5b-instruct"
    model_dir: str = ""
    adapter_dir: str = ""
    artifact_id: str = "seed-language-provider-v1"
    artifact_root: str = ""
    training_corpus: str = ""
    training_report: str = ""
    safety_report: str = ""
    content_digests: tuple[tuple[str, str], ...] = ()
    artifact_digest: str = ""
    expires_at: float | None = None
    canary_id: str = LANGUAGE_PROVIDER_CANARY_FORMAT
    chat_enabled: bool = False
    max_tokens: int = 24
    temperature: float = 0.0
    health_failure_threshold: int = 3
    health_cooldown_seconds: float = 300.0
    health_minimum_accepted_rate: float = 0.5
    health_minimum_rate_probes: int = 8

    def __post_init__(self) -> None:
        if int(self.config_version) != LANGUAGE_PROVIDER_CONFIG_VERSION:
            raise ValueError(
                f"language provider config_version must be {LANGUAGE_PROVIDER_CONFIG_VERSION}"
            )
        if self.mode not in {"native", "structured", "raw", "lora", "guarded"}:
            raise ValueError(
                "language provider mode must be native, structured, raw, lora, or guarded"
            )
        if self.chat_enabled and self.mode != "guarded":
            raise ValueError("product chat requires an explicitly guarded language provider")
        if not str(self.provider):
            raise ValueError("language provider provider cannot be empty")
        if not str(self.backend_id):
            raise ValueError("language provider backend_id cannot be empty")
        if not str(self.artifact_id):
            raise ValueError("language provider artifact_id cannot be empty")
        if int(self.max_tokens) <= 0:
            raise ValueError("language provider max_tokens must be positive")
        if float(self.temperature) < 0.0:
            raise ValueError("language provider temperature cannot be negative")
        if self.expires_at is not None and (
            isinstance(self.expires_at, bool) or not math.isfinite(float(self.expires_at))
        ):
            raise ValueError("language provider expires_at must be finite")
        if not str(self.canary_id):
            raise ValueError("language provider canary_id cannot be empty")
        # Construct the watchdog policy once so its own invariants reject bad
        # thresholds here instead of at the first runtime probe.
        self.health_policy()

    def health_policy(self) -> LanguageProviderHealthPolicy:
        """Return the runtime health watchdog policy declared by this selection."""

        return LanguageProviderHealthPolicy(
            failure_threshold=int(self.health_failure_threshold),
            cooldown_seconds=float(self.health_cooldown_seconds),
            minimum_accepted_rate=float(self.health_minimum_accepted_rate),
            minimum_rate_probes=int(self.health_minimum_rate_probes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_version": self.config_version,
            "mode": self.mode,
            "provider": self.provider,
            "backend_id": self.backend_id,
            "model_dir": self.model_dir,
            "adapter_dir": self.adapter_dir,
            "artifact_id": self.artifact_id,
            "artifact_root": self.artifact_root,
            "training_corpus": self.training_corpus,
            "training_report": self.training_report,
            "safety_report": self.safety_report,
            "content_digests": {str(role): str(digest) for role, digest in self.content_digests},
            "artifact_digest": self.artifact_digest,
            "expires_at": self.expires_at,
            "canary_id": self.canary_id,
            "chat_enabled": self.chat_enabled,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "health_failure_threshold": self.health_failure_threshold,
            "health_cooldown_seconds": self.health_cooldown_seconds,
            "health_minimum_accepted_rate": self.health_minimum_accepted_rate,
            "health_minimum_rate_probes": self.health_minimum_rate_probes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> LanguageProviderConfig:
        values = dict(payload or {})
        version = int(values.get("config_version", 1))
        mode = str(values.get("mode", "native"))
        # Version 1 used structured-stub as the product default.  Keep an
        # explicitly versioned structured selection intact, but migrate old
        # unversioned defaults to the readable native surface.
        if version == 1 and mode == "structured":
            mode = "native"
        content_digests = values.get("content_digests", {})
        if isinstance(content_digests, Mapping):
            digest_pairs = tuple(
                (str(role), str(digest)) for role, digest in content_digests.items()
            )
        elif isinstance(content_digests, (list, tuple)):
            digest_pairs = tuple(
                (str(item[0]), str(item[1]))
                for item in content_digests
                if isinstance(item, (list, tuple)) and len(item) == 2
            )
        else:
            raise ValueError("language provider content_digests must be a mapping or sequence")
        return cls(
            config_version=LANGUAGE_PROVIDER_CONFIG_VERSION,
            mode=mode,
            provider=str(values.get("provider", "qwen")),
            backend_id=str(values.get("backend_id", "qwen2.5-0.5b-instruct")),
            model_dir=str(values.get("model_dir", "")),
            adapter_dir=str(values.get("adapter_dir", "")),
            artifact_id=str(values.get("artifact_id", "seed-language-provider-v1")),
            artifact_root=str(values.get("artifact_root", "")),
            training_corpus=str(values.get("training_corpus", "")),
            training_report=str(values.get("training_report", "")),
            safety_report=str(values.get("safety_report", "")),
            content_digests=digest_pairs,
            artifact_digest=str(values.get("artifact_digest", "")),
            expires_at=(None if values.get("expires_at") is None else float(values["expires_at"])),
            canary_id=str(values.get("canary_id", LANGUAGE_PROVIDER_CANARY_FORMAT)),
            chat_enabled=_as_bool(values.get("chat_enabled", False)),
            max_tokens=int(values.get("max_tokens", 24)),
            temperature=float(values.get("temperature", 0.0)),
            health_failure_threshold=int(values.get("health_failure_threshold", 3)),
            health_cooldown_seconds=float(values.get("health_cooldown_seconds", 300.0)),
            health_minimum_accepted_rate=float(values.get("health_minimum_accepted_rate", 0.5)),
            health_minimum_rate_probes=int(values.get("health_minimum_rate_probes", 8)),
        )

    @classmethod
    def from_environment(cls, base: LanguageProviderConfig | None = None) -> LanguageProviderConfig:
        """Apply explicit ``SEED_LANGUAGE_*`` overrides to a base selection."""

        current = (base or cls()).to_dict()
        names = {
            "MODE": "mode",
            "PROVIDER": "provider",
            "BACKEND_ID": "backend_id",
            "MODEL_DIR": "model_dir",
            "ADAPTER_DIR": "adapter_dir",
            "ARTIFACT_ID": "artifact_id",
            "ARTIFACT_ROOT": "artifact_root",
            "TRAINING_CORPUS": "training_corpus",
            "TRAINING_REPORT": "training_report",
            "SAFETY_REPORT": "safety_report",
            "ARTIFACT_DIGEST": "artifact_digest",
            "EXPIRES_AT": "expires_at",
            "CANARY_ID": "canary_id",
            "CHAT_ENABLED": "chat_enabled",
            "MAX_TOKENS": "max_tokens",
            "TEMPERATURE": "temperature",
            "HEALTH_FAILURE_THRESHOLD": "health_failure_threshold",
            "HEALTH_COOLDOWN_SECONDS": "health_cooldown_seconds",
            "HEALTH_MINIMUM_ACCEPTED_RATE": "health_minimum_accepted_rate",
            "HEALTH_MINIMUM_RATE_PROBES": "health_minimum_rate_probes",
        }
        for suffix, field_name in names.items():
            value = os.environ.get(f"SEED_LANGUAGE_{suffix}")
            if value is None:
                continue
            if field_name in {
                "max_tokens",
                "health_failure_threshold",
                "health_minimum_rate_probes",
            }:
                current[field_name] = int(value)
            elif field_name in {
                "temperature",
                "expires_at",
                "health_cooldown_seconds",
                "health_minimum_accepted_rate",
            }:
                current[field_name] = float(value)
            else:
                current[field_name] = value
        return cls.from_dict(current)


@dataclass(frozen=True)
class SeedConfig:
    """Compatibility configuration; Taiji owns cognitive architecture state."""

    taiji: TaijiConfig = field(default_factory=TaijiConfig)
    language_provider: LanguageProviderConfig = field(default_factory=LanguageProviderConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "taiji": self.taiji.to_dict(),
            "language_provider": self.language_provider.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SeedConfig:
        return cls(
            taiji=TaijiConfig.from_dict(dict(payload["taiji"])),
            language_provider=LanguageProviderConfig.from_dict(payload.get("language_provider")),
        )
