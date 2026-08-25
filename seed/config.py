"""Top-level configuration for the Seed product/runtime wrapper."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from taiji import TaijiConfig


@dataclass(frozen=True)
class LanguageProviderConfig:
    """Product-side selection for Taiji's replaceable terminal language organ."""

    mode: str = "structured"
    provider: str = "qwen"
    backend_id: str = "qwen2.5-0.5b-instruct"
    model_dir: str = ""
    adapter_dir: str = ""
    artifact_id: str = "seed-language-provider-v1"
    artifact_root: str = ""
    training_corpus: str = ""
    training_report: str = ""
    safety_report: str = ""
    max_tokens: int = 24
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"structured", "raw", "lora", "guarded"}:
            raise ValueError("language provider mode must be structured, raw, lora, or guarded")
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

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> LanguageProviderConfig:
        values = dict(payload or {})
        return cls(
            mode=str(values.get("mode", "structured")),
            provider=str(values.get("provider", "qwen")),
            backend_id=str(values.get("backend_id", "qwen2.5-0.5b-instruct")),
            model_dir=str(values.get("model_dir", "")),
            adapter_dir=str(values.get("adapter_dir", "")),
            artifact_id=str(values.get("artifact_id", "seed-language-provider-v1")),
            artifact_root=str(values.get("artifact_root", "")),
            training_corpus=str(values.get("training_corpus", "")),
            training_report=str(values.get("training_report", "")),
            safety_report=str(values.get("safety_report", "")),
            max_tokens=int(values.get("max_tokens", 24)),
            temperature=float(values.get("temperature", 0.0)),
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
            "MAX_TOKENS": "max_tokens",
            "TEMPERATURE": "temperature",
        }
        for suffix, field_name in names.items():
            value = os.environ.get(f"SEED_LANGUAGE_{suffix}")
            if value is None:
                continue
            if field_name == "max_tokens":
                current[field_name] = int(value)
            elif field_name == "temperature":
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
