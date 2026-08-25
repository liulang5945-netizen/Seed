"""Compatibility wrapper for the packaged Seed provider boundary.

Training/evaluation scripts keep their historical import path, while the
client runtime uses the implementation in ``seed.language_provider``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from seed.language_provider import load_qwen_language_provider
from taiji import LanguageProviderArtifact, TSKV8Adapter


def attach_qwen_language_provider(
    adapter: TSKV8Adapter,
    artifact: LanguageProviderArtifact,
    *,
    model_dir: Path,
    prompt_builder: Callable[[Any], str] | None = None,
    artifact_root: Path | None = None,
) -> Any:
    """Preserve the training-script API while delegating to Seed runtime."""

    del prompt_builder
    return load_qwen_language_provider(
        adapter,
        artifact,
        model_dir=model_dir,
        artifact_root=artifact_root,
    )


__all__ = ["attach_qwen_language_provider", "load_qwen_language_provider"]
