"""Canonical artifact vocabulary owned by the Taiji platform.

An artifact describes a persisted capability asset.  It is deliberately not
the same thing as a runtime, provider, or model family: the runtime consumes
an artifact, while a language provider artifact only owns the language
realization boundary.
"""

from __future__ import annotations

from typing import Final

ARTIFACT_TYPES: Final[tuple[str, ...]] = (
    "taiji_checkpoint",
    "language_provider_artifact",
    "legacy_benchmark_artifact",
)


def is_artifact_type(value: object) -> bool:
    """Return whether ``value`` is one of the canonical artifact types."""

    return isinstance(value, str) and value in ARTIFACT_TYPES
