"""Seed-owned declarative external Skill registry."""

from __future__ import annotations

from typing import Any

from seed_platform.evolution_adapters import SkillArtifactAdapter
from seed_platform.source_registry import DeclarativeSourceRegistry


class SkillRegistry(DeclarativeSourceRegistry):
    def __init__(self, adapter: Any | None = None) -> None:
        super().__init__(SkillArtifactAdapter() if adapter is None else adapter)

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> SkillRegistry:
        return super().from_checkpoint(payload, adapter=SkillArtifactAdapter())


__all__ = ["SkillRegistry"]
