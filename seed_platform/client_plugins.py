"""Seed-owned client plugin manifest registry.

Only declarative affordances and lifecycle are recorded here.  The protected
desktop root shell never loads plugin executable source through this registry.
"""

from __future__ import annotations

from typing import Any

from seed_platform.evolution_adapters import ClientPluginArtifactAdapter
from seed_platform.source_registry import DeclarativeSourceRegistry


class ClientPluginRegistry(DeclarativeSourceRegistry):
    def __init__(self, adapter: Any | None = None) -> None:
        super().__init__(ClientPluginArtifactAdapter() if adapter is None else adapter)

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> ClientPluginRegistry:
        return super().from_checkpoint(payload, adapter=ClientPluginArtifactAdapter())


__all__ = ["ClientPluginRegistry"]
