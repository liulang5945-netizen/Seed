"""Top-level configuration for the Seed product/runtime wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from taiji import TaijiConfig


@dataclass(frozen=True)
class SeedConfig:
    """Compatibility configuration; Taiji owns cognitive architecture state."""

    taiji: TaijiConfig = field(default_factory=TaijiConfig)

    def to_dict(self) -> dict[str, Any]:
        return {"taiji": self.taiji.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SeedConfig:
        return cls(taiji=TaijiConfig.from_dict(dict(payload["taiji"])))
