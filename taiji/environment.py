"""Minimal active-environment contract for native Taiji interaction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class EnvironmentOutcome:
    """What the environment returns after one Taiji motor action."""

    sensation: int
    reward: float
    terminal: bool = False
    success: bool | None = None


@runtime_checkable
class TaijiEnvironment(Protocol):
    """Protocol for environments whose transitions depend on Taiji actions."""

    def reset(self) -> tuple[int, Sequence[int]]:
        """Return the initial sensation and currently afforded actions."""

    def step(self, action_symbol: int) -> EnvironmentOutcome:
        """Execute an action and return its sensation, reward and terminal flag."""


@runtime_checkable
class TaijiToolEnvironment(Protocol):
    """Protocol for environments that accept structured tool organs."""

    def execute_tool(
        self, tool_name: str, parameters: Mapping[str, Any]
    ) -> EnvironmentOutcome:
        """Execute one structured tool call and return its causal outcome."""
