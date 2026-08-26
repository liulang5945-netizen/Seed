"""Taiji-owned persistent world state for the P3 causal vertical slice."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .contracts import WorldState, WorldTransition

WORLD_STATE_CHECKPOINT_FORMAT = "taiji-world-state-v1"


def _value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _value_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _value_equal(item_left, item_right)
            for item_left, item_right in zip(left, right, strict=False)
        )
    return left == right


def _world_state_equal(left: WorldState, right: WorldState) -> bool:
    return _value_equal(left.to_payload(), right.to_payload())


class TaijiWorldState:
    """Own, checkpoint and validate the state changed by real interventions.

    The store keeps structured objects/events inside the Taiji checkpoint.  An
    environment may propose a transition, but it cannot silently mutate the
    current state or supply an untracked Python fact table.
    """

    def __init__(self, initial: WorldState, *, history_limit: int | None = None) -> None:
        if history_limit is not None and int(history_limit) < 0:
            raise ValueError("history_limit cannot be negative")
        self._state = initial
        self._history_limit = None if history_limit is None else int(history_limit)
        self._history: list[WorldTransition] = []

    @property
    def state(self) -> WorldState:
        """Return a detached state snapshot safe for callers to inspect."""

        return WorldState.from_payload(self._state.to_payload())

    @property
    def history(self) -> tuple[WorldTransition, ...]:
        return tuple(WorldTransition.from_payload(item.to_payload()) for item in self._history)

    def apply(self, transition: WorldTransition) -> WorldState:
        """Commit one causally linked action/outcome/next-state transition."""

        if not _world_state_equal(self._state, transition.before):
            raise ValueError("world transition does not start from the owned current state")
        self._state = transition.after
        self._history.append(transition)
        if self._history_limit is not None:
            if self._history_limit:
                del self._history[: -self._history_limit]
            else:
                self._history.clear()
        return self.state

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": WORLD_STATE_CHECKPOINT_FORMAT,
            "history_limit": self._history_limit,
            "state": self._state.to_payload(),
            "history": [item.to_payload() for item in self._history],
        }

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> TaijiWorldState:
        if checkpoint.get("format") != WORLD_STATE_CHECKPOINT_FORMAT:
            raise ValueError("unsupported Taiji world-state checkpoint format")
        history_limit = checkpoint.get("history_limit")
        world = cls(
            WorldState.from_payload(checkpoint["state"], device=device),
            history_limit=None if history_limit is None else int(history_limit),
        )
        world._history = [
            WorldTransition.from_payload(item, device=device)
            for item in checkpoint.get("history", ())
        ]
        if world._history:
            if not _world_state_equal(world._history[-1].after, world._state):
                raise ValueError("world-state checkpoint history does not end at current state")
            if any(
                not _world_state_equal(left.after, right.before)
                for left, right in zip(world._history, world._history[1:], strict=False)
            ):
                raise ValueError("world-state checkpoint history is not contiguous")
        return world
