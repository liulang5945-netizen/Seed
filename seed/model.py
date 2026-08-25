"""Seed product/runtime compatibility boundary hosting Taiji."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from taiji import (
    TaijiConsolidation,
    TaijiDecision,
    TaijiOutcome,
    TaijiState,
    TaijiStep,
    TSKV8Adapter,
)

from .config import SeedConfig


class Seed:
    """Product-facing runtime wrapper for the current Taiji kernel.

    The public class name and ``substrate`` attribute remain compatible with
    ``seed-native-v1`` checkpoints while Taiji Native Architecture v1 is
    introduced. Seed owns product lifecycle and serialization wrapping; Taiji
    owns perception, cognition, learning and action. No second cognitive
    algorithm may be hidden in this runtime boundary.
    """

    CHECKPOINT_FORMAT = "seed-native-v1"

    def __init__(
        self,
        config: SeedConfig | None = None,
        *,
        device: torch.device | str = "cpu",
        episode_id: str = "episode-0",
    ) -> None:
        self.config = config or SeedConfig()
        # ``substrate`` is a compatibility attribute.  The actual hosted
        # architecture is the Taiji v1 contract surface exposed by this
        # TSK-v8 adapter; old callers continue to see a Taiji instance.
        self.substrate = TSKV8Adapter(
            self.config.taiji,
            device=device,
            episode_id=episode_id,
        )

    @property
    def architecture(self) -> TSKV8Adapter:
        """The Taiji architecture hosted by Seed (legacy alias: substrate)."""

        return self.substrate

    @property
    def device(self) -> torch.device:
        return self.substrate.device

    @property
    def tick(self) -> int:
        return self.substrate.tick

    def snapshot(self) -> TaijiState:
        return self.substrate.snapshot()

    def reset_dynamics(self, *, episode_id: str | None = None) -> None:
        self.substrate.reset_dynamics(episode_id=episode_id)

    def observe(
        self,
        symbol: int,
        *,
        learn: bool = True,
        learn_motor: bool | None = None,
        use_memory: bool = True,
    ) -> TaijiStep:
        return self.substrate.observe(
            symbol,
            learn=learn,
            learn_motor=learn_motor,
            use_memory=use_memory,
        )

    def act(
        self,
        available_actions: Sequence[int],
        *,
        sample: bool = True,
    ) -> TaijiDecision:
        return self.substrate.act(available_actions, sample=sample)

    def settle_action(
        self,
        reward: float,
        *,
        learn: bool = True,
        learn_memory: bool = True,
        provenance: str = "experienced",
    ) -> TaijiOutcome:
        return self.substrate.settle_action(
            reward,
            learn=learn,
            learn_memory=learn_memory,
            provenance=provenance,
        )

    def consolidate(
        self,
        *,
        cycles: int = 1,
        learn: bool = True,
    ) -> TaijiConsolidation:
        return self.substrate.consolidate(cycles=cycles, learn=learn)

    def learn_bytes(
        self,
        data: bytes,
        *,
        epochs: int = 1,
        include_boundary: bool = True,
    ) -> dict[str, float]:
        return self.substrate.learn_bytes(
            data,
            epochs=epochs,
            include_boundary=include_boundary,
        )

    def score_bytes(
        self,
        data: bytes,
        *,
        include_boundary: bool = True,
    ) -> dict[str, float]:
        return self.substrate.score_bytes(data, include_boundary=include_boundary)

    def generate(
        self,
        prompt: bytes,
        length: int,
        *,
        stop_at_boundary: bool = False,
        sample: bool = False,
        reset: bool = True,
    ) -> bytes:
        return self.substrate.generate(
            prompt,
            length,
            stop_at_boundary=stop_at_boundary,
            sample=sample,
            reset=reset,
        )

    def parameter_count(self, *, active_only: bool = True) -> int:
        return self.substrate.parameter_count(active_only=active_only)

    def dense_equivalent_parameter_count(self) -> int:
        return self.substrate.dense_equivalent_parameter_count()

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            # Canonical v1 cognitive envelope.  ``substrate`` remains a
            # readable compatibility payload for seed-native-v1 consumers.
            "taiji": self.substrate.native_checkpoint(),
            "substrate": self.substrate.checkpoint(),
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError("unsupported Seed checkpoint format")
        actual = SeedConfig.from_dict(checkpoint["config"])
        if actual != self.config:
            raise ValueError("checkpoint configuration does not match Seed")
        if "taiji" in checkpoint:
            self.substrate.restore_native(checkpoint["taiji"])
        else:
            # Read old seed-native-v1 checkpoints produced before P1.
            self.substrate.restore(checkpoint["substrate"])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> Seed:
        if checkpoint.get("format") != cls.CHECKPOINT_FORMAT:
            raise ValueError("unsupported Seed checkpoint format")
        config = SeedConfig.from_dict(checkpoint["config"])
        model = cls(config, device=device)
        model.restore(checkpoint)
        return model
