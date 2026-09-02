"""Fixed-capacity cue binding population for native episodic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .sparse import bound_norm


@dataclass(frozen=True)
class CueBindingResult:
    """The slot selected for one cue route, without exposing an answer."""

    slot_index: int | None
    similarity: float
    allocated: bool
    replaced: bool


class CueBindingBank:
    """A bounded competitive population of cue prototypes.

    The bank owns cue identity only.  It has no action, outcome, or external
    answer table.  A write either matches a sufficiently similar occupied
    assembly or allocates the least-used free/recyclable assembly; a read is
    strictly non-mutating.  This makes slot collisions and structural release
    observable before the population is connected to a readout.
    """

    CHECKPOINT_FORMAT = "taiji-cue-binding-v1"

    def __init__(
        self,
        capacity: int,
        pattern_dim: int,
        *,
        match_threshold: float = 0.85,
        update_rate: float = 0.10,
        device: torch.device | str = "cpu",
    ) -> None:
        if int(capacity) <= 0 or int(pattern_dim) <= 0:
            raise ValueError("cue binding capacity and pattern_dim must be positive")
        if not 0.0 < float(match_threshold) <= 1.0:
            raise ValueError("cue binding match_threshold must be in (0, 1]")
        if not 0.0 < float(update_rate) <= 1.0:
            raise ValueError("cue binding update_rate must be in (0, 1]")
        self.capacity = int(capacity)
        self.pattern_dim = int(pattern_dim)
        self.match_threshold = float(match_threshold)
        self.update_rate = float(update_rate)
        self.device = torch.device(device)
        self.prototypes = torch.zeros(
            (self.capacity, self.pattern_dim), device=self.device, dtype=torch.float32
        )
        self.occupied = torch.zeros(self.capacity, device=self.device, dtype=torch.bool)
        self.visits = torch.zeros(self.capacity, device=self.device, dtype=torch.long)
        self.allocation_count = 0
        self.match_count = 0
        self.replacement_count = 0

    def _normalize(self, pattern: torch.Tensor) -> torch.Tensor:
        if pattern.shape != (self.pattern_dim,):
            raise ValueError(
                f"cue binding pattern must be ({self.pattern_dim},), got {tuple(pattern.shape)}"
            )
        value = pattern.to(self.device, dtype=torch.float32)
        if float(value.norm().item()) <= 1e-8:
            raise ValueError("cue binding pattern cannot be empty")
        return bound_norm(value, 1.0)

    def route(self, pattern: torch.Tensor, *, learn: bool) -> CueBindingResult:
        """Match or allocate one assembly; reads do not mutate any state."""

        normalized = self._normalize(pattern)
        occupied_indices = torch.nonzero(self.occupied, as_tuple=False).flatten()
        if occupied_indices.numel():
            scores = torch.full((self.capacity,), float("-inf"), device=self.device)
            scores[occupied_indices] = self.prototypes[occupied_indices] @ normalized
            best_index = int(scores.argmax().item())
            best_similarity = float(scores[best_index].item())
            if best_similarity >= self.match_threshold:
                if learn:
                    blended = (
                        (1.0 - self.update_rate) * self.prototypes[best_index]
                        + self.update_rate * normalized
                    )
                    self.prototypes[best_index] = bound_norm(blended, 1.0)
                    self.visits[best_index] += 1
                    self.match_count += 1
                return CueBindingResult(
                    slot_index=best_index,
                    similarity=best_similarity,
                    allocated=False,
                    replaced=False,
                )
            if not learn:
                return CueBindingResult(
                    slot_index=None,
                    similarity=best_similarity,
                    allocated=False,
                    replaced=False,
                )

        free_indices = torch.nonzero(~self.occupied, as_tuple=False).flatten()
        replaced = not bool(free_indices.numel())
        if replaced:
            slot_index = int(self.visits.argmin().item())
            self.replacement_count += 1
        else:
            slot_index = int(free_indices[0].item())
        self.prototypes[slot_index] = normalized
        self.occupied[slot_index] = True
        self.visits[slot_index] = 1
        self.allocation_count += 1
        return CueBindingResult(
            slot_index=slot_index,
            similarity=0.0 if not occupied_indices.numel() else best_similarity,
            allocated=True,
            replaced=replaced,
        )

    def release(self, slot_index: int) -> None:
        index = int(slot_index)
        if not 0 <= index < self.capacity:
            raise ValueError("cue binding slot index is outside capacity")
        self.prototypes[index].zero_()
        self.occupied[index] = False
        self.visits[index] = 0

    @property
    def occupied_count(self) -> int:
        return int(self.occupied.sum().item())

    def slot_code(self, slot_index: int | None) -> torch.Tensor:
        code = torch.zeros(self.capacity, device=self.device)
        if slot_index is not None:
            index = int(slot_index)
            if not 0 <= index < self.capacity or not bool(self.occupied[index].item()):
                raise ValueError("cue binding slot is not occupied")
            code[index] = 1.0
        return code

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "capacity": self.capacity,
            "pattern_dim": self.pattern_dim,
            "match_threshold": self.match_threshold,
            "update_rate": self.update_rate,
            "prototypes": self.prototypes.detach().cpu().clone(),
            "occupied": self.occupied.detach().cpu().clone(),
            "visits": self.visits.detach().cpu().clone(),
            "allocation_count": self.allocation_count,
            "match_count": self.match_count,
            "replacement_count": self.replacement_count,
        }

    def load_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError("unsupported cue binding checkpoint format")
        expected = (
            self.capacity,
            self.pattern_dim,
            self.match_threshold,
            self.update_rate,
        )
        actual = (
            int(payload["capacity"]),
            int(payload["pattern_dim"]),
            float(payload["match_threshold"]),
            float(payload["update_rate"]),
        )
        if actual != expected:
            raise ValueError("cue binding checkpoint architecture does not match")
        prototypes = payload["prototypes"].detach().to(self.device, dtype=torch.float32)
        occupied = payload["occupied"].detach().to(self.device, dtype=torch.bool)
        visits = payload["visits"].detach().to(self.device, dtype=torch.long)
        if prototypes.shape != self.prototypes.shape:
            raise ValueError("cue binding prototype shape does not match architecture")
        if occupied.shape != self.occupied.shape or visits.shape != self.visits.shape:
            raise ValueError("cue binding state shape does not match architecture")
        if not torch.isfinite(prototypes).all() or (visits < 0).any():
            raise ValueError("cue binding checkpoint contains invalid values")
        self.prototypes = prototypes.clone()
        self.occupied = occupied.clone()
        self.visits = visits.clone()
        self.allocation_count = int(payload.get("allocation_count", 0))
        self.match_count = int(payload.get("match_count", 0))
        self.replacement_count = int(payload.get("replacement_count", 0))
        if min(self.allocation_count, self.match_count, self.replacement_count) < 0:
            raise ValueError("cue binding counters cannot be negative")
