"""Taiji-owned content-addressed episodic memory contracts and store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .contracts import EpisodicMemoryRecord

EPISODIC_MEMORY_CHECKPOINT_FORMAT = "taiji-episodic-memory-v1"


@dataclass(frozen=True)
class EpisodicMemoryHit:
    """A retrieved experience and its content similarity."""

    record: EpisodicMemoryRecord
    score: float


class EpisodicMemoryStore:
    """Persistent event memory with a configurable capacity and vector index.

    The store keeps real experience records and ranks them by cue similarity;
    it does not allocate a fixed semantic slot or maintain a domain fact table.
    """

    def __init__(self, *, capacity: int = 1024, cue_dim: int | None = None) -> None:
        if int(capacity) <= 0:
            raise ValueError("episodic memory capacity must be positive")
        if cue_dim is not None and int(cue_dim) <= 0:
            raise ValueError("episodic memory cue_dim must be positive")
        self.capacity = int(capacity)
        self.cue_dim = None if cue_dim is None else int(cue_dim)
        self._records: dict[str, EpisodicMemoryRecord] = {}

    @property
    def records(self) -> tuple[EpisodicMemoryRecord, ...]:
        return tuple(self._records.values())

    @property
    def count(self) -> int:
        return len(self._records)

    def write(self, record: EpisodicMemoryRecord) -> None:
        if not isinstance(record, EpisodicMemoryRecord):
            raise TypeError("episodic memory store accepts EpisodicMemoryRecord")
        if self.cue_dim is None:
            self.cue_dim = int(record.cue.numel())
        if record.cue.numel() != self.cue_dim:
            raise ValueError("episodic memory cue dimension does not match the store")
        self._records.pop(record.memory_id, None)
        self._records[record.memory_id] = EpisodicMemoryRecord.from_payload(record.to_payload())
        while len(self._records) > self.capacity:
            oldest_id = next(iter(self._records))
            del self._records[oldest_id]

    def retrieve(self, cue: torch.Tensor, *, limit: int = 1) -> tuple[EpisodicMemoryHit, ...]:
        if cue.ndim != 1:
            raise ValueError("episodic retrieval cue must be a vector")
        if self.cue_dim is not None and cue.numel() != self.cue_dim:
            raise ValueError("episodic retrieval cue dimension does not match the store")
        if int(limit) <= 0:
            return ()
        if not self._records:
            return ()
        records = tuple(self._records.values())
        query = cue.detach().to(dtype=torch.float32)
        query_norm = torch.linalg.vector_norm(query)
        if float(query_norm) <= 1e-8:
            scores = [0.0 for _ in records]
        else:
            scores = [
                float(
                    torch.dot(query, record.cue.to(query.device, dtype=query.dtype))
                    / (
                        query_norm
                        * torch.linalg.vector_norm(record.cue.to(query.device, dtype=query.dtype))
                        + 1e-8
                    )
                )
                for record in records
            ]
        order = sorted(range(len(self._records)), key=lambda index: (-scores[index], -index))
        return tuple(
            EpisodicMemoryHit(records[index], scores[index]) for index in order[: int(limit)]
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": EPISODIC_MEMORY_CHECKPOINT_FORMAT,
            "capacity": self.capacity,
            "cue_dim": self.cue_dim,
            "records": [record.to_payload() for record in self._records.values()],
        }

    @classmethod
    def from_checkpoint(
        cls, payload: dict[str, Any], *, device: torch.device | str = "cpu"
    ) -> EpisodicMemoryStore:
        if payload.get("format") != EPISODIC_MEMORY_CHECKPOINT_FORMAT:
            raise ValueError("unsupported episodic memory checkpoint format")
        store = cls(capacity=int(payload["capacity"]), cue_dim=payload.get("cue_dim"))
        for item in payload.get("records", ()):
            store.write(EpisodicMemoryRecord.from_payload(item, device=device))
        return store
