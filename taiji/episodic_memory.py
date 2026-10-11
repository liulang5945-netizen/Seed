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
        # ㊵-681（DEBT-G99）：显式撤销与容量淘汰是两件事，各记一列，不许合并。
        self._expired: list[str] = []
        self.evicted_count = 0
        self.implicit_provenance_count = 0

    @property
    def records(self) -> tuple[EpisodicMemoryRecord, ...]:
        return tuple(self._records.values())

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def expired_memory_ids(self) -> tuple[str, ...]:
        """Explicitly revoked ids.  Empty means "nothing was revoked", not "nothing expired"."""

        return tuple(self._expired)

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
            self.evicted_count += 1

    def expire(self, *memory_ids: str) -> int:
        """Revoke named episodic records by source-level decision (DEBT-G99).

        This deletes from the store only: it is not a claim that anything was
        forgotten from trained weights, and it must never be described that way.
        """

        removed = 0
        for memory_id in memory_ids:
            if not str(memory_id):
                raise ValueError("episodic expire requires a memory_id")
            if self._records.pop(str(memory_id), None) is not None:
                removed += 1
            self._expired.append(str(memory_id))
        return removed

    def retrieve(
        self, cue: torch.Tensor, *, limit: int = 1, episode_id: str | None = None
    ) -> tuple[EpisodicMemoryHit, ...]:
        """Rank records by cue similarity; `episode_id` narrows the candidate set.

        `episode_id=None` (the default) searches every record, which is what the
        product's per-step caller does today — the storage layer still has no
        session gate, and DEBT-G106 stays open until a caller asks for one.
        """

        if cue.ndim != 1:
            raise ValueError("episodic retrieval cue must be a vector")
        if self.cue_dim is not None and cue.numel() != self.cue_dim:
            raise ValueError("episodic retrieval cue dimension does not match the store")
        if int(limit) <= 0:
            return ()
        records = tuple(
            record
            for record in self._records.values()
            if episode_id is None or record.episode_id == episode_id
        )
        if not records:
            return ()
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
        order = sorted(range(len(records)), key=lambda index: (-scores[index], -index))
        return tuple(
            EpisodicMemoryHit(records[index], scores[index]) for index in order[: int(limit)]
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": EPISODIC_MEMORY_CHECKPOINT_FORMAT,
            "capacity": self.capacity,
            "cue_dim": self.cue_dim,
            "records": [record.to_payload() for record in self._records.values()],
            "expired_memory_ids": list(self._expired),
            "evicted_count": self.evicted_count,
            "implicit_provenance_count": self.implicit_provenance_count,
        }

    @classmethod
    def from_checkpoint(
        cls, payload: dict[str, Any], *, device: torch.device | str = "cpu"
    ) -> EpisodicMemoryStore:
        if payload.get("format") != EPISODIC_MEMORY_CHECKPOINT_FORMAT:
            raise ValueError("unsupported episodic memory checkpoint format")
        store = cls(capacity=int(payload["capacity"]), cue_dim=payload.get("cue_dim"))
        for item in payload.get("records", ()):
            if "provenance" not in item:
                store.implicit_provenance_count += 1
            store.write(EpisodicMemoryRecord.from_payload(item, device=device))
        # Optional keys: a v1 payload written before DEBT-G99 simply has none of them.
        store._expired = [str(item) for item in payload.get("expired_memory_ids", ())]
        store.evicted_count = int(payload.get("evicted_count", 0))
        # 这一列量的是"本次载入时有几条记录没声明来源"，由上面的循环现数；
        # 不去读件里那个历史值，否则回填会把刚数出来的数覆盖成 0。
        return store
