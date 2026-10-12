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
        # ㊵-745（PLAN-N4-11 §1.1 / DEBT-G122 修法乙）：对象↔情节的绑定是**派生索引**，
        # 从每条记录的 object_ids 现建，刻意不进 checkpoint()/to_payload()——加持久键会扩迁移面。
        self._object_of: dict[str, tuple[str, ...]] = {}
        self._objects: dict[str, list[str]] = {}

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

    def _unbind(self, memory_id: str) -> None:
        for object_id in self._object_of.pop(memory_id, ()):
            bucket = self._objects.get(object_id)
            if bucket is None:
                continue
            bucket[:] = [held for held in bucket if held != memory_id]
            if not bucket:
                del self._objects[object_id]

    def _bind(self, record: EpisodicMemoryRecord) -> None:
        self._unbind(record.memory_id)
        self._object_of[record.memory_id] = tuple(record.object_ids)
        for object_id in record.object_ids:
            self._objects.setdefault(object_id, []).append(record.memory_id)

    def write(self, record: EpisodicMemoryRecord) -> None:
        if not isinstance(record, EpisodicMemoryRecord):
            raise TypeError("episodic memory store accepts EpisodicMemoryRecord")
        if self.cue_dim is None:
            self.cue_dim = int(record.cue.numel())
        if record.cue.numel() != self.cue_dim:
            raise ValueError("episodic memory cue dimension does not match the store")
        self._records.pop(record.memory_id, None)
        self._records[record.memory_id] = EpisodicMemoryRecord.from_payload(record.to_payload())
        self._bind(self._records[record.memory_id])
        while len(self._records) > self.capacity:
            oldest_id = next(iter(self._records))
            del self._records[oldest_id]
            self._unbind(oldest_id)
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
            self._unbind(str(memory_id))
            self._expired.append(str(memory_id))
        return removed

    def object_identifiers(self) -> tuple[str, ...]:
        """当前有绑定的对象标识；空元组＝没人绑，不代表库里没有记录。"""

        return tuple(self._objects)

    def bound_memory_ids(self, object_id: str) -> tuple[str, ...]:
        if str(object_id) not in self._objects:
            raise ValueError(f"no episodic records are bound to object {object_id!r}")
        return tuple(self._objects[str(object_id)])

    def rename_object(self, old_object_id: str, new_object_id: str) -> int:
        """只换绑定名，不碰记录本身——这样「同一对象换了名字」才是一件能被测的事。"""

        old, new = str(old_object_id), str(new_object_id)
        if not old or not new:
            raise ValueError("episodic object rename requires both identifiers")
        bucket = self._objects.pop(old, None)
        if bucket is None:
            raise ValueError(f"no episodic records are bound to object {old!r}")
        if new in self._objects:
            raise ValueError(f"object {new!r} is already bound; rename would merge two objects")
        self._objects[new] = bucket
        for memory_id in bucket:
            self._object_of[memory_id] = tuple(
                new if held == old else held for held in self._object_of[memory_id]
            )
        return len(bucket)

    def retrieve(
        self,
        cue: torch.Tensor,
        *,
        limit: int = 1,
        episode_id: str | None = None,
        object_id: str | None = None,
    ) -> tuple[EpisodicMemoryHit, ...]:
        """Rank records by cue similarity; `episode_id` and `object_id` narrow the candidate set.

        `episode_id=None` (the default) searches every record, which is what the
        product's per-step caller does today — the storage layer still has no
        session gate, and DEBT-G106 stays open until a caller asks for one.

        `object_id=None` (the default) keeps today's candidate set bit-for-bit
        (PLAN-N4-11 J-N4h-1).  Naming an object narrows candidates to the records
        bound to it; naming an object with no binding raises instead of returning
        `()`, because "zero hits" and "this key does not exist" must not collapse
        into one answer (DEBT-G58, DEBT-G122).
        """

        if cue.ndim != 1:
            raise ValueError("episodic retrieval cue must be a vector")
        if self.cue_dim is not None and cue.numel() != self.cue_dim:
            raise ValueError("episodic retrieval cue dimension does not match the store")
        if int(limit) <= 0:
            return ()
        bound: set[str] | None = None
        if object_id is not None:
            if str(object_id) not in self._objects:
                raise ValueError(f"no episodic records are bound to object {object_id!r}")
            bound = set(self._objects[str(object_id)])
        records = tuple(
            record
            for record in self._records.values()
            if (episode_id is None or record.episode_id == episode_id)
            and (bound is None or record.memory_id in bound)
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
