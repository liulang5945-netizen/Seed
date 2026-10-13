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
        # ㊵-757（PLAN-N4-12 §1.3／DEBT-G122 八裁⑴ 甲）：属性位是**第二**可寻址维，形状与对象通道同构，
        # 同样是派生索引（不进 checkpoint()）；它不进打分——retrieve 只用它收窄候选集。
        self._attribute_of: dict[str, tuple[str, ...]] = {}
        self._attributes: dict[str, list[str]] = {}
        self.implicit_attribute_count = 0

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

    @staticmethod
    def _detach(
        of: dict[str, tuple[str, ...]], buckets: dict[str, list[str]], memory_id: str
    ) -> None:
        for key in of.pop(memory_id, ()):
            bucket = buckets.get(key)
            if bucket is None:
                continue
            bucket[:] = [held for held in bucket if held != memory_id]
            if not bucket:
                del buckets[key]

    @classmethod
    def _attach(
        cls,
        of: dict[str, tuple[str, ...]],
        buckets: dict[str, list[str]],
        memory_id: str,
        keys: tuple[str, ...],
    ) -> None:
        cls._detach(of, buckets, memory_id)
        of[memory_id] = tuple(keys)
        for key in keys:
            buckets.setdefault(key, []).append(memory_id)

    def _unbind(self, memory_id: str) -> None:
        self._detach(self._object_of, self._objects, memory_id)
        self._detach(self._attribute_of, self._attributes, memory_id)

    def _bind(self, record: EpisodicMemoryRecord) -> None:
        self._attach(self._object_of, self._objects, record.memory_id, record.object_ids)
        self._attach(self._attribute_of, self._attributes, record.memory_id, record.attribute_ids)

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

    def _rename_binding(
        self,
        of: dict[str, tuple[str, ...]],
        buckets: dict[str, list[str]],
        old: str,
        new: str,
        kind: str,
    ) -> int:
        if not old or not new:
            raise ValueError(f"episodic {kind} rename requires both identifiers")
        bucket = buckets.pop(old, None)
        if bucket is None:
            raise ValueError(f"no episodic records are bound to {kind} {old!r}")
        if new in buckets:
            raise ValueError(f"{kind} {new!r} is already bound; rename would merge two {kind}s")
        buckets[new] = bucket
        for memory_id in bucket:
            of[memory_id] = tuple(new if held == old else held for held in of[memory_id])
        return len(bucket)

    def rename_object(self, old_object_id: str, new_object_id: str) -> int:
        """只换绑定名，不碰记录本身——这样「同一对象换了名字」才是一件能被测的事。"""

        return self._rename_binding(
            self._object_of, self._objects, str(old_object_id), str(new_object_id), "object"
        )

    def rename_attribute(self, old_attribute_id: str, new_attribute_id: str) -> int:
        """属性维的改名与对象维同形：动索引不动记录，且拒绝把两枚属性并成一枚（PLAN-N4-12 §1.5）。"""

        return self._rename_binding(
            self._attribute_of,
            self._attributes,
            str(old_attribute_id),
            str(new_attribute_id),
            "attribute",
        )

    def attribute_identifiers(self) -> tuple[str, ...]:
        """当前有绑定的属性标识；空元组＝没人绑，不代表库里没有记录（DEBT-G58 同族）。"""

        return tuple(self._attributes)

    def retrieve(
        self,
        cue: torch.Tensor,
        *,
        limit: int = 1,
        episode_id: str | None = None,
        object_id: str | None = None,
        attribute: str | None = None,
    ) -> tuple[EpisodicMemoryHit, ...]:
        """Rank records by cue similarity; `episode_id`, `object_id` and `attribute` only narrow it.

        `episode_id=None` (the default) searches every record, which is what the
        product's per-step caller does today — the storage layer still has no
        session gate, and DEBT-G106 stays open until a caller asks for one.

        `object_id=None` (the default) keeps today's candidate set bit-for-bit
        (PLAN-N4-11 J-N4h-1).  Naming an object narrows candidates to the records
        bound to it; naming an object with no binding raises instead of returning
        `()`, because "zero hits" and "this key does not exist" must not collapse
        into one answer (DEBT-G58, DEBT-G122).

        `attribute=None` (the default) is neutral the same way, and the attribute
        channel is a **third conjunct**, never a reweighting: scores are computed
        from `cue` only (PLAN-N4-12 G-N4s-1), so swapping two records'
        `attribute_ids` moves the candidate set without moving any score.
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
        attributed: set[str] | None = None
        if attribute is not None:
            if str(attribute) not in self._attributes:
                raise ValueError(f"no episodic records are bound to attribute {attribute!r}")
            attributed = set(self._attributes[str(attribute)])
        records = tuple(
            record
            for record in self._records.values()
            if (episode_id is None or record.episode_id == episode_id)
            and (bound is None or record.memory_id in bound)
            and (attributed is None or record.memory_id in attributed)
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
            "implicit_attribute_count": self.implicit_attribute_count,
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
            # ㊵-757（PLAN-N4-12 §1.2）：属性键在存量件里缺席是**正常**的，所以只出版计数，不响亮拒绝。
            if "attribute_ids" not in item:
                store.implicit_attribute_count += 1
            store.write(EpisodicMemoryRecord.from_payload(item, device=device))
        # Optional keys: a v1 payload written before DEBT-G99 simply has none of them.
        store._expired = [str(item) for item in payload.get("expired_memory_ids", ())]
        store.evicted_count = int(payload.get("evicted_count", 0))
        # 这两列 implicit_* 量的是"本次载入时有几条记录没声明来源／属性位"，由上面的循环现数；
        # 不去读件里那个历史值，否则回填会把刚数出来的数覆盖成 0。
        return store
