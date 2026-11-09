"""Written bridge: the phrase->name map stored as address-layer records (R11).

R6-R10b established that no closed-form read-out recovers a name from a phrase
once the exact (phrase, name) row is gone: the one-wording bank gives a
constructive zero, the two-wording bank gives a lambda-fragile middle band.
The conclusion is architectural -- the mapping has to be *written*, not fitted.

This module is the minimal implementation of that write.  It is deliberately a
separate object, not a change to ``AddressTable`` (whose identity is chunk
fingerprints, not a token set) and not a change to ``LivingCoreReference``:
nothing in the default graph imports it, so the closed-state path cannot move.

Record shape (one per (phrase, slot+name) observation):

* key   -- the phrase's 2-byte token *set* (order-free, no positions, deduped),
  the same "short substring" granularity the byte-level probes use.
* value -- (slot, name), e.g. ("place", "书架"); names never enter the key.

Write outcomes, all counted and all reachable:

* ``appended``  -- the key is new.
* ``merged``    -- same key, same value: count+1, no new record.
* ``replaced``  -- same key, different value: the old value moves to history.
* ``rejected_batches`` / ``rollbacks`` -- one batch carries two different values
  for one key.  The batch is validated before it is applied; the table is
  restored from a snapshot, so "rolled back" is byte equality, not a promise.

Read outcomes (structural, no thresholds): an address anchors a query only when
it shares a token that occurs in exactly one address.  Exactly one anchor
answers; two or more say ``ambiguous_multiple_anchors``; zero says ``no_match``
with a reason.  A read never invents a value: ``value`` is ``None`` in both
non-answer branches (unlike ``AddressTable.read``, which always returns top-1).

Zero training, zero optimizer steps, zero parameters, zero torch tensors.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

TOKEN_WIDTH = 2
START_CAPACITY = 2


def phrase_key(text: str | bytes) -> frozenset[str]:
    """Order-free set of 2-byte tokens, hex-encoded.  Deterministic, no state."""
    data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    return frozenset(
        data[index : index + TOKEN_WIDTH].hex() for index in range(0, len(data) - 1, TOKEN_WIDTH)
    )


def value_key(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, str)):
        return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class BridgeTable:
    """Phrase -> (slot, name) records with merge / replace / batch rollback."""

    def __init__(self, start_capacity: int = START_CAPACITY) -> None:
        self.capacity = int(start_capacity)
        self.peak_capacity = int(start_capacity)
        self.records: list[dict[str, Any]] = []
        self.batches = 0
        self.appended = 0
        self.merged = 0
        self.replaced = 0
        self.rejected_batches = 0
        self.rollbacks = 0
        self.rejected_reasons: dict[str, int] = {}

    # -- bookkeeping ---------------------------------------------------------
    @property
    def generation(self) -> int:
        return self.peak_capacity // START_CAPACITY - 1

    def values_at(self, index: int) -> list[Any]:
        return list(self.records[index]["values"])

    def current(self, index: int) -> Any:
        values = self.records[index]["values"]
        return values[-1] if values else None

    def history_at(self, index: int) -> list[Any]:
        return list(self.records[index]["history"])

    def digest(self) -> str:
        """Byte-level digest of the whole table, counters included."""
        payload = {
            "capacity": self.capacity,
            "peak_capacity": self.peak_capacity,
            "batches": self.batches,
            "appended": self.appended,
            "merged": self.merged,
            "replaced": self.replaced,
            "rejected_batches": self.rejected_batches,
            "rollbacks": self.rollbacks,
            "rejected_reasons": dict(sorted(self.rejected_reasons.items())),
            "records": self._record_payload(),
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def content_digest(self) -> str:
        """Digest of the addresses only: the rollback proof.

        Counters are excluded on purpose -- rejecting a batch *must* move
        ``rejected_batches`` and ``rollbacks``, so only the content can be the
        evidence that nothing was applied.
        """
        payload = {
            "capacity": self.capacity,
            "peak_capacity": self.peak_capacity,
            "records": self._record_payload(),
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _record_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "key": sorted(record["key"]),
                "values": [value_key(value) for value in record["values"]],
                "history": [value_key(value) for value in record["history"]],
                "count": record["count"],
            }
            for record in self.records
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "peak_capacity": self.peak_capacity,
            "records": [
                {
                    "key": record["key"],
                    "values": list(record["values"]),
                    "history": list(record["history"]),
                    "count": record["count"],
                }
                for record in self.records
            ],
            "counters": (self.appended, self.merged, self.replaced),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.capacity = snapshot["capacity"]
        self.peak_capacity = snapshot["peak_capacity"]
        self.records = [
            {
                "key": record["key"],
                "values": list(record["values"]),
                "history": list(record["history"]),
                "count": record["count"],
            }
            for record in snapshot["records"]
        ]
        self.appended, self.merged, self.replaced = snapshot["counters"]

    # -- write ---------------------------------------------------------------
    def find(self, key: frozenset[str]) -> int | None:
        for index, record in enumerate(self.records):
            if record["key"] == key:
                return index
        return None

    def write(self, items: Iterable[tuple[str | bytes, Any]]) -> dict[str, Any]:
        items = list(items)
        self.batches += 1
        digest_before = self.content_digest()

        first_value: dict[frozenset[str], str] = {}
        conflict_positions: list[int] = []
        for position, (text, value) in enumerate(items):
            key = phrase_key(text)
            if not key:
                self.rejected_batches += 1
                self.rejected_reasons["key_too_short"] = (
                    self.rejected_reasons.get("key_too_short", 0) + 1
                )
                self.rollbacks += 1
                return {
                    "action": "rejected_batch_key_too_short",
                    "items": len(items),
                    "rejected_item": position,
                    "batch": self.batches,
                    "digest_unchanged": self.content_digest() == digest_before,
                }
            encoded = value_key(value)
            if key not in first_value:
                first_value[key] = encoded
            elif first_value[key] != encoded:
                conflict_positions.append(position)
        if conflict_positions:
            # Nothing has been applied yet; the snapshot restore is the proof.
            snapshot = self.snapshot()
            self.restore(snapshot)
            self.rejected_batches += 1
            self.rollbacks += 1
            reason = "batch_address_collision"
            self.rejected_reasons[reason] = self.rejected_reasons.get(reason, 0) + 1
            return {
                "action": "rejected_batch_address_collision",
                "items": len(items),
                "reason": reason,
                "conflict_items": conflict_positions,
                "batch": self.batches,
                "digest_unchanged": self.content_digest() == digest_before,
            }

        planned: list[dict[str, Any]] = []
        for position, (text, value) in enumerate(items):
            key = phrase_key(text)
            index = self.find(key)
            if index is None:
                while len(self.records) >= self.capacity:
                    self.capacity *= 2
                    self.peak_capacity = max(self.peak_capacity, self.capacity)
                self.records.append({"key": key, "values": [value], "history": [], "count": 1})
                self.appended += 1
                planned.append(
                    {"item": position, "address": len(self.records) - 1, "action": "appended"}
                )
                continue
            if value_key(self.current(index)) == value_key(value):
                self.records[index]["count"] = int(self.records[index]["count"]) + 1
                self.merged += 1
                planned.append({"item": position, "address": index, "action": "merged_duplicate"})
                continue
            record = self.records[index]
            record["history"].append(record["values"][-1])
            record["values"].append(value)
            record["count"] = int(record["count"]) + 1
            self.replaced += 1
            planned.append({"item": position, "address": index, "action": "replaced"})

        return {
            "action": "accepted",
            "items": len(items),
            "batch": self.batches,
            "slots": len(self.records),
            "appended": sum(1 for row in planned if row["action"] == "appended"),
            "merged": sum(1 for row in planned if row["action"] == "merged_duplicate"),
            "replaced": sum(1 for row in planned if row["action"] == "replaced"),
            "items_detail": planned,
        }

    # -- read ----------------------------------------------------------------
    def read(self, query: str | bytes) -> dict[str, Any]:
        return self.read_tokens(phrase_key(query))

    def read_tokens(self, tokens: frozenset[str]) -> dict[str, Any]:
        document_frequency: dict[str, int] = {}
        for record in self.records:
            for token in record["key"]:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        unique = {token for token, count in document_frequency.items() if count == 1}
        anchors = [
            index for index, record in enumerate(self.records) if (record["key"] & tokens & unique)
        ]
        unknown = sorted(token for token in tokens if document_frequency.get(token, 0) == 0)
        if len(anchors) == 1:
            return {
                "decision": "matched",
                "address": anchors[0],
                "value": self.current(anchors[0]),
                "anchors": anchors,
                "reason": "single_anchor",
                "unknown_tokens": unknown,
                "query_tokens": len(tokens),
            }
        if len(anchors) > 1:
            return {
                "decision": "ambiguous_multiple_anchors",
                "address": None,
                "value": None,
                "anchors": anchors,
                "reason": "multiple_anchors",
                "unknown_tokens": unknown,
                "query_tokens": len(tokens),
            }
        return {
            "decision": "no_match",
            "address": None,
            "value": None,
            "anchors": [],
            "reason": "unknown_token" if unknown else "no_unique_token",
            "unknown_tokens": unknown,
            "query_tokens": len(tokens),
        }


def bridge_records(
    object_descriptions: dict[str, str], place_descriptions: dict[str, str]
) -> list[tuple[str, tuple[str, str]]]:
    """The 9 written records: phrase -> (slot, name).  Names never enter the key."""
    rows: list[tuple[str, tuple[str, str]]] = []
    for name, phrase in object_descriptions.items():
        rows.append((phrase, ("object", name)))
    for name, phrase in place_descriptions.items():
        rows.append((phrase, ("place", name)))
    return rows
