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

* key   -- the phrase's token *set* (order-free, no positions, deduped).  The
  token unit is a parameter: R11 used 2-byte windows (``phrase_key``); R12 adds
  UTF-8 code points (``codepoint_key``) and code points with their character
  index (``codepoint_offset_key``).  R13 adds the identity arm: the *ordered*
  characters (``sequence_key``) read by containment (``read_mode="sequence"``),
  because a set of characters cannot tell "shares one character" from "same
  phrase".  Nothing else in the table changes.
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
with a reason.  With ``read_mode="sequence"`` the anchor rule is containment
instead of token sharing (see ``read_sequence``).  A read never invents a value:
``value`` is ``None`` in both non-answer branches (unlike ``AddressTable.read``,
which always returns top-1).

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
    """Order-free set of 2-byte tokens, hex-encoded.  Deterministic, no state.

    **This is the R11 key unit and must stay byte-identical**: R11's frozen
    reading is reproduced by this function alone.  Named ``byte_pair_key`` in
    R12 reports; ``phrase_key`` stays the default for older callers.
    """
    data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    return frozenset(
        data[index : index + TOKEN_WIDTH].hex() for index in range(0, len(data) - 1, TOKEN_WIDTH)
    )


def byte_pair_key(text: str | bytes) -> frozenset[str]:
    """R12 name for the R11 unit: the 2-byte window set (byte-identical)."""
    return phrase_key(text)


def _decoded(text: str | bytes) -> str | None:
    """The text as characters, or None when the bytes are not valid UTF-8."""
    if isinstance(text, str):
        return text
    try:
        return bytes(text).decode("utf-8")
    except UnicodeDecodeError:
        return None


def codepoint_key(text: str | bytes) -> frozenset[str]:
    """Order-free set of UTF-8 code points (one whole character per token).

    Bytes are decoded strictly: a byte slice that cuts a character in half is
    not text, so it has an empty key and can never address a record.
    """
    characters = _decoded(text)
    return frozenset() if characters is None else frozenset(characters)


def codepoint_offset_key(text: str | bytes) -> frozenset[str]:
    """Code points tagged with their character index: the offset-aware arm.

    Diagnostic only in R12: it is the pre-registered treatment for positional
    drift, and the report must state which failure class it does *not* treat.
    """
    characters = _decoded(text)
    if characters is None:
        return frozenset()
    return frozenset(f"{index}:{character}" for index, character in enumerate(characters))


def sequence_key(text: str | bytes) -> tuple[str, ...]:
    """The characters in order, multiplicity kept: the R13 identity arm.

    A set of characters cannot tell "shares one character" from "same phrase";
    a sequence can.  Read side pairs this with ``read_mode="sequence"``, which
    answers only when a record's whole sequence occurs *contiguously* in the
    query.
    """
    characters = _decoded(text)
    return () if characters is None else tuple(characters)


def value_key(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, str)):
        return value.decode("utf-8", "replace") if isinstance(value, (bytes, bytearray)) else value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def contains_sequence(query: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    """True when ``pattern`` occurs contiguously inside ``query`` (both non-empty)."""
    width = len(pattern)
    if width == 0 or width > len(query):
        return False
    return any(query[start : start + width] == pattern for start in range(len(query) - width + 1))


class BridgeTable:
    """Phrase -> (slot, name) records with merge / replace / batch rollback."""

    def __init__(
        self,
        start_capacity: int = START_CAPACITY,
        key_fn: Any = phrase_key,
        read_mode: str = "token",
        on_conflict: str = "replace",
    ) -> None:
        self.key_fn = key_fn
        self.read_mode = read_mode
        self.on_conflict = on_conflict
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
                "key": (
                    sorted(record["key"])
                    if isinstance(record["key"], frozenset)
                    else list(record["key"])
                ),
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
    def find(self, key: Any) -> int | None:
        for index, record in enumerate(self.records):
            if record["key"] == key:
                return index
        return None

    def write(self, items: Iterable[tuple[str | bytes, Any]]) -> dict[str, Any]:
        items = list(items)
        self.batches += 1
        digest_before = self.content_digest()

        first_value: dict[Any, str] = {}
        conflict_positions: list[int] = []
        for position, (text, value) in enumerate(items):
            key = self.key_fn(text)
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
            if self.on_conflict == "reject":
                bound = self.find(key)
                if bound is not None and value_key(self.current(bound)) != encoded:
                    # First writer keeps the address; the whole batch is not applied.
                    self.rejected_batches += 1
                    self.rollbacks += 1
                    reason = "address_already_bound"
                    self.rejected_reasons[reason] = self.rejected_reasons.get(reason, 0) + 1
                    return {
                        "action": "rejected_batch_address_already_bound",
                        "items": len(items),
                        "rejected_item": position,
                        "batch": self.batches,
                        "bound": value_key(self.current(bound)),
                        "attempted": encoded,
                        "digest_unchanged": self.content_digest() == digest_before,
                    }
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
            key = self.key_fn(text)
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
        key = self.key_fn(query)
        if self.read_mode == "sequence":
            return self.read_sequence(key)
        if self.read_mode != "token":
            raise ValueError(f"unknown read_mode: {self.read_mode}")
        return self.read_tokens(key)

    def read_sequence(self, key: tuple[str, ...]) -> dict[str, Any]:
        """Answer only when one record's whole sequence occurs inside the query.

        Identity, not similarity: a query that merely shares characters with a
        record cannot answer, so "shares one character" and "same phrase" stop
        being the same event.  An empty or undecodable query never answers.
        """
        if not key:
            return {
                "decision": "no_match",
                "address": None,
                "value": None,
                "anchors": [],
                "reason": "empty_key",
                "unknown_tokens": [],
                "query_tokens": 0,
            }
        anchors = [
            index
            for index, record in enumerate(self.records)
            if contains_sequence(key, tuple(record["key"]))
        ]
        if len(anchors) == 1:
            return {
                "decision": "matched",
                "address": anchors[0],
                "value": self.current(anchors[0]),
                "anchors": anchors,
                "reason": "single_containment",
                "unknown_tokens": [],
                "query_tokens": len(key),
            }
        if len(anchors) > 1:
            return {
                "decision": "ambiguous_multiple_anchors",
                "address": None,
                "value": None,
                "anchors": anchors,
                "reason": "multiple_containments",
                "unknown_tokens": [],
                "query_tokens": len(key),
            }
        return {
            "decision": "no_match",
            "address": None,
            "value": None,
            "anchors": [],
            "reason": "not_contained",
            "unknown_tokens": [],
            "query_tokens": len(key),
        }

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
