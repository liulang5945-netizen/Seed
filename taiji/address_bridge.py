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
* ``rejected_batches`` / ``rollbacks`` -- the batch would bind one key to a
  different value (or carries two values for one key).  The batch is validated
  before it is applied; the table is restored from a snapshot, so "rolled back"
  is byte equality, not a promise.

There is exactly **one** conflict rule: the first writer keeps the address and
the later batch is rejected.  R11's ``replace`` rule is deliberately gone: a
bridge whose default is "the last writer wins" can silently re-point an entity,
and R14 measured that the difference is not cosmetic.  A caller cannot select a
different rule because there is no parameter and no branch for one.

Correction is **not** a second conflict rule.  ``correct`` is a separate
compare-and-swap operation: the caller must present the value currently bound to
the address, the whole batch is validated before any mutation, and every applied
correction is appended to an audit log (old value, new value).  A plain
``write`` of a conflicting value is still rejected, so no blind later writer can
re-point an entity.  Old values are never readable as current again; they remain
only in the log.

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
    return first_occurrence(query, pattern) is not None


def first_occurrence(query: tuple[str, ...], pattern: tuple[str, ...]) -> int | None:
    """Start index of the first contiguous occurrence, or None."""
    width = len(pattern)
    if width == 0 or width > len(query):
        return None
    for start in range(len(query) - width + 1):
        if query[start : start + width] == pattern:
            return start
    return None


MASK_CHARACTER = "\x00"


class BridgeTable:
    """Phrase -> (slot, name) records with merge, first-writer-wins and rollback."""

    def __init__(
        self,
        start_capacity: int = START_CAPACITY,
        key_fn: Any = phrase_key,
        read_mode: str = "token",
        allowed_authorities: tuple[str, ...] = (),
    ) -> None:
        self.key_fn = key_fn
        self.read_mode = read_mode
        self.allowed_authorities = tuple(allowed_authorities)
        self.capacity = int(start_capacity)
        self.peak_capacity = int(start_capacity)
        self.records: list[dict[str, Any]] = []
        self.batches = 0
        self.appended = 0
        self.merged = 0
        self.rejected_batches = 0
        self.rollbacks = 0
        self.rejected_reasons: dict[str, int] = {}
        self.correction_batches = 0
        self.corrected = 0
        self.rejected_corrections = 0
        self.corrections: list[dict[str, Any]] = []

    # -- bookkeeping ---------------------------------------------------------
    @property
    def generation(self) -> int:
        return self.peak_capacity // START_CAPACITY - 1

    def current(self, index: int) -> Any:
        return self.records[index]["value"]

    def digest(self) -> str:
        """Byte-level digest of the whole table, counters included."""
        payload = {
            "capacity": self.capacity,
            "peak_capacity": self.peak_capacity,
            "batches": self.batches,
            "appended": self.appended,
            "merged": self.merged,
            "rejected_batches": self.rejected_batches,
            "rollbacks": self.rollbacks,
            "rejected_reasons": dict(sorted(self.rejected_reasons.items())),
            "correction_batches": self.correction_batches,
            "corrected": self.corrected,
            "rejected_corrections": self.rejected_corrections,
            "allowed_authorities": list(self.allowed_authorities),
            "corrections": self._correction_payload(),
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
                "value": value_key(record["value"]),
                "count": record["count"],
            }
            for record in self.records
        ]

    def _correction_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "key": row["key"],
                "from": value_key(row["from"]),
                "to": value_key(row["to"]),
                "authority": row.get("authority", ""),
                "reason": row.get("reason", ""),
            }
            for row in self.corrections
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "peak_capacity": self.peak_capacity,
            "records": [
                {"key": record["key"], "value": record["value"], "count": record["count"]}
                for record in self.records
            ],
            "counters": (self.appended, self.merged),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.capacity = snapshot["capacity"]
        self.peak_capacity = snapshot["peak_capacity"]
        self.records = [
            {"key": record["key"], "value": record["value"], "count": record["count"]}
            for record in snapshot["records"]
        ]
        self.appended, self.merged = snapshot["counters"]

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
            # Nothing has been applied yet: validation ends before any mutation.
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
                self.records.append({"key": key, "value": value, "count": 1})
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
            raise AssertionError("unreachable: an unequal binding was rejected above")

        return {
            "action": "accepted",
            "items": len(items),
            "batch": self.batches,
            "slots": len(self.records),
            "appended": sum(1 for row in planned if row["action"] == "appended"),
            "merged": sum(1 for row in planned if row["action"] == "merged_duplicate"),
            "items_detail": planned,
        }

    def correct(
        self,
        items: Iterable[tuple[str | bytes, Any, Any]],
        *,
        authority: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        """Explicit compare-and-swap correction, appended to an audit log.

        ``items`` are ``(text, expected_value, new_value)``.  The whole batch is
        validated before anything is mutated:

        * the address must exist (``rejected_batch_unknown_address``);
        * the value currently bound must equal ``expected_value`` byte for byte
          (``rejected_batch_correction_mismatch``) -- the expected value *is* the
          credential, so a caller that does not know the current value cannot
          re-point the address;
        * the new value must differ (``rejected_batch_correction_no_change``);
        * one key may not appear twice (``rejected_batch_address_collision`` for
          different new values, ``rejected_batch_correction_duplicate_key`` for
          the same one).

        ``authority`` and ``reason`` are provenance, not a second credential.
        When ``allowed_authorities`` is non-empty, a missing authority is
        rejected as ``rejected_batch_missing_authority`` and an unlisted one as
        ``rejected_batch_unknown_authority``; both reject before any row is
        planned.  An empty allow-list keeps the earlier behaviour (any caller
        may correct) and records the empty provenance.

        On acceptance every row appends ``{key, from, to, authority, reason}`` to
        ``corrections`` and the current value becomes ``new_value``.  ``count``
        is not touched: a correction is not an observation.  The old value is
        never readable again; it survives only in the log.
        """
        items = list(items)
        self.correction_batches += 1
        digest_before = self.content_digest()
        if self.allowed_authorities:
            if not authority:
                self.rejected_corrections += 1
                return {
                    "action": "rejected_batch_missing_authority",
                    "items": len(items),
                    "batch": self.correction_batches,
                    "digest_unchanged": self.content_digest() == digest_before,
                }
            if authority not in self.allowed_authorities:
                self.rejected_corrections += 1
                return {
                    "action": "rejected_batch_unknown_authority",
                    "items": len(items),
                    "authority": authority,
                    "batch": self.correction_batches,
                    "digest_unchanged": self.content_digest() == digest_before,
                }
        planned: list[dict[str, Any]] = []
        seen: dict[Any, str] = {}
        for position, (text, expected, new_value) in enumerate(items):
            key = self.key_fn(text)
            if not key:
                self.rejected_corrections += 1
                return {
                    "action": "rejected_batch_correction_key_too_short",
                    "items": len(items),
                    "rejected_item": position,
                    "batch": self.correction_batches,
                    "digest_unchanged": self.content_digest() == digest_before,
                }
            index = self.find(key)
            if index is None:
                self.rejected_corrections += 1
                return {
                    "action": "rejected_batch_unknown_address",
                    "items": len(items),
                    "rejected_item": position,
                    "batch": self.correction_batches,
                    "digest_unchanged": self.content_digest() == digest_before,
                }
            bound = value_key(self.current(index))
            expected_key = value_key(expected)
            new_key = value_key(new_value)
            if bound != expected_key:
                self.rejected_corrections += 1
                return {
                    "action": "rejected_batch_correction_mismatch",
                    "items": len(items),
                    "rejected_item": position,
                    "bound": bound,
                    "attempted_expected": expected_key,
                    "batch": self.correction_batches,
                    "digest_unchanged": self.content_digest() == digest_before,
                }
            if new_key == expected_key:
                self.rejected_corrections += 1
                return {
                    "action": "rejected_batch_correction_no_change",
                    "items": len(items),
                    "rejected_item": position,
                    "batch": self.correction_batches,
                    "digest_unchanged": self.content_digest() == digest_before,
                }
            if key in seen:
                self.rejected_corrections += 1
                action = (
                    "rejected_batch_address_collision"
                    if seen[key] != new_key
                    else "rejected_batch_correction_duplicate_key"
                )
                return {
                    "action": action,
                    "items": len(items),
                    "rejected_item": position,
                    "batch": self.correction_batches,
                    "digest_unchanged": self.content_digest() == digest_before,
                }
            seen[key] = new_key
            planned.append(
                {
                    "item": position,
                    "address": index,
                    "key": text,
                    "from": self.current(index),
                    "to": new_value,
                }
            )
        for row in planned:
            self.records[row["address"]]["value"] = row["to"]
            self.corrections.append(
                {
                    "key": row["key"],
                    "from": row["from"],
                    "to": row["to"],
                    "authority": authority,
                    "reason": reason,
                }
            )
            self.corrected += 1
        return {
            "action": "corrected",
            "items": len(items),
            "batch": self.correction_batches,
            "corrected": len(planned),
            "items_detail": [
                {"item": row["item"], "address": row["address"], "action": "corrected"}
                for row in planned
            ],
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

    def _earliest_contained(self, query: tuple[str, ...], reason: str) -> dict[str, Any]:
        """The stored key that starts earliest inside ``query``.

        Reading order decides, not key length: a length rule cannot separate
        "the object wording and the place wording are both five characters".
        Two different records that start at the same earliest position abstain
        (``ambiguous_same_start``): the read must not pick one silently.
        """
        best_start: int | None = None
        winners: list[tuple[int, int, tuple[str, ...]]] = []
        for index, record in enumerate(self.records):
            key = tuple(record["key"])
            start = first_occurrence(query, key)
            if start is None:
                continue
            if best_start is None or start < best_start:
                best_start = start
                winners = [(index, start, key)]
            elif start == best_start:
                winners.append((index, start, key))
        if not winners:
            return {
                "decision": "no_match",
                "address": None,
                "value": None,
                "reason": reason,
                "span_start": None,
                "span_end": None,
                "key_chars": 0,
                "candidates": 0,
            }
        if len(winners) > 1:
            return {
                "decision": "ambiguous_same_start",
                "address": None,
                "value": None,
                "reason": "two_records_share_the_earliest_start",
                "span_start": None,
                "span_end": None,
                "key_chars": 0,
                "candidates": len(winners),
                "addresses": [row[0] for row in winners],
            }
        index, start, key = winners[0]
        return {
            "decision": "matched",
            "address": index,
            "value": self.current(index),
            "reason": "single_earliest_containment",
            "span_start": start,
            "span_end": start + len(key),
            "key_chars": len(key),
            "candidates": 1,
        }

    def read_sentence(self, query: str | bytes, max_passes: int = 2) -> dict[str, Any]:
        """Repeated passes over one sentence: answer, mask that span, answer again.

        Pass rule (fixed, not tuned): the key that starts earliest answers; a
        same-start collision between two records abstains.  Masking rule (fixed,
        not tuned): the **first** occurrence of that record's whole character
        sequence is replaced by ``MASK_CHARACTER`` repeated to the same width.
        The width is preserved, so positions after the span do not shift and the
        next pass sees the same sentence minus exactly the answered wording.

        ``max_passes`` bounds the loop (default 2 = the R15 reading).  The loop
        stops on the first non-answer, on a repeated address, or when the bound
        is reached; ``stopped_at_pass`` and ``final_exit`` say which of the three
        happened, so "answered fewer than expected" is never reported without
        the layer it stopped at.

        Exit names: ``answered_pair`` (two distinct entities), ``answered_triple``
        (three), ``answered_multi`` (four or more), ``answered_single`` (one
        entity), ``no_match`` (nothing found), ``ambiguous_same_start`` (two
        records claim the same earliest position -- abstain, never pick one),
        ``same_address_repeated`` (the same key occurs twice, so it is one
        entity, not two), ``empty_key``.
        """
        if max_passes < 1:
            raise ValueError("max_passes must be at least 1")
        characters = _decoded(query)
        if not characters:
            return {
                "decision": "empty_key",
                "answers": [],
                "passes": [],
                "distinct_addresses": 0,
                "query_chars": 0,
                "mask": None,
                "masks": [],
                "stopped_at_pass": 0,
                "final_exit": "empty_key",
            }
        passes: list[dict[str, Any]] = []
        answers: list[dict[str, Any]] = []
        masks: list[dict[str, Any]] = []
        remaining = tuple(characters)
        for pass_index in range(1, max_passes + 1):
            reason = "not_contained" if pass_index == 1 else "masked_no_match"
            result = self._earliest_contained(remaining, reason)
            passes.append({"pass": pass_index, **result})
            if result["decision"] != "matched":
                break
            if any(row["address"] == result["address"] for row in answers):
                passes[-1]["decision"] = "same_address_repeated"
                passes[-1]["reason"] = "same_key_answered_twice"
                break
            start = result["span_start"]
            width = result["key_chars"]
            answers.append(
                {
                    "address": result["address"],
                    "value": result["value"],
                    "span_start": start,
                    "span_end": result["span_end"],
                    "key_chars": width,
                }
            )
            masked = remaining[:start] + (MASK_CHARACTER,) * width + remaining[start + width :]
            masks.append(
                {
                    "character": "\\u0000",
                    "rule": "first occurrence of the answered key, same width, positions preserved",
                    "pass": pass_index,
                    "span_start": start,
                    "span_end": result["span_end"],
                    "masked_query": "".join(masked),
                }
            )
            remaining = masked
        count = len(answers)
        if count == 2:
            decision = "answered_pair"
        elif count == 3:
            decision = "answered_triple"
        elif count > 3:
            decision = "answered_multi"
        elif count == 1 and passes and passes[-1]["decision"] == "same_address_repeated":
            decision = "answered_single_duplicate_key"
        elif count == 1:
            decision = "answered_single"
        else:
            decision = passes[0]["decision"] if passes else "empty_key"
        return {
            "decision": decision,
            "answers": answers,
            "passes": passes,
            "distinct_addresses": len({row["address"] for row in answers}),
            "query_chars": len(characters),
            "mask": masks[0] if masks else None,
            "masks": masks,
            "stopped_at_pass": len(passes),
            "final_exit": passes[-1]["decision"] if passes else "empty_key",
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
