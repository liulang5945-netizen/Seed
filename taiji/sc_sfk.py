"""Content-defined chunk addressing (local-predicate chunking); pure Python.

Identity definition
-------------------
A record's bytes are cut at every position where a **local** gear-hash predicate
fires: mask ``0x3F`` on the rolling hash, whose low six bits are a function of the
last six bytes only.  The rolling hash is never reset and there is no
"distance from the previous cut" rule (no minimum length, no maximum length), so
the cut set is a function of the bytes alone.  Chunks shorter than
``MIN_CHUNK = 16`` bytes are dropped on emission; they are never merged or
extended, because length is looked at only after the cut set is fixed.  A chunk's
fingerprint is ``sha256(chunk)[:16].hex()``.

The record identity ``S(r)`` is the **ordered fingerprint sequence** of the chunks
that remain after dropping every chunk that overlaps the first 16 bytes or the
last 16 bytes of the record (the protection band).  The band is what makes the
identity a function of the interior only: any prefix or suffix appended to a
query changes at most the first / last chunk of the chunking, both of which lie
outside ``S(r)``.

A query's key ``C(Q)`` is the fingerprint sequence of **all** its chunks; the
query is not banded.  A record is anchored iff ``S(r)`` appears as a **contiguous
subsequence** of ``C(Q)``.  Zero, one and several anchors are recorded as
readings (``no_anchor`` / ``matched`` / ``ambiguous_candidates``); a read always
returns the top-1 record, ranked by covered blocks descending, coverage ratio
descending, address index ascending.  Nothing outside the chunk sequence can
manufacture an anchor: an anchor requires the whole interior of the target record
to be present, in order, contiguously.

Write policy
------------
Never a batch rollback, never an eviction, never a silent overwrite.  Each record
is admitted or rejected **alone**:

* rejected if fewer than two identity chunks remain (record too short to have an
  interior);
* rejected if an address with the same identity already holds a different value;
* rejected if its identity is a contiguous subsequence of an existing identity or
  an existing identity is a contiguous subsequence of it (containment clash);
* otherwise appended; the same identity with the same value is a merge.

The old ``DF = 1`` unique-word anchors, the ``DF = 0`` guard and the G-A/G-B/G-C
batch checks were deleted with the old definition; they are not kept as a second
criterion in any branch.
"""

from __future__ import annotations

import hashlib
from typing import Any

START_CAPACITY = 2
WINDOW = 16
MIN_CHUNK = 16
MASK = 0x3F
PROTECT = 16
MASK64 = 0xFFFFFFFFFFFFFFFF
GEAR_SALT = b"taiji-gear-v1"


def _gear_table() -> tuple[int, ...]:
    """Deterministic 64-bit gear table; no seed, no process state."""
    return tuple(
        int.from_bytes(hashlib.sha256(GEAR_SALT + bytes([value])).digest()[:8], "big")
        for value in range(256)
    )


GEAR = _gear_table()


def chunks(raw: bytes) -> list[tuple[int, int, str]]:
    """Content-defined chunks as ``(start, end, fingerprint)`` in order.

    The cut set is ``{i : (h_i & MASK) == 0}`` where ``h_i`` is the rolling hash
    over ``raw[:i]``.  ``MASK`` reads the low six bits, which depend only on the
    six bytes ending at ``i``; the hash is never reset, so the cut set is a
    function of the bytes alone -- a prefix moves every interior cut by exactly
    the prefix length and a suffix moves none of them.

    Chunks shorter than ``MIN_CHUNK`` are dropped at emission (their bytes stay
    invisible to the address).  Length is examined only *after* the cut set is
    fixed, so a short chunk can never move a cut point.
    """
    cuts: list[int] = []
    rolling = 0
    for index, value in enumerate(raw):
        rolling = ((rolling << 1) + GEAR[value]) & MASK64
        if (rolling & MASK) == 0:
            cuts.append(index + 1)
    blocks: list[tuple[int, int, str]] = []
    start = 0
    for end in cuts + [len(raw)]:
        if end - start >= MIN_CHUNK:
            blocks.append((start, end, _fingerprint(raw[start:end])))
        start = end
    return blocks


def _fingerprint(block: bytes) -> str:
    return hashlib.sha256(block).hexdigest()[:16]


def identity_blocks(raw: bytes) -> list[tuple[int, int, str]]:
    """Chunks outside the head/tail protection band, in order."""
    blocks = chunks(raw)
    end = len(raw) - PROTECT
    return [block for block in blocks if block[0] >= PROTECT and block[1] <= end]


def identity(raw: bytes) -> tuple[str, ...]:
    """``S(r)``: the ordered fingerprints of the interior chunks."""
    return tuple(block[2] for block in identity_blocks(raw))


def features(raw: bytes) -> tuple[str, ...]:
    """``C(Q)``: the ordered fingerprints of **all** chunks (query key)."""
    return tuple(block[2] for block in chunks(raw))


def longest_contiguous_run(
    needle: tuple[str, ...], haystack: tuple[str, ...]
) -> tuple[int, int, int]:
    """Longest run of ``needle`` that occurs contiguously in ``haystack``.

    Returns ``(length, needle_start, haystack_start)``.  ``length == len(needle)``
    with ``needle_start == 0`` means containment.
    """
    best = (0, 0, 0)
    for outer, token in enumerate(needle):
        for inner, candidate in enumerate(haystack):
            if token != candidate:
                continue
            length = 0
            while (
                outer + length < len(needle)
                and inner + length < len(haystack)
                and needle[outer + length] == haystack[inner + length]
            ):
                length += 1
            if length > best[0]:
                best = (length, outer, inner)
    return best


def is_subsequence(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    """``short`` occurs contiguously in ``long`` (including equality)."""
    if not short or len(short) > len(long):
        return False
    return longest_contiguous_run(short, long)[0] == len(short)


def _payload_bytes(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return 8


def _value_key(value: Any) -> Any:
    return value if isinstance(value, (bytes, bytearray, str)) else repr(value)


class AddressTable:
    """Content-defined chunk table: append-only, no rollback, no eviction."""

    def __init__(self, start_capacity: int = START_CAPACITY) -> None:
        self.capacity = int(start_capacity)
        self.peak_capacity = int(start_capacity)
        self.addresses: list[dict[str, Any]] = []
        self.expansions = 0
        self.merged = 0
        self.appended = 0
        self.rejected_records = 0
        self.rejected_reasons: dict[str, int] = {}
        # Structural zeros: there is no code path that could raise them.
        self.replaced = 0
        self.evictions = 0
        self.silent_overwrites = 0
        self.rollbacks = 0
        self.rejected_batches = 0

    # -- statistics ---------------------------------------------------------
    @property
    def generation(self) -> int:
        return self.expansions

    def nbytes(self) -> int:
        total = 0
        for address in self.addresses:
            total += len(address["raw"])
            total += sum(len(fingerprint) for fingerprint in address["identity"])
            total += sum(_payload_bytes(value) for value in address["values"])
            for entry in address["history"]:
                total += _payload_bytes(entry.get("value"))
        return total

    # -- batch API ----------------------------------------------------------
    def find(self, ident: tuple[str, ...]) -> int | None:
        for index, address in enumerate(self.addresses):
            if address["identity"] == ident:
                return index
        return None

    def value_at(self, index: int) -> Any:
        values = self.addresses[index]["values"]
        return values[-1] if values else None

    def _admission(self, ident: tuple[str, ...], value: Any) -> str | None:
        if len(ident) < 2:
            return "identity_too_short"
        existing = self.find(ident)
        if existing is not None:
            if _value_key(self.value_at(existing)) == _value_key(value):
                return "merge"
            return "identity_collision_different_value"
        for address in self.addresses:
            other = address["identity"]
            if is_subsequence(ident, other) or is_subsequence(other, ident):
                return "identity_containment_conflict"
        return None

    def write_batch_checked(self, items: list[tuple[bytes, Any]]) -> dict[str, Any]:
        """Write records one by one; a bad record never rolls back the batch."""
        slots_before = len(self.addresses)
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for position, (raw, value) in enumerate(items):
            raw = bytes(raw)
            ident = identity(raw)
            reason = self._admission(ident, value)
            if reason == "merge":
                index = self.find(ident)
                self.addresses[index]["count"] = int(self.addresses[index]["count"]) + 1
                self.merged += 1
                accepted.append({"item": position, "address": index, "action": "merged_duplicate"})
                continue
            if reason is not None:
                self.rejected_records += 1
                self.rejected_reasons[reason] = self.rejected_reasons.get(reason, 0) + 1
                rejected.append({"item": position, "reason": reason, "identity_blocks": len(ident)})
                continue
            while len(self.addresses) >= self.capacity:
                self.capacity *= 2
                self.expansions += 1
                self.peak_capacity = max(self.peak_capacity, self.capacity)
            self.addresses.append(
                {
                    "identity": ident,
                    "raw": raw,
                    "values": [value],
                    "history": [],
                    "count": 1,
                }
            )
            self.appended += 1
            accepted.append(
                {"item": position, "address": len(self.addresses) - 1, "action": "appended"}
            )
        return {
            "action": "accepted_with_rejections" if rejected else "accepted",
            "items": len(items),
            "slots_before": slots_before,
            "slots_after": len(self.addresses),
            "accepted_items": len(accepted),
            "accepted": accepted,
            "rejected": rejected,
        }

    def write_batch(self, items: list[tuple[bytes, Any]]) -> dict[str, Any]:
        return self.write_batch_checked(items)

    # -- read ---------------------------------------------------------------
    def read(self, raw: bytes) -> dict[str, Any]:
        """Always return a top-1 record and a legal slice of its stored bytes."""
        query = features(raw)
        ranked: list[dict[str, Any]] = []
        for index, address in enumerate(self.addresses):
            ident = address["identity"]
            run, needle_start, haystack_start = longest_contiguous_run(ident, query)
            contained = run == len(ident) and needle_start == 0
            ranked.append(
                {
                    "index": index,
                    "covered": run,
                    "ratio": (run / len(ident)) if ident else 0.0,
                    "contained": contained,
                    "needle_start": needle_start,
                    "haystack_start": haystack_start,
                }
            )
        ranked.sort(key=lambda row: (-row["covered"], -row["ratio"], row["index"]))
        anchors = [row["index"] for row in ranked if row["contained"]]
        if len(anchors) == 1:
            decision = "matched"
        elif len(anchors) > 1:
            decision = "ambiguous_multiple_anchors"
        else:
            decision = "no_match"
        top = ranked[0] if ranked else None
        output: dict[str, Any] | None = None
        value = None
        if top is not None:
            address = self.addresses[top["index"]]
            start, end = self._output_span(top, address)
            slice_bytes = address["raw"][start:end]
            output = {
                "record": top["index"],
                "start": start,
                "end": end,
                "length": end - start,
                "sha256": hashlib.sha256(slice_bytes).hexdigest(),
                "full_record": start == 0 and end == len(address["raw"]),
            }
            value = self.value_at(top["index"])
        top1 = float(top["ratio"]) if top is not None else 0.0
        top2 = float(ranked[1]["ratio"]) if len(ranked) > 1 else 0.0
        return {
            "decision": decision,
            "address": top["index"] if top is not None else None,
            "value": value,
            "anchors": anchors,
            "ambiguous_candidates": len(anchors),
            "no_anchor": not anchors,
            "covered_blocks": top["covered"] if top is not None else 0,
            "coverage_ratio": round(top1, 9),
            "contained": bool(top["contained"]) if top is not None else False,
            "output": output,
            "query_blocks": len(query),
            "top1": round(top1, 9),
            "top2": round(top2, 9),
            "margin": round(top1 - top2, 9),
            "unique_features_in_query": 0,
            "unknown_features": 0,
        }

    def _output_span(self, top: dict[str, Any], address: dict[str, Any]) -> tuple[int, int]:
        """Full record on containment, else the byte span of the best run."""
        raw = address["raw"]
        if top["contained"] or top["covered"] == 0:
            return 0, len(raw)
        blocks = identity_blocks(raw)
        first = top["needle_start"]
        last = first + top["covered"] - 1
        return blocks[first][0], blocks[last][1]

    def target_of(self, raw: bytes) -> tuple[int | None, float]:
        query = features(raw)
        best_index, best_ratio = None, -1.0
        for index, address in enumerate(self.addresses):
            ident = address["identity"]
            run = longest_contiguous_run(ident, query)[0]
            ratio = (run / len(ident)) if ident else 0.0
            if ratio > best_ratio:
                best_index, best_ratio = index, ratio
        return best_index, best_ratio
