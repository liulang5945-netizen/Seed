"""Window-set addressing: identity is the set of every 16-byte window of a record.

Definition
----------
For a record ``r`` let ``W(r) = { H(r[i:i+WINDOW_BYTES]) : 0 <= i <= len(r) - WINDOW_BYTES }``
with ``H(block) = sha256(block).hexdigest()[:16]``.  ``identity(r)`` returns those
fingerprints **with multiplicity** (one token per offset); equality/merge uses the
full ordered tuple, coverage uses the multiset of tokens.

A query ``q`` is turned into the same multiset ``W(q)``.  A record is anchored iff
its multiset is **covered** by the query's multiset with multiplicity::

    for every token t: count_q(t) >= count_r(t)      (and identity(r) is not empty)

Multiplicity was chosen over set inclusion, and it is frozen here.  Reason: set
inclusion lets one occurrence in the query pay for a window that occurs twice in
the record (repeated table rows, repeated separators), which is a strictly weaker
promise for the same cost -- counting is one extra dict lookup and adds no numeric
constant.  Multiset coverage implies set inclusion, never the other way.

No protection band
------------------
Every window of the identity lies **entirely inside** ``r`` by construction.  For
any prefix ``p`` and suffix ``s`` and every ``i`` in ``0 .. len(r)-16``, the window
at offset ``|p| + i`` in ``p + r + s`` is exactly ``r[i:i+16]``, so ``identity(r)``
is covered by ``identity(p + r + s)`` (extra windows that cross the joins are
irrelevant -- coverage is one-directional).  There is nothing to protect: the old
``PROTECT = 6`` band belonged to a cut predicate that read bytes *before* the cut;
a fixed 16-byte window reads only its own bytes.  ``PROTECT`` is therefore absent,
not defaulted.

Shortest addressable length
---------------------------
``MIN_ADDRESSABLE = WINDOW_BYTES = 16`` bytes, exactly: a 16-byte record has one
window; a 15-byte record has none and is rejected alone.  An empty identity is
**not** addressable, because ``identity(r) = {}`` is covered by every query and
would anchor every record (the same counterexample the old one-block floor
forbade).  The bound is necessary **and sufficient**: length alone decides whether
the identity is non-empty (content decides nothing here).

Fallback
--------
Unanchored reads are ranked by the frozen window overlap
``|set(I(r)) & set(W(q))| / |set(I(r))|`` descending, then address index
ascending.  ``fallback_basis`` is ``window_overlap`` when the chosen record shares
at least one window, else ``address_order``.  No threshold is involved: the
fallback decides *who* is printed, never *whether*.

Impersonation cost (what this definition does and does not buy)
--------------------------------------------------------------
If ``identity(r)`` is covered by ``W(q)`` then ``q`` must contain every 16-byte
window of ``r`` **as a substring, with the record's multiplicity**.  That does
**not** imply that ``q`` contains ``r`` contiguously, and it does not imply that
``q`` must reproduce ``r`` almost verbatim.  Counterexample (order loss): let
``q = r[0:16] + r[1:17] + r[2:18] + ...``; every window of ``r`` occurs in ``q``
at least as often as in ``r``, so the anchor fires, while ``q`` is a reassembly
(roughly 16x longer) that never contains ``r`` as a contiguous run.  The honest
statement is therefore: the identity forces *window coverage*, not *verbatim
contiguity*; the claim "almost verbatim reproduction is required" cannot be
derived from this definition.

Failure constructions (each checks one design choice)
-----------------------------------------------------
* Empty identity kept: ``identity(r) = {}`` is covered by every query, so every
  query anchors a sub-16-byte record.  Rejected at write time, floor one window.
* Set inclusion instead of multiplicity: an entry with a repeated window (two
  identical table rows) is anchored by a query that carries that window once.
* Order kept in the comparison (contiguous sequence): a prefix/suffix can move
  cuts, which is exactly the failure the old chunk definition had to bandage with
  ``PROTECT``; windows need no such repair.
* Low-entropy windows: a record whose identity is one 16-byte run of spaces is
  covered by any query containing that run, so it is anchored by unrelated
  queries.  This is a real specificity limit of any fixed-window identity and is
  reported per group, not hidden.

Write policy
------------
Append-only, never a batch rollback, never an eviction, never a silent overwrite.
A record is rejected **alone** only when its identity is empty (``len(r) < 16``).
Identity containment is *not* a rejection reason: under a window identity a
heading or a table row is legitimately a substring of a longer record, and
rejecting containment would filter exactly the short entries this definition
exists to admit.  Multiple anchors are a reading (``ambiguous_multiple_anchors``)
and still produce a top-1 output, as frozen.
"""

from __future__ import annotations

import hashlib
from typing import Any

WINDOW_BYTES = 16
MIN_ADDRESSABLE = WINDOW_BYTES
FINGERPRINT_HEX = 16
START_CAPACITY = 2


def _fingerprint(block: bytes) -> str:
    return hashlib.sha256(block).hexdigest()[:FINGERPRINT_HEX]


def windows(raw: bytes) -> list[tuple[int, str]]:
    """Every fully internal 16-byte window as ``(offset, fingerprint)``."""
    last = len(raw) - WINDOW_BYTES
    return [
        (offset, _fingerprint(raw[offset : offset + WINDOW_BYTES])) for offset in range(last + 1)
    ]


def identity(raw: bytes) -> tuple[str, ...]:
    """Multiset of window fingerprints, in offset order."""
    return tuple(fingerprint for _offset, fingerprint in windows(raw))


def features(raw: bytes) -> tuple[str, ...]:
    """Query key: the same multiset of window fingerprints."""
    return identity(raw)


def counts_of(tokens: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def covers(record_counts: dict[str, int], query_counts: dict[str, int]) -> bool:
    """Multiset coverage: every window of the record appears in the query."""
    if not record_counts:
        return False
    return all(query_counts.get(token, 0) >= need for token, need in record_counts.items())


def window_overlap(record_set: frozenset[str], query_set: frozenset[str]) -> float:
    """Frozen fallback ratio ``|I(r) & W(q)| / |I(r)|``."""
    if not record_set:
        return 0.0
    return len(record_set & query_set) / len(record_set)


def window_address_audit() -> dict[str, Any]:
    """Numeric surface of the window rule, for the freeze/run tolerance audit."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()
        ):
            names.append(node.targets[0].id)
    constants = {
        name: globals()[name]
        for name in names
        if isinstance(globals().get(name), (int, float))
        and not isinstance(globals().get(name), bool)
    }
    scope = ("read", "covers", "window_overlap", "identity", "windows")
    literals: dict[str, list[float]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in scope:
            values = sorted(
                {
                    float(item.value)
                    for item in ast.walk(node)
                    if isinstance(item, ast.Constant)
                    and isinstance(item.value, (int, float))
                    and not isinstance(item.value, bool)
                }
            )
            literals[node.name] = values
    return {
        "constants": constants,
        "read_path_literals": literals,
        "protection": {
            "band_present": False,
            "MIN_ADDRESSABLE_matches_definition": bool(MIN_ADDRESSABLE == WINDOW_BYTES),
            "MIN_ADDRESSABLE": MIN_ADDRESSABLE,
            "WINDOW_BYTES": WINDOW_BYTES,
        },
    }


def _payload_bytes(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return 8


def _value_key(value: Any) -> Any:
    return value if isinstance(value, (bytes, bytearray, str)) else repr(value)


class WindowAddressTable:
    """Window-set address table: append-only, no rollback, no eviction."""

    def __init__(self, start_capacity: int = START_CAPACITY) -> None:
        self.capacity = int(start_capacity)
        self.peak_capacity = int(start_capacity)
        self.addresses: list[dict[str, Any]] = []
        self.expansions = 0
        self.merged = 0
        self.appended = 0
        self.rejected_records = 0
        self.rejected_reasons: dict[str, int] = {}
        self.replaced = 0
        self.evictions = 0
        self.silent_overwrites = 0
        self.rollbacks = 0
        self.rejected_batches = 0

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

    def find(self, ident: tuple[str, ...]) -> int | None:
        for index, address in enumerate(self.addresses):
            if address["identity"] == ident:
                return index
        return None

    def value_at(self, index: int) -> Any:
        values = self.addresses[index]["values"]
        return values[-1] if values else None

    def _admission(self, ident: tuple[str, ...], value: Any) -> str | None:
        if not ident:
            return "window_identity_empty"
        existing = self.find(ident)
        if existing is not None:
            if _value_key(self.value_at(existing)) == _value_key(value):
                return "merge"
            return "identity_collision_different_value"
        return None

    def write_batch_checked(self, items: list[tuple[bytes, Any]]) -> dict[str, Any]:
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
                rejected.append(
                    {"item": position, "reason": reason, "identity_blocks": len(set(ident))}
                )
                continue
            while len(self.addresses) >= self.capacity:
                self.capacity *= 2
                self.expansions += 1
                self.peak_capacity = max(self.peak_capacity, self.capacity)
            self.addresses.append(
                {
                    "identity": ident,
                    "identity_counts": counts_of(ident),
                    "identity_set": frozenset(ident),
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

    def read(self, raw: bytes) -> dict[str, Any]:
        query = features(raw)
        query_counts = counts_of(query)
        query_set = frozenset(query)
        ranked: list[dict[str, Any]] = []
        for index, address in enumerate(self.addresses):
            ident_set = address["identity_set"]
            shared = len(ident_set & query_set)
            ratio = (shared / len(ident_set)) if ident_set else 0.0
            contained = shared == len(ident_set) and covers(
                address["identity_counts"], query_counts
            )
            ranked.append(
                {
                    "index": index,
                    "covered": shared,
                    "ratio": ratio,
                    "contained": contained,
                    "shared": shared,
                    "overlap": ratio,
                }
            )
        anchors = [row for row in ranked if row["contained"]]
        anchors.sort(key=lambda row: (-row["ratio"], -row["covered"], row["index"]))
        if len(anchors) == 1:
            decision = "matched"
        elif len(anchors) > 1:
            decision = "ambiguous_multiple_anchors"
        else:
            decision = "no_match"
        if anchors:
            pool = anchors
            top = pool[0]
            fallback = False
            fallback_reason = "none"
        else:
            ranked.sort(key=lambda row: (-row["overlap"], row["index"]))
            pool = ranked
            top = pool[0] if pool else None
            fallback = top is not None
            if top is None:
                fallback_reason = "empty_table"
            elif top["shared"] > 0:
                fallback_reason = "no_anchor"
            else:
                fallback_reason = "no_overlap"
        output: dict[str, Any] | None = None
        value = None
        unit = "none"
        if top is not None:
            address = self.addresses[top["index"]]
            if anchors:
                start, end = 0, len(address["raw"])
                unit_index = 0
            else:
                start, end, unit_index = self._sentence_span(address["raw"], query_set)
            unit = "record" if start == 0 and end == len(address["raw"]) else "sentence"
            slice_bytes = address["raw"][start:end]
            output = {
                "record": top["index"],
                "start": start,
                "end": end,
                "length": end - start,
                "sha256": hashlib.sha256(slice_bytes).hexdigest(),
                "full_record": start == 0 and end == len(address["raw"]),
                "unit": unit,
                "unit_index": unit_index,
            }
            value = self.value_at(top["index"])
        if anchors:
            fallback_basis = "none"
        elif top is not None and top["shared"] > 0:
            fallback_basis = "window_overlap"
        else:
            fallback_basis = "address_order"
        top1 = float(top["overlap"] if fallback else top["ratio"]) if top is not None else 0.0
        top2 = float(pool[1]["overlap"] if fallback else pool[1]["ratio"]) if len(pool) > 1 else 0.0
        return {
            "decision": decision,
            "address": top["index"] if top is not None else None,
            "value": value,
            "anchors": [row["index"] for row in anchors],
            "ambiguous_candidates": len(anchors),
            "no_anchor": not anchors,
            "fallback": fallback,
            "fallback_reason": fallback_reason,
            "fallback_basis": fallback_basis,
            "unit": unit,
            "covered_blocks": top["covered"] if top is not None else 0,
            "coverage_ratio": round(float(top["ratio"]) if top is not None else 0.0, 9),
            "shared_blocks": top["shared"] if top is not None else 0,
            "overlap_ratio": round(float(top["overlap"]) if top is not None else 0.0, 9),
            "contained": bool(top["contained"]) if top is not None else False,
            "output": output,
            "query_blocks": len(set(query)),
            "top1": round(top1, 9),
            "top2": round(top2, 9),
            "margin": round(top1 - top2, 9),
            "unique_features_in_query": 0,
            "unknown_features": 0,
        }

    def _sentence_span(self, raw: bytes, query_set: frozenset[str]) -> tuple[int, int, int]:
        from taiji.sc_sfk import sentences

        spans = sentences(raw)
        if len(spans) == 1:
            return 0, len(raw), 0
        best: tuple[tuple[float, int, int], int, int, int] | None = None
        for index, (start, end) in enumerate(spans):
            sentence_set = frozenset(features(raw[start:end]))
            ratio = window_overlap(sentence_set, query_set)
            key = (-ratio, -len(sentence_set & query_set), index)
            if best is None or key < best[0]:
                best = (key, start, end, index)
        assert best is not None
        return best[1], best[2], best[3]

    def target_of(self, raw: bytes) -> tuple[int | None, float]:
        query_set = frozenset(features(raw))
        best_index, best_ratio = None, -1.0
        for index, address in enumerate(self.addresses):
            ratio = window_overlap(address["identity_set"], query_set)
            if ratio > best_ratio:
                best_index, best_ratio = index, ratio
        return best_index, best_ratio
