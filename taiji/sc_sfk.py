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

The record identity ``S(r)`` is the **ordered fingerprint sequence** of every
chunk ``[a, b)`` with ``a >= PROTECT`` and ``b < len(r)``:

* ``PROTECT = WINDOW = 6`` is the true dependency width of the predicate -- six
  bytes, no more.  A cut at offset ``a`` reads ``r[a-6:a]``, so a prefix can only
  disturb cuts at offsets ``a < 6``.  That is exactly the band, and it is the
  smallest band for which the shift theorem below holds.
* ``b < len(r)`` drops only the **final** chunk, the one whose right edge is the
  buffer end.  Any suffix rewrites that chunk's bytes, so it can never be part of
  the identity; every other chunk ends at a real cut and is untouched by a suffix.

Because only whole chunks are kept, ``S(r)`` is a contiguous run of the record's
own chunk list (the dropped chunks sit at the two ends only).

A query's key ``C(Q)`` is the fingerprint sequence of **all** its chunks; the
query is not banded.  A record is anchored iff ``S(r)`` appears as a **contiguous
subsequence** of ``C(Q)``.

Shortest addressable length
---------------------------
``MIN_ADDRESSABLE = MIN_CHUNK + PROTECT + 1 = 16 + 6 + 1 = 23`` bytes.  One
identity chunk needs ``a >= 6``, ``b - a >= 16`` and ``b <= len(r) - 1``, hence
``len(r) >= 23``.  The bound is **necessary, not sufficient**: whether cuts at
``a`` and ``b`` actually exist is content-dependent, so a record of 23 bytes or
more can still be unaddressable.  A record shorter than 23 bytes has no identity
at all and is rejected alone.

Shift invariance (theorem, checkable line by line)
--------------------------------------------------
Premise: the cut predicate at offset ``i`` is ``pred(r, i) = ((h_i & 0x3F) == 0)``
with ``h_i = sum_{k<i} G[r[k]] * 2^{i-1-k} mod 2^64``; modulo 64 only the last
six terms survive, so ``pred(r, i)`` depends on ``r[i-6:i]`` alone.  Identity
keeps every chunk ``[a, b)`` with ``a >= 6``, ``b - a >= 16``, ``b < len(r)``.

Equality: for any prefix ``p`` and suffix ``s``, ``S(r)`` is a contiguous
subsequence of ``C(p + r + s)``.

* Prefix: for every offset ``i`` with ``6 <= i <= len(r)``,
  ``pred(p + r, |p| + i) = pred(r, i)`` because the six bytes ending at
  ``|p| + i`` are ``r[i-6:i]``.  Hence every chunk ``[a, b)`` of ``r`` with
  ``a >= 6`` is the chunk ``[a + |p|, b + |p|)`` of ``p + r`` with identical
  bytes and the same fingerprint.
* Suffix: for every offset ``i <= len(r)``, ``pred(r + s, i) = pred(r, i)``
  because the window still ends inside ``r``.  Hence every chunk ``[a, b)`` with
  ``b <= len(r) - 1`` survives byte-identically; only the final chunk (whose
  right edge is the buffer end) changes.
* Both: the retained chunks of ``r`` keep their order and their fingerprints,
  nothing is inserted between them, and the two ends of ``p + r + s`` can only
  contribute chunks outside ``S(r)``.  Therefore ``S(r)`` occurs contiguously.

Failure constructions (each one is a counterexample to a wrong constant):

* ``PROTECT < 6``: take ``r`` with a cut at offset ``a = 4`` and a chunk
  ``[4, 20)``; prepend one byte.  The window ending at 4 becomes ``p + r[0:4]``,
  the cut can vanish, ``[4, 20)`` merges into the first chunk and its fingerprint
  disappears.  Six is the smallest band that rules this out.
* No final-chunk drop: append one byte to ``r``.  The final chunk becomes
  ``[last_cut, len(r)+1)`` with a new fingerprint, so an identity that contained
  it loses its anchor.
* No ``MIN_CHUNK`` drop (short chunks merged instead of discarded): merging
  changes the start of the next chunk, so its bytes -- and every fingerprint
  after it -- become prefix-dependent again.
* Keep the first chunk (``a >= 0``): a 15-byte lead-in makes that chunk absorb
  the prefix, so its fingerprint differs and no record with that chunk anchors.
* Allow an **empty** identity: ``S(r) = ()`` is a contiguous subsequence of every
  ``C(Q)``, so every query anchors every such record -- the exact mis-point the
  one-block floor forbids.  This is why the floor is 1, not 0.

Reads, anchors, the fallback and the output unit
-----------------------------------------------
Zero, one and several anchors are recorded as readings (``no_anchor`` /
``matched`` / ``ambiguous_candidates``); a read always returns the top-1 record
and a legal, **complete text unit** of its stored bytes.

* Anchored: choose among the anchors by covered blocks descending, coverage ratio
  descending, address index ascending; the output is the **whole record**
  (``unit = record``).
* Not anchored (``no_anchor`` label, printed in the result): rank **every** record
  by block-level overlap -- ``shared / (|S(r)| + |C(Q)| - shared)``, the multiset
  Jaccard index between the query's chunks and the record's identity chunks --
  then by address index ascending (``fallback_basis`` is ``record_overlap`` when
  some block is shared, else ``address_order``).  ``fallback_reason`` is
  ``no_anchor`` when some block is shared and ``no_overlap`` when none is.  The
  fallback decides **who** is printed, never **whether** anything is printed, and
  it introduces no pass/reject threshold: an anchored record always outranks the
  fallback, and anchoring itself is exact (coverage ratio 1.0), not a line.
* The printed unit is always a **complete record or a complete sentence**
  (``sentences()``): boundaries sit only at a sentence terminator, a newline or
  the record end, and every boundary is a character boundary of the stored UTF-8
  text.  Within the chosen record the sentence is the one with the highest
  block-level overlap with the query, ties broken by sentence order.  A fragment
  that cuts a character, a word or a sentence is never produced -- a fallback
  shows the most likely **whole sentence** instead of debris.
* Every output carries its boundary provenance: ``unit`` (``record`` /
  ``sentence``), ``start`` / ``end`` byte offsets, ``fallback`` plus
  ``fallback_reason`` and ``fallback_basis``.  These are readings; they do not
  change whether an output is produced.

Nothing outside the chunk sequence can manufacture an anchor: an anchor requires
the whole ``S(r)`` to be present, in order, contiguously.  The one thing a short
record gives up is *specificity*, not the rule: a single-block identity certifies
at least ``MIN_CHUNK = 16`` verbatim interior bytes (plus the cut-context
condition), where a long record certifies many such blocks.  An empty identity
would anchor every query -- that is why the floor is one block, not zero.

Write policy
------------
Never a batch rollback, never an eviction, never a silent overwrite.  Each record
is admitted or rejected **alone**:

* rejected if **no** identity chunk remains (record shorter than 23 bytes, or no
  cut pair inside the addressable band);
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
WINDOW = 6
MIN_CHUNK = 16
MASK = 0x3F
PROTECT = WINDOW
MIN_ADDRESSABLE = MIN_CHUNK + PROTECT + 1
MASK64 = 0xFFFFFFFFFFFFFFFF
GEAR_SALT = b"taiji-gear-v1"
SENTENCE_TERMINATORS = frozenset("。！？；…\n")


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
    """Chunks that start past the prefix band and end before the buffer end.

    ``block[0] >= PROTECT`` keeps only cuts whose six-byte window lies inside the
    record; ``block[1] < len(raw)`` drops the final chunk, whose right edge moves
    with any suffix.
    """
    blocks = chunks(raw)
    return [block for block in blocks if block[0] >= PROTECT and block[1] < len(raw)]


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


def block_overlap(ident: tuple[str, ...], query: tuple[str, ...]) -> tuple[int, int, float]:
    """Multiset block overlap: ``(shared, union, shared / union)``.

    ``shared`` counts each fingerprint as often as it occurs in both sequences,
    so a block repeated twice in the record and once in the query is shared once.
    The ratio is the Jaccard index; it is a *ranking* signal for the fallback,
    never a pass/reject line.
    """
    counts: dict[str, int] = {}
    for token in query:
        counts[token] = counts.get(token, 0) + 1
    shared = 0
    for token in ident:
        if counts.get(token, 0) > 0:
            counts[token] -= 1
            shared += 1
    union = len(ident) + len(query) - shared
    return shared, union, (shared / union) if union else 0.0


def sentences(raw: bytes) -> list[tuple[int, int]]:
    """Byte spans of the complete sentences of ``raw``.

    A sentence ends after a terminator in ``SENTENCE_TERMINATORS`` (a newline
    counts) or at the end of the record.  Terminators are whole characters of the
    stored UTF-8 text, so every span boundary is also a character boundary; the
    leading whitespace of a span is skipped, never the trailing one.  A record
    with no terminator is one sentence covering the whole record.  No numeric
    threshold is involved: membership in a fixed character set decides a
    boundary, and the set is categorical, not a tolerance.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    offset = 0
    for char in raw.decode("utf-8"):
        offset += len(char.encode("utf-8"))
        if char in SENTENCE_TERMINATORS:
            spans.append((start, offset))
            start = offset
    if start < len(raw):
        spans.append((start, len(raw)))
    cleaned: list[tuple[int, int]] = []
    for begin, end in spans:
        while begin < end and raw[begin : begin + 1] in b" \t\r\n":
            begin += 1
        if end > begin:
            cleaned.append((begin, end))
    return cleaned or [(0, len(raw))]


def chunk_address_audit() -> dict[str, Any]:
    """Numeric surface of the address rule, for the freeze/run tolerance audit.

    Returns every upper-case module constant that is an ``int``/``float`` and
    every numeric literal that appears inside the read path (``read``,
    ``block_overlap``, ``identity_blocks``, ``sentences``, ``_sentence_span``).  A
    new constant or a new literal is a *visible* change; the runner compares this
    dict with the frozen copy, so no tolerance can be added silently.
    """
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
    scope = (
        "read",
        "block_overlap",
        "identity_blocks",
        "sentences",
        "_sentence_span",
    )
    literals: dict[str, list[float]] = {}
    for walked in ast.walk(tree):
        if isinstance(walked, ast.FunctionDef) and walked.name in scope:
            values = sorted(
                {
                    float(item.value)
                    for item in ast.walk(walked)
                    if isinstance(item, ast.Constant)
                    and isinstance(item.value, (int, float))
                    and not isinstance(item.value, bool)
                }
            )
            literals[walked.name] = values
    return {
        "constants": constants,
        "read_path_literals": literals,
        "protection": {
            "PROTECT_equals_WINDOW": bool(PROTECT == WINDOW),
            "MIN_ADDRESSABLE_matches_definition": bool(MIN_ADDRESSABLE == MIN_CHUNK + PROTECT + 1),
            "MIN_ADDRESSABLE": MIN_ADDRESSABLE,
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
        if not ident:
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
                assert index is not None
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
        """Always return a top-1 record and a legal slice of its stored bytes.

        Anchored reads choose among the anchors; an unanchored read is labelled
        (``fallback`` / ``fallback_reason``) and ranked by block overlap.  The
        label is a reading: it never changes whether an output is produced.
        """
        query = features(raw)
        ranked: list[dict[str, Any]] = []
        for index, address in enumerate(self.addresses):
            ident = address["identity"]
            run, needle_start, haystack_start = longest_contiguous_run(ident, query)
            contained = run == len(ident) and needle_start == 0
            shared, union, overlap = block_overlap(ident, query)
            ranked.append(
                {
                    "index": index,
                    "covered": run,
                    "ratio": (run / len(ident)) if ident else 0.0,
                    "contained": contained,
                    "needle_start": needle_start,
                    "haystack_start": haystack_start,
                    "shared": shared,
                    "union": union,
                    "overlap": overlap,
                }
            )
        anchors = [row for row in ranked if row["contained"]]
        anchors.sort(key=lambda row: (-row["covered"], -row["ratio"], row["index"]))
        if len(anchors) == 1:
            decision = "matched"
        elif len(anchors) > 1:
            decision = "ambiguous_multiple_anchors"
        else:
            decision = "no_match"
        top: dict[str, Any] | None
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
                start, end, unit_index = self._sentence_span(address["raw"], query)
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
            fallback_basis = "record_overlap"
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
            "query_blocks": len(query),
            "top1": round(top1, 9),
            "top2": round(top2, 9),
            "margin": round(top1 - top2, 9),
            "unique_features_in_query": 0,
            "unknown_features": 0,
        }

    def _sentence_span(self, raw: bytes, query: tuple[str, ...]) -> tuple[int, int, int]:
        """Span of the whole sentence with the highest block overlap with the query.

        Ties go to the earlier sentence.  A record with a single sentence returns
        its whole span, so callers label it ``record``.  The span always comes
        from ``sentences()``: no character, word or sentence is ever cut.
        """
        spans = sentences(raw)
        if len(spans) == 1:
            return 0, len(raw), 0
        best: tuple[tuple[float, int, int], int, int, int] | None = None
        for index, (start, end) in enumerate(spans):
            shared, _union, ratio = block_overlap(features(raw[start:end]), query)
            key = (-ratio, -shared, index)
            if best is None or key < best[0]:
                best = (key, start, end, index)
        assert best is not None
        return best[1], best[2], best[3]

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
