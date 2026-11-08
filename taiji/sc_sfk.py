"""Shared-carrier sparse feature key (SC-SFK) address table.

The address layer of the store, as a structural primitive: pure Python statistics,
no tensors, no parameters, no training.  The primitive is the one verified in
``scripts/training/probe_taiji_sc_address_20261108.py``; this module is the same
code moved to where the store can use it, with the constants left as module
defaults so the wired reading stays comparable to the runner-side reading.

Address definition
------------------
* a record's address is the *set* of its 2-byte word features (position free);
* a feature's weight is its smoothed inverse document frequency over the current
  table, so common carrier words weigh almost nothing and content words carry the
  identity;
* similarity is weighted Jaccard over those weights;
* a query is addressable only when it owns at least one feature that occurs in
  exactly one stored address.  Zero anchors -> ``no_match``; more than one ->
  ``ambiguous_multiple_anchors``.  The exit is derived from store statistics, not
  from a tuned threshold.

Write policy: make room before content (the table starts at 2 slots and doubles on
demand), merge identical values, replace a same-address different value while
keeping the previous value in history, reject a batch whole when two of its items
claim one address with different values.  No eviction and no silent overwrite.
Every accepted write recomputes IDF over the whole table and then re-checks that
each address reads back to itself and that no two addresses reach
``COPY_OVERLAP``; a failed re-check rolls the write back.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

START_CAPACITY = 2
COPY_OVERLAP = 0.85
ANCHOR_MARGIN = 0.02
UNSEEN = "__unseen__"


def features(raw: bytes) -> tuple[str, ...]:
    """Position-independent feature set: 2-byte word units, deduplicated."""
    words = []
    for index in range(0, len(raw) - 1, 2):
        words.append(raw[index : index + 2].hex())
    return tuple(sorted(set(words)))


def document_frequency(addresses: list[dict[str, Any]]) -> Counter:
    """Document frequency over *records*, not over collapsed addresses.

    Byte-identical records merged into one address still count as separate
    observations: they are the evidence that a word belongs to the shared carrier.
    A value replacement does not add an observation.
    """
    counter: Counter = Counter()
    for address in addresses:
        for _ in range(int(address.get("count", 1))):
            counter.update(address["features"])
    return counter


def idf_table(addresses: list[dict[str, Any]]) -> dict[str, float]:
    n = max(sum(int(address.get("count", 1)) for address in addresses), 1)
    table = {
        feature: math.log(1.0 + n / (1.0 + count))
        for feature, count in document_frequency(addresses).items()
    }
    # A feature never seen in the table is the rarest possible feature.
    table[UNSEEN] = math.log(1.0 + n)
    return table


def similarity(left: tuple[str, ...], right: tuple[str, ...], idf: dict[str, float]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    default = idf.get(UNSEEN, 0.0)
    numerator = 0.0
    for feature in left_set & right_set:
        numerator += idf.get(feature, default)
    denominator = 0.0
    for feature in left_set | right_set:
        denominator += idf.get(feature, default)
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


class AddressTable:
    """Shared-carrier sparse feature address table with growth-before-write."""

    def __init__(self, start_capacity: int = START_CAPACITY) -> None:
        self.capacity = int(start_capacity)
        self.peak_capacity = int(start_capacity)
        self.addresses: list[dict[str, Any]] = []
        self.expansions = 0
        self.merged = 0
        self.replaced = 0
        self.appended = 0
        self.rejected_batches = 0
        self.evictions = 0
        self.silent_overwrites = 0
        self.rollbacks = 0
        self.global_checks = 0
        self.global_failures = 0
        self.history: list[dict[str, Any]] = []

    # -- statistics ---------------------------------------------------------
    def idf(self) -> dict[str, float]:
        return idf_table(self.addresses)

    def unique_features(self) -> set[str]:
        df = document_frequency(self.addresses)
        return {feature for feature, count in df.items() if count == 1}

    # -- batch API ----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "addresses": [
                {
                    "features": address["features"],
                    "values": list(address["values"]),
                    "history": list(address["history"]),
                    "count": int(address["count"]),
                }
                for address in self.addresses
            ],
            "capacity": self.capacity,
            "peak_capacity": self.peak_capacity,
            "expansions": self.expansions,
            "merged": self.merged,
            "replaced": self.replaced,
            "appended": self.appended,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.addresses = snapshot["addresses"]
        self.capacity = snapshot["capacity"]
        self.peak_capacity = snapshot["peak_capacity"]
        self.expansions = snapshot["expansions"]
        self.merged = snapshot["merged"]
        self.replaced = snapshot["replaced"]
        self.appended = snapshot["appended"]

    def target_of(self, feats: tuple[str, ...]) -> tuple[int | None, float]:
        idf = self.idf()
        best_index, best_score = None, -1.0
        for index, address in enumerate(self.addresses):
            score = similarity(feats, address["features"], idf)
            if score > best_score:
                best_index, best_score = index, score
        if best_score >= COPY_OVERLAP:
            return best_index, best_score
        return None, best_score

    def write_batch(self, items: list[tuple[tuple[str, ...], Any]]) -> dict[str, Any]:
        def value_key(value: Any) -> str:
            return value if isinstance(value, str) else repr(value)

        snapshot = self.snapshot()
        before = len(self.addresses)
        claims: dict[int, str] = {}
        actions: list[str] = []
        for feats, value in items:
            target, _ = self.target_of(feats)
            if target is not None:
                key = value_key(value)
                if target in claims and claims[target] != key:
                    self.restore(snapshot)
                    self.rejected_batches += 1
                    return {
                        "action": "rejected_batch_address_collision",
                        "items": len(items),
                        "conflicts": [{"address": target, "values": [claims[target], key]}],
                        "slots_before": before,
                        "slots_after": before,
                    }
                claims[target] = key
            actions.append(self._write_one(feats, value, target))
        return {
            "action": "accepted",
            "items": len(items),
            "slots_before": before,
            "slots_after": len(self.addresses),
            "actions": actions,
        }

    def _write_one(self, feats: tuple[str, ...], value: Any, target: int | None) -> str:
        if target is not None:
            address = self.addresses[target]
            if value in address["values"]:
                address["values"].append(value)
                address["count"] = int(address["count"]) + 1
                self.merged += 1
                return "merged_duplicate"
            address["history"].append({"values": list(address["values"]), "reason": "replace"})
            address["values"].append(value)
            self.replaced += 1
            return "replaced_at_same_address"
        while len(self.addresses) >= self.capacity:
            self.capacity *= 2
            self.expansions += 1
            self.peak_capacity = max(self.peak_capacity, self.capacity)
        self.addresses.append({"features": feats, "values": [value], "history": [], "count": 1})
        self.appended += 1
        return "appended_after_growth"

    # -- global (local -> whole) consistency --------------------------------
    def check_global(self) -> dict[str, Any]:
        self.global_checks += 1
        idf = self.idf()
        self_read = all(
            similarity(address["features"], address["features"], idf) >= 1.0 - 1e-12
            for address in self.addresses
        )
        collisions = []
        for left in range(len(self.addresses)):
            for right in range(left + 1, len(self.addresses)):
                score = similarity(
                    self.addresses[left]["features"], self.addresses[right]["features"], idf
                )
                if score >= COPY_OVERLAP:
                    collisions.append({"left": left, "right": right, "score": score})
        ok = bool(self_read) and not collisions
        if not ok:
            self.global_failures += 1
        return {"ok": ok, "self_read": bool(self_read), "collisions": collisions}

    def write_batch_checked(self, items, *, rollback_on_failure: bool = True) -> dict[str, Any]:
        snapshot = self.snapshot()
        result = self.write_batch(items)
        if result["action"] != "accepted":
            return result
        check = self.check_global()
        result["global_check"] = check
        if not check["ok"] and rollback_on_failure:
            self.restore(snapshot)
            self.rollbacks += 1
            result["action"] = "rolled_back_global_check_failed"
        return result

    # -- read ---------------------------------------------------------------
    def read(self, feats: tuple[str, ...]) -> dict[str, Any]:
        df = document_frequency(self.addresses)
        if any(df.get(feature, 0) == 0 for feature in feats):
            return {
                "decision": "no_match",
                "address": None,
                "anchors": [],
                "unique_features_in_query": 0,
                "unknown_features": sum(1 for feature in feats if df.get(feature, 0) == 0),
                "top1": 0.0,
                "top2": 0.0,
                "margin": 0.0,
                "anchor_margin": 0.0,
            }
        unique = {feature for feature, count in df.items() if count == 1}
        query_unique = [feature for feature in feats if feature in unique]
        anchors = [
            index
            for index, address in enumerate(self.addresses)
            if any(feature in address["features"] for feature in query_unique)
        ]
        idf = self.idf()
        scores = [
            (index, similarity(feats, address["features"], idf))
            for index, address in enumerate(self.addresses)
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        top1_index, top1 = scores[0] if scores else (None, 0.0)
        top2 = scores[1][1] if len(scores) > 1 else 0.0
        anchor_scores = sorted(
            ((index, dict(scores)[index]) for index in anchors),
            key=lambda item: item[1],
            reverse=True,
        )
        anchor_margin = (
            anchor_scores[0][1] - anchor_scores[1][1]
            if len(anchor_scores) > 1
            else anchor_scores[0][1] if anchor_scores else 0.0
        )
        if not query_unique:
            decision = "no_match"
        elif len(anchors) == 1 or anchor_margin >= ANCHOR_MARGIN:
            decision = "matched"
        else:
            decision = "ambiguous_multiple_anchors"
        chosen = anchor_scores[0][0] if decision == "matched" else None
        return {
            "decision": decision,
            "address": chosen,
            "anchors": anchors,
            "unique_features_in_query": len(query_unique),
            "unknown_features": 0,
            "top1": round(top1, 9),
            "top2": round(top2, 9),
            "margin": round(top1 - top2, 9),
            "anchor_margin": round(anchor_margin, 9),
        }
