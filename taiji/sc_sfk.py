"""Shared-carrier sparse feature key (SC-SFK) address table.

The address layer of the store, as a structural primitive: pure Python statistics,
no tensors, no parameters, no training.

Address definition
------------------
* a record's address is the *set* of its 2-byte word features (position free);
* two records share an address iff their feature sets are equal -- there is no
  numeric similarity threshold on the write path and no exit depends on one;
* a word is unique when its document frequency is 1, and a query's anchors are
  the addresses that own one of its unique words;
* read exits: zero anchors -> ``no_match`` (the explicit "do not know"), more
  than one -> ``ambiguous_multiple_anchors`` (the explicit "cannot tell"), exactly
  one -> ``matched``.  All three are set conditions.

Write policy: make room before content (the table starts at 2 slots and doubles on
demand), merge an equal value, replace a same-address different value while
keeping the previous value in history, reject a batch whole when two of its items
claim the same feature set with different values.  No eviction and no silent
overwrite.  Every accepted write recomputes DF over the whole table and then
checks three structural properties: no record is mis-addressed (G-A), no write
removes the anchor of an address it did not touch (G-B), and no two addresses
share a feature set (G-C).  A failed check rolls the whole batch back.

The weighted Jaccard similarity and its IDF table are kept as *readings* only:
they are reported as ``top1``/``top2``/``margin`` and never decide an exit.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

START_CAPACITY = 2
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


def _value_key(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _payload_bytes(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return 8


def unique_features(addresses: list[dict[str, Any]]) -> set[str]:
    df = document_frequency(addresses)
    return {feature for feature, count in df.items() if count == 1}


def addressable(addresses: list[dict[str, Any]]) -> set[int]:
    """Indices whose feature set contains at least one table-unique word."""
    unique = unique_features(addresses)
    return {
        index
        for index, address in enumerate(addresses)
        if any(feature in unique for feature in address["features"])
    }


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
        # Structural zeros, not counters: the table has no eviction and no silent
        # overwrite path at all.  ``write_batch_checked`` asserts they stay 0.
        self.evictions = 0
        self.silent_overwrites = 0
        self.rollbacks = 0
        self.global_checks = 0
        self.global_failures = 0

    # -- statistics ---------------------------------------------------------
    @property
    def generation(self) -> int:
        """Capacity generation; the same number the session layer reports."""
        return self.expansions

    def idf(self) -> dict[str, float]:
        return idf_table(self.addresses)

    def unique_features(self) -> set[str]:
        return unique_features(self.addresses)

    def nbytes(self) -> int:
        """Payload bytes of the table (features, current values, history values).

        Python object overhead is excluded; every payload is a bytes/str object, so
        this is the number that should be added to ``state_bytes`` when the table
        is live.
        """
        total = 0
        for address in self.addresses:
            total += sum(len(feature) for feature in address["features"])
            total += sum(_payload_bytes(value) for value in address["values"])
            for entry in address["history"]:
                total += _payload_bytes(entry.get("value"))
        return total

    # -- batch API ----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "addresses": [
                {
                    "features": address["features"],
                    "values": list(address["values"]),
                    "history": [dict(entry) for entry in address["history"]],
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

    def find(self, feats: tuple[str, ...]) -> int | None:
        """Exact-set address lookup; the only write-side identity rule."""
        for index, address in enumerate(self.addresses):
            if address["features"] == feats:
                return index
        return None

    def value_at(self, index: int) -> Any:
        values = self.addresses[index]["values"]
        return values[-1] if values else None

    def target_of(self, feats: tuple[str, ...]) -> tuple[int | None, float]:
        """Exact match, else the best diagnostic similarity (never an exit)."""
        index = self.find(feats)
        if index is not None:
            return index, 1.0
        idf = self.idf()
        best_index, best_score = None, -1.0
        for candidate, address in enumerate(self.addresses):
            score = similarity(feats, address["features"], idf)
            if score > best_score:
                best_index, best_score = candidate, score
        return best_index, best_score

    def write_batch(self, items: list[tuple[tuple[str, ...], Any]]) -> dict[str, Any]:
        snapshot = self.snapshot()
        before = len(self.addresses)
        addressable_before = addressable(self.addresses)
        claims: dict[tuple[str, ...], str] = {}
        touched: set[int] = set()
        actions: list[str] = []
        for feats, value in items:
            key = _value_key(value)
            if feats in claims and claims[feats] != key:
                self.restore(snapshot)
                self.rejected_batches += 1
                return {
                    "action": "rejected_batch_address_collision",
                    "items": len(items),
                    "conflicts": [{"features": list(feats), "values": [claims[feats], key]}],
                    "slots_before": before,
                    "slots_after": before,
                }
            claims[feats] = key
            target = self.find(feats)
            if target is not None:
                touched.add(target)
            actions.append(self._write_one(feats, value, target))
        return {
            "action": "accepted",
            "items": len(items),
            "slots_before": before,
            "slots_after": len(self.addresses),
            "actions": actions,
            "addressable_before": addressable_before,
            "touched": touched,
        }

    def _write_one(self, feats: tuple[str, ...], value: Any, target: int | None) -> str:
        if target is not None:
            address = self.addresses[target]
            # ``count`` is the number of observations of this record, not the
            # number of byte-identical merges: a re-observation whose fast snapshot
            # moved is still an observation and still raises the carrier frequency.
            address["count"] = int(address["count"]) + 1
            if value == address["values"][-1]:
                self.merged += 1
                return "merged_duplicate"
            # Only the displaced current value is kept: old values are exactly the
            # history entries, in order, so memory stays O(observations) and no old
            # value is ever dropped.
            address["history"].append({"value": address["values"][-1], "reason": "replace"})
            address["values"][-1] = value
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
    def check_global(
        self,
        *,
        before_addressable: set[int] | None = None,
        touched: set[int] | None = None,
    ) -> dict[str, Any]:
        """Structural whole-table check G-A / G-B / G-C.

        G-A  no record is mis-addressed: reading a record's own features returns
             either that record or an explicit abstention, never another address;
        G-B  no collateral loss: an address that was addressable before the batch
             and was not touched by it is still addressable afterwards;
        G-C  identity uniqueness: no two addresses share a feature set.
        """
        self.global_checks += 1
        sets = [address["features"] for address in self.addresses]
        distinct = len(set(sets)) == len(sets)
        misaddressed = []
        for index, feats in enumerate(sets):
            result = self.read(feats)
            if result["address"] not in (None, index):
                misaddressed.append({"record": index, "read_as": result["address"]})
        after = addressable(self.addresses)
        collateral: list[int] = []
        if before_addressable is not None:
            touched = set(touched or ())
            collateral = sorted(before_addressable - touched - after)
        ok = bool(distinct) and not misaddressed and not collateral
        if not ok:
            self.global_failures += 1
        return {
            "ok": ok,
            "distinct": bool(distinct),
            "missing_g1": misaddressed,
            "collateral_loss": collateral,
            "addressable_before": (
                len(before_addressable) if before_addressable is not None else None
            ),
            "addressable_after": len(after),
        }

    def write_batch_checked(self, items, *, rollback_on_failure: bool = True) -> dict[str, Any]:
        assert self.evictions == 0 and self.silent_overwrites == 0
        snapshot = self.snapshot()
        result = self.write_batch(items)
        if result["action"] != "accepted":
            return result
        check = self.check_global(
            before_addressable=result["addressable_before"],
            touched=result["touched"],
        )
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
                "value": None,
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
        elif len(anchors) == 1:
            decision = "matched"
        else:
            decision = "ambiguous_multiple_anchors"
        chosen = anchors[0] if decision == "matched" else None
        return {
            "decision": decision,
            "address": chosen,
            "value": self.value_at(chosen) if chosen is not None else None,
            "anchors": anchors,
            "unique_features_in_query": len(query_unique),
            "unknown_features": 0,
            "top1": round(top1, 9),
            "top2": round(top2, 9),
            "margin": round(top1 - top2, 9),
            "anchor_margin": round(anchor_margin, 9),
        }
