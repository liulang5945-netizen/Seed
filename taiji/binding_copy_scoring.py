"""15.1 rev2 raw-byte scoring and entity-stage views.

The scorer takes canonical action bytes, never displayed text.  It is the
single normalization used by the new evaluation: strict UTF-8, one canonical
27-byte action plus EOS, 64 output bytes / 65 predictions, with early EOS,
parse failure and truncation left in the denominator.
"""

from __future__ import annotations

from typing import Any

from taiji.binding_copy_data import ACTION_BYTES, DataError

SCORING_SCHEMA = "taiji-binding-copy-scoring-v1"
MAX_OUTPUT_BYTES = 64
MAX_PREDICTIONS = 65
INVALID_PREFIX = "<invalid-utf8:"
INVALID_SUFFIX = ">"


def normalize_expected(value: bytes | tuple[int, ...] | list[int]) -> bytes:
    """One normalization for bytes and integer tuple/list; no text inference."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, tuple) and all(isinstance(item, int) for item in value):
        return bytes(value)
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return bytes(value)
    raise DataError(f"expected must be bytes or an integer tuple/list, not {type(value)!r}")


def raw_from_text(text: str, utf8: bool) -> bytes:
    if utf8:
        return text.encode("utf-8")
    if text.startswith(INVALID_PREFIX) and text.endswith(INVALID_SUFFIX):
        return bytes.fromhex(text[len(INVALID_PREFIX) : -len(INVALID_SUFFIX)])
    raise DataError("invalid generated text has no stored hex form")


def _first_divergence(raw: bytes, expected: bytes) -> int:
    for index in range(min(len(raw), len(expected))):
        if raw[index] != expected[index]:
            return index
    if len(raw) != len(expected):
        return min(len(raw), len(expected))
    return -1


def score_generation(
    raw: bytes,
    expected: bytes | tuple[int, ...] | list[int],
    terminated: bool,
) -> dict[str, Any]:
    expected_bytes = normalize_expected(expected)
    raw_bytes = bytes(raw)
    try:
        raw_bytes.decode("utf-8")
        utf8 = True
    except UnicodeDecodeError:
        utf8 = False
    full = raw_bytes == expected_bytes
    object_exact = len(raw_bytes) == ACTION_BYTES and raw_bytes[3:15] == expected_bytes[3:15]
    place_exact = len(raw_bytes) == ACTION_BYTES and raw_bytes[21:27] == expected_bytes[21:27]
    entity_exact = bool(object_exact and place_exact)
    completed = bool(full and terminated)
    empty = len(raw_bytes) == 0
    early_eos = bool(terminated and 0 < len(raw_bytes) < ACTION_BYTES)
    truncated = bool(len(raw_bytes) >= MAX_OUTPUT_BYTES and not terminated)
    parsed = bool(terminated and utf8)
    first_divergence = _first_divergence(raw_bytes, expected_bytes)
    if completed:
        reason = "completed"
    elif empty:
        reason = "empty"
    elif early_eos:
        reason = "early_eos"
    elif truncated:
        reason = "truncated"
    elif not utf8:
        reason = "invalid_utf8"
    elif not terminated:
        reason = "not_terminated"
    else:
        reason = "wrong_bytes"
    return {
        "schema": SCORING_SCHEMA,
        "bytes": len(raw_bytes),
        "predictions": min(len(raw_bytes) + 1, MAX_PREDICTIONS),
        "utf8": bool(utf8),
        "terminated": bool(terminated),
        "full_sentence": bool(full),
        "object_exact": bool(object_exact),
        "place_exact": bool(place_exact),
        "entity_exact": entity_exact,
        "completed": completed,
        "empty": bool(empty),
        "early_eos": bool(early_eos),
        "truncated": bool(truncated),
        "parsed": bool(parsed),
        "first_divergence": first_divergence,
        "reason": reason,
        "expected_bytes": len(expected_bytes),
        "denominator": True,
    }


def entity_stage(
    raw: bytes,
    expected: bytes | tuple[int, ...] | list[int],
) -> dict[str, Any]:
    expected_bytes = normalize_expected(expected)
    raw_bytes = bytes(raw)
    object_exact = len(raw_bytes) == ACTION_BYTES and raw_bytes[3:15] == expected_bytes[3:15]
    place_exact = len(raw_bytes) == ACTION_BYTES and raw_bytes[21:27] == expected_bytes[21:27]
    return {
        "object_bytes": [3, 15] if len(raw_bytes) >= 15 else [3, len(raw_bytes)],
        "place_bytes": [21, 27] if len(raw_bytes) >= 27 else [21, len(raw_bytes)],
        "object_exact": bool(object_exact),
        "place_exact": bool(place_exact),
        "entity_exact": bool(object_exact and place_exact),
    }


def score_14_6_rows(rows: list[dict]) -> dict[str, Any]:
    """Deterministic self-check against the already re-scored 14.6 rows."""
    scored: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    utf8_bytes_match = 0
    for index, row in enumerate(rows):
        raw = raw_from_text(row["generated_text"], row["utf8"])
        expected = row["expected"].encode("utf-8")
        result = score_generation(raw, expected, row["terminated"])
        stored = {
            "completed": bool(row["completed"]),
            "full_sentence": bool(row["full_sentence"]),
            "entity_exact": bool(row["entity_exact"]),
            "utf8": bool(row["utf8"]),
            "bytes": int(row["bytes"]),
        }
        current = {
            "completed": result["completed"],
            "full_sentence": result["full_sentence"],
            "entity_exact": result["entity_exact"],
            "utf8": result["utf8"],
            "bytes": result["bytes"],
        }
        if current == stored:
            utf8_bytes_match += 1
        else:
            mismatches.append(
                {
                    "index": index,
                    "stored": stored,
                    "current": current,
                    "reason": result["reason"],
                }
            )
        scored.append(
            {
                "index": index,
                "stored": stored,
                "current": current,
                "first_divergence": result["first_divergence"],
                "entity_stage": entity_stage(raw, expected),
                "reason": result["reason"],
            }
        )
    counts = {
        "completed": sum(1 for item in scored if item["current"]["completed"]),
        "full_sentence": sum(1 for item in scored if item["current"]["full_sentence"]),
        "entity_exact": sum(1 for item in scored if item["current"]["entity_exact"]),
        "utf8": sum(1 for item in scored if item["current"]["utf8"]),
        "n": len(scored),
    }
    stored_counts = {
        "completed": sum(1 for item in scored if item["stored"]["completed"]),
        "full_sentence": sum(1 for item in scored if item["stored"]["full_sentence"]),
        "entity_exact": sum(1 for item in scored if item["stored"]["entity_exact"]),
        "utf8": sum(1 for item in scored if item["stored"]["utf8"]),
        "n": len(scored),
    }
    return {
        "n": len(scored),
        "matched": utf8_bytes_match,
        "mismatches": mismatches,
        "counts": counts,
        "stored_counts": stored_counts,
        "scored": scored,
    }


def self_test(
    canonical: bytes,
    other: bytes,
    candidates: tuple[str, ...],
) -> dict[str, Any]:
    """Oracle/constant/list/wrong-name/truncation/byte-type/empty/early controls."""
    canonical_bytes = normalize_expected(canonical)
    other_bytes = normalize_expected(other)
    oracle = score_generation(canonical_bytes, canonical_bytes, True)
    constant = score_generation(other_bytes, canonical_bytes, True)
    listed = score_generation("".join(candidates).encode("utf-8"), canonical_bytes, True)
    wrong_name = score_generation(other_bytes, canonical_bytes, True)
    truncation = score_generation(
        (canonical_bytes + b"\xe4\xb8" * 19)[:MAX_OUTPUT_BYTES], canonical_bytes, False
    )
    byte_type = score_generation(canonical_bytes, tuple(canonical_bytes), True)
    empty = score_generation(b"", canonical_bytes, True)
    early_eos = score_generation(canonical_bytes[:10], canonical_bytes, True)
    invalid_utf8 = score_generation(b"\xe7" * 22, canonical_bytes, True)
    not_terminated = score_generation(canonical_bytes, canonical_bytes, False)
    cases = {
        "oracle": oracle,
        "constant": constant,
        "list": listed,
        "wrong_name": wrong_name,
        "truncation": truncation,
        "byte_type": byte_type,
        "empty": empty,
        "early_eos": early_eos,
        "invalid_utf8": invalid_utf8,
        "not_terminated": not_terminated,
    }
    passed = (
        oracle["completed"]
        and not constant["completed"]
        and not listed["completed"]
        and not wrong_name["completed"]
        and not truncation["completed"]
        and truncation["truncated"]
        and byte_type["completed"]
        and not empty["completed"]
        and empty["empty"]
        and not early_eos["completed"]
        and early_eos["early_eos"]
        and not invalid_utf8["completed"]
        and not invalid_utf8["utf8"]
        and not not_terminated["completed"]
        and not not_terminated["parsed"]
    )
    return {"passed": passed, "cases": cases}
