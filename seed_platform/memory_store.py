"""Taiji-native memory: the append-only journal of what actually happened.

Three other things in this repository carry the word *memory*, and this module is
none of them: ``seed_platform.memory`` is a RAM watchdog stub,
``neuroplex.agent_ext.memory_manager`` (legacy, never mounted) is a set of no-op
stubs, and ``seed_platform.turn_records`` is the recursive loop's *statistics*
ring.  This module is the runtime's own episodic memory — one append-only jsonl
journal plus a recall view, so the sleep phases (and later the harness) read back
what happened instead of recomputing it.

An entry is free-form by ``kind`` (the journal is a product-level record, not a
schema, so a new kind must not need a runtime change to exist).  What is fixed is
the envelope: text, session, turn, tags, importance, source, metadata, the
content digest, and the moment it was recorded.  Writes are idempotent per
content, and duplicates that a second process wrote are collapsed on read, so a
retrying reporter cannot inflate the journal.

Recall is keyword-based and deliberately so: the honest statement is "matching
text, importance, and recency", not "semantic memory".  Nothing here embeds text.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from hashlib import sha256
from typing import Any

from seed_platform.paths import get_external_path

logger = logging.getLogger("SeedPlatform.MemoryStore")

_LOCK = threading.Lock()
_DIR_NAME = "memory"
_ENTRY_FILE = "entries.jsonl"
# One entry is a bounded excerpt, not a transcript: the session log already keeps
# the full text, and a journal nobody can read whole is not a memory.
_MAX_TEXT_CHARS = 4000
_MAX_TAG_CHARS = 60
_MAX_TAGS = 16
_MAX_FIELD_CHARS = 200
_DEFAULT_IMPORTANCE = 0.3
# Recall ranks recency against this half-life; a week is the horizon over which
# "recent" still means something for a session-scale journal.
_RECENCY_HALF_LIFE_SECONDS = 7 * 24 * 3600
# Reads stay bounded on a long-lived runtime; the journal itself is not pruned.
_MAX_READ_ENTRIES = 20000
# Process-local idempotency: a reporter that retries one write must not append the
# same content twice.  Cross-process duplicates are collapsed by digest on read.
_SEEN_DIGESTS: set[str] = set()

_RECALL_WEIGHTS = {"overlap": 0.6, "importance": 0.25, "recency": 0.15}


def _entries_dir() -> str:
    """Return (and create) the directory the memory journal lives in."""

    return get_external_path(os.path.join("data", _DIR_NAME))


def _clip(value: Any, limit: int) -> str:
    """Clip one free-text field to its budget."""

    return str(value or "")[:limit]


def _normalize_tags(tags: Any) -> list[str]:
    """Clip, drop empties, and de-duplicate tags while keeping the caller's order."""

    names = [_clip(tag, _MAX_TAG_CHARS) for tag in tags or [] if str(tag or "").strip()]
    return list(dict.fromkeys(names))[:_MAX_TAGS]


def _clamp_importance(value: Any) -> float:
    """Clamp a reported importance into 0..1, defaulting to the journal's baseline."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_IMPORTANCE
    if math.isnan(number) or math.isinf(number):
        return _DEFAULT_IMPORTANCE
    return min(1.0, max(0.0, number))


def digest_of(kind: str, text: str, session_id: str = "", turn: Any = None) -> str:
    """Return the identity of one entry: its kind, its session, its turn, its text.

    Two reports of the same turn are the same entry; a re-run of one turn with a
    different answer is not, because the text is part of the identity.
    """

    basis = "\x1f".join((str(kind), str(session_id), str(turn), str(text)))
    return sha256(basis.encode("utf-8")).hexdigest()


def record(
    kind: str,
    text: str,
    *,
    session_id: str = "",
    turn: Any = None,
    tags: Any = None,
    importance: Any = None,
    source: str = "",
    metadata: Any = None,
) -> dict[str, Any]:
    """Append one entry, or report that this content is already in the journal.

    Returns the outcome rather than raising: a caller reporting a turn must not
    fail its turn because memory could not be written.
    """

    body = str(text or "").strip()
    if not body:
        return {"status": "skipped", "reason": "empty text"}
    clean_kind = _clip(kind, _MAX_FIELD_CHARS).strip()
    if not clean_kind:
        return {"status": "skipped", "reason": "empty kind"}
    session = _clip(session_id, _MAX_FIELD_CHARS)
    digest = digest_of(clean_kind, body, session, turn)
    entry = {
        "entry_id": uuid.uuid4().hex[:16],
        "kind": clean_kind,
        "text": body[:_MAX_TEXT_CHARS],
        "session_id": session,
        "turn": turn,
        "tags": _normalize_tags(tags),
        "importance": _clamp_importance(importance),
        "source": _clip(source, _MAX_FIELD_CHARS),
        "metadata": metadata if isinstance(metadata, dict) else {},
        "digest": digest,
        "recorded_at": time.time(),
    }
    with _LOCK:
        if digest in _SEEN_DIGESTS:
            return {"status": "duplicate", "digest": digest}
        _SEEN_DIGESTS.add(digest)
        try:
            os.makedirs(_entries_dir(), exist_ok=True)
            with open(os.path.join(_entries_dir(), _ENTRY_FILE), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:  # pragma: no cover - defensive, never breaks a caller
            _SEEN_DIGESTS.discard(digest)
            logger.debug("【memory_store.record】写入失败（非致命）: %s", e)
            return {"status": "failed", "reason": str(e)}
    return {"status": "recorded", "entry_id": entry["entry_id"], "digest": digest}


def _read() -> list[dict[str, Any]]:
    """Read the journal, collapsing cross-process duplicates, newest last."""

    path = os.path.join(_entries_dir(), _ENTRY_FILE)
    if not os.path.exists(path):
        return []
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            digest = str(entry.get("digest") or "")
            if digest and digest in seen:
                continue
            if digest:
                seen.add(digest)
            entries.append(entry)
    return entries[-_MAX_READ_ENTRIES:]


def entries(
    *,
    kind: str = "",
    session_id: str = "",
    tag: str = "",
    limit: int = _MAX_READ_ENTRIES,
) -> list[dict[str, Any]]:
    """Return journal entries matching the given exact filters, newest last."""

    selected = [
        entry
        for entry in _read()
        if (not kind or str(entry.get("kind")) == kind)
        and (not session_id or str(entry.get("session_id")) == session_id)
        and (not tag or tag in (entry.get("tags") or []))
    ]
    if limit <= 0:
        return selected
    return selected[-limit:]


def _tokens(query: str) -> list[str]:
    """Split a query into match terms: whitespace words plus their lowercase form."""

    terms = [term for term in str(query or "").split() if term]
    return [term.lower() for term in terms]


def _overlap(entry: dict[str, Any], terms: list[str]) -> float:
    """Fraction of query terms the entry's text or tags contain."""

    if not terms:
        return 0.0
    haystack = " ".join(
        [
            str(entry.get("text") or ""),
            str(entry.get("kind") or ""),
            *[str(tag) for tag in entry.get("tags") or []],
        ]
    ).lower()
    matched = sum(1 for term in terms if term in haystack)
    return matched / len(terms)


def _recency(entry: dict[str, Any], now: float) -> float:
    """Exponential decay of the entry's age against the recall half-life."""

    age = max(0.0, now - float(entry.get("recorded_at") or 0.0))
    return 0.5 ** (age / _RECENCY_HALF_LIFE_SECONDS)


def recall(
    query: str = "",
    *,
    kind: str = "",
    session_id: str = "",
    tag: str = "",
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Return the entries that best match a query, highest score first.

    An empty query ranks by importance and recency alone; a non-empty query drops
    entries that match no term, because returning "the most important unrelated
    entry" would be a worse answer than returning nothing.
    """

    terms = _tokens(query)
    now = time.time()
    ranked: list[dict[str, Any]] = []
    for entry in entries(kind=kind, session_id=session_id, tag=tag):
        overlap = _overlap(entry, terms)
        if terms and overlap <= 0.0:
            continue
        score = (
            _RECALL_WEIGHTS["overlap"] * overlap
            + _RECALL_WEIGHTS["importance"] * float(entry.get("importance") or 0.0)
            + _RECALL_WEIGHTS["recency"] * _recency(entry, now)
        )
        ranked.append({**entry, "score": round(score, 6)})
    ranked.sort(
        key=lambda item: (item["score"], float(item.get("recorded_at") or 0.0)), reverse=True
    )
    return ranked[: max(0, limit)]


def status() -> dict[str, Any]:
    """Return what the journal holds, for operators and the Life surface."""

    try:
        journal = _read()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("【memory_store.status】处理失败（非致命）: %s", e)
        return {"available": False, "reason": str(e)}

    by_kind: dict[str, int] = {}
    tags: set[str] = set()
    sessions: set[str] = set()
    for entry in journal:
        key = str(entry.get("kind") or "unknown")
        by_kind[key] = by_kind.get(key, 0) + 1
        tags.update(str(tag) for tag in entry.get("tags") or [])
        if entry.get("session_id"):
            sessions.add(str(entry["session_id"]))
    return {
        "available": True,
        "directory": _entries_dir(),
        "entries": len(journal),
        "by_kind": by_kind,
        "tags": len(tags),
        "sessions": len(sessions),
        "last_recorded_at": max(
            [float(entry.get("recorded_at") or 0.0) for entry in journal] or [0.0]
        ),
    }
