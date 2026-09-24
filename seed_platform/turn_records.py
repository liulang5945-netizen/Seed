"""Seed-native turn records: the input ring of the Taiji recursive loop.

The legacy chat path feeds ``RecursiveImprover`` (strategy ring) and
``EvolutionEngine`` (task ring) from ``api.chat_strategies``.  A Seed-native
turn took the other branch and recorded nothing, so every harness turn stayed
invisible to the loop and the sleep phases that analyse it had no samples.

This module is the native counterpart.  It appends one jsonl line per record
under the shared external data root and keeps the legacy field vocabulary
(``strategy_type``/``strategy_content``/``task``/``success``/``quality_score``)
so a later consolidation pass, or the Life panel, can read native and legacy
samples through one shape.  Records are written with an append handle instead of
rewriting the whole file the way the legacy improver does, and every failure
here is non-fatal to the request that produced it.  A record carries the
harness Session that produced it, and a task outcome also carries the tools the
turn offered the model — the caller's request metadata, never model input.

It also collects the assembled system prompt as a **constraint seed**: the
native corpus holds dialogue only, so a constraint reaches the model through
training rather than through a prompt prefix.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from hashlib import sha256
from typing import Any

from seed_platform.paths import get_external_path

logger = logging.getLogger("SeedPlatform.TurnRecords")

_LOCK = threading.Lock()
_DIR_NAME = "turn_records"
_STRATEGY_FILE = "strategy_records.jsonl"
_TASK_FILE = "task_outcomes.jsonl"
_CONSTRAINT_FILE = "constraint_seeds.jsonl"
_MAX_FIELD_CHARS = 200
# The constraint seeds are training material, not a log: keep the whole prompt
# (one assembled system prompt), still bounded so one runaway caller cannot
# inflate the file.
_MAX_CONSTRAINT_CHARS = 2000
# Process-local dedup: the assembled system prompt is stable across turns, so
# without this every turn would append the same seed.
_SEEN_CONSTRAINTS: set[str] = set()
# Statistics stay bounded on a long-lived runtime; the summary is a read for
# operators and consolidation, not an audit of every turn ever taken.
_MAX_SUMMARY_LINES = 20000


def _records_dir() -> str:
    """Return (and create) the directory native turn records live in."""

    return get_external_path(os.path.join("data", _DIR_NAME))


def _clip(value: Any) -> str:
    """Clip a free-text field to the legacy record budget."""

    return str(value or "")[:_MAX_FIELD_CHARS]


def _attribution(session_id: str) -> dict[str, Any]:
    """The attribution key a record carries when the caller knows the Session."""

    sid = str(session_id or "")
    return {} if not sid else {"session_id": sid[:_MAX_FIELD_CHARS]}


def _offered(tools_offered: Any) -> dict[str, Any]:
    """The tool names a turn *offered*, clipped to the record budget.

    Offered is not used: a turn's actually-used tools are its ``tool_choice``
    strategy records.  The offered set is kept beside the outcome because
    "what was on offer and what was chosen" is the sample a tool-selection
    analysis needs, and it is invisible from the used set alone.
    """

    names = [_clip(name) for name in tools_offered or [] if str(name or "").strip()]
    unique = list(dict.fromkeys(names))
    return {} if not unique else {"tools_offered": unique}


def _append(filename: str, payload: dict[str, Any]) -> None:
    """Append one record, tagged with the moment and the branch that produced it."""

    record = dict(payload)
    record.setdefault("recorded_at", time.time())
    record.setdefault("source", "seed.native.chat")
    line = json.dumps(record, ensure_ascii=False)
    with _LOCK:
        os.makedirs(_records_dir(), exist_ok=True)
        with open(os.path.join(_records_dir(), filename), "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def record_strategy(
    strategy_type: str,
    strategy_content: str,
    task: str,
    success: bool,
    quality_score: float,
    session_id: str = "",
) -> None:
    """Record one strategy use: the strategy ring's input.

    Mirrors ``RecursiveImprover.record_strategy`` so sleep-time analysis sees the
    same three kinds (``prompt``/``tool_choice``/``reflection``) from a native
    turn as it does from a legacy one.  ``session_id`` attributes the sample to
    the harness Session that produced it.
    """

    try:
        _append(
            _STRATEGY_FILE,
            {
                "kind": "strategy",
                "strategy_type": str(strategy_type),
                "strategy_content": _clip(strategy_content),
                "task": _clip(task),
                "success": bool(success),
                "quality_score": float(quality_score),
                **_attribution(session_id),
            },
        )
    except Exception as e:  # pragma: no cover - defensive, never breaks a turn
        logger.debug("【record_strategy】处理失败（非致命）: %s", e)


def record_task_outcome(
    task: str,
    success: bool,
    final_answer: str = "",
    error: str = "",
    session_id: str = "",
    tools_offered: Any = None,
) -> None:
    """Record one task outcome: the task ring's input."""

    try:
        _append(
            _TASK_FILE,
            {
                "kind": "task",
                "task": _clip(task),
                "success": bool(success),
                "final_answer": _clip(final_answer),
                "error": _clip(error),
                **_attribution(session_id),
                **_offered(tools_offered),
            },
        )
    except Exception as e:  # pragma: no cover - defensive, never breaks a turn
        logger.debug("【record_task_outcome】处理失败（非致命）: %s", e)


def _read(filename: str, limit: int = _MAX_SUMMARY_LINES) -> list[dict[str, Any]]:
    """Read at most ``limit`` most recent records of one kind."""

    path = os.path.join(_records_dir(), filename)
    if not os.path.exists(path):
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-limit:]


def record_constraint(system_prompt: str) -> None:
    """Collect one assembled system prompt as a *training* seed, not a request field.

    The native corpus is dialogue only (``{"text": "问：…\\n答：…"}``), so a system
    prompt has no inference-time form the model was ever trained on: splicing it
    into the byte stream would invent an out-of-distribution prefix, while
    dropping it leaves the prompt with no effect at all.  The honest channel for
    a constraint is therefore the corpus, not the prompt — so the native branch
    keeps what the caller already sends and lets training internalise it later.
    Identical prompts are recorded once per process.
    """

    text = (system_prompt or "").strip()
    if not text:
        return
    digest = sha256(text.encode("utf-8")).hexdigest()
    with _LOCK:
        if digest in _SEEN_CONSTRAINTS:
            return
        _SEEN_CONSTRAINTS.add(digest)
    try:
        _append(
            _CONSTRAINT_FILE,
            {
                "kind": "constraint",
                "sha256": digest,
                "text": text[:_MAX_CONSTRAINT_CHARS],
            },
        )
    except Exception as e:  # pragma: no cover - defensive, never breaks a turn
        logger.debug("【record_constraint】处理失败（非致命）: %s", e)


def strategies(limit: int = _MAX_SUMMARY_LINES) -> list[dict[str, Any]]:
    """Return recorded strategy samples, oldest first, most recent ``limit`` kept."""

    try:
        return _read(_STRATEGY_FILE, limit)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("【strategies】读取失败（非致命）: %s", e)
        return []


def tasks(limit: int = _MAX_SUMMARY_LINES) -> list[dict[str, Any]]:
    """Return recorded task outcomes, oldest first, most recent ``limit`` kept."""

    try:
        return _read(_TASK_FILE, limit)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("【tasks】读取失败（非致命）: %s", e)
        return []


def constraints() -> list[dict[str, Any]]:
    """Return the distinct constraint seeds collected so far, newest first.

    Deduplication happens here rather than on write so the file stays a plain
    append-only log a later corpus builder can replay.
    """

    seen: dict[str, dict[str, Any]] = {}
    for record in _read(_CONSTRAINT_FILE):
        digest = str(record.get("sha256") or "")
        if digest and digest not in seen:
            seen[digest] = record
    return sorted(
        seen.values(), key=lambda record: float(record.get("recorded_at") or 0), reverse=True
    )


def summarize() -> dict[str, Any]:
    """Return what the native rings hold, for the Life surface and consolidation.

    Kept deliberately cheap: counts plus the most recent sample of each kind, so
    a caller can tell "the ring is filling" from "the ring has never seen a turn"
    without loading the whole history into a payload.
    """

    try:
        strategies = _read(_STRATEGY_FILE)
        tasks = _read(_TASK_FILE)
        seeds = constraints()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("【summarize】处理失败（非致命）: %s", e)
        return {"available": False, "reason": str(e)}

    by_type: dict[str, int] = {}
    for record in strategies:
        key = str(record.get("strategy_type") or "unknown")
        by_type[key] = by_type.get(key, 0) + 1
    succeeded = sum(1 for record in tasks if record.get("success"))
    return {
        "available": True,
        "directory": _records_dir(),
        "strategies": len(strategies),
        "strategies_by_type": by_type,
        "tasks": len(tasks),
        "tasks_succeeded": succeeded,
        "tasks_failed": len(tasks) - succeeded,
        "constraint_seeds": len(seeds),
        "last_recorded_at": max(
            [float(record.get("recorded_at") or 0) for record in strategies + tasks + seeds]
            or [0.0]
        ),
    }
