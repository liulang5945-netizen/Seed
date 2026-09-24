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
here is non-fatal to the request that produced it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from seed_platform.paths import get_external_path

logger = logging.getLogger("SeedPlatform.TurnRecords")

_LOCK = threading.Lock()
_DIR_NAME = "turn_records"
_STRATEGY_FILE = "strategy_records.jsonl"
_TASK_FILE = "task_outcomes.jsonl"
_MAX_FIELD_CHARS = 200
# Statistics stay bounded on a long-lived runtime; the summary is a read for
# operators and consolidation, not an audit of every turn ever taken.
_MAX_SUMMARY_LINES = 20000


def _records_dir() -> str:
    """Return (and create) the directory native turn records live in."""

    return get_external_path(os.path.join("data", _DIR_NAME))


def _clip(value: Any) -> str:
    """Clip a free-text field to the legacy record budget."""

    return str(value or "")[: _MAX_FIELD_CHARS]


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
) -> None:
    """Record one strategy use: the strategy ring's input.

    Mirrors ``RecursiveImprover.record_strategy`` so sleep-time analysis sees the
    same three kinds (``prompt``/``tool_choice``/``reflection``) from a native
    turn as it does from a legacy one.
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
            },
        )
    except Exception as e:  # pragma: no cover - defensive, never breaks a turn
        logger.debug("【record_strategy】处理失败（非致命）: %s", e)


def record_task_outcome(
    task: str,
    success: bool,
    final_answer: str = "",
    error: str = "",
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


def summarize() -> dict[str, Any]:
    """Return what the native rings hold, for the Life surface and consolidation.

    Kept deliberately cheap: counts plus the most recent sample of each kind, so
    a caller can tell "the ring is filling" from "the ring has never seen a turn"
    without loading the whole history into a payload.
    """

    try:
        strategies = _read(_STRATEGY_FILE)
        tasks = _read(_TASK_FILE)
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
        "last_recorded_at": max(
            [float(record.get("recorded_at") or 0) for record in strategies + tasks] or [0.0]
        ),
    }