"""Block until a P3b campaign arm reaches a stage target (or stops), then report.

The campaigns run for ~50 hours, so polling them turn by turn wastes both turns and CPU.
This waits on the condition instead: it exits as soon as either arm has ``--stages`` scored
checkpoints, or either arm has written a stop verdict, or the deadline hits.  Output is
ASCII so it can be redirected to a log safely.

Usage::

    python -X utf8 scripts/training/wait_p3b_stage.py --stages 6 --deadline-seconds 25200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARMS = ("treatment", "control")
#: Only the current key is read.  The two campaigns started 2026-09-15 16:25 write their stop
#: verdict under the field's previous name (loaded before the rename); those two artifacts are
#: read with a one-off command, and the reserved token stays out of this repository's files.
STOP_KEY = "campaign_stop"


def read_arm(arm: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "reports" / f"taiji_p3b_campaign_{arm}_20260915.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # rewritten atomically; a torn read just retries
        return {}


def stop_of(report: dict[str, Any]) -> Any:
    return report.get(STOP_KEY)


def reached(stages: dict[str, Any], target: int) -> bool:
    return any(len(report.get("stages") or []) >= target for report in stages.values())


def wait(target: int, deadline_seconds: int, poll_seconds: int) -> dict[str, Any]:
    deadline = time.time() + deadline_seconds
    state: dict[str, Any] = {arm: {"stages": 0, "status": "missing"} for arm in ARMS}
    while time.time() < deadline:
        stages = {arm: read_arm(arm) for arm in ARMS}
        stops = {arm: stop_of(stages[arm]) for arm in ARMS}
        state = {
            arm: {
                "stages": len((stages[arm] or {}).get("stages") or []),
                "status": (stages[arm] or {}).get("status"),
                "stop": stops[arm],
                "latest_tick": next(
                    (
                        row.get("tick")
                        for row in reversed((stages[arm] or {}).get("stages") or [])
                        if row.get("tick")
                    ),
                    None,
                ),
            }
            for arm in ARMS
        }
        if any(stops.values()) or reached(stages, target):
            return {"reason": "condition", "state": state}
        time.sleep(poll_seconds)
    return {"reason": "deadline", "state": state}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="等待 P3b 阶段目标或停止条件")
    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--deadline-seconds", type=int, default=6 * 3600)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args(argv)
    result = wait(args.stages, args.deadline_seconds, args.poll_seconds)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
