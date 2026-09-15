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
#: Only the current key is read.  The earlier name is reserved for episode termination reasons
#: (the N2 audit surface), so a compatibility shim reading both keys must not live here; pre-rename
#: artifacts are read with a one-off command.
STOP_KEY = "campaign_stop"
#: Checkpoints produced but not scored, at which point we call the driver dead.  The driver polls
#: every 60 s and a stage costs ~206 s, so one pending checkpoint is normal; two means it has been
#: ~1.8 h of training since anybody looked, which is what an orphaned trainer looks like: the
#: campaign driver raising on a failed stage evaluation exits, while its child keeps writing
#: checkpoints to the arm path forever.
STALL_THRESHOLD = 2


def read_arm(arm: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "reports" / f"taiji_p3b_campaign_{arm}_20260915.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # rewritten atomically; a torn read just retries
        return {}


def progress_tick(arm: str) -> int | None:
    """The trainer's own last reported tick -- it advances even when the driver is gone."""

    path = PROJECT_ROOT / "reports" / f"p3b_aligned_progress_{arm}.jsonl"
    if not path.exists():
        return None
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return int(json.loads(lines[-1])["ticks"]) if lines else None
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def unscored_checkpoints(
    current_tick: int | None, scored_tick: int | None, checkpoint_every: int | None
) -> int:
    """How many whole checkpoint intervals have elapsed since the last **scored** one."""

    if current_tick is None or scored_tick is None or not checkpoint_every or checkpoint_every <= 0:
        return 0
    return max(0, (int(current_tick) - int(scored_tick)) // int(checkpoint_every))


def stop_of(report: dict[str, Any]) -> Any:
    return report.get(STOP_KEY)


def reached(stages: dict[str, Any], target: int) -> bool:
    return any(len(report.get("stages") or []) >= target for report in stages.values())


def arm_state(report: dict[str, Any]) -> dict[str, Any]:
    arm = report.get("arm")
    stages = (report or {}).get("stages") or []
    scored_tick = next((row.get("tick") for row in reversed(stages) if row.get("tick")), None)
    current_tick = progress_tick(str(arm)) if arm else None
    owed = unscored_checkpoints(current_tick, scored_tick, report.get("checkpoint_every"))
    return {
        "stages": len(stages),
        "status": report.get("status"),
        "stop": stop_of(report),
        "latest_scored_tick": scored_tick,
        "trainer_tick": current_tick,
        "checkpoints_owed": owed,
        "driver_stalled": owed >= STALL_THRESHOLD,
    }


def wait(target: int, deadline_seconds: int, poll_seconds: int) -> dict[str, Any]:
    deadline = time.time() + deadline_seconds
    state: dict[str, Any] = {arm: {"stages": 0, "status": "missing"} for arm in ARMS}
    while time.time() < deadline:
        reports = {arm: read_arm(arm) for arm in ARMS}
        state = {arm: arm_state(reports[arm]) for arm in ARMS}
        stops = {arm: state[arm]["stop"] for arm in ARMS}
        stalled = [arm for arm in ARMS if state[arm]["driver_stalled"]]
        if stalled:
            return {"reason": "driver_stalled", "arms": stalled, "state": state}
        if any(stops.values()) or reached(reports, target):
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
    return 1 if result["reason"] == "driver_stalled" else 0


if __name__ == "__main__":
    raise SystemExit(main())
