"""Deadline-bearing drive task and its fixture validity checks (22.6.0).

22.6 asks whether the subject chooses to seek, replay, consolidate or wait, and
whether those choices predict the development they produce. None of that can be
measured unless the task actually has three properties: a deadline that binds, an
action whose value is real, and a situation where waiting is the correct move.

This module freezes the schedule and its arithmetic only. It does not step the
world, load a model, or read any project data:

  * the schedule is a declared table of decision opportunities per session;
  * each opportunity states when the target becomes knowable, when acting pays,
    what a guess before the reveal pays, and when the deadline closes;
  * ``assess_fixture`` returns a named verdict, and the plan's own stop condition
    is one of its failures: if waiting is optimal because there is no task at all,
    the fixture is invalid rather than the subject being idle.

Values are exact integers on a tick grid, so a reader can recompute every number
by hand and a red test cannot pass by rounding.
"""

from __future__ import annotations

import hashlib
import json

SCHEMA = "taiji-u2-drive-task-v1"
SESSIONS = 3
OPPORTUNITIES = 16
OPERATIONS = (
    "act",
    "express",
    "inspect",
    "ask",
    "replay",
    "consolidate",
    "grow",
    "rest",
    "wait",
)

# One session is thirteen valued opportunities plus a contiguous prompt-free
# stretch of three. Composition of the thirteen: seven pay only if acted on
# immediately, four only after the target is revealed at tick 3, two only after a
# result that arrives two ticks late.
COMPOSITION = (("early", 7), ("late", 4), ("pending", 2))
QUIET_POSITIONS = (6, 7, 8)  # contiguous, no new prompt, value as "early"
QUIET_PROFILE = ("early", "late", "early")

# "early": now or never — deferring by one tick already forfeits the value, so a
# policy that inspects before every action pays for the habit.
# "late":  acting at tick 0 is a guess worth nothing; the target is knowable at
#          tick 3 and pays until tick 6.
# "pending": the previous action's result arrives two ticks late; acting earlier
#          is the same valueless guess.
PROFILES = {
    "early": {"reveal": 0, "start": 0, "end": 0, "deadline": 6, "full": 1, "guess": 0},
    "late": {"reveal": 3, "start": 3, "end": 6, "deadline": 8, "full": 1, "guess": 0},
    "pending": {"reveal": 2, "start": 2, "end": 5, "deadline": 7, "full": 1, "guess": 0},
}

HELP_BYTES_PER_SESSION = 96
FEEDBACK_AT = 5
FEEDBACK_AFFECTS = (9, 12)


def _permutation(seed: int, count: int, tag: str) -> list[int]:
    """Explicit Fisher-Yates driven by digest bytes, not by a library PRNG."""
    order = list(range(count))
    key = hashlib.sha256(f"{SCHEMA}|{seed}|{tag}".encode()).digest()
    for index in range(count - 1, 0, -1):
        key = hashlib.sha256(key).digest()
        swap = int.from_bytes(key[:4], "little") % (index + 1)
        order[index], order[swap] = order[swap], order[index]
    return order


def _profiles(seed: int, session: int) -> list[str]:
    flat: list[str] = []
    for tag, repeats in COMPOSITION:
        flat.extend([tag] * repeats)
    if len(flat) != OPPORTUNITIES - len(QUIET_POSITIONS):
        raise ValueError("drive_task_composition_is_wrong")
    order = _permutation(seed, len(flat), f"session{session}")
    shuffled = [flat[index] for index in order]
    result: list[str] = []
    cursor = 0
    for index in range(OPPORTUNITIES):
        if index in QUIET_POSITIONS:
            result.append(QUIET_PROFILE[QUIET_POSITIONS.index(index)])
        else:
            result.append(shuffled[cursor])
            cursor += 1
    return result


def build_schedule(seed: int, *, sessions: int = SESSIONS) -> dict:
    """Build the declared schedule. The same seed always yields the same bytes."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("invalid_drive_task_seed")
    if sessions != SESSIONS:
        raise ValueError("drive_task_requires_three_sessions")
    rows: list[dict] = []
    for session in range(sessions):
        profiles = _profiles(seed, session)
        opportunities = []
        for index, tag in enumerate(profiles):
            profile = PROFILES[tag]
            quiet = index in QUIET_POSITIONS
            opportunity = {
                "index": index,
                "kind": tag,
                "prompt": None if quiet else f"session{session}:step{index}",
                "reveal_tick": profile["reveal"],
                "window": [profile["start"], profile["end"]],
                "deadline_tick": profile["deadline"],
                "value_full": profile["full"],
                "value_guess": profile["guess"],
            }
            if tag == "pending":
                opportunity["pending_from"] = index - 1 if index else None
                opportunity["result_delay_ticks"] = profile["reveal"]
            opportunities.append(opportunity)
        rows.append(
            {
                "session": session,
                "opportunities": opportunities,
                "quiet_run": list(QUIET_POSITIONS),
                "feedback_change": {
                    "at": FEEDBACK_AT,
                    "from": "colour_cue",
                    "to": "position_cue",
                    "affects": list(FEEDBACK_AFFECTS),
                },
                "deadline_tick": max(o["deadline_tick"] for o in opportunities),
                "help": {
                    "bytes": HELP_BYTES_PER_SESSION,
                    "removed_arm": True,
                    "note": "the same byte allowance is granted to the helped arm only",
                },
            }
        )
    schedule = {
        "schema": SCHEMA,
        "seed": seed,
        "sessions": rows,
        "rights": {
            "operations": list(OPERATIONS),
            "public": [
                "goal_raw",
                "request_raw",
                "authorized_records",
                "observations",
                "coordinates",
                "action",
                "legal_mask",
                "deadline",
            ],
            "private": ["body_id", "position", "velocity", "target", "score_only", "seed"],
            "resource": {"world_ticks_per_session": 8, "decision_opportunities": OPPORTUNITIES},
        },
        "stop_conditions": {
            "wait_idle_optimal": "fixture_invalid",
            "deadline_not_binding": "fixture_invalid",
            "no_feedback_change": "fixture_invalid",
            "no_pending_result": "fixture_invalid",
        },
    }
    schedule["digest"] = schedule_digest(schedule)
    return schedule


def schedule_digest(schedule: dict) -> str:
    body = {key: value for key, value in schedule.items() if key != "digest"}
    text = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def value_of(opportunity: dict, operation: str, tick: int) -> int:
    """Value of performing ``operation`` at ``tick`` in this opportunity."""
    if operation not in OPERATIONS:
        raise ValueError(f"unknown_operation:{operation}")
    if operation != "act":
        return 0
    if tick < 0 or tick > opportunity["deadline_tick"]:
        return 0
    if tick < opportunity["reveal_tick"]:
        return opportunity["value_guess"]
    start, end = opportunity["window"]
    return opportunity["value_full"] if start <= tick <= end else 0


def best_now(opportunity: dict) -> int:
    return value_of(opportunity, "act", 0)


def best_deferred(opportunity: dict) -> int:
    """Best value reachable by spending at least one tick before acting."""
    return max(
        (value_of(opportunity, "act", tick) for tick in range(1, opportunity["deadline_tick"] + 1)),
        default=0,
    )


def _flat(schedule: dict) -> list[dict]:
    return [row for session in schedule["sessions"] for row in session["opportunities"]]


def assess_fixture(schedule: dict) -> dict:
    """Named validity verdict for a schedule. No model, no world, no data."""
    failures: list[str] = []
    sessions = schedule.get("sessions") or []
    if len(sessions) != SESSIONS:
        failures.append("session_count")
    if any(len(row.get("opportunities") or []) != OPPORTUNITIES for row in sessions):
        failures.append("opportunity_count")

    flat = _flat(schedule)
    total_now = sum(best_now(o) for o in flat)
    total_deferred = sum(best_deferred(o) for o in flat)
    total_best = sum(max(best_now(o), best_deferred(o)) for o in flat)

    # The plan's literal stop: if there is no task opportunity, waiting forever is
    # optimal and the fixture — not the subject — is what failed.
    if total_best == 0:
        failures.append("wait_idle_optimal")
        failures.append("no_action_value")

    # A deadline that does not bind makes the card untestable: value must be zero
    # strictly after the deadline, and some window must close before the deadline.
    for opportunity in flat:
        if value_of(opportunity, "act", opportunity["deadline_tick"] + 1) != 0:
            failures.append("deadline_not_binding")
            break
    if not any(o["window"][1] < o["deadline_tick"] for o in flat):
        failures.append("window_equals_deadline_everywhere")

    if not any(best_deferred(o) > best_now(o) for o in flat):
        failures.append("wait_never_optimal")
    if not any(best_now(o) > best_deferred(o) for o in flat):
        failures.append("wait_always_optimal")

    for row in sessions:
        change = row.get("feedback_change") or {}
        if not change or not change.get("affects"):
            failures.append("no_feedback_change")
        elif min(change["affects"]) <= change.get("at", OPPORTUNITIES):
            failures.append("feedback_change_affects_nothing_later")
        quiet = row.get("quiet_run") or []
        if len(quiet) < 3 or quiet[-1] - quiet[0] + 1 != len(quiet):
            failures.append("no_contiguous_prompt_free_run")
        elif any(row["opportunities"][index]["prompt"] is not None for index in quiet):
            failures.append("prompt_inside_the_quiet_run")
        if not any(o["kind"] == "pending" for o in row["opportunities"]):
            failures.append("no_pending_result")
        help_row = row.get("help") or {}
        if help_row.get("bytes", 0) <= 0 or not help_row.get("removed_arm"):
            failures.append("help_not_metered_or_not_removed")

    failures = list(dict.fromkeys(failures))
    return {
        "schema": SCHEMA,
        "digest": schedule.get("digest"),
        "valid": not failures,
        "verdict": "fixture_valid" if not failures else "fixture_invalid",
        "failures": failures,
        "totals": {"act_now": total_now, "deferred": total_deferred, "oracle": total_best},
        "counts": {
            "sessions": len(sessions),
            "opportunities": len(flat),
            "requires_waiting": sum(1 for o in flat if best_deferred(o) > best_now(o)),
            "requires_immediate": sum(1 for o in flat if best_now(o) > best_deferred(o)),
            "pending_results": sum(1 for o in flat if o["kind"] == "pending"),
            "prompt_free": sum(1 for o in flat if o["prompt"] is None),
        },
        "checked": [
            "session_count",
            "opportunity_count",
            "wait_idle_optimal",
            "no_action_value",
            "deadline_not_binding",
            "window_equals_deadline_everywhere",
            "wait_never_optimal",
            "wait_always_optimal",
            "no_feedback_change",
            "feedback_change_affects_nothing_later",
            "no_contiguous_prompt_free_run",
            "prompt_inside_the_quiet_run",
            "no_pending_result",
            "help_not_metered_or_not_removed",
        ],
        "not_checked_here": [
            "actual execution receipts (22.6.2)",
            "policy learning and net utility (22.6.2-22.6.5)",
        ],
    }
