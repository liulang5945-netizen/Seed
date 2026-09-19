"""Same-entry selection evidence for the HANDOFF-M4 product path.

Runs three deterministic group episodes through the product execution entry
(``taiji.collab_handoff.execute_group_episode``) and records the event log:
the evidence that the product path (not an instrument) triggers failure
handoff and the terminal stop reasons, with ``rule_revision`` on every event.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from taiji.collab_handoff import (  # noqa: E402
    STOP_ALL_MEMBERS_BLOCKED,
    STOP_ALL_MEMBERS_EXHAUSTED,
    ExecutionStep,
    FailureHandoffPolicy,
    MemberCall,
    execute_group_episode,
)

REPORT_FORMAT = "taiji-collab-handoff-entry-evidence-v1"
REPORT = Path("reports/taiji_collab_handoff_entry_evidence_20260919.json")


def _scenario_handoff(state: dict[str, Any]):
    """m0 binds and fails once (handoff to m1), m1 executes to the goal."""

    def invoke(member: str, cue_count: int) -> MemberCall:
        return MemberCall(member=member, bind_failure=None)

    def execute(member: str) -> bool:
        executed = state["m0_failures"] >= 1
        if member == "m0" and not executed:
            state["m0_failures"] += 1
            return False
        state["done"] = True
        return True

    def goal() -> bool:
        return bool(state["done"])

    return invoke, execute, goal


def _scenario_all_blocked(state: dict[str, Any]):
    def invoke(member: str, cue_count: int) -> MemberCall:
        return MemberCall(member=member, bind_failure=None)

    def execute(member: str) -> bool:
        return False

    def goal() -> bool:
        return False

    return invoke, execute, goal


def _scenario_exhausted(state: dict[str, Any]):
    def invoke(member: str, cue_count: int) -> MemberCall:
        return MemberCall(member=member, bind_failure="bind_error")

    def execute(member: str) -> bool:
        return False

    def goal() -> bool:
        return False

    return invoke, execute, goal


def main() -> int:
    members = ("m0", "m1", "m2")
    policy = FailureHandoffPolicy(members, rule_revision=1)
    episodes = []
    for name, scenario in (
        ("handoff_then_goal", _scenario_handoff),
        ("all_members_blocked", _scenario_all_blocked),
        ("all_members_exhausted", _scenario_exhausted),
    ):
        state: dict[str, Any] = {"m0_failures": 0, "done": False}
        invoke, execute, goal = scenario(state)
        episodes.append(execute_group_episode(
            active_members=members,
            episode_id=name,
            policy=policy,
            invoke_member=invoke,
            execute_chosen=execute,
            goal_reached=goal,
            max_steps=8,
        ))

    stops = {episode["stop_reason"] for episode in episodes}
    handoff = episodes[0]
    handoff_events = [
        event
        for event in handoff["events"]
        if event["kind"] == "member_executed" and event["member"] != "m0"
    ]
    evidence = {
        "format": REPORT_FORMAT,
        "version": 1,
        "contract": "plans/reference/M5_COLLAB_HANDOFF_M4_PRODUCTIZATION_DRAFT_20260919.md",
        "composition_rule": policy.composition_rule,
        "rule_revision": policy.rule_revision,
        "entry": "taiji.collab_handoff.execute_group_episode",
        "episodes": episodes,
        "checks": {
            "all_events_carry_rule_revision": all(
                "rule_revision" in event for episode in episodes for event in episode["events"]
            ),
            "handoff_demonstrated": any(
                event["kind"] == "member_executed" and event["member"] == "m1"
                for event in handoff["events"]
            ) and bool(handoff_events),
            "all_members_blocked_demonstrated": STOP_ALL_MEMBERS_BLOCKED in stops,
            "all_members_exhausted_demonstrated": STOP_ALL_MEMBERS_EXHAUSTED in stops,
            "handoff_episode_reached_goal": handoff["stop_reason"] == "goal_reached",
        },
        "outcome": "passed",
    }
    checks = evidence["checks"]
    evidence["outcome"] = "passed" if all(checks.values()) else "failed"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outcome": evidence["outcome"], "checks": checks}, ensure_ascii=False))
    return 0 if evidence["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
