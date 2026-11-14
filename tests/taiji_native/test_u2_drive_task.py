"""Contract tests for the 22.6.0 drive task and its validity checks.

The schedule is arithmetic only, so every assertion here is a recomputation a
reader can repeat by hand. Two of the tests are cross-checks against files this
module does not own: the nine operations must match the world contract frozen in
22.4, and the public information list must match the same contract's permissions.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from taiji.u2_drive_task import (
    OPERATIONS,
    OPPORTUNITIES,
    SESSIONS,
    assess_fixture,
    best_deferred,
    best_now,
    build_schedule,
    value_of,
)

ROOT = Path(__file__).resolve().parents[2]
WORLD_CONTRACT = ROOT / "plans/manifests/taiji_u2_22_4_2_world_contract_20261112.json"


def _contract() -> dict:
    return json.loads(WORLD_CONTRACT.read_text(encoding="utf-8"))


def test_schedule_is_deterministic_and_seed_dependent() -> None:
    first = build_schedule(20261110)
    second = build_schedule(20261110)
    other = build_schedule(20261111)
    assert first["digest"] == second["digest"]
    assert first["digest"] != other["digest"]
    assert first["sessions"] == second["sessions"]


def test_every_session_has_sixteen_opportunities_and_three_quiet_steps() -> None:
    schedule = build_schedule(20261110)
    assert len(schedule["sessions"]) == SESSIONS
    for row in schedule["sessions"]:
        assert len(row["opportunities"]) == OPPORTUNITIES
        prompts = [o["prompt"] for o in row["opportunities"]]
        assert prompts.count(None) == 3
        first = prompts.index(None)
        assert prompts[first : first + 3] == [None, None, None]


def test_deadline_closes_the_window_and_value_is_zero_after_it() -> None:
    for row in build_schedule(20261110)["sessions"]:
        for opportunity in row["opportunities"]:
            assert value_of(opportunity, "act", opportunity["deadline_tick"] + 1) == 0
            assert value_of(opportunity, "act", -1) == 0
            start, end = opportunity["window"]
            assert value_of(opportunity, "act", start) == opportunity["value_full"]
            assert value_of(opportunity, "act", end) == opportunity["value_full"]
            assert value_of(opportunity, "act", end + 1) == 0


def test_only_act_carries_value() -> None:
    row = build_schedule(20261110)["sessions"][0]
    opportunity = row["opportunities"][0]
    for operation in OPERATIONS:
        expected = opportunity["value_full"] if operation == "act" else 0
        assert value_of(opportunity, operation, opportunity["window"][0]) == expected


def test_waiting_is_sometimes_required_and_sometimes_wrong() -> None:
    flat = [o for row in build_schedule(20261110)["sessions"] for o in row["opportunities"]]
    assert sum(1 for o in flat if best_deferred(o) > best_now(o)) > 0
    assert sum(1 for o in flat if best_now(o) > best_deferred(o)) > 0


def test_feedback_change_lands_before_the_opportunities_it_affects() -> None:
    for row in build_schedule(20261110)["sessions"]:
        change = row["feedback_change"]
        assert change["affects"]
        assert min(change["affects"]) > change["at"]
        assert change["from"] != change["to"]


def test_pending_result_arrives_after_a_real_delay() -> None:
    pending = [
        o
        for row in build_schedule(20261110)["sessions"]
        for o in row["opportunities"]
        if o["kind"] == "pending"
    ]
    assert pending
    for opportunity in pending:
        assert opportunity["result_delay_ticks"] > 0
        assert opportunity["result_delay_ticks"] == opportunity["reveal_tick"]
        assert best_now(opportunity) == 0


def test_rights_match_the_frozen_world_contract() -> None:
    contract = _contract()
    schedule = build_schedule(20261110)
    assert list(OPERATIONS) == contract["action"]["operations"]
    assert schedule["rights"]["public"] == contract["permissions"]["public"]


def test_declared_schedule_is_valid() -> None:
    verdict = assess_fixture(build_schedule(20261110))
    assert verdict["valid"], verdict["failures"]
    assert verdict["verdict"] == "fixture_valid"
    assert verdict["totals"]["oracle"] > verdict["totals"]["act_now"] > 0
    assert verdict["counts"]["pending_results"] == SESSIONS * 2


def test_idle_schedule_is_rejected_as_fixture_invalid() -> None:
    """The plan's own stop: no task opportunity means waiting forever is optimal."""
    schedule = build_schedule(20261110)
    for row in schedule["sessions"]:
        for opportunity in row["opportunities"]:
            opportunity["value_full"] = 0
            opportunity["value_guess"] = 0
    verdict = assess_fixture(schedule)
    assert not verdict["valid"]
    assert verdict["verdict"] == "fixture_invalid"
    assert "wait_idle_optimal" in verdict["failures"]


def test_deadline_that_never_closes_early_is_rejected() -> None:
    schedule = build_schedule(20261110)
    for row in schedule["sessions"]:
        for opportunity in row["opportunities"]:
            opportunity["deadline_tick"] = opportunity["window"][1]
    verdict = assess_fixture(schedule)
    assert "window_equals_deadline_everywhere" in verdict["failures"]


def test_schedule_that_never_rewards_waiting_is_rejected() -> None:
    schedule = build_schedule(20261110)
    for row in schedule["sessions"]:
        for opportunity in row["opportunities"]:
            opportunity["kind"] = "early"
            opportunity["reveal_tick"] = 0
            opportunity["window"] = [0, 0]
            opportunity["deadline_tick"] = 6
            opportunity["value_guess"] = 0
    verdict = assess_fixture(schedule)
    assert "wait_never_optimal" in verdict["failures"]


def test_prompt_inside_the_quiet_run_is_rejected() -> None:
    schedule = build_schedule(20261110)
    schedule["sessions"][0]["opportunities"][7]["prompt"] = "interrupt"
    verdict = assess_fixture(schedule)
    assert "prompt_inside_the_quiet_run" in verdict["failures"]


def test_missing_pending_result_is_rejected() -> None:
    schedule = build_schedule(20261110)
    for row in schedule["sessions"]:
        for opportunity in row["opportunities"]:
            if opportunity["kind"] == "pending":
                opportunity["kind"] = "early"
    verdict = assess_fixture(schedule)
    assert "no_pending_result" in verdict["failures"]


def test_feedback_change_without_later_effect_is_rejected() -> None:
    schedule = build_schedule(20261110)
    for row in schedule["sessions"]:
        row["feedback_change"]["affects"] = [row["feedback_change"]["at"]]
    verdict = assess_fixture(schedule)
    assert "feedback_change_affects_nothing_later" in verdict["failures"]


def test_assess_does_not_mutate_the_schedule() -> None:
    schedule = build_schedule(20261110)
    before = copy.deepcopy(schedule)
    assess_fixture(schedule)
    assert schedule == before
