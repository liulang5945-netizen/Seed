"""HANDOFF-M4 product-component gates (implementation gate, zero training).

Contract: plans/reference/M5_COLLAB_HANDOFF_M4_PRODUCTIZATION_DRAFT_20260919.md §3

The core gate is instrument-product alignment: a VERBATIM reference copy of
the P5.2b gate's decision block (the frozen measurement instrument) is
cross-checked against ``taiji.collab_handoff.FailureHandoffPolicy`` on
randomized scenarios -- the product rule must reproduce the instrument's
decisions exactly.  Also pinned: revision-0 reproducibility, terminal stop
reasons, blocked-release after success, payload round-trip, validation, and
the per-member cue rule.
"""

from __future__ import annotations

import random

import pytest

from taiji.collab_handoff import (
    COMPOSITION_RULE,
    RULE_REVISION,
    STOP_ALL_MEMBERS_BLOCKED,
    STOP_ALL_MEMBERS_EXHAUSTED,
    ExecutionStep,
    FailureHandoffPolicy,
    MemberCall,
    member_cue_count,
    member_own_steps,
)


def _reference_decision(active_members, calls, steps):
    """VERBATIM reference of the P5.2b gate's decision block
    (eval_taiji_p5_2b_group_causal_corpora_gate.py::_member_episode, WP-3
    landed revision 1).  Kept literally so drift shows up as a test failure."""

    by_member = {c["member"]: c for c in calls}
    bindable = [
        by_member[member]
        for member in active_members
        if member in by_member and by_member[member]["bind_failure"] is None
    ]
    if not bindable:
        return None, "all_members_exhausted"
    last_success_index = max(
        (index for index, step in enumerate(steps) if step.get("executed")),
        default=-1,
    )
    blocked = {
        step.get("chosen")
        for index, step in enumerate(steps)
        if step.get("chosen") is not None
        and not step.get("executed")
        and index > last_success_index
    }
    chosen = next((call for call in bindable if call["member"] not in blocked), None)
    if chosen is None:
        return None, "all_members_blocked"
    return chosen, None


def _random_scenario(rng: random.Random):
    members = [f"m{index}" for index in range(rng.randint(2, 5))]
    calls = [
        {"member": member, "bind_failure": None if rng.random() < 0.7 else "bind_error"}
        for member in members
    ]
    steps = []
    for _ in range(rng.randint(0, 8)):
        chosen = rng.choice(members)
        steps.append({"chosen": chosen, "executed": rng.random() < 0.6})
    return members, calls, steps


# --------------------------------------------------------------------------- #
# Gate 1: instrument-product alignment on randomized scenarios
# --------------------------------------------------------------------------- #


def test_gate1_alignment_with_gate_reference_on_random_scenarios() -> None:
    rng = random.Random(20260919)
    for _ in range(400):
        members, calls, steps = _random_scenario(rng)
        policy = FailureHandoffPolicy(members, rule_revision=RULE_REVISION)
        decision = policy.select(
            [MemberCall(member=c["member"], bind_failure=c["bind_failure"]) for c in calls],
            [ExecutionStep(chosen=s["chosen"], executed=s["executed"]) for s in steps],
        )
        ref_chosen, ref_stop = _reference_decision(members, calls, steps)
        expected_chosen = ref_chosen["member"] if ref_chosen else None
        assert decision.stop == ref_stop
        assert (decision.chosen.member if decision.chosen else None) == expected_chosen


def test_gate1_reference_exercises_all_outcomes() -> None:
    """The fuzz above must have covered exhausted / blocked / chosen paths."""

    seen = set()
    rng = random.Random(7)
    for _ in range(600):
        members, calls, steps = _random_scenario(rng)
        policy = FailureHandoffPolicy(members, rule_revision=RULE_REVISION)
        decision = policy.select(
            [MemberCall(member=c["member"], bind_failure=c["bind_failure"]) for c in calls],
            [ExecutionStep(chosen=s["chosen"], executed=s["executed"]) for s in steps],
        )
        seen.add(decision.stop or "chosen")
    assert {"chosen", STOP_ALL_MEMBERS_EXHAUSTED, STOP_ALL_MEMBERS_BLOCKED} <= seen


# --------------------------------------------------------------------------- #
# Gate 2: handoff semantics on deterministic scenarios (contract §2 rules)
# --------------------------------------------------------------------------- #


def _policy(members=("m0", "m1", "m2")):
    return FailureHandoffPolicy(members)


def test_gate2_handoff_skips_blocked_member() -> None:
    policy = _policy()
    calls = [
        MemberCall("m0", None),
        MemberCall("m1", None),
        MemberCall("m2", "bind_error"),
    ]
    # m0 was chosen after the last success and its attempt did not execute
    steps = [
        ExecutionStep(chosen="m0", executed=False),
    ]
    decision = policy.select(calls, steps)
    assert decision.chosen.member == "m1"
    assert decision.stop is None


def test_gate2_blocked_released_after_success() -> None:
    policy = _policy()
    calls = [MemberCall("m0", None), MemberCall("m1", None)]
    # a success happened after m0's failed attempt: m0 is no longer blocked
    steps = [
        ExecutionStep(chosen="m0", executed=False),
        ExecutionStep(chosen="m1", executed=True),
    ]
    decision = policy.select(calls, steps)
    assert decision.chosen.member == "m0"


def test_gate2_all_blocked_terminal() -> None:
    policy = _policy(("m0", "m1"))
    calls = [MemberCall("m0", None), MemberCall("m1", None)]
    steps = [
        ExecutionStep(chosen="m0", executed=False),
        ExecutionStep(chosen="m1", executed=False),
    ]
    decision = policy.select(calls, steps)
    assert decision.chosen is None
    assert decision.stop == STOP_ALL_MEMBERS_BLOCKED


def test_gate2_all_unbindable_exhausted() -> None:
    policy = _policy()
    calls = [MemberCall("m0", "bind_error"), MemberCall("m1", "bind_error")]
    decision = policy.select(calls, [])
    assert decision.chosen is None
    assert decision.stop == STOP_ALL_MEMBERS_EXHAUSTED


def test_gate2_revision0_ignores_blocked() -> None:
    calls = [MemberCall("m0", None), MemberCall("m1", None)]
    steps = [ExecutionStep(chosen="m0", executed=False)]
    policy = FailureHandoffPolicy(("m0", "m1"), rule_revision=0)
    decision = policy.select(calls, steps)
    assert decision.chosen.member == "m0"  # first bindable, blocked set ignored
    assert FailureHandoffPolicy(("m0", "m1"), rule_revision=0).composition_rule == COMPOSITION_RULE


# --------------------------------------------------------------------------- #
# Gate 3: payload round-trip, validation, cue rule
# --------------------------------------------------------------------------- #


def test_gate3_payload_roundtrip_and_validation() -> None:
    policy = _policy(("m0", "m1", "m2"))
    payload = policy.to_payload()
    assert payload["composition_rule"] == COMPOSITION_RULE
    assert payload["rule_revision"] == RULE_REVISION
    restored = FailureHandoffPolicy.from_payload(payload)
    assert restored.active_members == policy.active_members
    assert restored.rule_revision == policy.rule_revision
    with pytest.raises(ValueError, match="not a collab handoff policy"):
        FailureHandoffPolicy.from_payload({"format": "other"})
    with pytest.raises(ValueError, match="at least two"):
        FailureHandoffPolicy(("m0",))
    with pytest.raises(ValueError, match="unique"):
        FailureHandoffPolicy(("m0", "m0"))
    with pytest.raises(ValueError, match="unsupported rule revision"):
        FailureHandoffPolicy(("m0", "m1"), rule_revision=2)
    with pytest.raises(ValueError, match="non-active member"):
        policy.select([MemberCall("mX", None)], [])
    with pytest.raises(ValueError, match="duplicate member call"):
        policy.select([MemberCall("m0", None), MemberCall("m0", None)], [])


def test_gate3_member_cue_rule() -> None:
    steps = [
        ExecutionStep(chosen="m0", executed=True),
        ExecutionStep(chosen="m1", executed=False),
        ExecutionStep(chosen="m0", executed=True),
    ]
    assert member_own_steps(steps, "m0") == 2
    assert member_cue_count(steps, "m0") == 3  # own steps + 1
    assert member_own_steps(steps, "m1") == 0
    assert member_cue_count(steps, "m1") == 1
    assert member_own_steps(steps, "m2") == 0
