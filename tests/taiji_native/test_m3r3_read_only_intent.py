from __future__ import annotations

import pytest

from scripts.training.eval_taiji_m3r1_native_observation import _course_registry
from scripts.training.eval_taiji_m3r2_task_state_sequence import _world
from scripts.training.eval_taiji_m3r3_read_only_intent import (
    _catalog_semantics,
    run_gate,
)
from seed_platform.workbench import WorkbenchEnvironment
from taiji import (
    ContentPlan,
    Goal,
    NativeReadOnlyIntentPlanner,
    ReadOnlyAbstention,
    ReadOnlyIntentDecision,
    ReadOnlyIntentPolicy,
)


def test_read_only_intent_policy_and_decision_roundtrip() -> None:
    from scripts.training.eval_taiji_m3r2_task_state_sequence import (
        _sequence_for_split,
        build_transition_course,
    )

    corpus, observations = build_transition_course()
    sequence = _sequence_for_split(observations["test"])
    policy = ReadOnlyIntentPolicy(
        routes=(
            ("content:inspect-language", "workspace.read"),
            (
                "content:clarify-toolchain",
                "workspace.programming_language.resolve",
            ),
        )
    )
    planner = NativeReadOnlyIntentPlanner(policy)
    environment = WorkbenchEnvironment(
        root="tests/fixtures/m3_workbench_projects/native_observation_test",
        programming_language_registry=_course_registry(),
    )
    goal, content = _catalog_semantics(corpus, sequence[1], tick=1)
    decision = planner.propose(
        observation=sequence[1],
        world=_world(sequence[1], tick=1),
        goal=goal,
        content=content,
        capability_snapshot=environment.capability_snapshot,
        tick=1,
    )
    restored = ReadOnlyIntentDecision.from_payload(decision.to_payload())
    assert decision.accepted is True
    assert decision.action_intent is not None
    assert restored.to_payload() == decision.to_payload()
    assert (
        NativeReadOnlyIntentPlanner.from_checkpoint(planner.checkpoint()).checkpoint()
        == planner.checkpoint()
    )


def test_read_only_abstention_and_missing_target_recovery_contract() -> None:
    from scripts.training.eval_taiji_m3r2_task_state_sequence import (
        _sequence_for_split,
        build_transition_course,
    )

    _, observations = build_transition_course()
    sequence = _sequence_for_split(observations["test"])
    environment = WorkbenchEnvironment(
        root="tests/fixtures/m3_workbench_projects/native_observation_test",
        programming_language_registry=_course_registry(),
    )
    recovery_policy = ReadOnlyIntentPolicy(
        routes=(("content:recover-target", "workspace.list"),),
        route_parameters=(("content:recover-target", (("path", "."),)),),
    )
    planner = NativeReadOnlyIntentPlanner(recovery_policy)
    abstention = planner.abstain(
        observation=sequence[1],
        capability_snapshot=environment.capability_snapshot,
        reason_code="input_confidence_below_floor",
        next_step="request_clarification",
        confidence=0.3,
    )
    restored_abstention = ReadOnlyAbstention.from_payload(abstention.to_payload())
    assert restored_abstention.to_payload() == abstention.to_payload()
    assert abstention.to_payload()["action_intent"] is None

    recovery_goal = Goal(
        goal_id="goal:recover-target",
        description="Recover a missing Workbench target.",
        priority=0.9,
    )
    recovery_content = ContentPlan(
        content_id="content:recover-target",
        intent_id="intent:recover-target",
        intent_kind="request_information",
        semantic_slots={},
        source_goal_id=recovery_goal.goal_id,
        expected_outcome="workspace candidate listing",
        tick=1,
    )
    decision = planner.propose(
        observation=sequence[1],
        world=_world(sequence[1], tick=1),
        goal=recovery_goal,
        content=recovery_content,
        capability_snapshot=environment.capability_snapshot,
        tick=1,
    )
    assert decision.accepted is True
    assert decision.action_intent is not None
    assert decision.action_intent.kind == "workspace.list"
    assert decision.action_intent.parameters == {"path": "."}
    restored_policy = NativeReadOnlyIntentPlanner.from_checkpoint(planner.checkpoint())
    assert restored_policy.checkpoint() == planner.checkpoint()
    with pytest.raises(ValueError, match="route parameters must be mappings"):
        ReadOnlyIntentPolicy.from_payload(
            {
                "format": "taiji-read-only-intent-policy-v1",
                "version": 1,
                "routes": {"content:recover-target": "workspace.list"},
                "route_parameters": {"content:recover-target": ["path", "."]},
            }
        )


def test_m3r3_native_read_only_intent_gate(tmp_path) -> None:
    report = run_gate(tmp_path / "m3r3.json")
    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert all(report["gates"].values())
