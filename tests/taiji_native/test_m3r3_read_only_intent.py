from __future__ import annotations

from scripts.training.eval_taiji_m3r1_native_observation import _course_registry
from scripts.training.eval_taiji_m3r2_task_state_sequence import _world
from scripts.training.eval_taiji_m3r3_read_only_intent import (
    _catalog_semantics,
    run_gate,
)
from seed_platform.workbench import WorkbenchEnvironment
from taiji import (
    NativeReadOnlyIntentPlanner,
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
    assert NativeReadOnlyIntentPlanner.from_checkpoint(
        planner.checkpoint()
    ).checkpoint() == planner.checkpoint()


def test_m3r3_native_read_only_intent_gate(tmp_path) -> None:
    report = run_gate(tmp_path / "m3r3.json")
    assert report["status"] == "passed"
    assert report["can_promote"] is False
    assert all(report["gates"].values())
