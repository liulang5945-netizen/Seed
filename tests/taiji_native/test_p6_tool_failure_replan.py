from __future__ import annotations

from taiji import (
    EnvironmentOutcome,
    EpisodicMemoryStore,
    GenerationController,
    Goal,
    GoalPlanner,
    ImaginedRollout,
    PlanningCandidate,
    TSKV8Adapter,
    WorldAction,
)


class FailingThenRecoveringEnvironment:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute_tool(self, tool_name: str, parameters: dict[str, object]) -> EnvironmentOutcome:
        del parameters
        self.calls.append(tool_name)
        recovered = tool_name == "weather.recover.v1"
        return EnvironmentOutcome(
            sensation=98 if recovered else 99,
            reward=0.4 if recovered else -1.0,
            success=recovered,
            terminal=recovered,
        )


def _candidate(rollout_id: str, kind: str, reward: float, progress: float) -> PlanningCandidate:
    return PlanningCandidate(
        candidate_id=f"{rollout_id}-step-0",
        action=WorldAction(
            action_id=f"{rollout_id}-action-0",
            kind=kind,
            tick=1,
            provenance="imagined",
        ),
        predicted_reward=reward,
        success_probability=0.9,
        expected_progress=progress,
        uncertainty=0.1,
    )


def test_tool_failure_replans_and_recovers_through_existing_planner() -> None:
    environment = FailingThenRecoveringEnvironment()
    adapter = TSKV8Adapter()
    adapter.attach_goal_planner(GoalPlanner())
    adapter.attach_generation_controller(GenerationController())
    adapter.attach_episodic_memory(EpisodicMemoryStore(capacity=8))
    adapter.set_goals((Goal("stay-informed", "get current information", priority=1.0),))
    adapter.observe(97, learn=False)

    adapter.plan_rollouts(
        (
            ImaginedRollout(
                rollout_id="lookup-first",
                goal_id="stay-informed",
                confidence=0.9,
                steps=(_candidate("lookup-first", "lookup_weather", 1.0, 0.8),),
            ),
        )
    )
    adapter.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("lookup_weather", "idle"),
        use_plan=True,
    )
    first_call = adapter.generate_tool_call(tool_name="weather.lookup.v1")
    first_outcome = adapter.execute_tool_call(environment, call=first_call, learn=False)

    assert first_outcome.success is False
    assert adapter.replan_required is True
    assert adapter.last_rollout_prediction_error is not None
    assert adapter.last_rollout_prediction_error > 0.25

    adapter.plan_rollouts(
        (
            ImaginedRollout(
                rollout_id="recover-second",
                goal_id="stay-informed",
                confidence=0.95,
                steps=(_candidate("recover-second", "recover_weather", 0.2, 0.7),),
            ),
        )
    )
    adapter.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("recover_weather", "idle"),
        use_plan=True,
    )
    second_call = adapter.generate_tool_call(tool_name="weather.recover.v1")
    second_outcome = adapter.execute_tool_call(environment, call=second_call, learn=False)

    assert second_outcome.success is True
    assert adapter.replan_required is False
    assert environment.calls == ["weather.lookup.v1", "weather.recover.v1"]
    assert adapter._episodic_memory is not None
    assert adapter._episodic_memory.count == 2

