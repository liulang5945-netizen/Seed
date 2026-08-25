from __future__ import annotations

import pytest

from taiji import (
    EnvironmentOutcome,
    EpisodicMemoryStore,
    GenerationController,
    Goal,
    GoalPlanner,
    PlanningCandidate,
    TaijiToolEnvironment,
    TSKV8Adapter,
    WorldAction,
)


class WeatherToolEnvironment:
    def execute_tool(self, tool_name: str, parameters: dict[str, object]) -> EnvironmentOutcome:
        valid = tool_name == "weather.lookup.v1" and parameters["action_symbol"] == 10
        return EnvironmentOutcome(
            sensation=98 if valid else 99,
            reward=1.0 if valid else -1.0,
            success=valid,
            terminal=True,
        )


def _adapter(*, generation: bool) -> TSKV8Adapter:
    adapter = TSKV8Adapter()
    adapter.attach_goal_planner(GoalPlanner())
    adapter.set_goals((Goal("stay-informed", "get current information", priority=1.0),))
    adapter.attach_episodic_memory(EpisodicMemoryStore(capacity=8))
    if generation:
        adapter.attach_generation_controller(GenerationController())
    adapter.observe(97, learn=False)
    adapter.plan_actions(
        (
            PlanningCandidate(
                candidate_id="lookup",
                action=WorldAction(action_id="lookup", kind="lookup_weather", tick=1),
                predicted_reward=0.8,
                success_probability=0.9,
                expected_progress=0.8,
            ),
            PlanningCandidate(
                candidate_id="idle",
                action=WorldAction(action_id="idle", kind="idle", tick=1),
                predicted_reward=0.0,
                success_probability=0.4,
                expected_progress=0.0,
            ),
        )
    )
    adapter.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("lookup_weather", "idle"),
        use_plan=True,
    )
    return adapter


def test_structured_tool_execution_writes_outcome_and_memory() -> None:
    environment = WeatherToolEnvironment()
    assert isinstance(environment, TaijiToolEnvironment)
    adapter = _adapter(generation=True)
    call = adapter.generate_tool_call(tool_name="weather.lookup.v1")

    outcome = adapter.execute_tool_call(environment, call=call, learn=False)

    assert outcome.intent_id == call.intent_id
    assert outcome.reward == 1.0
    assert outcome.success is True
    assert outcome.terminal is True
    assert adapter.generation_trace is not None
    store = adapter._episodic_memory
    assert store is not None
    assert store.count == 1
    assert store.records[0].outcome == outcome
    assert store.records[0].action_intent is not None
    assert store.records[0].action_intent.kind == "lookup_weather"


def test_direct_byte_lesion_cannot_execute_structured_tool() -> None:
    adapter = _adapter(generation=False)
    with pytest.raises(RuntimeError, match="generated ToolCall"):
        adapter.execute_tool_call(WeatherToolEnvironment(), learn=False)

