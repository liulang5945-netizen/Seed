from __future__ import annotations

from taiji import (
    ActionIntent,
    ContentPlan,
    ExpressionPlan,
    GenerationController,
    Goal,
    GoalPlanner,
    PlanningCandidate,
    StructuredToolCallCodec,
    ToolCall,
    TSKV8Adapter,
    WorldAction,
)


def test_content_expression_tool_call_round_trip_is_data_driven() -> None:
    intent = ActionIntent(
        intent_id="episode:intent:7",
        kind="lookup_weather",
        parameters={"city": "Shanghai", "days": 3},
        source_goal_id="stay-informed",
        expected_outcome="weather result",
        confidence=0.8,
        tick=7,
    )
    trace = GenerationController().generate_tool_call(
        intent,
        tool_name="weather.lookup.v1",
        channel="rpc",
    )

    assert isinstance(trace.content, ContentPlan)
    assert trace.content.semantic_slots == intent.parameters
    assert isinstance(trace.expression, ExpressionPlan)
    assert trace.expression.modality == "tool"
    assert trace.expression.channel == "rpc"
    assert isinstance(trace.tool_call, ToolCall)
    assert trace.tool_call.parameters == intent.parameters
    assert trace.tool_call.source_goal_id == "stay-informed"
    restored = StructuredToolCallCodec.decode(trace.encoded)
    assert restored == trace.tool_call
    world_action = restored.to_world_action()
    assert world_action.action_id == intent.intent_id
    assert world_action.kind == "weather.lookup.v1"
    assert dict(world_action.parameters) == dict(intent.parameters)


def test_adapter_owns_tool_generation_and_native_checkpoint() -> None:
    adapter = TSKV8Adapter()
    adapter.attach_generation_controller(GenerationController())
    adapter.attach_goal_planner(GoalPlanner())
    adapter.set_goals((Goal("stay-informed", "get current information", priority=1.0),))
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
        ),
        goal_id=None,
    )
    adapter.act(
        (10, 11),
        sample=False,
        procedural_action_kinds=("lookup_weather", "idle"),
        use_plan=True,
    )
    call = adapter.generate_tool_call(tool_name="weather.lookup.v1")
    assert call.tool_name == "weather.lookup.v1"
    assert call.parameters == {
        "action_symbol": call.parameters["action_symbol"],
        "available_actions": (10, 11),
    }
    assert adapter.generation_trace is not None

    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    assert restored.generation_trace == adapter.generation_trace
    assert restored.generate_tool_call(tool_name="weather.lookup.v1") == call
