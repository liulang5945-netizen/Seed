from __future__ import annotations

from taiji import (
    ActionIntent,
    EnvironmentOutcome,
    GenerationController,
    StructuredToolCallCodec,
    TSKV8Adapter,
    WorldAction,
)


class UnseenMapsToolEnvironment:
    def execute_tool(self, tool_name: str, parameters: dict[str, object]) -> EnvironmentOutcome:
        expected_query = {
            "regions": ["east", "south"],
            "filters": {"forecast": {"kind": "hourly", "probability": 0.8}},
        }
        valid = (
            tool_name == "maps.search.v42"
            and parameters.get("query") == expected_query
            and parameters.get("limits") == {"start": 2, "end": 4}
        )
        return EnvironmentOutcome(
            sensation=98 if valid else 99,
            reward=1.0 if valid else -1.0,
            success=valid,
        )


def test_unseen_tool_and_nested_parameters_are_not_table_mapped() -> None:
    nested = {
        "query": {
            "regions": ["east", "south"],
            "filters": {"forecast": {"kind": "hourly", "probability": 0.8}},
        },
        "limits": {"start": 2, "end": 4},
    }
    intent = ActionIntent(
        intent_id="episode:intent:unseen-tool",
        kind="maps_search",
        parameters=nested,
        confidence=0.9,
        tick=0,
    )
    trace = GenerationController().generate_tool_call(
        intent,
        tool_name="maps.search.v42",
        channel="rpc",
    )
    restored = StructuredToolCallCodec.decode(trace.encoded)
    assert restored.tool_name == "maps.search.v42"
    assert restored.parameters == nested
    assert restored.to_world_action().kind == "maps.search.v42"


def test_parameter_order_is_not_a_fixed_mapping_and_adapter_executes_it() -> None:
    first = ActionIntent(
        intent_id="episode:intent:order",
        kind="maps_search",
        parameters={"query": {"region": "east"}, "limits": {"end": 4, "start": 2}},
        confidence=0.9,
    )
    second = ActionIntent(
        intent_id="episode:intent:order",
        kind="maps_search",
        parameters={"limits": {"start": 2, "end": 4}, "query": {"region": "east"}},
        confidence=0.9,
    )
    controller = GenerationController()
    first_trace = controller.generate_tool_call(first, tool_name="maps.search.v42")
    second_trace = controller.generate_tool_call(second, tool_name="maps.search.v42")
    assert first_trace.encoded == second_trace.encoded

    adapter = TSKV8Adapter()
    adapter.attach_generation_controller(controller)
    adapter.observe(97, learn=False)
    world_action = WorldAction(
        action_id="pending",
        kind="maps_search",
        tick=adapter.tick,
        parameters={
            "query": {
                "regions": ["east", "south"],
                "filters": {"forecast": {"kind": "hourly", "probability": 0.8}},
            },
            "limits": {"start": 2, "end": 4},
        },
    )
    adapter.act(
        (10,),
        sample=False,
        procedural_action_kinds=("maps_search",),
        world_action=world_action,
    )
    call = adapter.generate_tool_call(tool_name="maps.search.v42")
    outcome = adapter.execute_tool_call(UnseenMapsToolEnvironment(), call=call, learn=False)
    assert outcome.success is True

