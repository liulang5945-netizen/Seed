"""Evaluate unseen structured tool names, nested parameters, and key order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    ActionIntent,
    EnvironmentOutcome,
    GenerationController,
    StructuredToolCallCodec,
    TSKV8Adapter,
    WorldAction,
)

MANIFEST_FORMAT = "taiji-p6-unseen-tool-transfer-manifest-v1"
REPORT_FORMAT = "taiji-p6-unseen-tool-transfer-v1"


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


def evaluate() -> dict[str, object]:
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
    )
    controller = GenerationController()
    trace = controller.generate_tool_call(intent, tool_name="maps.search.v42", channel="rpc")
    restored = StructuredToolCallCodec.decode(trace.encoded)
    reordered = ActionIntent(
        intent_id=intent.intent_id,
        kind=intent.kind,
        parameters={"limits": {"end": 4, "start": 2}, "query": nested["query"]},
        confidence=intent.confidence,
    )
    reordered_trace = controller.generate_tool_call(
        reordered,
        tool_name="maps.search.v42",
        channel="rpc",
    )

    adapter = TSKV8Adapter()
    adapter.attach_generation_controller(controller)
    adapter.observe(97, learn=False)
    adapter.act(
        (10,),
        sample=False,
        procedural_action_kinds=("maps_search",),
        world_action=WorldAction(
            action_id="pending",
            kind="maps_search",
            tick=adapter.tick,
            parameters=nested,
        ),
    )
    adapter_call = adapter.generate_tool_call(tool_name="maps.search.v42")
    outcome = adapter.execute_tool_call(UnseenMapsToolEnvironment(), call=adapter_call, learn=False)
    gate_passed = bool(
        restored.parameters == nested
        and trace.encoded == reordered_trace.encoded
        and adapter_call.parameters["query"] == nested["query"]
        and outcome.success is True
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "unseen_tool_name": trace.tool_call.tool_name,
            "nested_parameter_round_trip": restored.parameters == nested,
            "order_invariant_codec": trace.encoded == reordered_trace.encoded,
            "adapter_parameter_preservation": adapter_call.parameters["query"] == nested["query"],
            "execution_success": outcome.success,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "unseen tool name, nested parameters, and reordered keys survive generation/codec/execution without a fixed mapping",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "execute an unseen structured tool with nested parameters and reordered keys",
        "lesions": ["fixed_tool_table", "flat_parameter_only", "order_sensitive_codec"],
        "signals": ["tool_name", "nested_parameters", "key_order", "execution_success"],
        "boundary": "unseen-tool transfer Gate only; no broad tool ecosystem or language generalization claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_unseen_tool_transfer_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_unseen_tool_transfer_baseline_20260825.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

