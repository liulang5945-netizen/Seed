"""Evaluate Taiji content, expression, and structured tool-call generation."""

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
    GenerationController,
    StructuredToolCallCodec,
)

MANIFEST_FORMAT = "taiji-p6-generation-manifest-v1"
REPORT_FORMAT = "taiji-p6-generation-v1"


def evaluate() -> dict[str, object]:
    intent = ActionIntent(
        intent_id="episode:intent:0",
        kind="lookup_weather",
        parameters={"city": "Shanghai", "days": 3},
        source_goal_id="stay-informed",
        expected_outcome="weather result",
        confidence=0.8,
        tick=0,
    )
    trace = GenerationController().generate_tool_call(
        intent,
        tool_name="weather.lookup.v1",
        channel="rpc",
    )
    decoded = StructuredToolCallCodec.decode(trace.encoded)
    gate_passed = bool(
        trace.content.intent_id == intent.intent_id
        and trace.content.semantic_slots == intent.parameters
        and trace.expression.modality == "tool"
        and trace.expression.channel == "rpc"
        and decoded == trace.tool_call
        and decoded.parameters == intent.parameters
        and decoded.to_world_action().action_id == intent.intent_id
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "content_intent_kind": trace.content.intent_kind,
            "expression_modality": trace.expression.modality,
            "expression_channel": trace.expression.channel,
            "tool_name": trace.tool_call.tool_name,
            "parameter_keys": sorted(trace.tool_call.parameters),
            "encoded_bytes": len(trace.encoded),
            "codec_round_trip": decoded == trace.tool_call,
            "world_action_kind": decoded.to_world_action().kind,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "content and expression preserve intent semantics and structured tool bytes round-trip into a world action",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "preserve one ActionIntent through content, expression, tool-call, and UTF-8 codec stages",
        "lesions": ["direct_byte_output", "content_plan", "expression_plan", "codec_round_trip"],
        "signals": ["intent_kind", "semantic_slots", "tool_name", "world_action"],
        "boundary": "structured tool generation Gate only; no language-model training or general language fluency claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_generation_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_generation_baseline_20260825.json",
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

