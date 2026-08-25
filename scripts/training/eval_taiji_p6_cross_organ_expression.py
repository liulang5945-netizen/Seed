"""Evaluate semantic consistency when one content plan feeds two organs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import ActionIntent, GenerationController  # noqa: E402

MANIFEST_FORMAT = "taiji-p6-cross-organ-expression-manifest-v1"
REPORT_FORMAT = "taiji-p6-cross-organ-expression-v1"


def evaluate() -> dict[str, object]:
    intent = ActionIntent(
        intent_id="episode:intent:cross-organ",
        kind="report_status",
        parameters={"topic": "weather", "detail": {"level": "summary"}},
        source_goal_id="stay-informed",
        expected_outcome="user receives status",
        confidence=0.85,
        tick=4,
    )
    controller = GenerationController()
    content = controller.plan_content(intent)
    tool_expression = controller.plan_expression(content, modality="tool", channel="rpc")
    text_expression = controller.plan_expression(content, modality="text", channel="message")
    same_content = bool(
        tool_expression.content_id == text_expression.content_id == content.content_id
        and tool_expression.fields == text_expression.fields
        and tool_expression.fields["semantic_slots"] == intent.parameters
        and tool_expression.confidence == text_expression.confidence == intent.confidence
    )
    gate_passed = bool(
        same_content
        and tool_expression.modality == "tool"
        and text_expression.modality == "text"
        and tool_expression.channel != text_expression.channel
        and content.source_goal_id == intent.source_goal_id
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "content_id": content.content_id,
            "tool_modality": tool_expression.modality,
            "text_modality": text_expression.modality,
            "tool_channel": tool_expression.channel,
            "text_channel": text_expression.channel,
            "semantic_slots_consistent": same_content,
            "goal_provenance": content.source_goal_id,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "one ContentPlan feeds tool and text expression organs without changing semantic slots, confidence, or goal provenance",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "render one content plan through tool and text expression organs",
        "lesions": ["independent_content_per_organ", "expression_overwrites_goal", "direct_byte_content"],
        "signals": ["content_id", "semantic_slots", "confidence", "goal_provenance", "modality"],
        "boundary": "structured cross-organ consistency Gate only; text modality is not a language fluency claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_cross_organ_expression_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_cross_organ_expression_baseline_20260825.json",
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

