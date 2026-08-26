"""Evaluate structured text-organ codec semantic fidelity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    ContentPlan,
    GenerationController,
    TextExpressionCodec,
)

MANIFEST_FORMAT = "taiji-p6-text-organ-codec-manifest-v1"
REPORT_FORMAT = "taiji-p6-text-organ-codec-v1"


def evaluate() -> dict[str, object]:
    content = ContentPlan(
        content_id="holdout:intent:content",
        intent_id="holdout:intent",
        intent_kind="forecast_digest",
        semantic_slots={"format": "digest", "regions": ["east", "south"]},
        source_goal_id="stay-informed",
        expected_outcome="user receives forecast",
        confidence=0.68,
        provenance="selected",
        tick=4,
    )
    expression = GenerationController().plan_expression(
        content,
        modality="text",
        channel="message",
    )
    encoded = TextExpressionCodec.encode(expression)
    restored = TextExpressionCodec.decode(encoded)
    gate_passed = bool(
        restored == expression
        and restored.fields["semantic_slots"] == content.semantic_slots
        and restored.confidence == content.confidence
        and restored.source_goal_id == content.source_goal_id
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "encoded_bytes": len(encoded),
            "modality": restored.modality,
            "channel": restored.channel,
            "semantic_slots_round_trip": restored.fields["semantic_slots"]
            == content.semantic_slots,
            "confidence_round_trip": restored.confidence == content.confidence,
            "goal_provenance_round_trip": restored.source_goal_id == content.source_goal_id,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "structured text expression preserves semantic slots, confidence, and goal provenance through UTF-8 codec",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "round-trip a selected ContentPlan through the structured text organ codec",
        "lesions": ["direct_byte_content", "goal_provenance_drop", "semantic_slot_rewrite"],
        "signals": ["modality", "channel", "semantic_slots", "confidence", "source_goal_id"],
        "boundary": "structured text-organ codec Gate only; no fluency, syntax, or language-model capability claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_text_organ_codec_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_text_organ_codec_baseline_20260825.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
