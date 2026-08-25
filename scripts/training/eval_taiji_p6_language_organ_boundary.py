"""Evaluate the replaceable terminal language-organ boundary."""

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
    StructuredTextLanguageOrgan,
    TextExpressionCodec,
    TSKV8Adapter,
)

MANIFEST_FORMAT = "taiji-p6-language-organ-boundary-manifest-v1"
REPORT_FORMAT = "taiji-p6-language-organ-boundary-v1"


def evaluate() -> dict[str, object]:
    controller = GenerationController()
    adapter = TSKV8Adapter()
    expression = controller.plan_expression(
        ContentPlan(
            content_id="language:intent:content",
            intent_id="language:intent",
            intent_kind="render_message",
            semantic_slots={"topic": "status", "format": "concise"},
            source_goal_id="language-goal",
            expected_outcome="user receives a message",
            confidence=0.8,
            provenance="selected",
            tick=0,
        ),
        modality="text",
        channel="message",
    )
    adapter.attach_generation_controller(controller)
    adapter.attach_language_organ(StructuredTextLanguageOrgan())
    emission = adapter.emit_language(expression)
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    round_trip = TextExpressionCodec.decode(emission.text_bytes) == expression
    checkpoint_round_trip = restored.last_language_emission == emission
    cognition_unchanged = adapter.cognitive_snapshot().action_intent is None
    gate_passed = bool(round_trip and checkpoint_round_trip and cognition_unchanged)
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "backend": emission.backend,
            "encoded_bytes": len(emission.text_bytes),
            "expression_round_trip": round_trip,
            "native_checkpoint_round_trip": checkpoint_round_trip,
            "cognition_unchanged": cognition_unchanged,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "a replaceable terminal language organ emits Taiji-owned text expressions and restores without owning cognition",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "verify the terminal language-organ interface and lesion boundary",
        "lesions": ["language_organ_detached", "direct_byte_content", "cognitive_state_mutation"],
        "signals": ["backend", "expression_round_trip", "native_checkpoint_round_trip", "cognition_unchanged"],
        "boundary": "structured stub proves interface ownership only; no natural-language fluency or decoder capability claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_language_organ_boundary_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_language_organ_boundary_baseline_20260825.json",
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
