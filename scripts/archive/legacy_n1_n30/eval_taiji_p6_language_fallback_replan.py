"""Evaluate language fallback consumption by Taiji content replanning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    ContentCandidate,
    ContentSelector,
    ExternalTextDecoderLanguageOrgan,
    GenerationController,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    TSKV8Adapter,
    ValidatedLanguageOrgan,
)

MANIFEST_FORMAT = "taiji-p6-language-fallback-replan-manifest-v1"
REPORT_FORMAT = "taiji-p6-language-fallback-replan-v1"
BACKEND_ID = "mature-decoder-v1"


class RecoveryTextDecoder:
    """A deterministic external effector used only for the replan Gate."""

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        del max_tokens, temperature
        return "当前状态稳定。" if prompt == "recovery" else "操作员收到一条消息。"


def _candidates() -> tuple[ContentCandidate, ContentCandidate]:
    return (
        ContentCandidate(
            candidate_id="status",
            intent_id="language:intent",
            intent_kind="render_message",
            semantic_slots={"topic": "status"},
            required_terms=("稳定",),
            confidence=0.8,
        ),
        ContentCandidate(
            candidate_id="recovery",
            intent_id="language:intent",
            intent_kind="render_message",
            semantic_slots={"topic": "recovery"},
            required_terms=("稳定",),
            confidence=0.8,
        ),
    )


def evaluate() -> dict[str, object]:
    registry = LanguageBackendRegistry.default()
    registry.register(
        LanguageBackendSpec(
            backend_id=BACKEND_ID,
            family="external-decoder",
            training_contract="expression-to-text-v1",
        )
    )
    adapter = TSKV8Adapter()
    adapter.attach_generation_controller(GenerationController())
    adapter.attach_content_selector(ContentSelector())
    adapter.attach_language_backend_registry(registry)
    adapter.attach_language_organ(
        ValidatedLanguageOrgan(
            ExternalTextDecoderLanguageOrgan(
                RecoveryTextDecoder(),
                prompt_builder=lambda expression: expression.fields["semantic_slots"]["topic"],
                backend_id=BACKEND_ID,
            )
        )
    )
    adapter.observe(97, learn=False)
    candidates = _candidates()
    first = adapter.select_content(candidates)
    first_emission = adapter.emit_language()
    first_error = adapter.last_content_prediction_error
    replanned = adapter.replan_content_after_language_fallback(candidates)
    second_expression = adapter.express_selected_content(modality="text", channel="message")
    second_emission = adapter.emit_language(second_expression)
    adapter.attach_language_organ(None)
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    gate_passed = bool(
        first.selected.candidate_id == "status"
        and first_emission.fallback_used
        and first_error is not None
        and first_error > 0.0
        and replanned.selected.candidate_id == "recovery"
        and second_expression.fields["semantic_slots"] == {"topic": "recovery"}
        and not second_emission.fallback_used
        and adapter.language_fallback_count == 1
        and not adapter.replan_required
        and restored.last_content_selection == replanned
        and restored.language_fallback_count == 1
        and restored.replan_required is False
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "first_content": first.selected.candidate_id,
            "first_fallback": first_emission.fallback_used,
            "first_content_prediction_error": first_error,
            "replacement_content": replanned.selected.candidate_id,
            "replacement_expression_slots": dict(second_expression.fields["semantic_slots"]),
            "replacement_fallback": second_emission.fallback_used,
            "language_fallback_count": adapter.language_fallback_count,
            "final_replan_required": adapter.replan_required,
            "checkpoint_replacement_content": restored.last_content_selection.selected.candidate_id,
            "checkpoint_fallback_count": restored.language_fallback_count,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "unsafe language realization is credited as a content failure, an alternative ContentPlan is selected and expressed safely, and checkpoint state remains consistent",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "consume a language realization fallback through alternative Taiji content replanning",
        "lesions": [
            "no_fallback_feedback",
            "no_alternative_content_replan",
            "checkpoint_feedback_loss",
        ],
        "signals": [
            "first_fallback",
            "content_prediction_error",
            "replacement_content",
            "final_replan_required",
        ],
        "boundary": "content/expression replan Gate only; no claim about open-domain language quality",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_p6_language_fallback_replan_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_p6_language_fallback_replan_baseline_20260825.json",
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
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
