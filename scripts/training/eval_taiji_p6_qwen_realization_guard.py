"""Run the Taiji-owned realization validator around a real local Qwen provider."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_p6_qwen_provider import (  # noqa: E402
    BACKEND_ID,
    HOLDOUT_CASES,
    QwenTextDecoder,
    _prompt,
)
from taiji import (  # noqa: E402
    ContentPlan,
    ExternalTextDecoderLanguageOrgan,
    GenerationController,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageRealizationValidator,
    TextExpressionCodec,
    TSKV8Adapter,
    ValidatedLanguageOrgan,
)

MANIFEST_FORMAT = "taiji-p6-qwen-realization-guard-manifest-v1"
REPORT_FORMAT = "taiji-p6-qwen-realization-guard-v1"


def _expression(controller: GenerationController, case: dict[str, object]):
    content = ContentPlan(
        content_id=f"qwen:holdout:{case['case_id']}:content",
        intent_id=f"qwen:holdout:{case['case_id']}:intent",
        intent_kind=str(case["intent_kind"]),
        semantic_slots=dict(case["semantic_slots"]),
        required_terms=tuple(str(term) for term in case["required_terms"]),
        source_goal_id="qwen-language-holdout-goal",
        expected_outcome="operator receives a concise message",
        confidence=0.82,
        provenance="holdout",
        tick=0,
    )
    return controller.plan_expression(content, modality="text", channel="message")


def evaluate(model_dir: Path) -> dict[str, object]:
    decoder = QwenTextDecoder(model_dir)
    controller = GenerationController()
    registry = LanguageBackendRegistry.default()
    registry.register(
        LanguageBackendSpec(
            backend_id=BACKEND_ID,
            family="external-causal-decoder",
            training_contract="expression-to-text-v1",
        )
    )
    primary = ExternalTextDecoderLanguageOrgan(
        decoder,
        prompt_builder=_prompt,
        backend_id=BACKEND_ID,
        max_tokens=24,
        temperature=0.0,
    )
    guarded = ValidatedLanguageOrgan(
        primary,
        validator=LanguageRealizationValidator(minimum_coverage=1.0),
    )
    adapter = TSKV8Adapter()
    adapter.attach_language_backend_registry(registry)
    adapter.attach_language_organ(guarded)
    action_before = adapter.cognitive_snapshot().action_intent
    results: list[dict[str, object]] = []
    for case in HOLDOUT_CASES:
        expression = _expression(controller, case)
        emission = adapter.emit_language(expression)
        validation = emission.validation
        fallback_semantics_preserved = False
        if emission.fallback_used:
            fallback_semantics_preserved = (
                TextExpressionCodec.decode(emission.text_bytes) == expression
            )
        results.append(
            {
                "case_id": case["case_id"],
                "fallback_used": emission.fallback_used,
                "validation_accepted": bool(validation and validation.accepted),
                "coverage": 0.0 if validation is None else validation.coverage,
                "reason": None if validation is None else validation.reason,
                "missing_terms": [] if validation is None else list(validation.missing_terms),
                "fallback_semantics_preserved": fallback_semantics_preserved,
            }
        )
    adapter.attach_language_organ(None)
    lesion_passed = False
    try:
        adapter.emit_language(_expression(controller, HOLDOUT_CASES[0]))
    except RuntimeError:
        lesion_passed = True
    cognition_unchanged = (
        action_before is None and adapter.cognitive_snapshot().action_intent is None
    )
    all_safe = all(
        bool(result["validation_accepted"]) or bool(result["fallback_semantics_preserved"])
        for result in results
    )
    fallback_count = sum(bool(result["fallback_used"]) for result in results)
    gate_passed = bool(all_safe and lesion_passed and cognition_unchanged)
    return {
        "format": REPORT_FORMAT,
        "model": {"backend": BACKEND_ID, "model_dir_name": model_dir.name, "device": "cpu"},
        "metrics": {
            "holdout_cases": len(results),
            "safe_realization_rate": sum(
                bool(result["validation_accepted"]) or bool(result["fallback_semantics_preserved"])
                for result in results
            )
            / len(results),
            "fallback_count": fallback_count,
            "organ_lesion": lesion_passed,
            "cognition_unchanged": cognition_unchanged,
        },
        "cases": results,
        "gate": {
            "passed": gate_passed,
            "criterion": "every real decoder output is either semantically accepted or replaced by a lossless structured fallback, and the organ remains lesionable",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "guard real Qwen language realization with Taiji-owned semantic validation and structured fallback",
        "lesions": [
            "language_organ_detached",
            "missing_required_terms",
            "structured_fallback_semantic_loss",
        ],
        "signals": [
            "safe_realization_rate",
            "fallback_count",
            "organ_lesion",
            "cognition_unchanged",
        ],
        "boundary": "safety/fallback Gate only; fallback is structured output, not a fluency or intelligence claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_qwen_realization_guard_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_qwen_realization_guard_baseline_20260825.json",
    )
    args = parser.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"local Qwen model directory not found: {args.model}")
    report = evaluate(args.model)
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
