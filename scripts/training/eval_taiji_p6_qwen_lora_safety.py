"""Evaluate a trained Qwen LoRA adapter at Taiji's safety boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_p6_qwen_provider import (  # noqa: E402
    BACKEND_ID,
    HOLDOUT_CASES,
    QwenTextDecoder,
    _prompt,
)
from scripts.training.eval_taiji_p6_qwen_realization_guard import (  # noqa: E402
    _expression,
)
from taiji import (  # noqa: E402
    ExternalTextDecoderLanguageOrgan,
    GenerationController,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageRealizationValidator,
    TSKV8Adapter,
    ValidatedLanguageOrgan,
)

MANIFEST_FORMAT = "taiji-p6-qwen-lora-safety-manifest-v1"
REPORT_FORMAT = "taiji-p6-qwen-lora-safety-v1"


def _raw_metrics(outputs: list[str], cases: tuple[dict[str, object], ...]) -> dict[str, object]:
    recalls = []
    for output, case in zip(outputs, cases, strict=True):
        terms = tuple(str(term) for term in case["required_terms"])
        recalls.append(sum(term in output for term in terms) / max(1, len(terms)))
    return {
        "output_nonempty_rate": sum(bool(output) for output in outputs) / len(outputs),
        "required_term_recall": sum(recalls) / len(recalls),
        "outputs": outputs,
    }


def evaluate(model_dir: Path, adapter_dir: Path) -> dict[str, object]:
    decoder = QwenTextDecoder(model_dir)
    controller = GenerationController()
    expressions = tuple(_expression(controller, case) for case in HOLDOUT_CASES)
    registry = LanguageBackendRegistry.default()
    registry.register(
        LanguageBackendSpec(
            backend_id=BACKEND_ID,
            family="external-causal-decoder-lora",
            training_contract="expression-to-text-v1",
        )
    )
    raw_organ = ExternalTextDecoderLanguageOrgan(
        decoder,
        prompt_builder=_prompt,
        backend_id=BACKEND_ID,
        max_tokens=24,
        temperature=0.0,
    )
    adapter = TSKV8Adapter()
    adapter.attach_language_backend_registry(registry)
    adapter.attach_language_organ(raw_organ)
    raw_outputs = [
        adapter.emit_language(expression).text_bytes.decode("utf-8") for expression in expressions
    ]
    adapted_model = PeftModel.from_pretrained(decoder.model, adapter_dir, is_trainable=False)
    adapted_model.eval()
    decoder.model = adapted_model
    guarded = ValidatedLanguageOrgan(
        ExternalTextDecoderLanguageOrgan(
            decoder,
            prompt_builder=_prompt,
            backend_id=BACKEND_ID,
            max_tokens=24,
            temperature=0.0,
        ),
        validator=LanguageRealizationValidator(minimum_coverage=1.0),
    )
    adapter.attach_language_organ(guarded)
    guarded_results: list[dict[str, object]] = []
    for expression, case in zip(expressions, HOLDOUT_CASES, strict=True):
        adapter.reset_dynamics()
        emission = adapter.emit_language(expression)
        validation = emission.validation
        guarded_results.append(
            {
                "case_id": case["case_id"],
                "output_text": emission.text_bytes.decode("utf-8"),
                "fallback_used": emission.fallback_used,
                "validation_accepted": bool(validation and validation.accepted),
                "coverage": 0.0 if validation is None else validation.coverage,
                "missing_terms": [] if validation is None else list(validation.missing_terms),
                "replan_required": adapter.replan_required,
            }
        )
    # Disable the adapter in-place: this is the provider's rollback path.
    adapted_model.disable_adapter_layers()
    rollback_outputs = [
        decoder.generate(_prompt(expression), max_tokens=24, temperature=0.0)
        for expression in expressions
    ]
    rollback_matches_raw = rollback_outputs == raw_outputs
    adapter.attach_language_organ(None)
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    fallback_count = sum(bool(result["fallback_used"]) for result in guarded_results)
    safe_rate = sum(
        bool(result["validation_accepted"]) or bool(result["fallback_used"])
        for result in guarded_results
    ) / len(guarded_results)
    gate_passed = bool(
        safe_rate == 1.0
        and all(
            bool(result["replan_required"]) for result in guarded_results if result["fallback_used"]
        )
        and all(
            not bool(result["replan_required"])
            for result in guarded_results
            if not result["fallback_used"]
        )
        and rollback_matches_raw
        and restored.cognitive_snapshot().action_intent is None
    )
    return {
        "format": REPORT_FORMAT,
        "model": {
            "backend": BACKEND_ID,
            "model_dir_name": model_dir.name,
            "adapter_dir_name": adapter_dir.name,
            "device": str(next(decoder.model.parameters()).device),
        },
        "raw": _raw_metrics(raw_outputs, HOLDOUT_CASES),
        "adapted": {
            "cases": guarded_results,
            "safe_realization_rate": safe_rate,
            "fallback_count": fallback_count,
            "outputs": [str(result["output_text"]) for result in guarded_results],
        },
        "rollback": {
            "outputs_match_raw": rollback_matches_raw,
            "outputs": rollback_outputs,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "trained external LoRA is guarded by Taiji semantic validation/fallback/replan, rollback reproduces raw output, and cognition remains unchanged",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "load a trained Qwen LoRA provider into Taiji's validator/fallback/replan boundary",
        "lesions": [
            "raw_provider_quality",
            "semantic_validator",
            "adapter_rollback",
            "cognition_dependency",
        ],
        "signals": [
            "raw_required_term_recall",
            "safe_realization_rate",
            "fallback_count",
            "replan_required",
            "rollback_outputs_match_raw",
        ],
        "boundary": "integration safety Gate only; passing does not make the external decoder a Taiji cognition owner",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_qwen_lora_safety_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_qwen_lora_safety_baseline_20260825.json",
    )
    args = parser.parse_args()
    if not args.model.is_dir() or not args.adapter_dir.is_dir():
        raise SystemExit("model and adapter directories must exist")
    report = evaluate(args.model, args.adapter_dir)
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
