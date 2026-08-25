"""Measure a provider train/holdout boundary without training Taiji cognition."""

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
    QwenTextDecoder,
    _prompt,
)
from taiji import (  # noqa: E402
    ContentPlan,
    ExternalTextDecoderLanguageOrgan,
    GenerationController,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageTrainingCorpus,
    LanguageTrainingExample,
    TSKV8Adapter,
)

MANIFEST_FORMAT = "taiji-p6-language-train-holdout-manifest-v1"
REPORT_FORMAT = "taiji-p6-language-train-holdout-v1"

TRAIN_CASES = (
    {
        "case_id": "database-train",
        "intent_kind": "render_database_notice",
        "semantic_slots": {"系统": "数据库", "状态": "正常", "受众": "操作员"},
        "target_text": "数据库运行正常。",
        "required_terms": ("数据库",),
    },
    {
        "case_id": "interface-train",
        "intent_kind": "render_interface_notice",
        "semantic_slots": {"服务": "接口", "状态": "维护", "受众": "操作员"},
        "target_text": "接口进入维护。",
        "required_terms": ("接口",),
    },
    {
        "case_id": "cache-recovery-train",
        "intent_kind": "render_cache_recovery",
        "semantic_slots": {"服务": "缓存", "状态": "恢复", "受众": "操作员"},
        "target_text": "缓存已经恢复。",
        "required_terms": ("缓存", "恢复"),
    },
    {
        "case_id": "cache-warning-train",
        "intent_kind": "render_cache_warning",
        "semantic_slots": {"服务": "缓存", "级别": "警告", "受众": "操作员"},
        "target_text": "缓存出现警告。",
        "required_terms": ("缓存", "警告"),
    },
)

HOLDOUT_CASES = (
    {
        "case_id": "incident-holdout",
        "intent_kind": "render_incident_alert",
        "semantic_slots": {"级别": "警告", "系统": "数据库", "动作": "请检查"},
        "target_text": "数据库出现警告，请检查。",
        "required_terms": ("警告", "数据库"),
    },
    {
        "case_id": "recovery-holdout",
        "intent_kind": "render_recovery_notice",
        "semantic_slots": {"状态": "恢复", "服务": "接口", "受众": "操作员"},
        "target_text": "接口已经恢复。",
        "required_terms": ("恢复", "接口"),
    },
)


def _example(
    controller: GenerationController,
    case: dict[str, object],
    *,
    split: str,
) -> LanguageTrainingExample:
    content = ContentPlan(
        content_id=f"qwen:{split}:{case['case_id']}:content",
        intent_id=f"qwen:{split}:{case['case_id']}:intent",
        intent_kind=str(case["intent_kind"]),
        semantic_slots=dict(case["semantic_slots"]),
        required_terms=tuple(str(term) for term in case["required_terms"]),
        source_goal_id="qwen-language-train-holdout-goal",
        expected_outcome="operator receives a concise message",
        confidence=0.82,
        provenance=split,
        tick=0,
    )
    expression = controller.plan_expression(content, modality="text", channel="message")
    return LanguageTrainingExample(
        example_id=f"qwen:{split}:{case['case_id']}:example",
        expression=expression,
        target_text=str(case["target_text"]),
        split=split,
        provenance="human-reviewed",
    )


def _measure(
    adapter: TSKV8Adapter,
    examples: tuple[LanguageTrainingExample, ...],
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for example in examples:
        emission = adapter.emit_language(example.expression)
        output_text = emission.text_bytes.decode("utf-8")
        terms = example.expression.required_terms
        matched = tuple(term for term in terms if term in output_text)
        results.append(
            {
                "example_id": example.example_id,
                "output_text": output_text,
                "output_nonempty": bool(output_text),
                "required_terms": list(terms),
                "required_term_recall": len(matched) / max(1, len(terms)),
                "prompt_or_json_leak": any(
                    marker in output_text
                    for marker in ("semantic_slots", "intent_kind", "expected_outcome", "{", "}")
                ),
            }
        )
    return {
        "examples": results,
        "output_nonempty_rate": sum(bool(result["output_nonempty"]) for result in results)
        / len(results),
        "required_term_recall": sum(float(result["required_term_recall"]) for result in results)
        / len(results),
        "prompt_leakage_rate": sum(not bool(result["prompt_or_json_leak"]) for result in results)
        / len(results),
    }


def evaluate(model_dir: Path) -> dict[str, object]:
    decoder = QwenTextDecoder(model_dir)
    controller = GenerationController()
    corpus = LanguageTrainingCorpus(
        train=tuple(_example(controller, case, split="train") for case in TRAIN_CASES),
        holdout=tuple(_example(controller, case, split="holdout") for case in HOLDOUT_CASES),
    )
    registry = LanguageBackendRegistry.default()
    registry.register(
        LanguageBackendSpec(
            backend_id=BACKEND_ID,
            family="external-causal-decoder",
            training_contract="expression-to-text-v1",
        )
    )
    adapter = TSKV8Adapter()
    adapter.attach_language_backend_registry(registry)
    adapter.attach_language_organ(
        ExternalTextDecoderLanguageOrgan(
            decoder,
            prompt_builder=_prompt,
            backend_id=BACKEND_ID,
            max_tokens=24,
            temperature=0.0,
        )
    )
    train_metrics = _measure(adapter, corpus.train)
    holdout_metrics = _measure(adapter, corpus.holdout)
    adapter.attach_language_organ(None)
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    restored_corpus = LanguageTrainingCorpus.from_payload(corpus.to_payload())
    lesion_passed = False
    try:
        adapter.emit_language(corpus.holdout[0].expression)
    except RuntimeError:
        lesion_passed = True
    split_ids = {example.example_id for example in corpus.train}.isdisjoint(
        example.example_id for example in corpus.holdout
    )
    split_expression_ids = {
        example.expression.expression_id for example in corpus.train
    }.isdisjoint(example.expression.expression_id for example in corpus.holdout)
    boundary_passed = bool(
        restored_corpus == corpus
        and split_ids
        and split_expression_ids
        and lesion_passed
        and restored._language_backend_registry.get(BACKEND_ID).family
        == "external-causal-decoder"
    )
    raw_quality_passed = bool(
        holdout_metrics["output_nonempty_rate"] == 1.0
        and holdout_metrics["required_term_recall"] >= 0.67
        and holdout_metrics["prompt_leakage_rate"] == 1.0
    )
    return {
        "format": REPORT_FORMAT,
        "model": {
            "backend": BACKEND_ID,
            "model_dir_name": model_dir.name,
            "device": str(next(decoder.model.parameters()).device),
        },
        "training": {
            "training_applied": False,
            "reason": "this Gate validates the provider data boundary and baseline; no external trainer or weight update is hidden here",
            "train_examples": len(corpus.train),
            "holdout_examples": len(corpus.holdout),
            "corpus_round_trip": restored_corpus == corpus,
            "split_example_ids_disjoint": split_ids,
            "split_expression_ids_disjoint": split_expression_ids,
        },
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "gates": {
            "train_holdout_boundary": {
                "passed": boundary_passed,
                "criterion": "provider receives disjoint Taiji-owned train/holdout ExpressionPlan examples, registry/checkpoint round-trip is preserved, and the organ remains lesionable",
            },
            "raw_provider_quality": {
                "passed": raw_quality_passed,
                "criterion": "raw external provider reaches non-empty holdout text, required-term recall >= 0.67, and no structured prompt leakage",
            },
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "separate LanguageTrainingExample train/holdout data and measure a real external provider without leaking cognition",
        "lesions": ["split_leakage", "hidden_provider_training", "language_organ_detached"],
        "signals": ["corpus_round_trip", "split_disjoint", "train_metrics", "holdout_metrics", "raw_provider_quality"],
        "boundary": "provider data/quality baseline only; no claim that Qwen training has occurred or that the decoder owns Taiji cognition",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_language_train_holdout_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_language_train_holdout_baseline_20260825.json",
    )
    args = parser.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"local Qwen model directory not found: {args.model}")
    report = evaluate(args.model)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["gates"]["train_holdout_boundary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
