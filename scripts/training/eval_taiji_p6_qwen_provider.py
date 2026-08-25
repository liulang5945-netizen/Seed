"""Run a real local Qwen decoder through Taiji's external language-organ boundary.

The model and tokenizer remain external runtime assets.  This script imports
``transformers`` only at the integration edge; ``taiji`` itself stays free of
Legacy/Transformer imports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    ContentPlan,
    ExternalTextDecoderLanguageOrgan,
    GenerationController,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageTrainingExample,
    TSKV8Adapter,
)

MANIFEST_FORMAT = "taiji-p6-qwen-provider-manifest-v1"
REPORT_FORMAT = "taiji-p6-qwen-provider-v1"
BACKEND_ID = "qwen2.5-0.5b-instruct"

HOLDOUT_CASES = (
    {
        "case_id": "status",
        "intent_kind": "render_status_digest",
        "semantic_slots": {"状态": "稳定", "受众": "操作员", "格式": "一句话"},
        "target_text": "当前状态稳定。",
        "required_terms": ("稳定",),
    },
    {
        "case_id": "maintenance",
        "intent_kind": "render_maintenance_notice",
        "semantic_slots": {"事件": "维护窗口", "时间": "今晚十点", "受众": "操作员"},
        "target_text": "今晚十点有维护窗口。",
        "required_terms": ("维护窗口", "今晚十点"),
    },
    {
        "case_id": "incident",
        "intent_kind": "render_incident_alert",
        "semantic_slots": {"级别": "警告", "系统": "数据库", "动作": "请检查"},
        "target_text": "数据库出现警告，请检查。",
        "required_terms": ("警告", "数据库"),
    },
)


class QwenTextDecoder:
    """Small adapter around a locally cached Hugging Face causal decoder."""

    def __init__(self, model_dir: Path) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen provider requires transformers; install the optional integration dependencies"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.float32,
        )
        self.model.eval()

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        del temperature
        formatted_prompt = prompt
        if getattr(self.tokenizer, "chat_template", None):
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = self.tokenizer(formatted_prompt, return_tensors="pt")
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        new_tokens = generated[0, prompt_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _prompt(expression) -> str:
    payload = {
        "intent_kind": expression.fields.get("intent_kind", ""),
        "channel": expression.channel,
        "semantic_slots": expression.fields.get("semantic_slots", {}),
        "expected_outcome": expression.fields.get("expected_outcome", ""),
    }
    return (
        "请把下面的结构化表达实现为简洁、自然的中文消息。"
        "必须保留语义槽位中的全部关键值，不要添加输入中没有的事实。"
        "不要输出字段名或 JSON，只输出最终消息：\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


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
    organ = ExternalTextDecoderLanguageOrgan(
        decoder,
        prompt_builder=_prompt,
        backend_id=BACKEND_ID,
        max_tokens=24,
        temperature=0.0,
    )
    adapter = TSKV8Adapter()
    adapter.attach_language_backend_registry(registry)
    adapter.attach_language_organ(organ)
    action_before = adapter.cognitive_snapshot().action_intent
    holdout_results: list[dict[str, object]] = []
    training_examples: list[LanguageTrainingExample] = []
    for case in HOLDOUT_CASES:
        content = ContentPlan(
            content_id=f"qwen:holdout:{case['case_id']}:content",
            intent_id=f"qwen:holdout:{case['case_id']}:intent",
            intent_kind=str(case["intent_kind"]),
            semantic_slots=dict(case["semantic_slots"]),
            source_goal_id="qwen-language-holdout-goal",
            expected_outcome="operator receives a concise message",
            confidence=0.82,
            provenance="holdout",
            tick=0,
        )
        expression = controller.plan_expression(content, modality="text", channel="message")
        training_example = LanguageTrainingExample(
            example_id=f"qwen:holdout:{case['case_id']}:example",
            expression=expression,
            target_text=str(case["target_text"]),
            split="holdout",
            provenance="human-reviewed",
        )
        training_examples.append(training_example)
        output_text = adapter.emit_language(expression).text_bytes.decode("utf-8")
        required_terms = tuple(str(term) for term in case["required_terms"])
        term_hits = sum(term in output_text for term in required_terms)
        holdout_results.append(
            {
                "case_id": case["case_id"],
                "expression_id": expression.expression_id,
                "target_text": training_example.target_text,
                "output_text": output_text,
                "output_nonempty": bool(output_text),
                "required_terms": list(required_terms),
                "required_term_recall": term_hits / max(1, len(required_terms)),
                "prompt_or_json_leak": any(
                    marker in output_text for marker in ("semantic_slots", "intent_kind", "expected_outcome", "{", "}")
                ),
            }
        )
    adapter.attach_language_organ(None)
    lesion_passed = False
    try:
        adapter.emit_language(expression)
    except RuntimeError:
        lesion_passed = True
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    cognition_unchanged = action_before is None and adapter.cognitive_snapshot().action_intent is None
    contract_round_trip = all(
        LanguageTrainingExample.from_payload(example.to_payload()) == example
        for example in training_examples
    )
    registry_round_trip = restored._language_backend_registry.get(BACKEND_ID).family == (
        "external-causal-decoder"
    )
    output_nonempty_rate = sum(result["output_nonempty"] for result in holdout_results) / len(holdout_results)
    required_term_recall = sum(result["required_term_recall"] for result in holdout_results) / len(holdout_results)
    prompt_leakage_rate = sum(not result["prompt_or_json_leak"] for result in holdout_results) / len(holdout_results)
    gate_passed = bool(
        output_nonempty_rate == 1.0
        and required_term_recall >= 0.67
        and prompt_leakage_rate == 1.0
        and lesion_passed
        and cognition_unchanged
        and contract_round_trip
        and registry_round_trip
    )
    return {
        "format": REPORT_FORMAT,
        "model": {
            "backend": BACKEND_ID,
            "model_dir_name": model_dir.name,
            "device": str(next(decoder.model.parameters()).device),
        },
        "metrics": {
            "holdout_cases": len(holdout_results),
            "output_nonempty_rate": output_nonempty_rate,
            "required_term_recall": required_term_recall,
            "prompt_leakage_rate": prompt_leakage_rate,
            "organ_lesion": lesion_passed,
            "cognition_unchanged": cognition_unchanged,
            "training_contract_round_trip": contract_round_trip,
            "registry_checkpoint_round_trip": registry_round_trip,
        },
        "samples": holdout_results,
        "gate": {
            "passed": gate_passed,
            "criterion": "a real local decoder maps varied holdout ExpressionPlan values to non-empty text with minimum semantic-term coverage and no structured-prompt leakage, while remaining lesionable",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "run a local Qwen causal decoder as an external Taiji language organ",
        "required_assets": ["local checkpoint directory", "local tokenizer files", "transformers", "safetensors"],
        "lesions": ["language_organ_detached", "cognitive_state_mutation", "registry_state_loss"],
        "signals": ["output_nonempty_rate", "required_term_recall", "prompt_leakage_rate", "organ_lesion", "cognition_unchanged", "training_contract_round_trip", "registry_checkpoint_round_trip"],
        "boundary": "small holdout realization Gate only; no fluency, factuality, or general intelligence claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_qwen_provider_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_qwen_provider_baseline_20260825.json",
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
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
