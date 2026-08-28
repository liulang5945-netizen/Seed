"""Train and rollback a small Qwen LoRA language-organ adapter.

This script is deliberately an integration-edge trainer.  It updates only an
external decoder adapter from Taiji-owned ``LanguageTrainingCorpus`` examples;
it never enters the Taiji cognitive checkpoint or changes an ``ActionIntent``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_p6_language_train_holdout import (  # noqa: E402
    HOLDOUT_CASES,
    TRAIN_CASES,
    _example,
    _measure,
)
from scripts.training.eval_taiji_p6_qwen_provider import (  # noqa: E402
    BACKEND_ID,
    _prompt,
)
from taiji import (  # noqa: E402
    ExternalTextDecoderLanguageOrgan,
    GenerationController,
    LanguageBackendRegistry,
    LanguageBackendSpec,
    LanguageRealizationGate,
    LanguageTrainingCorpus,
    TSKV8Adapter,
)

MANIFEST_FORMAT = "taiji-p6-qwen-lora-provider-manifest-v1"
REPORT_FORMAT = "taiji-p6-qwen-lora-provider-v1"
LORA_RANK = 4
LORA_ALPHA = 8
LORA_TARGET_MODULES = ("q_proj", "v_proj")
DEFAULT_EPOCHS = 4
DEFAULT_LEARNING_RATE = 5e-4


class QwenLoRATextDecoder:
    """Minimal decoder wrapper shared by raw and adapted provider paths."""

    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def _formatted_prompt(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        del temperature
        formatted_prompt = self._formatted_prompt(prompt)
        encoded = self.tokenizer(formatted_prompt, return_tensors="pt")
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        return self.tokenizer.decode(
            generated[0, prompt_length:],
            skip_special_tokens=True,
        ).strip()


def _training_batch(tokenizer, example) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = _prompt(example.expression)
    prompt_text = QwenLoRATextDecoder(tokenizer, None)._formatted_prompt(prompt)
    if getattr(tokenizer, "chat_template", None):
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": example.target_text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        full_text = f"{prompt}\n{example.target_text}"
    encoded = tokenizer(full_text, return_tensors="pt")
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
    labels = encoded["input_ids"].clone()
    labels[:, :prompt_tokens] = -100
    return encoded["input_ids"], labels


def _train_lora(
    model, tokenizer, corpus: LanguageTrainingCorpus, *, epochs: int, learning_rate: float
) -> dict[str, object]:
    config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )
    adapted = get_peft_model(model, config)
    adapted.config.use_cache = False
    adapted.train()
    trainable = tuple(parameter for parameter in adapted.parameters() if parameter.requires_grad)
    if not trainable:
        raise RuntimeError("Qwen LoRA trainer found no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate))
    losses: list[float] = []
    for _ in range(int(epochs)):
        for example in corpus.train:
            input_ids, labels = _training_batch(tokenizer, example)
            output = adapted(input_ids=input_ids, labels=labels)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError("Qwen LoRA trainer produced a non-finite loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    adapted.eval()
    return {
        "model": adapted,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "optimizer_steps": len(losses),
        "losses": losses,
    }


def evaluate(
    model_dir: Path,
    *,
    output_dir: Path,
    epochs: int = 1,
    learning_rate: float = 1e-4,
) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float32,
    )
    model.eval()
    controller = GenerationController()
    corpus = LanguageTrainingCorpus(
        train=tuple(_example(controller, case, split="train") for case in TRAIN_CASES),
        holdout=tuple(_example(controller, case, split="holdout") for case in HOLDOUT_CASES),
    )
    registry = LanguageBackendRegistry.default()
    registry.register(
        LanguageBackendSpec(
            backend_id=BACKEND_ID,
            family="external-causal-decoder-lora",
            training_contract="expression-to-text-v1",
        )
    )
    decoder = QwenLoRATextDecoder(tokenizer, model)
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
    base_train = _measure(adapter, corpus.train)
    base_holdout = _measure(adapter, corpus.holdout)
    training = _train_lora(model, tokenizer, corpus, epochs=epochs, learning_rate=learning_rate)
    adapted_model = training["model"]
    decoder.model = adapted_model
    adapted_train = _measure(adapter, corpus.train)
    adapted_holdout = _measure(adapter, corpus.holdout)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapted_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    checkpoint_model = PeftModel.from_pretrained(
        model,
        output_dir,
        is_trainable=False,
    )
    checkpoint_model.eval()
    checkpoint_decoder = QwenLoRATextDecoder(tokenizer, checkpoint_model)
    rollback_decoder = QwenLoRATextDecoder(tokenizer, model)
    rollback_organ = ExternalTextDecoderLanguageOrgan(
        rollback_decoder,
        prompt_builder=_prompt,
        backend_id=BACKEND_ID,
        max_tokens=24,
        temperature=0.0,
    )
    rollback_reference_organ = ExternalTextDecoderLanguageOrgan(
        QwenLoRATextDecoder(tokenizer, model),
        prompt_builder=_prompt,
        backend_id=BACKEND_ID,
        max_tokens=24,
        temperature=0.0,
    )
    realization_gate = LanguageRealizationGate(
        minimum_required_term_coverage=1.0,
        minimum_readable_rate=1.0,
        maximum_fallback_rate=0.0,
    ).evaluate(
        organ,
        corpus,
        rollback_organ=rollback_organ,
        rollback_reference_organ=rollback_reference_organ,
        checkpoint_loader=lambda payload: ExternalTextDecoderLanguageOrgan(
            checkpoint_decoder,
            prompt_builder=_prompt,
            backend_id=str(payload["backend"]),
            max_tokens=int(payload.get("max_tokens", 24)),
            temperature=float(payload.get("temperature", 0.0)),
        ),
    )
    adapted_outputs = [str(example["output_text"]) for example in adapted_holdout["examples"]]
    adapted_model.disable_adapter_layers()
    rollback_holdout = _measure(adapter, corpus.holdout)
    rollback_outputs = [str(example["output_text"]) for example in rollback_holdout["examples"]]
    rollback_outputs_match_base = rollback_outputs == [
        str(example["output_text"]) for example in base_holdout["examples"]
    ]
    adapter.attach_language_organ(None)
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    holdout_improved = float(adapted_holdout["required_term_recall"]) > float(
        base_holdout["required_term_recall"]
    )
    gate_passed = bool(
        training["trainable_parameter_count"] > 0
        and training["optimizer_steps"] == len(corpus.train) * int(epochs)
        and holdout_improved
        and float(adapted_holdout["prompt_leakage_rate"]) == 1.0
        and float(adapted_holdout["output_nonempty_rate"]) == 1.0
        and rollback_outputs_match_base
        and bool(realization_gate["gate"]["passed"])
        and restored.cognitive_snapshot().action_intent is None
    )
    return {
        "format": REPORT_FORMAT,
        "model": {
            "backend": BACKEND_ID,
            "model_dir_name": model_dir.name,
            "device": str(next(model.parameters()).device),
            "adapter_dir": str(output_dir),
        },
        "training": {
            "training_applied": True,
            "method": "peft-lora",
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "trainable_parameter_count": training["trainable_parameter_count"],
            "optimizer_steps": training["optimizer_steps"],
            "losses": training["losses"],
            "base_checkpoint_untouched": True,
            "checkpoint_continuation": bool(realization_gate["checkpoint"]["outputs_match"]),
        },
        "base_metrics": {"train": base_train, "holdout": base_holdout},
        "adapted_metrics": {"train": adapted_train, "holdout": adapted_holdout},
        "rollback": {
            "holdout": rollback_holdout,
            "outputs_match_base": rollback_outputs_match_base,
            "adapted_outputs": adapted_outputs,
        },
        "expression_to_text_gate": realization_gate,
        "gate": {
            "passed": gate_passed,
            "criterion": "a real LoRA adapter updates only external provider parameters, improves the same holdout required-term recall without leakage, and disabling the adapter reproduces the raw provider output",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "train a rollbackable Qwen LoRA language-organ adapter from Taiji-owned train examples and evaluate the unchanged holdout",
        "lesions": ["no_lora_update", "adapter_disabled_rollback", "taiji_cognition_dependency"],
        "signals": [
            "trainable_parameter_count",
            "holdout_required_term_recall",
            "prompt_leakage_rate",
            "rollback_outputs_match_base",
        ],
        "boundary": "external provider adaptation only; LoRA never owns Taiji goals, memory, planning, or action decisions",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_qwen_lora_provider_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_qwen_lora_provider_baseline_20260825.json",
    )
    args = parser.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"local Qwen model directory not found: {args.model}")
    if args.epochs <= 0 or args.learning_rate <= 0.0:
        raise SystemExit("epochs and learning-rate must be positive")
    report = evaluate(
        args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
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
