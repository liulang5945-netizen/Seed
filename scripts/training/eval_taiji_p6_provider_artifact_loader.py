"""Smoke-test the manifest-selected raw/LoRA/guarded provider loader."""

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
    _prompt,
)
from scripts.training.eval_taiji_p6_qwen_realization_guard import (  # noqa: E402
    _expression,
)
from scripts.training.load_taiji_qwen_provider import (  # noqa: E402
    attach_qwen_language_provider,
)
from taiji import (  # noqa: E402
    GenerationController,
    LanguageProviderArtifact,
    TSKV8Adapter,
)

MANIFEST_FORMAT = "taiji-p6-provider-artifact-loader-manifest-v1"
REPORT_FORMAT = "taiji-p6-provider-artifact-loader-v1"


def evaluate(
    model_dir: Path,
    *,
    mode: str,
    adapter_dir: Path | None,
) -> dict[str, object]:
    artifact = LanguageProviderArtifact(
        artifact_id=f"qwen-{mode}-provider-v1",
        backend_id=BACKEND_ID,
        mode=mode,
        base_model=str(model_dir),
        adapter_path=None if adapter_dir is None else str(adapter_dir),
        training_corpus=(
            None if mode == "raw" else "reports/taiji_p6_language_train_holdout_baseline_20260825.json"
        ),
        training_report=(
            None if mode == "raw" else "reports/taiji_p6_qwen_lora_provider_baseline_20260825.json"
        ),
        safety_report=(
            None if mode != "guarded" else "reports/taiji_p6_qwen_lora_safety_baseline_20260825.json"
        ),
        default_enabled=False,
    )
    adapter = TSKV8Adapter()
    decoder = attach_qwen_language_provider(
        adapter,
        artifact,
        model_dir=model_dir,
        prompt_builder=_prompt,
    )
    expression = _expression(GenerationController(), HOLDOUT_CASES[0])
    emission = adapter.emit_language(expression)
    validation = emission.validation
    adapter.attach_language_organ(None)
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    safe = (
        validation is not None and (validation.accepted or emission.fallback_used)
        if mode == "guarded"
        else bool(emission.text_bytes)
    )
    gate_passed = bool(
        safe
        and adapter.language_provider_artifact == artifact
        and restored.language_provider_artifact == artifact
        and restored.cognitive_snapshot().action_intent is None
        and decoder.model.training is False
    )
    return {
        "format": REPORT_FORMAT,
        "artifact": artifact.to_payload(),
        "mode": mode,
        "emission": {
            "backend": emission.backend,
            "fallback_used": emission.fallback_used,
            "validation": None if validation is None else validation.to_payload(),
        },
        "checkpoint": {
            "artifact_round_trip": restored.language_provider_artifact == artifact,
            "cognition_unchanged": restored.cognitive_snapshot().action_intent is None,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "artifact-selected provider mode loads at the integration edge, Taiji checkpoint preserves the manifest, and guarded output remains safe without cognition ownership",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "load raw/LoRA/guarded provider modes from one LanguageProviderArtifact manifest",
        "lesions": ["artifact_metadata_loss", "wrong_provider_mode", "cognition_dependency"],
        "signals": ["artifact_round_trip", "fallback_used", "validation", "cognition_unchanged"],
        "boundary": "loader configuration Gate only; provider quality remains governed by separate train/holdout and safety reports",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw", "lora", "guarded"), default="guarded")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_provider_artifact_loader_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p6_provider_artifact_loader_baseline_20260825.json",
    )
    args = parser.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"local Qwen model directory not found: {args.model}")
    if args.mode in {"lora", "guarded"} and (args.adapter_dir is None or not args.adapter_dir.is_dir()):
        raise SystemExit("LoRA and guarded modes require an existing --adapter-dir")
    report = evaluate(args.model, mode=args.mode, adapter_dir=args.adapter_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
