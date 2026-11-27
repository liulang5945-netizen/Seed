"""Run a real local Qwen model through the Taiji semantic-provider boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import QwenSemanticEvidenceProvider, Seed  # noqa: E402
from taiji import language_provider_content_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-p6-1c-qwen-semantic-provider-v1"


def evaluate(model_dir: Path, *, expected_model_digest: str, prompt: str) -> dict[str, object]:
    provider = QwenSemanticEvidenceProvider.from_model_dir(
        model_dir,
        expected_model_digest=expected_model_digest,
    )
    runtime = SeedRuntime(
        Seed(episode_id="p6-1c-qwen-semantic-provider"),
        semantic_provider=provider,
    )
    result = runtime.interpret_workbench_task(prompt, constraints=("只读",))
    proposal = result["provider_evidence"]
    gate_passed = bool(
        result["semantic_provider"]["state"] == "attached"
        and result["execution"]["action_intent"] is None
        and result["execution"]["tool_call"] is None
        and result["execution"]["side_effects"] is False
        and runtime.workbench_audit.events == ()
        and proposal["input_digest"]
        and proposal["evidence_digest"]
        and result["interpretation"]["status"] in {"resolved", "candidate"}
    )
    return {
        "format": REPORT_FORMAT,
        "model": {
            "model_dir": str(model_dir),
            "model_digest": language_provider_content_digest(model_dir),
            "backend_id": provider.artifact.backend_id,
        },
        "provider_artifact": provider.artifact.to_payload(),
        "request": {
            "prompt": prompt,
            "constraints": ["只读"],
        },
        "admission": result,
        "checkpoint": provider.checkpoint(),
        "gate": {
            "passed": gate_passed,
            "criterion": (
                "real Qwen output is admitted only as content-addressed semantic evidence; "
                "Taiji retains interpretation ownership and no Workbench side effect occurs"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-digest", required=True)
    parser.add_argument("--prompt", default="读取 README.md 并确认当前内容")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p6_1c_qwen_semantic_provider_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate(
        args.model,
        expected_model_digest=args.expected_model_digest,
        prompt=args.prompt,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
