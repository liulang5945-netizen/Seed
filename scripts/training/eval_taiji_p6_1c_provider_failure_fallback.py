"""Verify semantic-provider failure falls back to a Goal-only Taiji boundary."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import Seed  # noqa: E402
from taiji import SemanticEvidenceProposal, SemanticProviderRequest  # noqa: E402

REPORT_FORMAT = "taiji-w7-p6-1c-semantic-provider-failure-fallback-v1"


class _FailingSemanticProvider:
    """Deterministic provider fault used only by this boundary evaluator."""

    provider_id = "eval.semantic-provider.failure"

    def propose(self, request: SemanticProviderRequest) -> SemanticEvidenceProposal:
        del request
        raise RuntimeError("synthetic semantic provider outage")

    def checkpoint(self) -> Mapping[str, object]:
        return {
            "format": "taiji-semantic-provider-test-descriptor-v1",
            "provider_id": self.provider_id,
        }


def evaluate(prompt: str) -> dict[str, object]:
    runtime = SeedRuntime(
        Seed(episode_id="p6-1c-semantic-provider-failure-fallback"),
        semantic_provider=_FailingSemanticProvider(),
    )
    result = runtime.interpret_workbench_task(prompt)
    fallback = result.get("semantic_provider_fallback")
    checks = {
        "provider_degraded": result["semantic_provider"]["state"] == "degraded",
        "failure_reason_is_explicit": (
            result["semantic_provider"]["reason_code"] == "semantic_provider_failed"
        ),
        "goal_only_candidate_returned": result["interpretation"]["status"] == "candidate",
        "fallback_mode_is_goal_only": fallback == {
            "mode": "goal_only",
            "reason_code": "semantic_provider_failed",
        },
        "no_provider_evidence_admitted": "provider_evidence" not in result,
        "no_execution_authority": (
            result["execution"]["action_intent"] is None
            and result["execution"]["tool_call"] is None
            and result["execution"]["side_effects"] is False
        ),
        "no_workbench_side_effect": runtime.workbench_audit.events == (),
    }
    return {
        "format": REPORT_FORMAT,
        "request": {"prompt": prompt},
        "result": result,
        "provider_checkpoint": runtime.semantic_provider.checkpoint(),
        "checks": checks,
        "gate": {
            "passed": all(checks.values()),
            "criterion": (
                "a provider failure is observable and recoverable as a Goal-only candidate; "
                "no evidence, execution authority, or Workbench side effect crosses the boundary"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="读取 README.md")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_p6_1c_provider_failure_fallback_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate(args.prompt)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
