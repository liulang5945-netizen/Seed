"""P6-1a canary: independent semantic provider seam before Workbench planning."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import Seed  # noqa: E402
from taiji import (  # noqa: E402
    SemanticEvidenceProposal,
    SemanticEvidenceProvider,
    SemanticProviderRequest,
)

REPORT_FORMAT = "taiji-w7-p6-1a-semantic-provider-interface-v1"


class _CanaryProvider:
    provider_id = "deterministic-p6-1a.semantic"

    def propose(self, request: SemanticProviderRequest) -> SemanticEvidenceProposal:
        return SemanticEvidenceProposal.from_frame(
            request.frame,
            provider_id=self.provider_id,
            goal_description="检查目标文件并准备工作台",
            constraints=request.constraints,
            context_digest=request.context_digest,
            semantic_steps=(
                {
                    "description": "读取目标文件并确认当前内容",
                    "semantic_slots": {"operation": "inspect", "path": "README.md"},
                },
            ),
            confidence=0.9,
            ambiguity=0.1,
            provenance="p6-1a-canary.provider",
            tick=request.frame.timestamp,
        )

    def checkpoint(self) -> Mapping[str, object]:
        return {
            "format": "taiji-p6-1a-provider-descriptor-v1",
            "provider_id": self.provider_id,
        }


def evaluate() -> dict[str, object]:
    provider = _CanaryProvider()
    runtime = SeedRuntime(
        Seed(episode_id="p6-1a-semantic-provider-interface"),
        semantic_provider=provider,
    )
    prompt = "读取 README.md 并准备工作台"
    _, frame = runtime._task_frame(prompt)
    first_request = SemanticProviderRequest.from_frame(frame, constraints=("只读",))
    second_request = SemanticProviderRequest.from_frame(frame, constraints=("只读",))
    result = runtime.interpret_workbench_task(prompt, constraints=("只读",))

    metrics = {
        "provider_implements_versioned_interface": isinstance(
            provider, SemanticEvidenceProvider
        ),
        "request_is_content_addressed_and_repeatable": (
            first_request.request_id == second_request.request_id
            and first_request.input_digest
            == first_request.to_payload()["input_digest"]
        ),
        "request_has_no_execution_authority": not any(
            field in first_request.to_payload()
            for field in ("action_intent", "capability_id", "parameter_bindings", "tool")
        ),
        "taiji_admits_provider_evidence_before_any_plan": (
            result["interpretation"]["status"] == "resolved"
            and bool(result["decomposition"]["steps"])
            and result["execution"]["action_intent"] is None
            and result["execution"]["tool_call"] is None
            and result["execution"]["side_effects"] is False
        ),
        "no_workbench_side_effect_at_interpretation_boundary": (
            runtime.workbench_audit.events == ()
        ),
        "unavailable_mode_is_explicit": (
            SeedRuntime(Seed(episode_id="p6-1a-unavailable")).semantic_provider_status()
            == {
                "interface": "taiji-semantic-provider-interface-v1",
                "state": "unavailable",
                "provider_id": "",
                "evidence_enabled": "false",
                "reason_code": "semantic_provider_not_attached",
            }
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "task": "Independent semantic provider request and admission seam",
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "An explicitly attached semantic provider may return only validated, "
                "content-addressed evidence; no provider output may become a capability, "
                "ActionIntent, tool call, or Workbench side effect at interpretation time."
            ),
        },
        "boundary": (
            "This Gate establishes the provider seam and honest unavailable state. It does "
            "not claim real model quality, broad language understanding, automatic provider "
            "selection, Workbench execution, CUDA, CI, or open-domain autonomy."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = evaluate()
    report_path = PROJECT_ROOT / "reports" / "taiji_w7_p6_1a_semantic_provider_interface_20260831.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
