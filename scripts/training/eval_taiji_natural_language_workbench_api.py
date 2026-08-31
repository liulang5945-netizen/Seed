"""P2-13 canary: expose the Taiji-owned write protocol through product APIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import create_app  # noqa: E402
from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import Seed  # noqa: E402
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import SemanticEvidenceProposal  # noqa: E402

REPORT_FORMAT = "taiji-w7-p2-13-natural-language-workbench-api-v1"
TARGET_PATH = "reports/.p2-13-api-fixture.txt"
ORIGINAL_CONTENT = "Seed API source\n"
UPDATED_CONTENT = "Taiji API source\n"


def _runtime(seed: int, checkpoint_path: Path) -> SeedRuntime:
    runtime = SeedRuntime(
        Seed(episode_id=f"p2-13-natural-language-api-{seed}"),
        checkpoint_path=checkpoint_path,
    )
    runtime._workbench_environment = WorkbenchEnvironment(PROJECT_ROOT)
    return runtime


def _proposal(runtime: SeedRuntime, prompt: str) -> SemanticEvidenceProposal:
    _, frame = runtime._task_frame(prompt)
    return SemanticEvidenceProposal.from_frame(
        frame,
        provider_id="deterministic-p2-13-canary",
        goal_description=prompt,
        semantic_steps=(
            {
                "description": "读取当前待编辑文件",
                "semantic_slots": {"operation": "read", "path": TARGET_PATH},
                "expected_outcome": "获得当前内容和 digest",
            },
            {
                "description": "将文件中的 Seed 替换为 Taiji",
                "semantic_slots": {
                    "operation": "patch",
                    "path": TARGET_PATH,
                    "edit": {
                        "kind": "replace_text",
                        "find": "Seed",
                        "replace": "Taiji",
                    },
                },
                "expected_outcome": "生成一个可预览、可审批、可 undo 的文本替换",
            },
        ),
        confidence=0.95,
        ambiguity=0.05,
        provenance="p2-13-canary.provider",
        tick=runtime.model.tick,
    )


def evaluate() -> dict[str, object]:
    fixture_path = PROJECT_ROOT / TARGET_PATH
    checkpoint_path = PROJECT_ROOT / "checkpoints" / ".p2-13-natural-language-api.pt"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.unlink(missing_ok=True)
    fixture_path.write_text(ORIGINAL_CONTENT, encoding="utf-8")
    prompt = "请把 reports/.p2-13-api-fixture.txt 中的 Seed 改成 Taiji"

    try:
        with patch(
            "seed_platform.workbench.get_setting",
            lambda key, default=None: str(PROJECT_ROOT)
            if key == "workspace_path"
            else default,
        ):
            runtime = _runtime(13, checkpoint_path)
            with patch("api.seed_runtime._runtime", runtime):
                with TestClient(create_app(startup_tasks=False)) as client:
                    capabilities = client.get("/api/workbench/capabilities")
                    snapshot_id = capabilities.json()["snapshot_id"]
                    proposal = _proposal(runtime, prompt).to_payload()
                    plan_payload = {
                        "prompt": prompt,
                        "semantic_evidence": proposal,
                        "snapshot_id": snapshot_id,
                        "loop_id": "p2-13-api-loop",
                        "max_steps": 2,
                        "max_budget_units": 2.0,
                        "resource_budget": 0.8,
                    }
                    plan_response = client.post(
                        "/api/chat/workbench/natural-language/plan",
                        json=plan_payload,
                    )
                    if plan_response.status_code != 200:
                        raise AssertionError(
                            f"natural-language plan route failed: "
                            f"HTTP {plan_response.status_code} {plan_response.text}"
                        )
                    plan = plan_response.json()
                    if "approval_requirements" not in plan or "provider_evidence" not in plan:
                        raise AssertionError(
                            f"natural-language plan route returned unexpected payload: {plan}"
                        )
                    patch_slots = plan["provider_evidence"]["semantic_steps"][1]["semantic_slots"]
                    plan_id = plan.get("plan_id")
                    requirement = plan["approval_requirements"][0]
                    request_id = requirement["request_id"]
                    unchanged_before_approval = (
                        fixture_path.read_text(encoding="utf-8") == ORIGINAL_CONTENT
                    )

                    missing = client.post(
                        "/api/chat/workbench/natural-language/execute",
                        json={"plan_id": plan_id, "approval_tokens": {}},
                    )
                    missing_payload = missing.json()
                    unchanged_after_missing = (
                        fixture_path.read_text(encoding="utf-8") == ORIGINAL_CONTENT
                    )

                    approval = client.post(
                        "/api/chat/workbench/natural-language/approve",
                        json={"plan_id": plan_id, "request_id": request_id},
                    )
                    approval_payload = approval.json()
                    duplicate_approval = client.post(
                        "/api/chat/workbench/natural-language/approve",
                        json={"plan_id": plan_id, "request_id": request_id},
                    )
                    duplicate_payload = duplicate_approval.json()
                    execute = client.post(
                        "/api/chat/workbench/natural-language/execute",
                        json={
                            "plan_id": plan_id,
                            "approval_tokens": {
                                request_id: approval_payload["approval_token"]
                            },
                        },
                    )
                    execute_payload = execute.json()
                    updated = fixture_path.read_text(encoding="utf-8")

                    fixture_path.write_text(ORIGINAL_CONTENT, encoding="utf-8")
                    stale_plan = client.post(
                        "/api/chat/workbench/natural-language/plan",
                        json={
                            **plan_payload,
                            "loop_id": "p2-13-stale-loop",
                            "semantic_evidence": _proposal(runtime, prompt).to_payload(),
                        },
                    ).json()
                    runtime.chat("推进当前 Taiji tick")
                    stale_approval = client.post(
                        "/api/chat/workbench/natural-language/approve",
                        json={
                            "plan_id": stale_plan["plan_id"],
                            "request_id": stale_plan["approval_requirements"][0]["request_id"],
                        },
                    )
                    expired = client.post(
                        "/api/chat/workbench/natural-language/execute",
                        json={"plan_id": "missing-p2-13-plan", "approval_tokens": {}},
                    )

                    schema = client.get("/openapi.json").json()
                    plan_schema = schema["paths"][
                        "/api/chat/workbench/natural-language/plan"
                    ]["post"]["requestBody"]["content"]["application/json"]["schema"]
                    plan_schema_text = json.dumps(plan_schema, ensure_ascii=False)

        metrics = {
            "product_plan_route_returns_taiji_plan": (
                plan_response.status_code == 200
                and plan["status"] == "needs_approval"
                and bool(plan_id)
                and len(plan["approval_requirements"]) == 1
            ),
            "api_does_not_reexpose_final_patch_or_bindings": (
                "patch" not in patch_slots
                and "before_digest" not in patch_slots
                and "expected_after_digest" not in patch_slots
                and "parameter_bindings" not in plan_schema_text
                and "action_intent" not in plan_schema_text
            ),
            "missing_approval_is_visible_and_side_effect_free": (
                missing.status_code == 200
                and missing_payload["status"] == "rejected"
                and missing_payload["reason_code"] == "capability_requires_approval"
                and missing_payload["execution"]["status"] == "not_executed"
                and unchanged_after_missing
            ),
            "approval_is_idempotent_and_write_completes": (
                approval.status_code == 200
                and duplicate_approval.status_code == 200
                and approval_payload["approval_token"] == duplicate_payload["approval_token"]
                and execute.status_code == 200
                and execute_payload["status"] == "completed"
                and unchanged_before_approval
                and updated == UPDATED_CONTENT
                and checkpoint_path.is_file()
            ),
            "stale_and_unknown_plans_fail_closed": (
                stale_approval.status_code == 400
                and expired.status_code == 400
                and fixture_path.read_text(encoding="utf-8") == ORIGINAL_CONTENT
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "task": "Taiji-owned natural-language Workbench plan/approve/execute product API",
            "routes": {
                "plan": "/api/chat/workbench/natural-language/plan",
                "approve": "/api/chat/workbench/natural-language/approve",
                "execute": "/api/chat/workbench/natural-language/execute",
            },
            "metrics": metrics,
            "trace": {
                "plan_id": plan_id,
                "plan_status": plan["status"],
                "missing_approval": {
                    "http_status": missing.status_code,
                    "status": missing_payload["status"],
                    "reason_code": missing_payload["reason_code"],
                },
                "duplicate_approval_same_token": (
                    approval_payload["approval_token"] == duplicate_payload["approval_token"]
                ),
                "stale_approval_http_status": stale_approval.status_code,
                "unknown_plan_http_status": expired.status_code,
            },
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "The product boundary must expose Taiji's plan/approval/execute protocol, "
                    "keep patch and final binding ownership in Taiji, make approval idempotent, "
                    "and fail closed on missing or stale plans."
                ),
            },
            "boundary": (
                "This Gate validates the native API and frontend transport boundary with deterministic "
                "semantic evidence. It does not claim a real provider can generate semantic evidence, "
                "a complete chat UI journey, CUDA, CI, or open-domain autonomy."
            ),
        }
    finally:
        checkpoint_path.unlink(missing_ok=True)
        fixture_path.unlink(missing_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report_path = PROJECT_ROOT / "reports" / "taiji_w7_p2_13_natural_language_workbench_api_20260831.json"
    report = evaluate()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
