"""Evaluate Taiji-owned semantic edit grounding and controlled Workbench writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import Seed  # noqa: E402
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import ActionIntent, SemanticEvidenceProposal  # noqa: E402

REPORT_FORMAT = "taiji-w7-p2-12-natural-language-write-v1"
TARGET_PATH = "reports/.p2-12-edit-fixture.txt"
ORIGINAL_CONTENT = "Seed editor source\n"
UPDATED_CONTENT = "Taiji editor source\n"


def _runtime(seed: int, checkpoint_path: Path) -> SeedRuntime:
    runtime = SeedRuntime(
        Seed(episode_id=f"p2-12-natural-language-write-{seed}"),
        checkpoint_path=checkpoint_path,
    )
    runtime._workbench_environment = WorkbenchEnvironment(PROJECT_ROOT)
    return runtime


def _proposal(runtime: SeedRuntime, prompt: str) -> SemanticEvidenceProposal:
    _, frame = runtime._task_frame(prompt)
    return SemanticEvidenceProposal.from_frame(
        frame,
        provider_id="deterministic-p2-12-canary",
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
        provenance="p2-12-canary.provider",
        tick=runtime.model.tick,
    )


def _plan(runtime: SeedRuntime, prompt: str, loop_id: str) -> dict[str, object]:
    return runtime.plan_natural_language_workbench_task(
        prompt,
        _proposal(runtime, prompt),
        snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
        loop_id=loop_id,
        max_steps=2,
        max_budget_units=2.0,
        resource_budget=0.8,
    )


def evaluate() -> dict[str, object]:
    fixture_path = PROJECT_ROOT / TARGET_PATH
    checkpoint_path = PROJECT_ROOT / "checkpoints" / ".p2-12-natural-language-write.pt"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.unlink(missing_ok=True)
    fixture_path.write_text(ORIGINAL_CONTENT, encoding="utf-8")
    prompt = "请把 reports/.p2-12-edit-fixture.txt 中的 Seed 改成 Taiji"

    try:
        with patch(
            "seed_platform.workbench.get_setting",
            lambda key, default=None: str(PROJECT_ROOT)
            if key == "workspace_path"
            else default,
        ):
            runtime = _runtime(11, checkpoint_path)
            plan = _plan(runtime, prompt, "p2-12-write-loop")
            provider_steps = plan["provider_evidence"]["semantic_steps"]
            provider_patch_step = provider_steps[1]["semantic_slots"]
            approval_requirements = plan["approval_requirements"]
            patch_request_id = approval_requirements[0]["request_id"]
            preview = runtime.approve_planned_natural_language_workbench_task(
                plan["plan_id"],
                patch_request_id,
            )
            unchanged_before_execute = fixture_path.read_text(encoding="utf-8") == ORIGINAL_CONTENT
            approved = runtime.execute_planned_natural_language_workbench_task(
                plan["plan_id"],
                {patch_request_id: preview["approval_token"]},
            )
            updated_digest = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
            updated_content = fixture_path.read_text(encoding="utf-8")
            patch_step = approved["execution"]["steps"][1]
            patch_outcome = patch_step["outcome"]
            undo_token = patch_outcome["transaction"]["undo_token"]
            restored = SeedRuntime.load(checkpoint_path)
            undo_intent = ActionIntent(
                intent_id="p2-12-undo-edit",
                kind="workspace.undo",
                parameters={"undo_token": undo_token},
                confidence=1.0,
                tick=restored.model.tick,
            )
            undo_preview = restored.preview_workbench_intent(
                undo_intent,
                snapshot_id=restored.workbench_environment.capability_snapshot.snapshot_id,
            )
            undo_execution = restored.execute_workbench_intent(
                undo_intent,
                snapshot_id=restored.workbench_environment.capability_snapshot.snapshot_id,
                approval_token=undo_preview["approval"]["approval_token"],
                learn=False,
            )
            restored_content = fixture_path.read_text(encoding="utf-8")
            undo_restored = restored_content == ORIGINAL_CONTENT

            no_approval_checkpoint = (
                PROJECT_ROOT / "checkpoints" / ".p2-12-no-approval.pt"
            )
            no_approval_checkpoint.unlink(missing_ok=True)
            no_approval_runtime = _runtime(29, no_approval_checkpoint)
            no_approval_plan = _plan(
                no_approval_runtime,
                prompt,
                "p2-12-no-approval-loop",
            )
            no_approval = no_approval_runtime.execute_planned_natural_language_workbench_task(
                no_approval_plan["plan_id"],
                {},
            )
            no_approval_unchanged = fixture_path.read_text(encoding="utf-8") == ORIGINAL_CONTENT
            no_approval_checkpoint.unlink(missing_ok=True)

            conflict_checkpoint = PROJECT_ROOT / "checkpoints" / ".p2-12-conflict.pt"
            conflict_checkpoint.unlink(missing_ok=True)
            conflict_runtime = _runtime(47, conflict_checkpoint)
            conflict_plan = _plan(conflict_runtime, prompt, "p2-12-conflict-loop")
            conflict_request_id = conflict_plan["approval_requirements"][0]["request_id"]
            conflict_approval = (
                conflict_runtime.approve_planned_natural_language_workbench_task(
                    conflict_plan["plan_id"],
                    conflict_request_id,
                )
            )
            fixture_path.write_text("Seed editor source\nexternal change\n", encoding="utf-8")
            conflict = conflict_runtime.execute_planned_natural_language_workbench_task(
                conflict_plan["plan_id"],
                {conflict_request_id: conflict_approval["approval_token"]},
            )
            conflict_step = conflict["execution"]["steps"][1]
            conflict_content = fixture_path.read_text(encoding="utf-8")
            conflict_checkpoint.unlink(missing_ok=True)

        metrics = {
            "provider_submits_semantic_edit_not_patch": (
                "patch" not in provider_patch_step
                and "before_digest" not in provider_patch_step
                and "expected_after_digest" not in provider_patch_step
            ),
            "taiji_derives_current_digest_checked_patch": (
                plan["status"] == "needs_approval"
                and len(approval_requirements) == 1
                and approval_requirements[0]["capability_id"] == "workspace.apply_patch"
                and preview["preview"]["mutation"]["before_digest"]
                != preview["preview"]["mutation"]["after_digest"]
                and unchanged_before_execute
            ),
            "explicit_preview_approval_and_write_succeed": (
                preview["policy"]["reason_code"] == "capability_requires_approval"
                and approved["status"] == "completed"
                and approved["execution"]["status"] == "completed"
                and approved["execution"]["steps"]
                and all(item["success"] for item in approved["execution"]["steps"])
                and unchanged_before_execute
                and updated_content == UPDATED_CONTENT
                and updated_digest == preview["preview"]["mutation"]["after_digest"]
            ),
            "checkpoint_undo_and_recovery_restore_original": (
                checkpoint_path.is_file()
                and bool(undo_preview["approval"]["approval_token"])
                and undo_execution["outcome"]["status"] == "success"
                and undo_restored
            ),
            "missing_approval_fails_closed_before_write": (
                no_approval["status"] == "rejected"
                and no_approval["reason_code"] == "capability_requires_approval"
                and no_approval["execution"]["status"] == "not_executed"
                and no_approval_unchanged
            ),
            "digest_conflict_stops_before_patch_mutation": (
                conflict["status"] == "failed"
                and conflict_step["success"] is False
                and conflict_step["outcome"]["error_code"] == "transaction_conflict"
                and conflict_content == "Seed editor source\nexternal change\n"
            ),
        }
        return {
            "format": REPORT_FORMAT,
            "task": "Taiji-owned natural-language semantic edit and controlled Workbench write",
            "plan": {
                "status": plan["status"],
                "approval_requirements": approval_requirements,
                "provider_patch_semantic_slots": provider_patch_step,
            },
            "trace": {
                "approved": {
                    "status": approved["status"],
                    "execution_status": approved["execution"]["status"],
                    "unchanged_before_execute": unchanged_before_execute,
                    "updated_content": updated_content,
                    "updated_digest": updated_digest,
                    "expected_updated_digest": preview["preview"]["mutation"]["after_digest"],
                    "step_statuses": [
                        {"success": item["success"], "error_code": item.get("error_code", "")}
                        for item in approved["execution"].get("steps", [])
                    ],
                },
                "undo": {
                    "execution_status": undo_execution["outcome"]["status"],
                    "restored": undo_restored,
                },
                "missing_approval": {
                    "status": no_approval["status"],
                    "reason_code": no_approval["reason_code"],
                    "execution_status": no_approval["execution"]["status"],
                    "unchanged": no_approval_unchanged,
                },
            },
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "Taiji must derive a digest-checked structured patch from live file evidence, "
                    "require explicit preview/approval before writing, checkpoint the outcome, "
                    "support undo/recovery, and stop on missing approval or stale file state"
                ),
            },
            "boundary": (
                "This Gate covers one deterministic UTF-8 replacement on CPU. It does not claim "
                "general code editing, real provider quality, CUDA, CI, or open-domain autonomy."
            ),
        }
    finally:
        checkpoint_path.unlink(missing_ok=True)
        fixture_path.unlink(missing_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p2_12_natural_language_write_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
