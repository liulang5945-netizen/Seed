"""Run the P2-5 real reversible IDE programming-task canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import seed_platform.workbench as workbench_module  # noqa: E402
from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import Seed  # noqa: E402
from taiji import ActionIntent, InputFrame, TaskInterpretation  # noqa: E402

REPORT_FORMAT = "taiji-w7-p2-reversible-ide-canary-v1"
TARGET_PATH = "api/app.py"
OLD_TEXT = "Taiji FastAPI application factory."
NEW_TEXT = "Taiji native FastAPI application factory."


def _task_frame() -> InputFrame:
    return InputFrame(
        input_id="p2-5-ide-task",
        modality="text",
        payload=b"update api/app.py",
        source="scripts.p2_5",
        provenance="scripts.p2_5",
    )


def evaluate() -> dict[str, object]:
    checkpoint_path = PROJECT_ROOT / "reports" / f".tmp-p2-5-{uuid.uuid4().hex}.pt"
    source_path = PROJECT_ROOT / TARGET_PATH
    original_digest = ""
    patch_undo_token = ""
    runtime: SeedRuntime | None = None
    restored: SeedRuntime | None = None
    with patch.object(workbench_module, "default_workspace_root", lambda: PROJECT_ROOT):
        try:
            runtime = SeedRuntime(Seed(episode_id="p2-5-reversible-ide"))
            architecture = runtime.model.architecture
            architecture.ingest_input(_task_frame(), learn=False)
            snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id

            read_intent = ActionIntent(
                intent_id="p2-5-read-source",
                kind="workspace.read",
                parameters={"path": TARGET_PATH},
                confidence=1.0,
                tick=architecture.tick,
            )
            read_execution = runtime.execute_workbench_intent(
                read_intent,
                snapshot_id=snapshot_id,
                learn=False,
            )
            read_result = dict(read_execution["outcome"]["result"])
            original_content = str(read_result["content"])
            original_digest = str(read_result["digest"])

            interpretation = architecture.interpret_task_input(
                _task_frame(),
                goal_description="更新 api/app.py 的应用工厂说明",
                status="resolved",
                confidence=0.9,
                ambiguity=0.1,
            )
            planned = runtime.plan_language_selection(
                snapshot_id=snapshot_id,
                path=TARGET_PATH,
                resource_budget=0.8,
            )
            decision = architecture.last_executive_decision
            if decision is None or decision.action_intent.kind != "editor.set_language":
                raise AssertionError("P2-5 language planner did not produce editor.set_language")
            language_execution = runtime.execute_workbench_intent(
                decision.action_intent,
                snapshot_id=snapshot_id,
                learn=False,
            )
            language_result = dict(language_execution["outcome"]["result"])

            start = original_content.index(OLD_TEXT)
            updated_content = (
                original_content[:start]
                + NEW_TEXT
                + original_content[start + len(OLD_TEXT) :]
            )
            updated_digest = hashlib.sha256(updated_content.encode("utf-8")).hexdigest()
            patch_intent = ActionIntent(
                intent_id="p2-5-apply-source-patch",
                kind="workspace.apply_patch",
                parameters={
                    "path": TARGET_PATH,
                    "before_digest": original_digest,
                    "patch": {
                        "kind": "text_replace",
                        "operations": [
                            {"start": start, "end": start + len(OLD_TEXT), "text": NEW_TEXT}
                        ],
                    },
                    "expected_after_digest": updated_digest,
                },
                source_goal_id=interpretation.goal_id,
                confidence=0.9,
                tick=architecture.tick,
            )
            preview = runtime.preview_workbench_intent(
                patch_intent,
                snapshot_id=snapshot_id,
            )
            approval = preview.get("approval") or {}
            approval_token = str(approval.get("approval_token", ""))
            if not approval_token:
                raise AssertionError("P2-5 patch preview did not issue approval")
            preview_kept_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            patch_execution = runtime.execute_workbench_intent(
                patch_intent,
                snapshot_id=snapshot_id,
                approval_token=approval_token,
                learn=False,
            )
            patch_outcome = dict(patch_execution["outcome"])
            patch_transaction = dict(patch_outcome["transaction"] or {})
            patch_undo_token = str(patch_transaction.get("undo_token", ""))
            changed_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()

            runtime.save(checkpoint_path)
            restored = SeedRuntime.load(checkpoint_path)
            restored_language_selections = restored.workbench_environment.language_state_checkpoint()[
                "selections"
            ]
            restored_language = next(
                (
                    item
                    for item in restored_language_selections
                    if item.get("path") == TARGET_PATH
                ),
                {},
            )
            undo_intent = ActionIntent(
                intent_id="p2-5-undo-source-patch",
                kind="workspace.undo",
                parameters={"undo_token": patch_undo_token},
                confidence=1.0,
                tick=restored.model.tick,
            )
            undo_preview = restored.preview_workbench_intent(
                undo_intent,
                snapshot_id=restored.workbench_environment.capability_snapshot.snapshot_id,
            )
            undo_approval = undo_preview.get("approval") or {}
            undo_execution = restored.execute_workbench_intent(
                undo_intent,
                snapshot_id=restored.workbench_environment.capability_snapshot.snapshot_id,
                approval_token=str(undo_approval.get("approval_token", "")),
                learn=False,
            )
            restored_read = restored.execute_workbench_intent(
                ActionIntent(
                    intent_id="p2-5-read-after-recovery",
                    kind="workspace.read",
                    parameters={"path": TARGET_PATH},
                    confidence=1.0,
                    tick=restored.model.tick,
                ),
                snapshot_id=restored.workbench_environment.capability_snapshot.snapshot_id,
                learn=False,
            )
            restored_content = str(restored_read["outcome"]["result"]["content"])
            metrics = {
                "read_outcome_provided_content_digest": (
                    read_execution["outcome"]["status"] == "success"
                    and original_digest == hashlib.sha256(original_content.encode("utf-8")).hexdigest()
                ),
                "language_plan_reaches_and_executes_editor_selection": (
                    planned["assessment"]["selection_state"] == "resolved"
                    and planned["planner"]["status"] == "planned"
                    and language_execution["policy"]["decision"] == "allow"
                    and language_result["selection_state"] == "taiji_selection"
                ),
                "patch_is_previewed_approved_and_applied": (
                    preview["policy"]["reason_code"] == "capability_requires_approval"
                    and preview_kept_digest == original_digest
                    and patch_outcome["status"] == "success"
                    and changed_digest == updated_digest
                    and bool(patch_undo_token)
                ),
                "checkpoint_restores_language_and_undo_state": (
                    checkpoint_path.is_file()
                    and restored_language.get("selection_state") == "taiji_selection"
                    and bool(undo_approval.get("approval_token"))
                    and undo_execution["outcome"]["status"] == "success"
                ),
                "recovery_restores_file_and_keeps_read_capability": (
                    restored_content == original_content
                    and hashlib.sha256(source_path.read_bytes()).hexdigest() == original_digest
                    and restored_read["outcome"]["status"] == "success"
                ),
            }
        finally:
            if patch_undo_token and source_path.is_file():
                current_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
                if current_digest != original_digest and runtime is not None:
                    runtime.workbench_environment.execute_tool(
                        "workspace.undo",
                        {"undo_token": patch_undo_token},
                    )
            if checkpoint_path.exists():
                checkpoint_path.unlink()

    return {
        "format": REPORT_FORMAT,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "a resolved Taiji task reads a real source file, grounds an IDE language "
                "selection, previews and approves a reversible patch, executes it, persists "
                "language and undo state, then recovers after checkpoint restart"
            ),
        },
        "trace": {
            "target_path": TARGET_PATH,
            "read_digest": original_digest,
            "updated_digest": updated_digest,
            "language_id": language_result["programming_language_id"],
            "audit_phases": [event.phase for event in runtime.workbench_audit.events]
            if runtime is not None
            else [],
        },
        "gap": {
            "current": (
                "the real IDE canary is executable for resolved task evidence, but ordinary "
                "natural-language semantic resolution and multi-step task decomposition remain gated"
            ),
            "next": (
                "connect a bounded semantic resolver to this canary, then verify restart "
                "continuation, failure recovery, budget exhaustion, and old-task retention"
            ),
        },
        "boundary": (
            "This Gate does not claim open-domain autonomy, unrestricted file writing, "
            "provider quality, CUDA, CI, structural growth benefit, or AGI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p2_reversible_ide_canary_20260831.json",
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
