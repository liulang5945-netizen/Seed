"""Run the P2-6 bounded IDE failure, budget, and restart continuation Gate."""

from __future__ import annotations

import argparse
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
from seed_platform.workbench import WorkbenchActionRequest  # noqa: E402
from taiji import ActionIntent, InputFrame  # noqa: E402

REPORT_FORMAT = "taiji-w7-p2-ide-restart-recovery-v1"
TARGET_PATH = "api/app.py"
RECOVERY_PATH = "plans/README.md"


def _frame() -> InputFrame:
    return InputFrame(
        input_id="p2-6-ide-task",
        modality="text",
        payload=b"continue the IDE task",
        source="scripts.p2_6",
        provenance="scripts.p2_6",
    )


def _request(runtime: SeedRuntime, intent: ActionIntent) -> WorkbenchActionRequest:
    environment = runtime.workbench_environment
    return WorkbenchActionRequest.from_action_intent(
        intent,
        snapshot_id=environment.capability_snapshot.snapshot_id,
        capability_registry_snapshot_id=environment.capability_registry.snapshot_id,
    )


def evaluate() -> dict[str, object]:
    checkpoint_path = PROJECT_ROOT / "reports" / f".tmp-p2-6-{uuid.uuid4().hex}.pt"
    missing_path = f"p2-6-missing-{uuid.uuid4().hex}.txt"
    with patch.object(workbench_module, "default_workspace_root", lambda: PROJECT_ROOT):
        runtime = SeedRuntime(Seed(episode_id="p2-6-ide-restart-recovery"))
        runtime.checkpoint_path = checkpoint_path
        architecture = runtime.model.architecture
        architecture.ingest_input(_frame(), learn=False)
        snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
        tick = architecture.tick
        intents = (
            ActionIntent(
                intent_id="p2-6-read-source",
                kind="workspace.read",
                parameters={"path": TARGET_PATH},
                confidence=1.0,
                tick=tick,
            ),
            ActionIntent(
                intent_id="p2-6-resolve-language",
                kind="workspace.programming_language.resolve",
                parameters={"path": TARGET_PATH},
                confidence=1.0,
                tick=tick,
            ),
            ActionIntent(
                intent_id="p2-6-fail-missing-read",
                kind="workspace.read",
                parameters={"path": missing_path},
                confidence=1.0,
                tick=tick,
            ),
        )
        requests = tuple(_request(runtime, intent) for intent in intents)
        preflight = runtime.preflight_workbench_loop(
            requests,
            loop_id="p2-6-failure-loop",
            max_steps=3,
            max_budget_units=3.0,
        )
        failure_run = runtime.execute_preflighted_workbench_loop(
            intents,
            requests,
            loop_id="p2-6-failure-loop",
            preflight_id=str(preflight["preflight_id"]),
            max_steps=3,
            max_budget_units=3.0,
        )
        failure_checkpoint_exists = checkpoint_path.is_file()
        restored = SeedRuntime.load(checkpoint_path)
        restored_loop_state = dict(restored._workbench_loop_state)
        recovery_intent = ActionIntent(
            intent_id="p2-6-recovery-read",
            kind="workspace.read",
            parameters={"path": RECOVERY_PATH},
            confidence=1.0,
            tick=restored.model.tick,
        )
        recovery_request = _request(restored, recovery_intent)
        recovery_preflight = restored.preflight_workbench_loop(
            (recovery_request,),
            loop_id="p2-6-recovery-loop",
            max_steps=1,
            max_budget_units=1.0,
        )
        recovery_run = restored.execute_preflighted_workbench_loop(
            (recovery_intent,),
            (recovery_request,),
            loop_id="p2-6-recovery-loop",
            preflight_id=str(recovery_preflight["preflight_id"]),
            max_steps=1,
            max_budget_units=1.0,
        )
        final_restored = SeedRuntime.load(checkpoint_path)

        budget_runtime = SeedRuntime(Seed(episode_id="p2-6-budget-boundary"))
        budget_runtime.checkpoint_path = checkpoint_path
        budget_architecture = budget_runtime.model.architecture
        budget_architecture.ingest_input(_frame(), learn=False)
        budget_tick = budget_architecture.tick
        budget_intents = tuple(
            ActionIntent(
                intent_id=f"p2-6-budget-{index}",
                kind="workspace.read",
                parameters={"path": path},
                confidence=1.0,
                tick=budget_tick,
            )
            for index, path in enumerate((TARGET_PATH, RECOVERY_PATH, "README.md"))
        )
        budget_requests = tuple(_request(budget_runtime, intent) for intent in budget_intents)
        budget_preflight = budget_runtime.preflight_workbench_loop(
            budget_requests,
            loop_id="p2-6-budget-loop",
            max_steps=3,
            max_budget_units=2.0,
        )

    metrics = {
        "preflight_accepts_bounded_failure_loop": (
            preflight["accepted"] is True
            and preflight["step_count"] == 3
            and preflight["checkpoint"]["boundary"] == "after_each_step"
        ),
        "failure_stops_after_successful_prefix_and_checkpoints": (
            failure_run["status"] == "failed"
            and failure_run["stopped_at"] == 2
            and failure_run["completed_prefix"] == 2
            and failure_run["steps"][0]["success"] is True
            and failure_run["steps"][1]["success"] is True
            and failure_run["steps"][2]["success"] is False
            and failure_run["steps"][2]["outcome"]["error_code"] == "not_found"
            and failure_checkpoint_exists
        ),
        "restart_restores_failed_loop_and_continues_with_fresh_request": (
            restored_loop_state["status"] == "failed"
            and set(restored_loop_state["committed_request_ids"]) == {
                request.request_id for request in requests
            }
            and recovery_preflight["accepted"] is True
            and recovery_run["status"] == "completed"
            and final_restored._workbench_loop_state["status"] == "completed"
            and recovery_run["steps"][0]["outcome"]["result"]["path"] == RECOVERY_PATH
        ),
        "budget_exhaustion_is_rejected_before_execution": (
            budget_preflight["accepted"] is False
            and budget_preflight["error_code"] == "loop_budget_limit"
            and budget_runtime.workbench_audit.events == ()
        ),
        "old_workbench_capability_remains_available_after_recovery": (
            final_restored.workbench_environment.capability_snapshot.get("workspace.read")
            is not None
            and final_restored.workbench_environment.capability_snapshot.get(
                "workspace.programming_language.resolve"
            )
            is not None
            and len(final_restored.workbench_audit.events) > 0
        ),
    }

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return {
        "format": REPORT_FORMAT,
        "metrics": metrics,
        "trace": {
            "target_path": TARGET_PATH,
            "recovery_path": RECOVERY_PATH,
            "missing_path": missing_path,
            "failed_loop_checkpoint": str(checkpoint_path),
            "failed_loop_committed_request_ids": list(
                restored_loop_state.get("committed_request_ids", ())
            ),
        },
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "a bounded Workbench loop must stop on a real failed step, checkpoint its "
                "prefix, reject over-budget plans before execution, restart from the failed "
                "state, continue with a fresh request, and preserve old capabilities"
            ),
        },
        "gap": {
            "current": (
                "bounded failure and restart continuation are proven for structured task "
                "evidence, while ordinary natural-language semantic resolution remains gated"
            ),
            "next": (
                "connect bounded semantic task decomposition to the same loop, then measure "
                "old-task retention and new-task transfer longitudinally"
            ),
        },
        "boundary": (
            "This Gate does not claim unrestricted autonomy, provider quality, CUDA, CI, "
            "open-domain growth benefit, or AGI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p2_ide_restart_recovery_20260831.json",
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
