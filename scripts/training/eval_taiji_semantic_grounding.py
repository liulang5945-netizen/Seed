"""Evaluate Taiji-owned semantic-slot grounding without external bindings."""

from __future__ import annotations

import argparse
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
from taiji import SemanticEvidenceProposal  # noqa: E402

REPORT_FORMAT = "taiji-w7-p2-9-semantic-grounding-v1"
LEARNER_SEEDS = (11, 29, 47)
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def _proposal(
    runtime: SeedRuntime,
    prompt: str,
    *,
    operation: str = "read",
    path: str = "README.md",
    confidence: float = 0.95,
    ambiguity: float = 0.05,
    forbidden: bool = False,
) -> SemanticEvidenceProposal:
    _, frame = runtime._task_frame(prompt)
    slots = {"operation": operation, "path": path}
    if forbidden:
        slots["parameters"] = {"path": path}
    return SemanticEvidenceProposal.from_frame(
        frame,
        provider_id="deterministic-p2-9-canary",
        goal_description=prompt,
        semantic_steps=(
            {
                "description": "根据语义操作读取用户指定的工作区文件",
                "semantic_slots": slots,
                "expected_outcome": "获得当前文件内容",
            },
        ),
        confidence=confidence,
        ambiguity=ambiguity,
        provenance="p2-9-canary.provider",
        tick=runtime.model.tick,
    )


def _runtime(seed: int, checkpoint_path: Path) -> SeedRuntime:
    runtime = SeedRuntime(
        Seed(episode_id=f"p2-9-semantic-grounding-{seed}"),
        checkpoint_path=checkpoint_path,
    )
    runtime._workbench_environment = WorkbenchEnvironment(PROJECT_ROOT)
    return runtime


def _run_success(seed: int) -> dict[str, object]:
    checkpoint_path = CHECKPOINT_DIR / f".p2-9-semantic-grounding-{seed}.pt"
    checkpoint_path.unlink(missing_ok=True)
    prompt = "请读取 README.md"
    with patch(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(PROJECT_ROOT)
        if key == "workspace_path"
        else default,
    ):
        runtime = _runtime(seed, checkpoint_path)
        proposal = _proposal(runtime, prompt)
        result = runtime.execute_natural_language_workbench_task(
            prompt,
            proposal,
            snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
            loop_id=f"p2-9-semantic-grounding-loop-{seed}",
            max_steps=1,
            max_budget_units=1.0,
            resource_budget=0.8,
        )
        checkpoint_saved = checkpoint_path.is_file()
        restored = SeedRuntime.load(checkpoint_path)
        restored._workbench_environment = WorkbenchEnvironment(PROJECT_ROOT)
        restored_status = restored.status()
        checkpoint_path.unlink(missing_ok=True)
    action_intent = result["planning"]["action_intents"][0]
    planning_step = result["planning"]["steps"][0]
    step = result["execution"]["steps"][0]
    return {
        "seed": seed,
        "status": result["status"],
        "selected_kind": action_intent["kind"],
        "selected_source_goal_id": action_intent["source_goal_id"],
        "goal_id": result["goal"]["goal_id"],
        "grounding_source": planning_step["grounding_source"],
        "grounding_count": len(planning_step["grounding"]),
        "execution_status": result["execution"]["status"],
        "step_success": step["success"],
        "checkpoint_saved": checkpoint_saved,
        "restored_checkpoint_name": restored_status["name"],
        "workbench_event_count": len(runtime.workbench_audit.events),
        "provider_has_no_final_binding": not any(
            key in result["provider_evidence"]
            for key in (
                "action",
                "capability_id",
                "intent",
                "parameters",
                "parameter_binding",
                "parameter_bindings",
                "tool",
                "tool_id",
            )
        ),
    }


def evaluate() -> dict[str, object]:
    runs = [_run_success(seed) for seed in LEARNER_SEEDS]
    unresolved_checkpoint = CHECKPOINT_DIR / ".p2-9-semantic-grounding-unresolved.pt"
    unresolved_checkpoint.unlink(missing_ok=True)
    with patch(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(PROJECT_ROOT)
        if key == "workspace_path"
        else default,
    ):
        unresolved_runtime = _runtime(101, unresolved_checkpoint)
        unresolved_prompt = "请处理 README.md"
        unresolved = unresolved_runtime.execute_natural_language_workbench_task(
            unresolved_prompt,
            _proposal(unresolved_runtime, unresolved_prompt, operation="unsupported"),
            snapshot_id=unresolved_runtime.workbench_environment.capability_snapshot.snapshot_id,
            loop_id="p2-9-unresolved",
        )
        unresolved_stops_before_workbench = (
            unresolved["status"] == "needs_clarification"
            and unresolved["reason_code"] == "semantic_grounding_unresolved"
            and unresolved["execution"]["side_effects"] is False
            and unresolved_runtime.workbench_audit.events == ()
        )
        try:
            _proposal(unresolved_runtime, unresolved_prompt, forbidden=True)
        except ValueError as exc:
            forbidden_rejected = "execution field" in str(exc)
        else:
            forbidden_rejected = False
    unresolved_checkpoint.unlink(missing_ok=True)

    metrics = {
        "three_independent_seeds": len(runs) == len(LEARNER_SEEDS),
        "taiji_contract_grounding_used": all(
            item["grounding_source"] == "taiji-semantic-contract"
            and item["grounding_count"] == 1
            for item in runs
        ),
        "taiji_creates_intent_without_external_binding": all(
            item["selected_kind"] == "workspace.read"
            and item["selected_source_goal_id"] == item["goal_id"]
            for item in runs
        ),
        "bounded_workbench_execution_succeeds": all(
            item["status"] == "completed"
            and item["execution_status"] == "completed"
            and item["step_success"]
            and item["workbench_event_count"] > 0
            for item in runs
        ),
        "checkpoint_saves_and_restores": all(
            item["checkpoint_saved"]
            and item["restored_checkpoint_name"]
            == f"seed:.p2-9-semantic-grounding-{item['seed']}.pt"
            for item in runs
        ),
        "provider_cannot_supply_final_binding": all(
            item["provider_has_no_final_binding"] for item in runs
        ),
        "unresolved_grounding_stops_before_workbench": unresolved_stops_before_workbench,
        "provider_execution_fields_fail_closed": forbidden_rejected,
    }
    return {
        "format": REPORT_FORMAT,
        "task": "Taiji-owned semantic-slot grounding without external parameter bindings",
        "runs": runs,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "Taiji must select a unique live capability and derive its execution parameters "
                "from registered semantic contracts without caller parameter bindings; "
                "unresolved or executable provider evidence must stop before Workbench mutation"
            ),
        },
        "boundary": (
            "This Gate validates declarative grounding for one read-only semantic task. It does "
            "not claim real provider quality, broad language coverage, multi-step autonomy, "
            "write/terminal parameter synthesis, CUDA, CI, or open-domain self-evolution."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p2_9_semantic_grounding_20260831.json",
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
