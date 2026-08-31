"""Evaluate multi-step Taiji-owned grounding and Workbench recovery."""

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

REPORT_FORMAT = "taiji-w7-p2-10-multistep-grounding-recovery-v1"
LEARNER_SEEDS = (11, 29, 47)
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def _proposal(
    runtime: SeedRuntime,
    prompt: str,
    steps: tuple[tuple[str, str], ...],
) -> SemanticEvidenceProposal:
    _, frame = runtime._task_frame(prompt)
    return SemanticEvidenceProposal.from_frame(
        frame,
        provider_id="deterministic-p2-10-canary",
        goal_description=prompt,
        semantic_steps=tuple(
            {
                "description": f"执行语义操作：{operation}",
                "semantic_slots": {"operation": operation, "path": path},
                "expected_outcome": "保留可审计的工作区证据",
            }
            for operation, path in steps
        ),
        confidence=0.95,
        ambiguity=0.05,
        provenance="p2-10-canary.provider",
        tick=runtime.model.tick,
    )


def _runtime(seed: int, checkpoint_path: Path) -> SeedRuntime:
    runtime = SeedRuntime(
        Seed(episode_id=f"p2-10-multistep-grounding-{seed}"),
        checkpoint_path=checkpoint_path,
    )
    runtime._workbench_environment = WorkbenchEnvironment(PROJECT_ROOT)
    return runtime


def _run_success(seed: int) -> dict[str, object]:
    checkpoint_path = CHECKPOINT_DIR / f".p2-10-multistep-grounding-{seed}.pt"
    checkpoint_path.unlink(missing_ok=True)
    prompt = "请读取并检查 README.md"
    steps = (("read", "README.md"), ("stat", "README.md"))
    with patch(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(PROJECT_ROOT)
        if key == "workspace_path"
        else default,
    ):
        runtime = _runtime(seed, checkpoint_path)
        result = runtime.execute_natural_language_workbench_task(
            prompt,
            _proposal(runtime, prompt, steps),
            snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
            loop_id=f"p2-10-multistep-loop-{seed}",
            max_steps=2,
            max_budget_units=2.0,
            resource_budget=0.8,
        )
        checkpoint_saved = checkpoint_path.is_file()
        step_payloads = result["planning"]["steps"]
        restored = SeedRuntime.load(checkpoint_path)
        restored_status = restored.status()
        checkpoint_path.unlink(missing_ok=True)
    return {
        "seed": seed,
        "status": result["status"],
        "step_count": len(step_payloads),
        "grounding_sources": [item["grounding_source"] for item in step_payloads],
        "grounding_kinds": [item["grounding"][0]["action_kind"] for item in step_payloads],
        "execution_status": result["execution"]["status"],
        "completed_prefix": result["execution"]["completed_prefix"],
        "step_successes": [item["success"] for item in result["execution"]["steps"]],
        "checkpoint_saved": checkpoint_saved,
        "restored_checkpoint_name": restored_status["name"],
        "workbench_event_count": len(runtime.workbench_audit.events),
    }


def _run_failure_and_recovery() -> dict[str, object]:
    checkpoint_path = CHECKPOINT_DIR / ".p2-10-multistep-recovery.pt"
    checkpoint_path.unlink(missing_ok=True)
    failed_prompt = "请读取并检查缺失文件"
    failed_steps = (("read", "missing/p2-10.md"), ("stat", "missing/p2-10.md"))
    recovered_prompt = "请读取 README.md"
    recovered_steps = (("read", "README.md"),)
    with patch(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(PROJECT_ROOT)
        if key == "workspace_path"
        else default,
    ):
        runtime = _runtime(101, checkpoint_path)
        failed = runtime.execute_natural_language_workbench_task(
            failed_prompt,
            _proposal(runtime, failed_prompt, failed_steps),
            snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
            loop_id="p2-10-failure-loop",
            max_steps=2,
            max_budget_units=2.0,
            resource_budget=0.8,
        )
        checkpoint_saved = checkpoint_path.is_file()
        restored = SeedRuntime.load(checkpoint_path)
        restored._workbench_environment = WorkbenchEnvironment(PROJECT_ROOT)
        recovered = restored.execute_natural_language_workbench_task(
            recovered_prompt,
            _proposal(restored, recovered_prompt, recovered_steps),
            snapshot_id=restored.workbench_environment.capability_snapshot.snapshot_id,
            loop_id="p2-10-fresh-recovery-loop",
            max_steps=1,
            max_budget_units=1.0,
            resource_budget=0.8,
        )
        checkpoint_path.unlink(missing_ok=True)
    return {
        "failure_status": failed["execution"]["status"],
        "failure_completed_prefix": failed["execution"]["completed_prefix"],
        "failure_stopped_at": failed["execution"].get("stopped_at"),
        "checkpoint_saved_after_failure": checkpoint_saved,
        "fresh_recovery_status": recovered["status"],
        "fresh_recovery_execution": recovered["execution"]["status"],
        "fresh_recovery_success": recovered["execution"]["steps"][0]["success"],
    }


def evaluate() -> dict[str, object]:
    runs = [_run_success(seed) for seed in LEARNER_SEEDS]
    recovery = _run_failure_and_recovery()
    metrics = {
        "three_independent_seeds": len(runs) == len(LEARNER_SEEDS),
        "each_semantic_step_is_grounded_by_taiji": all(
            item["step_count"] == 2
            and item["grounding_sources"] == [
                "taiji-semantic-contract",
                "taiji-semantic-contract",
            ]
            and item["grounding_kinds"] == ["workspace.read", "workspace.stat"]
            for item in runs
        ),
        "multi_step_execution_succeeds": all(
            item["status"] == "completed"
            and item["execution_status"] == "completed"
            and item["completed_prefix"] == 2
            and item["step_successes"] == [True, True]
            and item["workbench_event_count"] > 0
            for item in runs
        ),
        "checkpoint_roundtrip_after_multistep": all(
            item["checkpoint_saved"]
            and item["restored_checkpoint_name"]
            == f"seed:.p2-10-multistep-grounding-{item['seed']}.pt"
            for item in runs
        ),
        "failure_stops_at_first_step": (
            recovery["failure_status"] == "failed"
            and recovery["failure_completed_prefix"] == 0
            and recovery["failure_stopped_at"] == 0
        ),
        "failure_checkpoint_and_fresh_recovery": (
            recovery["checkpoint_saved_after_failure"]
            and recovery["fresh_recovery_status"] == "completed"
            and recovery["fresh_recovery_execution"] == "completed"
            and recovery["fresh_recovery_success"]
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "task": "Taiji-owned multi-step semantic grounding and checkpoint recovery",
        "runs": runs,
        "recovery": recovery,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "Taiji must independently ground and execute at least two semantic steps "
                "without external parameter bindings, checkpoint the sequence, stop at the "
                "first failure, and accept a fresh recovered request"
            ),
        },
        "boundary": (
            "This Gate covers two read-only semantic operations. It does not claim broad "
            "language understanding, write/terminal parameter synthesis, real provider quality, "
            "CUDA, CI, or open-domain self-evolution."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_p2_10_multistep_grounding_recovery_20260831.json",
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
