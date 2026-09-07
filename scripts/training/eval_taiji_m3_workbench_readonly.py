"""Run the first project-disjoint, read-only Taiji Workbench task Gate.

This Gate exercises the current runtime against small real project trees.  It
does not train a language provider and it does not grant write, terminal, or
MCP authority.  The provider submits only semantic operations; live file and
language evidence is derived by Taiji/Seed at the Workbench boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "m3_workbench_projects"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import Seed  # noqa: E402
from seed_platform.programming_languages import (  # noqa: E402
    ProgrammingLanguageRegistry,
)
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import ActionIntent, SemanticEvidenceProposal  # noqa: E402

REPORT_FORMAT = "taiji-m3-workbench-readonly-v1"
REPORT_VERSION = 1


def _workspace_digest(root: Path) -> str:
    entries: list[dict[str, str]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "digest": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return hashlib.sha256(
        json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _runtime(
    project_id: str,
    root: Path,
    checkpoint_path: Path,
    *,
    registry: ProgrammingLanguageRegistry | None = None,
) -> SeedRuntime:
    runtime = SeedRuntime(
        Seed(episode_id=f"m3-workbench-readonly-{project_id}"),
        checkpoint_path=checkpoint_path,
    )
    runtime._workbench_environment = WorkbenchEnvironment(
        root,
        programming_language_registry=registry,
    )
    return runtime


def _proposal(
    runtime: SeedRuntime,
    prompt: str,
    path: str,
    *,
    include_language: bool = True,
) -> SemanticEvidenceProposal:
    _, frame = runtime._task_frame(prompt)
    steps: list[dict[str, Any]] = [
        {
            "description": "读取当前工作区目标文件",
            "semantic_slots": {"operation": "read", "path": path},
            "expected_outcome": "获得当前文件内容与摘要证据",
        }
    ]
    if include_language:
        steps.append(
            {
                "description": "根据当前文件证据判断编程语言",
                "semantic_slots": {
                    "operation": "resolve-language",
                    "path": path,
                },
                "expected_outcome": "得到有来源的编程语言判断",
            }
        )
    return SemanticEvidenceProposal.from_frame(
        frame,
        provider_id="deterministic-m3-readonly-canary",
        goal_description=prompt,
        semantic_steps=tuple(steps),
        confidence=0.95,
        ambiguity=0.05,
        provenance="m3-readonly-canary.provider",
        tick=runtime.model.tick,
    )


def _missing_toolchain_registry() -> ProgrammingLanguageRegistry:
    """Keep the missing-toolchain case deterministic across host machines."""

    default = ProgrammingLanguageRegistry.default()
    definitions = tuple(
        replace(
            definition,
            toolchain_commands=("m3-toolchain-that-is-not-installed",),
        )
        if definition.language_id == "rust"
        else definition
        for definition in default.definitions
    )
    return ProgrammingLanguageRegistry(definitions)


def _run_positive_case(
    project_id: str,
    relative_root: str,
    path: str,
    expected_language: str,
    *,
    registry: ProgrammingLanguageRegistry | None = None,
) -> dict[str, Any]:
    root = (FIXTURE_ROOT / relative_root).resolve()
    checkpoint_path = PROJECT_ROOT / "checkpoints" / f".m3-workbench-readonly-{project_id}.pt"
    checkpoint_path.unlink(missing_ok=True)
    prompt = f"请读取 {path}，判断编程语言并形成只读工作台结论"
    before_digest = _workspace_digest(root)
    with patch(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(root) if key == "workspace_path" else default,
    ):
        runtime = _runtime(project_id, root, checkpoint_path, registry=registry)
        snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
        proposal = _proposal(runtime, prompt, path)
        result = runtime.execute_natural_language_workbench_task(
            prompt,
            proposal,
            snapshot_id=snapshot_id,
            loop_id=f"m3-readonly-{project_id}",
            max_steps=2,
            max_budget_units=2.0,
            resource_budget=0.8,
        )
        provider_steps = result["provider_evidence"]["semantic_steps"]
        planning_steps = result["planning"]["steps"]
        execution_steps = result["execution"]["steps"]
        resolve_planning = planning_steps[1]
        resolve_execution = execution_steps[1]
        language_evidence = resolve_planning["language_resolution_evidence"]
        language_result = resolve_execution["outcome"]["result"]
        selected = planning_steps[0]["planner"]["decision"]["selected"]
        content_plan = selected["content_plan"]
        checkpoint_saved = checkpoint_path.is_file()
        restored = SeedRuntime.load(checkpoint_path)
        restored_selection = next(
            (
                item
                for item in restored.workbench_environment.language_state_checkpoint()[
                    "selections"
                ]
                if item.get("path") == path
            ),
            {},
        )
        recovery_prompt = f"恢复后再次读取 {path}"
        recovery = restored.execute_natural_language_workbench_task(
            recovery_prompt,
            _proposal(restored, recovery_prompt, path, include_language=False),
            snapshot_id=restored.workbench_environment.capability_snapshot.snapshot_id,
            loop_id=f"m3-readonly-recovery-{project_id}",
            max_steps=1,
            max_budget_units=1.0,
            resource_budget=0.8,
        )
    after_digest = _workspace_digest(root)
    checkpoint_path.unlink(missing_ok=True)
    return {
        "project_id": project_id,
        "project_root": relative_root,
        "path": path,
        "expected_language": expected_language,
        "status": result["status"],
        "goal_present": bool(result.get("goal")),
        "content_plan_present": bool(content_plan),
        "provider_has_no_final_language_id": (
            "programming_language_id" not in provider_steps[1]["semantic_slots"]
        ),
        "language_evidence_language": language_evidence["programming_language_id"],
        "language_result_language": language_result["programming_language_id"],
        "language_result_state": language_result["selection_state"],
        "language_evidence_digest": language_evidence["file_digest"],
        "language_result_digest": language_result["file_digest"],
        "available_for_language": language_result["execution_snapshot"][
            "available_for_language"
        ],
        "planning_sources": [item["grounding_source"] for item in planning_steps],
        "execution_capabilities": [
            item["capability_id"] for item in execution_steps
        ],
        "execution_status": result["execution"]["status"],
        "step_successes": [item["success"] for item in execution_steps],
        "side_effects": result["execution"]["side_effects"],
        "checkpoint_saved": checkpoint_saved,
        "restored_selection_state": restored_selection.get("selection_state"),
        "restored_language_digest": restored_selection.get("file_digest"),
        "recovery_status": recovery["status"],
        "recovery_step_success": recovery["execution"]["steps"][0]["success"],
        "workspace_unchanged": before_digest == after_digest,
    }


def _run_clarification_case(
    case_id: str,
    relative_root: str,
    path: str,
    expected_reason: str,
) -> dict[str, Any]:
    root = (FIXTURE_ROOT / relative_root).resolve()
    checkpoint_path = PROJECT_ROOT / "checkpoints" / f".m3-workbench-readonly-{case_id}.pt"
    checkpoint_path.unlink(missing_ok=True)
    prompt = f"请读取 {path} 并判断编程语言"
    before_digest = _workspace_digest(root)
    with patch(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(root) if key == "workspace_path" else default,
    ):
        runtime = _runtime(case_id, root, checkpoint_path)
        result = runtime.execute_natural_language_workbench_task(
            prompt,
            _proposal(runtime, prompt, path),
            snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
            loop_id=f"m3-readonly-{case_id}",
            max_steps=2,
            max_budget_units=2.0,
            resource_budget=0.8,
        )
        phases = [event.phase for event in runtime.workbench_audit.events]
    after_digest = _workspace_digest(root)
    checkpoint_path.unlink(missing_ok=True)
    return {
        "case_id": case_id,
        "project_root": relative_root,
        "path": path,
        "status": result["status"],
        "reason_code": result.get("reason_code"),
        "expected_reason": expected_reason,
        "has_action_intents": "action_intents" in (result.get("planning") or {}),
        "side_effects": result["execution"]["side_effects"],
        "audit_phases": phases,
        "workspace_unchanged": before_digest == after_digest,
    }


def _run_boundary_controls() -> dict[str, Any]:
    root = (FIXTURE_ROOT / "python_app").resolve()
    runtime = _runtime(
        "boundary-controls",
        root,
        PROJECT_ROOT / "checkpoints" / ".m3-workbench-readonly-boundary.pt",
    )
    snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
    stale = runtime.execute_workbench_intent(
        ActionIntent(
            intent_id="m3-stale-read",
            kind="workspace.read",
            parameters={"path": "src/app.py"},
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        snapshot_id=f"stale:{snapshot_id}",
        learn=False,
    )
    diagnostics = runtime.execute_workbench_intent(
        ActionIntent(
            intent_id="m3-diagnostics-read",
            kind="editor.diagnostics.read",
            parameters={},
            confidence=1.0,
            tick=runtime.model.tick,
        ),
        snapshot_id=snapshot_id,
        learn=False,
    )
    phases = [event.phase for event in runtime.workbench_audit.events]
    checkpoint_path = PROJECT_ROOT / "checkpoints" / ".m3-workbench-readonly-boundary.pt"
    checkpoint_path.unlink(missing_ok=True)
    return {
        "stale_status": stale["outcome"]["status"],
        "stale_reason": stale["policy"]["reason_code"],
        "stale_has_execution_phase": "executing" in phases,
        "diagnostics_status": diagnostics["outcome"]["status"],
        "diagnostics_reason": diagnostics["policy"]["reason_code"],
        "diagnostics_has_execution_phase": (
            diagnostics["outcome"]["status"] == "success"
            or "executing" in phases
        ),
        "diagnostics_descriptor_disabled": not bool(
            runtime.workbench_environment.capability_snapshot.get(
                "editor.diagnostics.read"
            ).enabled
        ),
    }


def evaluate() -> dict[str, Any]:
    projects = [
        _run_positive_case(
            "python",
            "python_app",
            "src/app.py",
            "python",
        ),
        _run_positive_case(
            "typescript",
            "typescript_app",
            "src/index.ts",
            "typescript",
        ),
        _run_positive_case(
            "missing-toolchain",
            "missing_toolchain",
            "src/main.rs",
            "rust",
            registry=_missing_toolchain_registry(),
        ),
    ]
    ambiguous = _run_clarification_case(
        "ambiguous-language",
        "ambiguous_header",
        "src/shared.h",
        "language_evidence_ambiguous",
    )
    invalid = _run_clarification_case(
        "invalid-target",
        "python_app",
        "src/does_not_exist.py",
        "workspace_target_not_found",
    )
    controls = _run_boundary_controls()
    metrics = {
        "project_disjoint_fixture_families": len(
            {item["project_root"] for item in projects}
        )
        == len(projects),
        "readonly_projects_complete": all(
            item["status"] == "completed"
            and item["execution_status"] == "completed"
            and item["step_successes"] == [True, True]
            for item in projects
        ),
        "goal_and_content_plan_are_structured": all(
            item["goal_present"] and item["content_plan_present"] for item in projects
        ),
        "provider_submits_semantics_not_final_language": all(
            item["provider_has_no_final_language_id"] for item in projects
        ),
        "language_is_derived_from_live_evidence": all(
            item["language_evidence_language"] == item["expected_language"]
            and item["language_result_language"] == item["expected_language"]
            and item["language_evidence_digest"] == item["language_result_digest"]
            for item in projects
        ),
        "missing_toolchain_is_observed_without_execution": (
            projects[2]["language_result_language"] == "rust"
            and projects[2]["available_for_language"] == []
            and projects[2]["side_effects"] is False
        ),
        "only_read_capabilities_are_executed": all(
            set(item["execution_capabilities"])
            <= {"workspace.read", "workspace.programming_language.resolve"}
            and item["side_effects"] is False
            for item in projects
        ),
        "checkpoint_and_restart_recover_language_evidence": all(
            item["checkpoint_saved"]
            and item["restored_selection_state"] == "resolved"
            and item["restored_language_digest"] == item["language_result_digest"]
            and item["recovery_status"] == "completed"
            and item["recovery_step_success"]
            for item in projects
        ),
        "ambiguous_language_clarifies_before_execution": (
            ambiguous["status"] == "needs_clarification"
            and ambiguous["reason_code"] == ambiguous["expected_reason"]
            and not ambiguous["has_action_intents"]
            and ambiguous["audit_phases"] == []
            and ambiguous["workspace_unchanged"]
        ),
        "invalid_target_clarifies_before_execution": (
            invalid["status"] == "needs_clarification"
            and invalid["reason_code"] == invalid["expected_reason"]
            and not invalid["has_action_intents"]
            and invalid["audit_phases"] == []
            and invalid["workspace_unchanged"]
        ),
        "stale_snapshot_fails_closed": (
            controls["stale_status"] == "rejected"
            and controls["stale_reason"] == "stale_capability_snapshot"
            and not controls["stale_has_execution_phase"]
        ),
        "diagnostics_capability_fails_closed_without_editor": (
            controls["diagnostics_descriptor_disabled"]
            and controls["diagnostics_status"] == "rejected"
            and controls["diagnostics_reason"] == "capability_not_connected"
            and not controls["diagnostics_has_execution_phase"]
        ),
        "fixture_workspaces_are_unchanged": all(
            item["workspace_unchanged"] for item in projects
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "status": "passed" if all(metrics.values()) else "failed",
        "task": "M3 first project-disjoint Taiji Workbench read-only task",
        "projects": projects,
        "clarification_cases": {"ambiguous_language": ambiguous, "invalid_target": invalid},
        "boundary_controls": controls,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "can_promote": False,
            "criterion": (
                "Taiji must form a structured goal/content plan from provider semantics, "
                "bind file/language facts from the live Workbench, recover after restart, "
                "and clarify before execution on ambiguity, invalid targets, stale snapshots, "
                "or disconnected diagnostics."
            ),
        },
        "boundary": (
            "This is a provider-assisted, native Workbench boundary canary. It does not "
            "prove open-domain language understanding, native semantic learning from raw "
            "text, diagnostics from a connected editor, write/terminal/MCP autonomy, CUDA, "
            "or AGI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m3_workbench_readonly_20260907.json",
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
