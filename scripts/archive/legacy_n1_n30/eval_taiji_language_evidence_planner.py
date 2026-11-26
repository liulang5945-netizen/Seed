"""Run the P2-4 Taiji language-evidence planning Gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import seed_platform.workbench as workbench_module  # noqa: E402
from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import Seed  # noqa: E402
from seed_platform.programming_languages import (  # noqa: E402
    ProgrammingLanguageDefinition,
    ProgrammingLanguageRegistry,
)
from seed_platform.workbench import WorkbenchActionRequest, WorkbenchEnvironment  # noqa: E402
from taiji import InputFrame, TaskInterpretation  # noqa: E402

REPORT_FORMAT = "taiji-w7-p2-language-evidence-planner-v1"


def _frame(input_id: str) -> InputFrame:
    return InputFrame(
        input_id=input_id,
        modality="text",
        payload=b"open api/app.py",
        source="scripts.p2_4",
        provenance="scripts.p2_4",
    )


def _admit_resolved_task(runtime: SeedRuntime, input_id: str) -> None:
    architecture = runtime.model.architecture
    frame = _frame(input_id)
    architecture.ingest_input(frame, learn=False)
    architecture.admit_task_interpretation(
        TaskInterpretation.from_input(
            frame,
            status="resolved",
            confidence=0.9,
            ambiguity=0.1,
            tick=architecture.tick,
        )
    )


def _mapping_parameters(action_payload: dict[str, object]) -> dict[str, object]:
    raw = action_payload.get("parameters", {})
    if not isinstance(raw, dict) or raw.get("kind") != "mapping":
        return {}
    value = raw.get("value", {})
    return dict(value) if isinstance(value, dict) else {}


def evaluate() -> dict[str, object]:
    with patch.object(workbench_module, "default_workspace_root", lambda: PROJECT_ROOT):
        runtime = SeedRuntime(Seed(episode_id="p2-4-language-evidence"))
        _admit_resolved_task(runtime, "p2-4-language-task")
        snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
        planned = runtime.plan_language_selection(
            snapshot_id=snapshot_id,
            path="api/app.py",
            resource_budget=0.8,
        )
        action_payload = planned["execution"]["action_intent"] or {}
        action_parameters = _mapping_parameters(action_payload)

        override_runtime = SeedRuntime(Seed(episode_id="p2-4-user-override"))
        override_environment = override_runtime.workbench_environment
        override_snapshot = override_environment.capability_snapshot.snapshot_id
        override_request = WorkbenchActionRequest(
            request_id="p2-4-user-override-request",
            intent_id="p2-4-user-override-intent",
            capability_id="editor.set_language",
            parameters={
                "path": "api/app.py",
                "programming_language_id": "javascript",
                "user_override": True,
            },
            snapshot_id=override_snapshot,
            confidence=1.0,
        )
        override_policy = override_environment.policy_for(override_request)
        override_outcome = override_environment.execute_tool(
            "editor.set_language",
            override_request.parameters,
        )
        _admit_resolved_task(override_runtime, "p2-4-override-task")
        override_planned = override_runtime.plan_language_selection(
            snapshot_id=override_runtime.workbench_environment.capability_snapshot.snapshot_id,
            path="api/app.py",
            resource_budget=0.8,
        )

        ambiguous_registry = ProgrammingLanguageRegistry(
            (
                ProgrammingLanguageDefinition("alpha", "Alpha", "alpha", extensions=(".py",)),
                ProgrammingLanguageDefinition("beta", "Beta", "beta", extensions=(".py",)),
                ProgrammingLanguageDefinition("plaintext", "Plain text", "plaintext"),
            )
        )
        ambiguous_runtime = SeedRuntime(Seed(episode_id="p2-4-ambiguous"))
        ambiguous_runtime._workbench_environment = WorkbenchEnvironment(
            PROJECT_ROOT,
            programming_language_registry=ambiguous_registry,
        )
        _admit_resolved_task(ambiguous_runtime, "p2-4-ambiguous-task")
        ambiguous = ambiguous_runtime.plan_language_selection(
            snapshot_id=ambiguous_runtime.workbench_environment.capability_snapshot.snapshot_id,
            path="api/app.py",
            resource_budget=0.8,
        )

    metrics = {
        "language_evidence_grounds_taiji_action_intent": (
            planned["assessment"]["programming_language_id"] == "python"
            and planned["assessment"]["selection_state"] == "resolved"
            and planned["planner"]["status"] == "planned"
            and action_payload.get("kind") == "editor.set_language"
            and action_parameters.get("programming_language_id") == "python"
        ),
        "planning_does_not_execute_or_write": (
            planned["execution"]["status"] == "not_executed"
            and planned["execution"]["side_effects"] is False
            and runtime.workbench_audit.events == ()
        ),
        "user_override_has_priority": (
            override_policy.decision == "allow"
            and override_outcome.success
            and override_planned["assessment"]["selection_state"] == "user_override"
            and override_planned["planner"]["status"] == "already_selected"
            and override_planned["execution"]["action_intent"] is None
        ),
        "ambiguous_language_stops_before_action_intent": (
            ambiguous["assessment"]["selection_state"] == "ambiguous"
            and ambiguous["planner"]["status"] == "needs_clarification"
            and ambiguous["planner"]["reason_code"] == "language_evidence_ambiguous"
            and ambiguous["execution"]["action_intent"] is None
            and ambiguous_runtime.workbench_audit.events == ()
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "live Workbench language evidence is assessed before Taiji planning; only "
                "resolved evidence produces a non-executing editor.set_language intent, "
                "explicit user override wins, and ambiguous evidence asks"
            ),
        },
        "gap": {
            "current": (
                "language evidence can ground a proposed editor language intent, but the "
                "intent is not yet admitted and executed through the reversible UI policy"
            ),
            "next": (
                "run a real reversible IDE canary: preview, policy admission, execute, "
                "verify language state, checkpoint, and undo"
            ),
        },
        "boundary": (
            "This Gate does not claim natural-language semantic resolution, file editing, "
            "provider quality, CUDA, CI, open-domain gains, or AGI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p2_language_evidence_planner_20260831.json",
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
