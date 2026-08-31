"""Evaluate the Taiji-owned IDE language evidence and execution chain."""

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
from taiji import ActionIntent, SemanticEvidenceProposal  # noqa: E402

REPORT_FORMAT = "taiji-w7-p2-11-ide-language-chain-v1"
TARGET_PATH = "api/app.py"
AMBIGUOUS_PATH = "direct-workbench-r5b-20260830-b/test-37/shared.h"


def _runtime(seed: int, checkpoint_path: Path) -> SeedRuntime:
    runtime = SeedRuntime(
        Seed(episode_id=f"p2-11-ide-language-chain-{seed}"),
        checkpoint_path=checkpoint_path,
    )
    runtime._workbench_environment = WorkbenchEnvironment(PROJECT_ROOT)
    return runtime


def _proposal(
    runtime: SeedRuntime,
    prompt: str,
    path: str,
    *,
    include_language_chain: bool = True,
) -> SemanticEvidenceProposal:
    _, frame = runtime._task_frame(prompt)
    steps = [
        {
            "description": "读取当前工作区文件",
            "semantic_slots": {"operation": "read", "path": path},
            "expected_outcome": "获得当前文件内容证据",
        },
    ]
    if include_language_chain:
        steps.extend(
            (
                {
                    "description": "解析当前文件的编程语言",
                    "semantic_slots": {
                        "operation": "resolve-language",
                        "path": path,
                    },
                    "expected_outcome": "获得当前文件语言证据",
                },
                {
                    "description": "按当前文件证据切换编辑器语言",
                    "semantic_slots": {
                        "operation": "set-language",
                        "path": path,
                        "user_override": False,
                    },
                    "expected_outcome": "编辑器选择与当前文件语言一致",
                },
            )
        )
    return SemanticEvidenceProposal.from_frame(
        frame,
        provider_id="deterministic-p2-11-canary",
        goal_description=prompt,
        semantic_steps=tuple(steps),
        confidence=0.95,
        ambiguity=0.05,
        provenance="p2-11-canary.provider",
        tick=runtime.model.tick,
    )


def _run_success(seed: int) -> dict[str, object]:
    checkpoint_path = PROJECT_ROOT / "checkpoints" / f".p2-11-language-chain-{seed}.pt"
    checkpoint_path.unlink(missing_ok=True)
    prompt = "请读取 api/app.py，识别语言并同步编辑器语言"
    with patch(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(PROJECT_ROOT)
        if key == "workspace_path"
        else default,
    ):
        runtime = _runtime(seed, checkpoint_path)
        result = runtime.execute_natural_language_workbench_task(
            prompt,
            _proposal(runtime, prompt, TARGET_PATH),
            snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id,
            loop_id=f"p2-11-language-chain-loop-{seed}",
            max_steps=3,
            max_budget_units=3.0,
            resource_budget=0.8,
        )
        checkpoint_saved = checkpoint_path.is_file()
        restored = SeedRuntime.load(checkpoint_path)
        restored_language = next(
            (
                item
                for item in restored.workbench_environment.language_state_checkpoint()[
                    "selections"
                ]
                if item.get("path") == TARGET_PATH
            ),
            {},
        )
        recovery_prompt = "恢复后再次读取 api/app.py"
        recovery = restored.execute_natural_language_workbench_task(
            recovery_prompt,
            _proposal(
                restored,
                recovery_prompt,
                TARGET_PATH,
                include_language_chain=False,
            ),
            snapshot_id=restored.workbench_environment.capability_snapshot.snapshot_id,
            loop_id=f"p2-11-language-chain-recovery-{seed}",
            max_steps=1,
            max_budget_units=1.0,
            resource_budget=0.8,
        )
        checkpoint_path.unlink(missing_ok=True)

    planning_steps = result["planning"]["steps"]
    execution_steps = result["execution"]["steps"]
    provider_steps = result["provider_evidence"]["semantic_steps"]
    language_step = planning_steps[2]
    language_execution = execution_steps[2]
    language_result = language_execution["outcome"]["result"]
    language_intent = result["planning"]["action_intents"][2]
    language_intent_parameters = language_intent["parameters"]["value"]
    return {
        "seed": seed,
        "status": result["status"],
        "planning_kinds": [item["grounding"][0]["action_kind"] for item in planning_steps],
        "grounding_sources": [item["grounding_source"] for item in planning_steps],
        "provider_set_step_has_no_final_language_id": (
            "programming_language_id" not in provider_steps[2]["semantic_slots"]
        ),
        "taiji_derived_language_id": language_intent_parameters.get(
            "programming_language_id"
        ),
        "language_evidence_digest": language_step["language_evidence"]["file_digest"],
        "language_result_digest": language_result["file_digest"],
        "language_result_state": language_result["selection_state"],
        "execution_status": result["execution"]["status"],
        "step_successes": [item["success"] for item in execution_steps],
        "checkpoint_saved": checkpoint_saved,
        "restored_language_state": restored_language.get("selection_state"),
        "recovery_status": recovery["status"],
        "recovery_step_success": recovery["execution"]["steps"][0]["success"],
    }


def _run_user_override_and_ambiguity() -> dict[str, object]:
    checkpoint_path = PROJECT_ROOT / "checkpoints" / ".p2-11-language-policy.pt"
    checkpoint_path.unlink(missing_ok=True)
    prompt = "请识别并同步 api/app.py 的编辑器语言"
    with patch(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(PROJECT_ROOT)
        if key == "workspace_path"
        else default,
    ):
        runtime = _runtime(101, checkpoint_path)
        snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
        override = runtime.execute_workbench_intent(
            ActionIntent(
                intent_id="p2-11-user-override",
                kind="editor.set_language",
                parameters={
                    "path": TARGET_PATH,
                    "programming_language_id": "python",
                    "user_override": True,
                },
                confidence=0.1,
                tick=runtime.model.tick,
            ),
            snapshot_id=snapshot_id,
            learn=False,
        )
        blocked = runtime.execute_natural_language_workbench_task(
            prompt,
            _proposal(runtime, prompt, TARGET_PATH),
            snapshot_id=snapshot_id,
            loop_id="p2-11-user-override-loop",
            max_steps=3,
            max_budget_units=3.0,
        )
        override_state = runtime.workbench_environment.language_state_checkpoint()["selections"]
        checkpoint_path.unlink(missing_ok=True)

    ambiguous_checkpoint = PROJECT_ROOT / "checkpoints" / ".p2-11-ambiguous-language.pt"
    ambiguous_checkpoint.unlink(missing_ok=True)
    with patch(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(PROJECT_ROOT)
        if key == "workspace_path"
        else default,
    ):
        ambiguous_runtime = _runtime(103, ambiguous_checkpoint)
        ambiguous_prompt = "请识别并同步 shared.h 的编辑器语言"
        ambiguous = ambiguous_runtime.execute_natural_language_workbench_task(
            ambiguous_prompt,
            _proposal(ambiguous_runtime, ambiguous_prompt, AMBIGUOUS_PATH),
            snapshot_id=ambiguous_runtime.workbench_environment.capability_snapshot.snapshot_id,
            loop_id="p2-11-ambiguous-language-loop",
            max_steps=3,
            max_budget_units=3.0,
        )
    ambiguous_checkpoint.unlink(missing_ok=True)

    selected_override = next(
        item for item in override_state if item.get("path") == TARGET_PATH
    )
    return {
        "override_setup_status": override["outcome"]["status"],
        "override_setup_state": selected_override["selection_state"],
        "override_blocked_status": blocked["status"],
        "override_blocked_reason": blocked["reason_code"],
        "override_blocked_has_no_execution": blocked["execution"]["side_effects"] is False,
        "ambiguous_status": ambiguous["status"],
        "ambiguous_reason": ambiguous["reason_code"],
        "ambiguous_has_no_action_intents": "action_intents" not in ambiguous["planning"],
        "ambiguous_has_no_workbench_events": ambiguous_runtime.workbench_audit.events == (),
    }


def evaluate() -> dict[str, object]:
    runs = [_run_success(seed) for seed in (11, 29, 47)]
    policy = _run_user_override_and_ambiguity()
    metrics = {
        "three_independent_seeds": len(runs) == 3,
        "provider_submits_semantics_not_final_language_binding": all(
            item["provider_set_step_has_no_final_language_id"] for item in runs
        ),
        "taiji_owns_three_step_language_chain": all(
            item["planning_kinds"]
            == [
                "workspace.read",
                "workspace.programming_language.resolve",
                "editor.set_language",
            ]
            and item["grounding_sources"]
            == [
                "taiji-semantic-contract",
                "taiji-semantic-contract",
                "taiji-semantic-contract+workbench-language-evidence",
            ]
            for item in runs
        ),
        "language_id_comes_from_live_workbench_evidence": all(
            item["taiji_derived_language_id"] == "python"
            and item["language_evidence_digest"] == item["language_result_digest"]
            for item in runs
        ),
        "three_step_execution_and_outcome_succeed": all(
            item["status"] == "completed"
            and item["execution_status"] == "completed"
            and item["step_successes"] == [True, True, True]
            and item["language_result_state"] == "taiji_selection"
            for item in runs
        ),
        "language_selection_checkpoint_and_recovery": all(
            item["checkpoint_saved"]
            and item["restored_language_state"] == "taiji_selection"
            and item["recovery_status"] == "completed"
            and item["recovery_step_success"]
            for item in runs
        ),
        "user_override_has_priority_before_new_action_intent": (
            policy["override_setup_status"] == "success"
            and policy["override_setup_state"] == "user_override"
            and policy["override_blocked_status"] == "needs_clarification"
            and policy["override_blocked_reason"] == "user_override_has_priority"
            and policy["override_blocked_has_no_execution"]
        ),
        "ambiguous_language_stops_before_execution": (
            policy["ambiguous_status"] == "needs_clarification"
            and policy["ambiguous_reason"] == "language_evidence_ambiguous"
            and policy["ambiguous_has_no_action_intents"]
            and policy["ambiguous_has_no_workbench_events"]
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "task": "Taiji-owned IDE read, language evidence, and editor selection chain",
        "runs": runs,
        "policy": policy,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "provider semantic evidence must let Taiji ground read -> language resolve "
                "-> editor selection from live file evidence, checkpoint the result, recover "
                "after restart, and stop before ActionIntent on override or ambiguity"
            ),
        },
        "boundary": (
            "This Gate proves one bounded IDE language chain on deterministic CPU evidence. "
            "It does not claim broad language understanding, real provider quality, CUDA, "
            "CI, unrestricted IDE autonomy, or AGI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p2_11_ide_language_chain_20260831.json",
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
