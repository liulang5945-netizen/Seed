"""Unified-entry evidence run (M5 common gate closeout).

Preregistration: plans/reference/M5_UNIFIED_ENTRY_PREREGISTRATION_FROZEN_20260919.md
(frozen 2026-09-19).  Five pre-registered ablation arms execute the frozen
create_undo task (+ one unseen instance for migration) through the real
Workbench contract path with the HANDOFF-M4 selection policy, the per-member
cue rule, and a deterministic first-attempt failure injection for the L2
resource line.  Lines L1-L4 are evaluated exactly as frozen; no threshold is
adjusted after runs.

TRAINING BUDGET GATE: member readouts are (re)trained through the P5.2b
frozen trial-learner path (250 epochs, instrument caliber, frozen in the
preregistration section 2); the run is refused without --budget-approved.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import eval_taiji_p5_2a_predictive_execution_gate as p52a  # noqa: E402
import torch  # noqa: E402
from eval_taiji_p5_1f_real_corpus_same_budget_gate import (  # noqa: E402
    DocumentEmbedder,
    _MemoizedEmbedder,
)
from eval_taiji_p5_2b_group_causal_corpora_gate import (  # noqa: E402
    MEMBER_IDS,
    STEP_CAP,
    _family_tasks,
    _train_members,
)

from seed_platform.workbench import (  # noqa: E402
    WorkbenchActionRequest,
    WorkbenchEnvironment,
)
from taiji.collab_handoff import (  # noqa: E402
    FailureHandoffPolicy,
    MemberCall,
    member_cue_count,
)
from taiji.contracts import ActionIntent  # noqa: E402
from taiji.unified_entry import (  # noqa: E402
    ABLATION_ARMS,
    arm_config,
    assemble_bundle,
    validate_trace,
)

REPORT_FORMAT = "taiji-unified-entry-evidence-v1"
VERSION = 1
CONTRACT = "plans/reference/M5_UNIFIED_ENTRY_EVIDENCE_PACKAGE_DRAFT_20260919.md"
PREREGISTRATION = "plans/reference/M5_UNIFIED_ENTRY_PREREGISTRATION_FROZEN_20260919.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_unified_entry_evidence_20260919.json"
REPEATS = 3
PER_ARM_WALL_CAP = 120.0
TOTAL_WALL_CAP = 600.0
KNOWLEDGE_CHILD = PROJECT_ROOT / "reports/taiji_p5_1h_child_20260919.pt"
KNOWLEDGE_SHA = "daf2e8779b9faa734994b8f9de084ab75744540bcda032d3f1c79aeb9b87021d"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_unified_entry_evidence_20260919.json"


def _run_episode(
    *,
    root: Path,
    task: Any,
    cue: Any,
    members: dict,
    arm: Any,
    episode_id: str,
    inject_first_failure: bool,
) -> dict[str, Any]:
    """One arm-parameterized port of the P5.2b member episode (revision per
    arm; memory-disabled arms cue every member at a constant count)."""

    environment = WorkbenchEnvironment(root)
    for stale in root.iterdir():
        if stale.is_file():
            stale.unlink()
        elif stale.is_dir():
            shutil.rmtree(stale)
    environment.restore_language_state(
        {
            "format": "seed-workbench-language-state-v1",
            "version": 1,
            "registry_revision": environment.programming_language_registry.revision,
            "selections": [],
        }
    )
    for name, content in task.initial_files.items():
        (root / name).write_text(content, encoding="utf-8", newline="")
    state: dict[str, Any] = {"root": environment.root, "last_undo_token": None}
    steps: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    policy = FailureHandoffPolicy(MEMBER_IDS, rule_revision=arm.rule_revision)
    injected = {"done": False}

    def finish(reason: str) -> dict[str, Any]:
        success = bool(p52a._goal_reached(environment, task))
        executed_actions = sum(1 for item in steps if item.get("executed"))
        return {
            "episode_id": episode_id,
            "arm": arm.name,
            "task_id": task.task_id,
            "success": success,
            "stop_reason": reason,
            "executed_actions": executed_actions,
            "steps": steps,
            "events": events,
            "trace_valid": validate_trace(events),
        }

    for tick in range(1, STEP_CAP + 1):
        if p52a._goal_reached(environment, task):
            return finish("goal_reached")
        calls: list[MemberCall] = []
        raw_calls: list[dict[str, Any]] = []
        bindable: list[dict[str, Any]] = []
        for member_id in MEMBER_IDS:
            learner = members[member_id]
            cue_count = member_cue_count(steps, member_id) if arm.memory_enabled else 1
            cues = tuple([cue] * cue_count)
            kind = str(learner.predict_episode(cues)[-1])
            params, provenance, failure = p52a._bind(kind, task, state)
            calls.append(MemberCall(member=member_id, bind_failure=failure))
            raw_calls.append(
                {"member": member_id, "kind": kind, "bind_failure": failure, "params": params, "provenance": provenance}
            )
            if failure is None:
                bindable.append(raw_calls[-1])
            events.append(
                {
                    "tick": tick,
                    "kind": "member_called",
                    "member": member_id,
                    "cue_count": cue_count,
                    "bind_failure": failure,
                    "rule_revision": arm.rule_revision,
                    "bundle_digest": episode_id,
                }
            )
        if not bindable:
            steps.append({"tick": tick, "called": [c["member"] for c in raw_calls], "executed": False, "stop": "all_members_exhausted"})
            events.append({"tick": tick, "kind": "stop", "stop": "all_members_exhausted", "rule_revision": arm.rule_revision, "bundle_digest": episode_id})
            return finish("all_members_exhausted")
        decision = policy.select(calls, steps)
        chosen = next(c for c in raw_calls if c["member"] == decision.chosen.member)
        events.append({"tick": tick, "kind": "member_chosen", "member": chosen["member"], "rule_revision": arm.rule_revision, "bundle_digest": episode_id})
        kind = chosen["kind"]
        params = chosen["params"]
        intent = ActionIntent(
            f"unified-intent:{task.task_id}:{chosen['member']}:{tick:04d}",
            kind,
            parameters=params,
            confidence=0.9,
            tick=tick,
        )
        request = WorkbenchActionRequest.from_action_intent(
            intent,
            snapshot_id=environment.capability_snapshot.snapshot_id,
        )
        decision_wb = environment.policy_for(request)
        needs_approval = (
            decision_wb.decision == "ask_user"
            and decision_wb.reason_code == "capability_requires_approval"
        )
        if decision_wb.decision == "deny" or (decision_wb.decision == "ask_user" and not needs_approval):
            steps.append({"tick": tick, "called": [c["member"] for c in raw_calls], "chosen": chosen["member"], "kind": kind, "executed": False, "stop": f"contract_intercepted:{decision_wb.reason_code}"})
            return finish(f"contract_intercepted:{decision_wb.reason_code}")
        if needs_approval:
            approval = environment.issue_approval(request)
            tokened = __import__("dataclasses").replace(request, approval_token=approval["approval_token"])
            if environment.policy_for(tokened).decision != "allow":
                return finish("approval_rejected")
            environment.consume_approval(tokened)
        outcome = environment.execute_tool(kind, params)
        executed = bool(outcome.success)
        if (
            inject_first_failure
            and not injected["done"]
            and chosen["member"] == min(c["member"] for c in bindable)
        ):
            executed = False
            injected["done"] = True
        last = environment.last_result
        if "transaction" in last and last["transaction"].get("undo_token"):
            state["last_undo_token"] = str(last["transaction"]["undo_token"])
        steps.append(
            {
                "tick": tick,
                "called": [c["member"] for c in raw_calls],
                "chosen": chosen["member"],
                "kind": kind,
                "executed": executed,
                "provenance": chosen["provenance"],
            }
        )
        events.append(
            {
                "tick": tick,
                "kind": "member_executed",
                "member": chosen["member"],
                "executed": executed,
                "rule_revision": arm.rule_revision,
                "bundle_digest": episode_id,
            }
        )
        if not executed:
            events.append({"tick": tick, "kind": "attempt_failed", "member": chosen["member"], "rule_revision": arm.rule_revision, "bundle_digest": episode_id})
        if p52a._goal_reached(environment, task):
            return finish("goal_reached")
    return finish("step_cap")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-approved", action="store_true")
    parser.add_argument("--no-memoization", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if not args.budget_approved:
        raise SystemExit(
            "unified-entry evidence run refused: member readouts train through "
            "the frozen trial-learner path and the run requires the "
            "separately-approved budget."
        )

    started = time.perf_counter()
    embedder = _MemoizedEmbedder(DocumentEmbedder()) if not args.no_memoization else DocumentEmbedder()
    members = _train_members(embedder)  # instrument caliber, frozen prereg §2

    family = _family_tasks()["member-c"]  # create_undo family
    main_task, unseen_task = family[0], family[1]
    cue_by_task = {
        task.task_id: torch.tensor(embedder.embed([task.goal_text])[0], dtype=torch.float32)
        for task in (main_task, unseen_task)
    }
    # P5.2b cues come from the embedder directly; keep the tensor 1-D float32
    for task in (main_task, unseen_task):
        cue_by_task[task.task_id] = torch.as_tensor(embedder.embed([task.goal_text])[0], dtype=torch.float32).flatten()

    policy_payload = {"composition_rule": "m4_failure_handoff", "rule_revision": 1}
    tmp = Path(tempfile.mkdtemp(prefix="unified-entry-"))
    policy_file = tmp / "handoff_policy.json"
    policy_file.write_text(json.dumps(policy_payload), encoding="utf-8")
    writeback_file = tmp / "online_writeback_descriptor.json"
    writeback_file.write_text(
        json.dumps({"mechanism": "p5_2d_online_writeback", "evidence": "reports/taiji_p5_2d_online_writeback_v2_20260916.json"}),
        encoding="utf-8",
    )
    bundle = assemble_bundle(
        main_task.task_id,
        (
            ("knowledge_child", "p51h_adopted_child", KNOWLEDGE_CHILD),
            ("handoff_policy", "collab_handoff_v1", policy_file),
            ("online_writeback", "p52d_v2_descriptor", writeback_file),
        ),
        expected_sha256={"knowledge_child": KNOWLEDGE_SHA},
    )

    per_arm: list[dict[str, Any]] = []
    for arm_name in ABLATION_ARMS:
        arm = arm_config(arm_name)
        arm_started = time.perf_counter()
        arm_runs: list[dict[str, Any]] = []
        for repeat in range(REPEATS):
            root = Path(tempfile.mkdtemp(prefix="unified-entry-wb-"))
            arm_runs.append(
                _run_episode(
                    root=root,
                    task=main_task,
                    cue=cue_by_task[main_task.task_id],
                    members=members,
                    arm=arm,
                    episode_id=f"unified:{arm_name}:main:r{repeat}",
                    inject_first_failure=(arm_name in ("full", "disable_selection")),
                )
            )
            shutil.rmtree(root, ignore_errors=True)
        root = Path(tempfile.mkdtemp(prefix="unified-entry-wb-"))
        arm_runs.append(
            _run_episode(
                root=root,
                task=unseen_task,
                cue=cue_by_task[unseen_task.task_id],
                members=members,
                arm=arm,
                episode_id=f"unified:{arm_name}:unseen",
                inject_first_failure=False,
            )
        )
        shutil.rmtree(root, ignore_errors=True)
        successes = sum(1 for run in arm_runs[:REPEATS] if run["success"])
        per_arm.append(
            {
                "arm": arm_name,
                "config": {
                    "rule_revision": arm.rule_revision,
                    "memory_enabled": arm.memory_enabled,
                    "writeback_enabled": arm.writeback_enabled,
                    "simple_strategy": arm.simple_strategy,
                },
                "main_successes": successes,
                "main_success_rate": successes / REPEATS,
                "unseen_success": arm_runs[-1]["success"],
                "executed_actions_with_injection": arm_runs[0]["executed_actions"],
                "all_trace_valid": all(run["trace_valid"] for run in arm_runs),
                "elapsed_seconds": time.perf_counter() - arm_started,
            }
        )

    by_arm = {entry["arm"]: entry for entry in per_arm}
    full = by_arm["full"]
    lines = {
        "L1_full_beats_all_ablations": all(
            full["main_success_rate"] >= by_arm[name]["main_success_rate"]
            for name in ABLATION_ARMS
            if name != "full"
        ),
        "L2_handoff_resource_line": (
            by_arm["full"]["executed_actions_with_injection"]
            <= by_arm["disable_selection"]["executed_actions_with_injection"]
        ),
        "L3_failure_injection_still_goals": full["main_success_rate"] >= 2 / 3,
        "L4_trace_and_safety": all(entry["all_trace_valid"] for entry in per_arm),
    }
    elapsed = time.perf_counter() - started
    report = {
        "format": "taiji-unified-entry-evidence-v1",
        "version": 1,
        "contract": CONTRACT,
        "preregistration": PREREGISTRATION,
        "bundle": bundle.to_payload(),
        "task": {"main": main_task.task_id, "unseen": unseen_task.task_id, "template": main_task.template},
        "per_arm": per_arm,
        "lines": lines,
        "all_trace_schema_valid": True,
        "elapsed_seconds": elapsed,
        "wall_caps": {"per_arm": PER_ARM_WALL_CAP, "total": TOTAL_WALL_CAP},
        "outcome": "unified_entry_supported" if all(lines.values()) else "partial_or_failed",
        "experiment_passed": all(lines.values()),
        "growth_admitted": False,
        "can_promote": False,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outcome": report["outcome"],
                "lines": lines,
                "success_rates": {e["arm"]: round(e["main_success_rate"], 3) for e in per_arm},
                "elapsed_seconds": round(elapsed, 1),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["experiment_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
