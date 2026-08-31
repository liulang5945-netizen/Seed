"""Evaluate terminal governance after editor/MCP structural transfer."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training.eval_taiji_continuous_structural_growth import (  # noqa: E402
    _admit,
    _build_model,
    _pressure,
)
from scripts.training.eval_taiji_cross_domain_structural_gain import (  # noqa: E402
    _candidate,
    _cross_domain_training_examples,
    _run_cross_domain_task,
    _run_native_actions,
)
from scripts.training.eval_taiji_online_interaction_structural_bridge import (  # noqa: E402
    _run_online_feedbacks,
)
from scripts.training.eval_taiji_open_domain_interaction_gain import (  # noqa: E402
    build_train_corpus,
)
from seed import Seed  # noqa: E402
from seed_platform.workbench import (  # noqa: E402
    WorkbenchActionRequest,
    WorkbenchEnvironment,
)
from taiji import (  # noqa: E402
    ActionIntent,
    StructuralWorkspaceRouter,
    TSKV8Adapter,
    WorkspaceCandidate,
    WorkspaceRouter,
    WorkspaceRoutingExample,
)

REPORT_FORMAT = "taiji-w7-p4-12-terminal-three-domain-governance-v1"
LEARNER_SEEDS = (11, 29, 47)
FIT_EPOCHS = 40


def _terminal_action(marker: str, *, failing: bool = False) -> tuple[str, dict[str, object]]:
    code = (
        "raise SystemExit(9)"
        if failing
        else f"print({marker!r})"
    )
    return (
        "terminal.run",
        {
            "argv": [sys.executable, "-c", code],
            "cwd": ".",
            "timeout_seconds": 5.0,
            "output_limit": 1024,
            "env": {},
            "env_allowlist": [],
            "expected_artifacts": [],
            "execution_kind": "test",
        },
    )


TERMINAL_TRAIN_ACTIONS = (
    (
        "terminal-editor-readme",
        (
            ("editor.open", {"path": "README.md"}),
            _terminal_action("terminal-train-readme"),
        ),
    ),
    (
        "terminal-mcp",
        (
            ("mcp.list", {}),
            _terminal_action("terminal-train-mcp"),
        ),
    ),
    (
        "terminal-editor-pyproject",
        (
            ("editor.open", {"path": "pyproject.toml"}),
            _terminal_action("terminal-train-pyproject"),
        ),
    ),
)

TERMINAL_NEGATIVE_TRAIN_ACTIONS = (
    (
        "terminal-distractor-negative",
        (
            ("editor.open", {"path": "README.md"}),
            _terminal_action("terminal-distractor", failing=True),
        ),
    ),
)

TARGET_TASKS = (
        (
            "three-domain-readme",
            (
                ("editor.open", {"path": "README.md"}),
                ("mcp.list", {}),
                _terminal_action("terminal-train-readme"),
            ),
            _terminal_action("terminal-distractor", failing=True),
        ),
    (
            "three-domain-pyproject",
            (
                ("editor.open", {"path": "pyproject.toml"}),
                ("mcp.list", {}),
                _terminal_action("terminal-train-pyproject"),
            ),
            _terminal_action("terminal-distractor", failing=True),
        ),
)


def _training_examples() -> tuple[tuple[WorkspaceRoutingExample, ...], dict[str, object]]:
    base_examples, base_source = _cross_domain_training_examples()
    terminal_examples: list[WorkspaceRoutingExample] = []
    terminal_records: list[dict[str, object]] = []
    for family, actions in TERMINAL_TRAIN_ACTIONS:
        record = _run_native_actions(split="train", context_label=family, actions=actions)
        candidates = tuple(_candidate(action) for action in actions)
        relevant_ids = tuple(
            candidate.candidate_id
            for candidate, observed in zip(candidates, record["actions"])
            if bool(observed["success"])
        )
        if not record["all_success"] or len(relevant_ids) != len(actions):
            raise AssertionError(f"terminal training action failed: {family}")
        terminal_examples.append(
            WorkspaceRoutingExample(
                candidates=candidates,
                relevant_ids=relevant_ids,
                tick=len(base_examples) + len(terminal_examples) + 1,
            )
        )
        terminal_records.append(
            {
                "family": family,
                "capability_domains": sorted({item[0].split(".")[0] for item in actions}),
                "native_statuses": [item["status"] for item in record["actions"]],
                "approval_required": [item["approval_required"] for item in record["actions"]],
                "approval_granted": [item["approval_granted"] for item in record["actions"]],
                "preview_validated": [item["preview_validated"] for item in record["actions"]],
                "native_audit_event_count": record["native_audit_event_count"],
                "native_checkpoint_roundtrip": record["native_checkpoint_roundtrip"],
            }
        )
    for family, actions in TERMINAL_NEGATIVE_TRAIN_ACTIONS:
        record = _run_native_actions(split="train", context_label=family, actions=actions)
        candidates = tuple(_candidate(action) for action in actions)
        relevant_ids = tuple(
            candidate.candidate_id
            for candidate, observed in zip(candidates, record["actions"])
            if bool(observed["success"])
        )
        if not relevant_ids or bool(record["actions"][-1]["success"]):
            raise AssertionError(f"terminal negative training action did not record failure: {family}")
        terminal_examples.append(
            WorkspaceRoutingExample(
                candidates=candidates,
                relevant_ids=relevant_ids,
                tick=len(base_examples) + len(terminal_examples) + 1,
            )
        )
        terminal_records.append(
            {
                "family": family,
                "capability_domains": sorted({item[0].split(".")[0] for item in actions}),
                "native_statuses": [item["status"] for item in record["actions"]],
                "approval_required": [item["approval_required"] for item in record["actions"]],
                "approval_granted": [item["approval_granted"] for item in record["actions"]],
                "preview_validated": [item["preview_validated"] for item in record["actions"]],
                "native_audit_event_count": record["native_audit_event_count"],
                "native_checkpoint_roundtrip": record["native_checkpoint_roundtrip"],
                "expected_terminal_failure": True,
            }
        )
    return tuple((*base_examples, *terminal_examples)), {
        "base_source": base_source,
        "base_example_count": len(base_examples),
        "terminal_example_count": len(terminal_examples),
        "example_count": len(base_examples) + len(terminal_examples),
        "terminal_records": terminal_records,
        "score_source": "native Workbench outcome status and success",
    }


def _task_candidates(
    required_actions: tuple[tuple[str, dict[str, object]], ...],
    distractor: tuple[str, dict[str, object]],
) -> tuple[tuple[WorkspaceCandidate, ...], dict[str, tuple[str, dict[str, object]]], tuple[str, ...]]:
    actions = (*required_actions, distractor)
    candidates = tuple(_candidate(action) for action in actions)
    action_by_id = {
        candidate.candidate_id: action for candidate, action in zip(candidates, actions)
    }
    required_ids = tuple(_candidate(action).candidate_id for action in required_actions)
    return candidates, action_by_id, required_ids


def _approved_request(runtime: SeedRuntime, intent: ActionIntent) -> WorkbenchActionRequest:
    environment = runtime.workbench_environment
    mcp_snapshot_id = (
        environment.mcp_registry.snapshot_id if intent.kind.startswith("mcp.") else ""
    )
    request = WorkbenchActionRequest.from_action_intent(
        intent,
        snapshot_id=environment.capability_snapshot.snapshot_id,
        mcp_registry_snapshot_id=mcp_snapshot_id,
        capability_registry_snapshot_id=environment.capability_registry.snapshot_id,
    )
    decision = environment.policy_for(request)
    if decision.decision != "allow":
        if decision.reason_code != "capability_requires_approval":
            raise AssertionError(f"terminal request was not governable: {decision.to_payload()}")
        approval = environment.issue_approval(request)
        request = replace(request, approval_token=str(approval["approval_token"]))
    return request


def _terminal_governance_probe(seed: int) -> dict[str, object]:
    """Exercise no-approval rejection, failure stop, checkpoint, and fresh recovery."""

    # The managed Windows runner can create a TemporaryDirectory but denies
    # Python's exclusive file creation inside it.  Keep the probe isolated by
    # using a dot-prefixed checkpoint in the repository's existing writable
    # checkpoint directory; the successful path removes it before returning.
    with nullcontext():
        checkpoint_path = PROJECT_ROOT / "checkpoints" / f".p4-12-terminal-recovery-{seed}.pt"
        checkpoint_path.unlink(missing_ok=True)
        with patch(
            "seed_platform.workbench.get_setting",
            lambda key, default=None: str(PROJECT_ROOT) if key == "workspace_path" else default,
        ):
            runtime = SeedRuntime(
                Seed(episode_id=f"terminal-governance-{seed}"),
                checkpoint_path=checkpoint_path,
            )
            runtime._workbench_environment = WorkbenchEnvironment(PROJECT_ROOT)
            snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
            safe_intent = ActionIntent(
                intent_id=f"p4-12-unapproved-{seed}",
                kind="terminal.run",
                parameters=_terminal_action("unapproved")[1],
                confidence=1.0,
                tick=runtime.model.tick,
            )
            unapproved = runtime.execute_workbench_intent(
                safe_intent,
                snapshot_id=snapshot_id,
                learn=False,
            )
            approved_request = _approved_request(runtime, safe_intent)
            approved = runtime.execute_workbench_intent(
                safe_intent,
                snapshot_id=snapshot_id,
                approval_token=approved_request.approval_token,
                learn=False,
            )
            failure_intent = ActionIntent(
                intent_id=f"p4-12-failure-{seed}",
                kind="terminal.run",
                parameters=_terminal_action("failure", failing=True)[1],
                confidence=1.0,
                tick=runtime.model.tick,
            )
            recovery_intent = ActionIntent(
                intent_id=f"p4-12-recovery-{seed}",
                kind="terminal.run",
                parameters=_terminal_action("recovered")[1],
                confidence=1.0,
                tick=runtime.model.tick,
            )
            failure_request = _approved_request(runtime, failure_intent)
            recovery_request = _approved_request(runtime, recovery_intent)
            preflight = runtime.preflight_workbench_loop(
                (failure_request, recovery_request),
                loop_id=f"p4-12-failure-loop-{seed}",
                max_steps=2,
                max_budget_units=4.0,
            )
            failed_run = runtime.execute_preflighted_workbench_loop(
                (failure_intent, recovery_intent),
                (failure_request, recovery_request),
                loop_id=f"p4-12-failure-loop-{seed}",
                preflight_id=preflight["preflight_id"],
                max_steps=2,
                max_budget_units=4.0,
                learn=False,
            )
            if not checkpoint_path.exists():
                raise AssertionError("terminal failure loop did not save a checkpoint")
            restored = SeedRuntime.load(checkpoint_path)
            restored._workbench_environment = WorkbenchEnvironment(PROJECT_ROOT)
            fresh_recovery_intent = ActionIntent(
                intent_id=f"p4-12-fresh-recovery-{seed}",
                kind="terminal.run",
                parameters=_terminal_action("fresh-recovery")[1],
                confidence=1.0,
                tick=restored.model.tick,
            )
            fresh_request = _approved_request(restored, fresh_recovery_intent)
            recovery_preflight = restored.preflight_workbench_loop(
                (fresh_request,),
                loop_id=f"p4-12-recovery-loop-{seed}",
                max_steps=1,
                max_budget_units=2.0,
            )
            recovered_run = restored.execute_preflighted_workbench_loop(
                (fresh_recovery_intent,),
                (fresh_request,),
                loop_id=f"p4-12-recovery-loop-{seed}",
                preflight_id=recovery_preflight["preflight_id"],
                max_steps=1,
                max_budget_units=2.0,
                learn=False,
            )
            checkpoint_saved = checkpoint_path.exists()
            checkpoint_path.unlink(missing_ok=True)
            return {
                "unapproved_status": unapproved["outcome"]["status"],
                "unapproved_success": bool(unapproved["outcome"]["success"]),
                "approved_status": approved["outcome"]["status"],
                "approved_success": bool(approved["outcome"]["success"]),
                "approved_shell_free": approved["outcome"]["result"].get("shell") is False,
                "approved_output_limit": approved["outcome"]["result"].get("output_limit"),
                "failure_loop_status": failed_run["status"],
                "failure_loop_stopped_at": failed_run.get("stopped_at"),
                "failure_loop_completed_prefix": failed_run["completed_prefix"],
                "failure_step_success": failed_run["steps"][0]["success"],
                "recovery_status": recovered_run["status"],
                "recovery_step_success": recovered_run["steps"][0]["success"],
                "checkpoint_exists": checkpoint_saved,
            }


def evaluate() -> dict[str, object]:
    corpus, _ = build_train_corpus()
    train_examples, training_source = _training_examples()
    runs: list[dict[str, object]] = []
    for seed in LEARNER_SEEDS:
        controller, feedbacks, rounds = _run_online_feedbacks(corpus=corpus, seed=seed)
        pressure = _pressure(feedbacks, controller, seed=seed, cycle=4)
        model, region = _build_model(f"terminal-three-domain-{seed}")
        initial_checkpoint = model.native_checkpoint()
        workspace = StructuralWorkspaceRouter(
            WorkspaceRouter(256, capacity=2, seed=seed),
            region,
        )
        workspace.fit(train_examples, epochs=FIT_EPOCHS, learning_rate=0.2)
        parent_workspace_checkpoint = workspace.checkpoint()
        model, admission = _admit(model, pressure, unit_id="u2")
        workspace.rebind(model.neuron_regions[0])
        grown_model_checkpoint = model.native_checkpoint()
        grown_workspace_checkpoint = workspace.checkpoint()
        continued_workspace = StructuralWorkspaceRouter.from_checkpoint(
            grown_workspace_checkpoint,
            region=TSKV8Adapter.from_native_checkpoint(grown_model_checkpoint).neuron_regions[0],
        )
        structural_runs = [
            _run_cross_domain_task(
                router=continued_workspace,
                task_label=task_label,
                required_actions=required_actions,
                distractor=distractor,
                seed=seed,
                method="terminal_three_domain_growth",
            )
            for task_label, required_actions, distractor in TARGET_TASKS
        ]
        controls: dict[str, StructuralWorkspaceRouter] = {}
        for offset, method in enumerate(("interaction_weight_only", "router_only", "memory_only"), 1):
            control_model = TSKV8Adapter.from_native_checkpoint(initial_checkpoint)
            control = StructuralWorkspaceRouter(
                WorkspaceRouter(256, capacity=2, seed=seed + 1000 + offset),
                control_model.neuron_regions[0],
            )
            control.fit(train_examples, epochs=FIT_EPOCHS, learning_rate=0.2)
            controls[method] = control
        control_runs = {
            method: [
                _run_cross_domain_task(
                    router=router,
                    task_label=task_label,
                    required_actions=required_actions,
                    distractor=distractor,
                    seed=seed,
                    method=method,
                )
                for task_label, required_actions, distractor in TARGET_TASKS
            ]
            for method, router in controls.items()
        }
        retention_runs = []
        from scripts.training.eval_taiji_structural_workspace_net_gain import _run_routed_task

        for task_label, required_keys, distractor_key in (
            ("future-bec", ("b", "e", "c"), "f"),
            ("future-ade", ("a", "d", "e"), "g"),
        ):
            retention_runs.append(
                _run_routed_task(
                    router=continued_workspace,
                    task_label=f"terminal-retention-{task_label}",
                    required_keys=required_keys,
                    distractor_key=distractor_key,
                    seed=seed,
                    method="terminal_three_domain_retention",
                )
            )
        lesion_model = TSKV8Adapter.from_native_checkpoint(grown_model_checkpoint)
        lesion_workspace = StructuralWorkspaceRouter.from_checkpoint(
            grown_workspace_checkpoint,
            region=lesion_model.neuron_regions[0],
        )
        proposal = lesion_model.topology_proposals[-1]
        specification = tuple(
            (
                key,
                len(lesion_model.neuron_regions[0].unit_ids)
                if key == "existing_unit_count"
                else value,
            )
            for key, value in proposal.specification
        )
        lesion_model.neuron_regions[0].lesion_topology_proposal(
            replace(proposal, status="pending", specification=specification)
        )
        lesion_workspace.rebind(lesion_model.neuron_regions[0])
        lesion_run = _run_cross_domain_task(
            router=lesion_workspace,
            task_label="terminal-three-domain-lesion",
            required_actions=TARGET_TASKS[0][1],
            distractor=TARGET_TASKS[0][2],
            seed=seed,
            method="terminal_three_domain_lesion",
        )
        topology_before_rollback = str(admission["topology_before"])
        rollback = model.rollback_structural_candidate(admission["candidate_id"])
        workspace.rebind(model.neuron_regions[0])
        runs.append(
            {
                "seed": seed,
                "online_rounds": list(rounds),
                "online_feedback_ids": [item.feedback_id for item in feedbacks],
                "pressure": pressure.to_payload(),
                "admission": admission,
                "structural_runs": structural_runs,
                "retention_runs": retention_runs,
                "control_runs": control_runs,
                "lesion_run": lesion_run,
                "governance_probe": _terminal_governance_probe(seed),
                "parent_capacity": parent_workspace_checkpoint["router"]["capacity"],
                "grown_capacity": continued_workspace.capacity,
                "lesion_capacity": lesion_run["capacity"],
                "rollback_capacity": workspace.capacity,
                "topology_before_rollback": topology_before_rollback,
                "topology_after_rollback": model._structural_topology_digest(
                    model.native_checkpoint()
                ),
                "budget_before": admission["budget_before"],
                "budget_after_admission": admission["budget_after"],
                "budget_after_rollback": model.cognitive_snapshot().development.structural_budget,
                "rollback_status": "rolled_back" if rollback else "rollback_failed",
                "workspace_checkpoint_roundtrip": (
                    continued_workspace.checkpoint()["checkpoint_digest"]
                    == grown_workspace_checkpoint["checkpoint_digest"]
                ),
            }
        )

    structural_scores = [
        float(item["task_score"])
        for run in runs
        for item in run["structural_runs"]
    ]
    structural_mean = sum(structural_scores) / len(structural_scores)
    control_means = {
        method: sum(
            float(item["task_score"])
            for run in runs
            for item in run["control_runs"][method]
        )
        / (len(runs) * len(TARGET_TASKS))
        for method in ("interaction_weight_only", "router_only", "memory_only")
    }
    metrics = {
        "three_independent_seeds": len(runs) == len(LEARNER_SEEDS),
        "training_contains_actual_terminal_records": (
            training_source["terminal_example_count"]
            == len(TERMINAL_TRAIN_ACTIONS) + len(TERMINAL_NEGATIVE_TRAIN_ACTIONS)
            and all(
                set(record["capability_domains"]) in (
                    {"editor", "terminal"},
                    {"mcp", "terminal"},
                )
                for record in training_source["terminal_records"]
            )
            and all(
                all(status == "success" for status in record["native_statuses"])
                and record["native_audit_event_count"] > 0
                and record["native_checkpoint_roundtrip"]
                and record["approval_granted"][1]
                and record["preview_validated"][1]
                for record in training_source["terminal_records"]
                if not record.get("expected_terminal_failure", False)
            )
            and all(
                record.get("expected_terminal_failure", False)
                and record["native_statuses"] == ["success", "error"]
                and record["native_audit_event_count"] > 0
                and record["native_checkpoint_roundtrip"]
                and record["approval_granted"][1]
                and record["preview_validated"][1]
                for record in training_source["terminal_records"]
                if record.get("expected_terminal_failure", False)
            )
        ),
        "three_domain_structural_gain": structural_mean == 1.0
        and all(
            set(item["capability_domains"]) == {"editor", "mcp", "terminal"}
            for run in runs
            for item in run["structural_runs"]
        ),
        "three_domain_growth_beats_fixed_capacity_controls": all(
            structural_mean > score for score in control_means.values()
        ),
        "old_editor_mcp_workspace_retention": all(
            all(item["task_score"] == 1.0 for item in run["retention_runs"])
            for run in runs
        ),
        "terminal_requires_approval_and_respects_resources": all(
            all(
                any(
                    capability_id == "terminal.run"
                    and approval_required
                    and approval_granted
                    and preview_validated
                    and result.get("shell") is False
                    and int(result.get("output_limit", 0)) <= 1024
                    and "timeout_seconds" in parameters
                    and "expected_artifacts" in parameters
                    for capability_id, parameters, approval_required, approval_granted, preview_validated, result in zip(
                        item["selected_capability_ids"],
                        item["selected_parameters"],
                        item["approval_required"],
                        item["approval_granted"],
                        item["preview_validated"],
                        item["raw_action_results"],
                    )
                )
                for item in run["structural_runs"]
            )
            for run in runs
        ),
        "native_status_and_checkpoint_evidence": all(
            item["native_event_count"] > 0
            and item["native_audit_event_count"] > 0
            and item["native_checkpoint_roundtrip"]
            for run in runs
            for item in run["structural_runs"]
        ) and all(
            item["native_world_event_count"] > 0
            and item["native_checkpoint_replay"]
            for run in runs
            for item in run["retention_runs"]
        ),
        "controls_remain_at_parent_capacity": all(
            all(
                item["capacity"] == 2
                for item in run["control_runs"][method]
            )
            for run in runs
            for method in ("interaction_weight_only", "router_only", "memory_only")
        ),
        "capacity_tracks_growth_lesion_and_rollback": all(
            run["parent_capacity"] == 2
            and run["grown_capacity"] == 3
            and run["lesion_capacity"] == 2
            and run["rollback_capacity"] == 2
            for run in runs
        ),
        "growth_is_checkpointable": all(run["workspace_checkpoint_roundtrip"] for run in runs),
        "lesion_removes_three_domain_gain": all(
            run["lesion_run"]["task_score"] == 0.0 for run in runs
        ),
        "terminal_failure_stops_and_fresh_recovery_succeeds": all(
            probe["unapproved_status"] == "rejected"
            and not probe["unapproved_success"]
            and probe["approved_status"] == "success"
            and probe["approved_success"]
            and probe["approved_shell_free"]
            and probe["failure_loop_status"] == "failed"
            and probe["failure_loop_stopped_at"] == 0
            and probe["failure_loop_completed_prefix"] == 0
            and not probe["failure_step_success"]
            and probe["recovery_status"] == "completed"
            and probe["recovery_step_success"]
            and probe["checkpoint_exists"]
            for run in runs
            for probe in (run["governance_probe"],)
        ),
        "rollback_restores_topology_and_budget": all(
            run["rollback_status"] == "rolled_back"
            and run["topology_before_rollback"] == run["topology_after_rollback"]
            and run["budget_after_rollback"] == run["budget_before"]
            for run in runs
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "task": "terminal governance and three-domain structural workspace gain",
        "capability_domains": ["editor", "mcp", "terminal"],
        "target_tasks": [item[0] for item in TARGET_TASKS],
        "learner_seeds": list(LEARNER_SEEDS),
        "structural_mean_score": structural_mean,
        "control_mean_scores": control_means,
        "training_source": training_source,
        "runs": runs,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "an online-evidence-backed admission must expand live capacity, complete two "
                "unseen three-action editor+MCP+terminal tasks, retain old capabilities, "
                "beat fixed-capacity controls, require explicit terminal approval and bounded "
                "execution, stop on terminal failure, recover from checkpoint, and rollback "
                "topology and budget"
            ),
        },
        "boundary": (
            "This is bounded three-domain governance evidence. It does not claim unrestricted "
            "terminal autonomy, arbitrary command safety, capability discovery, AGI, full "
            "language understanding, or CUDA acceleration."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_p4_12_terminal_three_domain_governance_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"gate": report["gate"], "metrics": report["metrics"]}, ensure_ascii=False))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
