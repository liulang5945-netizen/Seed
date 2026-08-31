"""Evaluate structural workspace gain across editor and MCP capability domains."""

from __future__ import annotations

import argparse
import json
import sys
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
from scripts.training.eval_taiji_online_interaction_structural_bridge import (  # noqa: E402
    _run_online_feedbacks,
)
from scripts.training.eval_taiji_open_domain_interaction_gain import (  # noqa: E402
    build_train_corpus,
)
from scripts.training.eval_taiji_structural_workspace_net_gain import (  # noqa: E402
    FEATURE_DIM,
    _candidate,
    _run_routed_task,
)
from scripts.training.eval_taiji_structural_workspace_net_gain import (
    _training_examples as _base_training_examples,
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

REPORT_FORMAT = "taiji-w7-p4-11-cross-domain-structural-gain-v1"
LEARNER_SEEDS = (11, 29, 47)

EDITOR_MCP_TRAIN_ACTIONS = (
    (
        "editor-mcp-readme",
        (
            ("editor.open", {"path": "README.md"}),
            ("mcp.list", {}),
        ),
    ),
    (
        "editor-mcp-pyproject",
        (
            ("editor.open", {"path": "pyproject.toml"}),
            ("mcp.list", {}),
        ),
    ),
    (
        "editor-mcp-requirements",
        (
            ("editor.open", {"path": "requirements.txt"}),
            ("mcp.list", {}),
        ),
    ),
)

CROSS_DOMAIN_TASKS = (
    (
        "cross-editor-mcp-readme-pyproject",
        (
            ("editor.open", {"path": "README.md"}),
            ("editor.open", {"path": "pyproject.toml"}),
            ("mcp.list", {}),
        ),
        ("editor.open", {"path": "missing-cross-domain.txt"}),
    ),
    (
        "cross-editor-mcp-readme-requirements",
        (
            ("editor.open", {"path": "README.md"}),
            ("editor.open", {"path": "requirements.txt"}),
            ("mcp.list", {}),
        ),
        ("editor.open", {"path": "missing-cross-domain.txt"}),
    ),
)


def _native_action(
    runtime: SeedRuntime,
    *,
    capability_id: str,
    parameters: dict[str, object],
    ordinal: int,
) -> dict[str, object]:
    """Execute one structured ActionIntent through the native Workbench seam."""

    environment = runtime.workbench_environment
    intent = ActionIntent(
        intent_id=f"p4-11:{capability_id}:{ordinal}",
        kind=capability_id,
        parameters=parameters,
        confidence=1.0,
        tick=runtime.model.tick,
    )
    mcp_snapshot_id = (
        environment.mcp_registry.snapshot_id if capability_id.startswith("mcp.") else ""
    )
    request = WorkbenchActionRequest.from_action_intent(
        intent,
        snapshot_id=environment.capability_snapshot.snapshot_id,
        mcp_registry_snapshot_id=mcp_snapshot_id,
        capability_registry_snapshot_id=environment.capability_registry.snapshot_id,
    )
    policy = environment.policy_for(request)
    approval_token = ""
    preview_validated = False
    if policy.decision == "ask_user" and policy.reason_code == "capability_requires_approval":
        approval = environment.issue_approval(request)
        approval_token = str(approval["approval_token"])
        preview_validated = bool(approval.get("preview", {}).get("validated"))
    elif policy.decision != "allow":
        raise AssertionError(
            f"cross-domain native action was not admitted: {capability_id} {policy.to_payload()}"
        )
    result = runtime.execute_workbench_intent(
        intent,
        snapshot_id=environment.capability_snapshot.snapshot_id,
        approval_token=approval_token,
        mcp_registry_snapshot_id=mcp_snapshot_id,
        capability_registry_snapshot_id=environment.capability_registry.snapshot_id,
        learn=False,
    )
    outcome = result.get("outcome")
    if not isinstance(outcome, dict):
        raise AssertionError("native Workbench action did not return an outcome")
    return {
        "capability_id": capability_id,
        "parameters": dict(parameters),
        "policy_decision": policy.decision,
        "policy_reason": policy.reason_code,
        "approval_required": policy.reason_code == "capability_requires_approval",
        "approval_granted": bool(approval_token),
        "preview_validated": preview_validated,
        "status": outcome.get("status"),
        "success": bool(outcome.get("success", False)),
        "error_code": outcome.get("error_code", ""),
        "result": dict(outcome.get("result", {})),
        "request_id": outcome.get("request_id", ""),
        "tick": outcome.get("tick", 0),
    }


def _run_native_actions(
    *,
    split: str,
    context_label: str,
    actions: tuple[tuple[str, dict[str, object]], ...],
) -> dict[str, object]:
    """Run editor/MCP actions and retain native status, audit, and checkpoint evidence."""

    episode_id = f"p4-11-{split}-{context_label}"
    with patch(
        "seed_platform.workbench.get_setting",
        lambda key, default=None: str(PROJECT_ROOT) if key == "workspace_path" else default,
    ):
        runtime = SeedRuntime(Seed(episode_id=episode_id))
        runtime._workbench_environment = WorkbenchEnvironment(PROJECT_ROOT)
        runtime.model.architecture.observe(65, learn=False)
        records = [
            _native_action(
                runtime,
                capability_id=capability_id,
                parameters=parameters,
                ordinal=index,
            )
            for index, (capability_id, parameters) in enumerate(actions, start=1)
        ]
        checkpoint = runtime.model.architecture.native_checkpoint()
        restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
        state = runtime.model.architecture.cognitive_snapshot()
        return {
            "episode_id": episode_id,
            "actions": records,
            "native_event_count": len(state.events),
            "native_audit_event_count": len(runtime._workbench_audit.events),
            "native_checkpoint_format": checkpoint["format"],
            "native_checkpoint_roundtrip": (
                restored._structural_topology_digest(restored.native_checkpoint())
                == runtime.model.architecture._structural_topology_digest(checkpoint)
            ),
            "all_success": all(
                item["status"] == "success" and item["success"] for item in records
            ),
        }


def _cross_domain_training_examples() -> tuple[tuple[WorkspaceRoutingExample, ...], dict[str, object]]:
    examples: list[WorkspaceRoutingExample] = []
    records: list[dict[str, object]] = []
    for family, actions in EDITOR_MCP_TRAIN_ACTIONS:
        record = _run_native_actions(split="train", context_label=family, actions=actions)
        candidates = tuple(_candidate(action) for action in actions)
        relevant_ids = tuple(
            candidate.candidate_id
            for candidate, observed in zip(candidates, record["actions"])
            if bool(observed["success"])
        )
        if not record["all_success"] or len(relevant_ids) != len(actions):
            raise AssertionError(f"cross-domain training action failed: {family}")
        examples.append(
            WorkspaceRoutingExample(
                candidates=candidates,
                relevant_ids=relevant_ids,
                tick=len(examples) + 1,
            )
        )
        records.append(
            {
                "family": family,
                "capability_domains": sorted({item[0].split(".")[0] for item in actions}),
                "native_statuses": [item["status"] for item in record["actions"]],
                "native_audit_event_count": record["native_audit_event_count"],
                "native_checkpoint_roundtrip": record["native_checkpoint_roundtrip"],
            }
        )
    base_examples, base_source = _base_training_examples()
    return tuple((*base_examples, *examples)), {
        "base_example_count": len(base_examples),
        "cross_domain_example_count": len(examples),
        "example_count": len(base_examples) + len(examples),
        "base_source": base_source,
        "cross_domain_records": records,
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


def _run_cross_domain_task(
    *,
    router: StructuralWorkspaceRouter,
    task_label: str,
    required_actions: tuple[tuple[str, dict[str, object]], ...],
    distractor: tuple[str, dict[str, object]],
    seed: int,
    method: str,
) -> dict[str, object]:
    candidates, action_by_id, required_ids = _task_candidates(required_actions, distractor)
    selection = router.route(candidates, tick=300 + seed, mode="learned")
    actions = tuple(action_by_id[item] for item in selection.selected_ids)
    native = _run_native_actions(
        split="holdout",
        context_label=f"{task_label}-{method}-{seed}",
        actions=actions,
    )
    action_success = tuple(bool(item["success"]) for item in native["actions"])
    complete = set(selection.selected_ids) == set(required_ids)
    all_selected_actions_succeeded = bool(actions) and all(action_success)
    task_score = 1.0 if complete and all_selected_actions_succeeded else 0.0
    return {
        "task": task_label,
        "method": method,
        "seed": seed,
        "capability_domains": sorted({item[0].split(".")[0] for item in actions}),
        "capacity": selection.capacity,
        "candidate_ids": list(selection.candidate_ids),
        "selected_ids": list(selection.selected_ids),
        "required_ids": list(required_ids),
        "selected_action_count": len(actions),
        "selected_capability_ids": [item["capability_id"] for item in native["actions"]],
        "selected_parameters": [item["parameters"] for item in native["actions"]],
        "raw_action_status": [item["status"] for item in native["actions"]],
        "raw_action_success": list(action_success),
        "raw_action_results": [item["result"] for item in native["actions"]],
        "approval_required": [item["approval_required"] for item in native["actions"]],
        "approval_granted": [item["approval_granted"] for item in native["actions"]],
        "preview_validated": [item["preview_validated"] for item in native["actions"]],
        "task_complete": complete,
        "all_selected_actions_succeeded": all_selected_actions_succeeded,
        "task_score": task_score,
        "score_source": "native Workbench outcome status/success plus required-set completion",
        "native_event_count": native["native_event_count"],
        "native_audit_event_count": native["native_audit_event_count"],
        "native_checkpoint_roundtrip": native["native_checkpoint_roundtrip"],
    }


def _control_routers(
    *,
    checkpoint: dict[str, object],
    train_examples: tuple[WorkspaceRoutingExample, ...],
    seed: int,
) -> dict[str, StructuralWorkspaceRouter]:
    controls: dict[str, StructuralWorkspaceRouter] = {}
    for offset, method in enumerate(("interaction_weight_only", "router_only", "memory_only"), 1):
        model = TSKV8Adapter.from_native_checkpoint(checkpoint)
        router = StructuralWorkspaceRouter(
            WorkspaceRouter(FEATURE_DIM, capacity=2, seed=seed + 1000 + offset),
            model.neuron_regions[0],
        )
        router.fit(train_examples, epochs=120, learning_rate=0.2)
        controls[method] = router
    return controls


def evaluate() -> dict[str, object]:
    corpus, _ = build_train_corpus()
    train_examples, training_source = _cross_domain_training_examples()
    runs: list[dict[str, object]] = []
    for seed in LEARNER_SEEDS:
        controller, feedbacks, rounds = _run_online_feedbacks(corpus=corpus, seed=seed)
        pressure = _pressure(feedbacks, controller, seed=seed, cycle=3)
        model, region = _build_model(f"cross-domain-structural-{seed}")
        initial_checkpoint = model.native_checkpoint()
        workspace = StructuralWorkspaceRouter(
            WorkspaceRouter(FEATURE_DIM, capacity=2, seed=seed),
            region,
        )
        workspace.fit(train_examples, epochs=120, learning_rate=0.2)
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
                method="structural_growth",
            )
            for task_label, required_actions, distractor in CROSS_DOMAIN_TASKS
        ]
        retention_runs = [
            _run_routed_task(
                router=continued_workspace,
                task_label=f"retention-{task_label}",
                required_keys=required_keys,
                distractor_key=distractor_key,
                seed=seed,
                method="cross_domain_retention",
            )
            for task_label, required_keys, distractor_key in (
                ("future-bec", ("b", "e", "c"), "f"),
                ("future-ade", ("a", "d", "e"), "g"),
            )
        ]
        controls = _control_routers(
            checkpoint=initial_checkpoint,
            train_examples=train_examples,
            seed=seed,
        )
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
                for task_label, required_actions, distractor in CROSS_DOMAIN_TASKS
            ]
            for method, router in controls.items()
        }
        lesion_model = TSKV8Adapter.from_native_checkpoint(grown_model_checkpoint)
        lesion_workspace = StructuralWorkspaceRouter.from_checkpoint(
            grown_workspace_checkpoint,
            region=lesion_model.neuron_regions[0],
        )
        proposal = lesion_model.topology_proposals[-1]
        lesion_specification = tuple(
            (
                key,
                len(lesion_model.neuron_regions[0].unit_ids)
                if key == "existing_unit_count"
                else value,
            )
            for key, value in proposal.specification
        )
        lesion_workspace.rebind(lesion_model.neuron_regions[0])
        lesion_model.neuron_regions[0].lesion_topology_proposal(
            replace(
                proposal,
                status="pending",
                specification=lesion_specification,
            )
        )
        lesion_run = _run_cross_domain_task(
            router=lesion_workspace,
            task_label="cross-domain-lesion",
            required_actions=CROSS_DOMAIN_TASKS[0][1],
            distractor=CROSS_DOMAIN_TASKS[0][2],
            seed=seed,
            method="structural_growth_lesion",
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
        / (len(runs) * len(CROSS_DOMAIN_TASKS))
        for method in ("interaction_weight_only", "router_only", "memory_only")
    }
    metrics = {
        "three_independent_seeds": len(runs) == len(LEARNER_SEEDS),
        "training_contains_actual_editor_and_mcp_records": (
            training_source["cross_domain_example_count"] == len(EDITOR_MCP_TRAIN_ACTIONS)
            and all(
                set(record["capability_domains"]) == {"editor", "mcp"}
                and all(status == "success" for status in record["native_statuses"])
                and record["native_audit_event_count"] > 0
                and record["native_checkpoint_roundtrip"]
                for record in training_source["cross_domain_records"]
            )
        ),
        "cross_domain_structural_gain": structural_mean == 1.0,
        "cross_domain_growth_beats_fixed_capacity_controls": all(
            structural_mean > score for score in control_means.values()
        ),
        "old_workspace_capability_retention": all(
            all(item["task_score"] == 1.0 for item in run["retention_runs"])
            for run in runs
        ),
        "native_status_and_checkpoint_evidence": all(
            item["score_source"].startswith("native Workbench")
            and item["native_event_count"] > 0
            and item["native_audit_event_count"] > 0
            and item["native_checkpoint_roundtrip"]
            for run in runs
            for item in run["structural_runs"]
        )
        and all(
            item["score_source"].startswith("native Workbench")
            and item["native_world_event_count"] > 0
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
        "growth_is_checkpointable": all(
            run["workspace_checkpoint_roundtrip"] for run in runs
        ),
        "lesion_removes_cross_domain_gain": all(
            run["lesion_run"]["task_score"] == 0.0 for run in runs
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
        "task": "editor and MCP cross-domain structural workspace gain with old-capability retention",
        "capability_domains": ["editor", "mcp"],
        "target_tasks": [item[0] for item in CROSS_DOMAIN_TASKS],
        "learner_seeds": list(LEARNER_SEEDS),
        "structural_mean_score": structural_mean,
        "control_mean_scores": control_means,
        "training_source": training_source,
        "runs": runs,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "an online-evidence-backed neuron admission must expand the live workspace, "
                "complete two unseen three-action tasks spanning editor and MCP, retain prior "
                "workspace tasks, beat fixed-capacity controls, and preserve native status, "
                "checkpoint, lesion, resource, and rollback evidence"
            ),
        },
        "boundary": (
            "This is bounded cross-domain capacity evidence. It does not claim autonomous "
            "capability discovery, unrestricted architecture design, AGI, terminal safety, "
            "full language understanding, or CUDA acceleration."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_p4_11_cross_domain_structural_gain_20260831.json",
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
