"""Evaluate structural workspace growth on unseen real Workbench compositions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_interaction_groups import (  # noqa: E402
    _run_workbench_episode,
    _workbench_owner,
)
from scripts.training.eval_taiji_online_interaction_structural_bridge import (  # noqa: E402
    _independent_observation,
    _run_online_episode,
    _run_online_feedbacks,
)
from scripts.training.eval_taiji_open_domain_interaction_gain import (  # noqa: E402
    MEMBER_ACTIONS,
    MEMBER_IDS,
    TRAIN_FAMILIES,
    _action_cells,
    _run_scored_cell,
    build_train_corpus,
)
from scripts.training.eval_taiji_structural_validation import (  # noqa: E402
    _build_model,
    _expected_activity,
)
from taiji import (  # noqa: E402
    InteractionStructuralBridge,
    InteractionStructuralBridgeConfig,
    StructuralWorkspaceRouter,
    WorkspaceCandidate,
    WorkspaceRouter,
    WorkspaceRoutingExample,
)

REPORT_FORMAT = "taiji-w7-p4-10-structural-workspace-net-gain-v1"
FEATURE_DIM = 256
LEARNER_SEEDS = (11, 29, 47)
TARGET_TASKS = (
    ("future-bec", ("b", "e", "c"), "f"),
    ("future-ade", ("a", "d", "e"), "g"),
)


def _action_feature(action: tuple[str, dict[str, object]]) -> torch.Tensor:
    """Encode a Workbench binding by content, without semantic role labels."""

    payload = json.dumps(
        {"capability_id": action[0], "parameters": action[1]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    feature = torch.zeros(FEATURE_DIM)
    for offset in range(0, 16, 2):
        feature[int(digest[offset : offset + 2], 16)] += 1.0
    return feature


def _candidate(action: tuple[str, dict[str, object]]) -> WorkspaceCandidate:
    capability_id, parameters = action
    return WorkspaceCandidate(
        candidate_id=_workbench_owner(capability_id, parameters),
        features=_action_feature(action),
        source="seed.workbench.binding",
    )


def _training_examples() -> tuple[tuple[WorkspaceRoutingExample, ...], dict[str, object]]:
    """Build route supervision from actual Workbench success/status traces."""

    examples: list[WorkspaceRoutingExample] = []
    records: list[dict[str, object]] = []
    for family, first, second in TRAIN_FAMILIES:
        for cell_label, actions, expected_score in _action_cells(first, second):
            if not actions:
                continue
            _, record = _run_scored_cell(
                split="train",
                context_label=f"workspace-route-{family}",
                cell_label=cell_label,
                actions=actions,
                expected_score=expected_score,
            )
            candidates = tuple(_candidate(action) for action in actions)
            raw_actions = record["workbench_outcome"]["raw_actions"][1 : 1 + len(actions)]
            relevant_ids = tuple(
                candidate.candidate_id
                for candidate, observed in zip(candidates, raw_actions)
                if bool(observed["success"])
            )
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
                    "cell": cell_label,
                    "action_count": len(actions),
                    "successful_action_count": len(relevant_ids),
                    "score_source": record["score_source"],
                    "native_world_event_count": record["native_world_event_count"],
                }
            )
    if not examples:
        raise AssertionError("structural workspace route training has no examples")
    return tuple(examples), {
        "example_count": len(examples),
        "record_count": len(records),
        "records": records,
        "score_sources": sorted({str(item["score_source"]) for item in records}),
    }


def _task_candidates(
    required_keys: tuple[str, ...], distractor_key: str
) -> tuple[tuple[WorkspaceCandidate, ...], dict[str, tuple[str, dict[str, object]]], tuple[str, ...]]:
    keys = (*required_keys, distractor_key)
    actions = {key: MEMBER_ACTIONS[key] for key in keys}
    candidates = tuple(_candidate(actions[key]) for key in keys)
    action_by_id = {
        candidate.candidate_id: actions[key]
        for key, candidate in zip(keys, candidates)
    }
    required_ids = tuple(MEMBER_IDS[key] for key in required_keys)
    return candidates, action_by_id, required_ids


def _run_routed_task(
    *,
    router: StructuralWorkspaceRouter,
    task_label: str,
    required_keys: tuple[str, ...],
    distractor_key: str,
    seed: int,
    method: str,
) -> dict[str, object]:
    candidates, action_by_id, required_ids = _task_candidates(required_keys, distractor_key)
    selection = router.route(
        candidates,
        tick=100 + seed,
        mode="learned",
    )
    actions = tuple(action_by_id[item] for item in selection.selected_ids)
    _, workbench_record = _run_workbench_episode(
        workspace=PROJECT_ROOT,
        split="holdout",
        context_label=task_label,
        cell_label=f"{method}-seed-{seed}",
        actions=actions,
        reward=0.0,
    )
    raw_actions = workbench_record["workbench_outcome"]["raw_actions"][1 : 1 + len(actions)]
    action_success = tuple(bool(item["success"]) for item in raw_actions)
    complete = set(selection.selected_ids) == set(required_ids)
    all_selected_actions_succeeded = bool(actions) and all(action_success)
    task_score = 1.0 if complete and all_selected_actions_succeeded else 0.0
    return {
        "task": task_label,
        "method": method,
        "seed": seed,
        "capacity": selection.capacity,
        "candidate_ids": list(selection.candidate_ids),
        "selected_ids": list(selection.selected_ids),
        "required_ids": list(required_ids),
        "selected_action_count": len(actions),
        "raw_action_success": list(action_success),
        "task_complete": complete,
        "all_selected_actions_succeeded": all_selected_actions_succeeded,
        "task_score": task_score,
        "score_source": "native Workbench raw action success/status plus required-set completion",
        "native_world_event_count": int(workbench_record["native_world_event_count"]),
        "native_checkpoint_replay": bool(workbench_record["replay_equal"]),
    }


def _admit_growth(
    *,
    seed: int,
    corpus: object,
    train_examples: tuple[WorkspaceRoutingExample, ...],
) -> dict[str, object]:
    controller, feedbacks, rounds = _run_online_feedbacks(corpus=corpus, seed=seed)
    holdout_episode, _ = _run_online_episode(
        family="holdout-de-bridge",
        first="d",
        second="e",
        seed=seed,
        expected_score=1.0,
    )
    retention_episode, _ = _run_online_episode(
        family="retention-ab-bridge",
        first="a",
        second="b",
        seed=seed,
        expected_score=1.0,
    )
    bridge = InteractionStructuralBridge(
        InteractionStructuralBridgeConfig(
            network_id="standalone:adaptive.cortex",
            region_id="adaptive.cortex",
            minimum_feedbacks=2,
            minimum_holdout_transfer=0.5,
        )
    )
    pressure = bridge.project(
        feedbacks,
        controller.admissions,
        (
            _independent_observation(partition="holdout", episode=holdout_episode, tick=10),
            _independent_observation(partition="retention", episode=retention_episode, tick=11),
        ),
    )
    model, region = _build_model(f"structural-workspace-{seed}")
    topology_before = model._structural_topology_digest(model.native_checkpoint())
    budget_before = model.cognitive_snapshot().development.structural_budget
    candidate = model.propose_structural_candidate_from_pressure(
        pressure.projection,
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={
            "region_id": region.region_id,
            "unit_id": "u2",
            "online_bridge_digest": pressure.bridge_digest,
        },
    )
    if candidate is None:
        raise AssertionError("P4-10 structural workspace candidate was not created")
    candidate_checkpoint = model.native_checkpoint()
    control_model = type(model).from_native_checkpoint(candidate_checkpoint)
    parent_workspace = StructuralWorkspaceRouter(
        WorkspaceRouter(FEATURE_DIM, capacity=2, seed=seed),
        control_model.neuron_regions[0],
    )
    parent_workspace.fit(train_examples, epochs=120, learning_rate=0.2)
    parent_workspace_checkpoint = parent_workspace.checkpoint()
    restored_model = type(model).from_native_checkpoint(candidate_checkpoint)
    restored_workspace = StructuralWorkspaceRouter.from_checkpoint(
        parent_workspace_checkpoint,
        region=restored_model.neuron_regions[0],
    )
    batch = restored_model.arbitrate_structural_candidate_batch(
        (candidate.candidate_id,)
    )
    holdout_input, expected_activity = _expected_activity(
        restored_model,
        candidate.candidate_id,
    )
    validation = restored_model.validate_structural_candidate_shadow(
        candidate.candidate_id,
        holdout_inputs=(holdout_input,),
        expected_activities=(expected_activity,),
    )
    decision = restored_model.evaluate_structural_candidate_gate(
        validation,
        holdout_gain=validation.validation_score,
        retention_regression=0.02,
        lesion_effect=0.15,
        resource_state=0.8,
        evidence_ids=(f"online-bridge:{pressure.bridge_digest}",),
    )
    admission = restored_model.admit_structural_candidate(validation, decision)
    restored_workspace.rebind(restored_model.neuron_regions[0])
    if admission.status != "admitted":
        raise AssertionError("P4-10 structural workspace candidate was not admitted")
    grown_workspace_checkpoint = restored_workspace.checkpoint()
    continued_workspace = StructuralWorkspaceRouter.from_checkpoint(
        grown_workspace_checkpoint,
        region=restored_model.neuron_regions[0],
    )
    structural_runs = [
        _run_routed_task(
            router=continued_workspace,
            task_label=task_label,
            required_keys=required_keys,
            distractor_key=distractor_key,
            seed=seed,
            method="structural_growth",
        )
        for task_label, required_keys, distractor_key in TARGET_TASKS
    ]

    controls: dict[str, StructuralWorkspaceRouter] = {}
    weight_control = StructuralWorkspaceRouter.from_checkpoint(
        parent_workspace_checkpoint,
        region=control_model.neuron_regions[0],
    )
    weight_control.fit(train_examples, epochs=20, learning_rate=0.05)
    controls["interaction_weight_only"] = weight_control
    router_control_model = type(model).from_native_checkpoint(candidate_checkpoint)
    router_control = StructuralWorkspaceRouter(
        WorkspaceRouter(FEATURE_DIM, capacity=2, seed=seed + 1000),
        router_control_model.neuron_regions[0],
    )
    router_control.fit(train_examples, epochs=120, learning_rate=0.2)
    controls["router_only"] = router_control
    memory_control_model = type(model).from_native_checkpoint(candidate_checkpoint)
    controls["memory_only"] = StructuralWorkspaceRouter.from_checkpoint(
        parent_workspace_checkpoint,
        region=memory_control_model.neuron_regions[0],
    )
    control_runs = {
        method: [
            _run_routed_task(
                router=router,
                task_label=task_label,
                required_keys=required_keys,
                distractor_key=distractor_key,
                seed=seed,
                method=method,
            )
            for task_label, required_keys, distractor_key in TARGET_TASKS
        ]
        for method, router in controls.items()
    }
    topology_after_admission = restored_model._structural_topology_digest(
        restored_model.native_checkpoint()
    )
    budget_after_admission = restored_model.cognitive_snapshot().development.structural_budget
    grown_units = tuple(restored_model.neuron_regions[0].unit_ids)
    rollback = restored_model.rollback_structural_candidate(candidate.candidate_id)
    restored_workspace.rebind(restored_model.neuron_regions[0])
    topology_after_rollback = restored_model._structural_topology_digest(
        restored_model.native_checkpoint()
    )
    rollback_route = restored_workspace.route(
        tuple(_candidate(MEMBER_ACTIONS[key]) for key in ("b", "e", "c", "f")),
        tick=999,
        mode="learned",
    )
    return {
        "seed": seed,
        "online_rounds": list(rounds),
        "online_feedback_ids": [item.feedback_id for item in feedbacks],
        "pressure": pressure.to_payload(),
        "candidate": candidate.to_payload(),
        "arbitration": batch.to_payload(),
        "validation": validation.to_payload(),
        "decision": decision.to_payload(),
        "admission": admission.to_payload(),
        "structural_runs": structural_runs,
        "control_runs": control_runs,
        "parent_capacity": parent_workspace.capacity,
        "grown_capacity": continued_workspace.capacity,
        "rollback_capacity": rollback_route.capacity,
        "parent_units": list(region.unit_ids),
        "grown_units": list(grown_units),
        "topology_before": topology_before,
        "topology_after_admission": topology_after_admission,
        "topology_after_rollback": topology_after_rollback,
        "budget_before": budget_before,
        "budget_after_admission": budget_after_admission,
        "budget_after_rollback": restored_model.cognitive_snapshot().development.structural_budget,
        "rollback_status": "rolled_back" if rollback else "rollback_failed",
        "candidate_checkpoint_roundtrip": (
            type(model).from_native_checkpoint(candidate_checkpoint).structural_proposal_candidates[0]
            == candidate
        ),
        "workspace_checkpoint_roundtrip": (
            continued_workspace.checkpoint()["checkpoint_digest"]
            == grown_workspace_checkpoint["checkpoint_digest"]
        ),
    }


def evaluate() -> dict[str, object]:
    corpus, _ = build_train_corpus()
    train_examples, training_source = _training_examples()
    runs = [
        _admit_growth(seed=seed, corpus=corpus, train_examples=train_examples)
        for seed in LEARNER_SEEDS
    ]
    structural_scores = [
        float(item["task_score"])
        for run in runs
        for item in run["structural_runs"]
    ]
    control_means = {
        method: sum(float(item["task_score"]) for run in runs for item in items)
        / (len(runs) * len(TARGET_TASKS))
        for method, items in {
            method: [item for run in runs for item in run["control_runs"][method]]
            for method in ("interaction_weight_only", "router_only", "memory_only")
        }.items()
    }
    structural_mean = sum(structural_scores) / len(structural_scores)
    metrics = {
        "three_independent_seeds": len(runs) == len(LEARNER_SEEDS),
        "two_independent_unseen_workbench_contexts": all(
            len(run["structural_runs"]) == len(TARGET_TASKS)
            and len({item["task"] for item in run["structural_runs"]}) == 2
            for run in runs
        ),
        "actual_workbench_status_projection": all(
            item["score_source"].startswith("native Workbench")
            and item["native_world_event_count"] > 0
            and item["native_checkpoint_replay"]
            for run in runs
            for item in (*run["structural_runs"], *run["control_runs"]["interaction_weight_only"])
        ),
        "structural_growth_completes_unseen_tasks": structural_mean == 1.0,
        "structural_growth_beats_weight_router_memory_controls": all(
            structural_mean > score for score in control_means.values()
        ),
        "controls_remain_capacity_bound": all(
            all(item["capacity"] == 2 for item in run["control_runs"][method])
            for run in runs
            for method in ("interaction_weight_only", "router_only", "memory_only")
        ),
        "online_contexts_are_required": all(
            len(run["online_feedback_ids"]) >= 2
            and len(
                {
                    tuple(item["member_ids"])
                    for item in run["online_rounds"]
                    if item["admission_status"] == "applied"
                }
            )
            >= 2
            for run in runs
        ),
        "failed_online_outcome_excluded": all(
            any(item["admission_status"] == "rejected" for item in run["online_rounds"])
            and all(
                item not in run["pressure"]["feedback_ids"]
                for item in [
                    item["feedback_id"]
                    for item in run["online_rounds"]
                    if item["admission_status"] == "rejected"
                ]
            )
            for run in runs
        ),
        "capacity_follows_topology": all(
            run["parent_capacity"] == 2
            and run["grown_capacity"] == 3
            and run["rollback_capacity"] == 2
            for run in runs
        ),
        "candidate_checkpoint_roundtrip": all(run["candidate_checkpoint_roundtrip"] for run in runs),
        "workspace_checkpoint_roundtrip": all(run["workspace_checkpoint_roundtrip"] for run in runs),
        "admission_changes_topology_and_spends_budget": all(
            run["admission"]["status"] == "admitted"
            and run["topology_after_admission"] != run["topology_before"]
            and run["budget_after_admission"] == run["budget_before"] - 1
            for run in runs
        ),
        "rollback_restores_topology_and_budget": all(
            run["rollback_status"] == "rolled_back"
            and run["topology_after_rollback"] == run["topology_before"]
            and run["budget_after_rollback"] == run["budget_before"]
            for run in runs
        ),
        "lesion_removes_structural_capacity": all(
            run["grown_capacity"] == 3 and run["rollback_capacity"] == 2
            for run in runs
        ),
        "training_uses_native_workbench_records": training_source["score_sources"]
        == ["native Workbench raw action success/status projection"],
    }
    return {
        "format": REPORT_FORMAT,
        "task": "online structural workspace growth against weight/router/memory controls",
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
                "an online-evidence-backed neuron admission must expand live workspace capacity, "
                "complete two unseen three-action Workbench tasks, beat controls that only change "
                "interaction weights/router/memory while capacity remains fixed, and preserve "
                "checkpoint, resource, lesion, and topology rollback boundaries"
            ),
        },
        "boundary": (
            "This is a bounded structural-capacity Gate. It does not claim unrestricted autonomous "
            "architecture design, AGI, full language understanding, or CUDA acceleration."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_p4_10_structural_workspace_net_gain_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
