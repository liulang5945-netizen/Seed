"""Evaluate a second independent structural-growth cycle on real Workbench tasks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_online_interaction_structural_bridge import (  # noqa: E402
    _feedback_from_run,
    _independent_observation,
    _run_online_episode,
    _run_online_feedbacks,
)
from scripts.training.eval_taiji_open_domain_interaction_gain import (  # noqa: E402
    MEMBER_IDS,
    build_train_corpus,
)
from scripts.training.eval_taiji_structural_validation import (  # noqa: E402
    _expected_activity,
)
from scripts.training.eval_taiji_structural_workspace_net_gain import (  # noqa: E402
    FEATURE_DIM,
    _run_routed_task,
    _training_examples,
)
from taiji import (  # noqa: E402
    AdaptiveNeuronRegion,
    AdaptiveStructuralGrowthController,
    InteractionStructuralBridge,
    InteractionStructuralBridgeConfig,
    StructuralGrowthDynamics,
    StructuralWorkspaceRouter,
    TaijiConfig,
    TSKV8Adapter,
    WorkspaceRouter,
)

REPORT_FORMAT = "taiji-w7-p4-10b-continuous-structural-growth-v1"
LEARNER_SEEDS = (11, 29, 47)
STAGE_ONE_TASKS = (
    ("future-bec", ("b", "e", "c"), "f"),
    ("future-ade", ("a", "d", "e"), "g"),
)
STAGE_TWO_TASKS = (
    ("future-abce", ("a", "b", "c", "e"), "f"),
    ("future-acde", ("a", "c", "d", "e"), "g"),
)


def _build_model(episode_id: str) -> tuple[TSKV8Adapter, AdaptiveNeuronRegion]:
    config = TaijiConfig(
        alphabet_size=257,
        boundary_symbol=256,
        region_sizes=(8, 6),
        synapse_fan_in=3,
        motor_fan_in=4,
        lateral_fan_in=3,
        memory_units=12,
        memory_fan_in=3,
        memory_readout_fan_in=4,
        memory_meta_dim=4,
        memory_time_dim=4,
        memory_episode_dim=4,
        development_structural_budget=2,
        seed=71,
    )
    model = TSKV8Adapter(config, episode_id=episode_id)
    region = AdaptiveNeuronRegion(
        region_id="adaptive.cortex",
        input_dim=5,
        unit_ids=("u0", "u1"),
        fan_in=2,
        generator=torch.Generator().manual_seed(7),
    )
    model.attach_adaptive_neuron_region(region)
    model.attach_structural_growth_controller(
        AdaptiveStructuralGrowthController(
            dynamics=StructuralGrowthDynamics(
                ema_rate=1.0,
                error_threshold=0.0,
                holdout_transfer_threshold=0.0,
                minimum_resource_state=0.0,
                required_error_steps=1,
            )
        )
    )
    return model, region


def _extend_online_feedbacks(
    controller: object,
    feedbacks: tuple[object, ...],
    rounds: tuple[dict[str, object], ...],
    *,
    seed: int,
) -> tuple[object, tuple[object, ...], tuple[dict[str, object], ...]]:
    """Append new successful contexts after the first online cycle."""

    all_feedbacks = list(feedbacks)
    all_rounds = list(rounds)
    for round_index, (family, first, second) in enumerate(
        (("online-ae-next", "a", "e"), ("online-de-next", "d", "e")),
        start=4,
    ):
        members = tuple(sorted((MEMBER_IDS[first], MEMBER_IDS[second])))
        selected = controller.select((members,), resource_budget=2.0)
        if selected is None:
            raise AssertionError(f"second-cycle online controller did not select {family}")
        candidate = selected[1]
        parent = controller.checkpoint()
        episode, record = _run_online_episode(
            family=family,
            first=first,
            second=second,
            seed=seed,
            expected_score=1.0,
        )
        feedback = _feedback_from_run(
            candidate=candidate,
            parent_digest=str(parent["checkpoint_digest"]),
            episode=episode,
            record=record,
            first=first,
            second=second,
        )
        admission = controller.apply_feedback(feedback)
        if admission.status == "applied":
            all_feedbacks.append(feedback)
        all_rounds.append(
            {
                "round": round_index,
                "family": family,
                "member_ids": list(members),
                "feedback_id": feedback.feedback_id,
                "outcome_id": feedback.outcome_id,
                "outcome_success": feedback.outcome.success,
                "admission_status": admission.status,
                "admission_reason": admission.reason,
                "native_world_event_count": int(record["native_world_event_count"]),
                "native_checkpoint_replay": bool(record["replay_equal"]),
            }
        )
        if admission.status != "applied":
            raise AssertionError(f"second-cycle online feedback was not admitted: {family}")
    return controller, tuple(all_feedbacks), tuple(all_rounds)


def _pressure(
    feedbacks: tuple[object, ...],
    controller: object,
    *,
    seed: int,
    cycle: int,
) -> object:
    holdout_episode, _ = _run_online_episode(
        family=f"holdout-cycle-{cycle}",
        first="d",
        second="e",
        seed=seed,
        expected_score=1.0,
    )
    retention_episode, _ = _run_online_episode(
        family=f"retention-cycle-{cycle}",
        first="a",
        second="b",
        seed=seed,
        expected_score=1.0,
    )
    return InteractionStructuralBridge(
        InteractionStructuralBridgeConfig(
            network_id="standalone:adaptive.cortex",
            region_id="adaptive.cortex",
            minimum_feedbacks=2,
            minimum_holdout_transfer=0.5,
        )
    ).project(
        feedbacks,
        controller.admissions,
        (
            _independent_observation(
                partition="holdout", episode=holdout_episode, tick=10 + cycle
            ),
            _independent_observation(
                partition="retention", episode=retention_episode, tick=11 + cycle
            ),
        ),
    )


def _admit(
    model: TSKV8Adapter,
    pressure: object,
    *,
    unit_id: str,
) -> tuple[TSKV8Adapter, dict[str, object]]:
    region = model.neuron_regions[0]
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
            "unit_id": unit_id,
            "cycle_bridge_digest": pressure.bridge_digest,
        },
    )
    if candidate is None:
        raise AssertionError(f"structural candidate {unit_id} was not created")
    candidate_checkpoint = model.native_checkpoint()
    restored = type(model).from_native_checkpoint(candidate_checkpoint)
    restored_candidate = restored.structural_proposal_candidates[0]
    batch = restored.arbitrate_structural_candidate_batch((restored_candidate.candidate_id,))
    holdout_input, expected_activity = _expected_activity(
        restored, restored_candidate.candidate_id
    )
    validation = restored.validate_structural_candidate_shadow(
        restored_candidate.candidate_id,
        holdout_inputs=(holdout_input,),
        expected_activities=(expected_activity,),
    )
    decision = restored.evaluate_structural_candidate_gate(
        validation,
        holdout_gain=validation.validation_score,
        retention_regression=0.02,
        lesion_effect=0.15,
        resource_state=0.8,
        evidence_ids=(f"continuous-bridge:{pressure.bridge_digest}",),
    )
    admission = restored.admit_structural_candidate(validation, decision)
    topology_after = restored._structural_topology_digest(restored.native_checkpoint())
    budget_after = restored.cognitive_snapshot().development.structural_budget
    if admission.status != "admitted":
        raise AssertionError(f"structural candidate {unit_id} was not admitted")
    return restored, {
        "candidate_id": restored_candidate.candidate_id,
        "candidate": restored_candidate.to_payload(),
        "arbitration": batch.to_payload(),
        "validation": validation.to_payload(),
        "decision": decision.to_payload(),
        "admission": admission.to_payload(),
        "topology_before": topology_before,
        "topology_after": topology_after,
        "budget_before": budget_before,
        "budget_after": budget_after,
        "unit_ids": list(restored.neuron_regions[0].unit_ids),
        "candidate_checkpoint_roundtrip": (
            type(model).from_native_checkpoint(candidate_checkpoint).structural_proposal_candidates[0]
            == candidate
        ),
    }


def _control_routers(
    *,
    checkpoint: dict[str, object],
    workspace_checkpoint: dict[str, object],
    seed: int,
) -> dict[str, StructuralWorkspaceRouter]:
    controls: dict[str, StructuralWorkspaceRouter] = {}
    weight_model = TSKV8Adapter.from_native_checkpoint(checkpoint)
    weight_router = StructuralWorkspaceRouter.from_checkpoint(
        workspace_checkpoint,
        region=weight_model.neuron_regions[0],
    )
    controls["interaction_weight_only"] = weight_router
    router_model = TSKV8Adapter.from_native_checkpoint(checkpoint)
    router = StructuralWorkspaceRouter(
        WorkspaceRouter(FEATURE_DIM, capacity=2, seed=seed + 1000),
        router_model.neuron_regions[0],
    )
    controls["router_only"] = router
    memory_model = TSKV8Adapter.from_native_checkpoint(checkpoint)
    controls["memory_only"] = StructuralWorkspaceRouter.from_checkpoint(
        workspace_checkpoint,
        region=memory_model.neuron_regions[0],
    )
    return controls


def evaluate() -> dict[str, object]:
    corpus, _ = build_train_corpus()
    train_examples, training_source = _training_examples()
    runs: list[dict[str, object]] = []
    for seed in LEARNER_SEEDS:
        controller, feedbacks, first_rounds = _run_online_feedbacks(corpus=corpus, seed=seed)
        controller, feedbacks, all_rounds = _extend_online_feedbacks(
            controller,
            feedbacks,
            first_rounds,
            seed=seed,
        )
        first_pressure = _pressure(feedbacks[:2], controller, seed=seed, cycle=1)
        second_pressure = _pressure(feedbacks, controller, seed=seed, cycle=2)
        model, _ = _build_model(f"continuous-structural-{seed}")
        initial_checkpoint = model.native_checkpoint()
        workspace = StructuralWorkspaceRouter(
            WorkspaceRouter(FEATURE_DIM, capacity=2, seed=seed),
            model.neuron_regions[0],
        )
        workspace.fit(train_examples, epochs=120, learning_rate=0.2)
        parent_workspace_checkpoint = workspace.checkpoint()
        model, first_admission = _admit(model, first_pressure, unit_id="u2")
        workspace.rebind(model.neuron_regions[0])
        first_model_checkpoint = model.native_checkpoint()
        first_workspace_checkpoint = workspace.checkpoint()
        stage_one_after_first = [
            _run_routed_task(
                router=workspace,
                task_label=task_label,
                required_keys=required_keys,
                distractor_key=distractor_key,
                seed=seed,
                method="growth_cycle_1",
            )
            for task_label, required_keys, distractor_key in STAGE_ONE_TASKS
        ]
        resumed_model = type(model).from_native_checkpoint(first_model_checkpoint)
        resumed_workspace = StructuralWorkspaceRouter.from_checkpoint(
            first_workspace_checkpoint,
            region=resumed_model.neuron_regions[0],
        )
        resumed_model, second_admission = _admit(
            resumed_model,
            second_pressure,
            unit_id="u3",
        )
        resumed_workspace.rebind(resumed_model.neuron_regions[0])
        second_model_checkpoint = resumed_model.native_checkpoint()
        second_workspace_checkpoint = resumed_workspace.checkpoint()
        stage_one_after_second = [
            _run_routed_task(
                router=resumed_workspace,
                task_label=task_label,
                required_keys=required_keys,
                distractor_key=distractor_key,
                seed=seed,
                method="growth_cycle_2_retention",
            )
            for task_label, required_keys, distractor_key in STAGE_ONE_TASKS
        ]
        stage_two_after_second = [
            _run_routed_task(
                router=resumed_workspace,
                task_label=task_label,
                required_keys=required_keys,
                distractor_key=distractor_key,
                seed=seed,
                method="growth_cycle_2",
            )
            for task_label, required_keys, distractor_key in STAGE_TWO_TASKS
        ]
        controls = _control_routers(
            checkpoint=initial_checkpoint,
            workspace_checkpoint=parent_workspace_checkpoint,
            seed=seed,
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
                for task_label, required_keys, distractor_key in STAGE_TWO_TASKS
            ]
            for method, router in controls.items()
        }
        lesion_model = type(resumed_model).from_native_checkpoint(second_model_checkpoint)
        lesion_workspace = StructuralWorkspaceRouter.from_checkpoint(
            second_workspace_checkpoint,
            region=lesion_model.neuron_regions[0],
        )
        second_proposal = lesion_model.topology_proposals[-1]
        lesion_specification = tuple(
            (key, len(lesion_model.neuron_regions[0].unit_ids) if key == "existing_unit_count" else value)
            for key, value in second_proposal.specification
        )
        lesion_workspace.rebind(lesion_model.neuron_regions[0])
        lesion_model.neuron_regions[0].lesion_topology_proposal(
            replace(second_proposal, status="pending", specification=lesion_specification)
        )
        lesion_run = _run_routed_task(
            router=lesion_workspace,
            task_label="future-abce-lesion",
            required_keys=STAGE_TWO_TASKS[0][1],
            distractor_key=STAGE_TWO_TASKS[0][2],
            seed=seed,
            method="growth_cycle_2_lesion",
        )
        topology_before_rollback = resumed_model._structural_topology_digest(
            resumed_model.native_checkpoint()
        )
        second_rollback = resumed_model.rollback_structural_candidate(
            second_admission["candidate_id"]
        )
        resumed_workspace.rebind(resumed_model.neuron_regions[0])
        after_second_rollback_capacity = resumed_workspace.capacity
        first_rollback = resumed_model.rollback_structural_candidate(
            first_admission["candidate_id"]
        )
        resumed_workspace.rebind(resumed_model.neuron_regions[0])
        run = {
            "seed": seed,
            "online_rounds": list(all_rounds),
            "online_feedback_ids": [item.feedback_id for item in feedbacks],
            "first_pressure": first_pressure.to_payload(),
            "second_pressure": second_pressure.to_payload(),
            "first_admission": first_admission,
            "second_admission": second_admission,
            "stage_one_after_first": stage_one_after_first,
            "stage_one_after_second": stage_one_after_second,
            "stage_two_after_second": stage_two_after_second,
            "control_runs": control_runs,
            "lesion_run": lesion_run,
            "parent_capacity": 2,
            "first_growth_capacity": 3,
            "second_growth_capacity": 4,
            "after_second_rollback_capacity": after_second_rollback_capacity,
            "final_capacity": resumed_workspace.capacity,
            "topology_before_second_rollback": topology_before_rollback,
            "topology_after_full_rollback": resumed_model._structural_topology_digest(
                resumed_model.native_checkpoint()
            ),
            "budget_after_full_rollback": resumed_model.cognitive_snapshot().development.structural_budget,
            "second_rollback_status": "rolled_back" if second_rollback else "rollback_failed",
            "first_rollback_status": "rolled_back" if first_rollback else "rollback_failed",
            "second_workspace_checkpoint_roundtrip": (
                StructuralWorkspaceRouter.from_checkpoint(
                    second_workspace_checkpoint,
                    region=type(resumed_model).from_native_checkpoint(
                        second_model_checkpoint
                    ).neuron_regions[0],
                ).checkpoint()["checkpoint_digest"]
                == second_workspace_checkpoint["checkpoint_digest"]
            ),
        }
        runs.append(run)

    structural_stage_two = [
        float(item["task_score"])
        for run in runs
        for item in run["stage_two_after_second"]
    ]
    structural_mean = sum(structural_stage_two) / len(structural_stage_two)
    control_means = {
        method: sum(
            float(item["task_score"])
            for run in runs
            for item in run["control_runs"][method]
        )
        / (len(runs) * len(STAGE_TWO_TASKS))
        for method in ("interaction_weight_only", "router_only", "memory_only")
    }
    metrics = {
        "three_independent_seeds": len(runs) == len(LEARNER_SEEDS),
        "new_online_contexts_are_actual_and_successful": all(
            len(run["online_rounds"]) == 5
            and all(
                item["admission_status"] == "applied"
                and item["outcome_success"]
                and item["native_world_event_count"] > 0
                and item["native_checkpoint_replay"]
                for item in run["online_rounds"][3:]
            )
            for run in runs
        ),
        "second_pressure_contains_new_evidence": all(
            len(run["second_pressure"]["feedback_ids"]) == 4
            and all(
                f"online-feedback:{item}" in run["second_pressure"]["projection"]["evidence_ids"]
                for item in run["online_feedback_ids"]
            )
            for run in runs
        ),
        "failed_online_outcome_stays_excluded": all(
            any(item["admission_status"] == "rejected" for item in run["online_rounds"])
            and all(
                item["feedback_id"] not in run["second_pressure"]["feedback_ids"]
                for item in run["online_rounds"]
                if item["admission_status"] == "rejected"
            )
            for run in runs
        ),
        "first_growth_retains_after_second": all(
            all(item["task_score"] == 1.0 for item in run["stage_one_after_first"])
            and all(item["task_score"] == 1.0 for item in run["stage_one_after_second"])
            for run in runs
        ),
        "second_growth_completes_unseen_tasks": structural_mean == 1.0,
        "second_growth_beats_fixed_capacity_controls": all(
            structural_mean > score for score in control_means.values()
        ),
        "controls_stay_at_parent_capacity": all(
            all(item["capacity"] == 2 for item in run["control_runs"][method])
            for run in runs
            for method in ("interaction_weight_only", "router_only", "memory_only")
        ),
        "capacity_tracks_each_growth_cycle": all(
            run["parent_capacity"] == 2
            and run["first_growth_capacity"] == 3
            and run["second_growth_capacity"] == 4
            and run["after_second_rollback_capacity"] == 3
            and run["final_capacity"] == 2
            for run in runs
        ),
        "both_admissions_are_checkpointed_and_bounded": all(
            run["first_admission"]["candidate_checkpoint_roundtrip"]
            and run["second_admission"]["candidate_checkpoint_roundtrip"]
            and run["second_workspace_checkpoint_roundtrip"]
            and run["first_admission"]["budget_after"] == 1
            and run["second_admission"]["budget_after"] == 0
            for run in runs
        ),
        "lesion_removes_second_growth_gain": all(
            run["lesion_run"]["capacity"] == 3
            and run["lesion_run"]["task_score"] == 0.0
            for run in runs
        ),
        "full_rollback_restores_topology_and_budget": all(
            run["second_rollback_status"] == "rolled_back"
            and run["first_rollback_status"] == "rolled_back"
            and run["topology_after_full_rollback"]
            == run["first_admission"]["topology_before"]
            and run["budget_after_full_rollback"] == 2
            for run in runs
        ),
        "training_uses_native_workbench_records": training_source["score_sources"]
        == ["native Workbench raw action success/status projection"],
    }
    return {
        "format": REPORT_FORMAT,
        "task": "second independent online cycle and continuous structural growth",
        "stage_one_tasks": [item[0] for item in STAGE_ONE_TASKS],
        "stage_two_tasks": [item[0] for item in STAGE_TWO_TASKS],
        "learner_seeds": list(LEARNER_SEEDS),
        "structural_stage_two_mean_score": structural_mean,
        "control_mean_scores": control_means,
        "runs": runs,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "after one admitted growth, a new independent online evidence cycle must produce "
                "a second checkpointed neuron admission; prior tasks must be retained, two unseen "
                "four-action Workbench tasks must beat fixed-capacity weight/router/memory controls, "
                "lesion must remove the second-cycle gain, and sequential rollback must restore the parent"
            ),
        },
        "boundary": (
            "This is two-cycle bounded structural growth evidence. It does not claim unlimited growth, "
            "general intelligence, autonomous architecture design, or CUDA acceleration."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_p4_10b_continuous_structural_growth_20260831.json",
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
