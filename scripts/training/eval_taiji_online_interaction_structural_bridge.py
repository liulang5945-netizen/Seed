"""Evaluate the guarded bridge from online Workbench Outcomes to structure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_interaction_groups import (  # noqa: E402
    _run_workbench_episode,
)
from scripts.training.eval_taiji_open_domain_interaction_gain import (  # noqa: E402
    MEMBER_ACTIONS,
    MEMBER_IDS,
    _derive_score,
    _evaluator,
    _single_score,
    build_train_corpus,
)
from scripts.training.eval_taiji_structural_validation import (  # noqa: E402
    _build_model,
    _expected_activity,
)
from taiji import (  # noqa: E402
    InteractionGroupOnlineLearner,
    InteractionGroupOutcomeFeedback,
    InteractionGroupTransferLearner,
    InteractionStructuralBridge,
    InteractionStructuralBridgeConfig,
    Outcome,
    StructuralRuntimeObservation,
    build_member_evidence,
)

REPORT_FORMAT = "taiji-w7-p4-9-online-interaction-structural-bridge-v1"
LEARNER_SEEDS = (11, 29, 47)


def _target_member_ids(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((MEMBER_IDS[first], MEMBER_IDS[second])))


def _build_controller(*, corpus: object, seed: int) -> InteractionGroupOnlineLearner:
    train_revision = next(iter(corpus.train_checkpoint_revisions))
    profiles = build_member_evidence(
        corpus.train,
        source_trace_digest=corpus.train_trace_digest,
        checkpoint_revision=train_revision,
    )
    ordered_profiles = profiles if seed % 2 else tuple(reversed(profiles))
    learner = InteractionGroupTransferLearner(
        ridge=0.1,
        minimum_utility=-2.0,
        maximum_uncertainty=1.25,
    )
    train_groups = _evaluator().train_only_candidates(corpus)
    learner.observe_members(ordered_profiles)
    learner.observe_records(train_groups if seed % 3 else tuple(reversed(train_groups)))
    return InteractionGroupOnlineLearner(
        learner,
        minimum_interaction=0.1,
        maximum_feedback_uncertainty=1.25,
        maximum_resource_cost=2.0,
    )


def _run_online_episode(
    *,
    family: str,
    first: str,
    second: str,
    seed: int,
    expected_score: float,
) -> tuple[object, dict[str, object]]:
    actions = (MEMBER_ACTIONS[first], MEMBER_ACTIONS[second])
    episode, record = _run_workbench_episode(
        workspace=PROJECT_ROOT,
        split="online",
        context_label=family,
        cell_label=f"seed-{seed}",
        actions=actions,
        reward=expected_score,
    )
    derived_score = _derive_score(actions, record)
    if abs(float(derived_score) - expected_score) > 1e-12:
        raise AssertionError(
            f"native online Workbench score mismatch for {family}: "
            f"expected {expected_score}, derived {derived_score}"
        )
    return episode, {
        **record,
        "derived_task_score": derived_score,
        "score_source": "native Workbench raw action success/status projection",
    }


def _feedback_from_run(
    *,
    candidate: object,
    parent_digest: str,
    episode: object,
    record: dict[str, object],
    first: str,
    second: str,
) -> InteractionGroupOutcomeFeedback:
    outcome = Outcome.from_payload(record["workbench_outcome"]["outcome"])
    score = float(record["derived_task_score"])
    return InteractionGroupOutcomeFeedback.from_episode(
        candidate=candidate,
        parent_checkpoint_digest=parent_digest,
        episode=episode,
        outcome=outcome,
        realized_interaction=score - _single_score(first) - _single_score(second),
        contribution=score,
        uncertainty=float(candidate.uncertainty),
    )


def _independent_observation(
    *,
    partition: str,
    episode: object,
    tick: int,
) -> StructuralRuntimeObservation:
    return StructuralRuntimeObservation(
        network_id="standalone:adaptive.cortex",
        region_id="adaptive.cortex",
        tick=tick,
        usage=0.35,
        resource_pressure=0.2,
        prediction_error=0.15,
        learning_gain=0.25,
        holdout_transfer=0.8 if partition == "holdout" else 0.0,
        evidence_id=f"workbench-{partition}:outcome:{episode.outcome_id}",
        task_slice_id=f"workbench-{partition}:{episode.context_id}",
        partition=partition,
    )


def _run_online_feedbacks(
    *,
    corpus: object,
    seed: int,
) -> tuple[InteractionGroupOnlineLearner, tuple[object, ...], tuple[dict[str, object], ...]]:
    controller = _build_controller(corpus=corpus, seed=seed)
    feedbacks: list[InteractionGroupOutcomeFeedback] = []
    episodes: list[object] = []
    rounds: list[dict[str, object]] = []
    for round_index, (family, first, second, expected_score) in enumerate(
        (
            ("online-be", "b", "e", 1.0),
            ("online-bf", "b", "f", -1.0),
            ("online-ce", "c", "e", 1.0),
        ),
        start=1,
    ):
        parent = controller.checkpoint()
        members = _target_member_ids(first, second)
        if expected_score > 0.0:
            selected = controller.select((members,), resource_budget=2.0)
            if selected is None or selected[0].member_ids != members:
                raise AssertionError(f"online controller did not select target {family}")
            candidate = selected[1]
        else:
            candidate = controller.learner.candidate(members, allow_observed=False)
            if candidate is None:
                raise AssertionError(f"online failure candidate was unavailable: {family}")
        episode, record = _run_online_episode(
            family=family,
            first=first,
            second=second,
            seed=seed,
            expected_score=expected_score,
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
        episodes.append(episode)
        if admission.status == "applied":
            feedbacks.append(feedback)
        rounds.append(
            {
                "round": round_index,
                "family": family,
                "candidate_id": candidate.group_id,
                "member_ids": list(candidate.member_ids),
                "feedback_id": feedback.feedback_id,
                "outcome_id": feedback.outcome_id,
                "outcome_success": feedback.outcome.success,
                "admission_status": admission.status,
                "admission_reason": admission.reason,
                "native_world_event_count": int(record["native_world_event_count"]),
                "native_checkpoint_replay": bool(record["replay_equal"]),
            }
        )
        if round_index == 2:
            controller = InteractionGroupOnlineLearner.from_checkpoint(controller.checkpoint())
    if len(feedbacks) != 2:
        raise AssertionError("online structural bridge requires two applied feedbacks")
    return controller, tuple(feedbacks), tuple(rounds)


def evaluate() -> dict[str, object]:
    corpus, train_records = build_train_corpus()
    runs: list[dict[str, object]] = []
    for seed in LEARNER_SEEDS:
        controller, feedbacks, rounds = _run_online_feedbacks(corpus=corpus, seed=seed)
        holdout_episode, holdout_record = _run_online_episode(
            family="holdout-de",
            first="d",
            second="e",
            seed=seed,
            expected_score=1.0,
        )
        retention_episode, retention_record = _run_online_episode(
            family="retention-ab",
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
        independent = (
            _independent_observation(
                partition="holdout", episode=holdout_episode, tick=10
            ),
            _independent_observation(
                partition="retention", episode=retention_episode, tick=11
            ),
        )
        pressure = bridge.project(feedbacks, controller.admissions, independent)
        if not pressure.projection.evidence_ids:
            raise AssertionError("structural bridge projection is missing evidence")
        failed_feedback = next(
            item
            for item in controller.admissions
            if item.status == "rejected"
        )
        failed_excluded = failed_feedback.feedback_id not in pressure.feedback_ids

        model, region = _build_model(f"online-structural-{seed}")
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
            raise AssertionError("online structural bridge did not produce a candidate")
        candidate_checkpoint = model.native_checkpoint()
        restored_model = type(model).from_native_checkpoint(candidate_checkpoint)
        restored_candidate = restored_model.structural_proposal_candidates[0]
        batch = restored_model.arbitrate_structural_candidate_batch(
            (restored_candidate.candidate_id,)
        )
        if restored_candidate.candidate_id not in batch.selected_candidate_ids:
            raise AssertionError("structural arbitration did not reserve the bridge candidate")
        holdout_input, expected_activity = _expected_activity(
            restored_model, restored_candidate.candidate_id
        )
        validation = restored_model.validate_structural_candidate_shadow(
            restored_candidate.candidate_id,
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
        topology_after_admission = restored_model._structural_topology_digest(
            restored_model.native_checkpoint()
        )
        units_after_admission = tuple(restored_model.neuron_regions[0].unit_ids)
        budget_after_admission = (
            restored_model.cognitive_snapshot().development.structural_budget
        )
        rollback = restored_model.rollback_structural_candidate(restored_candidate.candidate_id)
        topology_after_rollback = restored_model._structural_topology_digest(
            restored_model.native_checkpoint()
        )
        run = {
            "seed": seed,
            "rounds": list(rounds),
            "feedback_ids": [item.feedback_id for item in feedbacks],
            "pressure": pressure.to_payload(),
            "candidate": restored_candidate.to_payload(),
            "arbitration": batch.to_payload(),
            "validation": validation.to_payload(),
            "decision": decision.to_payload(),
            "admission": admission.to_payload(),
            "rollback": {
                "candidate_id": restored_candidate.candidate_id,
                "status": "rolled_back" if rollback else "rollback_failed",
            },
            "holdout_outcome_id": holdout_record["workbench_outcome"]["outcome"]["intent_id"],
            "retention_outcome_id": retention_record["workbench_outcome"]["outcome"]["intent_id"],
            "parent_units": list(region.unit_ids),
            "post_admission_units": list(units_after_admission),
            "roundtrip_candidate": restored_candidate.to_payload() == candidate.to_payload(),
            "topology_before": topology_before,
            "topology_after_admission": topology_after_admission,
            "topology_after_rollback": topology_after_rollback,
            "budget_before": budget_before,
            "budget_after_admission": budget_after_admission,
            "budget_after_rollback": restored_model.cognitive_snapshot().development.structural_budget,
            "failed_feedback_excluded": failed_excluded,
        }
        runs.append(run)

    metrics = {
        "three_independent_seeds": len(runs) == len(LEARNER_SEEDS),
        "actual_workbench_online_outcomes": all(
            all(
                bool(item["outcome_id"])
                and bool(item["native_world_event_count"])
                and bool(item["native_checkpoint_replay"])
                for item in run["rounds"]
            )
            for run in runs
        ),
        "failed_online_outcome_excluded": all(run["failed_feedback_excluded"] for run in runs),
        "pressure_contains_online_evidence": all(
            all(f"online-feedback:{feedback_id}" in run["pressure"]["projection"]["evidence_ids"]
                for feedback_id in run["feedback_ids"])
            for run in runs
        ),
        "independent_holdout_and_retention_required": all(
            run["pressure"]["projection"]["holdout_window_count"] >= 1
            and run["pressure"]["projection"]["retention_window_count"] >= 1
            for run in runs
        ),
        "candidate_checkpoint_roundtrip": all(run["roundtrip_candidate"] for run in runs),
        "arbitration_selected_candidate": all(
            run["candidate"]["candidate_id"] in run["arbitration"]["selected_candidate_ids"]
            for run in runs
        ),
        "shadow_validation_is_topology_neutral": all(
            run["validation"]["status"] == "validated"
            and run["validation"]["topology_before_digest"]
            == run["validation"]["topology_after_digest"]
            for run in runs
        ),
        "admission_changes_topology": all(
            run["admission"]["status"] == "admitted"
            and run["topology_after_admission"] != run["topology_before"]
            for run in runs
        ),
        "rollback_restores_topology_and_budget": all(
            run["rollback"]["status"] == "rolled_back"
            and run["topology_after_rollback"] == run["topology_before"]
            and run["budget_after_admission"] == run["budget_before"] - 1
            and run["budget_after_rollback"] == run["budget_before"]
            for run in runs
        ),
        "train_inputs_are_native_workbench_projected": all(
            record["score_source"] == "native Workbench raw action success/status projection"
            for record in train_records
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "task": "online interaction evidence to governed structural candidate bridge",
        "learner_seeds": list(LEARNER_SEEDS),
        "runs": runs,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "actual admitted online Outcomes must form a sealed structural pressure only "
                "with independent holdout and retention evidence; the existing candidate "
                "arbitration, shadow validation, admission, and rollback lifecycle must remain "
                "the only topology mutation path"
            ),
        },
        "boundary": (
            "This Gate proves a bounded online-to-structure bridge. It does not claim unrestricted "
            "self-evolution, general intelligence, autonomous architecture design, or CUDA gain."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "taiji_w7_p4_9_online_interaction_structural_bridge_20260831.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
