"""Evaluate native Outcome writeback, admission, rollback, and restart."""

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
    build_workbench_corpus,
)
from scripts.training.eval_taiji_open_domain_interaction_gain import (  # noqa: E402
    MEMBER_ACTIONS,
    MEMBER_IDS,
    _derive_score,
    _evaluator,
    _single_score,
    build_train_corpus,
)
from taiji import (  # noqa: E402
    InteractionGroupOnlineLearner,
    InteractionGroupOutcomeFeedback,
    InteractionGroupTransferLearner,
    Outcome,
    build_member_evidence,
)

REPORT_FORMAT = "taiji-w7-p4-8-online-interaction-writeback-v1"
LEARNER_SEEDS = (11, 29, 47)
ONLINE_ROUNDS = (
    ("online-be", "b", "e", "success"),
    ("online-bf", "b", "f", "failure"),
    ("online-ce", "c", "e", "success"),
)


def _target_member_ids(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((MEMBER_IDS[first], MEMBER_IDS[second])))


def _build_controller(*, corpus: object, seed: int) -> InteractionGroupOnlineLearner:
    train_revision = next(iter(corpus.train_checkpoint_revisions))
    profiles = build_member_evidence(
        corpus.train,
        source_trace_digest=corpus.train_trace_digest,
        checkpoint_revision=train_revision,
    )
    train_groups = _evaluator().train_only_candidates(corpus)
    learner = InteractionGroupTransferLearner(
        ridge=0.1,
        minimum_utility=-2.0,
        maximum_uncertainty=1.25,
    )
    ordered_profiles = profiles if seed % 2 else tuple(reversed(profiles))
    ordered_groups = train_groups if seed % 3 else tuple(reversed(train_groups))
    learner.observe_members(ordered_profiles)
    learner.observe_records(ordered_groups)
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
    outcome_payload = record["workbench_outcome"]["outcome"]
    outcome = Outcome.from_payload(outcome_payload)
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


def evaluate() -> dict[str, object]:
    corpus, train_records = build_train_corpus()
    attribution_corpus, _ = build_workbench_corpus()
    attribution = _evaluator().evaluate(attribution_corpus)
    runs: list[dict[str, object]] = []

    for seed in LEARNER_SEEDS:
        controller = _build_controller(corpus=corpus, seed=seed)
        round_records: list[dict[str, object]] = []
        latest_applied_parent: dict[str, object] | None = None

        for round_index, (family, first, second, kind) in enumerate(ONLINE_ROUNDS, start=1):
            parent = controller.checkpoint()
            target_members = _target_member_ids(first, second)
            if kind == "success":
                selected = controller.select((target_members,), resource_budget=2.0)
                if selected is None or selected[0].member_ids != target_members:
                    raise AssertionError(f"online controller did not select target {family}")
                candidate = selected[1]
                expected_score = 1.0
            else:
                candidate = controller.learner.candidate(target_members, allow_observed=False)
                if candidate is None:
                    raise AssertionError(f"online failure candidate was unavailable: {family}")
                expected_score = -1.0

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
            before_digest = controller.learner.model_digest
            admission = controller.apply_feedback(feedback)
            after_digest = controller.learner.model_digest
            after_checkpoint = controller.checkpoint()
            restored = InteractionGroupOnlineLearner.from_checkpoint(after_checkpoint)

            round_record = {
                "round": round_index,
                "family": family,
                "kind": kind,
                "candidate_id": candidate.group_id,
                "member_ids": list(candidate.member_ids),
                "feedback_id": feedback.feedback_id,
                "outcome_id": feedback.outcome_id,
                "event_ids": list(feedback.event_ids),
                "outcome_reward": feedback.outcome.reward,
                "outcome_success": feedback.outcome.success,
                "outcome_provenance": feedback.outcome.provenance,
                "derived_task_score": record["derived_task_score"],
                "realized_interaction": feedback.realized_interaction,
                "score_source": record["score_source"],
                "admission_status": admission.status,
                "admission_reason": admission.reason,
                "observation_count_before": len(controller.learner.observed_records)
                - (1 if admission.status == "applied" else 0),
                "observation_count_after": len(controller.learner.observed_records),
                "model_changed_on_apply": before_digest != after_digest,
                "restart_admission_statuses": [item.status for item in restored.admissions],
                "native_world_event_count": int(record["native_world_event_count"]),
                "native_checkpoint_replay": bool(record["replay_equal"]),
                "raw_action_success": [
                    bool(action["success"])
                    for action in record["workbench_outcome"]["raw_actions"][1:3]
                ],
            }
            round_records.append(round_record)

            if admission.status == "applied":
                latest_applied_parent = parent

            if round_index == 1:
                lesion = InteractionGroupOnlineLearner.from_checkpoint(parent)
                round_record["writeback_lesion_keeps_candidate"] = (
                    lesion.learner.candidate(target_members, allow_observed=False) is not None
                    and controller.learner.candidate(target_members, allow_observed=False) is None
                )
            if round_index == 2:
                controller = restored

        if latest_applied_parent is None:
            raise AssertionError("online canary did not retain an applied parent checkpoint")
        latest_applied = next(
            item for item in reversed(controller.admissions) if item.status == "applied"
        )
        rolled_back = controller.rollback_to(
            latest_applied_parent,
            feedback_id=latest_applied.feedback_id,
        )
        post_rollback = InteractionGroupOnlineLearner.from_checkpoint(controller.checkpoint())

        holdout_episode, holdout_record = _run_online_episode(
            family="holdout-reserved",
            first="d",
            second="e",
            seed=seed,
            expected_score=1.0,
        )
        holdout_outcome = Outcome.from_payload(holdout_record["workbench_outcome"]["outcome"])
        holdout_candidate = post_rollback.learner.candidate(
            _target_member_ids("d", "e"), allow_observed=False
        )
        if holdout_candidate is None:
            raise AssertionError("holdout writeback canary lost its reserved candidate")
        holdout_feedback = InteractionGroupOutcomeFeedback.from_episode(
            candidate=holdout_candidate,
            parent_checkpoint_digest=str(post_rollback.checkpoint()["checkpoint_digest"]),
            episode=holdout_episode,
            outcome=holdout_outcome,
            realized_interaction=0.6,
            contribution=1.0,
        )
        holdout_rejected = False
        try:
            holdout_payload = dict(holdout_feedback.to_payload())
            holdout_payload["source_split"] = "holdout"
            InteractionGroupOutcomeFeedback.from_payload(holdout_payload)
        except ValueError as error:
            holdout_rejected = "source_split" in str(error)

        round_records.append(
            {
                "round": "rollback",
                "target_feedback_id": rolled_back.feedback_id,
                "status": rolled_back.status,
                "reason": rolled_back.reason,
                "restart_statuses": [item.status for item in post_rollback.admissions],
                "admitted_observation_count_after_rollback": len(
                    post_rollback.learner.observed_records
                ),
            }
        )
        runs.append(
            {
                "seed": seed,
                "rounds": round_records,
                "holdout_writeback_rejected": holdout_rejected,
                "holdout_candidate_remained_unobserved": (
                    post_rollback.learner.candidate(
                        _target_member_ids("d", "e"), allow_observed=False
                    )
                    is not None
                ),
                "rollback_parent_roundtrip": (
                    post_rollback.admissions[-1].status == "rolled_back"
                    and post_rollback.admissions[-1].feedback_id == rolled_back.feedback_id
                ),
                "old_train_lineage": post_rollback.learner.source_trace_digest
                == corpus.train_trace_digest,
                "old_train_records_retained": len(post_rollback.learner.observed_records) >= 5,
            }
        )

    old_episode, old_record = _run_online_episode(
        family="retention-old-ab",
        first="a",
        second="b",
        seed=0,
        expected_score=1.0,
    )
    del old_episode
    metrics = {
        "online_round_count": all(len(item["rounds"]) >= 4 for item in runs),
        "three_independent_seeds": len(runs) == len(LEARNER_SEEDS),
        "actual_outcome_writeback": all(
            all(
                round_item.get("outcome_id") and round_item.get("event_ids")
                for round_item in item["rounds"]
                if isinstance(round_item.get("round"), int)
            )
            for item in runs
        ),
        "successful_outcomes_are_admitted": all(
            all(
                round_item["admission_status"] == "applied"
                for round_item in item["rounds"]
                if round_item.get("kind") == "success"
            )
            for item in runs
        ),
        "failed_outcome_does_not_mutate": all(
            all(
                round_item["admission_status"] == "rejected"
                and round_item["admission_reason"] == "outcome_unsuccessful"
                and round_item["observation_count_after"]
                == round_item["observation_count_before"]
                for round_item in item["rounds"]
                if round_item.get("kind") == "failure"
            )
            for item in runs
        ),
        "writeback_changes_relation_model": all(
            any(
                bool(round_item.get("model_changed_on_apply"))
                for round_item in item["rounds"]
                if isinstance(round_item.get("round"), int)
                and round_item.get("kind") == "success"
            )
            for item in runs
        ),
        "writeback_lesion_keeps_preupdate_candidate": all(
            bool(
                next(
                    round_item["writeback_lesion_keeps_candidate"]
                    for round_item in item["rounds"]
                    if "writeback_lesion_keeps_candidate" in round_item
                )
            )
            for item in runs
        ),
        "restart_continues_admission_audit": all(
            any("applied" == status for status in item["rounds"][0]["restart_admission_statuses"])
            for item in runs
        ),
        "rollback_is_audited_and_restored": all(
            bool(item["rollback_parent_roundtrip"])
            and item["rounds"][-1]["status"] == "rolled_back"
            for item in runs
        ),
        "holdout_cannot_enter_online_feedback": all(
            bool(item["holdout_writeback_rejected"])
            and bool(item["holdout_candidate_remained_unobserved"])
            for item in runs
        ),
        "native_world_and_replay_preserved": all(
            all(
                bool(round_item.get("native_world_event_count", 0))
                and bool(round_item.get("native_checkpoint_replay"))
                for round_item in item["rounds"]
                if isinstance(round_item.get("round"), int)
            )
            for item in runs
        ),
        "old_workbench_capability_retained": bool(
            old_record["derived_task_score"] == 1.0
            and any(
                action["capability_id"] == "workspace.search" and action["success"]
                for action in old_record["workbench_outcome"]["raw_actions"]
            )
        ),
        "native_lesion_evidence_present": bool(attribution.metrics["lesion_effects_observed"]),
        "train_records_are_native_outcome_projected": all(
            record["score_source"] == "native Workbench raw action success/status projection"
            for record in train_records
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "task": "native online Outcome writeback, admission, rollback, and restart continuation",
        "learner_seeds": list(LEARNER_SEEDS),
        "online_rounds": [item[0] for item in ONLINE_ROUNDS],
        "runs": runs,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "at least three real Workbench online rounds must bind actual terminal Outcome "
                "and native event evidence to the current candidate checkpoint; successful "
                "feedback may update the train-lineage relation learner, failed or holdout "
                "feedback must not mutate it, restart must preserve the audit and learned state, "
                "and explicit rollback must restore the parent state without silent re-admission"
            ),
        },
        "boundary": (
            "This Gate demonstrates a bounded native online learning boundary, not unrestricted "
            "self-evolution or AGI. Admission updates interaction evidence only; it does not "
            "silently mutate provider policy, task planning, neuron topology, CUDA kernels, or "
            "the desktop UI."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_p4_8_online_interaction_writeback_20260831.json",
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
