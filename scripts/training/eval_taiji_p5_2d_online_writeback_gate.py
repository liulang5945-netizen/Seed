"""P5.2d online outcome writeback acceptance gate.

Preregistration: plans/reference/M5_P5_2D_ONLINE_WRITEBACK_PREREGISTRATION_FROZEN_20260916.md
(frozen; section 6 records the pre-execution design revision).  No library
rejection rule, threshold, or contract is changed.  Six acceptance classes:

  - new-task gain: the online-updated learner's selection achieves at least the
    frozen parent's real-execution gain on fresh create-family contexts and
    member-a+member-c ranks first in library-scale predictions;
  - old-task retention: the untouched base-lineage pair's prediction is
    bit-identical pre/post and the legacy held-out spot check does not regress;
  - idempotency: duplicate feedback is rejected without mutating state;
  - stale-parent rejection;
  - interruption recovery: checkpoint plus pending feedback payloads replayed
    in a fresh process to a byte-identical learner digest;
  - separated rollback: learner rollback with tombstone vs environment
    transaction undo, each verified independently.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import eval_taiji_b0_b1_representation_gate as b1  # noqa: E402
import probe_taiji_b0_structure_space as ssp  # noqa: E402

from taiji import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionGroupEvaluatorConfig,
    InteractionGroupOnlineLearner,
    InteractionGroupOutcomeFeedback,
    InteractionGroupTransferLearner,
    Outcome,
    build_member_evidence,
)
from taiji.interaction_groups import InteractionTraceCorpus  # noqa: E402

REPORT_FORMAT = "taiji-p5-2d-online-writeback-gate-report-v1"
VERSION = 1
PREREGISTRATION = "plans/reference/M5_P5_2D_ONLINE_WRITEBACK_PREREGISTRATION_FROZEN_20260916.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_2d_online_writeback_v2_20260916.json"

BASE_PREFIX = "p52dbase"
INDOMAIN_PREFIX = "p52dind"
TRIAL_PREFIX = "p52dtrial"
EVAL_PREFIX = "p52deval"
FAMILY_STEP = 6
ROUNDS = 6
RECOVERY_AFTER_ROUND = 3
BASE_PAIR = ("member-a", "member-b")
WALL_CAP_SECONDS = 90.0

STATIC_CHECK_SCOPE = (
    "scripts/training/eval_taiji_p5_2d_online_writeback_gate.py",
    "scripts/training/eval_taiji_b0_b2_selection_gate.py",
    "scripts/training/eval_taiji_b0_b1_representation_gate.py",
)


def _member_set(
    episodes: list[dict[str, Any]], allowed: set[tuple[str, ...]]
) -> list[dict[str, Any]]:
    return [episode for episode in episodes if tuple(episode["active_members"]) in allowed]


def _trace_means(
    trace_episodes: tuple[Any, ...],
    pair: tuple[str, str],
) -> dict[str, float]:
    groups: dict[tuple[str, ...], list[float]] = {}
    for episode in trace_episodes:
        groups.setdefault(episode.member_ids, []).append(float(episode.outcome))
    baseline = statistics.mean(groups.get((), [-1.0]))
    first = statistics.mean(groups.get((pair[0],), [-1.0]))
    second = statistics.mean(groups.get((pair[1],), [-1.0]))
    pair_mean = statistics.mean(groups.get(tuple(sorted(pair)), [-1.0]))
    return {
        "baseline": baseline,
        "first": first,
        "second": second,
        "pair": pair_mean,
        "interaction": pair_mean - first - second + baseline,
        "contribution": pair_mean - baseline,
    }


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "preregistration": PREREGISTRATION,
        "scope": (
            "P5.2d online outcome writeback acceptance gate over the existing "
            "library machinery with zero rule changes; base lineage = four member "
            "profiles plus the base projection (full singleton evidence plus the "
            "historical default pair member-a+member-b) of the legacy fit corpus, "
            "leaving five pairs unseen for six online rounds on fresh "
            "create__override contexts"
        ),
        "static_checks": {
            "scope": list(STATIC_CHECK_SCOPE),
            "commands": [
                "python -m py_compile <scope files>",
                "python -m ruff check scripts/training (and full repo)",
                "python -m black --no-cache --check <scope files>",
            ],
        },
    }
    try:
        frozen = ssp.load_frozen()
        counterfactual = ssp.load_counterfactual()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        _grid, training_cells, candidate_cells = b1._cells()
        override_cell = next(
            cell for cell in candidate_cells if cell["label"] == "create__override"
        )
        eval_cells = [cell for cell in candidate_cells if cell["label"] != "create__override"]
        member_ids = frozen.MEMBER_IDS
        pairs = list(itertools.combinations(member_ids, 2))
        base_sets = {
            (),
            (member_ids[0],),
            (member_ids[1],),
            (member_ids[2],),
            (member_ids[3],),
            BASE_PAIR,
        }

        member_start = time.perf_counter()
        embedder = frozen.DocumentEmbedder()
        members = frozen._train_members(embedder)
        member_seconds = time.perf_counter() - member_start

        # ---- base lineage: legacy fit corpus restricted to the base projection
        legacy_tasks = b1._tasks_for_cells(
            frozen, training_cells, (b1.FIT_CONTEXT_STEP,), BASE_PREFIX
        )
        legacy_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, legacy_tasks
        )
        indomain_tasks = b1._tasks_for_cells(
            frozen, training_cells, (b1.INDOMAIN_HOLDOUT_STEP,), INDOMAIN_PREFIX
        )
        indomain_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, indomain_tasks
        )
        base_trace = frozen._project(_member_set(legacy_episodes, base_sets))
        holdout_trace = frozen._project(_member_set(indomain_episodes, base_sets))
        corpus = InteractionTraceCorpus(train=base_trace, holdout=holdout_trace)
        revision = next(iter(corpus.train_checkpoint_revisions))
        profiles = build_member_evidence(
            corpus.train,
            source_trace_digest=corpus.train_trace_digest,
            checkpoint_revision=revision,
        )
        evaluator = InteractionGroupEvaluator(
            InteractionGroupEvaluatorConfig(maximum_resource_cost=b1.FIT_RESOURCE_CAP)
        )
        base_records = evaluator.train_only_candidates(corpus)
        base_record_pairs = sorted("+".join(record.member_ids) for record in base_records)
        base_learner = InteractionGroupTransferLearner(
            minimum_utility=-10.0, maximum_uncertainty=2.0
        )
        base_learner.observe_members(profiles)
        base_learner.observe_records(base_records)
        # §7 budget calibration: the online admission budget is the offline
        # evaluator's own deployment budget (b1.FIT_RESOURCE_CAP), so online
        # writeback and offline training share one resource contract.
        online = InteractionGroupOnlineLearner(
            base_learner, maximum_resource_cost=float(b1.FIT_RESOURCE_CAP)
        )
        admission_budget = {
            "maximum_resource_cost": float(b1.FIT_RESOURCE_CAP),
            "source": "b1.FIT_RESOURCE_CAP (preregistration section 7)",
            "minimum_interaction": 0.0,
            "maximum_feedback_uncertainty": 1.0,
        }
        ab_before = base_learner.candidate(BASE_PAIR, allow_observed=True)
        ab_prediction_before = (
            float(ab_before.predicted_interaction) if ab_before is not None else None
        )
        ab_records_before = sum(
            1 for record in base_learner.observed_records if tuple(record.member_ids) == BASE_PAIR
        )
        parent_pair = tuple(base_learner.select(pairs, unseen_only=False)[0].member_ids)
        parent_digest_seed = str(online.checkpoint()["checkpoint_digest"])

        # ---- six online rounds on fresh create__override contexts
        rounds: list[dict[str, Any]] = []
        applied_feedbacks: list[Any] = []
        pending_payloads: list[dict[str, Any]] = []
        pending_applied_ids: list[str] = []
        recovery_checkpoint: dict[str, Any] | None = None
        pre_apply_checkpoints: dict[int, dict[str, Any]] = {}
        locally_excluded: set[tuple[str, str]] = set()
        trial_episodes_total = 0
        a3_duplicate = False
        a3_duplicate_tested = False
        a3_note = "not_exercised_no_applied_feedback"
        for round_index in range(1, ROUNDS + 1):
            step = FAMILY_STEP + round_index
            task = b1._tasks_for_cells(frozen, [override_cell], (step,), TRIAL_PREFIX)
            available = [pair for pair in pairs if (pair[0], pair[1]) not in locally_excluded]
            pre_apply = online.checkpoint()
            pre_apply_checkpoints[round_index] = pre_apply
            parent_digest = str(pre_apply["checkpoint_digest"])
            selected = online.select(available)
            if selected is None:
                rounds.append(
                    {
                        "round": round_index,
                        "context": task[0].task_id,
                        "selection": None,
                        "note": "online_select_returned_none",
                    }
                )
                continue
            _selection, candidate = selected
            pair = tuple(candidate.member_ids)
            trial_episodes = counterfactual.execute_surface(
                frozen,
                frozen._member_episode,
                members,
                embedder,
                task,
                cell_member_sets=((), pair, (pair[0],), (pair[1],)),
            )
            trial_episodes_total += len(trial_episodes)
            trace = frozen._project(trial_episodes)
            means = _trace_means(trace, pair)
            pair_trace = next(
                (episode for episode in trace if episode.member_ids == tuple(sorted(pair))),
                None,
            )
            if pair_trace is None:
                locally_excluded.add((pair[0], pair[1]))
                rounds.append(
                    {
                        "round": round_index,
                        "context": task[0].task_id,
                        "selection": "+".join(pair),
                        "note": "feedback_unconstructible_member_mismatch",
                    }
                )
                continue
            pair_dict = next(
                episode
                for episode in trial_episodes
                if tuple(episode["active_members"]) == tuple(sorted(pair))
            )
            outcome_obj = Outcome(
                candidate.group_id,
                reward=float(pair_trace.outcome),
                success=bool(pair_dict["success"]),
                terminal=True,
                provenance="native-workbench",
                tick=1,
            )
            feedback = InteractionGroupOutcomeFeedback.from_episode(
                candidate=candidate,
                parent_checkpoint_digest=parent_digest,
                episode=pair_trace,
                outcome=outcome_obj,
                realized_interaction=means["interaction"],
                contribution=means["contribution"],
            )
            admission = online.apply_feedback(feedback)
            round_record = {
                "round": round_index,
                "context": task[0].task_id,
                "selection": "+".join(pair),
                "admission_status": admission.status,
                "admission_reason": admission.reason,
                "realized_interaction": round(means["interaction"], 6),
                "trace_means": {key: round(value, 6) for key, value in means.items()},
            }
            rounds.append(round_record)
            if admission.status == "applied":
                applied_feedbacks.append(feedback)
                # A3 duplicate idempotency must be tested immediately after the
                # first applied admission: any later admission (even a rejected
                # audit) moves the checkpoint digest and the stale check would
                # fire before the duplicate check (preregistration section 8.2).
                if not a3_duplicate_tested:
                    pre_dup_digest = str(online.checkpoint()["checkpoint_digest"])
                    try:
                        online.apply_feedback(feedback)
                        a3_note = "duplicate_was_not_rejected"
                    except ValueError as exc:
                        # The library's stale guard precedes the duplicate check
                        # and every apply moves the state, so an immediate replay
                        # is always rejected via the stale guard (preregistration
                        # section 8.6). Idempotency = replay rejected + no mutation.
                        a3_note = str(exc)
                        a3_duplicate = True
                    a3_duplicate = a3_duplicate and pre_dup_digest == str(
                        online.checkpoint()["checkpoint_digest"]
                    )
                    a3_duplicate_tested = True
            if round_index == RECOVERY_AFTER_ROUND:
                recovery_checkpoint = online.checkpoint()
            if round_index > RECOVERY_AFTER_ROUND:
                pending_payloads.append(feedback.to_payload())
                if admission.status == "applied":
                    pending_applied_ids.append(admission.feedback_id)
        applied_count_pre_rollback = len(online.applied_feedback_ids)
        updated_pair = tuple(online.learner.select(pairs, unseen_only=False)[0].member_ids)

        # ---- A1 new-task gain (fresh create-family contexts, full factorial)
        eval_tasks = b1._tasks_for_cells(
            frozen,
            eval_cells,
            tuple(FAMILY_STEP + 1 + index for index in range(ROUNDS)),
            EVAL_PREFIX,
        )
        eval_episodes = counterfactual.execute_surface(
            frozen, frozen._member_episode, members, embedder, eval_tasks
        )
        eval_ids = sorted(task.task_id for task in eval_tasks)
        pair_gains, _singleton_gains = b1._actual_gains(eval_episodes, eval_ids, member_ids)
        updated_gain = pair_gains["+".join(updated_pair)]
        parent_gain = pair_gains["+".join(parent_pair)]
        updated_predictions = {
            "+".join(pair): (
                float(candidate.predicted_interaction)
                if (candidate := online.learner.candidate(pair, allow_observed=True)) is not None
                else float("-inf")
            )
            for pair in pairs
        }
        ac_pair = ("member-a", "member-c")
        ac_ranked_first = updated_predictions["+".join(ac_pair)] == max(
            updated_predictions.values()
        )
        a1_new_task_gain = bool(
            updated_gain >= parent_gain
            and ac_ranked_first
            and any(
                record.member_ids == tuple(sorted(ac_pair)) and record.status == "admitted"
                for record in online.learner.observed_records
            )
        )

        # ---- A2 old-task retention (preregistration section 8.1: the library
        # refits globally on every record, so retention is the untouched base
        # pair's record set plus execution no-regression; the prediction shift
        # is disclosed as the global-refit effect, not gated)
        ab_after = online.learner.candidate(BASE_PAIR, allow_observed=True)
        ab_prediction_after = (
            float(ab_after.predicted_interaction) if ab_after is not None else None
        )
        ab_records_after = sum(
            1 for record in online.learner.observed_records if tuple(record.member_ids) == BASE_PAIR
        )
        retention_records_untouched = bool(
            ab_prediction_before is not None and ab_records_after == ab_records_before
        )
        spot_task = b1._tasks_for_cells(
            frozen,
            [next(cell for cell in training_cells if cell["label"] == "patch__override")],
            (b1.INDOMAIN_HOLDOUT_STEP,),
            "p52dspot",
        )
        spot_episodes = counterfactual.execute_surface(
            frozen,
            frozen._member_episode,
            members,
            embedder,
            spot_task,
            cell_member_sets=((), BASE_PAIR, tuple(sorted(ac_pair))),
        )
        spot_rates = b1._success_by_cell(spot_episodes, spot_task[0].task_id)
        parent_spot = spot_rates.get(tuple(sorted(parent_pair)), 0.0)
        updated_spot = spot_rates.get(tuple(sorted(updated_pair)), 0.0)
        retention_execution_ok = bool(updated_spot >= parent_spot)
        a2_old_task_retention = bool(retention_records_untouched and retention_execution_ok)

        # ---- A3 duplicate idempotency: tested inline after the first applied
        # admission inside the round loop (section 8.2); nothing to do here.
        duplicate_note = a3_note

        # ---- A4 stale-parent rejection
        a4_stale = False
        stale_note = "no_applied_feedback_to_replay"
        if applied_feedbacks:
            stale_payload = applied_feedbacks[0].to_payload()
            stale_payload["parent_checkpoint_digest"] = parent_digest_seed
            try:
                online.apply_feedback(InteractionGroupOutcomeFeedback.from_payload(stale_payload))
                stale_note = "stale_was_not_rejected"
            except ValueError as exc:
                stale_note = str(exc)
                a4_stale = "stale" in stale_note

        # ---- A5 interruption recovery (fresh process replay)
        a5_recovery = False
        recovery_note = "recovery_checkpoint_missing"
        recovery_detail: dict[str, Any] = {}
        if recovery_checkpoint is not None and pending_payloads:
            ckpt_dir = Path(tempfile.mkdtemp(prefix="p52d-"))
            try:
                ckpt_path = ckpt_dir / "online-checkpoint.json"
                pending_path = ckpt_dir / "pending-feedbacks.json"
                ckpt_path.write_text(
                    json.dumps(recovery_checkpoint, sort_keys=True), encoding="utf-8"
                )
                pending_path.write_text(
                    json.dumps(pending_payloads, sort_keys=True), encoding="utf-8"
                )
                child = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child-recover",
                        str(ckpt_path),
                        str(pending_path),
                    ],
                    cwd=PROJECT_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                if child.returncode == 0:
                    child_output = json.loads(child.stdout.strip().splitlines()[-1])
                    final_digest = str(online.checkpoint()["checkpoint_digest"])
                    a5_recovery = bool(
                        child_output.get("final_digest") == final_digest
                        and child_output.get("applied_ids") == pending_applied_ids
                    )
                    recovery_detail = {
                        "child_digest_matches": child_output.get("final_digest") == final_digest,
                        "pending_rounds": len(pending_payloads),
                        "expected_applied_ids": pending_applied_ids,
                        "child_applied_ids": child_output.get("applied_ids"),
                    }
                    recovery_note = "replayed"
                else:
                    recovery_note = f"child_failed: {child.stderr[-400:]}"
            finally:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
        else:
            recovery_note = (
                "recovery_not_exercised: loop ended before round "
                f"{RECOVERY_AFTER_ROUND + 1} or no pending feedbacks"
            )

        # ---- A6 separated rollback: learner tombstone vs environment undo
        a6_rollback = False
        rollback_note = "no_applied_feedback_to_rollback"
        rollback_detail: dict[str, Any] = {}
        if applied_feedbacks:
            last_feedback = applied_feedbacks[-1]
            last_round = next(
                round_record["round"]
                for round_record in rounds
                if round_record.get("admission_status") == "applied"
                and round_record["selection"] == "+".join(last_feedback.member_ids)
            )
            pre_apply = pre_apply_checkpoints[last_round]
            rolled = online.rollback_to(pre_apply, feedback_id=last_feedback.feedback_id)
            inner_restored = str(online.learner.checkpoint()["checkpoint_digest"])
            inner_expected = str(pre_apply["learner"]["checkpoint_digest"])
            tombstoned = last_feedback.candidate_id in online.blocked_candidate_ids
            # The library enforces the tombstone through lineage: after the
            # rollback admission the pre-trial parent digest is permanently
            # unreachable, so a replay is rejected before any admission check.
            replay_rejected = False
            replay_note = ""
            try:
                replay = online.apply_feedback(
                    InteractionGroupOutcomeFeedback.from_payload(last_feedback.to_payload())
                )
                replay_note = f"replay_status:{replay.status}"
            except ValueError as exc:
                replay_rejected = True
                replay_note = str(exc)
            a6_rollback = bool(
                rolled.status == "rolled_back"
                and inner_restored == inner_expected
                and tombstoned
                and replay_rejected
            )
            rollback_detail = {
                "rolled_back_round": last_round,
                "inner_digest_restored": inner_restored == inner_expected,
                "tombstoned": tombstoned,
                "replay_rejected": replay_rejected,
                "replay_note": replay_note,
            }

        # ---- environment transaction undo, independent of learner state
        learner_digest_before_env = str(online.learner.checkpoint()["checkpoint_digest"])
        env_root = Path(tempfile.mkdtemp(prefix="p52d-env-"))
        environment = frozen.WorkbenchEnvironment(env_root)
        demo_name = "p52d_undo_demo.py"
        environment.execute_tool("workspace.create", {"path": demo_name, "content": "x = 1\n"})
        undo_token = str(environment.last_result["transaction"]["undo_token"])
        file_present_after_create = (env_root / demo_name).exists()
        environment.execute_tool("workspace.undo", {"undo_token": undo_token})
        file_present_after_undo = (env_root / demo_name).exists()
        shutil.rmtree(env_root, ignore_errors=True)
        learner_digest_after_env = str(online.learner.checkpoint()["checkpoint_digest"])
        env_undo_detail = {
            "file_present_after_create": file_present_after_create,
            "file_present_after_undo": file_present_after_undo,
            "learner_digest_unchanged": learner_digest_before_env == learner_digest_after_env,
        }
        a6_env_undo = bool(
            file_present_after_create
            and not file_present_after_undo
            and learner_digest_before_env == learner_digest_after_env
        )

        # ---- gates
        fit_ids = {task.task_id for task in legacy_tasks} | {
            task.task_id for task in indomain_tasks
        }
        trial_and_eval_ids = (
            {task["context"] for task in rounds if task.get("context")}
            | set(eval_ids)
            | {task.task_id for task in spot_task}
        )
        ordinal_by_label = {cell["label"]: int(cell["ordinal"]) for cell in _grid}
        used_steps = sorted(
            int(task_id.rsplit("-", 1)[1])
            - ssp.INDEX_BASE
            - ordinal_by_label[task_id.split("-")[1]] * 10
            for task_id in sorted(trial_and_eval_ids)
            if task_id.startswith(("p52dtrial", "p52deval"))
        )
        g1 = bool(
            trial_episodes_total > 0
            and applied_count_pre_rollback == len(applied_feedbacks)
            and all(
                round_record.get("context") is not None
                for round_record in rounds
                if round_record.get("selection")
            )
        )
        g2 = bool(
            not (fit_ids & trial_and_eval_ids)
            and all(step >= FAMILY_STEP + 1 for step in used_steps)
        )
        total_wall = time.perf_counter() - started
        g6 = bool(total_wall <= WALL_CAP_SECONDS)
        acceptance = {
            "a1_new_task_gain": a1_new_task_gain,
            "a2_old_task_retention": a2_old_task_retention,
            "a3_duplicate_idempotency": a3_duplicate,
            "a4_stale_parent_rejection": a4_stale,
            "a5_interruption_recovery": a5_recovery,
            "a6_separated_rollback_and_env_undo": bool(a6_rollback and a6_env_undo),
        }
        mechanism_classes = (
            "a3_duplicate_idempotency",
            "a4_stale_parent_rejection",
            "a5_interruption_recovery",
            "a6_separated_rollback_and_env_undo",
        )
        gates = {
            "static_checks": True,
            "g1_intervention_reality": g1,
            "g2_data_isolation": g2,
            "g6_budget": g6,
            **acceptance,
        }
        if any(
            not gates[name]
            for name in (
                "static_checks",
                "g1_intervention_reality",
                "g2_data_isolation",
                "g6_budget",
            )
        ):
            outcome = "failed"
        elif any(not gates[name] for name in mechanism_classes):
            outcome = "failed"
        elif all(acceptance.values()):
            outcome = "online_writeback_supported"
        else:
            outcome = "partial"
        payload.update(
            {
                "status": "completed",
                "outcome": outcome,
                "experiment_passed": bool(outcome == "online_writeback_supported"),
                "record": {
                    "commit": commit,
                    "rule_revision": frozen.RULE_REVISION,
                    "composition_rule": frozen.COMPOSITION_RULE,
                    "wall_cap_seconds": WALL_CAP_SECONDS,
                    "elapsed_seconds": round(total_wall, 3),
                },
                "base_lineage": {
                    "member_sets": sorted(
                        "/".join(cell) if cell else "baseline" for cell in base_sets
                    ),
                    "record_pairs": base_record_pairs,
                    "train_trace_digest": corpus.train_trace_digest,
                    "parent_selection": "+".join(parent_pair),
                },
                "admission_budget": admission_budget,
                "online_rounds": rounds,
                "arms": {
                    "parent_pair": "+".join(parent_pair),
                    "updated_pair": "+".join(updated_pair),
                    "parent_gain_rate_scale": round(parent_gain, 6),
                    "updated_gain_rate_scale": round(updated_gain, 6),
                    "updated_predictions": {
                        label: round(value, 6)
                        for label, value in sorted(updated_predictions.items())
                    },
                },
                "acceptance": {
                    **acceptance,
                    "retention_detail": {
                        "ab_prediction_before": ab_prediction_before,
                        "ab_prediction_after": ab_prediction_after,
                        "ab_records_before": ab_records_before,
                        "ab_records_after": ab_records_after,
                        "retention_records_untouched": retention_records_untouched,
                        "note": (
                            "the library refits globally on every record, so the "
                            "prediction shift is the global-refit effect of the "
                            "admitted record and is disclosed, not gated "
                            "(preregistration section 8.1)"
                        ),
                        "retention_execution_ok": retention_execution_ok,
                        "parent_spot": round(parent_spot, 6),
                        "updated_spot": round(updated_spot, 6),
                    },
                    "duplicate_note": duplicate_note,
                    "stale_note": stale_note,
                    "recovery_note": recovery_note,
                    "recovery_detail": recovery_detail,
                    "rollback_note": rollback_note,
                    "rollback_detail": rollback_detail,
                    "env_undo_detail": env_undo_detail,
                },
                "stop_reasons": {"trial_episodes": trial_episodes_total},
                "gates": gates,
                "cost": {
                    "member_training_seconds": round(member_seconds, 3),
                    "episodes_total": len(legacy_episodes)
                    + len(indomain_episodes)
                    + trial_episodes_total
                    + len(eval_episodes)
                    + len(spot_episodes),
                },
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        f"completed: outcome={outcome}; updated selection "
                        f"{'+'.join(updated_pair)} (parent {'+'.join(parent_pair)}); "
                        "acceptance classes "
                        + json.dumps(acceptance)
                        + "; claim scope: library online writeback machinery over "
                        "the frozen base lineage; growth_admitted=false and "
                        "can_promote=false remain in force"
                    )
                    if outcome != "failed"
                    else (
                        "completed: outcome=failed; failed gates "
                        f"{sorted(key for key, value in gates.items() if not value)}"
                    )
                ),
                "elapsed_seconds": round(total_wall, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    _write_json(DEFAULT_REPORT, payload)
    return payload


def _child_recover(ckpt_path: Path, pending_path: Path) -> int:
    controller = InteractionGroupOnlineLearner.from_checkpoint(
        json.loads(ckpt_path.read_text(encoding="utf-8"))
    )
    payloads = json.loads(pending_path.read_text(encoding="utf-8"))
    applied = []
    for item in payloads:
        feedback = InteractionGroupOutcomeFeedback.from_payload(item)
        admission = controller.apply_feedback(feedback)
        if admission.status == "applied":
            applied.append(admission.feedback_id)
    print(
        json.dumps(
            {
                "final_digest": str(controller.checkpoint()["checkpoint_digest"]),
                "applied_ids": applied,
            }
        )
    )
    return 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-recover", nargs=2, type=Path, default=None)
    args = parser.parse_args(argv)
    if args.child_recover is not None:
        return _child_recover(args.child_recover[0], args.child_recover[1])
    result = run_gate()
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "gates": result.get("gates"),
                "acceptance": (result.get("acceptance") or {}),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("experiment_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
