"""R029 engineering-gap verification on the R021/R027 subject path.

Closes the five gaps declared in SPEC-TAIJI-CORE_ENGINEERING_R029: the §10.2
constrained feedback update, activity-policy learning, post-training lesion,
natural-Chinese task fixtures and subprocess CPU accounting. Fixtures, configs
and the timing helper are imported from the frozen R028 pricing script so both
reports price the same face. Synthetic/public Chinese fixtures only: no sealed
data, no product default change, no training admission.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_living_fragment_pricing_r028 as pricing  # noqa: E402
from taiji import constrained_learning as cl  # noqa: E402
from taiji import growth_credit as gc  # noqa: E402
from taiji import native_development as development  # noqa: E402
from taiji.activity_learning import (  # noqa: E402
    ActivityLearner,
    evaluate_activity_policy,
    masked_log_probabilities,
    train_activity_episode,
)
from taiji.native_activity import (  # noqa: E402
    ActivityEnvironment,
    ActivityPolicy,
    build_features,
)
from taiji.native_task_fixtures import answer_rounds, build_fixtures, shortcut_probe  # noqa: E402
from taiji.state_migration import (  # noqa: E402
    load_subject,
    resume_career,
    resume_subject,
    run_career,
    run_worker_job,
    save_subject,
)

SEED = 20261101
CONSTRAINT_STEPS = 8
CONSTRAINT_ETA = 1.0
CONSTRAINT_EPSILON = 2e-3
CONSTRAINT_RHO = 2e-2
ACTUAL_EPSILON = 3.75e-3
MAX_RETRIES = 4
UNCONSTRAINED_LR = 1e-3
FIT_DROP_GATE = 0.010
RETENTION_GATE = 0.030
CONTROL_DRIFT_FLOOR = 0.05
KKT_GATE = 1e-8
VIOLATION_GATE = 1e-12
ACTIVITY_TRAIN_ROUNDS = 256
ACTIVITY_EVAL_ROUNDS = 32
ACTIVITY_MAX_EXPOSURES = 256
LESION_LR = 1e-5
LESION_UPDATES = 8
BENEFIT_GATE = 1e-4
BUDGET_CPU_SECONDS = 1800
BUDGET_WALL_SECONDS = 3600
# R028 reported the fresh-spawn restore at 0.0 CPU seconds because it could only
# see the parent side; this is its bundle size on the same declared save face.
R028_BUNDLE_BYTES = 77304316
R028_SPAWN_PARENT_CPU_SECONDS = 0.0
RUNTIME_MODULES = ("taiji/activity_learning.py", "taiji/constrained_learning.py")
FORBIDDEN_RUNTIME_TOKENS = ("expected_target", "shortcut_label", "category", "gold")
REQUIRED_SECTIONS = (
    "constraint",
    "control",
    "activity",
    "lesion",
    "fixture",
    "worker",
    "boundary",
)


class Budget(Exception):
    pass


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def module_digest(module) -> str:
    payload = b""
    for name, parameter in module.named_parameters():
        payload += name.encode("utf-8")
        payload += parameter.detach().cpu().contiguous().numpy().tobytes()
    return digest_bytes(payload)


def parameter_norm(model) -> float:
    return float(
        torch.stack([parameter.detach().norm() for parameter in model.parameters()]).norm()
    )


def constraint_section(guard) -> dict:
    torch.manual_seed(SEED)
    model, _ = pricing.fresh(pricing.PRINCIPLE, "r029-constrained")
    norm_before = parameter_norm(model)
    fit_pre = development.evaluate_career_nll(model, pricing.LONG)
    holdout_pre = development.evaluate_career_nll(model, pricing.HOLDOUT)
    retention_pre = development.evaluate_career_nll(model, pricing.RETENTION)
    guard()
    rows, cpu, wall = pricing.timed(
        lambda: cl.constrained_feedback_updates(
            model,
            pricing.LONG,
            (pricing.RETENTION,),
            steps=CONSTRAINT_STEPS,
            eta=CONSTRAINT_ETA,
            epsilon=CONSTRAINT_EPSILON,
            rho=CONSTRAINT_RHO,
            actual_epsilon=ACTUAL_EPSILON,
            max_retries=MAX_RETRIES,
        )
    )
    fit_post = development.evaluate_career_nll(model, pricing.LONG)
    holdout_post = development.evaluate_career_nll(model, pricing.HOLDOUT)
    retention_post = development.evaluate_career_nll(model, pricing.RETENTION)
    guard()
    deltas = []
    for row in rows:
        diagnostics = row["diagnostics"]
        deltas.append(
            {
                "fit_loss_before": row["fit_loss_before"],
                "gradient_norm": row["gradient_norm"],
                "pre_projection_norm": CONSTRAINT_ETA * row["gradient_norm"],
                "delta_norm": diagnostics.delta_norm,
                "ball_active": diagnostics.ball_active,
                "active_constraints": list(diagnostics.active_constraints),
                "multipliers": diagnostics.multipliers,
                "kkt_residual": diagnostics.kkt_residual,
                "complementarity_residual": diagnostics.complementarity_residual,
                "constraint_values": diagnostics.constraint_values,
                "constraint_slacks": diagnostics.constraint_slacks,
                "max_violation": max(
                    [0.0] + [value - CONSTRAINT_EPSILON for value in diagnostics.constraint_values]
                ),
                "attempts": row["attempts"],
                "accepted": row["accepted"],
                "rho_used": row["rho_used"],
                "actual_old_max_abs_change": row["actual_old_max_abs_change"],
                "cpu_seconds": sum(row["timings"].values()),
                "timings": dict(row["timings"]),
            }
        )
    ulp = torch.finfo(torch.float64).eps * norm_before
    return {
        "shape": "principle",
        "fit_journal_events": pricing.LONG.event_count,
        "old_journals": ["r028-retention"],
        "holdout_journal": "r028-holdout (evaluation only, never constrained)",
        "declared": {
            "steps": CONSTRAINT_STEPS,
            "eta": CONSTRAINT_ETA,
            "epsilon": CONSTRAINT_EPSILON,
            "rho": CONSTRAINT_RHO,
            "actual_epsilon": ACTUAL_EPSILON,
            "max_retries": MAX_RETRIES,
        },
        "nll": {
            "fit_pre": fit_pre,
            "fit_post": fit_post,
            "fit_drop": fit_pre - fit_post,
            "holdout_pre": holdout_pre,
            "holdout_post": holdout_post,
            "delta_nll_holdout": holdout_post - holdout_pre,
            "retention_pre": retention_pre,
            "retention_post": retention_post,
            "drift_retention": abs(retention_post - retention_pre),
        },
        "parameter_norm_before": norm_before,
        "parameter_norm_after": parameter_norm(model),
        "ulp": ulp,
        "delta_norm_over_ulp": sum(row["delta_norm"] for row in deltas) / ulp,
        "max_kkt_residual": max(row["kkt_residual"] for row in deltas),
        "max_violation": max(row["max_violation"] for row in deltas),
        "max_complementarity": max(row["complementarity_residual"] for row in deltas),
        "accepted_steps": sum(row["accepted"] for row in deltas),
        "total_attempts": sum(row["attempts"] for row in deltas),
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "steps": deltas,
    }


def control_section(guard) -> dict:
    torch.manual_seed(SEED)
    model, _ = pricing.fresh(pricing.PRINCIPLE, "r029-unconstrained")
    retention_pre = development.evaluate_career_nll(model, pricing.RETENTION)
    holdout_pre = development.evaluate_career_nll(model, pricing.HOLDOUT)
    fit_pre = development.evaluate_career_nll(model, pricing.LONG)
    optimizer = torch.optim.Adam(model.parameters(), lr=UNCONSTRAINED_LR)
    guard()
    losses, cpu, wall = pricing.timed(
        lambda: development.declared_feedback_updates(
            model, pricing.LONG, optimizer, steps=CONSTRAINT_STEPS
        )
    )
    retention_post = development.evaluate_career_nll(model, pricing.RETENTION)
    holdout_post = development.evaluate_career_nll(model, pricing.HOLDOUT)
    fit_post = development.evaluate_career_nll(model, pricing.LONG)
    guard()
    return {
        "shape": "principle",
        "declared": {"lr": UNCONSTRAINED_LR, "steps": CONSTRAINT_STEPS, "optimizer": "Adam"},
        "update_losses": losses,
        "fit_drop": fit_pre - fit_post,
        "delta_nll_holdout": holdout_post - holdout_pre,
        "drift_retention": abs(retention_post - retention_pre),
        "cpu_seconds": cpu,
        "wall_seconds": wall,
    }


def activity_env(config, subject: str, seed: int):
    torch.manual_seed(seed)
    model, state = pricing.fresh(config, subject)
    env = ActivityEnvironment(store=pricing.RECORDS, goal=pricing.CHINESE_GOAL, now=10)
    env.message_queue.append(pricing.CHINESE_LONG_TARGET)
    env.pending_outcomes.append("r029-outcome")
    return model, state, env


def ledger_invariants(rows) -> dict:
    idle = [row for row in rows if row.get("activity") in ("wait", "maintain")]
    imagined = [row for row in rows if row.get("imagined")]
    return {
        "rows": len(rows),
        "idle_rows": len(idle),
        "idle_rows_with_microsteps": sum(1 for row in idle if row.get("step_delta", 0) != 0),
        "imagined_rows": len(imagined),
        "imagined_store_sizes": sorted({row["store_size_after"] for row in imagined}),
        "chosen_by": sorted({row["chosen_by"] for row in rows}),
        "exposures": [row.get("exposures") for row in rows if "exposures" in row],
        "mean_round_cpu_seconds": (
            sum(row["round_cpu_seconds"] for row in rows if "round_cpu_seconds" in row)
            / max(1, sum(1 for row in rows if "round_cpu_seconds" in row))
        ),
    }


def activity_section(guard, train_rounds: int, eval_rounds: int) -> dict:
    config = pricing.SMALL
    model, state, env = activity_env(config, "r029-activity-train", SEED)
    learner = ActivityLearner.build(config.width)
    policy_before, critic_before = module_digest(learner.policy), module_digest(learner.critic)
    last_event = torch.zeros(config.width, dtype=torch.float64)
    journal: list = []
    guard()
    train_rows, train_cpu, train_wall = pricing.timed(
        lambda: train_activity_episode(
            model,
            state,
            env,
            learner,
            pricing.SHORT,
            pricing.RETENTION,
            rounds=train_rounds,
            journal=journal,
            last_event=last_event,
            max_exposures=ACTIVITY_MAX_EXPOSURES,
        )[4]
    )
    policy_after, critic_after = module_digest(learner.policy), module_digest(learner.critic)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "learner.pt"
        torch.save(learner.state_dict(), path)
        roundtrip = ActivityLearner.build(config.width)
        roundtrip.load_state_dict(torch.load(path, weights_only=True))
        roundtrip_equal = _roundtrip_equal(learner, roundtrip)
    guard()

    # All three arms rebuild the identical subject from the same seed, so the only
    # variable is the selection head: trained policy, seeded policy, fixed ladder.
    arms, core_digests = {}, {}
    for name, control in (
        ("trained_model", False),
        ("untrained_model", False),
        ("fixed_threshold", True),
    ):
        arm_model, arm_state, arm_env = activity_env(config, "r029-activity-eval", SEED + 1)
        core_digests[name] = module_digest(arm_model)
        policy = (
            learner.policy if name == "trained_model" else ActivityPolicy(config.width).double()
        )
        _, _, _, _, rows, cumulative = evaluate_activity_policy(
            arm_model,
            arm_state,
            arm_env,
            policy,
            pricing.SHORT,
            pricing.RETENTION,
            rounds=eval_rounds,
            journal=[],
            last_event=torch.zeros(config.width, dtype=torch.float64),
            control=control,
        )
        arms[name] = {
            "cumulative_reward": cumulative,
            "activities": [row["activity"] for row in rows],
            "distinct_activities": sorted({row["activity"] for row in rows}),
            "external_triggers": sum(1 for row in rows if row.get("actual_reward", 0.0) > 0.0),
            "compute_cost_total": sum(row["compute_cost"] for row in rows),
            "ledger": ledger_invariants(rows),
        }
        guard()
    advantage = arms["trained_model"]["cumulative_reward"] - max(
        arms["untrained_model"]["cumulative_reward"], arms["fixed_threshold"]["cumulative_reward"]
    )
    verdict = "activity_learning_advantage" if advantage > 0 else "negative_activity_learning"
    trained_idle_only = set(arms["trained_model"]["distinct_activities"]) <= {"wait", "maintain"}
    advantage_is_cost_avoidance = bool(
        trained_idle_only and arms["trained_model"]["cumulative_reward"] == 0.0
    )

    greedy_model, greedy_state, greedy_env = activity_env(config, "r029-activity-eval", SEED + 1)
    goal_vector = greedy_model.encoder.goal_vector(greedy_env.goal)
    with torch.no_grad():
        features = build_features(greedy_state, goal_vector, 0.0, greedy_env).detach()
    trained_greedy = masked_log_probabilities(learner.policy, features, greedy_env).exp()
    seeded_greedy = masked_log_probabilities(
        ActivityPolicy(config.width).double(), features, greedy_env
    ).exp()
    return {
        "shape": "small",
        "declared": {
            "train_rounds": train_rounds,
            "eval_rounds": eval_rounds,
            "max_exposures": ACTIVITY_MAX_EXPOSURES,
            "trace_lambda": 0.0,
            "gamma": 0.99,
            "policy_lr": 1.0,
            "critic_lr": 0.1,
            "epsilon": 0.5,
            "interact_reward": 5e-2,
            "compute_cost": 1e-3,
            "retention_penalty": 0.1,
            "arm_seed": SEED + 1,
            "note": "TRACE_LAMBDA=0.0 is the declared value for this immediate-reward fixture.",
        },
        "policy_sha256_before": policy_before,
        "policy_sha256_after": policy_after,
        "critic_sha256_before": critic_before,
        "critic_sha256_after": critic_after,
        "policy_changed": policy_before != policy_after,
        "critic_changed": critic_before != critic_after,
        "state_roundtrip_bitwise": roundtrip_equal,
        "exposures": learner.exposures,
        "core_parameter_digests_identical_across_arms": len(set(core_digests.values())) == 1,
        "core_parameter_digests": core_digests,
        "training": {
            "rows": len(train_rows),
            "updated": sum(1 for row in train_rows if row.get("status") == "updated"),
            "stopped": [row for row in train_rows if row.get("status") != "updated"],
            "ledger": ledger_invariants(train_rows),
            "first_reward": train_rows[0].get("reward"),
            "last_reward": train_rows[-1].get("reward"),
            "cpu_seconds": train_cpu,
            "wall_seconds": train_wall,
        },
        "greedy_probabilities_trained": [float(value) for value in trained_greedy.detach()],
        "greedy_probabilities_seeded": [float(value) for value in seeded_greedy.detach()],
        "arms": arms,
        "advantage_over_best_control": advantage,
        "verdict": verdict,
        "trained_arm_chose_only_idle_activities": trained_idle_only,
        "advantage_is_idle_cost_avoidance": advantage_is_cost_avoidance,
        "advantage_disclosure": (
            "The trained arm's cumulative P_R is exactly zero when it only selects idle "
            "activities, so a positive advantage over the controls measures avoided compute "
            "cost, not task gain. The declared fixture places one external message in the "
            "queue, so the reward-bearing interact activity is legal in at most one of the "
            "evaluation rounds; 09.7 must design a recurring-trigger activity fixture before "
            "any autonomy or activity-learning claim is made."
            if advantage_is_cost_avoidance
            else "The trained arm selected at least one non-idle activity."
        ),
        "journal_entries": len(journal),
    }


def _roundtrip_equal(left: ActivityLearner, right: ActivityLearner) -> bool:
    if left.exposures != right.exposures:
        return False
    for source, target in ((left.policy, right.policy), (left.critic, right.critic)):
        source_state, target_state = source.state_dict(), target.state_dict()
        if set(source_state) != set(target_state):
            return False
        for name, value in source_state.items():
            if not torch.equal(value, target_state[name]):
                return False
    for name, trace in left.eligibility.items():
        if not torch.equal(trace, right.eligibility[name]):
            return False
    return True


def birth_inputs(parent):
    book, _ = gc.collect_credit(parent, pricing.LONG)
    steps = len(book.deltas)
    fit_window, check_window = range(steps // 2), range(steps // 2, steps)
    tau_map = {
        name: gc.calibrate_tau_noise(book, branch, fit_window)
        for branch, name in enumerate(book.branch_ids)
    }
    full_credits, fit_credits, verdicts = {}, {}, {}
    for branch, name in enumerate(book.branch_ids):
        full_credits[name] = gc.branch_credit(book, branch)
        fit_credits[name] = gc.branch_credit(book, branch, fit_window)
        verdicts[name] = gc.branch_verdict(book, branch, tau_map[name], fit_window, check_window)
    return book, full_credits, fit_credits, verdicts


def lesion_section(guard) -> dict:
    torch.manual_seed(SEED)
    parent, _ = pricing.fresh(pricing.PRINCIPLE, "r029-lesion")
    guard()
    (book, full_credits, fit_credits, verdicts), build_cpu, build_wall = pricing.timed(
        lambda: birth_inputs(parent)
    )
    guard()
    (child, provider, plan), birth_cpu, birth_wall = pricing.timed(
        lambda: development.coverage_birth_arm(parent, book, fit_credits, verdicts)
    )
    with torch.no_grad():
        parent_state, parent_steps = run_career(parent, pricing.LONG)
        child_state, child_steps = run_career(child, pricing.LONG, growth_gate_provider=provider)
    neutral_equal = _function_equal(parent_state, parent_steps, child_state, child_steps)

    # Restoration face: an untrained copy. Birth is function-neutral, so removing
    # the newborn units must bring the function back to the parent bitwise.
    untrained = copy.deepcopy(child)
    guard()
    untrained_report, lesion_cpu, lesion_wall = pricing.timed(lambda: gc.lesion_trained(untrained))
    with torch.no_grad():
        untrained_state, untrained_steps = run_career(untrained, pricing.LONG)
    restored_equal = _function_equal(parent_state, parent_steps, untrained_state, untrained_steps)

    # Contribution face: train the child, then remove only the newborn units. The
    # surviving parent rows are trained, so bitwise equality with the pre-training
    # parent is not the claim; preservation of those learned rows is.
    holdout_pre = development.evaluate_career_nll(child, pricing.HOLDOUT, provider)
    optimizer = torch.optim.Adam(child.parameters(), lr=LESION_LR)
    guard()
    losses, train_cpu, train_wall = pricing.timed(
        lambda: development.declared_feedback_updates(
            child, pricing.LONG, optimizer, provider, steps=LESION_UPDATES
        )
    )
    holdout_post = development.evaluate_career_nll(child, pricing.HOLDOUT, provider)
    units_before = int(child.R.shape[1])
    parent_units = child.growth_parent_units
    trained_r = child.R[:, :parent_units, :].detach().clone()
    trained_p = child.P[:, :, :parent_units].detach().clone()
    guard()
    trained_report = gc.lesion_trained(child)
    rows_preserved = bool(torch.equal(child.R, trained_r) and torch.equal(child.P, trained_p))
    holdout_lesion = development.evaluate_career_nll(child, pricing.HOLDOUT)
    benefit = holdout_pre - holdout_post
    benefit_after = holdout_pre - holdout_lesion
    benefit_removed = benefit - benefit_after
    if benefit < BENEFIT_GATE:
        verdict = "no_benefit_to_lesion"
    elif benefit_removed >= BENEFIT_GATE:
        verdict = "lesion_removes_benefit"
    else:
        verdict = "lesion_does_not_remove_benefit"
    guard()
    return {
        "shape": "principle",
        "declared": {
            "lr": LESION_LR,
            "updates": LESION_UPDATES,
            "benefit_gate": BENEFIT_GATE,
            "lr_source": "R028 frozen learning rate",
            "restoration_face": "untrained deepcopy of the same coverage_birth child",
            "contribution_face": "the trained coverage_birth child",
        },
        "units_total": plan["units_total"],
        "allocation": plan.get("allocation"),
        "birth_units_before_lesion": units_before,
        "birth_units_after_lesion": int(child.R.shape[1]),
        "growth_parent_units": parent_units,
        "neutral_birth_bitwise": neutral_equal,
        "lesion_restores_parent_bitwise": restored_equal,
        "lesion_preserves_trained_parent_rows": rows_preserved,
        "untrained_lesion_report": untrained_report,
        "update_losses": losses,
        "holdout_pre": holdout_pre,
        "holdout_post": holdout_post,
        "holdout_after_lesion": holdout_lesion,
        "benefit": benefit,
        "benefit_after_lesion": benefit_after,
        "benefit_removed": benefit - benefit_after,
        "trained_lesion_report": trained_report,
        "verdict": verdict,
        "contribution_disclosure": {
            "benefit_from_training": benefit,
            "benefit_attributable_to_newborn_units": benefit_removed,
            "newborn_share_of_benefit": (benefit_removed / benefit if benefit else None),
            "reading": (
                "Newborn units account for the share above of the measured holdout benefit; "
                "the remainder is carried by the inherited parent rows that training changed. "
                "A share below the declared gate means the growth mechanism has no measurable "
                "contribution on this fixture and must not be signed off as effective."
            ),
        },
        "credit_build": {"cpu_seconds": build_cpu, "wall_seconds": build_wall},
        "birth": {"cpu_seconds": birth_cpu, "wall_seconds": birth_wall},
        "untrained_lesion": {"cpu_seconds": lesion_cpu, "wall_seconds": lesion_wall},
        "training": {"cpu_seconds": train_cpu, "wall_seconds": train_wall},
    }


def _function_equal(state_a, steps_a, state_b, steps_b) -> bool:
    return bool(
        torch.equal(state_a.h, state_b.h)
        and len(steps_a) == len(steps_b)
        and all(
            torch.equal(left.probabilities, right.probabilities)
            for (left, _), (right, _) in zip(steps_a, steps_b, strict=True)
        )
    )


def fixture_section(guard) -> dict:
    fixtures = build_fixtures()
    probe = shortcut_probe(fixtures)
    source_guards = {}
    for relative in RUNTIME_MODULES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        source_guards[relative] = {token: text.count(token) for token in FORBIDDEN_RUNTIME_TOKENS}
    guard()
    return {
        "fixtures": [
            {
                "fixture_id": item.fixture_id,
                "category": item.category,
                "prompt_bytes": len(item.journal.prompt),
                "target_bytes": len(item.journal.target),
                "answer_denominator": item.answer_denominator,
                "answer_rounds": answer_rounds(item),
                "events": item.journal.event_count,
                "records": len(item.journal.records),
                "shortcut_label": item.shortcut_label,
            }
            for item in fixtures
        ],
        "shortcut_probe": probe,
        "runtime_source_token_counts": source_guards,
        "sealed_read": False,
        "corpus_split_read": [],
    }


def spawn_worker_timed(spec: dict) -> tuple[dict, float, float]:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=run_worker_job, args=(spec,))
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    process.start()
    process.join(timeout=600)
    cpu, wall = time.process_time() - cpu_start, time.perf_counter() - wall_start
    if process.exitcode != 0:
        raise RuntimeError(f"restore worker exitcode={process.exitcode}")
    return torch.load(spec["out"], weights_only=True), cpu, wall


def worker_section(guard) -> dict:
    torch.manual_seed(SEED)
    model, state = pricing.fresh(pricing.PRINCIPLE, "r029-worker")
    # The R028 restore face is mirrored exactly: one declared update so the Adam
    # state is populated, then the same journal-entry and pending payload. Without
    # this the bundle carries parameters only and the price is not comparable.
    optimizer = torch.optim.Adam(model.parameters(), lr=development.UPDATE_LR)

    def perform_update():
        _, steps = resume_career(model, state, pricing.SHORT, 0)
        terms = [output.probabilities[byte] for output, byte in steps if byte is not None]
        loss = -torch.stack(terms).log().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return steps[-1][0].state

    guard()
    state, update_cpu, update_wall = pricing.timed(perform_update)
    with tempfile.TemporaryDirectory() as directory:
        bundle_path = Path(directory) / "bundle.pt"
        out_path = Path(directory) / "worker.pt"
        guard()
        bundle_sha, save_cpu, save_wall = pricing.timed(
            lambda: save_subject(
                bundle_path,
                model,
                state,
                career=pricing.SHORT,
                journal_cursor=1,
                optimizer=optimizer,
                eligibility=torch.zeros(pricing.PRINCIPLE.groups, dtype=torch.float64),
                journal_entries=[{"seq": 0, "kind": "pricing_seed"}],
                pending={"outcome_id": "pricing", "reward": 0.1, "resolved": False},
            )
        )
        bundle_bytes = bundle_path.stat().st_size
        restored_model, restored_state, _ = resume_subject(load_subject(bundle_path))
        with torch.no_grad():
            local_state, local_steps = resume_career(
                restored_model, restored_state, pricing.SHORT, 1, limit=1
            )
        guard()
        worker, parent_cpu, parent_wall = spawn_worker_timed(
            {
                "bundle": str(bundle_path),
                "mode": "microstep",
                "out": str(out_path),
                "out_bundle": None,
            }
        )
        guard()
    return {
        "shape": "principle",
        "bundle_bytes": bundle_bytes,
        "r028_bundle_bytes": R028_BUNDLE_BYTES,
        "r028_spawn_parent_cpu_seconds": R028_SPAWN_PARENT_CPU_SECONDS,
        "bundle_face_matches_r028": bundle_bytes == R028_BUNDLE_BYTES,
        "optimizer_state_entries": len(optimizer.state),
        "bundle_sha256": bundle_sha,
        "declared_update_for_bundle": {"cpu_seconds": update_cpu, "wall_seconds": update_wall},
        "save_subject": {"cpu_seconds": save_cpu, "wall_seconds": save_wall},
        "worker_cpu_seconds": worker["worker_cpu_seconds"],
        "worker_wall_seconds": worker["worker_wall_seconds"],
        "parent_wait_cpu_seconds": parent_cpu,
        "parent_wait_wall_seconds": parent_wall,
        "child_cpu_exceeds_parent_wait_cpu": worker["worker_cpu_seconds"] > parent_cpu,
        "h_bitwise_equal": bool(torch.equal(worker["h"], local_state.h)),
        "probabilities_bitwise_equal": bool(
            torch.equal(worker["probabilities"], local_steps[0][0].probabilities)
        ),
    }


def boundary_section(guard) -> dict:
    import tests.taiji_native.test_subject_gates_r027 as r027

    original = r027.spawn_worker
    calls: list[dict] = []

    def wrapped(spec: dict) -> dict:
        result = original(spec)
        calls.append(
            {
                "mode": result.get("mode"),
                "worker_cpu_seconds": result.get("worker_cpu_seconds"),
                "worker_wall_seconds": result.get("worker_wall_seconds"),
            }
        )
        return result

    r027.spawn_worker = wrapped
    try:
        guard()
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        r027.test_r027_unified_life_sequence_gates()
        cpu, wall = time.process_time() - cpu_start, time.perf_counter() - wall_start
    finally:
        r027.spawn_worker = original
    missing = [row for row in calls if row["worker_cpu_seconds"] is None]
    return {
        "shape": "r027 declared small shape",
        "worker_calls": len(calls),
        "calls": calls,
        "calls_missing_child_cpu": len(missing),
        "child_cpu_total": sum(row["worker_cpu_seconds"] or 0.0 for row in calls),
        "child_wall_total": sum(row["worker_wall_seconds"] or 0.0 for row in calls),
        "sequence_parent_cpu_seconds": cpu,
        "sequence_wall_seconds": wall,
        "note": "sequence_parent_cpu_seconds excludes child CPU; child_cpu_total is self-reported.",
    }


def build_checks(sections: dict) -> list[dict]:
    constraint, control, activity = (
        sections["constraint"],
        sections["control"],
        sections["activity"],
    )
    lesion, fixture, worker, boundary = (
        sections["lesion"],
        sections["fixture"],
        sections["worker"],
        sections["boundary"],
    )
    ledger = activity["training"]["ledger"]
    return [
        {
            "id": "constraint_kkt_within_tolerance",
            "passed": constraint["max_kkt_residual"] <= KKT_GATE,
            "observed": constraint["max_kkt_residual"],
        },
        {
            "id": "constraint_violation_within_tolerance",
            "passed": constraint["max_violation"] <= VIOLATION_GATE,
            "observed": constraint["max_violation"],
        },
        {
            "id": "constraint_complementarity_within_tolerance",
            "passed": constraint["max_complementarity"] <= KKT_GATE,
            "observed": constraint["max_complementarity"],
        },
        {
            "id": "constraint_all_steps_accepted",
            "passed": constraint["accepted_steps"] == CONSTRAINT_STEPS,
            "observed": constraint["accepted_steps"],
        },
        {
            "id": "constraint_fit_drop_gate",
            "passed": constraint["nll"]["fit_drop"] >= FIT_DROP_GATE,
            "observed": constraint["nll"]["fit_drop"],
        },
        {
            "id": "constraint_retention_gate",
            "passed": constraint["nll"]["drift_retention"] <= RETENTION_GATE,
            "observed": constraint["nll"]["drift_retention"],
        },
        {
            "id": "constraint_negative_control_drift",
            "passed": control["drift_retention"] > CONTROL_DRIFT_FLOOR,
            "observed": control["drift_retention"],
        },
        {
            "id": "activity_parameters_changed",
            "passed": activity["policy_changed"] and activity["critic_changed"],
            "observed": [activity["policy_changed"], activity["critic_changed"]],
        },
        {
            "id": "activity_state_roundtrip_bitwise",
            "passed": activity["state_roundtrip_bitwise"],
            "observed": activity["state_roundtrip_bitwise"],
        },
        {
            "id": "activity_exposure_budget_respected",
            "passed": activity["exposures"] <= ACTIVITY_MAX_EXPOSURES,
            "observed": activity["exposures"],
        },
        {
            "id": "activity_idle_rows_fabricate_no_microsteps",
            "passed": ledger["idle_rows_with_microsteps"] == 0,
            "observed": ledger["idle_rows_with_microsteps"],
        },
        {
            "id": "activity_arms_share_the_same_core",
            "passed": activity["core_parameter_digests_identical_across_arms"],
            "observed": sorted(set(activity["core_parameter_digests"].values())),
        },
        {
            "id": "activity_verdict_has_advantage",
            "passed": activity["verdict"] == "activity_learning_advantage",
            "observed": activity["advantage_over_best_control"],
        },
        {
            "id": "lesion_birth_neutral_bitwise",
            "passed": lesion["neutral_birth_bitwise"],
            "observed": lesion["neutral_birth_bitwise"],
        },
        {
            "id": "lesion_restores_parent_bitwise",
            "passed": lesion["lesion_restores_parent_bitwise"],
            "observed": lesion["lesion_restores_parent_bitwise"],
        },
        {
            "id": "lesion_preserves_trained_parent_rows",
            "passed": lesion["lesion_preserves_trained_parent_rows"],
            "observed": lesion["lesion_preserves_trained_parent_rows"],
        },
        {
            "id": "lesion_verdict_declared",
            "passed": lesion["verdict"] in {"lesion_removes_benefit", "no_benefit_to_lesion"},
            "observed": lesion["verdict"],
        },
        {
            "id": "fixture_shortcut_ruler_bounded",
            "passed": all(
                0 < row["correct"] < row["total"]
                for row in fixture["shortcut_probe"]["as_given"].values()
            ),
            "observed": {
                name: row["correct"] for name, row in fixture["shortcut_probe"]["as_given"].items()
            },
        },
        {
            "id": "fixture_denominator_matches_answer_rounds",
            "passed": all(
                row["answer_denominator"] == row["answer_rounds"] > 1 for row in fixture["fixtures"]
            ),
            "observed": [row["answer_denominator"] for row in fixture["fixtures"]],
        },
        {
            "id": "fixture_runtime_modules_read_no_gold",
            "passed": all(
                count == 0
                for tokens in fixture["runtime_source_token_counts"].values()
                for count in tokens.values()
            ),
            "observed": fixture["runtime_source_token_counts"],
        },
        {
            "id": "worker_bundle_face_matches_r028",
            "passed": worker["bundle_face_matches_r028"] and worker["optimizer_state_entries"] > 0,
            "observed": [worker["bundle_bytes"], worker["optimizer_state_entries"]],
        },
        {
            "id": "worker_self_reports_cpu",
            "passed": worker["worker_cpu_seconds"] > 0.0
            and worker["worker_cpu_seconds"] <= worker["worker_wall_seconds"],
            "observed": worker["worker_cpu_seconds"],
        },
        {
            "id": "worker_cpu_not_a_parent_copy",
            "passed": worker["child_cpu_exceeds_parent_wait_cpu"],
            "observed": [worker["worker_cpu_seconds"], worker["parent_wait_cpu_seconds"]],
        },
        {
            "id": "worker_restore_bitwise",
            "passed": worker["h_bitwise_equal"] and worker["probabilities_bitwise_equal"],
            "observed": [worker["h_bitwise_equal"], worker["probabilities_bitwise_equal"]],
        },
        {
            "id": "boundary_worker_calls_all_reported",
            "passed": boundary["worker_calls"] == 8 and boundary["calls_missing_child_cpu"] == 0,
            "observed": [boundary["worker_calls"], boundary["calls_missing_child_cpu"]],
        },
    ]


def classify(checks: list[dict], budget_ok: bool) -> str:
    failed = {row["id"] for row in checks if not row["passed"]}
    if not budget_ok:
        return "budget_stop"
    if any(name.startswith("constraint_") for name in failed):
        return "failed_constraint"
    if any(name.startswith("activity_") for name in failed):
        return "failed_activity_learning"
    if any(name.startswith("lesion_") for name in failed):
        return "failed_lesion"
    if any(name.startswith("fixture_") for name in failed):
        return "failed_fixture"
    if any(name.startswith(("worker_", "boundary_")) for name in failed):
        return "failed_worker_accounting"
    if failed:
        return "engineering_failure"
    return "passed_r029_engineering_only"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--activity-train-rounds", type=int, default=ACTIVITY_TRAIN_ROUNDS)
    parser.add_argument("--activity-eval-rounds", type=int, default=ACTIVITY_EVAL_ROUNDS)
    parser.add_argument("--skip-boundaries", action="store_true")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("R029 report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    baseline = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def guard() -> None:
        if time.process_time() - total_cpu > BUDGET_CPU_SECONDS:
            raise Budget("cpu budget exhausted")
        if time.perf_counter() - total_wall > BUDGET_WALL_SECONDS:
            raise Budget("wall budget exhausted")

    budget_ok = True
    sections: dict = {}
    for name, function in (
        ("constraint", constraint_section),
        ("control", control_section),
        (
            "activity",
            lambda guard: activity_section(
                guard, args.activity_train_rounds, args.activity_eval_rounds
            ),
        ),
        ("lesion", lesion_section),
        ("fixture", fixture_section),
        ("worker", worker_section),
    ):
        try:
            sections[name] = function(guard)
        except Budget as exc:
            sections[name] = {"status": "budget_stop", "reason": str(exc)}
            budget_ok = False
            break
        print(f"section {name} done cpu={time.process_time() - total_cpu:.1f}", flush=True)
    if budget_ok and not args.skip_boundaries:
        try:
            sections["boundary"] = boundary_section(guard)
        except Budget as exc:
            sections["boundary"] = {"status": "budget_stop", "reason": str(exc)}
            budget_ok = False
    if args.skip_boundaries:
        sections["boundary"] = {"status": "skipped_by_flag", "worker_calls": 0}

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    budget_ok = (
        budget_ok and cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS
    )
    budget_check = {
        "id": "budget_not_exceeded",
        "passed": budget_ok,
        "observed": [cpu_seconds, wall_seconds],
    }
    incomplete = [
        key for key in REQUIRED_SECTIONS if key not in sections or "status" in sections[key]
    ]
    if not budget_ok or incomplete:
        checks = [budget_check]
        status = "budget_stop" if not budget_ok else "engineering_failure"
    else:
        checks = build_checks(sections) + [budget_check]
        status = classify(checks, budget_ok)
    report = {
        "schema": "taiji-r029-engineering-gaps-v1",
        "date": "2026-11-01",
        "baseline_commit": baseline,
        "status": status,
        "configs": {
            "small_reference": asdict(pricing.SMALL),
            "principle_reference": asdict(pricing.PRINCIPLE),
        },
        "faces": {
            "constraint": "principle shape, R028 long Chinese career as fit journal",
            "control": "principle shape, same journals, unconstrained Adam lr=1e-3",
            "activity": "small shape, R028 short career as P_R probe journal",
            "lesion": "principle shape, coverage_birth arm at the R028 frozen lr",
            "fixture": "shape independent generator and shortcut probe",
            "worker": "principle shape fresh-spawn microstep",
            "boundary": "R027 declared eight-boundary life sequence",
        },
        "sections": sections,
        "incomplete_sections": incomplete,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "checks": checks,
        "training_admitted": False,
        "task_capability_claimed": False,
        "default_model_replaced": False,
        "data_splits_read": [],
        "source_hashes": {
            relative: pricing.digest(ROOT / relative)
            for relative in (
                "taiji/living_core.py",
                "taiji/growth_credit.py",
                "taiji/native_development.py",
                "taiji/native_activity.py",
                "taiji/state_migration.py",
                "taiji/constrained_learning.py",
                "taiji/activity_learning.py",
                "taiji/native_task_fixtures.py",
                "scripts/training/verify_taiji_living_fragment_pricing_r028.py",
                "scripts/training/verify_taiji_r029_engineering_gaps.py",
                "tests/taiji_native/test_r029_engineering_gaps.py",
                "tests/taiji_native/test_subject_gates_r027.py",
            )
        },
        "limits": (
            "Engineering gaps only. Passing means the five declared mechanisms run and satisfy "
            "their declared gates on synthetic Chinese fixtures; it does not establish task "
            "capability, growth benefit, autonomy, retention of real skills or training admission. "
            "Two readings are disclosed rather than smoothed over: the activity arm's advantage "
            "can be pure compute-cost avoidance when the trained policy only selects idle "
            "activities, and the lesion arm can show that newborn units carry no measurable share "
            "of the holdout benefit. See sections.activity.advantage_disclosure and "
            "sections.lesion.contribution_disclosure for the measured values."
        ),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "cpu_seconds": round(cpu_seconds, 3),
                "wall_seconds": round(wall_seconds, 3),
                "failed_checks": [row["id"] for row in checks if not row["passed"]],
            },
            ensure_ascii=True,
        )
    )
    return int(status != "passed_r029_engineering_only")


if __name__ == "__main__":
    raise SystemExit(main())
