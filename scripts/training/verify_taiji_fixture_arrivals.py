"""09.6.2 fixture verification: recurring-arrival activity fixture and growth grid.

Both fixtures exist to restore discriminating power, so every gate here is a
reachability gate first: the activity fixture must be able to pay more than idle,
and the growth grid must publish every declared cell rather than the best one.
Synthetic Chinese fixtures only; no sealed data, no product default, no training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_living_fragment_pricing_r028 as pricing  # noqa: E402
from taiji import growth_credit as gc  # noqa: E402
from taiji import native_development as development  # noqa: E402
from taiji.activity_learning import (  # noqa: E402
    INTERACT_REWARD,
    ActivityLearner,
    activity_reward,
    evaluate_activity_policy,
    train_activity_episode,
)
from taiji.native_activity import (  # noqa: E402
    ActivityEnvironment,
    ActivityPolicy,
    run_round,
)
from taiji.native_activity_fixture import (  # noqa: E402
    MESSAGES,
    ArrivalSource,
    oracle_cumulative_reward,
    patches,
)

SEED = 20261101
ACTIVITY_ROUNDS = 32
ACTIVITY_TRAIN_ROUNDS = 256
ACTIVITY_MAX_EXPOSURES = 256
ORACLE_FLOOR = 0.5
TRAINED_ORACLE_SHARE = 0.9
MIN_NON_IDLE_ROUNDS = 8
GRID_LR = (1e-5, 1e-4)
GRID_STEPS = (8, 32, 128)
NEWBORN_SHARE_GATE = 1e-4
RETENTION_GATE = 0.030
BUDGET_CPU_SECONDS = 1800
BUDGET_WALL_SECONDS = 3600
RUNTIME_MODULES = ("taiji/activity_learning.py", "taiji/native_activity_fixture.py")
FORBIDDEN_RUNTIME_TOKENS = ("expected_target", "shortcut_label", "category", "gold")


class Budget(Exception):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def activity_env(subject: str, seed: int):
    torch.manual_seed(seed)
    model, state = pricing.fresh(pricing.SMALL, subject)
    env = ActivityEnvironment(store=pricing.RECORDS, goal=pricing.CHINESE_GOAL, now=0)
    return model, state, env


def scripted_arm(chooser, seed: int = SEED + 1) -> dict:
    model, state, env = activity_env("fixture-scripted", seed)
    policy = ActivityPolicy(pricing.SMALL.width).double()
    arrivals = ArrivalSource()
    last_event = torch.zeros(pricing.SMALL.width, dtype=torch.float64)
    activities, rewards, cumulative = [], [], 0.0
    for _ in range(ACTIVITY_ROUNDS):
        before = state
        with torch.no_grad():
            state, env, entry, last_event, _ = run_round(
                model,
                state,
                env,
                policy,
                [],
                last_event=last_event,
                prediction_error_value=0.0,
                chooser=chooser,
                arrivals=arrivals,
            )
        reward = activity_reward(
            model,
            before,
            state,
            pricing.SHORT,
            pricing.RETENTION,
            entry["step_delta"],
            actual_reward=INTERACT_REWARD if entry.get("external_trigger") else 0.0,
        )["reward"]
        cumulative += reward
        activities.append(entry["activity"])
        rewards.append(reward)
    interacts = [value for name, value in zip(activities, rewards) if name == "interact"]
    return {
        "cumulative_reward": cumulative,
        "activities": activities,
        "interactions": activities.count("interact"),
        "idle_rounds": sum(1 for name in activities if name in ("wait", "maintain")),
        "distinct_activities": sorted(set(activities)),
        "mean_interact_reward": (sum(interacts) / len(interacts)) if interacts else None,
    }


def trained_seed_diagnostic(seeds, guard) -> list[dict]:
    """Post-hoc diagnostic: how often does training reach the reachable optimum?

    This is reported, not gated: the declared gate stays the single seeded run.
    """
    rows = []
    for seed in seeds:
        model, state, env = activity_env("fixture-train", seed)
        learner = ActivityLearner.build(pricing.SMALL.width)
        guard()
        _, _, _ = pricing.timed(
            lambda: train_activity_episode(
                model,
                state,
                env,
                learner,
                pricing.SHORT,
                pricing.RETENTION,
                rounds=ACTIVITY_TRAIN_ROUNDS,
                journal=[],
                last_event=torch.zeros(pricing.SMALL.width, dtype=torch.float64),
                max_exposures=ACTIVITY_MAX_EXPOSURES,
                arrivals=ArrivalSource(),
            )
        )
        arm_model, arm_state, arm_env = activity_env("fixture-eval", seed + 1)
        _, _, _, _, rows_eval, cumulative = evaluate_activity_policy(
            arm_model,
            arm_state,
            arm_env,
            learner.policy,
            pricing.SHORT,
            pricing.RETENTION,
            rounds=ACTIVITY_ROUNDS,
            journal=[],
            last_event=torch.zeros(pricing.SMALL.width, dtype=torch.float64),
            arrivals=ArrivalSource(),
        )
        activities = [row["activity"] for row in rows_eval]
        rows.append(
            {
                "seed": seed,
                "cumulative_reward": cumulative,
                "oracle_share": cumulative / oracle_reward(),
                "interactions": activities.count("interact"),
                "distinct_activities": sorted(set(activities)),
            }
        )
        guard()
    return rows


def oracle_reward() -> float:
    return oracle_cumulative_reward(ACTIVITY_ROUNDS)["oracle_reward"]


def activity_section(guard) -> dict:
    oracle = oracle_cumulative_reward(ACTIVITY_ROUNDS)
    guard()
    idle = scripted_arm(lambda env, error: "wait")
    greedy = scripted_arm(lambda env, error: "interact" if env.message_queue else "wait")
    guard()

    model, state, env = activity_env("fixture-train", SEED)
    learner = ActivityLearner.build(pricing.SMALL.width)
    last_event = torch.zeros(pricing.SMALL.width, dtype=torch.float64)
    guard()
    train_rows, train_cpu, train_wall = pricing.timed(
        lambda: train_activity_episode(
            model,
            state,
            env,
            learner,
            pricing.SHORT,
            pricing.RETENTION,
            rounds=ACTIVITY_TRAIN_ROUNDS,
            journal=[],
            last_event=last_event,
            max_exposures=ACTIVITY_MAX_EXPOSURES,
            arrivals=ArrivalSource(),
        )[4]
    )
    guard()

    arms = {}
    for name, policy, control in (
        ("trained_model", learner.policy, False),
        ("untrained_model", ActivityPolicy(pricing.SMALL.width).double(), False),
        ("fixed_threshold", learner.policy, True),
    ):
        arm_model, arm_state, arm_env = activity_env("fixture-eval", SEED + 1)
        _, _, _, _, rows, cumulative = evaluate_activity_policy(
            arm_model,
            arm_state,
            arm_env,
            policy,
            pricing.SHORT,
            pricing.RETENTION,
            rounds=ACTIVITY_ROUNDS,
            journal=[],
            last_event=torch.zeros(pricing.SMALL.width, dtype=torch.float64),
            control=control,
            arrivals=ArrivalSource(),
        )
        activities = [row["activity"] for row in rows]
        arms[name] = {
            "cumulative_reward": cumulative,
            "interactions": activities.count("interact"),
            "idle_rounds": sum(1 for value in activities if value in ("wait", "maintain")),
            "distinct_activities": sorted(set(activities)),
            "oracle_share": cumulative / oracle["oracle_reward"],
        }
        guard()
    trained = arms["trained_model"]["cumulative_reward"]
    diagnostic = trained_seed_diagnostic((SEED + 11, SEED + 12), guard)
    diagnostic.insert(
        0,
        {
            "seed": SEED,
            "cumulative_reward": trained,
            "oracle_share": trained / oracle["oracle_reward"],
            "interactions": arms["trained_model"]["interactions"],
            "distinct_activities": arms["trained_model"]["distinct_activities"],
        },
    )
    fixed_at_oracle = (
        abs(arms["fixed_threshold"]["cumulative_reward"] - oracle["oracle_reward"]) <= 1e-9
    )
    return {
        "shape": "small",
        "declared": {
            "rounds": ACTIVITY_ROUNDS,
            "train_rounds": ACTIVITY_TRAIN_ROUNDS,
            "max_exposures": ACTIVITY_MAX_EXPOSURES,
            "interval_ticks": oracle["interval_ticks"],
            "message_count": len(MESSAGES),
            "patches_per_message": patches(MESSAGES[0]),
            "oracle_floor": ORACLE_FLOOR,
            "trained_oracle_share_gate": TRAINED_ORACLE_SHARE,
            "min_non_idle_rounds": MIN_NON_IDLE_ROUNDS,
        },
        "oracle": oracle,
        "scripted": {"idle": idle, "greedy_interact": greedy},
        "trained_model": arms["trained_model"],
        "untrained_model": arms["untrained_model"],
        "fixed_threshold": arms["fixed_threshold"],
        "trained_advantage_over_best_control": trained
        - max(
            arms["untrained_model"]["cumulative_reward"],
            arms["fixed_threshold"]["cumulative_reward"],
        ),
        "trained_advantage_over_idle": trained - idle["cumulative_reward"],
        "fixed_control_at_oracle": fixed_at_oracle,
        "seed_diagnostic": diagnostic,
        "seed_diagnostic_reading": (
            "Post-hoc diagnostic only, not a gate: counts how many declared seeds reach "
            "the reachable optimum. A split result means the learning rule, not the "
            "fixture, decides whether the optimum is found."
        ),
        "training": {
            "rows": len(train_rows),
            "updated": sum(1 for row in train_rows if row.get("status") == "updated"),
            "exposures": learner.exposures,
            "cpu_seconds": train_cpu,
            "wall_seconds": train_wall,
        },
        "disclosure": (
            "The fixed ladder already reaches the declared oracle, so a trained "
            "policy matching it demonstrates learning a simple rule, not the "
            "discovery of a better strategy. Idle stays at zero by construction."
        ),
    }


def growth_inputs(parent):
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
    return book, fit_credits, verdicts


def growth_cell(parent, book, fit_credits, verdicts, lr, steps, guard) -> dict:
    child, provider, plan = development.coverage_birth_arm(
        copy.deepcopy(parent), book, fit_credits, verdicts
    )
    holdout_pre = development.evaluate_career_nll(child, pricing.HOLDOUT, provider)
    retention_pre = development.evaluate_career_nll(child, pricing.RETENTION, provider)
    optimizer = torch.optim.Adam(child.parameters(), lr=lr)
    guard()
    losses, train_cpu, train_wall = pricing.timed(
        lambda: development.declared_feedback_updates(
            child, pricing.LONG, optimizer, provider, steps=steps
        )
    )
    holdout_post = development.evaluate_career_nll(child, pricing.HOLDOUT, provider)
    retention_post = development.evaluate_career_nll(child, pricing.RETENTION, provider)
    parent_units = child.growth_parent_units
    guard()
    lesion_report = gc.lesion_trained(child)
    holdout_lesion = development.evaluate_career_nll(child, pricing.HOLDOUT)
    benefit = holdout_pre - holdout_post
    benefit_after = holdout_pre - holdout_lesion
    share = (benefit - benefit_after) / benefit if benefit else None
    return {
        "lr": lr,
        "steps": steps,
        "units_total": plan["units_total"],
        "growth_parent_units": parent_units,
        "removed_units_per_group": lesion_report["removed_units_per_group"],
        "holdout_pre": holdout_pre,
        "holdout_post": holdout_post,
        "holdout_after_lesion": holdout_lesion,
        "benefit": benefit,
        "benefit_removed_by_lesion": benefit - benefit_after,
        "newborn_share": share,
        "retention_drift": abs(retention_post - retention_pre),
        "first_update_loss": losses[0],
        "last_update_loss": losses[-1],
        "reaches_gate": bool(
            share is not None
            and share >= NEWBORN_SHARE_GATE
            and abs(retention_post - retention_pre) <= RETENTION_GATE
        ),
        "cpu_seconds": train_cpu,
        "wall_seconds": train_wall,
    }


def growth_section(guard) -> dict:
    torch.manual_seed(SEED)
    parent, _ = pricing.fresh(pricing.PRINCIPLE, "fixture-growth")
    guard()
    (book, fit_credits, verdicts), build_cpu, build_wall = pricing.timed(
        lambda: growth_inputs(parent)
    )
    guard()
    cells = []
    for lr in GRID_LR:
        for steps in GRID_STEPS:
            cells.append(growth_cell(parent, book, fit_credits, verdicts, lr, steps, guard))
    reachable = [cell for cell in cells if cell["reaches_gate"]]
    best = max(
        cells,
        key=lambda cell: (cell["newborn_share"] if cell["newborn_share"] is not None else -1.0),
    )
    return {
        "shape": "principle",
        "declared": {
            "lr_grid": list(GRID_LR),
            "steps_grid": list(GRID_STEPS),
            "newborn_share_gate": NEWBORN_SHARE_GATE,
            "retention_gate": RETENTION_GATE,
        },
        "cells": cells,
        "cells_published": len(cells),
        "cells_expected": len(GRID_LR) * len(GRID_STEPS),
        "reachable_cells": [{"lr": cell["lr"], "steps": cell["steps"]} for cell in reachable],
        "best_cell": {
            "lr": best["lr"],
            "steps": best["steps"],
            "newborn_share": best["newborn_share"],
            "retention_drift": best["retention_drift"],
        },
        "verdict": (
            "growth_fixture_reachable" if reachable else "growth_fixture_no_reachable_cell"
        ),
        "credit_build": {"cpu_seconds": build_cpu, "wall_seconds": build_wall},
        "total_cpu_seconds": sum(cell["cpu_seconds"] for cell in cells),
    }


def source_guard() -> dict:
    counts = {}
    for relative in RUNTIME_MODULES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        counts[relative] = {token: text.count(token) for token in FORBIDDEN_RUNTIME_TOKENS}
    return counts


def build_checks(activity: dict, growth: dict, tokens: dict) -> list[dict]:
    oracle = activity["oracle"]
    return [
        {
            "id": "activity_fixture_reachable",
            "passed": oracle["oracle_reward"] >= ORACLE_FLOOR
            and oracle["legal_interactions"] >= ACTIVITY_ROUNDS // 2,
            "observed": [oracle["oracle_reward"], oracle["legal_interactions"]],
        },
        {
            "id": "activity_scripted_policy_reaches_floor",
            "passed": activity["scripted"]["greedy_interact"]["cumulative_reward"] >= ORACLE_FLOOR,
            "observed": activity["scripted"]["greedy_interact"]["cumulative_reward"],
        },
        {
            "id": "activity_interaction_pays",
            "passed": (activity["scripted"]["greedy_interact"]["mean_interact_reward"] or 0.0)
            > 0.01,
            "observed": activity["scripted"]["greedy_interact"]["mean_interact_reward"],
        },
        {
            "id": "activity_idle_is_zero",
            "passed": activity["scripted"]["idle"]["cumulative_reward"] == 0.0,
            "observed": activity["scripted"]["idle"]["cumulative_reward"],
        },
        {
            "id": "activity_trained_meets_oracle_share",
            "passed": activity["trained_model"]["cumulative_reward"]
            >= TRAINED_ORACLE_SHARE * oracle["oracle_reward"],
            "observed": activity["trained_model"]["oracle_share"],
        },
        {
            "id": "activity_trained_beats_idle",
            "passed": activity["trained_advantage_over_idle"] >= 0.1,
            "observed": activity["trained_advantage_over_idle"],
        },
        {
            "id": "activity_trained_not_idle_only",
            "passed": activity["trained_model"]["idle_rounds"]
            <= ACTIVITY_ROUNDS - MIN_NON_IDLE_ROUNDS,
            "observed": activity["trained_model"]["idle_rounds"],
        },
        {
            "id": "activity_fixed_control_disclosed",
            "passed": isinstance(activity["fixed_control_at_oracle"], bool),
            "observed": activity["fixed_control_at_oracle"],
        },
        {
            "id": "growth_grid_fully_published",
            "passed": growth["cells_published"] == growth["cells_expected"],
            "observed": [growth["cells_published"], growth["cells_expected"]],
        },
        {
            "id": "growth_grid_cells_all_measured",
            "passed": all(
                cell["last_update_loss"] == cell["last_update_loss"] for cell in growth["cells"]
            ),
            "observed": [cell["last_update_loss"] for cell in growth["cells"]],
        },
        {
            "id": "growth_verdict_recorded",
            "passed": growth["verdict"]
            in {"growth_fixture_reachable", "growth_fixture_no_reachable_cell"},
            "observed": growth["verdict"],
        },
        {
            "id": "runtime_modules_read_no_gold",
            "passed": all(count == 0 for row in tokens.values() for count in row.values()),
            "observed": tokens,
        },
    ]


def classify(checks: list[dict], growth: dict, budget_ok: bool) -> str:
    failed = {row["id"] for row in checks if not row["passed"]}
    if not budget_ok:
        return "budget_stop"
    if any(name.startswith("activity_") for name in failed):
        return "failed_activity_fixture"
    if any(name.startswith("growth_") or name.startswith("runtime_") for name in failed):
        return "failed_growth_fixture"
    if growth["verdict"] == "growth_fixture_no_reachable_cell":
        return "growth_fixture_no_reachable_cell"
    return "passed_09_6_2_fixture_engineering"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--activity-only", action="store_true")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("09.6.2 fixture report is write-once")

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

    sections: dict = {}
    budget_ok = True
    try:
        sections["activity"] = activity_section(guard)
        print(f"activity done cpu={time.process_time() - total_cpu:.1f}", flush=True)
        if args.activity_only:
            sections["growth"] = {"status": "skipped_by_flag"}
        else:
            sections["growth"] = growth_section(guard)
            print(f"growth done cpu={time.process_time() - total_cpu:.1f}", flush=True)
    except Budget as exc:
        budget_ok = False
        sections.setdefault("activity", {"status": "budget_stop", "reason": str(exc)})
        sections.setdefault("growth", {"status": "budget_stop", "reason": str(exc)})
    tokens = source_guard()

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
    incomplete = [key for key in ("activity", "growth") if "status" in sections.get(key, {})]
    if not budget_ok or incomplete:
        checks = [budget_check]
        status = "budget_stop" if not budget_ok else "engineering_failure"
    else:
        checks = build_checks(sections["activity"], sections["growth"], tokens) + [budget_check]
        status = classify(checks, sections["growth"], budget_ok)

    report = {
        "schema": "taiji-fixture-arrivals-v1",
        "date": "2026-11-01",
        "step": "09.6.2",
        "baseline_commit": baseline,
        "status": status,
        "configs": {
            "small_reference": asdict(pricing.SMALL),
            "principle_reference": asdict(pricing.PRINCIPLE),
        },
        "sections": sections,
        "incomplete_sections": incomplete,
        "runtime_source_token_counts": tokens,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "checks": checks,
        "training_admitted": False,
        "task_capability_claimed": False,
        "data_splits_read": [],
        "limits": (
            "Reachability instrumentation only. A reachable activity fixture and a "
            "reachable growth cell mean the question can be asked, not that the "
            "subject learned useful autonomy or that structural birth pays for itself."
        ),
        "source_hashes": {
            relative: digest(ROOT / relative)
            for relative in (
                "taiji/living_core.py",
                "taiji/growth_credit.py",
                "taiji/native_development.py",
                "taiji/native_activity.py",
                "taiji/activity_learning.py",
                "taiji/native_activity_fixture.py",
                "scripts/training/verify_taiji_living_fragment_pricing_r028.py",
                "scripts/training/verify_taiji_fixture_arrivals.py",
                "tests/taiji_native/test_fixture_arrivals.py",
            )
        },
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
    return int(status != "passed_09_6_2_fixture_engineering")


if __name__ == "__main__":
    raise SystemExit(main())
