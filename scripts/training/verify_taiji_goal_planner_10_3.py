"""10.3 verification: prediction/rollout/selection contract and four contrast arms.

No model is trained here. The reference planner is scored against the exhaustive
optimum, a greedy heuristic, random selection, a perturbed world model and a
deliberately wrong binding.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import taiji.goal_planner as planner_module  # noqa: E402
from taiji.goal_binding import Binding, bind  # noqa: E402
from taiji.goal_planner import DEPTH, run_arm, state_score  # noqa: E402
from taiji.native_goal_world import PLACES, build_fixtures, optimal  # noqa: E402

BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600


def wrong_binding(binding: Binding) -> Binding:
    conditions = []
    for obj, place, polarity in binding.conditions:
        shifted = PLACES[(PLACES.index(place) + 1) % len(PLACES)]
        conditions.append((obj, shifted, polarity))
    return Binding(tuple(conditions))


def scoring_takes_no_text() -> bool:
    tree = ast.parse(inspect.getsource(planner_module))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "state_score"
    ]
    if not calls:
        return False
    for call in calls:
        if any(keyword.arg in ("explanation", "text", "goal_text") for keyword in call.keywords):
            return False
    return list(inspect.signature(state_score).parameters) == ["state", "binding"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--occupy", action="store_true")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.3 report is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    fixtures = build_fixtures()
    rows = []
    for fixture in fixtures:
        binding = bind(fixture.goal.text, fixture.initial)
        optimum = optimal(fixture.initial, fixture.goal, occupy=args.occupy)["score"]
        arms = {
            "planner": run_arm(binding, fixture.initial, mode="planner", occupy=args.occupy),
            "greedy": run_arm(binding, fixture.initial, mode="greedy", occupy=args.occupy),
            "no_prediction": run_arm(
                binding, fixture.initial, mode="no_prediction", occupy=args.occupy
            ),
            "perturbed_model": run_arm(
                binding, fixture.initial, mode="planner", perturbed=True, occupy=args.occupy
            ),
            "wrong_goal": run_arm(
                wrong_binding(binding), fixture.initial, mode="planner", occupy=args.occupy
            ),
        }
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "optimal_score": optimum,
                "binding": [list(condition) for condition in binding.conditions],
                "arms": {
                    name: {
                        "score": result["score"],
                        "steps": len(result["ledger"]),
                        "imagined_lengths": [len(row["imagined"]) for row in result["ledger"]],
                        "all_predictions_match": (
                            all(row["predicted_matches_actual"] for row in result["ledger"])
                            if result["ledger"]
                            else None
                        ),
                        "ledger": result["ledger"],
                    }
                    for name, result in arms.items()
                },
            }
        )

    separation_ok = all(
        len(row["imagined_lengths"]) == row["steps"]
        and all(length <= DEPTH for length in row["imagined_lengths"])
        for row in rows
        for name, row in row["arms"].items()
        if name == "planner"
    )
    planner_matches = [
        row["arms"]["planner"]["all_predictions_match"]
        for row in rows
        if row["arms"]["planner"]["steps"]
    ]
    perturbed_mismatch = any(
        row["arms"]["perturbed_model"]["all_predictions_match"] is False
        for row in rows
        if row["arms"]["perturbed_model"]["steps"]
    )
    prediction_column_ok = all(planner_matches) and perturbed_mismatch
    planner_scores = [row["arms"]["planner"]["score"] for row in rows]
    greedy_scores = [row["arms"]["greedy"]["score"] for row in rows]
    planner_not_worse = all(
        p >= g - 1e-12 for p, g in zip(planner_scores, greedy_scores, strict=True)
    )
    planner_strictly_better = any(
        p > g + 1e-12 for p, g in zip(planner_scores, greedy_scores, strict=True)
    )
    wrong_goal_used = any(
        row["arms"]["wrong_goal"]["score"] < row["arms"]["planner"]["score"] - 1e-12 for row in rows
    )
    random_scores = [row["arms"]["no_prediction"]["score"] for row in rows]
    random_not_sweeping = any(
        r < p - 1e-12 for r, p in zip(random_scores, planner_scores, strict=True)
    )
    scored_without_text = scoring_takes_no_text()

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    checks = [
        {"id": "ledger_not_separated", "passed": separation_ok, "observed": separation_ok},
        {
            "id": "prediction_column_blind",
            "passed": prediction_column_ok,
            "observed": {
                "planner_all_match": all(planner_matches),
                "perturbed_mismatch": perturbed_mismatch,
            },
        },
        {
            "id": "planner_no_advantage",
            "passed": planner_not_worse and planner_strictly_better,
            "observed": {"planner": planner_scores, "greedy": greedy_scores},
        },
        {
            "id": "goal_condition_unused",
            "passed": wrong_goal_used,
            "observed": [row["arms"]["wrong_goal"]["score"] for row in rows],
        },
        {
            "id": "explanation_scored",
            "passed": scored_without_text,
            "observed": scored_without_text,
        },
        {
            "id": "fixture_too_narrow",
            "passed": random_not_sweeping,
            "observed": {"random": random_scores, "planner": planner_scores},
        },
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    failed = [row["id"] for row in checks if not row["passed"]]
    order = (
        "budget_not_exceeded",
        "ledger_not_separated",
        "prediction_column_blind",
        "planner_no_advantage",
        "goal_condition_unused",
        "explanation_scored",
        "fixture_too_narrow",
    )
    status = "passed_10_3_prediction_contract_only"
    for name in order:
        if name in failed:
            status = name
            break

    report = {
        "schema": "taiji-goal-planner-10-3-v1",
        "step": "10.3",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "depth": DEPTH,
            "arms": ["planner", "greedy", "no_prediction", "perturbed_model", "wrong_goal"],
            "fixtures": len(rows),
            "capacity_one": args.occupy,
        },
        "fixtures": rows,
        "checks": checks,
        "model_involved": False,
        "training_admitted": False,
        "task_capability_claimed": False,
        "limits": (
            "Reference planner baseline only. If the planner does not beat a greedy heuristic, the "
            "declared world is too shallow for this step and the task must be deepened before any "
            "planning claim; no model was involved and no capability is claimed."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_world.py",
                "taiji/goal_binding.py",
                "taiji/goal_planner.py",
                "scripts/training/verify_taiji_goal_planner_10_3.py",
                "tests/taiji_native/test_goal_planner.py",
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
                "planner": planner_scores,
                "greedy": greedy_scores,
                "random": random_scores,
                "wrong_goal": [row["arms"]["wrong_goal"]["score"] for row in rows],
                "perturbed": [row["arms"]["perturbed_model"]["score"] for row in rows],
                "cpu_seconds": round(cpu_seconds, 3),
                "failed": failed,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
