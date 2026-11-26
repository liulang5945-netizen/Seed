"""10.1b verification: capacity-1 places, the old path untouched, and planning separated.

Three questions in one run: does ``occupy=False`` still reproduce the 10.1 readings,
do the three 10.1 design gates still hold under capacity 1, and does a depth-2
planner now beat a greedy heuristic somewhere without ever being worse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji.goal_binding import bind  # noqa: E402
from taiji.goal_planner import DEPTH, run_arm  # noqa: E402
from taiji.native_goal_world import (  # noqa: E402
    SHORTCUT_POLICIES,
    build_fixtures,
    optimal,
    run_policy,
)

BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
SEED = 20261101
OLD_OPTIMA = {
    "f1-cup-to-sill-from-table": 0.99,
    "f2-seal-to-table-from-box": 0.99,
    "f3-cup-already-on-sill": 1.0,
    "f4-seal-to-table-from-sill": 0.99,
}


def shortlist(fixtures, *, occupy):
    rows = []
    for fixture in fixtures:
        best = optimal(fixture.initial, fixture.goal, occupy=occupy)
        policies = {
            name: run_policy(policy, fixture.initial, fixture.goal, seed=SEED, occupy=occupy)
            for name, policy in SHORTCUT_POLICIES.items()
        }
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "goal_id": fixture.goal.goal_id,
                "initial": [list(pair) for pair in fixture.initial.where],
                "optimal_score": best["score"],
                "optimal_moves": [list(move) for move in best["moves"]],
                "policies": {
                    name: {
                        "score": result["score"],
                        "reaches_optimal": abs(result["score"] - best["score"]) <= 1e-12,
                    }
                    for name, result in policies.items()
                },
            }
        )
    return rows


def gates(rows) -> dict:
    shared_initials = {}
    for row in rows:
        shared_initials.setdefault(tuple(map(tuple, row["initial"])), []).append(row)
    groups = [group for group in shared_initials.values() if len(group) > 1]
    goal_sensitive = bool(groups) and all(
        len({tuple(map(tuple, row["optimal_moves"])) for row in group}) > 1 for group in groups
    )
    by_goal = {}
    for row in rows:
        by_goal.setdefault(row["goal_id"], []).append(row)
    shared_goal_pairs = sum(len(group) - 1 for group in by_goal.values() if len(group) > 1)
    partition_ok = (
        len({(tuple(map(tuple, row["initial"])), row["goal_id"]) for row in rows}) >= 3
        and bool(groups)
        and shared_goal_pairs >= 2
    )
    blocked = all(
        any(not row["policies"][name]["reaches_optimal"] for row in rows)
        and not all(row["policies"][name]["reaches_optimal"] for row in rows)
        for name in SHORTCUT_POLICIES
    )
    return {
        "goal_sensitive": goal_sensitive,
        "partition_ok": partition_ok,
        "shortcuts_blocked": blocked,
        "shared_initial_groups": len(groups),
        "shared_goal_pairs": shared_goal_pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.1b report is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    fixtures = build_fixtures()
    old_rows = shortlist(fixtures[:4], occupy=False)
    deep_rows = shortlist(fixtures, occupy=True)
    old_gates = gates(old_rows)
    deep_gates = gates(deep_rows)

    default_unchanged = all(
        abs(row["optimal_score"] - OLD_OPTIMA[row["fixture_id"]]) <= 1e-12 for row in old_rows
    ) and not any(
        all(row["policies"][name]["reaches_optimal"] for row in old_rows)
        for name in SHORTCUT_POLICIES
    )

    planner_scores, greedy_scores, deep = [], [], []
    for fixture in fixtures:
        binding = bind(fixture.goal.text, fixture.initial)
        planner = run_arm(binding, fixture.initial, mode="planner", occupy=True)
        greedy = run_arm(binding, fixture.initial, mode="greedy", occupy=True)
        best = optimal(fixture.initial, fixture.goal, occupy=True)["score"]
        planner_scores.append(planner["score"])
        greedy_scores.append(greedy["score"])
        deep.append(
            {
                "fixture_id": fixture.fixture_id,
                "optimal_score": best,
                "planner_score": planner["score"],
                "greedy_score": greedy["score"],
                "planner_steps": len(planner["ledger"]),
                "greedy_steps": len(greedy["ledger"]),
            }
        )
    planner_not_worse = all(
        p >= g - 1e-12 for p, g in zip(planner_scores, greedy_scores, strict=True)
    )
    planner_strictly_better = any(
        p > g + 1e-12 for p, g in zip(planner_scores, greedy_scores, strict=True)
    )
    blocked_fixture = next(row for row in deep if row["fixture_id"].startswith("g1"))
    blocking_effective = blocked_fixture["greedy_score"] < blocked_fixture["optimal_score"] - 1e-12

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    checks = [
        {
            "id": "default_path_changed",
            "passed": default_unchanged,
            "observed": {row["fixture_id"]: row["optimal_score"] for row in old_rows},
        },
        {"id": "goal_insensitive", "passed": deep_gates["goal_sensitive"], "observed": old_gates},
        {
            "id": "partition_incomplete",
            "passed": deep_gates["partition_ok"],
            "observed": deep_gates,
        },
        {
            "id": "shortcut_can_solve",
            "passed": deep_gates["shortcuts_blocked"],
            "observed": deep_gates,
        },
        {
            "id": "no_planner_separation",
            "passed": planner_not_worse and planner_strictly_better,
            "observed": {"planner": planner_scores, "greedy": greedy_scores},
        },
        {"id": "blocking_not_effective", "passed": blocking_effective, "observed": blocked_fixture},
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    failed = [row["id"] for row in checks if not row["passed"]]
    status = "passed_10_1b_environment_deepened" if not failed else failed[0]

    report = {
        "schema": "taiji-goal-world-10-1b-v1",
        "step": "10.1b",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "capacity": 1,
            "depth": DEPTH,
            "fixtures": len(fixtures),
            "default_occupy": False,
            "old_optima": OLD_OPTIMA,
        },
        "old_path": old_rows,
        "deep_path": deep_rows,
        "planning": deep,
        "checks": checks,
        "model_involved": False,
        "training_admitted": False,
        "task_capability_claimed": False,
        "limits": (
            "Environment deepening only. Separation between the reference planner and a greedy "
            "heuristic says the task can now distinguish them, not that any model plans."
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
                "scripts/training/verify_taiji_goal_world_10_1b.py",
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
                "default_unchanged": default_unchanged,
                "deep": {"planner": planner_scores, "greedy": greedy_scores},
                "cpu_seconds": round(cpu_seconds, 3),
                "failed": failed,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
