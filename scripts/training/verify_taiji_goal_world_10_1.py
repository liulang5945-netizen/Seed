"""10.1 verification: does the task environment itself pass its three design gates?

No model is involved. The environment must show goal-sensitivity, a complete
partition, and resistance to four declared shortcut policies before any later
step is allowed to point a model at it.
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

from taiji.native_goal_world import (  # noqa: E402
    MAX_STEPS,
    OBJECTS,
    PLACES,
    SHORTCUT_POLICIES,
    build_fixtures,
    encode_actions,
    legal_moves,
    optimal,
    run_policy,
    score,
    step,
)

BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
SEED = 20261101


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.1 report is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    fixtures = build_fixtures()

    rows = []
    for fixture in fixtures:
        best = optimal(fixture.initial, fixture.goal)
        policies = {
            name: run_policy(policy, fixture.initial, fixture.goal, seed=SEED)
            for name, policy in SHORTCUT_POLICIES.items()
        }
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "goal_id": fixture.goal.goal_id,
                "goal_text": fixture.goal.text,
                "initial": [list(pair) for pair in fixture.initial.where],
                "optimal_score": best["score"],
                "optimal_moves": [list(move) for move in best["moves"]],
                "optimal_steps": len(best["moves"]),
                "policies": {
                    name: {
                        "score": result["score"],
                        "actions": result["actions"],
                        "reaches_optimal": abs(result["score"] - best["score"]) <= 1e-12,
                    }
                    for name, result in policies.items()
                },
            }
        )

    by_goal = {}
    for row in rows:
        by_goal.setdefault(row["goal_id"], []).append(row)
    shared_goal_pairs = sum(len(group) - 1 for group in by_goal.values() if len(group) > 1)
    initials = {}
    for row in rows:
        initials.setdefault(tuple(map(tuple, row["initial"])), []).append(row)
    shared_initial_groups = [group for group in initials.values() if len(group) > 1]
    goal_sensitive = bool(shared_initial_groups) and all(
        len({tuple(map(tuple, row["optimal_moves"])) for row in group}) > 1
        for group in shared_initial_groups
    )
    partition_ok = (
        len({(tuple(map(tuple, row["initial"])), row["goal_id"]) for row in rows}) >= 3
        and bool(shared_initial_groups)
        and shared_goal_pairs >= 2
    )

    policy_verdicts = {}
    for name in SHORTCUT_POLICIES:
        scores = [row["policies"][name]["score"] for row in rows]
        bests = [row["optimal_score"] for row in rows]
        policy_verdicts[name] = {
            "fails_some_fixture": any(s < b - 1e-12 for s, b in zip(scores, bests, strict=True)),
            "optimal_on_all": all(abs(s - b) <= 1e-12 for s, b in zip(scores, bests, strict=True)),
        }
    shortcuts_blocked = all(
        row["fails_some_fixture"] and not row["optimal_on_all"] for row in policy_verdicts.values()
    )

    state = fixtures[0].initial
    moves = legal_moves(state)
    first = encode_actions(moves, seed=SEED, step_index=0)
    second = encode_actions(moves, seed=SEED, step_index=1)
    moved = step(state, moves[0])
    reencoded = encode_actions(legal_moves(moved), seed=SEED, step_index=0)
    ids_differ = first != second
    same_move_different_id = any(
        payload["object"] == first["a0"]["object"] and payload["place"] == first["a0"]["place"]
        for payload in second.values()
    ) or any(
        payload["object"] == first["a0"]["object"] and payload["place"] == first["a0"]["place"]
        for payload in reencoded.values()
    )
    terminal_only = True
    for move in legal_moves(state):
        left = score(step(state, move), fixtures[0].goal)
        right = score(step(state, move), fixtures[0].goal)
        terminal_only = terminal_only and abs(left - right) <= 1e-12

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    checks = [
        {
            "id": "goal_changes_optimal_behavior",
            "passed": goal_sensitive,
            "observed": {row["fixture_id"]: row["optimal_moves"] for row in rows},
        },
        {
            "id": "partition_complete",
            "passed": partition_ok,
            "observed": {
                "fixtures": len(rows),
                "shared_initial_groups": len(shared_initial_groups),
                "shared_goal_pairs": shared_goal_pairs,
            },
        },
        {"id": "shortcuts_cannot_solve", "passed": shortcuts_blocked, "observed": policy_verdicts},
        {
            "id": "action_ids_carry_no_correctness",
            "passed": ids_differ and same_move_different_id,
            "observed": [list(first), list(second)],
        },
        {
            "id": "scoring_reads_terminal_state_only",
            "passed": terminal_only,
            "observed": terminal_only,
        },
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    failed = [row["id"] for row in checks if not row["passed"]]
    if "budget_not_exceeded" in failed:
        status = "budget_stop"
    elif "goal_changes_optimal_behavior" in failed:
        status = "goal_insensitive"
    elif "partition_complete" in failed:
        status = "partition_incomplete"
    elif "shortcuts_cannot_solve" in failed:
        status = "shortcut_can_solve"
    elif failed:
        status = "engineering_failure"
    else:
        status = "passed_10_1_environment_only"

    report = {
        "schema": "taiji-goal-world-10-1-v1",
        "step": "10.1",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "objects": list(OBJECTS),
            "places": list(PLACES),
            "max_steps": MAX_STEPS,
            "action_seed": SEED,
            "shortcut_policies": list(SHORTCUT_POLICIES),
            "search": "exhaustive breadth-first over 9 world states",
        },
        "fixtures": rows,
        "checks": checks,
        "model_involved": False,
        "training_admitted": False,
        "task_capability_claimed": False,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_world.py",
                "scripts/training/verify_taiji_goal_world_10_1.py",
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
                "fixtures": len(rows),
                "optimal": {row["fixture_id"]: row["optimal_score"] for row in rows},
                "shortcuts": {name: row["optimal_on_all"] for name, row in policy_verdicts.items()},
                "cpu_seconds": round(cpu_seconds, 2),
                "failed": failed,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
