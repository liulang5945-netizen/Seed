"""10.4 verification: the first capability reading where the subject itself chooses.

The subject is rebuilt exactly as the 09.8 closeout built it, its identity is checked
against that card, and its selections are compared with the greedy heuristic and a
seeded random arm. A losing subject is a valid, reportable outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_birth_direction as rule  # noqa: E402
import scripts.training.verify_taiji_living_fragment_pricing_r028 as pricing  # noqa: E402
from taiji import birth_direction as bd  # noqa: E402
from taiji import growth_credit as gc  # noqa: E402
from taiji import native_development as development  # noqa: E402
from taiji.goal_binding import bind  # noqa: E402
from taiji.goal_planner import state_score  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_goal_world import PLACES, build_fixtures, legal_moves, optimal, step  # noqa: E402
from taiji.subject_action_scorer import select_action  # noqa: E402

SEED = 20261101
PARTS = 5
UNITS_PER_GROUP = 3
STEPS = 8
LR = 1e-5
LIMIT = 6
ADVANTAGE_MARGIN = 0.05
CLOSEOUT_CARD = "reports/taiji_core_09_closeout_20261101.json"
BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200


class Budget(Exception):
    pass


def parent_digest(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_subject():
    torch.manual_seed(SEED)
    parent = LivingCoreReference(pricing.PRINCIPLE).double()
    directions = bd.consensus_directions(
        deepcopy(parent), pricing.LONG, parts=PARTS, count=UNITS_PER_GROUP
    )
    child, _plan = gc.birth(deepcopy(parent), directions, rule._credits(parent))
    provider = rule.open_gates(bd.RULE_NAME)
    optimizer = torch.optim.Adam(child.parameters(), lr=LR)
    development.declared_feedback_updates(child, pricing.LONG, optimizer, provider, steps=STEPS)
    return parent, child, provider


def play_subject(subject, provider, fixture) -> dict:
    binding = bind(fixture.goal.text, fixture.initial)
    state, actions, tables = fixture.initial, [], []
    for _ in range(LIMIT):
        if binding.satisfied(state):
            break
        moves = legal_moves(state, occupy=True)
        choice = select_action(subject, fixture.goal.text, moves, provider=provider)
        if choice["scored"] != len(moves):
            raise RuntimeError("selector scored a different number of actions than are legal")
        tables.append(choice["table"])
        state = step(state, choice["action"], occupy=True)
        actions.append(choice["action"])
    return {"score": state_score(state, binding), "actions": actions, "tables": tables}


def play_greedy(fixture) -> dict:
    binding = bind(fixture.goal.text, fixture.initial)
    targets = {obj: place for obj, place, polarity in binding.conditions if polarity > 0}
    state, actions = fixture.initial, []
    for _ in range(LIMIT):
        if binding.satisfied(state):
            break
        moves = legal_moves(state, occupy=True)
        chosen = min(
            moves,
            key=lambda move: (
                (
                    len(PLACES)
                    if move[0] not in targets
                    else abs(PLACES.index(move[1]) - PLACES.index(targets[move[0]]))
                ),
                PLACES.index(move[1]),
            ),
        )
        state = step(state, chosen, occupy=True)
        actions.append(chosen)
    return {"score": state_score(state, binding), "actions": actions}


def play_random(fixture, seed: int) -> dict:
    binding = bind(fixture.goal.text, fixture.initial)
    state, actions = fixture.initial, []
    for index in range(LIMIT):
        if binding.satisfied(state):
            break
        moves = legal_moves(state, occupy=True)
        chosen = moves[(seed + index + len(actions)) % len(moves)]
        state = step(state, chosen, occupy=True)
        actions.append(chosen)
    return {"score": state_score(state, binding), "actions": actions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.4 report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()

    def guard() -> None:
        if time.process_time() - total_cpu > BUDGET_CPU_SECONDS:
            raise Budget("cpu budget exhausted")
        if time.perf_counter() - total_wall > BUDGET_WALL_SECONDS:
            raise Budget("wall budget exhausted")

    card = json.loads((ROOT / CLOSEOUT_CARD).read_text(encoding="utf-8"))
    expected_identity = card["identity"]["parent_model_digest"]
    guard()
    parent, subject, provider = build_subject()
    identity_ok = parent_digest(parent) == expected_identity

    rows = []
    for fixture in build_fixtures():
        guard()
        first = play_subject(subject, provider, fixture)
        guard()
        repeat = play_subject(subject, provider, fixture)
        greedy = play_greedy(fixture)
        random_arm = play_random(fixture, SEED)
        best = optimal(fixture.initial, fixture.goal, occupy=True)["score"]
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "goal_text": fixture.goal.text,
                "optimal_score": best,
                "subject": {
                    "score": first["score"],
                    "actions": [list(move) for move in first["actions"]],
                    "tables": [
                        [{**row, "move": list(row["move"])} for row in table]
                        for table in first["tables"]
                    ],
                },
                "greedy": {
                    "score": greedy["score"],
                    "actions": [list(move) for move in greedy["actions"]],
                },
                "random": {
                    "score": random_arm["score"],
                    "actions": [list(move) for move in random_arm["actions"]],
                },
                "deterministic": first["actions"] == repeat["actions"],
                "scored_counts": [len(table) for table in first["tables"]],
                "legal_counts": [
                    len(legal_moves(fixture.initial, occupy=True)) for _ in first["tables"]
                ],
            }
        )

    subject_total = sum(row["subject"]["score"] for row in rows)
    greedy_total = sum(row["greedy"]["score"] for row in rows)
    random_total = sum(row["random"]["score"] for row in rows)
    deterministic = all(row["deterministic"] for row in rows)
    scored_ok = all(
        all(
            count == legal
            for count, legal in zip(row["scored_counts"], row["legal_counts"], strict=True)
        )
        for row in rows
        if row["scored_counts"]
    )
    if subject_total >= random_total + ADVANTAGE_MARGIN:
        outcome = "model_advantage_over_random"
    elif abs(subject_total - greedy_total) <= 1e-12:
        outcome = "model_matches_greedy"
    else:
        outcome = "model_below_random"

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif not identity_ok:
        status = "candidate_identity_mismatch"
    elif not deterministic:
        status = "nondeterministic_selection"
    elif not scored_ok:
        status = "selector_degenerate"
    else:
        status = outcome

    checks = [
        {
            "id": "candidate_identity_mismatch",
            "passed": identity_ok,
            "observed": [parent_digest(parent), expected_identity],
        },
        {
            "id": "nondeterministic_selection",
            "passed": deterministic,
            "observed": [row["deterministic"] for row in rows],
        },
        {
            "id": "selector_degenerate",
            "passed": scored_ok,
            "observed": [row["scored_counts"] for row in rows],
        },
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    report = {
        "schema": "taiji-subject-selection-10-4-v1",
        "step": "10.4",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "parts": PARTS,
            "units_per_group": UNITS_PER_GROUP,
            "steps": STEPS,
            "lr": LR,
            "limit": LIMIT,
            "advantage_margin": ADVANTAGE_MARGIN,
            "fixtures": len(rows),
            "selection": "subject conditional NLL over rendered action sentences, lowest wins",
        },
        "parent_model_digest": parent_digest(parent),
        "fixtures": rows,
        "totals": {"subject": subject_total, "greedy": greedy_total, "random": random_total},
        "model_involved": True,
        "capability_passed": outcome in {"model_advantage_over_random", "model_matches_greedy"},
        "checks": checks,
        "training_admitted": False,
        "limits": (
            "Six synthetic Chinese fixtures in the 10.1b world. A positive outcome means the subject "
            "beat random selection here; it is not natural-language continuous-task capability, not L3 "
            "evidence, and it does not close the other 11 preconditions."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/living_core.py",
                "taiji/growth_credit.py",
                "taiji/native_development.py",
                "taiji/birth_direction.py",
                "taiji/native_goal_world.py",
                "taiji/goal_binding.py",
                "taiji/subject_action_scorer.py",
                "scripts/training/verify_taiji_subject_selection_10_4.py",
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
                "totals": {
                    "subject": round(subject_total, 3),
                    "greedy": round(greedy_total, 3),
                    "random": round(random_total, 3),
                },
                "identity_ok": identity_ok,
                "cpu_seconds": round(cpu_seconds, 2),
                "failed": [row["id"] for row in checks if not row["passed"]],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
