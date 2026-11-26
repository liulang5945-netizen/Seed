"""14.2: rerun the 10.7f readout on a stratified split, changing only the split.

10.7f trained on ``rows[:6]`` and held out ``rows[6:]``. That ordered slice left the
whole place value 「木桌」 unseen as a correct answer (0/6 in train, 4/6 in held), so
the held-out set was a zero-shot-label test, not a conditioning test. 14.1 recorded
that defect.

This script keeps everything else identical to 10.7f -- same subject, same frozen
encoder, same 1,536-parameter bilinear readout, same 300 steps, same lr, same seed --
and changes only the partition: within each correct place, one state pair goes to
train and the other to held. Every place and every object therefore appears as a
correct answer in train, and held-out measures a new state with a seen goal.

Pass criterion: held-out six-way top-1 strictly above the best constant (1/6).
No capability claim. Read-only on the parent subject.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.action_verdict_scorer import state_text  # noqa: E402
from taiji.goal_conditioned_training import samples  # noqa: E402
from taiji.goal_state_readout import StateGoalActionReadout  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402

STEPS = 300
LR = 0.05
SEED = 0
RANK = 4
WIDTH = 128
BUDGET_CPU_SECONDS = 40
BUDGET_WALL_SECONDS = 120
SCHEMA = "taiji-split-rerun-v1"


def sentence(move) -> str:
    return f"把{move[0]}放到{move[1]}"


def all_moves() -> list[tuple[str, str]]:
    return [(obj, place) for obj in OBJECTS for place in PLACES]


def features(encoder, sample: dict) -> dict:
    moves = all_moves()
    with torch.no_grad():
        goal = encoder.goal_vector(sample["goal"].encode()).detach()
        state = encoder.context(state_text(sample["state"]).encode()).mean(dim=0).detach()
        actions = [encoder.context(sentence(move).encode()).mean(dim=0).detach() for move in moves]
    return {
        "goal_text": sample["goal"],
        "goal": goal,
        "state": state,
        "actions": actions,
        "moves": [list(move) for move in moves],
        "label": moves.index(tuple(sample["correct"])),
        "correct_move": list(sample["correct"]),
        "state_where": list(sample["state"].where),
    }


def stratified_split(rows: list[dict]) -> tuple[list[int], list[int], dict]:
    """Keep one state pair per place in train and the other in held."""
    by_place: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(rows):
        by_place[sample["correct"][1]].append(index)

    train: list[int] = []
    held: list[int] = []
    layout: dict[str, dict] = {}
    for place in PLACES:
        indices = by_place[place]
        if len(indices) != 4:
            raise SystemExit(f"place {place} has {len(indices)} rows, expected 4")
        # rows come as (stateA obj0, stateA obj1, stateB obj0, stateB obj1)
        train.extend(indices[:2])
        held.extend(indices[2:])
        layout[place] = {
            "train": indices[:2],
            "held": indices[2:],
            "train_state": rows[indices[0]]["state"].where,
            "held_state": rows[indices[2]]["state"].where,
        }
    return sorted(train), sorted(held), layout


def scores(head, goal: torch.Tensor, state: torch.Tensor, actions) -> torch.Tensor:
    return torch.stack([head.score(goal, state, action) for action in actions])


def _states(rows: list[dict], mode: str) -> list[torch.Tensor]:
    if mode == "original":
        return [row["state"] for row in rows]
    if mode == "zero":
        return [torch.zeros_like(row["state"]) for row in rows]
    if mode == "shuffled":
        return [rows[(index + 1) % len(rows)]["state"] for index, _ in enumerate(rows)]
    raise SystemExit(f"unknown state mode {mode!r}")


def run_head(train_rows: list[dict], held_rows: list[dict], mode: str = "original") -> dict:
    head = StateGoalActionReadout(width=WIDTH, rank=RANK, seed=SEED)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    train_states = _states(train_rows, mode)
    held_states = _states(held_rows, mode)
    losses: list[float] = []
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(
            [
                F.cross_entropy(
                    scores(head, row["goal"], train_states[index], row["actions"]),
                    torch.tensor(row["label"]),
                )
                for index, row in enumerate(train_rows)
            ]
        ).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        train_hits = sum(
            int(
                torch.argmax(scores(head, row["goal"], train_states[index], row["actions"])).item()
                == row["label"]
            )
            for index, row in enumerate(train_rows)
        )
        held_hits = sum(
            int(
                torch.argmax(scores(head, row["goal"], held_states[index], row["actions"])).item()
                == row["label"]
            )
            for index, row in enumerate(held_rows)
        )
        held_picks = []
        for index, row in enumerate(held_rows):
            picked = int(
                torch.argmax(scores(head, row["goal"], held_states[index], row["actions"])).item()
            )
            held_picks.append(
                {
                    "goal": row["goal_text"],
                    "correct": row["correct_move"],
                    "picked": row["moves"][picked],
                    "hit": picked == row["label"],
                }
            )
    return {
        "train_top1": train_hits,
        "train_total": len(train_rows),
        "held_top1": held_hits,
        "held_total": len(held_rows),
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "parameters": head.parameter_count(),
        "held_picks": held_picks,
    }


def best_constant(held_rows: list[dict]) -> dict:
    counts: dict[int, int] = defaultdict(int)
    for row in held_rows:
        counts[row["label"]] += 1
    label, hits = max(counts.items(), key=lambda item: item[1])
    return {"label": label, "hits": hits, "total": len(held_rows), "rate": hits / len(held_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("14.2 report is write-once")

    torch.set_num_threads(1)
    started = (time.process_time(), time.perf_counter())
    report: dict = {
        "schema": SCHEMA,
        "step": "14.2",
        "date": "2026-11-02",
        "revision": 2,
        "supersedes": "reports/taiji_core_14_2_split_rerun_v1_no_control_20261102.json",
        "supersede_reason": (
            "v1 measured only the original-state arm. The goal string and the correct "
            "action string are byte-identical in this fixture, so a 6/6 held-out score "
            "cannot be attributed to conditioning without state-removal controls."
        ),
        "status": "engineering_failure",
        "design": {
            "only_variable": "train/held partition",
            "held_from_10_7f": "rows[:6] / rows[6:] (ordered slice)",
            "held_14_2": "one state pair per correct place in train, the other in held",
            "steps": STEPS,
            "lr": LR,
            "seed": SEED,
            "rank": RANK,
            "width": WIDTH,
            "readout": "StateGoalActionReadout, bilinear ([g;s]U).(aV)",
            "changed_from_10_7f": "partition only; subject, encoder, readout shape, steps, lr and seed are identical",
        },
        "coverage": {},
        "readings": {},
        "verdict": {},
        "cost": {},
        "limits": (
            "Split audit rerun only. Same synthetic eleven-token world, four hundred "
            "and eighty byte budget untouched, single deterministic parent. A pass means "
            "goal-conditioned selection is learnable from these features under a fair "
            "split; it is not natural-language capability, not P2 and not L3."
        ),
    }

    parent, subject, provider = prior.build_subject()
    encoder = subject.encoder
    rows = samples()
    train_index, held_index, layout = stratified_split(rows)
    train_rows = [features(encoder, rows[i]) for i in train_index]
    held_rows = [features(encoder, rows[i]) for i in held_index]

    train_places = [row["correct_move"][1] for row in train_rows]
    held_places = [row["correct_move"][1] for row in held_rows]
    report["coverage"] = {
        "objects": list(OBJECTS),
        "places": list(PLACES),
        "total_samples": len(rows),
        "action_space": len(all_moves()),
        "train_index": train_index,
        "held_index": held_index,
        "train_correct_places": {place: train_places.count(place) for place in PLACES},
        "held_correct_places": {place: held_places.count(place) for place in PLACES},
        "every_place_in_train": all(place in train_places for place in PLACES),
        "every_object_in_train": all(
            any(row["correct_move"][0] == obj for row in train_rows) for obj in OBJECTS
        ),
        "state_overlap": sorted(
            {str(row["state_where"]) for row in train_rows}
            & {str(row["state_where"]) for row in held_rows}
        ),
        "layout": {
            place: {
                "train_state": [list(pair) for pair in layout[place]["train_state"]],
                "held_state": [list(pair) for pair in layout[place]["held_state"]],
            }
            for place in PLACES
        },
    }
    if not report["coverage"]["every_place_in_train"]:
        report["status"] = "engineering_failure"
        report["readings"] = {"reason": "stratification did not place every label value in train"}
        return _write(report, args, started)

    arm = run_head(train_rows, held_rows, "original")
    control_zero = run_head(train_rows, held_rows, "zero")
    control_shuffled = run_head(train_rows, held_rows, "shuffled")
    constant = best_constant(held_rows)
    chance = 1.0 / len(all_moves())
    control_best = max(control_zero["held_top1"], control_shuffled["held_top1"])
    state_contributes = arm["held_top1"] > control_best
    report["readings"] = {
        "arm": arm,
        "control_zero_state": control_zero,
        "control_shuffled_state": control_shuffled,
        "best_constant": constant,
        "chance_top1": chance,
        "pass_bar": "held_top1 > best_constant.hits",
        "train_fit_top1": f"{arm['train_top1']}/{arm['train_total']}",
        "held_top1": f"{arm['held_top1']}/{arm['held_total']}",
        "controls_held_top1": {
            "zero_state": control_zero["held_top1"],
            "shuffled_state": control_shuffled["held_top1"],
        },
    }
    passed = arm["held_top1"] > constant["hits"]
    shortcut = control_best >= arm["held_top1"] and arm["held_top1"] > constant["hits"]
    report["verdict"] = {
        "passed_over_constant": passed,
        "state_contributes": state_contributes,
        "shortcut_suspected": shortcut,
        "summary": (
            f"held {arm['held_top1']}/{arm['held_total']} vs best constant {constant['hits']}/{constant['total']}; "
            f"zero-state {control_zero['held_top1']}/{control_zero['held_total']}, "
            f"shuffled-state {control_shuffled['held_top1']}/{control_shuffled['held_total']}"
        ),
    }
    if shortcut:
        report["status"] = "goal_copy_shortcut_confirmed"
    elif passed and state_contributes:
        report["status"] = "selection_learnable_under_fair_split_with_state_contribution"
    elif passed:
        report["status"] = "selection_above_constant_but_state_not_needed"
    else:
        report["status"] = "selection_still_absent_under_fair_split"
    return _write(report, args, started)


def _write(report: dict, args, started) -> int:
    cpu = time.process_time() - started[0]
    wall = time.perf_counter() - started[1]
    report["cost"] = {
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "over_budget": cpu > BUDGET_CPU_SECONDS or wall > BUDGET_WALL_SECONDS,
    }
    if report["cost"]["over_budget"]:
        report["status"] = "budget_stop"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    print(
        json.dumps(
            {"status": report["status"], "verdict": report.get("verdict"), "cost": report["cost"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
