"""10.7e-1: rank the correct move against all six actions, using the head 10.7e saved.

10.7e only ever compared the correct move with the one alternative it was trained on.
This probe takes the *same saved head*, scores every (object, place) pair for each goal
and asks whether the correct move ranks first -- one in six by chance. It trains
nothing and loads no new parameters.
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

import torch  # noqa: E402

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.goal_action_readout import BilinearReadout  # noqa: E402
from taiji.goal_conditioned_training import samples  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402

TRAIN_STATES = 3
BUDGET_CPU_SECONDS = 120
BUDGET_WALL_SECONDS = 240
PASS_TOP1 = 5
CHANCE_TOP1 = 6
WEIGHTS = "output/taiji_core_10_7e/readout_rank4.pt"


def sentence(move) -> str:
    return f"把{move[0]}放到{move[1]}"


def load_head(path: Path) -> BilinearReadout:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    head = BilinearReadout(width=int(payload["width"]), rank=int(payload["rank"]))
    with torch.no_grad():
        head.left.copy_(payload["left"])
        head.right.copy_(payload["right"])
    return head


def rank_goal(head, encoder, goal: str, correct) -> dict:
    with torch.no_grad():
        goal_vector = encoder.goal_vector(goal.encode()).detach()
        scored = []
        for obj in OBJECTS:
            for place in PLACES:
                action = encoder.context(sentence((obj, place)).encode()).mean(dim=0)
                scored.append(((obj, place), float(head.score(goal_vector, action))))
    scored.sort(key=lambda item: item[1], reverse=True)
    order = [move for move, _ in scored]
    rank = order.index(tuple(correct)) + 1
    return {
        "goal": goal,
        "correct_move": list(correct),
        "rank": rank,
        "top1": rank == 1,
        "top_move": list(order[0]),
        "scores": [[list(move), score] for move, score in scored],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.7e-1 probe is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    encoder = subject.encoder
    head = load_head(ROOT / WEIGHTS)

    rows = samples()
    per_state = len(rows) // 6
    train_rows, held_rows = rows[: TRAIN_STATES * per_state], rows[TRAIN_STATES * per_state :]

    def evaluate(group):
        results = [rank_goal(head, encoder, row["goal"], row["correct"]) for row in group]
        top1 = sum(1 for row in results if row["top1"])
        return {
            "top1": top1,
            "total": len(results),
            "mean_rank": sum(row["rank"] for row in results) / len(results),
            "rows": results,
        }

    held = evaluate(held_rows)
    train = evaluate(train_rows)

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif held["top1"] >= PASS_TOP1:
        status = "readout_rank_top1_holds"
    else:
        status = "readout_rank_below"

    report = {
        "schema": "taiji-readout-rank-10-7e-1-v1",
        "step": "10.7e-1",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "weights": WEIGHTS,
            "actions_scored": len(OBJECTS) * len(PLACES),
            "chance_top1_per_goal": 1 / CHANCE_TOP1,
            "pass_top1": PASS_TOP1,
            "training": "none; the saved 10.7e head is evaluated as is",
        },
        "parent_model_digest": digest,
        "held_out": held,
        "train": train,
        "model_involved": True,
        "capability_claimed": status == "readout_rank_top1_holds",
        "training_admitted": False,
        "limits": (
            "Evaluation of the saved 10.7e head on a wider action set: six actions per goal "
            "instead of two. The head still reads only the goal and action encodings, and the "
            "goals here are the same six the fixture always had, so this strengthens the "
            "two-way reading rather than adding a new task. Not P2, no new training."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/goal_action_readout.py",
                "scripts/training/probe_readout_rank_10_7e1.py",
                WEIGHTS,
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
                "held_top1": held["top1"],
                "held_total": held["total"],
                "held_mean_rank": round(held["mean_rank"], 3),
                "train_top1": train["top1"],
                "train_total": train["total"],
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
