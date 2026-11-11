"""10.7e-2: the raw subject's own six-way ranking, as the floor the 10.7e head must beat.

10.7e-1 showed the trained head ranks the correct move first only 3/6 on held-out goals.
That number means nothing without the same measurement on the untouched subject: if the
raw likelihood already picks 3/6, the head added nothing. This probe ranks all six
actions per goal with ``subject_action_scorer.score_actions`` and reports, next to the
subject's own pick rate, the best fixed-move baseline.

It trains nothing and changes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.goal_conditioned_training import samples  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402
from taiji.subject_action_scorer import score_actions  # noqa: E402

TRAIN_STATES = 3
BUDGET_CPU_SECONDS = 240
BUDGET_WALL_SECONDS = 480


def all_moves() -> list[tuple[str, str]]:
    return [(obj, place) for obj in OBJECTS for place in PLACES]


def best_constant(rows) -> dict:
    counts = Counter(tuple(row["correct"]) for row in rows)
    move, count = counts.most_common(1)[0]
    return {"move": list(move), "hits": count, "total": len(rows)}


def evaluate(subject, rows, provider) -> dict:
    moves = all_moves()
    top1 = 0
    ranks = []
    picks = Counter()
    detail = []
    for row in rows:
        scored = score_actions(subject, row["goal"], moves, provider=provider)
        table = sorted(scored, key=lambda item: (item["nll"], item["move"]))
        order = [tuple(item["move"]) for item in table]
        rank = order.index(tuple(row["correct"])) + 1
        top1 += int(rank == 1)
        ranks.append(rank)
        picks[order[0]] += 1
        detail.append(
            {
                "goal": row["goal"],
                "correct_move": list(row["correct"]),
                "rank": rank,
                "top_move": list(order[0]),
                "nll_spread": table[-1]["nll"] - table[0]["nll"],
            }
        )
    return {
        "top1": top1,
        "total": len(rows),
        "mean_rank": sum(ranks) / len(ranks),
        "picks": [{"move": list(move), "count": count} for move, count in picks.items()],
        "best_constant": best_constant(rows),
        "rows": detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.7e-2 probe is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)

    rows = samples()
    per_state = len(rows) // 6
    train_rows, held_rows = rows[: TRAIN_STATES * per_state], rows[TRAIN_STATES * per_state :]

    held = evaluate(subject, held_rows, provider)
    train = evaluate(subject, train_rows, provider)

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    status = (
        "budget_stop"
        if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS
        else "raw_baseline_measured"
    )

    report = {
        "schema": "taiji-raw-rank-10-7e-2-v1",
        "step": "10.7e-2",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "actions_scored": len(all_moves()),
            "scorer": "taiji.subject_action_scorer.score_actions",
            "training": "none",
            "reference": "10.7e-1 head: held top-1 3/6",
        },
        "parent_model_digest": digest,
        "held_out": held,
        "train": train,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Baseline only: the untouched subject's own likelihood ranking over six actions. "
            "It is the floor for 10.7e-1, not a capability reading; no parameter changes and "
            "no training happens here."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/subject_action_scorer.py",
                "scripts/training/probe_raw_subject_rank_10_7e2.py",
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
                "held_best_constant": held["best_constant"],
                "train_top1": train["top1"],
                "held_mean_rank": round(held["mean_rank"], 3),
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
