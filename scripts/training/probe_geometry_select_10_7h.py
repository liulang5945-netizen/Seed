"""10.7h: read the six-way choice geometrically, with no fitted parameters at all.

10.7g showed the encoder's action vectors are almost perfectly additive in object and
place (99.99% of the energy in the two main effects). 10.7f showed a 1,536-parameter
head memorises six samples without transferring. This probe asks whether the additive
geometry alone is enough:

* directions come from the encoder only -- ``d_o`` = mean over places of the object's
  action vectors minus the grand mean, ``d_p`` likewise over objects;
* ``score(goal, action) = cos(g, d_o[action.o]) + cos(g, d_p[action.p])``;
* not one parameter is fitted to the task, and nothing is trained.

The control feeds a **different sample's goal** to the same scorer.
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
from taiji.goal_conditioned_training import samples  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402

TRAIN_STATES = 3
BUDGET_CPU_SECONDS = 180
BUDGET_WALL_SECONDS = 360
PASS_TOP1 = 5
ABSENT_TOP1 = 2


def sentence(obj: str, place: str) -> str:
    return f"把{obj}放到{place}"


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.double()
    second = second.double()
    norm = float(first.norm()) * float(second.norm())
    if norm == 0.0:
        return 0.0
    return float(torch.dot(first, second)) / norm


def directions(encoder) -> dict:
    with torch.no_grad():
        table = {
            (obj, place): encoder.context(sentence(obj, place).encode()).mean(dim=0).detach()
            for obj in OBJECTS
            for place in PLACES
        }
    grand = torch.stack(list(table.values())).mean(dim=0)
    object_direction = {
        obj: torch.stack([table[(obj, place)] for place in PLACES]).mean(dim=0) - grand
        for obj in OBJECTS
    }
    place_direction = {
        place: torch.stack([table[(obj, place)] for obj in OBJECTS]).mean(dim=0) - grand
        for place in PLACES
    }
    return {"object": object_direction, "place": place_direction}


def rank_row(goal_vector, dirs, correct) -> dict:
    scores = []
    for obj in OBJECTS:
        for place in PLACES:
            value = cosine(goal_vector, dirs["object"][obj]) + cosine(
                goal_vector, dirs["place"][place]
            )
            scores.append(((obj, place), value))
    scores.sort(key=lambda item: item[1], reverse=True)
    order = [move for move, _ in scores]
    rank = order.index(tuple(correct)) + 1
    return {"rank": rank, "top1": rank == 1, "top_move": list(order[0])}


def evaluate(encoder, dirs, rows, *, shift: int = 0) -> dict:
    top1 = 0
    detail = []
    with torch.no_grad():
        goals = [encoder.goal_vector(row["goal"].encode()).detach() for row in rows]
    for index, row in enumerate(rows):
        goal_vector = goals[(index + shift) % len(rows)]
        result = rank_row(goal_vector, dirs, row["correct"])
        top1 += int(result["top1"])
        detail.append(
            {
                "goal": row["goal"],
                "correct_move": list(row["correct"]),
                "rank": result["rank"],
                "top_move": result["top_move"],
            }
        )
    return {"top1": top1, "total": len(rows), "rows": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.7h probe is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    dirs = directions(subject.encoder)

    rows = samples()
    per_state = len(rows) // 6
    train_rows, held_rows = rows[: TRAIN_STATES * per_state], rows[TRAIN_STATES * per_state :]

    held = evaluate(subject.encoder, dirs, held_rows)
    train = evaluate(subject.encoder, dirs, train_rows)
    control = evaluate(subject.encoder, dirs, held_rows, shift=1)

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif held["top1"] >= PASS_TOP1:
        status = "geometry_selects"
    elif held["top1"] <= ABSENT_TOP1:
        status = "geometry_does_not_select"
    else:
        status = "geometry_inconclusive"

    report = {
        "schema": "taiji-geometry-10-7h-v1",
        "step": "10.7h",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "samples": len(rows),
            "actions": len(OBJECTS) * len(PLACES),
            "parameters_fitted": 0,
            "scorer": "cos(goal, d_object) + cos(goal, d_place)",
            "pass_top1": PASS_TOP1,
            "absent_top1": ABSENT_TOP1,
            "chance_top1": 1 / (len(OBJECTS) * len(PLACES)),
            "control": "held rows scored with the next sample's goal",
        },
        "parent_model_digest": digest,
        "held_out": held,
        "train": train,
        "control": control,
        "model_involved": True,
        "capability_claimed": status == "geometry_selects",
        "training_admitted": False,
        "limits": (
            "Zero fitted parameters: the directions come from the encoder's own action "
            "vectors and the goal enters as its encoding. It says whether the existing "
            "geometry already supports six-way choice on this fixture; it does not show the "
            "subject choosing, not natural language, not P2."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in ("scripts/training/probe_geometry_select_10_7h.py",)
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
                "train_top1": train["top1"],
                "control_top1": control["top1"],
                "total": held["total"],
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
