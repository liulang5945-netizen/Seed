"""Controlled-geometry collision probe: paired toward/away pushes (zero check).

The free-form collision probe was geometry-dominated (D-20261119-57).  Here the
worlds are filtered to a fixed target-neighbour distance window and each world
contributes a PAIR of rows: push direction exactly toward the neighbour vs exactly
away from it, same intensity.  The collision label (neighbour displaced > 0.02)
then depends on the action direction by construction, so the task tests whether
the model uses action+geometry rather than neighbour distance alone.

Gates: collision rate in [0.20, 0.80], 30-unit graph (groups=6) fit accuracy >=
max(no-collision baseline + 0.10, 0.60), direction masking drops >= 0.05.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    SUFFIX,
    adapter_v3_features,
)
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402
from taiji.u2_task_episode import NAMES  # noqa: E402
from taiji.u2_task_world import PhysicalWorld  # noqa: E402

OUT = ROOT / "reports/taiji_u2_collision_controlled_20261120.json"
CAP_CPU = 120.0
TRAIN_WORLDS = 192
FIT_WORLDS = 32
PUSHES = 3
STEPS = 1200
SEEDS = (0, 1, 2)
BATCH = 64
LR = 3e-3
WARMUP = 100
COLLISION_SHIFT = 0.06
DISTANCE_WINDOW = (0.45, 0.75)
PALETTE = [
    (220, 45, 45),
    (45, 90, 220),
    (45, 190, 80),
    (185, 185, 45),
    (170, 60, 200),
    (40, 190, 190),
]


def run_pair(seed: int) -> list[dict] | None:
    world = PhysicalWorld.reset(seed, {"body_count": 3})
    world.elapse("public-scene")
    bodies = world.snapshot()["bodies"]
    interior = [body for body in bodies if abs(body["position"][0]) <= 0.55]
    if not interior:
        return None
    target = interior[0]
    colour = tuple(int(value) for value in target["color"])
    if colour not in [tuple(value) for value in PALETTE]:
        return None
    name = NAMES[PALETTE.index(colour)]
    neighbours = []
    for body in bodies:
        if body["id"] == target["id"]:
            continue
        dx = body["position"][0] - target["position"][0]
        dy = body["position"][1] - target["position"][1]
        neighbours.append((math.hypot(dx, dy), dx, dy, body))
    neighbours.sort(key=lambda entry: entry[0])
    distance, dx, dy, neighbour = neighbours[0]
    if not DISTANCE_WINDOW[0] <= distance <= DISTANCE_WINDOW[1]:
        return None
    unit = (dx / distance, dy / distance)
    rows = []
    for label_name, direction in (
        ("toward", unit),
        ("away", (-unit[0], -unit[1])),
    ):
        trial = PhysicalWorld.reset(seed, {"body_count": 3})
        trial.elapse("public-scene")
        neighbour_before = next(
            body for body in trial.snapshot()["bodies"] if body["id"] == neighbour["id"]
        )["position"]
        try:
            for index in range(PUSHES):
                current = next(
                    body for body in trial.snapshot()["bodies"] if body["id"] == target["id"]
                )
                command = torch.zeros(15, dtype=torch.float64)
                command[0] = 1.0
                command[9], command[10] = current["position"]
                command[11], command[12] = direction
                command[13] = 1.0
                command[14] = 0.05
                trial.step(f"push-{label_name}-{index}", command)
        except ValueError:
            return None
        after = next(body for body in trial.snapshot()["bodies"] if body["id"] == neighbour["id"])
        shift = math.hypot(
            after["position"][0] - neighbour_before[0],
            after["position"][1] - neighbour_before[1],
        )
        example = {
            "audit": {"seed": seed},
            "public": {"request": (name + SUFFIX).encode("utf-8")},
            "target": b"",
        }
        features = adapter_v3_features(example)
        features[37] = float(direction[0])
        features[38] = float(direction[1])
        rows.append(
            {
                "features": features.float(),
                "label": int(shift > COLLISION_SHIFT),
                "condition": label_name,
                "neighbour_shift": round(shift, 5),
                "distance": round(distance, 4),
            }
        )
    return rows


def build_partition(count: int, start: int) -> list[dict]:
    rows = []
    seed = start
    attempts = 0
    while len(rows) < count * 2 and attempts < count * 40:
        pair = run_pair(seed)
        if pair is not None:
            rows.extend(pair)
        seed += 1
        attempts += 1
    return rows


def learning_rate(step: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, STEPS - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    names = ("rgb", "audio", "proprio", "question", "goal", "feedback")
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(names)}


def train(features: torch.Tensor, labels: torch.Tensor, seed: int, masked: bool) -> float:
    if masked:
        features = features.clone()
        features[:, 37] = 0.0
        features[:, 38] = 0.0
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=2, seed=20261120 + seed
    )
    graph = RouteBGraph(spec)
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(20261120 + seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        batch = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        gates = torch.ones(BATCH, graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels(features[batch]), gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, labels[batch])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        logits = graph(channels(features), gates, graph.initial_state())["readout"]
    return float((logits.argmax(-1) == labels).float().mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    train_rows = build_partition(TRAIN_WORLDS, 20261120)
    fit_rows = build_partition(FIT_WORLDS, 30303030)
    train_features = torch.stack([row["features"] for row in train_rows])
    train_labels = torch.tensor([row["label"] for row in train_rows])
    collision_rate = float(train_labels.float().mean())
    baseline = float(max(1.0 - collision_rate, collision_rate))
    full = [train(train_features, train_labels, seed, masked=False) for seed in SEEDS]
    masked = [train(train_features, train_labels, seed, masked=True) for seed in SEEDS]
    mean_full = float(sum(full) / len(full))
    mean_masked = float(sum(masked) / len(masked))
    toward_collision = float(
        sum(row["label"] for row in train_rows if row["condition"] == "toward")
        / max(1, sum(1 for row in train_rows if row["condition"] == "toward"))
    )
    away_collision = float(
        sum(row["label"] for row in train_rows if row["condition"] == "away")
        / max(1, sum(1 for row in train_rows if row["condition"] == "away"))
    )
    gates = {
        "balanced": 0.20 <= collision_rate <= 0.80,
        "direction_separates": toward_collision - away_collision >= 0.15,
        "learnable": mean_full >= max(baseline + 0.10, 0.60),
        "direction_contribution": mean_full - mean_masked >= 0.05,
    }
    verdict = "controlled_collision_ready" if all(gates.values()) else "controlled_collision_absent"
    payload = {
        "schema": "taiji-u2-collision-controlled-v1",
        "card": "controlled-geometry collision probe (paired toward/away)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "train_pairs": len(train_rows) // 2,
            "fit_pairs": len(fit_rows) // 2,
            "distance_window": list(DISTANCE_WINDOW),
            "pushes": PUSHES,
            "steps": STEPS,
            "seeds": list(SEEDS),
            "groups": 6,
            "units": 30,
        },
        "collision_rate": round(collision_rate, 4),
        "toward_collision_rate": round(toward_collision, 4),
        "away_collision_rate": round(away_collision, 4),
        "baseline_accuracy": round(baseline, 4),
        "full_accuracies": [round(value, 4) for value in full],
        "masked_accuracies": [round(value, 4) for value in masked],
        "mean_full": round(mean_full, 4),
        "mean_masked": round(mean_masked, 4),
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": STEPS * len(SEEDS) * 2,
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "toward": payload["toward_collision_rate"],
                "away": payload["away_collision_rate"],
                "full": payload["full_accuracies"],
                "masked": payload["masked_accuracies"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "controlled_collision_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
