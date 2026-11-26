"""Collision-dynamics feasibility probe (simulated, zero check).

Extends the action fragment to collision consequences: push an interior body
towards its nearest neighbour three times and label whether the neighbour is
displaced (> 0.02), i.e. an actual collision transfer.  Features are the adapter
v3 colour/position summary plus the push direction; ``groups=6`` routes all legal
channels (D-20261119-54).  A direction-masked arm checks that the action direction
is what carries the signal.

Feasibility gates: collision rate in [0.20, 0.80], a 30-unit graph reaches fit
accuracy >= max(no-collision baseline + 0.10, 0.60), and direction masking drops
accuracy by >= 0.05.  Zero check; the formal multi-dynamics card comes later.
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

OUT = ROOT / "reports/taiji_u2_collision_20261120.json"
CAP_CPU = 120.0
TRAIN_WORLDS = 256
FIT_WORLDS = 32
PUSHES = 3
STEPS = 1200
SEEDS = (0, 1, 2)
BATCH = 64
LR = 3e-3
WARMUP = 100
COLLISION_SHIFT = 0.02
PALETTE = [
    (220, 45, 45),
    (45, 90, 220),
    (45, 190, 80),
    (185, 185, 45),
    (170, 60, 200),
    (40, 190, 190),
]


def build_row(seed: int) -> dict | None:
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
    direction = None
    neighbour = None
    best = 10.0
    for body in bodies:
        if body["id"] == target["id"]:
            continue
        dx = body["position"][0] - target["position"][0]
        dy = body["position"][1] - target["position"][1]
        distance = math.hypot(dx, dy)
        if distance < best:
            best = distance
            base = (dx / max(distance, 1e-9), dy / max(distance, 1e-9))
            generator = torch.Generator().manual_seed(seed + 7)
            theta = float((torch.rand(1, generator=generator) * 2.0 - 1.0) * math.pi)
            rotated = (
                math.cos(theta) * base[0] - math.sin(theta) * base[1],
                math.sin(theta) * base[0] + math.cos(theta) * base[1],
            )
            direction = rotated
            intensity = 0.3 + 0.7 * float(torch.rand(1, generator=generator))
            neighbour = body
    if direction is None or neighbour is None:
        return None
    neighbour_before = list(neighbour["position"])
    try:
        for index in range(PUSHES):
            current = next(
                body for body in world.snapshot()["bodies"] if body["id"] == target["id"]
            )
            command = torch.zeros(15, dtype=torch.float64)
            command[0] = 1.0
            command[9], command[10] = current["position"]
            command[11], command[12] = direction
            command[13] = intensity
            command[14] = 0.05
            world.step(f"push-{index}", command)
    except ValueError:
        return None
    after = next(body for body in world.snapshot()["bodies"] if body["id"] == neighbour["id"])
    shift = math.hypot(
        after["position"][0] - neighbour_before[0], after["position"][1] - neighbour_before[1]
    )
    label = int(shift > COLLISION_SHIFT)
    example = {
        "audit": {"seed": seed},
        "public": {"request": (name + SUFFIX).encode("utf-8")},
        "target": b"",
    }
    features = adapter_v3_features(example)
    features[37] = float(direction[0])
    features[38] = float(direction[1])
    return {"features": features.float(), "label": label, "neighbour_shift": round(shift, 5)}


def build_partition(count: int, start: int) -> list[dict]:
    rows = []
    seed = start
    attempts = 0
    while len(rows) < count and attempts < count * 20:
        row = build_row(seed)
        if row is not None:
            rows.append(row)
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
    gates = {
        "collision_rate_in_range": 0.20 <= collision_rate <= 0.80,
        "learnable": mean_full >= max(baseline + 0.10, 0.60),
        "direction_contribution": mean_full - mean_masked >= 0.05,
    }
    verdict = "collision_dynamics_ready" if all(gates.values()) else "collision_dynamics_absent"
    payload = {
        "schema": "taiji-u2-collision-v1",
        "card": "collision-dynamics feasibility probe",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "train_worlds": len(train_rows),
            "fit_worlds": len(fit_rows),
            "pushes": PUSHES,
            "collision_shift": COLLISION_SHIFT,
            "steps": STEPS,
            "seeds": list(SEEDS),
            "groups": 6,
            "units": 30,
        },
        "collision_rate": round(collision_rate, 4),
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
                "collision_rate": payload["collision_rate"],
                "full": payload["full_accuracies"],
                "masked": payload["masked_accuracies"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "collision_dynamics_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
