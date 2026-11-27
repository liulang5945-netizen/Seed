"""Action-consequence task feasibility (N=3 pushes, position-bin label).

For each world: pick an interior body (|x| <= 0.55), read its colour and current
position bin, then apply three identical pushes (direction +x or -x, intensity 1,
pulse 0.05) and read the final position bin as the label.  Features are the
adapter-v3 perception summary plus the action direction/intensity packed into the
free dims; no labels or oracle positions are used as features.

Feasibility gates: label non-degenerate (each of 3 bins >= 15%), baseline
"predict current bin" accuracy < 0.6, bin-change fraction >= 0.2, and a 32-unit
graph trained for 500 steps reaches fit accuracy >= max(chance+0.1, baseline).
Zero check; capability signing waits for the formal action-fragment card.
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

OUT = ROOT / "reports/taiji_u2_action_consequence_20261120.json"
CAP_CPU = 80.0
TRAIN_WORLDS = 256
FIT_WORLDS = 32
PUSHES = 3
STEPS = 500
SEEDS = (0, 1, 2)
BATCH = 64
LR = 3e-3
WARMUP = 100


def bin_of(x: float) -> int:
    return 0 if x < -1.0 / 3.0 else (2 if x > 1.0 / 3.0 else 1)


def build_row(seed: int) -> dict | None:
    world = PhysicalWorld.reset(seed, {"body_count": 3})
    world.elapse("public-scene")
    bodies = world.snapshot()["bodies"]
    interior = [body for body in bodies if abs(body["position"][0]) <= 0.55]
    if not interior:
        return None
    target = interior[0]
    colour = tuple(int(value) for value in target["color"])
    if colour not in [
        tuple(int(v) for v in c)
        for c in [
            (220, 45, 45),
            (45, 90, 220),
            (45, 190, 80),
            (185, 185, 45),
            (170, 60, 200),
            (40, 190, 190),
        ]
    ]:
        return None
    name = NAMES[
        [
            tuple(int(v) for v in c)
            for c in [
                (220, 45, 45),
                (45, 90, 220),
                (45, 190, 80),
                (185, 185, 45),
                (170, 60, 200),
                (40, 190, 190),
            ]
        ].index(colour)
    ]
    position = target["position"]
    current_bin = bin_of(float(position[0]))
    direction = 1.0 if seed % 2 == 0 else -1.0
    try:
        for index in range(PUSHES):
            bodies = world.snapshot()["bodies"]
            target_now = next(body for body in bodies if body["id"] == target["id"])
            command = torch.zeros(15, dtype=torch.float64)
            command[0] = 1.0
            command[9], command[10] = target_now["position"]
            command[11], command[12] = direction, 0.0
            command[13] = 0.8
            command[14] = 0.05
            world.step(f"push-{index}", command)
    except ValueError:
        return None
    final = next(body for body in world.snapshot()["bodies"] if body["id"] == target["id"])
    final_bin = bin_of(float(final["position"][0]))
    example = {
        "audit": {"seed": seed},
        "public": {"request": (name + SUFFIX).encode("utf-8")},
        "target": b"",
    }
    features = adapter_v3_features(example)
    action = torch.zeros(48, dtype=torch.float64)
    action[37] = direction
    action[38] = 0.55 + float(current_bin) / 10.0
    features = torch.cat([features[:37], action[37:39], features[39:]])
    return {
        "features": features.float(),
        "current_bin": current_bin,
        "final_bin": final_bin,
        "changed": int(current_bin != final_bin),
        "direction": direction,
        "entity": name,
    }


def build_partition(partition: str, count: int, start: int) -> list[dict]:
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
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / max(1, STEPS - WARMUP)))


def train(data: dict, seed: int) -> float:
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=4, units_per_group=8, input_dim=8, hidden_dim=8, output_dim=3, seed=20261120 + seed
    )
    graph = RouteBGraph(spec)
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(20261120 + seed)
    features, labels = data["train_features"], data["train_labels"]
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
        gates = torch.ones(data["fit_features"].shape[0], graph.units.n_units)
        readout = graph(channels(data["fit_features"]), gates, graph.initial_state())["readout"]
    return float((readout.argmax(-1) == data["fit_labels"]).float().mean())


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    names = ("rgb", "audio", "proprio", "question", "goal", "feedback")
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(names)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    train_rows = build_partition("train", TRAIN_WORLDS, 20261120)
    fit_rows = build_partition("fit", FIT_WORLDS, 30303030)
    data = {
        "train_features": torch.stack([row["features"] for row in train_rows]),
        "train_labels": torch.tensor([row["final_bin"] for row in train_rows]),
        "fit_features": torch.stack([row["features"] for row in fit_rows]),
        "fit_labels": torch.tensor([row["final_bin"] for row in fit_rows]),
    }
    train_labels = data["train_labels"].tolist()
    distribution = {
        str(index): round(count / max(1, len(train_labels)), 4)
        for index, count in enumerate([train_labels.count(bin) for bin in (0, 1, 2)])
    }
    correct = [row["changed"] for row in train_rows]
    changed_fraction = round(float(sum(correct)) / max(1, len(correct)), 4)
    baseline = float(
        sum(1 for row in fit_rows if row["current_bin"] == row["final_bin"]) / max(1, len(fit_rows))
    )
    accuracies = [train(data, seed) for seed in SEEDS]
    mean_accuracy = float(sum(accuracies) / len(accuracies))
    gates = {
        "label_balance": min(distribution.values()) >= 0.15,
        "changed_fraction": changed_fraction >= 0.20,
        "baseline_below_0_6": baseline < 0.6,
        "learnable": mean_accuracy >= max(1.0 / 3.0 + 0.1, baseline),
    }
    verdict = "action_consequence_ready" if all(gates.values()) else "action_consequence_absent"
    payload = {
        "schema": "taiji-u2-action-consequence-v1",
        "card": "action-consequence task feasibility (N=3 pushes, position bin)",
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
            "steps": STEPS,
            "seeds": list(SEEDS),
            "output_classes": 3,
            "chance": round(1 / 3, 4),
            "feature_note": "adapter v3 summary + action direction/intensity in free dims",
        },
        "label_distribution": distribution,
        "changed_fraction": changed_fraction,
        "baseline_accuracy": round(baseline, 4),
        "accuracies": [round(value, 4) for value in accuracies],
        "mean_accuracy": round(mean_accuracy, 4),
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": STEPS * len(SEEDS),
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
                "distribution": distribution,
                "changed": changed_fraction,
                "baseline": round(baseline, 4),
                "acc": round(mean_accuracy, 4),
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "action_consequence_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
