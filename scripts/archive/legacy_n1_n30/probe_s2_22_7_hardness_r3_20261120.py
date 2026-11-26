"""22.7 task hardness calibration (zero check, <= 80 CPU).

r4 showed the 32-unit graph reaches fit accuracy 1.0 on the first-A adapter-v3
task, which leaves no headroom for a seven-arm growth comparison.  This probe
scanms a pre-registered grid of base capacities and step counts and selects the
configuration whose mean fit accuracy lands in [0.60, 0.85] (highest capacity
inside the band, then fewest steps), so growth can be measured with real
headroom.  Zero check/final reads; no capability signing.
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
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
    label_index,
)
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_7_hardness_r3_20261120.json"
CAP_CPU = 120.0
TARGET_LOW = 0.60
TARGET_HIGH = 0.85
GRID = ((4, 4), (5, 4))
STEP_GRID = (100, 200, 300, 400, 600)
SEEDS = (0, 1)
BATCH = 64
LR = 3e-3
WARMUP = 100
TRAIN_WORLDS = 256
FIT_WORLDS = 32


def learning_rate(step: int, steps: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, steps - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def split() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_examples = generate_examples("train", TRAIN_WORLDS)
    fit_examples = generate_examples("fit", FIT_WORLDS)
    x_train = torch.stack([adapter_v3_features(example) for example in train_examples]).float()
    y_train = torch.tensor([label_index(example) for example in train_examples])
    x_fit = torch.stack([adapter_v3_features(example) for example in fit_examples]).float()
    y_fit = torch.tensor([label_index(example) for example in fit_examples])
    return x_train, y_train, x_fit, y_fit


def run_config(
    groups: int,
    units_per_group: int,
    steps: int,
    seed: int,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    x_train, y_train, x_fit, y_fit = data
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=groups,
        units_per_group=units_per_group,
        input_dim=8,
        hidden_dim=8,
        output_dim=4,
        seed=20261120 + seed,
    )
    graph = RouteBGraph(spec)
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(20261120 + seed)
    indices = torch.arange(TRAIN_WORLDS)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)
        batch = indices[torch.randperm(TRAIN_WORLDS, generator=generator)[:BATCH]]
        channels = {
            name: x_train[batch, index * 8 : (index + 1) * 8]
            for index, name in enumerate(
                ("rgb", "audio", "proprio", "question", "goal", "feedback")
            )
        }
        gates = torch.ones(BATCH, graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels, gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, y_train[batch])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        channels = {
            name: x_fit[:, index * 8 : (index + 1) * 8]
            for index, name in enumerate(
                ("rgb", "audio", "proprio", "question", "goal", "feedback")
            )
        }
        gates = torch.ones(FIT_WORLDS, graph.units.n_units)
        readout = graph(channels, gates, graph.initial_state())["readout"]
    return float((readout.argmax(-1) == y_fit).float().mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    data = split()
    results = []
    for groups, units_per_group in GRID:
        for steps in STEP_GRID:
            accuracies = [run_config(groups, units_per_group, steps, seed, data) for seed in SEEDS]
            mean = float(sum(accuracies) / len(accuracies))
            results.append(
                {
                    "groups": groups,
                    "units_per_group": units_per_group,
                    "units": groups * units_per_group,
                    "steps": steps,
                    "accuracies": [round(value, 4) for value in accuracies],
                    "mean": round(mean, 4),
                }
            )
            if time.process_time() - started > CAP_CPU:
                break
        if time.process_time() - started > CAP_CPU:
            break
    admissible = [entry for entry in results if TARGET_LOW <= entry["mean"] <= TARGET_HIGH]
    selection = (
        sorted(
            admissible,
            key=lambda entry: (-entry["units"], entry["steps"]),
        )[0]
        if admissible
        else None
    )
    verdict = "hardness_calibrated" if selection is not None else "hardness_grid_missed_band"
    payload = {
        "schema": "taiji-u2-22-7-hardness-v1",
        "card": "22.7 task hardness calibration (adapter v3, target band 0.60-0.85)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "grid": [
                {
                    "groups": groups,
                    "units_per_group": units_per_group,
                    "units": groups * units_per_group,
                }
                for groups, units_per_group in GRID
            ],
            "step_grid": list(STEP_GRID),
            "seeds": list(SEEDS),
            "batch": BATCH,
            "lr": LR,
            "warmup": WARMUP,
            "target_band": [TARGET_LOW, TARGET_HIGH],
            "selection_rule": "highest units inside band, then fewest steps",
        },
        "results": results,
        "selection": selection,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": sum(entry["steps"] * len(SEEDS) for entry in results),
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "selection": selection, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
