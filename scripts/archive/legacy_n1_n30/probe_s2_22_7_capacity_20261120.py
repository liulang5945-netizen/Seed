"""22.7b capacity-headroom probe under the noisy-perception task family (sigma=0.15).

The midband probe (D-20261119-38) found sigma=0.15 puts the 32-unit baseline at
0.75 (band 0.60-0.85).  Before a new growth card can be pre-registered, this probe
must show that more units help on the SAME noisy task: accuracy should rise with
capacity and leave >= 0.05 headroom between 32 and 64 units.  Same noise protocol
as the midband probe (train noise resampled per batch, evaluation noise fixed).

Zero check/final; no capability signing; a failed probe keeps the family usable
only for the recursive-decision card, never for growth claims.
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

OUT = ROOT / "reports/taiji_u2_22_7_capacity_20261120.json"
CAP_CPU = 160.0
SIGMA = 0.15
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
GRID = ((2, 4), (4, 4), (4, 8), (8, 8))  # 8, 16, 32, 64 units
SEEDS = (0, 1)
STEPS = 1000
BATCH = 64
LR = 3e-3
WARMUP = 100
TRAIN_WORLDS = 256
FIT_WORLDS = 32
HEADROOM_MIN = 0.05


def learning_rate(step: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, STEPS - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def split() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_examples = generate_examples("train", TRAIN_WORLDS)
    fit_examples = generate_examples("fit", FIT_WORLDS)
    x_train = torch.stack([adapter_v3_features(example) for example in train_examples]).float()
    y_train = torch.tensor([label_index(example) for example in train_examples])
    x_fit = torch.stack([adapter_v3_features(example) for example in fit_examples]).float()
    y_fit = torch.tensor([label_index(example) for example in fit_examples])
    return x_train, y_train, x_fit, y_fit


def run(groups: int, units_per_group: int, seed: int, data: tuple) -> float:
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
    noise_generator = torch.Generator().manual_seed(303030 + seed)
    indices = torch.arange(TRAIN_WORLDS)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        batch = indices[torch.randperm(TRAIN_WORLDS, generator=generator)[:BATCH]]
        features = x_train[batch] + SIGMA * torch.randn(
            (BATCH, x_train.shape[1]), generator=noise_generator
        )
        channels = {
            name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)
        }
        gates = torch.ones(BATCH, graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels, gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, y_train[batch])
        loss.backward()
        optimizer.step()
    eval_generator = torch.Generator().manual_seed(404040 + seed)
    eval_features = x_fit + SIGMA * torch.randn(
        (FIT_WORLDS, x_fit.shape[1]), generator=eval_generator
    )
    with torch.no_grad():
        channels = {
            name: eval_features[:, index * 8 : (index + 1) * 8]
            for index, name in enumerate(CHANNELS)
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
        accuracies = [run(groups, units_per_group, seed, data) for seed in SEEDS]
        results.append(
            {
                "groups": groups,
                "units_per_group": units_per_group,
                "units": groups * units_per_group,
                "accuracies": [round(value, 4) for value in accuracies],
                "mean": round(float(sum(accuracies) / len(accuracies)), 4),
            }
        )
        if time.process_time() - started > CAP_CPU:
            break
    by_units = {entry["units"]: entry["mean"] for entry in results}
    headroom = (
        by_units.get(64, 0.0) - by_units.get(32, 0.0) if 64 in by_units and 32 in by_units else 0.0
    )
    monotone = all(
        by_units[small] <= by_units[large] + 0.02
        for small, large in zip(sorted(by_units), sorted(by_units)[1:], strict=False)
    )
    verdict = (
        "capacity_headroom_confirmed"
        if headroom >= HEADROOM_MIN and monotone and 64 in by_units
        else "capacity_headroom_absent"
    )
    payload = {
        "schema": "taiji-u2-22-7-capacity-v1",
        "card": "22.7b capacity headroom under sigma=0.15 noisy perception",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "sigma": SIGMA,
            "grid_units": [groups * units_per_group for groups, units_per_group in GRID],
            "seeds": list(SEEDS),
            "steps": STEPS,
            "batch": BATCH,
            "headroom_min": HEADROOM_MIN,
            "monotone_tolerance": 0.02,
        },
        "results": results,
        "headroom_32_64": round(headroom, 4),
        "monotone": bool(monotone),
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": STEPS * len(SEEDS) * len(results),
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
                "results": results,
                "headroom": round(headroom, 4),
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "capacity_headroom_confirmed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
