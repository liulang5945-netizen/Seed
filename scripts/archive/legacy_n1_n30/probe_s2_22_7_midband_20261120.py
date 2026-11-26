"""22.7 middle-band task-family feasibility: noisy-perception hardness probe.

The plain first-A adapter-v3 task is bimodal (learn by 200 steps or fail), so the
seven-arm growth comparison and recursive-development card have no headroom.
This probe adds fixed-seed Gaussian noise to the 48-dim adapter features (train
noise resampled per batch, evaluation noise fixed) on a 32-unit graph and scans
sigma to find a configuration whose fit accuracy lands in [0.60, 0.85].

Zero check/final; no capability signing; the result only informs whether a noisy
perception task family can reopen the growth/recursive evidence.
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

OUT = ROOT / "reports/taiji_u2_22_7_midband_20261120.json"
CAP_CPU = 170.0
TARGET_LOW = 0.60
TARGET_HIGH = 0.85
SIGMAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
SEEDS = (0, 1)
STEPS = 1000
BATCH = 64
LR = 3e-3
WARMUP = 100
TRAIN_WORLDS = 256
FIT_WORLDS = 32
GROUPS = 4
UNITS_PER_GROUP = 8


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


def run(sigma: float, seed: int, data: tuple) -> float:
    x_train, y_train, x_fit, y_fit = data
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=GROUPS,
        units_per_group=UNITS_PER_GROUP,
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
    noise_generator = torch.Generator().manual_seed(303030 + seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        batch = indices[torch.randperm(TRAIN_WORLDS, generator=generator)[:BATCH]]
        features = x_train[batch]
        if sigma > 0:
            features = features + sigma * torch.randn(features.shape, generator=noise_generator)
        channels = {
            name: features[:, index * 8 : (index + 1) * 8]
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
    eval_generator = torch.Generator().manual_seed(404040 + seed)
    eval_features = x_fit
    if sigma > 0:
        eval_features = eval_features + sigma * torch.randn(
            eval_features.shape, generator=eval_generator
        )
    with torch.no_grad():
        channels = {
            name: eval_features[:, index * 8 : (index + 1) * 8]
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
    for sigma in SIGMAS:
        accuracies = [run(sigma, seed, data) for seed in SEEDS]
        mean = float(sum(accuracies) / len(accuracies))
        results.append(
            {
                "sigma": sigma,
                "accuracies": [round(value, 4) for value in accuracies],
                "mean": round(mean, 4),
            }
        )
        if time.process_time() - started > CAP_CPU:
            break
    admissible = [entry for entry in results if TARGET_LOW <= entry["mean"] <= TARGET_HIGH]
    selection = sorted(admissible, key=lambda entry: entry["sigma"])[0] if admissible else None
    verdict = "middle_band_found" if selection is not None else "middle_band_absent"
    payload = {
        "schema": "taiji-u2-22-7-midband-v1",
        "card": "22.7 noisy-perception middle-band feasibility",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "base": f"{GROUPS}x{UNITS_PER_GROUP} units, {STEPS} steps, batch {BATCH}, lr {LR}",
            "train_worlds": TRAIN_WORLDS,
            "fit_worlds": FIT_WORLDS,
            "sigmas": list(SIGMAS),
            "seeds": list(SEEDS),
            "noise": "train noise resampled per batch; evaluation noise fixed",
            "target_band": [TARGET_LOW, TARGET_HIGH],
        },
        "results": results,
        "selection": selection,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": sum(1 for _ in results) * STEPS * len(SEEDS),
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
    return 0 if verdict == "middle_band_found" else 1


if __name__ == "__main__":
    raise SystemExit(main())
