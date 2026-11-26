"""22.8J1: appearance dose card - can the shared subject learn joint24 at a real dose? (0 check).

Same 30-unit RouteB subject as 22.8 S2, but the appearance rounds use 48 steps each
(8 appearance rounds per session x 3 sessions = 24 rounds -> 1,152 steps), which is on
the scale of the 22.x single-card joint24 probes (1,200-1,500 steps).  Gate: fit accuracy
>=0.60 and >= majority+0.10 on 3 paired seeds.  Other segment types are not trained
here (J1 only).  Cap 400 CPU, zero check/final.
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
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_8j1_dose_20261120.json"
CAP_CPU = 400.0
SEEDS = (20261120, 20261121, 20261122)
SESSIONS = 3
APPEARANCE_ROUNDS = 8
STEPS_PER_ROUND = 48
BATCH = 32
LR = 3e-3
WARMUP = 50
OUTPUT = 24
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
TARGET = 0.60


def lr_at(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def build_data() -> dict[str, torch.Tensor]:
    train = generate_examples("train", 256)
    fit = generate_examples("fit", 32)
    return {
        "x": torch.stack([adapter_v3_features(item) for item in train]).float(),
        "y": torch.tensor([joint_label(item) for item in train]),
        "x_fit": torch.stack([adapter_v3_features(item) for item in fit]).float(),
        "y_fit": torch.tensor([joint_label(item) for item in fit]),
    }


def build(seed: int) -> RouteBGraph:
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=OUTPUT, seed=seed
    )
    return RouteBGraph(spec)


def run_seed(bundle: dict, seed: int) -> dict:
    graph = build(seed)
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 3)
    total = SESSIONS * APPEARANCE_ROUNDS * STEPS_PER_ROUND
    step = 0
    for _session in range(SESSIONS):
        for _round in range(APPEARANCE_ROUNDS):
            for _ in range(STEPS_PER_ROUND):
                index = torch.randint(0, bundle["x"].shape[0], (BATCH,), generator=generator)
                gates = torch.ones(BATCH, graph.units.n_units)
                logits = graph(channels(bundle["x"][index]), gates, graph.initial_state())[
                    "readout"
                ]
                loss = loss_fn(logits, bundle["y"][index])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                graph.units.project_parameters_()
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, total)
                step += 1
    with torch.no_grad():
        gates = torch.ones(bundle["x_fit"].shape[0], graph.units.n_units)
        logits = graph(channels(bundle["x_fit"]), gates, graph.initial_state())["readout"]
    accuracy = float((logits.argmax(-1) == bundle["y_fit"]).float().mean())
    return {"seed": seed, "steps": step, "fit_accuracy": accuracy}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    bundle = build_data()
    counts = torch.bincount(bundle["y"], minlength=OUTPUT).float()
    majority = float(counts.max() / counts.sum())
    rows = [run_seed(bundle, seed) for seed in SEEDS]
    accuracies = [row["fit_accuracy"] for row in rows]
    gates = {
        "accuracy": min(accuracies) >= TARGET,
        "above_majority": min(accuracies) >= majority + 0.10,
    }
    verdict = "appearance_dose_verified" if all(gates.values()) else "appearance_dose_absent"
    payload = {
        "schema": "taiji-u2-22-8j1-dose-v1",
        "card": "22.8J1 appearance dose card (48 steps/round x 8 rounds x 3 sessions)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "sessions": SESSIONS,
            "appearance_rounds": APPEARANCE_ROUNDS,
            "steps_per_round": STEPS_PER_ROUND,
            "batch": BATCH,
            "target": TARGET,
            "majority": majority,
        },
        "rows": rows,
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "optimizer_steps": len(SEEDS) * SESSIONS * APPEARANCE_ROUNDS * STEPS_PER_ROUND,
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "rows": rows,
                "majority": majority,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "appearance_dose_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
