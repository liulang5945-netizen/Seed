"""22.5.2 S2 behaviour window: U2-A vs U2-B vs GRU at equal FLOPs (0 check).

Task: joint24 (colour x position, 24 classes) with a controlled three-tick evidence
stream — tick 0 carries the query/goal channels, tick 1 carries colour evidence, tick 2
carries the position evidence, so the label needs information from two ticks.  Arms:
U2 Route-A, U2 Route-B, GRU-16 (hidden chosen in S1); paired seeds; fit-face evaluation.
Gate G1: U2-B − GRU >= 0.05 with paired bootstrap 95% lower bound > 0.  Cap 3,000 CPU.
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

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.flops_22_5_2_20261120 import GruArm, run_route_a  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from taiji.u2_neural_units import RouteAGraph, RouteBGraph, UnitGraphSpec  # noqa: E402
from taiji.u2_scan import UnitScan  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_5_2_window_20261120.json"
CAP_CPU = 3000.0
SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124)
U2_STEPS = 1200
GRU_STEPS = 841
BATCH = 48
LR = 3e-3
WARMUP = 100
OUTPUT = 24
TICKS = 3
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
DELTA_MIN = 0.05
BOOTSTRAP = 2000


def lr_at(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def evidence_stream(example: dict) -> torch.Tensor:
    """Three-tick evidence arrival: query/goal, then colour, then position."""

    base = adapter_v3_features(example).float()
    query_part = torch.zeros(48)
    query_part[27:33] = base[27:33]  # entity stats + query
    query_part[36] = base[36]  # count
    colour_part = torch.zeros(48)
    colour_part[0:24] = base[0:24]
    colour_part[27:30] = base[27:30]
    position_part = torch.zeros(48)
    position_part[24:27] = base[24:27]
    position_part[33:36] = base[33:36]
    return torch.stack([query_part, colour_part, position_part])


def build_data() -> dict[str, torch.Tensor]:
    train_examples = generate_examples("train", 256)
    fit_examples = generate_examples("fit", 32)
    x_train = torch.stack([evidence_stream(example) for example in train_examples])
    y_train = torch.tensor([joint_label(example) for example in train_examples])
    x_fit = torch.stack([evidence_stream(example) for example in fit_examples])
    y_fit = torch.tensor([joint_label(example) for example in fit_examples])
    return {"x_train": x_train, "y_train": y_train, "x_fit": x_fit, "y_fit": y_fit}


def spec() -> UnitGraphSpec:
    return UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=OUTPUT, seed=20261120
    )


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[..., index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def train_u2(route: str, data: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    graph = RouteAGraph(spec()) if route == "A" else RouteBGraph(spec())
    scan = UnitScan(graph)
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 5)
    gates = torch.ones(BATCH, TICKS, graph.units.n_units)
    started = time.process_time()
    for step in range(U2_STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, U2_STEPS)
        index = torch.randint(0, data["x_train"].shape[0], (BATCH,), generator=generator)
        features = data["x_train"][index]
        if route == "A":
            logits = run_route_a(graph, features, gates)[:, -1]
        else:
            logits = scan.run(channels(features), gates)["readout"][:, -1]
        loss = loss_fn(logits, data["y_train"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()
    fit_gates = torch.ones(data["x_fit"].shape[0], TICKS, graph.units.n_units)
    with torch.no_grad():
        if route == "A":
            fit_logits = run_route_a(graph, data["x_fit"], fit_gates)[:, -1]
        else:
            fit_logits = scan.run(channels(data["x_fit"]), fit_gates)["readout"][:, -1]
    accuracy = float((fit_logits.argmax(-1) == data["y_fit"]).float().mean())
    return {"accuracy": accuracy, "cpu": time.process_time() - started}


def train_gru(data: dict, seed: int) -> dict:
    torch.manual_seed(seed + 11)
    model = GruArm(16)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 5)
    started = time.process_time()
    for step in range(GRU_STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, GRU_STEPS)
        index = torch.randint(0, data["x_train"].shape[0], (BATCH,), generator=generator)
        logits = model(data["x_train"][index])
        loss = loss_fn(logits, data["y_train"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        fit_logits = model(data["x_fit"])
    accuracy = float((fit_logits.argmax(-1) == data["y_fit"]).float().mean())
    return {"accuracy": accuracy, "cpu": time.process_time() - started}


def bootstrap(deltas: np.ndarray) -> tuple[float, float, float]:
    generator = np.random.default_rng(20261120)
    means = np.array(
        [deltas[generator.integers(0, len(deltas), len(deltas))].mean() for _ in range(BOOTSTRAP)]
    )
    return (
        float(means.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    data = build_data()
    labels = data["y_train"]
    counts = torch.bincount(labels, minlength=OUTPUT).float()
    majority = float(counts.max() / counts.sum())
    rows = []
    for seed in SEEDS:
        a = train_u2("A", data, seed)
        b = train_u2("B", data, seed)
        g = train_gru(data, seed)
        rows.append(
            {
                "seed": seed,
                "u2_a": round(a["accuracy"], 4),
                "u2_b": round(b["accuracy"], 4),
                "gru": round(g["accuracy"], 4),
                "delta_b_minus_gru": round(b["accuracy"] - g["accuracy"], 4),
                "cpu": round(a["cpu"] + b["cpu"] + g["cpu"], 2),
            }
        )
    deltas = np.array([row["delta_b_minus_gru"] for row in rows])
    delta_mean, delta_lb, delta_ub = bootstrap(deltas)
    gates = {
        "g1_advantage": delta_mean >= DELTA_MIN and delta_lb > 0.0,
        "g3_engineering": True,
        "g4_cost": time.process_time() - started <= CAP_CPU,
    }
    if gates["g1_advantage"] and gates["g4_cost"]:
        verdict = "arch_advantage_verified"
    elif delta_ub < 0.0:
        verdict = "arch_negative"
    elif gates["g3_engineering"] and gates["g4_cost"]:
        verdict = "arch_inconclusive"
    elif not gates["g4_cost"]:
        verdict = "budget_stop"
    else:
        verdict = "arch_negative"
    payload = {
        "schema": "taiji-u2-22-5-2-window-v1",
        "card": "22.5.2 S2 equal-FLOPs behaviour window: U2-A / U2-B / GRU-16 on three-tick joint24",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "u2_steps": U2_STEPS,
            "gru_steps": GRU_STEPS,
            "batch": BATCH,
            "ticks": TICKS,
            "evidence_stream": "t0 query/goal, t1 colour, t2 position",
            "majority_baseline": round(majority, 4),
            "gate": "mean delta (U2-B - GRU) >= 0.05 and paired bootstrap 95% lower bound > 0",
            "bootstrap": BOOTSTRAP,
        },
        "rows": rows,
        "delta_mean": round(delta_mean, 4),
        "delta_lb95": round(delta_lb, 4),
        "delta_ub95": round(delta_ub, 4),
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "optimizer_steps": len(SEEDS) * (U2_STEPS * 2 + GRU_STEPS),
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "rows": rows,
                "delta_mean": payload["delta_mean"],
                "delta_lb95": payload["delta_lb95"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "arch_advantage_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
