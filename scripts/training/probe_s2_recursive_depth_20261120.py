"""Recursive-depth probe: iterative inference within persistent unit state (0 check).

Pretrains a 30-unit joint24 graph (single tick), then trains a recurrent sequence
consisting of each sample repeated T=4 times (state carried), evaluating per-tick
accuracy.  Conditions (D-20261119-101): depth gain acc(T=4)-acc(T=1) >= 0.05;
shuffled-input control gain <= 0.02; later-tick credit (tick-4 loss <= tick-1 loss).
Baseline: same weights trained T=1 for the same extra steps.  3 seeds; cap 300 CPU.
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
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (
    adapter_v3_features,  # noqa: E402
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402

OUT = ROOT / "reports/taiji_u2_recursive_depth_20261120.json"
CAP_CPU = 300.0
TRAIN_WORLDS = 256
FIT_WORLDS = 32
BASE_STEPS = 1200
REC_STEPS = 600
BATCH = 32
TICKS = 4
LR = 3e-3
WARMUP = 100
SEEDS = (0, 1, 2)
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
DEPTH_GAIN_MIN = 0.05
SHUFFLED_GAIN_MAX = 0.02


def learning_rate(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def build_graph(seed: int) -> RouteBGraph:
    torch.manual_seed(seed)
    return RouteBGraph(
        UnitGraphSpec(
            groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=24, seed=seed
        )
    )


def train_plain(graph, features, labels, steps, seed):
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    graph.zero_grad(set_to_none=True)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)
        index = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        gates = torch.ones(BATCH, graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        logits = graph(channels(features[index]), gates, graph.initial_state())["readout"]
        loss = loss_fn(logits, labels[index])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def sequence(
    features: torch.Tensor, labels: torch.Tensor, index: torch.Tensor, ticks: int, shuffled: bool
):
    samples = features[index]
    labels_batch = labels[index]
    if shuffled:
        rolls = torch.roll(samples, shifts=1, dims=0)
        parts = [samples] + [rolls] * (ticks - 1)
    else:
        parts = [samples] * ticks
    sequence_features = torch.cat(parts, dim=0)
    sequence_labels = labels_batch.repeat(ticks)
    group = torch.arange(index.numel()).repeat(ticks)
    return sequence_features, sequence_labels, group


def per_tick_accuracy(
    graph, features, labels, ticks: int, shuffled: bool = False, samples: int = 16
) -> list[float]:
    index = torch.arange(samples)
    sequence_features, sequence_labels, _ = sequence(features, labels, index, ticks, shuffled)
    with torch.no_grad():
        gates = torch.ones(sequence_features.shape[0], graph.units.n_units)
        logits = graph(channels(sequence_features), gates, graph.initial_state())["readout"]
    predictions = logits.argmax(-1)
    return [
        float(
            (
                predictions[tick * samples : (tick + 1) * samples]
                == sequence_labels[tick * samples : (tick + 1) * samples]
            )
            .float()
            .mean()
        )
        for tick in range(ticks)
    ]


def train_recursive(graph, features, labels, steps, seed) -> tuple[float, float]:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    first_loss = 0.0
    last_loss = 0.0
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)
        index = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        sequence_features, sequence_labels, _ = sequence(features, labels, index, TICKS, False)
        gates = torch.ones(sequence_features.shape[0], graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        logits = graph(channels(sequence_features), gates, graph.initial_state())["readout"]
        loss = loss_fn(logits, sequence_labels)
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()
        if step == 0:
            first_loss = float(loss_fn(logits, sequence_labels))
        if step == steps - 1:
            last_loss = float(loss_fn(logits, sequence_labels))
    return first_loss, last_loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    train_examples = generate_examples("train", TRAIN_WORLDS)
    fit_examples = generate_examples("fit", FIT_WORLDS)
    train_features = torch.stack([adapter_v3_features(ex) for ex in train_examples]).float()
    train_labels = torch.tensor([joint_label(ex) for ex in train_examples])
    fit_features = torch.stack([adapter_v3_features(ex) for ex in fit_examples]).float()
    fit_labels = torch.tensor([joint_label(ex) for ex in fit_examples])
    results = []
    for seed in SEEDS:
        graph = build_graph(20261120 + seed)
        train_plain(graph, train_features, train_labels, BASE_STEPS, seed)
        pretrained_tick1 = per_tick_accuracy(graph, fit_features, fit_labels, 1)[0]
        first_loss, last_loss = train_recursive(
            graph, train_features, train_labels, REC_STEPS, seed + 5
        )
        ticks = per_tick_accuracy(graph, fit_features, fit_labels, TICKS)
        shuffled_ticks = per_tick_accuracy(graph, fit_features, fit_labels, TICKS, shuffled=True)
        baseline = build_graph(20261120 + seed + 1000)
        train_plain(baseline, train_features, train_labels, BASE_STEPS + REC_STEPS, seed + 6)
        baseline_tick1 = per_tick_accuracy(baseline, fit_features, fit_labels, 1)[0]
        results.append(
            {
                "seed": seed,
                "pretrained_tick1": round(pretrained_tick1, 4),
                "tick_accuracies": [round(value, 4) for value in ticks],
                "depth_gain": round(ticks[-1] - ticks[0], 6),
                "shuffled_gain": round(shuffled_ticks[-1] - shuffled_ticks[0], 6),
                "credit_first_loss": round(first_loss, 4),
                "credit_last_loss": round(last_loss, 4),
                "baseline_tick1": round(baseline_tick1, 4),
                "baseline_gain_vs_pretrained": round(baseline_tick1 - pretrained_tick1, 4),
            }
        )
    gains = [result["depth_gain"] for result in results]
    shuffled_gains = [result["shuffled_gain"] for result in results]
    gates = {
        "depth_gain": all(gain >= DEPTH_GAIN_MIN for gain in gains),
        "feedback_dependence": all(gain <= SHUFFLED_GAIN_MAX for gain in shuffled_gains),
        "credit_alive": all(
            result["credit_last_loss"] <= result["credit_first_loss"] + 1e-9 for result in results
        ),
        "baseline_control": all(
            result["baseline_gain_vs_pretrained"] <= 0.02 for result in results
        ),
    }
    if all(gates.values()):
        verdict = "recursive_depth_ready"
    elif gates["depth_gain"] and not (gates["feedback_dependence"] and gates["credit_alive"]):
        verdict = "recursive_depth_partial"
    else:
        verdict = "recursive_depth_absent"
    payload = {
        "schema": "taiji-u2-recursive-depth-v1",
        "card": "iterative inference depth within persistent unit state (T<=4)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "base_steps": BASE_STEPS,
            "recursive_steps": REC_STEPS,
            "depth_gain_min": DEPTH_GAIN_MIN,
            "shuffled_gain_max": SHUFFLED_GAIN_MAX,
            "baseline": "T=1 trained for the same extra steps",
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": (BASE_STEPS + REC_STEPS + BASE_STEPS + REC_STEPS) * len(SEEDS),
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "results": results, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "recursive_depth_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
