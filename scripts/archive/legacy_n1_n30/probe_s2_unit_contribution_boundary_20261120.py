"""Unit causal-contribution probe: newborn units' joint task contribution (zero check).

Trains a 30-unit joint24 graph, births 6 live-entry units at step 1,000 (two
batches of 3), continues training 1,000 more steps, then measures fit-accuracy
drops from (a) ablating the 6 newborns, (b) ablating 3 random mature units x 5
draws.  Causal contribution requires the newborn ablation drop >= 0.02 and larger
than the random-unit control by >= 0.01, on all 3 seeds, with a paired one-sided
95% lower bound > 0.

This probes "协同由实际因果影响和联合任务贡献验证" for a growth event; zero
check; cap 200 CPU.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (
    adapter_v3_features,  # noqa: E402
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from taiji.u2_neural_birth_multi import live_entry_birth_multi  # noqa: E402
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402

OUT = ROOT / "reports/taiji_u2_unit_contribution_boundary_20261120.json"
CAP_CPU = 200.0
TRAIN_WORLDS = 256
FIT_WORLDS = 32
PRE_STEPS = 750
POST_STEPS = 750
BATCH = 64
LR = 3e-3
WARMUP = 100
SEEDS = (0, 1, 2)
BIRTH_COUNT = 3
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")


def learning_rate(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def train(
    graph: RouteBGraph, features: torch.Tensor, labels: torch.Tensor, seed: int, steps: int
) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)
        batch = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        gates = torch.ones(BATCH, graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels(features[batch]), gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, labels[batch])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def correctness(graph: RouteBGraph, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        logits = graph(channels(features), gates, graph.initial_state())["readout"]
    return logits.argmax(-1) == labels


def ablate(graph: RouteBGraph, unit_ids: list[str]) -> None:
    for unit_id in unit_ids:
        graph.units.set_lifecycle(unit_id, "dormant")


def restore(graph: RouteBGraph, unit_ids: list[str]) -> None:
    for unit_id in unit_ids:
        graph.units.set_lifecycle(unit_id, "mature")


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
        torch.manual_seed(20261120 + seed)
        graph = RouteBGraph(
            UnitGraphSpec(
                groups=6,
                units_per_group=2,
                input_dim=8,
                hidden_dim=8,
                output_dim=24,
                seed=20261120 + seed,
            )
        )
        train(graph, train_features, train_labels, seed, PRE_STEPS)
        parents = [unit["unit_id"] for unit in graph.units.identities["units"][:3]]
        graph.units, _ = live_entry_birth_multi(graph.units, 0, parents, count=3)
        newborns = [unit["unit_id"] for unit in graph.units.identities["units"][-BIRTH_COUNT:]]
        for unit_id in newborns:
            graph.units.set_lifecycle(unit_id, "mature")
        train(graph, train_features, train_labels, seed + 100, POST_STEPS)
        base = correctness(graph, fit_features, fit_labels)
        base_accuracy = float(base.float().mean())

        ablate(graph, newborns)
        ablated_newborns = correctness(graph, fit_features, fit_labels)
        restore(graph, newborns)

        generator = torch.Generator().manual_seed(20261120 + seed + 7)
        old_units = [unit["unit_id"] for unit in graph.units.identities["units"][:30]]
        drops = []
        old_correct = []
        for _ in range(5):
            indices = torch.randperm(len(old_units), generator=generator)[:3].tolist()
            sample = [old_units[index] for index in indices]
            ablate(graph, sample)
            correct = correctness(graph, fit_features, fit_labels)
            restore(graph, sample)
            drops.append(base_accuracy - float(correct.float().mean()))
            old_correct.append(correct)
        old_drop = float(np.mean(drops))
        newborn_drop = base_accuracy - float(ablated_newborns.float().mean())
        paired = (base.to(torch.int32) - ablated_newborns.to(torch.int32)) - (
            base.to(torch.int32) - old_correct[0].to(torch.int32)
        )
        results.append(
            {
                "seed": seed,
                "base_accuracy": round(base_accuracy, 4),
                "newborn_ablation_drop": round(newborn_drop, 4),
                "old_random_drop_mean": round(old_drop, 4),
                "old_random_drops": [round(value, 4) for value in drops],
                "paired_correctness": paired.tolist(),
            }
        )
    gains = [result["newborn_ablation_drop"] for result in results]
    old_means = [result["old_random_drop_mean"] for result in results]
    paired_all = np.concatenate(
        [np.asarray(result["paired_correctness"], dtype=float) for result in results]
    )
    rng = np.random.default_rng(20261120)
    draws = rng.choice(paired_all, size=(5000, paired_all.size), replace=True).mean(axis=1)
    lower_bound = float(np.quantile(draws, 0.05))
    gates = {
        "newborn_drop_0_02": all(value >= 0.02 for value in gains),
        "beats_random_control": all(
            gains[index] >= old_means[index] + 0.01 for index in range(len(gains))
        ),
        "paired_lower_bound": lower_bound > 0.0,
    }
    verdict = "unit_contribution_pass" if all(gates.values()) else "unit_contribution_absent"
    payload = {
        "schema": "taiji-u2-unit-contribution-v1",
        "card": "newborn-unit causal contribution at the capacity boundary (12 units)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "pre_steps": PRE_STEPS,
            "post_steps": POST_STEPS,
            "birth_count": BIRTH_COUNT,
            "base_units": 12,
            "random_control_draws": 5,
            "seeds": list(SEEDS),
            "gates": "newborn drop >= 0.02, >= random + 0.01, paired LB > 0",
        },
        "results": results,
        "lower_bound": round(lower_bound, 4),
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": (PRE_STEPS + POST_STEPS) * len(SEEDS),
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
                "results": results,
                "lb": round(lower_bound, 4),
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "unit_contribution_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
