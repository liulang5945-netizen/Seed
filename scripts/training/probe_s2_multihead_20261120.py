"""Multi-head isolation feasibility: keep old ability exactly, learn a new one (0 check).

Option ① from D-20261119-73: task-specific readout heads on top of shared frozen
units.  Base graph trains joint24 with head A; then head A + units are frozen and a
fresh head B learns the action-consequence bins from the fixed unit features.  The
probe checks (a) old ability is retained bitwise (head A/unit params frozen), and
(b) the new task is learnable through head B alone.

Gates: old accuracy identical to pre-switch (drop == 0); new accuracy >= chance+0.1;
zero check; cap 120 CPU.
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
from scripts.training.probe_s2_action_consequence_20261120 import build_partition  # noqa: E402
from taiji.u2_multihead import MultiHeadGraph, freeze_module  # noqa: E402
from taiji.u2_neural_units import UnitGraphSpec  # noqa: E402

OUT = ROOT / "reports/taiji_u2_multihead_20261120.json"
CAP_CPU = 120.0
OLD_WORLDS = 256
OLD_FIT = 32
NEW_WORLDS = 256
NEW_FIT = 32
BASE_STEPS = 1200
HEAD_STEPS = 600
BATCH = 64
LR = 3e-3
WARMUP = 100
SEEDS = (0, 1, 2)
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")


def learning_rate(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def spec(seed: int) -> UnitGraphSpec:
    return UnitGraphSpec(
        groups=6,
        units_per_group=5,
        input_dim=8,
        hidden_dim=8,
        output_dim=24,
        seed=20261120 + seed,
    )


def train_base(graph, features, labels, seed: int) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(BASE_STEPS):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, BASE_STEPS)
        batch = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        gates = torch.ones(BATCH, graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels(features[batch]), gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, labels[batch])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def accuracy(graph, features, labels, head: int, classes: int | None = None) -> float:
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        logits = graph(channels(features), gates, graph.initial_state(), head=head)["readout"]
    if classes is not None:
        logits = logits[:, :classes]
    return float((logits.argmax(-1) == labels).float().mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    old_examples = generate_examples("train", OLD_WORLDS)
    old_fit_examples = generate_examples("fit", OLD_FIT)
    old_features = torch.stack([adapter_v3_features(ex) for ex in old_examples]).float()
    old_labels = torch.tensor([joint_label(ex) for ex in old_examples])
    old_fit_features = torch.stack([adapter_v3_features(ex) for ex in old_fit_examples]).float()
    old_fit_labels = torch.tensor([joint_label(ex) for ex in old_fit_examples])
    rows = build_partition("train", NEW_WORLDS + NEW_FIT, 20261120)
    new_features = torch.stack([row["features"] for row in rows[:NEW_WORLDS]])
    new_labels = torch.tensor([row["final_bin"] for row in rows[:NEW_WORLDS]])
    new_fit_features = torch.stack([row["features"] for row in rows[NEW_WORLDS:]])
    new_fit_labels = torch.tensor([row["final_bin"] for row in rows[NEW_WORLDS:]])

    results = []
    for seed in SEEDS:
        torch.manual_seed(20261120 + seed)
        graph = MultiHeadGraph(spec(seed), heads=2)
        train_base(graph, old_features, old_labels, seed)
        old_before = accuracy(graph, old_fit_features, old_fit_labels, head=0)
        freeze_module(graph.units)
        freeze_module(graph.heads[0])
        head_optimizer = torch.optim.Adam(graph.heads[1].parameters(), lr=1e-2)
        loss_fn = torch.nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(20261120 + seed + 9)
        for _ in range(HEAD_STEPS):
            batch = torch.randint(0, new_features.shape[0], (BATCH,), generator=generator)
            gates = torch.ones(BATCH, graph.units.n_units)
            head_optimizer.zero_grad(set_to_none=True)
            logits = graph(channels(new_features[batch]), gates, graph.initial_state(), head=1)[
                "readout"
            ]
            loss = loss_fn(logits[:, :3], new_labels[batch])
            loss.backward()
            head_optimizer.step()
        old_after = accuracy(graph, old_fit_features, old_fit_labels, head=0)
        new_accuracy = accuracy(graph, new_fit_features, new_fit_labels, head=1, classes=3)
        results.append(
            {
                "seed": seed,
                "old_before": round(old_before, 4),
                "old_after": round(old_after, 4),
                "retention_drop": round(old_before - old_after, 6),
                "new_accuracy": round(new_accuracy, 4),
            }
        )
    gates = {
        "old_exact": all(result["retention_drop"] <= 1e-9 for result in results),
        "new_learnable": all(result["new_accuracy"] >= 1 / 3 + 0.1 for result in results),
        "new_task_nonzero": all(result["new_accuracy"] > 0.0 for result in results),
    }
    verdict = "multihead_isolation_ready" if all(gates.values()) else "multihead_isolation_absent"
    payload = {
        "schema": "taiji-u2-multihead-v1",
        "card": "task-specific readout heads with frozen units (option 1 feasibility)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "base_steps": BASE_STEPS,
            "head_steps": HEAD_STEPS,
            "seeds": list(SEEDS),
            "note": "units and head A frozen; only head B trained on the new task",
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": (BASE_STEPS + HEAD_STEPS) * len(SEEDS),
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
    return 0 if verdict == "multihead_isolation_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
