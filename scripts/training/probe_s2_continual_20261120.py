"""Cross-session continual learning: new action task while keeping joint24 (zero check).

Session 1 trains joint24 (old ability).  Session 2 continues on the action-consequence
task (position-bin labels over the first three logits) with three arms:

* ``no_replay``: sequential fine-tune on action data only;
* ``replay``: same, but each batch mixes 80% action + 20% joint24 samples;
* ``scratch``: fresh graph trained on action data (new-task reference).

Gates: replay new accuracy >= no_replay - 0.05; replay old-task retention drop
<= 0.02; no_replay new accuracy >= chance + 0.1.  3 seeds, zero check, cap 200.
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
from scripts.training.probe_s2_action_consequence_20261120 import (  # noqa: E402
    build_partition,
)
from taiji.u2_neural_units import (  # noqa: E402
    RouteBGraph,
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    load_capsule,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_continual_20261120.json"
CAP_CPU = 200.0
OLD_WORLDS = 256
OLD_FIT = 32
NEW_WORLDS = 256
NEW_FIT = 32
OLD_STEPS = 1200
NEW_STEPS = 800
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


def make_graph(seed: int) -> RouteBGraph:
    torch.manual_seed(seed)
    return RouteBGraph(
        UnitGraphSpec(
            groups=6,
            units_per_group=5,
            input_dim=8,
            hidden_dim=8,
            output_dim=24,
            seed=20261120 + seed,
        )
    )


def train_mixed(
    graph: RouteBGraph,
    old_features: torch.Tensor,
    old_labels: torch.Tensor,
    new_features: torch.Tensor,
    new_labels: torch.Tensor,
    steps: int,
    seed: int,
    replay: bool,
) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)
        new_batch = torch.randint(0, new_features.shape[0], (BATCH,), generator=generator)
        features = new_features[new_batch]
        labels = new_labels[new_batch]
        if replay:
            old_batch = torch.randint(0, old_features.shape[0], (BATCH // 4,), generator=generator)
            features = torch.cat([features, old_features[old_batch]])
            labels = torch.cat([labels, old_labels[old_batch]])
        gates = torch.ones(features.shape[0], graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        readout = graph(channels(features), gates, graph.initial_state())["readout"]
        loss = loss_fn(readout, labels)
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def accuracy(
    graph: RouteBGraph, features: torch.Tensor, labels: torch.Tensor, classes: int
) -> float:
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        logits = graph(channels(features), gates, graph.initial_state())["readout"]
    if classes == 3:
        logits = logits[:, :3]
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
        base = make_graph(seed)
        train_mixed(
            base, old_features, old_labels, old_features, old_labels, OLD_STEPS, seed, False
        )
        old_before = accuracy(base, old_fit_features, old_fit_labels, 24)
        save_capsule(ROOT / f"output/s2_continual/base_seed{seed}.pt", build_capsule(base))

        sequential = make_graph(seed)
        apply_capsule(sequential, load_capsule(ROOT / f"output/s2_continual/base_seed{seed}.pt"))
        train_mixed(
            sequential,
            old_features,
            old_labels,
            new_features,
            new_labels,
            NEW_STEPS,
            seed + 1,
            False,
        )
        sequential_new = accuracy(sequential, new_fit_features, new_fit_labels, 3)
        sequential_old = accuracy(sequential, old_fit_features, old_fit_labels, 24)

        replay = make_graph(seed)
        apply_capsule(replay, load_capsule(ROOT / f"output/s2_continual/base_seed{seed}.pt"))
        train_mixed(
            replay, old_features, old_labels, new_features, new_labels, NEW_STEPS, seed + 2, True
        )
        replay_new = accuracy(replay, new_fit_features, new_fit_labels, 3)
        replay_old = accuracy(replay, old_fit_features, old_fit_labels, 24)

        scratch = make_graph(seed)
        train_mixed(
            scratch, old_features, old_labels, new_features, new_labels, NEW_STEPS, seed + 3, False
        )
        scratch_new = accuracy(scratch, new_fit_features, new_fit_labels, 3)

        results.append(
            {
                "seed": seed,
                "old_before": round(old_before, 4),
                "sequential": {"new": round(sequential_new, 4), "old": round(sequential_old, 4)},
                "replay": {"new": round(replay_new, 4), "old": round(replay_old, 4)},
                "scratch_new": round(scratch_new, 4),
                "replay_retention_drop": round(old_before - replay_old, 4),
                "sequential_retention_drop": round(old_before - sequential_old, 4),
            }
        )

    replay_new_values = [result["replay"]["new"] for result in results]
    sequential_new_values = [result["sequential"]["new"] for result in results]
    retention = [result["replay_retention_drop"] for result in results]
    gates = {
        "sequential_new_learnable": all(value >= 0.43 for value in sequential_new_values),
        "replay_keeps_old": all(value <= 0.02 for value in retention),
        "replay_not_worse_than_sequential": all(
            replay_new_values[index] >= sequential_new_values[index] - 0.05
            for index in range(len(results))
        ),
    }
    verdict = "continual_learning_pass" if all(gates.values()) else "continual_learning_absent"
    payload = {
        "schema": "taiji-u2-continual-v1",
        "card": "cross-session continual learning with replay (joint24 -> action bins)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "old_steps": OLD_STEPS,
            "new_steps": NEW_STEPS,
            "replay_fraction": 0.2,
            "seeds": list(SEEDS),
            "gates": "sequential new>=0.43; replay retention drop<=0.02; replay new>=sequential-0.05",
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": (OLD_STEPS + NEW_STEPS * 2) * len(SEEDS),
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
    return 0 if verdict == "continual_learning_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
