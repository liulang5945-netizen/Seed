"""Learnable two-step task: same/reverse direction with working-memory write-back (0 check).

Pairs label ``same = (direction1 == direction2)``; tick1 predicts direction1 (2 classes),
tick2 sees r2 plus a one-hot write-back of the predicted direction and predicts ``same``.
Single tick seeing only r2 cannot know direction1.  Arms: self / zero / shuffled / oracle.
Four pre-registered gates (D-20261119-111).  3 seeds; cap 250 CPU.
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
from scripts.training.probe_s2_action_consequence_20261120 import build_partition  # noqa: E402
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402
from taiji.u2_scan import UnitScan  # noqa: E402

OUT = ROOT / "reports/taiji_u2_learnable_two_step_balanced_20261120.json"
CAP_CPU = 250.0
PAIR_SEEDS = (20261120, 20261121, 20261122, 20261123, 20261124, 20261125)
STEPS = 1500
BATCH_PAIRS = 32
LR = 3e-3
WARMUP = 100
SEEDS = (0, 1, 2)
MEMORY_DIMS = (39, 40)
ORACLE_MIN = 0.85
SELF_MIN = 0.65
SELF_OVER_ZERO = 0.15
NECESSITY_DROP = 0.15
SHUFFLED_MAX = 0.55
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")


def learning_rate(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[..., index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def build_pairs() -> dict[str, list[dict]]:
    rows = []
    for seed in PAIR_SEEDS:
        rows.extend(build_partition("train", 256, seed))
    pairs = []
    for index in range(len(rows) // 2):
        first, second = rows[2 * index], rows[2 * index + 1]
        direction1 = 1.0 if first["direction"] > 0 else 0.0
        direction2 = 1.0 if second["direction"] > 0 else 0.0
        pairs.append(
            {
                "r1": first["features"],
                "r2": second["features"],
                "dir1": int(direction1),
                "same": int(direction1 == direction2),
            }
        )
    positives = [pair for pair in pairs if pair["same"] == 1]
    negatives = [pair for pair in pairs if pair["same"] == 0]
    generator = torch.Generator().manual_seed(20261120)

    def take(pool, count):
        order = torch.randperm(len(pool), generator=generator)[:count].tolist()
        return [pool[index] for index in order]

    count_train = min(len(positives), len(negatives))
    per_side_train = count_train // 2
    per_side_fit = min(64, count_train - per_side_train)
    train_pairs = take(positives, per_side_train) + take(negatives, per_side_train)
    fit_pairs = take(positives[per_side_train:], per_side_fit) + take(
        negatives[per_side_train:], per_side_fit
    )
    order = torch.randperm(len(train_pairs), generator=generator).tolist()
    train_pairs = [train_pairs[index] for index in order]
    return {"train": train_pairs, "fit": fit_pairs}


def build_graph(seed: int) -> RouteBGraph:
    torch.manual_seed(seed)
    return RouteBGraph(
        UnitGraphSpec(
            groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=2, seed=seed
        )
    )


def gates_for(graph: RouteBGraph, batch: int, ticks: int) -> torch.Tensor:
    return torch.ones(batch, ticks, graph.units.n_units)


def tick1_logits(graph, scan, batch) -> torch.Tensor:
    features = channels(torch.stack([pair["r1"] for pair in batch]).unsqueeze(1))
    return scan.run(features, gates_for(graph, len(batch), 1))["readout"][:, 0]


def tick2_logits(graph, scan, batch, writeback: torch.Tensor | None) -> torch.Tensor:
    seconds = torch.stack([pair["r2"] for pair in batch]).clone()
    if writeback is not None:
        seconds[:, list(MEMORY_DIMS)] = 0.0
        for item, value in enumerate(writeback.tolist()):
            seconds[item, MEMORY_DIMS[int(value)]] = 1.0
    features = channels(torch.stack([torch.stack([pair["r1"] for pair in batch]), seconds], dim=1))
    return scan.run(features, gates_for(graph, len(batch), 2))["readout"][:, 1]


def train(graph, scan, pairs, steps: int, seed: int) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)
        index = torch.randint(0, len(pairs), (BATCH_PAIRS,), generator=generator)
        batch = [pairs[position] for position in index.tolist()]
        dir1 = torch.tensor([pair["dir1"] for pair in batch])
        same = torch.tensor([pair["same"] for pair in batch])
        writeback = tick1_logits(graph, scan, batch).argmax(-1).detach()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(tick1_logits(graph, scan, batch), dir1) + loss_fn(
            tick2_logits(graph, scan, batch, writeback), same
        )
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def evaluate(graph, scan, pairs) -> dict[str, float]:
    dir1 = torch.tensor([pair["dir1"] for pair in pairs])
    same = torch.tensor([pair["same"] for pair in pairs])
    with torch.no_grad():
        predicted = tick1_logits(graph, scan, pairs).argmax(-1)
        generator = torch.Generator().manual_seed(20261120)
        shuffled = torch.randint(0, 2, (len(pairs),), generator=generator)
        self_logits = tick2_logits(graph, scan, pairs, predicted)
        zero_logits = tick2_logits(graph, scan, pairs, None)
        shuffled_logits = tick2_logits(graph, scan, pairs, shuffled)
        oracle_logits = tick2_logits(graph, scan, pairs, dir1)
    return {
        "direction_accuracy": float((predicted == dir1).float().mean()),
        "self": float((self_logits.argmax(-1) == same).float().mean()),
        "zero": float((zero_logits.argmax(-1) == same).float().mean()),
        "shuffled": float((shuffled_logits.argmax(-1) == same).float().mean()),
        "oracle": float((oracle_logits.argmax(-1) == same).float().mean()),
        "same_rate": float(same.float().mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    data = build_pairs()
    train_pairs, fit_pairs = data["train"], data["fit"]
    results = []
    for seed in SEEDS:
        graph = build_graph(20261120 + seed)
        scan = UnitScan(graph)
        train(graph, scan, train_pairs, STEPS, seed)
        scores = evaluate(graph, scan, fit_pairs)
        results.append({"seed": seed, **{key: round(value, 4) for key, value in scores.items()}})
    gates = {
        "task_learnable": all(result["oracle"] >= ORACLE_MIN for result in results),
        "self_over_zero": all(
            result["self"] >= SELF_MIN and result["self"] - result["zero"] >= SELF_OVER_ZERO
            for result in results
        ),
        "necessity": all(result["self"] - result["zero"] >= NECESSITY_DROP for result in results),
        "shuffled_control": all(result["shuffled"] <= SHUFFLED_MAX for result in results),
    }
    if all(gates.values()):
        verdict = "working_memory_writeback_ready"
    elif not gates["task_learnable"]:
        verdict = "task_not_learnable"
    elif sum(1 for value in gates.values() if value) >= 2:
        verdict = "working_memory_writeback_partial"
    else:
        verdict = "working_memory_writeback_absent"
    payload = {
        "schema": "taiji-u2-learnable-two-step-v1",
        "card": "same/reverse direction with working-memory write-back (balanced 1:1)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "pair_seeds": list(PAIR_SEEDS),
            "train_pairs": len(train_pairs),
            "fit_pairs": len(fit_pairs),
            "memory_dims": list(MEMORY_DIMS),
            "gates": "oracle>=0.85; self>=0.65 & self-zero>=0.15; necessity drop>=0.15; shuffled<=0.55",
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"optimizer_steps": STEPS * len(SEEDS), "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "results": results, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "working_memory_writeback_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
