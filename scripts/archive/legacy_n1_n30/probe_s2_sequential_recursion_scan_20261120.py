"""Sequential recursion with UnitScan: true per-sample tick1->tick2 recurrence (0 check).

Two-step action pairs (bin1 = r1.final_bin, bin2 = bin1 + (r2.final_bin - r2.current_bin));
tick1 sees r1 and must predict bin1, tick2 sees r2 and must predict bin2 with per-sample
state carried (UnitScan, batch x time separated).  Same four gates as D-20261119-104.
Controls: single-tick (r2 only, position proxy zeroed), swapped order, unrelated tick2.
3 seeds; cap 200 CPU.
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

OUT = ROOT / "reports/taiji_u2_sequential_recursion_scan_20261120.json"
CAP_CPU = 200.0
PAIRS = 128
FIT_PAIRS = 24
STEPS = 1500
BASE_STEPS = 1500
BATCH_PAIRS = 16
LR = 3e-3
WARMUP = 100
SEEDS = (0, 1, 2)
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
SEQ_MIN = 0.55
SEQ_OVER_TICK = 0.10
SHUFFLED_DROP = 0.10
TICK1_MIN = 0.50
UNRELATED_MAX = 0.45


def learning_rate(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[..., index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def make_pairs() -> dict[str, list[dict]]:
    rows = build_partition("train", 8 * (PAIRS + FIT_PAIRS), 20261120)
    pairs = []
    for index in range(len(rows) // 2):
        first, second = rows[2 * index], rows[2 * index + 1]
        bin1 = int(first["final_bin"])
        bin2 = bin1 + int(second["final_bin"]) - int(second["current_bin"])
        if 0 <= bin2 <= 2:
            pairs.append(
                {"r1": first["features"], "r2": second["features"], "bin1": bin1, "bin2": bin2}
            )
    return {"train": pairs[:PAIRS], "fit": pairs[PAIRS : PAIRS + FIT_PAIRS]}


def stack_sequences(
    pairs, mode: str = "ordered"
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    firsts, seconds, labels = [], [], []
    for pair in pairs:
        first, second = pair["r1"], pair["r2"]
        if mode == "swapped":
            first, second = second, first
        if mode == "unrelated":
            second = pair["r1"]
        firsts.append(first)
        seconds.append(second)
        labels.append([pair["bin1"], pair["bin2"]])
    features = torch.stack([torch.stack(firsts), torch.stack(seconds)], dim=1)  # [B,2,48]
    label_tensor = torch.tensor(labels)
    return channels(features), label_tensor[:, 0], label_tensor[:, 1]


def build_graph(seed: int) -> RouteBGraph:
    torch.manual_seed(seed)
    return RouteBGraph(
        UnitGraphSpec(
            groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=3, seed=seed
        )
    )


def gates_for(graph: RouteBGraph, batch: int, ticks: int) -> torch.Tensor:
    return torch.ones(batch, ticks, graph.units.n_units)


def train_sequence(graph, scan, pairs, steps: int, seed: int) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)
        index = torch.randint(0, len(pairs), (BATCH_PAIRS,), generator=generator)
        batch = [pairs[position] for position in index.tolist()]
        batch_channels, bin1, bin2 = stack_sequences(batch)
        optimizer.zero_grad(set_to_none=True)
        logits = scan.run(batch_channels, gates_for(graph, BATCH_PAIRS, 2))["readout"]
        loss = loss_fn(logits[:, 0], bin1) + loss_fn(logits[:, 1], bin2)
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def tick_accuracy(graph, scan, pairs, mode: str = "ordered") -> tuple[float, float]:
    batch_channels, bin1, bin2 = stack_sequences(pairs, mode)
    with torch.no_grad():
        logits = scan.run(batch_channels, gates_for(graph, len(pairs), 2))["readout"]
    tick1 = float((logits[:, 0].argmax(-1) == bin1).float().mean())
    tick2 = float((logits[:, 1].argmax(-1) == bin2).float().mean())
    return tick1, tick2


def train_single(graph, scan, pairs, steps: int, seed: int) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)
        index = torch.randint(0, len(pairs), (BATCH_PAIRS,), generator=generator)
        batch = [pairs[position] for position in index.tolist()]
        features = torch.stack([pair["r2"] for pair in batch]).clone()
        features[:, 38] = 0.0
        labels = torch.tensor([pair["bin2"] for pair in batch])
        optimizer.zero_grad(set_to_none=True)
        logits = scan.run(channels(features.unsqueeze(1)), gates_for(graph, len(batch), 1))[
            "readout"
        ]
        loss = loss_fn(logits[:, 0], labels)
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def single_accuracy(graph, scan, pairs) -> float:
    features = torch.stack([pair["r2"] for pair in pairs]).clone()
    features[:, 38] = 0.0
    labels = torch.tensor([pair["bin2"] for pair in pairs])
    with torch.no_grad():
        logits = scan.run(channels(features.unsqueeze(1)), gates_for(graph, len(pairs), 1))[
            "readout"
        ]
    return float((logits[:, 0].argmax(-1) == labels).float().mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    data = make_pairs()
    train_pairs, fit_pairs = data["train"], data["fit"]
    results = []
    for seed in SEEDS:
        graph = build_graph(20261120 + seed)
        scan = UnitScan(graph)
        train_sequence(graph, scan, train_pairs, STEPS, seed)
        tick1, tick2 = tick_accuracy(graph, scan, fit_pairs)
        _, shuffled_tick2 = tick_accuracy(graph, scan, fit_pairs, "swapped")
        _, unrelated_tick2 = tick_accuracy(graph, scan, fit_pairs, "unrelated")
        single = build_graph(20261120 + seed + 500)
        single_scan = UnitScan(single)
        train_single(single, single_scan, train_pairs, BASE_STEPS, seed + 3)
        single_acc = single_accuracy(single, single_scan, fit_pairs)
        results.append(
            {
                "seed": seed,
                "tick1": round(tick1, 4),
                "tick2": round(tick2, 4),
                "shuffled_tick2": round(shuffled_tick2, 4),
                "unrelated_tick2": round(unrelated_tick2, 4),
                "single_tick": round(single_acc, 4),
                "sequence_gain": round(tick2 - single_acc, 4),
                "shuffled_drop": round(tick2 - shuffled_tick2, 4),
            }
        )
    gates = {
        "sequential_over_single": all(
            result["tick2"] >= SEQ_MIN and result["sequence_gain"] >= SEQ_OVER_TICK
            for result in results
        ),
        "order_dependence": all(result["shuffled_drop"] >= SHUFFLED_DROP for result in results),
        "intermediate_credit": all(result["tick1"] >= TICK1_MIN for result in results),
        "tick2_information": all(result["unrelated_tick2"] <= UNRELATED_MAX for result in results),
    }
    passed = sum(1 for value in gates.values() if value)
    if passed == len(gates):
        verdict = "sequential_recursion_ready"
    elif passed >= 2 and gates["sequential_over_single"]:
        verdict = "sequential_recursion_partial"
    else:
        verdict = "sequential_recursion_absent"
    payload = {
        "schema": "taiji-u2-sequential-recursion-scan-v1",
        "card": "two-step action pairs with UnitScan per-sample recurrence",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "steps": STEPS,
            "pairs": len(train_pairs),
            "fit_pairs": len(fit_pairs),
            "implementation": "UnitScan batch x time separated (D-20261119-107)",
            "gates": "seq acc>=0.55 & >=single+0.10; shuffled drop>=0.10; tick1>=0.50; unrelated tick2<=0.45",
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": (STEPS + BASE_STEPS) * len(SEEDS),
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
    return 0 if verdict == "sequential_recursion_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
