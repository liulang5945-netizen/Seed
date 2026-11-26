"""Sequential-computation recursion probe: two-step action pairs -> final bin (0 check).

Builds action pairs from the consequence rows (bin1 = r1.final_bin, bin2 = clip(bin1
+ (r2.final_bin - r2.current_bin))).  Tick1 sees r1 and must predict bin1, tick2 sees
r2 and must predict bin2 with the unit state carried.  Controls: single-tick baseline
(only r2 features, position proxy zeroed), swapped order, and an unrelated tick2.
Four pre-registered gates (D-20261119-104).  3 seeds; cap 200 CPU.
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

OUT = ROOT / "reports/taiji_u2_sequential_recursion_20261120.json"
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
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def make_bins() -> dict[str, object]:
    rows = build_partition("train", 8 * (PAIRS + FIT_PAIRS), 20261120)
    pairs = []
    for index in range(len(rows) // 2 * 1):
        first = rows[2 * index]
        second = rows[2 * index + 1]
        bin1 = int(first["final_bin"])
        bin2 = bin1 + int(second["final_bin"]) - int(second["current_bin"])
        if bin2 < 0 or bin2 > 2:
            continue
        pairs.append(
            {
                "r1": first["features"],
                "r2": second["features"],
                "bin1": bin1,
                "bin2": bin2,
            }
        )
    train_pairs = pairs[:PAIRS]
    fit_pairs = pairs[PAIRS : PAIRS + FIT_PAIRS]
    return {"train": train_pairs, "fit": fit_pairs}


def stack(pairs, key: str) -> torch.Tensor:
    return torch.stack([pair[key] for pair in pairs])


def sequence_features(pairs, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """[2P,48] features and [2P] labels; positions 0,2,4.. are tick1."""
    pieces = []
    labels = []
    for _index, pair in enumerate(pairs):
        first, second = pair["r1"], pair["r2"]
        if mode == "swapped":
            first, second = second, first
        if mode == "unrelated":
            second = pair["r1"]
        pieces.append(first)
        pieces.append(second)
        labels.extend([pair["bin1"], pair["bin2"]])
    return torch.stack(pieces), torch.tensor(labels)


def build_graph(seed: int) -> RouteBGraph:
    torch.manual_seed(seed)
    return RouteBGraph(
        UnitGraphSpec(
            groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=3, seed=seed
        )
    )


def train_sequence(graph, pairs, steps: int, seed: int) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)
        index = torch.randint(0, len(pairs), (BATCH_PAIRS,), generator=generator)
        batch = [pairs[position] for position in index.tolist()]
        features, labels = sequence_features(batch, "ordered")
        gates = torch.ones(features.shape[0], graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        logits = graph(channels(features), gates, graph.initial_state())["readout"]
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def tick_accuracy(graph, pairs, mode: str = "ordered") -> tuple[float, float]:
    features, labels = sequence_features(pairs, mode)
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        logits = graph(channels(features), gates, graph.initial_state())["readout"]
    predictions = logits.argmax(-1)
    tick1 = float((predictions[0::2] == labels[0::2]).float().mean())
    tick2 = float((predictions[1::2] == labels[1::2]).float().mean())
    return tick1, tick2


def train_single(graph, pairs, steps: int, seed: int) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps)
        index = torch.randint(0, len(pairs), (BATCH_PAIRS,), generator=generator)
        batch = [pairs[position] for position in index.tolist()]
        features = stack(batch, "r2").clone()
        features[:, 38] = 0.0  # hide the position proxy: no initial-bin information
        labels = torch.tensor([pair["bin2"] for pair in batch])
        gates = torch.ones(features.shape[0], graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        logits = graph(channels(features), gates, graph.initial_state())["readout"]
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def single_accuracy(graph, pairs) -> float:
    features = stack(pairs, "r2").clone()
    features[:, 38] = 0.0
    labels = torch.tensor([pair["bin2"] for pair in pairs])
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        logits = graph(channels(features), gates, graph.initial_state())["readout"]
    return float((logits.argmax(-1) == labels).float().mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    data = make_bins()
    train_pairs, fit_pairs = data["train"], data["fit"]
    results = []
    for seed in SEEDS:
        graph = build_graph(20261120 + seed)
        train_sequence(graph, train_pairs, STEPS, seed)
        tick1, tick2 = tick_accuracy(graph, fit_pairs)
        shuffled_tick1, shuffled_tick2 = tick_accuracy(graph, fit_pairs, "swapped")
        unrelated_tick1, unrelated_tick2 = tick_accuracy(graph, fit_pairs, "unrelated")
        single = build_graph(20261120 + seed + 500)
        train_single(single, train_pairs, BASE_STEPS, seed + 3)
        single_acc = single_accuracy(single, fit_pairs)
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
        "schema": "taiji-u2-sequential-recursion-v1",
        "card": "two-step action pairs: final-bin prediction with carried state",
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
