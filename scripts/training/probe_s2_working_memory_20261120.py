"""Working-memory write-back probe: tick1 prediction as tick2 input (0 check).

Trains a two-step action-pair model where tick2 sees r2 plus a one-hot write-back of
the predicted bin1 in feature dims 39-41; evaluates self/zero/shuffled/oracle arms and
checks the four pre-registered gates (D-20261119-109).  3 seeds; cap 200 CPU.
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
from scripts.training.probe_s2_sequential_recursion_scan_20261120 import (  # noqa: E402
    channels,
    gates_for,
    make_pairs,
)
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402
from taiji.u2_scan import UnitScan  # noqa: E402

OUT = ROOT / "reports/taiji_u2_working_memory_20261120.json"
CAP_CPU = 200.0
STEPS = 1500
BATCH_PAIRS = 16
LR = 3e-3
WARMUP = 100
SEEDS = (0, 1, 2)
MEMORY_DIMS = (39, 40, 41)
SELF_MIN = 0.55
SELF_OVER_ZERO = 0.10
NECESSITY_DROP = 0.10
ORACLE_MIN = 0.65
SHUFFLED_MAX = 0.45


def learning_rate(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def build_graph(seed: int) -> RouteBGraph:
    torch.manual_seed(seed)
    return RouteBGraph(
        UnitGraphSpec(
            groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=3, seed=seed
        )
    )


def feature_sequences(batch, writeback: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    firsts = torch.stack([pair["r1"] for pair in batch])
    seconds = torch.stack([pair["r2"] for pair in batch]).clone()
    if writeback is not None:
        seconds[:, list(MEMORY_DIMS)] = 0.0
        for item, value in enumerate(writeback.tolist()):
            if 0 <= value <= 2:
                seconds[item, MEMORY_DIMS[value]] = 1.0
    return channels(torch.stack([firsts, seconds], dim=1))


def tick1_logits(graph, scan, batch) -> torch.Tensor:
    features = channels(torch.stack([pair["r1"] for pair in batch]).unsqueeze(1))
    return scan.run(features, gates_for(graph, len(batch), 1))["readout"][:, 0]


def tick2_logits(graph, scan, batch, writeback) -> torch.Tensor:
    features = feature_sequences(batch, writeback)
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
        bin1 = torch.tensor([pair["bin1"] for pair in batch])
        bin2 = torch.tensor([pair["bin2"] for pair in batch])
        writeback = tick1_logits(graph, scan, batch).argmax(-1).detach()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(tick1_logits(graph, scan, batch), bin1) + loss_fn(
            tick2_logits(graph, scan, batch, writeback), bin2
        )
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def evaluate(graph, scan, pairs) -> dict[str, float]:
    bin1 = torch.tensor([pair["bin1"] for pair in pairs])
    bin2 = torch.tensor([pair["bin2"] for pair in pairs])
    with torch.no_grad():
        predicted = tick1_logits(graph, scan, pairs).argmax(-1)
        generator = torch.Generator().manual_seed(20261120)
        shuffled = torch.randint(0, 3, (len(pairs),), generator=generator)
        self_logits = tick2_logits(graph, scan, pairs, predicted)
        zero_logits = tick2_logits(graph, scan, pairs, None)
        shuffled_logits = tick2_logits(graph, scan, pairs, shuffled)
        oracle_logits = tick2_logits(graph, scan, pairs, bin1)
    return {
        "tick1": float((predicted == bin1).float().mean()),
        "self": float((self_logits.argmax(-1) == bin2).float().mean()),
        "zero": float((zero_logits.argmax(-1) == bin2).float().mean()),
        "shuffled": float((shuffled_logits.argmax(-1) == bin2).float().mean()),
        "oracle": float((oracle_logits.argmax(-1) == bin2).float().mean()),
    }


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
        train(graph, scan, train_pairs, STEPS, seed)
        scores = evaluate(graph, scan, fit_pairs)
        results.append({"seed": seed, **{key: round(value, 4) for key, value in scores.items()}})
    gates = {
        "self_over_zero": all(
            result["self"] >= SELF_MIN and result["self"] - result["zero"] >= SELF_OVER_ZERO
            for result in results
        ),
        "necessity": all(result["self"] - result["zero"] >= NECESSITY_DROP for result in results),
        "oracle_reference": all(result["oracle"] >= ORACLE_MIN for result in results),
        "shuffled_control": all(result["shuffled"] <= SHUFFLED_MAX for result in results),
    }
    if all(gates.values()):
        verdict = "working_memory_writeback_ready"
    elif not gates["oracle_reference"]:
        verdict = "task_not_learnable"
    elif sum(1 for value in gates.values() if value) >= 2:
        verdict = "working_memory_writeback_partial"
    else:
        verdict = "working_memory_writeback_absent"
    payload = {
        "schema": "taiji-u2-working-memory-v1",
        "card": "explicit working-memory write-back of tick1 prediction into tick2 input",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "steps": STEPS,
            "memory_dims": list(MEMORY_DIMS),
            "gates": "self>=0.55 & self-zero>=0.10; necessity drop>=0.10; oracle>=0.65; shuffled<=0.45",
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
