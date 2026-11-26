"""Joint-gate explicit loop: combined state+write-back channel, with write-back-only sufficiency (0 check).

Variant of D-20261119-132/133: during training the persistent state is randomly reset
after step 1 (p=0.5), so the model learns to use both state and write-back.  Evaluation
checks the primary path (state kept) and the state-ablated path (write-back only).
Gates: self>=0.85; ablated self>=0.85; zero-writeback drop>=0.10; shuffled-order
drop>=0.10; oracle>=0.90.  3 seeds; cap 200 CPU.
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

OUT = ROOT / "reports/taiji_u2_sequential_loop_joint_20261120.json"
CAP_CPU = 200.0
FIT_TRIPLES = 128
STEPS = 1500
BATCH = 24
LR = 3e-3
WARMUP = 100
SEEDS = (0, 1, 2)
RESET_PROB = 0.5
MEMORY_DIMS = (39, 40)
INVALID_DIM = 41
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
SELF_MIN = 0.85
ABLATED_MIN = 0.85
ZERO_DROP = 0.10
ORDER_DROP = 0.10
ORACLE_MIN = 0.90


def learning_rate(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[..., index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def build_triples() -> dict[str, list[dict]]:
    rows = []
    for seed in (20261120, 20261121, 20261122, 20261123):
        rows.extend(build_partition("train", 256, seed))
    triples = []
    for index in range(len(rows) // 3):
        first, second, third = rows[3 * index : 3 * index + 3]
        d1 = 1 if first["direction"] > 0 else 0
        d3 = 1 if third["direction"] > 0 else 0
        triples.append(
            {
                "r": [first["features"], second["features"], third["features"]],
                "d1": d1,
                "d3": d3,
                "same13": int(d1 == d3),
            }
        )
    positives = [trip for trip in triples if trip["same13"] == 1]
    negatives = [trip for trip in triples if trip["same13"] == 0]
    generator = torch.Generator().manual_seed(20261120)

    def take(pool, count):
        order = torch.randperm(len(pool), generator=generator)[:count].tolist()
        return [pool[index] for index in order]

    count = min(len(positives), len(negatives))
    per_side_fit = min(FIT_TRIPLES // 2, count // 2)
    train = take(positives, count // 2) + take(negatives, count // 2)
    fit = take(positives[count // 2 :], per_side_fit) + take(negatives[count // 2 :], per_side_fit)
    order = torch.randperm(len(train), generator=generator).tolist()
    return {"train": [train[index] for index in order], "fit": fit}


def build_graph(seed: int) -> RouteBGraph:
    torch.manual_seed(seed)
    return RouteBGraph(
        UnitGraphSpec(
            groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=2, seed=seed
        )
    )


def inject(features: torch.Tensor, writeback: torch.Tensor | None) -> torch.Tensor:
    out = features.unsqueeze(1).clone()
    out[:, :, list(MEMORY_DIMS)] = 0.0
    if writeback is None:
        out[:, :, INVALID_DIM] = 1.0
    else:
        out[:, :, INVALID_DIM] = 0.0
        for item, value in enumerate(writeback.tolist()):
            out[item, 0, MEMORY_DIMS[int(value)]] = 1.0
    return out


def reset(state):
    return {"z": torch.zeros_like(state["z"]), "a": torch.zeros_like(state["a"])}


def forward_batch(
    graph,
    scan,
    batch,
    mode: str = "self",
    oracle: bool = False,
    swapped: bool = False,
    force_reset: bool = False,
    reset_prob: float = 0.0,
    generator=None,
):
    order = [2, 1, 0] if swapped else [0, 1, 2]
    state = None
    step1_logits = None
    previous = None
    do_reset = force_reset or (
        reset_prob > 0
        and generator is not None
        and float(torch.rand(1, generator=generator)) < reset_prob
    )
    for step in order:
        features = torch.stack([trip["r"][step] for trip in batch])
        if step == 0 or mode == "zero":
            writeback = None
        elif oracle:
            writeback = torch.tensor([trip["d1"] for trip in batch])
        elif mode == "shuffled":
            writeback = torch.tensor([trip["d1"] for trip in reversed(batch)])
        else:
            writeback = previous
        result = scan.run(
            channels(inject(features, writeback)),
            torch.ones(features.shape[0], 1, graph.units.n_units),
            state,
        )
        state = result["state"]
        logits = result["readout"][:, 0]
        if step == 0:
            step1_logits = logits
        if step == 2:
            return step1_logits, logits
        previous = logits.argmax(-1).detach()
        if do_reset:
            state = reset(state)
    raise RuntimeError("order_missing_final_step")


def final_accuracy(
    graph,
    scan,
    triples,
    mode: str = "self",
    oracle: bool = False,
    swapped: bool = False,
    force_reset: bool = False,
) -> float:
    labels = torch.tensor([trip["same13"] for trip in triples])
    with torch.no_grad():
        _, logits = forward_batch(graph, scan, triples, mode, oracle, swapped, force_reset)
    return float((logits.argmax(-1) == labels).float().mean())


def train(graph, scan, triples, seed: int) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, STEPS)
        index = torch.randint(0, len(triples), (BATCH,), generator=generator)
        batch = [triples[position] for position in index.tolist()]
        step1_logits, final_logits = forward_batch(
            graph, scan, batch, "self", reset_prob=RESET_PROB, generator=generator
        )
        loss = loss_fn(step1_logits, torch.tensor([trip["d1"] for trip in batch])) + loss_fn(
            final_logits, torch.tensor([trip["same13"] for trip in batch])
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    data = build_triples()
    results = []
    for seed in SEEDS:
        graph = build_graph(20261120 + seed)
        scan = UnitScan(graph)
        train(graph, scan, data["train"], seed)
        fit = data["fit"]
        self_acc = final_accuracy(graph, scan, fit, "self")
        ablated_acc = final_accuracy(graph, scan, fit, "self", force_reset=True)
        zero_acc = final_accuracy(graph, scan, fit, "zero")
        swapped_acc = final_accuracy(graph, scan, fit, "self", swapped=True)
        oracle_acc = final_accuracy(graph, scan, fit, "self", oracle=True)
        results.append(
            {
                "seed": seed,
                "self": round(self_acc, 4),
                "self_state_ablated": round(ablated_acc, 4),
                "zero_writeback": round(zero_acc, 4),
                "swapped_order": round(swapped_acc, 4),
                "oracle": round(oracle_acc, 4),
                "zero_drop": round(self_acc - zero_acc, 4),
                "order_drop": round(self_acc - swapped_acc, 4),
            }
        )
    gates = {
        "self_high": all(result["self"] >= SELF_MIN for result in results),
        "writeback_sufficient": all(
            result["self_state_ablated"] >= ABLATED_MIN for result in results
        ),
        "zero_necessity": all(result["zero_drop"] >= ZERO_DROP for result in results),
        "order_dependence": all(result["order_drop"] >= ORDER_DROP for result in results),
        "oracle_reference": all(result["oracle"] >= ORACLE_MIN for result in results),
    }
    if all(gates.values()):
        verdict = "sequential_loop_joint_ready"
    elif sum(1 for value in gates.values() if value) >= 3:
        verdict = "sequential_loop_joint_partial"
    else:
        verdict = "sequential_loop_joint_absent"
    payload = {
        "schema": "taiji-u2-sequential-loop-joint-v1",
        "card": "joint-gate 3-step loop: state+write-back training, write-back-only sufficiency",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "steps": STEPS,
            "reset_prob": RESET_PROB,
            "gates": "self>=0.85; state-ablated self>=0.85; zero drop>=0.10; order drop>=0.10; oracle>=0.90",
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
    return 0 if verdict == "sequential_loop_joint_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
