"""Memory-slot loop: dedicated K=2 slots carry step-1 direction to the final readout (0 check).

Three-step same13 task (D-20261119-135).  Memory slots are written every step through a
learned soft gate; training randomly zeroes the persistent unit state (p=0.5) so the
slots must carry information.  Arms: self, state-ablated, zero-memory, shuffled order,
oracle.  Gates: self>=0.85; ablated>=0.85; zero-memory drop>=0.15; order drop>=0.15;
oracle>=0.90.  3 seeds; cap 250 CPU.
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
from taiji.u2_memory import MemoryLoop  # noqa: E402
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402

OUT = ROOT / "reports/taiji_u2_memory_slots_20261120.json"
CAP_CPU = 250.0
STEPS = 1500
BATCH = 24
LR = 3e-3
WARMUP = 100
SEEDS = (0, 1, 2)
RESET_PROB = 0.5
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
SELF_MIN = 0.85
ABLATED_MIN = 0.85
ZERO_DROP = 0.15
ORDER_DROP = 0.15
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
    train = take(positives, count // 2) + take(negatives, count // 2)
    fit = take(positives[count // 2 :], min(64, count // 2)) + take(
        negatives[count // 2 :], min(64, count // 2)
    )
    order = torch.randperm(len(train), generator=generator).tolist()
    return {"train": [train[index] for index in order], "fit": fit}


def build_model(seed: int) -> MemoryLoop:
    torch.manual_seed(seed)
    graph = RouteBGraph(
        UnitGraphSpec(
            groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=2, seed=seed
        )
    )
    return MemoryLoop(graph, slots=2, dim=8)


def reset_state(state):
    return {"z": torch.zeros_like(state["z"]), "a": torch.zeros_like(state["a"])}


def run_loop(
    model: MemoryLoop,
    batch,
    mode: str = "self",
    force_reset: bool = False,
    swapped: bool = False,
    oracle: bool = False,
):
    model.memory.reset()
    order = [2, 1, 0] if swapped else [0, 1, 2]
    state = None
    step1_logits = None
    for position, step in enumerate(order):
        features = torch.stack([trip["r"][step] for trip in batch])
        logits, state, _activity = model.step(channels(features.unsqueeze(1)), state)
        if step == 0:
            step1_logits = logits
            if oracle:
                memory = torch.zeros(features.shape[0], model.memory.slots * model.memory.dim)
                for item, trip in enumerate(batch):
                    memory[item, trip["d1"]] = 1.0
                model.memory.memory = memory
        if mode == "zero":
            model.memory.memory = None
        if force_reset:
            state = reset_state(state)
        if step == 2:
            return step1_logits, logits
        if swapped and position == 1:
            pass
    raise RuntimeError("loop_missing_final_step")


def accuracy(
    model: MemoryLoop,
    triples,
    mode: str = "self",
    force_reset: bool = False,
    swapped: bool = False,
    oracle: bool = False,
) -> float:
    labels = torch.tensor([trip["same13"] for trip in triples])
    with torch.no_grad():
        _, logits = run_loop(model, triples, mode, force_reset, swapped, oracle)
    return float((logits.argmax(-1) == labels).float().mean())


def train(model: MemoryLoop, triples, seed: int) -> None:
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, STEPS)
        index = torch.randint(0, len(triples), (BATCH,), generator=generator)
        batch = [triples[position] for position in index.tolist()]
        do_reset = float(torch.rand(1, generator=generator)) < RESET_PROB
        step1_logits, final_logits = run_loop(model, batch, "self", force_reset=do_reset)
        loss = loss_fn(step1_logits, torch.tensor([trip["d1"] for trip in batch])) + loss_fn(
            final_logits, torch.tensor([trip["same13"] for trip in batch])
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.graph.units.project_parameters_()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    data = build_triples()
    results = []
    for seed in SEEDS:
        model = build_model(20261120 + seed)
        train(model, data["train"], seed)
        fit = data["fit"]
        self_acc = accuracy(model, fit, "self")
        ablated = accuracy(model, fit, "self", force_reset=True)
        zero = accuracy(model, fit, "zero")
        swapped = accuracy(model, fit, "self", swapped=True)
        oracle = accuracy(model, fit, "self", oracle=True)
        results.append(
            {
                "seed": seed,
                "self": round(self_acc, 4),
                "state_ablated": round(ablated, 4),
                "zero_memory": round(zero, 4),
                "swapped_order": round(swapped, 4),
                "oracle": round(oracle, 4),
                "zero_drop": round(self_acc - zero, 4),
                "order_drop": round(self_acc - swapped, 4),
            }
        )
    gates = {
        "self_high": all(result["self"] >= SELF_MIN for result in results),
        "single_channel": all(result["state_ablated"] >= ABLATED_MIN for result in results),
        "memory_necessity": all(result["zero_drop"] >= ZERO_DROP for result in results),
        "order_dependence": all(result["order_drop"] >= ORDER_DROP for result in results),
        "oracle_reference": all(result["oracle"] >= ORACLE_MIN for result in results),
    }
    if all(gates.values()):
        verdict = "memory_slot_ready"
    elif not gates["single_channel"]:
        verdict = "memory_single_channel_insufficient"
    elif sum(1 for value in gates.values() if value) >= 3:
        verdict = "memory_slot_partial"
    else:
        verdict = "memory_slot_absent"
    payload = {
        "schema": "taiji-u2-memory-slots-v1",
        "card": "K=2 dedicated working-memory slots over 3-step same13",
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
            "gates": "self>=0.85; ablated>=0.85; zero drop>=0.15; order drop>=0.15; oracle>=0.90",
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
    return 0 if verdict == "memory_slot_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
