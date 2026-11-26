"""Bounded key-value store F over per-tick unit activities with learned write/read (0 check).

The unit core stays frozen and the per-sample activity sequence is collected as in the
state-decode probe.  An experiment-level bounded store (K slots with learnable keys and
value projection) writes each tick with a softmax write address and reads at the final tick
with a learned query address.  Gate: bounded-store accuracy >= 0.60 and >= no-store mean + 0.10,
with the external-history attention as upper bound.  Zero check/final.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import channels, lr_at  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402

OUT = ROOT / "reports/taiji_u2_bounded_store_20261121.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
TRAIN = 256
FIT = 96
FIT_OFFSET = 10_000_000
TICKS = 10
DIM = 48
BODY = 3
BODY_STRIDE = 12
TICK_OFFSET = 36
TICK_SLOTS = 12
STEPS = 1200
BATCH = 64
LR = 3e-3
SLOTS = 8
SLOT_DIM = 16
GATE_MIN = 0.60
MARGIN = 0.10


def episode(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target_colour = int(torch.randint(0, 6, (1,), generator=generator))
    target_position = int(torch.randint(0, 3, (1,), generator=generator))
    ticks = []
    for tick_index in range(TICKS):
        features = torch.zeros(DIM)
        for body in range(BODY):
            if tick_index == 0 and body == 0:
                colour, position = target_colour, target_position
            else:
                colour = int(torch.randint(0, 6, (1,), generator=generator))
                position = int(torch.randint(0, 3, (1,), generator=generator))
            base = body * BODY_STRIDE
            features[base + colour] = 1.0
            features[base + 6 + position] = 1.0
        features[TICK_OFFSET + (tick_index % TICK_SLOTS)] = 1.0
        ticks.append(features)
    return {"ticks": torch.stack(ticks), "label": target_position}


def build(count: int, seed: int) -> list[dict]:
    return [episode(seed + index) for index in range(count)]


def record_activity(graph, rows: list[dict]) -> dict[str, torch.Tensor]:
    labels, sequences = [], []
    for row in rows:
        state = graph.initial_state()
        activities = []
        for tick_index in range(TICKS):
            gates = torch.ones(1, graph.units.n_units)
            with torch.no_grad():
                result = graph(channels(row["ticks"][tick_index : tick_index + 1]), gates, state)
            activities.append(result["unit_result"]["activity"].squeeze(0))
            state = result["next_state"]
        sequences.append(torch.stack(activities))
        labels.append(row["label"])
    return {"activities": torch.stack(sequences).float(), "labels": torch.tensor(labels)}


class BoundedStore(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.keys = nn.Parameter(torch.randn(SLOTS, SLOT_DIM) * 0.1)
        self.write_in = nn.Linear(dim, SLOT_DIM)
        self.write_out = nn.Linear(dim, SLOT_DIM)
        self.read_query = nn.Linear(dim, SLOT_DIM)
        self.decoder = nn.Sequential(nn.Linear(SLOT_DIM + dim, 64), nn.Tanh(), nn.Linear(64, 3))

    def forward(self, activities: torch.Tensor) -> torch.Tensor:
        batch = activities.shape[0]
        values = torch.zeros(batch, SLOTS, SLOT_DIM)
        for tick_index in range(activities.shape[1]):
            address = torch.softmax(self.write_in(activities[:, tick_index]) @ self.keys.T, dim=-1)
            candidate = torch.tanh(self.write_out(activities[:, tick_index])).unsqueeze(1)
            values = (1.0 - address.unsqueeze(-1)) * values + address.unsqueeze(-1) * candidate
        query = self.read_query(activities[:, -1])
        scores = torch.softmax(query @ self.keys.T, dim=-1)
        read = torch.einsum("bk,bkd->bd", scores, values)
        return self.decoder(torch.cat([read, activities[:, -1]], dim=-1))


class MeanRead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 3))

    def forward(self, activities: torch.Tensor) -> torch.Tensor:
        return self.decoder(activities.mean(dim=1))


def train_head(head: nn.Module, batch: dict[str, torch.Tensor], seed: int) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, batch["activities"].shape[0], (BATCH,), generator=generator)
        loss = loss_fn(head(batch["activities"][index]), batch["labels"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def accuracy(head: nn.Module, batch: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        logits = head(batch["activities"])
    return float((logits.argmax(-1) == batch["labels"]).float().mean())


def history_accuracy(
    train: dict[str, torch.Tensor], fit: dict[str, torch.Tensor], seed: int
) -> float:
    class Attn(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.pos = nn.Parameter(torch.zeros(TICKS, dim))
            self.token = nn.Sequential(nn.Linear(dim, 64), nn.Tanh())
            self.query = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
            self.out = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 3))

        def forward(self, activities: torch.Tensor) -> torch.Tensor:
            embed = self.token(activities + self.pos)
            key = embed.mean(dim=1)
            scores = torch.softmax(torch.einsum("btd,bd->bt", embed, self.query(key)), dim=-1)
            context = torch.einsum("bt,btd->bd", scores, embed)
            return self.out(torch.cat([context, key], dim=-1))

    head = Attn(train["activities"].shape[-1])
    train_head(head, train, seed)
    return accuracy(head, fit)


def run_seed(seed: int) -> dict:
    graph, _ = train_joint_parent(seed)
    for parameter in graph.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    train = record_activity(graph, build(TRAIN, seed))
    fit = record_activity(graph, build(FIT, seed + FIT_OFFSET))
    results = {}
    head = BoundedStore(train["activities"].shape[-1])
    train_head(head, train, seed)
    results["bounded_store"] = accuracy(head, fit)
    head = MeanRead(train["activities"].shape[-1])
    train_head(head, train, seed)
    results["mean"] = accuracy(head, fit)
    results["history"] = history_accuracy(train, fit, seed)
    return {"seed": seed, "arms": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {arm: sum(row["arms"][arm] for row in rows) / len(rows) for arm in rows[0]["arms"]}
    gates = {
        "store_level": means["bounded_store"] >= GATE_MIN,
        "store_gain": means["bounded_store"] >= means["mean"] + MARGIN,
    }
    verdict = "bounded_store_verified" if all(gates.values()) else "bounded_store_partial"
    payload = {
        "schema": "taiji-u2-bounded-store-v1",
        "card": "experiment-level bounded key-value store over per-tick unit activities",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "slots": SLOTS,
            "slot_dim": SLOT_DIM,
            "train": TRAIN,
            "fit": FIT,
            "fit_offset": FIT_OFFSET,
            "steps": STEPS,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "unit core frozen; the store is an experiment-level bounded interface with learned addresses, not unit-state retention",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "means": means, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "bounded_store_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
