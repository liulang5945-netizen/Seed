"""Experiment-level persistent-state probe with a learned write gate (0 check).

The unit graph has no per-sample write protection (z is overwritten by the current input), so
the earlier per-sample decoding found no retention beyond t0.  This probe keeps the frozen
unit core untouched and adds an experiment-level recurrent buffer over the per-tick unit
activities: g_t = sigmoid(w . a_t + b), s_t = (1 - g_t) s_{t-1} + g_t a_t.  A decoder reads s_T
to predict the t0 target.  Controls: last-activity only, running mean, and an external-history
attention (upper bound).  Gate: gated buffer accuracy >= 0.60 and above the no-gate control.
Zero check/final.
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

OUT = ROOT / "reports/taiji_u2_persistent_gate_20261121.json"
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
STEPS = 800
BATCH = 64
LR = 3e-3
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
    labels = []
    sequences = []
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


class GatedBuffer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, 1)
        self.decoder = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 3))

    def forward(self, activities: torch.Tensor, gated: bool = True) -> torch.Tensor:
        state = activities[:, 0]
        for tick_index in range(1, activities.shape[1]):
            if gated:
                gate = torch.sigmoid(self.gate(activities[:, tick_index]))
                state = (1.0 - gate) * state + gate * activities[:, tick_index]
            else:
                state = state + activities[:, tick_index]
        if not gated:
            state = state / activities.shape[1]
        return self.decoder(state)


def train_head(head: nn.Module, batch: dict[str, torch.Tensor], gated: bool, seed: int) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, batch["activities"].shape[0], (BATCH,), generator=generator)
        loss = loss_fn(head(batch["activities"][index], gated=gated), batch["labels"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def accuracy(head: nn.Module, batch: dict[str, torch.Tensor], gated: bool) -> float:
    with torch.no_grad():
        logits = head(batch["activities"], gated=gated)
    return float((logits.argmax(-1) == batch["labels"]).float().mean())


def raw_replay_accuracy(
    train: dict[str, torch.Tensor], fit: dict[str, torch.Tensor], seed: int
) -> float:
    # external-history attention on the same activities (upper bound)
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
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        index = torch.randint(0, train["activities"].shape[0], (BATCH,), generator=generator)
        loss = loss_fn(head(train["activities"][index]), train["labels"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = head(fit["activities"])
    return float((logits.argmax(-1) == fit["labels"]).float().mean())


def run_seed(seed: int) -> dict:
    graph, _ = train_joint_parent(seed)
    for parameter in graph.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    train = record_activity(graph, build(TRAIN, seed))
    fit = record_activity(graph, build(FIT, seed + FIT_OFFSET))
    results = {}
    head = GatedBuffer(train["activities"].shape[-1])
    train_head(head, train, True, seed)
    results["gated"] = accuracy(head, fit, True)
    head = GatedBuffer(train["activities"].shape[-1])
    train_head(head, train, False, seed)
    results["mean"] = accuracy(head, fit, False)
    results["history"] = raw_replay_accuracy(train, fit, seed)
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
        "gated_level": means["gated"] >= GATE_MIN,
        "gated_gain": means["gated"] >= means["mean"] + MARGIN,
    }
    verdict = "persistent_gate_verified" if all(gates.values()) else "persistent_gate_partial"
    payload = {
        "schema": "taiji-u2-persistent-gate-v1",
        "card": "experiment-level gated persistent buffer over per-tick unit activities",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "train": TRAIN,
            "fit": FIT,
            "fit_offset": FIT_OFFSET,
            "steps": STEPS,
            "gate_min": GATE_MIN,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "unit core unchanged; gated buffer is an experiment-level interface (F/W direction), not proof of unit-state retention",
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
    return 0 if verdict == "persistent_gate_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
