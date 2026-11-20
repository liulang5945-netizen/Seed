"""Working-memory load probe: 1 vs 3 vs 5 simultaneous targets across 5 ticks (0 check).

t0 shows N target bodies (unique colours); t1-t3 are distractor scenes; t4 asks the position
of the k-th target (query carries k).  Arms: no_state (t4 only), replay (t0+t4), state_carry
(all ticks, carried unit state).  Capacity pressure shows up as a large replay-carried gap at
N=3/5 while N=1 stays near replay.  Zero check/final.
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
from scripts.training.run_s2_23_0b_joint_20261120 import accuracy as skill_accuracy  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import (
    build_data as build_skill_data,  # noqa: E402
)
from scripts.training.run_s2_23_0b_joint_20261120 import channels, lr_at  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402

OUT = ROOT / "reports/taiji_u2_wm_load_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
LOADS = (1, 3, 5)
TICKS = 5
BODIES = 5
TRAIN = 512
FIT = 128
BATCH = 32
STEPS = 1500
LR = 3e-3
LOAD_DROP = 0.15
FIT_OFFSET = 10_000_000
RETENTION_MAX = 0.02


def episode(seed: int, load: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    colours = torch.randperm(6, generator=generator)[:load].tolist()
    positions = [int(torch.randint(0, 3, (1,), generator=generator)) for _ in range(load)]
    ticks = []
    for tick_index in range(TICKS):
        features = torch.zeros(80)
        for body in range(BODIES):
            if tick_index == 0 and body < load:
                colour, position = colours[body], positions[body]
            else:
                colour = int(torch.randint(0, 6, (1,), generator=generator))
                position = int(torch.randint(0, 3, (1,), generator=generator))
            features[body * 8 + 2 + colour] = 1.0
            features[body * 8 + 6 + position] = 1.0
        features[40 + tick_index] = 1.0
        ticks.append(features)
    query = torch.zeros(13)
    index = int(torch.randint(0, load, (1,), generator=generator))
    query[8 + index] = 1.0
    return {"ticks": torch.stack(ticks), "query": query, "label": positions[index]}


def build(count: int, seed: int, load: int) -> dict[str, torch.Tensor]:
    rows = [episode(seed + index, load) for index in range(count)]
    return {
        "ticks": torch.stack([row["ticks"] for row in rows]).float(),
        "queries": torch.stack([row["query"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


class TickRead(nn.Module):
    def __init__(self, token_dim: int, width: int = 64) -> None:
        super().__init__()
        self.query1 = nn.Sequential(nn.Linear(13, width), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(width + 13, width), nn.Tanh())
        self.token = nn.Sequential(nn.Linear(token_dim, width), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 4))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        embed = self.token(tokens)
        q1 = self.query1(query)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, q1), dim=-1)
        context1 = torch.einsum("bt,btd->bd", scores, embed)
        q2 = self.query2(torch.cat([context1, query], dim=-1))
        scores2 = torch.softmax(torch.einsum("btd,bd->bt", embed, q2), dim=-1)
        context2 = torch.einsum("bt,btd->bd", scores2, embed)
        return self.out(torch.cat([context2, q1], dim=-1))


def collect_activity(graph, ticks: torch.Tensor, use: list[int]) -> torch.Tensor:
    state = graph.initial_state()
    activities = []
    for tick_index in range(ticks.shape[1]):
        gates = torch.ones(ticks.shape[0], graph.units.n_units)
        result = graph(channels(ticks[:, tick_index]), gates, state)
        if tick_index in use:
            activities.append(result["unit_result"]["activity"])
        state = result["next_state"]
    return torch.stack(activities, dim=1)


def train_head(head: nn.Module, batch: dict[str, torch.Tensor], seed: int) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, batch["activities"].shape[0], (BATCH,), generator=generator)
        loss = loss_fn(
            head(batch["activities"][index], batch["queries"][index]), batch["labels"][index]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def accuracy(head: nn.Module, batch: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        logits = head(batch["activities"], batch["queries"])
    return float((logits.argmax(-1) == batch["labels"]).float().mean())


def run_seed(seed: int) -> dict:
    graph, _ = train_joint_parent(seed)
    skill_data = build_skill_data(seed)
    baseline = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    for parameter in graph.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    loads = {}
    last = TICKS - 1
    for load in LOADS:
        train = build(TRAIN, seed, load)
        fit = build(FIT, seed + FIT_OFFSET, load)
        arms = {}
        for arm, use in (
            ("no_state", [last]),
            ("replay", [0, last]),
            ("state_carry", list(range(TICKS))),
        ):
            train_activity = collect_activity(graph, train["ticks"], use).detach()
            fit_activity = collect_activity(graph, fit["ticks"], use).detach()
            head = TickRead(train_activity.shape[-1])
            train_head(
                head,
                {
                    "activities": train_activity,
                    "queries": train["queries"],
                    "labels": train["labels"],
                },
                seed + load,
            )
            arms[arm] = accuracy(
                head,
                {"activities": fit_activity, "queries": fit["queries"], "labels": fit["labels"]},
            )
        loads[str(load)] = arms
    after = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    damage = {skill: max(0.0, baseline[skill] - after[skill]) for skill in baseline}
    return {"seed": seed, "loads": loads, "retention_damage": damage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {}
    for load in LOADS:
        means[str(load)] = {
            arm: sum(row["loads"][str(load)][arm] for row in rows) / len(rows)
            for arm in ("no_state", "replay", "state_carry")
        }
    gaps = {
        str(load): means[str(load)]["replay"] - means[str(load)]["state_carry"] for load in LOADS
    }
    gates = {
        "load_pressure": gaps["3"] >= LOAD_DROP or gaps["5"] >= LOAD_DROP,
        "load1_ok": gaps["1"] <= 0.05,
        "retention": all(max(row["retention_damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = "memory_load_pressure" if all(gates.values()) else "memory_load_comparable"
    payload = {
        "schema": "taiji-u2-wm-load-v1",
        "card": "working-memory load probe: 1 vs 3 vs 5 simultaneous targets",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "loads": list(LOADS),
            "ticks": TICKS,
            "train": TRAIN,
            "fit": FIT,
            "steps": STEPS,
            "load_drop": LOAD_DROP,
        },
        "rows": rows,
        "means": means,
        "gaps": gaps,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "same frozen parent and readout family; query identifies the target index",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "means": means,
                "gaps": gaps,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "memory_load_pressure" else 1


if __name__ == "__main__":
    raise SystemExit(main())
