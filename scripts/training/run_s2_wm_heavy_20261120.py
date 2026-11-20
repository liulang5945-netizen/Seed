"""Heavy working-memory load probe: 10 targets over 20 ticks and serial-order recall (0 check).

Task A (load): t0 shows 10 target bodies; 19 distractor ticks; the final query asks the
position of the k-th target.  Task B (order): 5 targets appear one per tick in the first five
ticks; the final query asks the position of the k-th target in appearance order.
Arms: no_state (final tick only), replay (target ticks + final), state_carry (all ticks).
Pressure shows up as replay - carry >= 0.15.  Zero check/final.
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

OUT = ROOT / "reports/taiji_u2_wm_heavy_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
TRAIN = 384
FIT = 128
BATCH = 32
STEPS = 1500
LR = 3e-3
PRESSURE = 0.15
FIT_OFFSET = 10_000_000
RETENTION_MAX = 0.02
LOAD_BODIES = 10
LOAD_TICKS = 20
ORDER_TARGETS = 5
ORDER_TICKS = 20
DIM = 128


def make_features(bodies: list[tuple[int, int]], tick_index: int, ticks: int) -> torch.Tensor:
    features = torch.zeros(DIM)
    for body, (colour, position) in enumerate(bodies):
        features[body * 8 + 2 + colour] = 1.0
        features[body * 8 + 6 + position] = 1.0
    features[100 + (tick_index % 10)] = 1.0
    features[110 + min(ticks, 10) - 1] = 1.0
    return features


def load_episode(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    colours = torch.randperm(10, generator=generator)[:LOAD_BODIES].tolist()
    positions = [int(torch.randint(0, 3, (1,), generator=generator)) for _ in range(LOAD_BODIES)]
    ticks = []
    for tick_index in range(LOAD_TICKS):
        if tick_index == 0:
            bodies = list(zip(colours, positions, strict=True))
        else:
            bodies = [
                (
                    int(torch.randint(0, 6, (1,), generator=generator)),
                    int(torch.randint(0, 3, (1,), generator=generator)),
                )
                for _ in range(LOAD_BODIES)
            ]
        ticks.append(make_features(bodies, tick_index, LOAD_TICKS))
    index = int(torch.randint(0, LOAD_BODIES, (1,), generator=generator))
    query = torch.zeros(11)
    query[10] = 1.0
    query[index] = 1.0
    return {"ticks": torch.stack(ticks), "query": query, "label": positions[index]}


def order_episode(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    colours = torch.randperm(10, generator=generator)[:ORDER_TARGETS].tolist()
    positions = [int(torch.randint(0, 3, (1,), generator=generator)) for _ in range(ORDER_TARGETS)]
    ticks = []
    for tick_index in range(ORDER_TICKS):
        if tick_index < ORDER_TARGETS:
            bodies = [(colours[tick_index], positions[tick_index])]
            bodies += [
                (
                    int(torch.randint(0, 6, (1,), generator=generator)),
                    int(torch.randint(0, 3, (1,), generator=generator)),
                )
                for _ in range(4)
            ]
        else:
            bodies = [
                (
                    int(torch.randint(0, 6, (1,), generator=generator)),
                    int(torch.randint(0, 3, (1,), generator=generator)),
                )
                for _ in range(5)
            ]
        ticks.append(make_features(bodies, tick_index, ORDER_TICKS))
    index = int(torch.randint(0, ORDER_TARGETS, (1,), generator=generator))
    query = torch.zeros(11)
    query[10] = 1.0
    query[5 + index] = 1.0
    return {"ticks": torch.stack(ticks), "query": query, "label": positions[index]}


def build(count: int, seed: int, task: str) -> dict[str, torch.Tensor]:
    maker = load_episode if task == "load" else order_episode
    rows = [maker(seed + index) for index in range(count)]
    return {
        "ticks": torch.stack([row["ticks"] for row in rows]).float(),
        "queries": torch.stack([row["query"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


class TickRead(nn.Module):
    def __init__(self, token_dim: int, width: int = 64) -> None:
        super().__init__()
        self.query1 = nn.Sequential(nn.Linear(11, width), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(width + 11, width), nn.Tanh())
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


def collect_final_reset(graph, ticks: torch.Tensor) -> torch.Tensor:
    gates = torch.ones(ticks.shape[0], graph.units.n_units)
    with torch.no_grad():
        result = graph(channels(ticks[:, -1]), gates, graph.initial_state())
    return result["unit_result"]["activity"].unsqueeze(1)


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
    tasks = {}
    for task, target_ticks, ticks in (
        ("load", [0], LOAD_TICKS),
        ("order", list(range(ORDER_TARGETS)), ORDER_TICKS),
    ):
        train = build(TRAIN, seed, task)
        fit = build(FIT, seed + FIT_OFFSET, task)
        last = ticks - 1
        arms = {}
        for arm, use in (
            ("stateless_reset", [last]),
            ("carry_last", [last]),
            ("replay", [*target_ticks, last]),
            ("state_carry", list(range(ticks))),
        ):
            if arm == "stateless_reset":
                train_activity = collect_final_reset(graph, train["ticks"]).detach()
                fit_activity = collect_final_reset(graph, fit["ticks"]).detach()
            else:
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
                seed,
            )
            arms[arm] = accuracy(
                head,
                {"activities": fit_activity, "queries": fit["queries"], "labels": fit["labels"]},
            )
        tasks[task] = arms
    after = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    damage = {skill: max(0.0, baseline[skill] - after[skill]) for skill in baseline}
    return {"seed": seed, "tasks": tasks, "retention_damage": damage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {
        task: {
            arm: sum(row["tasks"][task][arm] for row in rows) / len(rows)
            for arm in ("stateless_reset", "carry_last", "replay", "state_carry")
        }
        for task in ("load", "order")
    }
    gaps = {task: means[task]["replay"] - means[task]["state_carry"] for task in means}
    gates = {
        "pressure": gaps["load"] >= PRESSURE or gaps["order"] >= PRESSURE,
        "retention": all(max(row["retention_damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = "memory_load_pressure" if all(gates.values()) else "memory_load_comparable"
    payload = {
        "schema": "taiji-u2-wm-heavy-v1",
        "card": "heavy working-memory load: 10 targets/20 ticks and serial-order recall",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "load_bodies": LOAD_BODIES,
            "load_ticks": LOAD_TICKS,
            "order_targets": ORDER_TARGETS,
            "order_ticks": ORDER_TICKS,
            "pressure": PRESSURE,
        },
        "rows": rows,
        "means": means,
        "gaps": gaps,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "synthetic 10-body scenes; replay arm gets the target ticks and the final tick",
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
