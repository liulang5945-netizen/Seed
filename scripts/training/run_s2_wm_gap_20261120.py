"""Working-memory retention decay probe: gaps 5/10/20/40 ticks (0 check).

Same 5-tick protocol as the working-memory card, but the number of distractor ticks between
the target (t0) and the query (last tick) is swept over 5/10/20/40.  Arms: no_state (query
tick only), replay (target + query, upper bound), state_carry (all ticks, carried unit
state).  Reports the largest gap where state_carry stays within 0.05 of replay and >= 0.60.
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
from scripts.training.run_s2_23_0b_joint_20261120 import accuracy as skill_accuracy  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import (
    build_data as build_skill_data,  # noqa: E402
)
from scripts.training.run_s2_23_0b_joint_20261120 import channels, lr_at  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402

OUT = ROOT / "reports/taiji_u2_wm_gap_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
GAPS = (5, 10, 20, 40)
TRAIN = 512
FIT = 128
BODY = 3
BATCH = 32
STEPS = 1500
LR = 3e-3
KEEP_MIN = 0.60
TOL = 0.05
RETENTION_MAX = 0.02


def episode(seed: int, gap: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target_colour = int(torch.randint(0, 6, (1,), generator=generator))
    target_position = int(torch.randint(0, 3, (1,), generator=generator))
    ticks = []
    for tick_index in range(gap):
        features = torch.zeros(64)
        for body in range(BODY):
            if tick_index == 0 and body == 0:
                colour, position = target_colour, target_position
            else:
                colour = int(torch.randint(0, 6, (1,), generator=generator))
                position = int(torch.randint(0, 3, (1,), generator=generator))
            features[body * 8 + 2 + colour] = 1.0
            features[body * 8 + 6 + position] = 1.0
        features[24 + (tick_index % 5)] = 1.0
        features[29 + (tick_index // 5) % 6] = 1.0
        ticks.append(features)
    query = torch.zeros(8)
    query[7] = 1.0
    return {"ticks": torch.stack(ticks), "query": query, "label": target_position}


def build(count: int, seed: int, gap: int) -> dict[str, torch.Tensor]:
    rows = [episode(seed + index, gap) for index in range(count)]
    return {
        "ticks": torch.stack([row["ticks"] for row in rows]).float(),
        "queries": torch.stack([row["query"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


class TickRead(nn.Module):
    def __init__(self, token_dim: int, width: int = 64) -> None:
        super().__init__()
        self.query1 = nn.Sequential(nn.Linear(8, width), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(width + 8, width), nn.Tanh())
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
    batch = ticks.shape[0]
    state = graph.initial_state()
    activities = []
    for tick_index in range(ticks.shape[1]):
        gates = torch.ones(batch, graph.units.n_units)
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
    gaps = {}
    for gap in GAPS:
        train = build(TRAIN, seed, gap)
        fit = build(FIT, seed + 1, gap)
        last = gap - 1
        arms = {}
        for arm, use in (
            ("no_state", [last]),
            ("replay", [0, last]),
            ("state_carry", list(range(gap))),
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
                seed + gap,
            )
            arms[arm] = accuracy(
                head,
                {"activities": fit_activity, "queries": fit["queries"], "labels": fit["labels"]},
            )
        gaps[str(gap)] = arms
    after = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    damage = {skill: max(0.0, baseline[skill] - after[skill]) for skill in baseline}
    return {"seed": seed, "gaps": gaps, "retention_damage": damage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {}
    for gap in GAPS:
        means[str(gap)] = {
            arm: sum(row["gaps"][str(gap)][arm] for row in rows) / len(rows)
            for arm in ("no_state", "replay", "state_carry")
        }
    boundary = 0
    for gap in GAPS:
        carry = means[str(gap)]["state_carry"]
        replay = means[str(gap)]["replay"]
        if carry >= KEEP_MIN and carry >= replay - TOL:
            boundary = gap
    gates = {
        "boundary_found": boundary > 0,
        "retention": all(max(row["retention_damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = "memory_decay_boundary" if all(gates.values()) else "memory_decay_partial"
    payload = {
        "schema": "taiji-u2-wm-gap-v1",
        "card": "working-memory retention decay over gaps 5/10/20/40",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "gaps": list(GAPS),
            "train": TRAIN,
            "fit": FIT,
            "steps": STEPS,
        },
        "rows": rows,
        "means": means,
        "boundary": boundary,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "same frozen parent and readout as the working-memory card; gap is distractor ticks between target and query",
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
                "boundary": boundary,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "memory_decay_boundary" else 1


if __name__ == "__main__":
    raise SystemExit(main())
