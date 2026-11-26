"""Chinese continuous-task probe: holding the first instruction over 5 ticks (0 check).

Each tick carries one high-density Chinese instruction (character bag projected into the
48-dim input).  t0 is the target instruction, t1-t3 are distractor instructions, t4 is the
query; the label is the target instruction's colour x direction (16 classes).  Arms:
stateless_reset (query tick only, state reset), replay (target + query), state_carry (all
ticks), all with the positional-attention readout.  Replay must reach 0.80 before the carry
comparison is interpreted.  Zero check/final.
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

OUT = ROOT / "reports/taiji_u2_cn_continuous_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
TRAIN = 512
FIT = 128
TICKS = 5
STEPS = 1500
BATCH = 32
LR = 3e-3
COLOURS = ("红", "蓝", "绿", "黄")
DIRECTIONS = ("左", "右", "前", "后")
TEMPLATES = ("把{c}球放到{d}边", "{c}球放{d}边", "让{c}球去{d}边")
VOCAB = "把球放到边让去红蓝绿黄左右前后"
CHAR_INDEX = {char: index for index, char in enumerate(VOCAB)}
PROJECTION = torch.randn(len(VOCAB), 48, generator=torch.Generator().manual_seed(41))
REPLAY_MIN = 0.80
CARRY_MIN = 0.60
TOL = 0.10


def instruction_tick(colour: str, direction: str, template_index: int) -> torch.Tensor:
    text = TEMPLATES[template_index].format(c=colour, d=direction)
    bag = torch.zeros(len(VOCAB))
    for char in text:
        if char in CHAR_INDEX:
            bag[CHAR_INDEX[char]] += 1.0
    return bag @ PROJECTION


def episode(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    colour = COLOURS[int(torch.randint(0, 4, (1,), generator=generator))]
    direction = DIRECTIONS[int(torch.randint(0, 4, (1,), generator=generator))]
    template = int(torch.randint(0, 3, (1,), generator=generator))
    ticks = [instruction_tick(colour, direction, template)]
    for _ in range(TICKS - 1):
        distractor_colour = COLOURS[int(torch.randint(0, 4, (1,), generator=generator))]
        distractor_direction = DIRECTIONS[int(torch.randint(0, 4, (1,), generator=generator))]
        ticks.append(
            instruction_tick(
                distractor_colour,
                distractor_direction,
                int(torch.randint(0, 3, (1,), generator=generator)),
            )
        )
    query = torch.zeros(8)
    query[7] = 1.0
    return {
        "ticks": torch.stack(ticks),
        "queries": query,
        "label": COLOURS.index(colour) * 4 + DIRECTIONS.index(direction),
    }


def build(count: int, seed: int) -> dict[str, torch.Tensor]:
    rows = [episode(seed + index) for index in range(count)]
    return {
        "ticks": torch.stack([row["ticks"] for row in rows]).float(),
        "queries": torch.stack([row["queries"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


class PosAttentionRead(nn.Module):
    def __init__(
        self, token_dim: int, classes: int, width: int = 64, positions: int = TICKS
    ) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(positions, token_dim))
        self.token = nn.Sequential(nn.Linear(token_dim, width), nn.Tanh())
        self.query1 = nn.Sequential(nn.Linear(width, width), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(width + 8, width), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, classes))

    def forward(self, sequence: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        embed = self.token(sequence + self.pos[: sequence.shape[1]])
        key = embed.mean(dim=1)
        q1 = self.query1(key)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, q1), dim=-1)
        context1 = torch.einsum("bt,btd->bd", scores, embed)
        q2 = self.query2(torch.cat([context1, query], dim=-1))
        scores2 = torch.softmax(torch.einsum("btd,bd->bt", embed, q2), dim=-1)
        context2 = torch.einsum("bt,btd->bd", scores2, embed)
        return self.out(torch.cat([context2, q1], dim=-1))


def collect_activity(
    graph, ticks: torch.Tensor, use: list[int], reset: bool = False
) -> torch.Tensor:
    state = graph.initial_state()
    activities = []
    for tick_index in range(ticks.shape[1]):
        if reset and tick_index != ticks.shape[1] - 1:
            continue
        gates = torch.ones(ticks.shape[0], graph.units.n_units)
        with torch.no_grad():
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
    train = build(TRAIN, seed)
    fit = build(FIT, seed + FIT_OFFSET)
    last = TICKS - 1
    arms = {}
    for arm in ("stateless_reset", "replay", "state_carry"):
        if arm == "stateless_reset":
            train_activity = collect_activity(graph, train["ticks"], [last], reset=True).detach()
            fit_activity = collect_activity(graph, fit["ticks"], [last], reset=True).detach()
        else:
            use = [0, last] if arm == "replay" else list(range(TICKS))
            train_activity = collect_activity(graph, train["ticks"], use).detach()
            fit_activity = collect_activity(graph, fit["ticks"], use).detach()
        head = PosAttentionRead(train_activity.shape[-1], 16, positions=train_activity.shape[1])
        train_head(
            head,
            {"activities": train_activity, "queries": train["queries"], "labels": train["labels"]},
            seed,
        )
        arms[arm] = accuracy(
            head, {"activities": fit_activity, "queries": fit["queries"], "labels": fit["labels"]}
        )
    after = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    damage = {skill: max(0.0, baseline[skill] - after[skill]) for skill in baseline}
    return {"seed": seed, "arms": arms, "retention_damage": damage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {arm: sum(row["arms"][arm] for row in rows) / len(rows) for arm in rows[0]["arms"]}
    gates = {
        "replay_learnable": means["replay"] >= REPLAY_MIN,
        "carry_learnable": means["state_carry"] >= CARRY_MIN,
        "near_replay": means["state_carry"] >= means["replay"] - TOL,
        "retention": all(max(row["retention_damage"].values()) <= 0.02 for row in rows),
    }
    verdict = "chinese_continuous_verified" if all(gates.values()) else "chinese_continuous_partial"
    payload = {
        "schema": "taiji-u2-cn-continuous-v1",
        "card": "Chinese continuous task: hold the first instruction over 5 ticks",
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
            "classes": 16,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "each tick is one high-density Chinese instruction bag; positional-attention readout over per-tick unit activities",
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
    return 0 if verdict == "chinese_continuous_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
