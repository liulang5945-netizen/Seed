"""Action grounding with scene-as-token-sequence and an instruction tick (0 check).

The flat 48-dim concatenation used in the earlier action probes is replaced by four ticks:
three object ticks (colour x position per object) followed by one high-density Chinese
instruction tick.  A positional-attention readout over the carried unit activities predicts
the action (target body slot x direction, 12 classes).  Arms: joint (4 ticks), scene only
(3 ticks), instruction only (1 tick); replay is the same readout on raw features and must
reach 0.80 before the unit readout is interpreted.  Zero check/final.
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

OUT = ROOT / "reports/taiji_u2_action_seq_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
TRAIN = 512
FIT = 128
STEPS = 1500
BATCH = 64
LR = 3e-3
COLOURS = ("红", "蓝", "绿", "黄")
DIRECTIONS = ("左", "右", "前", "后")
TEMPLATES = ("把{c}球放到{d}边", "{c}球放{d}边", "让{c}球去{d}边")
VOCAB = "把球放到边让去红蓝绿黄左右前后"
CHAR_INDEX = {char: index for index, char in enumerate(VOCAB)}
PROJECTION = torch.randn(len(VOCAB), 8, generator=torch.Generator().manual_seed(61))
CLASSES = 12
REPLAY_MIN = 0.80
JOINT_MIN = 0.60
MARGIN = 0.10


def instruction_bag(colour: str, direction: str) -> torch.Tensor:
    text = TEMPLATES[0].format(c=colour, d=direction)
    bag = torch.zeros(len(VOCAB))
    for char in text:
        if char in CHAR_INDEX:
            bag[CHAR_INDEX[char]] += 1.0
    return bag @ PROJECTION


def episode(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    colours = torch.randperm(4, generator=generator)[:3].tolist()
    positions = torch.randint(0, 4, (3,), generator=generator).tolist()
    target_slot = int(torch.randint(0, 3, (1,), generator=generator))
    direction = DIRECTIONS[int(torch.randint(0, 4, (1,), generator=generator))]
    ticks = []
    for slot in range(3):
        tick = torch.zeros(48)
        tick[colours[slot]] = 1.0
        tick[4 + positions[slot]] = 1.0
        ticks.append(tick)
    instruction = torch.zeros(48)
    instruction[:8] = instruction_bag(COLOURS[colours[target_slot]], direction)
    ticks.append(instruction)
    return {
        "ticks": torch.stack(ticks),
        "label": target_slot * 4 + DIRECTIONS.index(direction),
    }


def build(count: int, seed: int) -> dict[str, torch.Tensor]:
    rows = [episode(seed + index) for index in range(count)]
    return {
        "ticks": torch.stack([row["ticks"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


class PosAttentionRead(nn.Module):
    def __init__(self, token_dim: int, classes: int, width: int = 64, positions: int = 4) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(positions, token_dim))
        self.token = nn.Sequential(nn.Linear(token_dim, width), nn.Tanh())
        self.query1 = nn.Sequential(nn.Linear(width, width), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(width, width), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, classes))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        embed = self.token(sequence + self.pos[: sequence.shape[1]])
        key = embed.mean(dim=1)
        q1 = self.query1(key)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, q1), dim=-1)
        context1 = torch.einsum("bt,btd->bd", scores, embed)
        q2 = self.query2(context1)
        scores2 = torch.softmax(torch.einsum("btd,bd->bt", embed, q2), dim=-1)
        context2 = torch.einsum("bt,btd->bd", scores2, embed)
        return self.out(torch.cat([context2, q1], dim=-1))


def unit_sequence(graph, ticks: torch.Tensor) -> torch.Tensor:
    state = graph.initial_state()
    activities = []
    for tick_index in range(ticks.shape[1]):
        gates = torch.ones(ticks.shape[0], graph.units.n_units)
        with torch.no_grad():
            result = graph(channels(ticks[:, tick_index]), gates, state)
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
        index = torch.randint(0, batch["sequence"].shape[0], (BATCH,), generator=generator)
        loss = loss_fn(head(batch["sequence"][index]), batch["labels"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def accuracy(head: nn.Module, batch: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        logits = head(batch["sequence"])
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
    train_units = unit_sequence(graph, train["ticks"])
    fit_units = unit_sequence(graph, fit["ticks"])
    results = {}
    for arm, train_slice, fit_slice in (
        ("joint", slice(None), slice(None)),
        ("scene_only", slice(0, 3), slice(0, 3)),
        ("instruction_only", slice(3, 4), slice(3, 4)),
    ):
        head = PosAttentionRead(train_units.shape[-1], CLASSES)
        train_head(head, {"sequence": train_units[:, train_slice], "labels": train["labels"]}, seed)
        results[arm] = accuracy(
            head, {"sequence": fit_units[:, fit_slice], "labels": fit["labels"]}
        )
    replay = PosAttentionRead(train["ticks"].shape[-1], CLASSES)
    train_head(replay, {"sequence": train["ticks"], "labels": train["labels"]}, seed)
    results["replay"] = accuracy(replay, {"sequence": fit["ticks"], "labels": fit["labels"]})
    after = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    damage = {skill: max(0.0, baseline[skill] - after[skill]) for skill in baseline}
    return {"seed": seed, "arms": results, "retention_damage": damage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {arm: sum(row["arms"][arm] for row in rows) / len(rows) for arm in rows[0]["arms"]}
    best_single = max(means["scene_only"], means["instruction_only"])
    gates = {
        "replay_learnable": means["replay"] >= REPLAY_MIN,
        "joint_gain": means["joint"] >= best_single + MARGIN,
        "joint_level": means["joint"] >= JOINT_MIN,
        "retention": all(max(row["retention_damage"].values()) <= 0.02 for row in rows),
    }
    verdict = "action_seq_verified" if all(gates.values()) else "action_seq_partial"
    payload = {
        "schema": "taiji-u2-action-seq-v1",
        "card": "action grounding with scene ticks + instruction tick and positional attention",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "classes": CLASSES,
            "train": TRAIN,
            "fit": FIT,
            "fit_offset": FIT_OFFSET,
            "steps": STEPS,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "three object ticks plus one instruction tick; positional-attention readout; replay uses raw features",
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
    return 0 if verdict == "action_seq_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
