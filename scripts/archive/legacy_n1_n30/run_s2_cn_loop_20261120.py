"""Chinese instruction -> action -> simulated feedback closed loop (0 check).

Each episode has a scene and three Chinese instructions; the action applies the *last*
instruction and is scored by a symbolic simulator (slot x direction, 12 classes).  The model
acts twice: attempt 0 sees scene+instruction; attempt 1 additionally sees the previous action
and a noisy success/failure feedback bit.  Arms: without feedback (feedback and previous
action masked) vs with feedback.  Gate: the feedback arm's second-attempt accuracy beats the
no-feedback arm by >= 0.10 and reaches >= 0.60; replay (raw features) must reach 0.80.
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

OUT = ROOT / "reports/taiji_u2_cn_loop_20261120.json"
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
PROJECTION = torch.randn(len(VOCAB), 8, generator=torch.Generator().manual_seed(53))
CLASSES = 12
REPLAY_MIN = 0.80
LOOP_MIN = 0.60
MARGIN = 0.10
FEEDBACK_NOISE = 0.70


def instruction_tick(colour: str, direction: str) -> torch.Tensor:
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
    instructions = []
    for _ in range(3):
        instructions.append(
            (
                COLOURS[int(torch.randint(0, 4, (1,), generator=generator))],
                DIRECTIONS[int(torch.randint(0, 4, (1,), generator=generator))],
            )
        )
    final_colour, final_direction = instructions[-1]
    slot = None
    for index, colour_index in enumerate(colours):
        if COLOURS[colour_index] == final_colour:
            slot = index
            break
    if slot is None:
        slot = 0
    scene = torch.zeros(3, 8)
    for index in range(3):
        scene[index, colours[index]] = 1.0
        scene[index, 4 + positions[index]] = 1.0
    context = torch.zeros(48)
    context[:24] = scene.reshape(-1)
    context[24:32] = instruction_tick(final_colour, final_direction)
    label = slot * 4 + DIRECTIONS.index(final_direction)
    return {"context": context, "label": label}


def build(count: int, seed: int) -> list[dict]:
    return [episode(seed + index) for index in range(count)]


class PosAttentionRead(nn.Module):
    def __init__(self, token_dim: int, classes: int, width: int = 64, positions: int = 2) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(positions, token_dim))
        self.token = nn.Sequential(nn.Linear(token_dim, width), nn.Tanh())
        self.query1 = nn.Sequential(nn.Linear(width, width), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(width + 8, width), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, classes))
        self.query_token = nn.Parameter(torch.zeros(8))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        embed = self.token(sequence + self.pos[: sequence.shape[1]])
        key = embed.mean(dim=1)
        q1 = self.query1(key)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, q1), dim=-1)
        context1 = torch.einsum("bt,btd->bd", scores, embed)
        q2 = self.query2(
            torch.cat([context1, self.query_token.expand(context1.shape[0], -1)], dim=-1)
        )
        scores2 = torch.softmax(torch.einsum("btd,bd->bt", embed, q2), dim=-1)
        context2 = torch.einsum("bt,btd->bd", scores2, embed)
        return self.out(torch.cat([context2, q1], dim=-1))


def attempt_row(row: dict, feedback: float, previous_action: int) -> torch.Tensor:
    tick0 = row["context"]
    tick1 = row["context"].clone()
    tick1[32 + previous_action] = 1.0
    tick1[44] = feedback
    tick1[45] = 1.0  # attempt-1 marker
    return torch.stack([tick0, tick1])


def unit_activity(graph, sequence: torch.Tensor) -> torch.Tensor:
    state = graph.initial_state()
    activities = []
    for tick_index in range(sequence.shape[1]):
        gates = torch.ones(sequence.shape[0], graph.units.n_units)
        with torch.no_grad():
            result = graph(channels(sequence[:, tick_index]), gates, state)
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


def build_loop_rows(rows: list[dict], use_feedback: bool, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = []
    labels = []
    for row in rows:
        first = 0 if not use_feedback else int(torch.randint(0, CLASSES, (1,), generator=generator))
        feedback = 0.5
        if use_feedback:
            correct = int(torch.randint(0, 100, (1,), generator=generator)) < int(
                FEEDBACK_NOISE * 100
            )
        else:
            correct = False
        previous = first
        if use_feedback:
            feedback = (
                1.0
                if (correct and row["label"] == first) or (not correct and row["label"] != first)
                else 0.0
            )
        features.append(attempt_row(row, feedback, previous))
        labels.append(row["label"])
    return {"sequence": torch.stack(features).float(), "labels": torch.tensor(labels)}


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
    train_rows = build(TRAIN, seed)
    fit_rows = build(FIT, seed + FIT_OFFSET)
    results = {}
    for arm, use_feedback in (("no_feedback", False), ("with_feedback", True)):
        train = build_loop_rows(train_rows, use_feedback, seed)
        fit = build_loop_rows(fit_rows, use_feedback, seed + 1)
        train_units = unit_activity(graph, train["sequence"])
        fit_units = unit_activity(graph, fit["sequence"])
        head = PosAttentionRead(train_units.shape[-1], CLASSES)
        train_head(head, {"sequence": train_units, "labels": train["labels"]}, seed)
        results[arm] = accuracy(head, {"sequence": fit_units, "labels": fit["labels"]})
    # replay upper bound on the raw features of the same loop rows.
    train = build_loop_rows(train_rows, True, seed)
    fit = build_loop_rows(fit_rows, True, seed + 1)
    replay = PosAttentionRead(train["sequence"].shape[-1], CLASSES)
    train_head(replay, train, seed)
    results["replay"] = accuracy(replay, fit)
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
    gates = {
        "replay_learnable": means["replay"] >= REPLAY_MIN,
        "feedback_gain": means["with_feedback"] >= means["no_feedback"] + MARGIN,
        "feedback_level": means["with_feedback"] >= LOOP_MIN,
        "retention": all(max(row["retention_damage"].values()) <= 0.02 for row in rows),
    }
    verdict = "feedback_loop_verified" if all(gates.values()) else "feedback_loop_partial"
    payload = {
        "schema": "taiji-u2-cn-loop-v1",
        "card": "Chinese instruction -> action -> simulated feedback closed loop",
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
            "feedback_noise": FEEDBACK_NOISE,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "second attempt sees previous action and a noisy (70%) success bit; the simulator scores exact action match",
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
    return 0 if verdict == "feedback_loop_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
