"""A3c: natural-wording Chinese word-order extension (5 train / 5 held-out, seen chars only) (0 check).

Colours 8; five natural training templates and five natural held-out templates whose
characters all appear in the training templates.  The task is still "the second mentioned
colour" (8 classes).  Character ticks run through the frozen 23.0b parent; positional attention
is compared with mean pooling; replay must reach 0.80 before the unit readout is interpreted.
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

from scripts.training.run_s2_23_0b_joint_20261120 import accuracy as skill_accuracy  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import (
    build_data as build_skill_data,  # noqa: E402
)
from scripts.training.run_s2_23_0b_joint_20261120 import channels, lr_at  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402

from scripts.training.check_report_run_conditions import write_report  # noqa: E402

OUT = ROOT / "reports/taiji_u2_cn_vocab3_20261121.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
TRAIN = 1280
FIT = 320
MAX_LEN = 24
STEPS = 1800
BATCH = 64
LR = 3e-3
COLOURS = ("红", "蓝", "绿", "黄", "紫", "橙", "黑", "白")
TRAIN_TEMPLATES = (
    "请把{c1}和{c2}里后说的那个拿给我",
    "先提{c1}再提{c2}，要后者",
    "{c1}与{c2}中，后提到的那个",
    "我要{c1}和{c2}里后说的",
    "从{c1}、{c2}里挑后说的那个",
)
FIT_TEMPLATES = (
    "请给我{c1}和{c2}中后说的那个",
    "先说{c1}后说{c2}，要后者",
    "{c1}与{c2}中后说的那个",
    "挑{c1}与{c2}里后说的",
    "从{c1}、{c2}里，我要后说的那个",
)


def template_chars(template: str) -> set[str]:
    return set(template.format(c1="红", c2="蓝")) - {"红", "蓝"}


TRAIN_CHARS = set().union(*(template_chars(t) for t in TRAIN_TEMPLATES)) | set("".join(COLOURS))
FIT_CHARS = set().union(*(template_chars(t) for t in FIT_TEMPLATES))
assert FIT_CHARS <= TRAIN_CHARS, (
    "unseen characters in held-out templates",
    sorted(FIT_CHARS - TRAIN_CHARS),
)
VOCAB = sorted(TRAIN_CHARS)
CHAR_INDEX = {char: index for index, char in enumerate(VOCAB)}
PROJECTION = torch.randn(len(VOCAB), 8, generator=torch.Generator().manual_seed(107))
CLASSES = len(COLOURS)
REPLAY_MIN = 0.80
POS_MIN = 0.60
MARGIN = 0.10


def char_tick(char: str) -> torch.Tensor:
    tick = torch.zeros(48)
    index = CHAR_INDEX.get(char)
    if index is not None:
        one_hot = torch.zeros(len(VOCAB))
        one_hot[index] = 1.0
        tick[:8] = one_hot @ PROJECTION
    return tick


def episode(seed: int, template: str) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    first = int(torch.randint(0, CLASSES, (1,), generator=generator))
    second = int(torch.randint(0, CLASSES - 1, (1,), generator=generator))
    if second >= first:
        second += 1
    text = template.format(c1=COLOURS[first], c2=COLOURS[second])
    ticks = [char_tick(char) for char in text[:MAX_LEN]]
    while len(ticks) < MAX_LEN:
        ticks.append(torch.zeros(48))
    return {"ticks": torch.stack(ticks), "label": second}


def build(count: int, seed: int, templates: tuple[str, ...]) -> dict[str, torch.Tensor]:
    rows = [episode(seed + index, templates[index % len(templates)]) for index in range(count)]
    return {
        "ticks": torch.stack([row["ticks"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


class PosAttentionRead(nn.Module):
    def __init__(self, token_dim: int, classes: int, width: int = 64) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(MAX_LEN, token_dim))
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


class PoolRead(nn.Module):
    def __init__(self, token_dim: int, classes: int, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.net(sequence.mean(dim=1))


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


def train_head(head: nn.Module, batch: dict[str, torch.Tensor], key: str, seed: int) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, batch[key].shape[0], (BATCH,), generator=generator)
        loss = loss_fn(head(batch[key][index]), batch["labels"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def accuracy(head: nn.Module, batch: dict[str, torch.Tensor], key: str) -> float:
    with torch.no_grad():
        logits = head(batch[key])
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
    train = build(TRAIN, seed, TRAIN_TEMPLATES)
    fit = build(FIT, seed + FIT_OFFSET, FIT_TEMPLATES)
    train_units = unit_sequence(graph, train["ticks"])
    fit_units = unit_sequence(graph, fit["ticks"])
    results = {}
    head = PosAttentionRead(train_units.shape[-1], CLASSES)
    train_head(head, {"sequence": train_units, "labels": train["labels"]}, "sequence", seed)
    results["pos_attn"] = accuracy(
        head, {"sequence": fit_units, "labels": fit["labels"]}, "sequence"
    )
    head = PoolRead(train_units.shape[-1], CLASSES)
    train_head(head, {"sequence": train_units, "labels": train["labels"]}, "sequence", seed)
    results["mean_pool"] = accuracy(
        head, {"sequence": fit_units, "labels": fit["labels"]}, "sequence"
    )
    replay = PosAttentionRead(train["ticks"].shape[-1], CLASSES)
    train_head(replay, {"sequence": train["ticks"], "labels": train["labels"]}, "sequence", seed)
    results["replay"] = accuracy(
        replay, {"sequence": fit["ticks"], "labels": fit["labels"]}, "sequence"
    )
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
        "pos_attn_level": means["pos_attn"] >= POS_MIN,
        "pos_attn_gain": means["pos_attn"] >= means["mean_pool"] + MARGIN,
        "retention": all(max(row["retention_damage"].values()) <= 0.02 for row in rows),
    }
    if not gates["replay_learnable"]:
        verdict = "presentation_failure"
    elif all(gates.values()):
        verdict = "a3c_verified"
    else:
        verdict = "a3c_partial"
    payload = {
        "schema": "taiji-u2-cn-vocab3-v1",
        "card": "A3c natural-wording Chinese word-order (5 train / 5 held-out, seen chars)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "colors": len(COLOURS),
            "train_templates": len(TRAIN_TEMPLATES),
            "fit_templates": len(FIT_TEMPLATES),
            "train": TRAIN,
            "fit": FIT,
            "fit_offset": FIT_OFFSET,
            "max_len": MAX_LEN,
            "vocab": len(VOCAB),
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "all held-out characters are covered by the training templates; positional attention vs mean pooling; replay on raw features",
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
    return 0 if verdict == "a3c_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
