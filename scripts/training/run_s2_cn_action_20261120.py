"""Chinese instruction x scene action fusion probe (0 check).

Scene: three bodies with colours and positions (presented in random order).  Instruction:
"把{colour}球放到{direction}边".  The label is the *target body index* (which body matches the
named colour) crossed with the direction (12 classes): the instruction names the colour and
direction but only the scene reveals which body slot carries that colour.  Arms:
instruction-only (scene zeroed), scene-only (instruction zeroed), joint.  Joint must beat the
best single arm by >= 0.10 and reach >= 0.60.  Zero check/final.
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

OUT = ROOT / "reports/taiji_u2_cn_action_20261120.json"
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
VOCAB_SIZE = len(VOCAB)
PROJECTION = torch.randn(VOCAB_SIZE, 8, generator=torch.Generator().manual_seed(7))
CLASSES = 12
JOINT_MIN = 0.60
MARGIN = 0.10


def instruction_bag(text: str) -> torch.Tensor:
    bag = torch.zeros(VOCAB_SIZE)
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
    template = TEMPLATES[int(torch.randint(0, len(TEMPLATES), (1,), generator=generator))]
    text = template.format(c=COLOURS[colours[target_slot]], d=direction)
    scene = torch.zeros(3, 8)
    for slot in range(3):
        scene[slot, colours[slot]] = 1.0
        scene[slot, 4 + positions[slot]] = 1.0
    instruction = instruction_bag(text)
    features = torch.zeros(48)
    features[:24] = scene.reshape(-1)
    features[24:32] = instruction
    return {
        "features": features,
        "label": target_slot * 4 + DIRECTIONS.index(direction),
    }


def build(count: int, seed: int) -> dict[str, torch.Tensor]:
    rows = [episode(seed + index) for index in range(count)]
    return {
        "features": torch.stack([row["features"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


class Read(nn.Module):
    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, CLASSES),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def train_head(head: nn.Module, batch: dict[str, torch.Tensor], seed: int) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, batch["features"].shape[0], (BATCH,), generator=generator)
        loss = loss_fn(head(batch["features"][index]), batch["labels"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def accuracy(head: nn.Module, batch: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        logits = head(batch["features"])
    return float((logits.argmax(-1) == batch["labels"]).float().mean())


def zero_scene(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    features = batch["features"].clone()
    features[:, :24] = 0.0
    return {**batch, "features": features}


def zero_instruction(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    features = batch["features"].clone()
    features[:, 24:] = 0.0
    return {**batch, "features": features}


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

    def activity(features: torch.Tensor) -> torch.Tensor:
        gates = torch.ones(features.shape[0], graph.units.n_units)
        with torch.no_grad():
            result = graph(channels(features), gates, graph.initial_state())
        return result["unit_result"]["activity"]

    arms = {}
    for name, train_batch, fit_batch in (
        ("instruction_only", zero_scene(train), zero_scene(fit)),
        ("scene_only", zero_instruction(train), zero_instruction(fit)),
        ("joint", train, fit),
    ):
        head = Read()
        train_head(
            head,
            {"features": activity(train_batch["features"]), "labels": train_batch["labels"]},
            seed,
        )
        arms[name] = accuracy(
            head, {"features": activity(fit_batch["features"]), "labels": fit_batch["labels"]}
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
    best_single = max(means["instruction_only"], means["scene_only"])
    gates = {
        "joint_fusion": means["joint"] >= best_single + MARGIN and means["joint"] >= JOINT_MIN,
        "retention": all(max(row["retention_damage"].values()) <= 0.02 for row in rows),
    }
    verdict = "multimodal_action_verified" if all(gates.values()) else "multimodal_action_partial"
    payload = {
        "schema": "taiji-u2-cn-action-v1",
        "card": "Chinese instruction x scene action fusion",
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
            "joint_min": JOINT_MIN,
            "margin": MARGIN,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "scene and instruction are concatenated into one 48-dim tick; label needs both (body slot from scene, colour/direction from instruction)",
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
    return 0 if verdict == "multimodal_action_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
