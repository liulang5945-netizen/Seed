"""Chinese instruction grounding probe: held-out wording template (0 check).

A small Chinese vocabulary builds four wording templates for the same action ("colour" x
"direction", 16 classes).  The model sees a character one-hot sequence projected to 48 dims,
runs the frozen 23.0b parent, and a readout predicts the action class.  Training uses
templates 0-2; fit uses the held-out template 3 (unseen wording) with disjoint fit seeds.
Arms: correct instruction, shuffled-label control, null instruction.  A replay/"oracle"
upper bound encodes the action symbolically to check task learnability.
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

OUT = ROOT / "reports/taiji_u2_chinese_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
TRAIN_VARIANTS = 12
FIT_VARIANTS = 16
STEPS = 1500
BATCH = 64
LR = 3e-3
INPUT_DIM = 48
COLOURS = ("红", "蓝", "绿", "黄")
DIRECTIONS = ("左", "右", "前", "后")
TEMPLATES = (
    "把{c}球放到{d}边",
    "{c}球放{d}边",
    "让{c}球去{d}边",
    "把{c}球放去{d}边",  # held-out wording, same characters
)
VOCAB = "把球放到边让去你要移红蓝绿黄左右前后"
CHAR_INDEX = {char: index for index, char in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PROJECTION = torch.randn(VOCAB_SIZE, INPUT_DIM, generator=torch.Generator().manual_seed(12345))
REPLAY_MIN = 0.60
NULL_MAX = 0.30


def encode(text: str) -> torch.Tensor:
    ids = [CHAR_INDEX[char] for char in text if char in CHAR_INDEX]
    bag = torch.zeros(VOCAB_SIZE)
    for index in ids[:16]:
        bag[index] += 1.0
    return bag @ PROJECTION


def make_examples(seed: int, template_ids: range, variants: int) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    rows = []
    combos = [(colour, direction) for colour in COLOURS for direction in DIRECTIONS]
    for variant in range(variants):
        order = torch.randperm(len(combos), generator=generator).tolist()
        for position in order:
            colour, direction = combos[position]
            template = TEMPLATES[template_ids[variant % len(template_ids)]]
            text = template.format(c=colour, d=direction)
            rows.append(
                {
                    "features": encode(text),
                    "label": COLOURS.index(colour) * 4 + DIRECTIONS.index(direction),
                }
            )
    return rows


def build(rows: list[dict], shuffle_seed: int | None = None) -> dict[str, torch.Tensor]:
    features = torch.stack([row["features"] for row in rows]).float()
    labels = torch.tensor([row["label"] for row in rows])
    if shuffle_seed is not None:
        generator = torch.Generator().manual_seed(shuffle_seed)
        labels = labels[torch.randperm(len(labels), generator=generator)]
    return {"features": features, "labels": labels}


class Read(nn.Module):
    def __init__(self, input_dim: int, classes: int, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, classes),
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
    train_rows = make_examples(seed, range(3), TRAIN_VARIANTS)
    fit_rows = make_examples(seed + FIT_OFFSET, range(3, 4), FIT_VARIANTS)
    train = build(train_rows)
    fit = build(fit_rows)

    def activity(features: torch.Tensor) -> torch.Tensor:
        gates = torch.ones(features.shape[0], graph.units.n_units)
        with torch.no_grad():
            result = graph(channels(features), gates, graph.initial_state())
        return result["unit_result"]["activity"]

    train_activity = activity(train["features"])
    fit_activity = activity(fit["features"])
    arms = {}
    for arm, labels in (
        ("correct", train["labels"]),
        ("shuffled", build(train_rows, shuffle_seed=seed + 3)["labels"]),
        ("null", torch.zeros_like(train["labels"])),
    ):
        head = Read(train_activity.shape[-1], 16)
        if arm == "null":
            train_head(head, {"features": torch.zeros_like(train_activity), "labels": labels}, seed)
        else:
            train_head(head, {"features": train_activity, "labels": labels}, seed)
        arms[arm] = accuracy(head, {"features": fit_activity, "labels": fit["labels"]})
    # Symbolic replay upper bound on the same held-out wording.
    replay = Read(INPUT_DIM, 16)
    train_head(replay, {"features": train["features"], "labels": train["labels"]}, seed)
    arms["replay"] = accuracy(replay, {"features": fit["features"], "labels": fit["labels"]})
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
    means = {
        arm: sum(row["arms"][arm] for row in rows) / len(rows)
        for arm in ("correct", "shuffled", "null", "replay")
    }
    gates = {
        "replay_learnable": means["replay"] >= REPLAY_MIN,
        "correct_better_than_control": means["correct"] >= means["shuffled"] + 0.20
        and means["correct"] >= means["null"] + 0.20,
        "retention": all(max(row["retention_damage"].values()) <= 0.02 for row in rows),
    }
    verdict = (
        "chinese_grounding_verified"
        if all(gates.values())
        else ("chinese_probe_partial" if gates["replay_learnable"] else "chinese_probe_invalid")
    )
    payload = {
        "schema": "taiji-u2-chinese-v1",
        "card": "Chinese instruction grounding with held-out wording template",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "train_templates": list(range(3)),
            "fit_template": 3,
            "train_variants": TRAIN_VARIANTS,
            "fit_variants": FIT_VARIANTS,
            "fit_offset": FIT_OFFSET,
            "classes": 16,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "character one-hot projected to 48 dims; held-out wording template; replay arm reads the same projected features without the unit graph",
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
    return 0 if verdict == "chinese_grounding_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
