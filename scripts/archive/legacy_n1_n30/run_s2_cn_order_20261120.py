"""Chinese word-order sensitivity probe: bag-of-chars vs character-sequence ticks (0 check).

Instructions name two colours and ask for the *second mentioned* one ("{c1}和{c2}取第二个"),
so the label depends on token order and a bag-of-characters encoding cannot recover it.
Encodings: bag projection (order-insensitive) and per-character ticks fed through the frozen
23.0b parent with carried state (order-sensitive).  Training uses three templates, fit uses a
held-out template; fit seeds are disjoint.  Gate: sequence beats bag by >= 0.10 on the
held-out wording.  Zero check/final.
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

OUT = ROOT / "reports/taiji_u2_cn_order_20261120.json"
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
TEMPLATES = (
    "{c1}和{c2}取第二个",
    "{c1}与{c2}取第二个",
    "{c1}跟{c2}取第二个",
    "从{c1}{c2}取第二个",  # held-out wording
)
VOCAB = "和与跟取第二个从红蓝绿黄"
CHAR_INDEX = {char: index for index, char in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PROJECTION = torch.randn(VOCAB_SIZE, INPUT_DIM, generator=torch.Generator().manual_seed(20261120))
MAX_LEN = 12


def char_ids(text: str) -> list[int]:
    return [CHAR_INDEX[char] for char in text if char in CHAR_INDEX]


def bag_features(text: str) -> torch.Tensor:
    bag = torch.zeros(VOCAB_SIZE)
    for index in char_ids(text):
        bag[index] += 1.0
    return bag @ PROJECTION


def seq_features(text: str) -> torch.Tensor:
    ids = char_ids(text)[:MAX_LEN]
    table = torch.zeros(MAX_LEN, VOCAB_SIZE)
    for position, index in enumerate(ids):
        table[position, index] = 1.0
    return table @ PROJECTION


def make_examples(seed: int, template_ids: range, variants: int) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    rows = []
    pairs = [(first, second) for first in COLOURS for second in COLOURS if first != second]
    for variant in range(variants):
        order = torch.randperm(len(pairs), generator=generator).tolist()
        for position in order:
            first, second = pairs[position]
            text = TEMPLATES[template_ids[variant % len(template_ids)]].format(c1=first, c2=second)
            rows.append(
                {
                    "text": text,
                    "bag": bag_features(text),
                    "seq": seq_features(text),
                    "label": COLOURS.index(second),
                }
            )
    return rows


def build(rows: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "bag": torch.stack([row["bag"] for row in rows]).float(),
        "seq": torch.stack([row["seq"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


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
    train = build(make_examples(seed, range(3), TRAIN_VARIANTS))
    fit = build(make_examples(seed + FIT_OFFSET, range(3, 4), FIT_VARIANTS))

    def unit_bag(features: torch.Tensor) -> torch.Tensor:
        gates = torch.ones(features.shape[0], graph.units.n_units)
        with torch.no_grad():
            result = graph(channels(features), gates, graph.initial_state())
        return result["unit_result"]["activity"]

    def unit_seq(sequence: torch.Tensor) -> torch.Tensor:
        state = graph.initial_state()
        activities = []
        for tick_index in range(sequence.shape[1]):
            gates = torch.ones(sequence.shape[0], graph.units.n_units)
            with torch.no_grad():
                result = graph(channels(sequence[:, tick_index]), gates, state)
            activities.append(result["unit_result"]["activity"])
            state = result["next_state"]
        stacked = torch.stack(activities, dim=1)
        return stacked.mean(dim=1)

    train_bag = unit_bag(train["bag"])
    fit_bag = unit_bag(fit["bag"])
    train_seq = unit_seq(train["seq"])
    fit_seq = unit_seq(fit["seq"])
    arms = {}
    for name, train_feature, fit_feature in (
        ("bag_units", train_bag, fit_bag),
        ("seq_units", train_seq, fit_seq),
    ):
        head = Read(train_feature.shape[-1], 4)
        train_head(head, {"features": train_feature, "labels": train["labels"]}, "features", seed)
        arms[name] = accuracy(head, {"features": fit_feature, "labels": fit["labels"]}, "features")
    for name, train_feature, fit_feature in (
        ("bag_raw", train["bag"], fit["bag"]),
        (
            "seq_raw",
            train["seq"].reshape(train["seq"].shape[0], -1),
            fit["seq"].reshape(fit["seq"].shape[0], -1),
        ),
    ):
        head = Read(train_feature.shape[-1], 4)
        train_head(head, {"features": train_feature, "labels": train["labels"]}, "features", seed)
        arms[name] = accuracy(head, {"features": fit_feature, "labels": fit["labels"]}, "features")
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
        "order_signal": means["seq_units"] >= means["bag_units"] + 0.10,
        "retention": all(max(row["retention_damage"].values()) <= 0.02 for row in rows),
    }
    verdict = "chinese_word_order_verified" if all(gates.values()) else "chinese_word_order_partial"
    payload = {
        "schema": "taiji-u2-cn-order-v1",
        "card": "Chinese word-order sensitivity: bag vs character-sequence ticks",
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
            "fit_offset": FIT_OFFSET,
            "classes": 4,
            "max_len": MAX_LEN,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "labels depend on the second mentioned colour; bag encoding cannot recover order; sequence ticks carry order through the graph",
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
    return 0 if verdict == "chinese_word_order_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
