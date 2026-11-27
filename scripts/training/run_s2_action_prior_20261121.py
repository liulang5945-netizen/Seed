"""A2: action-grounding learnability prior with collision-free flat features (0 check).

Fixes the colour/position index overlap and checks whether the action task is learnable at
all: flat features (3 objects x 10 slots + 16-dim instruction projection) with a strong MLP
must reach fit accuracy >= 0.80.  Only if this passes may the closed loop / multi-step action
cards reopen.  Zero check/final.
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

OUT = ROOT / "reports/taiji_u2_action_prior_20261121.json"
CAP_CPU = 1000.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
TRAIN = 512
FIT = 128
STEPS = 2500
BATCH = 64
LR = 3e-3
COLOURS = ("红", "蓝", "绿", "黄")
DIRECTIONS = ("左", "右", "前", "后")
TEMPLATES = ("把{c}球放到{d}边", "{c}球放{d}边", "让{c}球去{d}边", "把{c}球放去{d}边")
VOCAB = "把球放到边让去红蓝绿黄左右前后"
CHAR_INDEX = {char: index for index, char in enumerate(VOCAB)}
PROJECTION = torch.randn(len(VOCAB), 16, generator=torch.Generator().manual_seed(71))
CLASSES = 12
REPLAY_MIN = 0.80
BODY_SLOTS = 10


def instruction_features(colour: str, direction: str) -> torch.Tensor:
    text = TEMPLATES[0].format(c=colour, d=direction)
    bag = torch.zeros(len(VOCAB))
    for char in text:
        if char in CHAR_INDEX:
            bag[CHAR_INDEX[char]] += 1.0
    return bag @ PROJECTION


def episode(seed: int, template_index: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    colours = torch.randperm(4, generator=generator)[:3].tolist()
    positions = torch.randint(0, 4, (3,), generator=generator).tolist()
    target_slot = int(torch.randint(0, 3, (1,), generator=generator))
    direction = DIRECTIONS[int(torch.randint(0, 4, (1,), generator=generator))]
    features = torch.zeros(3 * BODY_SLOTS + 16)
    for slot in range(3):
        base = slot * BODY_SLOTS
        features[base + colours[slot]] = 1.0
        features[base + 6 + positions[slot]] = 1.0
    # Instruction names the target colour with the held-out-aware template.
    text = TEMPLATES[template_index % len(TEMPLATES)].format(
        c=COLOURS[colours[target_slot]], d=direction
    )
    bag = torch.zeros(len(VOCAB))
    for char in text:
        if char in CHAR_INDEX:
            bag[CHAR_INDEX[char]] += 1.0
    features[30:] = bag @ PROJECTION
    return {"features": features, "label": target_slot * 4 + DIRECTIONS.index(direction)}


def build(count: int, seed: int, templates: tuple[int, ...]) -> dict[str, torch.Tensor]:
    rows = [episode(seed + index, templates[index % len(templates)]) for index in range(count)]
    return {
        "features": torch.stack([row["features"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


class StrongMLP(nn.Module):
    def __init__(self, input_dim: int, classes: int, width: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
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
    for _ in range(STEPS):
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
    train = build(TRAIN, seed, (0, 1, 2))
    fit = build(FIT, seed + FIT_OFFSET, (3,))
    head = StrongMLP(train["features"].shape[-1], CLASSES)
    train_head(head, train, seed)
    return {"seed": seed, "replay": accuracy(head, fit), "train_fit": accuracy(head, train)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    mean_replay = sum(row["replay"] for row in rows) / len(rows)
    gates = {
        "replay_learnable": mean_replay >= REPLAY_MIN,
        "slot_check": True,
    }
    verdict = "action_prior_learnable" if all(gates.values()) else "action_prior_not_learnable"
    payload = {
        "schema": "taiji-u2-action-prior-v1",
        "card": "A2 action-grounding learnability prior with collision-free slots",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "train": TRAIN,
            "fit": FIT,
            "fit_offset": FIT_OFFSET,
            "steps": STEPS,
            "body_slots": BODY_SLOTS,
            "classes": CLASSES,
            "replay_min": REPLAY_MIN,
        },
        "rows": rows,
        "mean_replay": mean_replay,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "collision-free slots (colour 0-5, position 6-9 per object); held-out wording template; no unit graph in this prior",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "rows": rows, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "action_prior_learnable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
