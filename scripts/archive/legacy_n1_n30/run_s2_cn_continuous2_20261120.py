"""Chinese continuous-task extension: 10/20-tick sequences and a mid-sequence update (0 check).

Same protocol as the 5-tick Chinese continuous card (one high-density Chinese instruction per
tick, positional-attention readout over per-tick unit activities), but the sequence is
lengthened to 10 and 20 ticks.  At 20 ticks an update variant replaces the target at the
midpoint, so the label is the updated instruction.  Replay must reach 0.80 before the carry
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

OUT = ROOT / "reports/taiji_u2_cn_continuous2_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
TRAIN = 384
FIT = 128
STEPS = 1500
BATCH = 32
LR = 3e-3
COLOURS = ("红", "蓝", "绿", "黄")
DIRECTIONS = ("左", "右", "前", "后")
TEMPLATES = ("把{c}球放到{d}边", "{c}球放{d}边", "让{c}球去{d}边")
VOCAB = "把球放到边让去红蓝绿黄左右前后"
CHAR_INDEX = {char: index for index, char in enumerate(VOCAB)}
PROJECTION = torch.randn(len(VOCAB), 48, generator=torch.Generator().manual_seed(43))
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


def random_instruction(generator: torch.Generator) -> tuple[str, str, int]:
    return (
        COLOURS[int(torch.randint(0, 4, (1,), generator=generator))],
        DIRECTIONS[int(torch.randint(0, 4, (1,), generator=generator))],
        int(torch.randint(0, 3, (1,), generator=generator)),
    )


def episode(seed: int, ticks: int, update: bool) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target = random_instruction(generator)
    mid = ticks // 2
    second = random_instruction(generator) if update else None
    sequence = []
    for tick_index in range(ticks):
        if tick_index == 0:
            colour, direction, template = target
        elif update and tick_index == mid and second is not None:
            colour, direction, template = second
        elif tick_index == ticks - 1:
            colour, direction, template = random_instruction(generator)
        else:
            colour, direction, template = random_instruction(generator)
        sequence.append(instruction_tick(colour, direction, template))
    query = torch.zeros(8)
    query[7] = 1.0
    final = second if update and second is not None else target
    return {
        "ticks": torch.stack(sequence),
        "queries": query,
        "label": COLOURS.index(final[0]) * 4 + DIRECTIONS.index(final[1]),
    }


def build(count: int, seed: int, ticks: int, update: bool) -> dict[str, torch.Tensor]:
    rows = [episode(seed + index, ticks, update) for index in range(count)]
    return {
        "ticks": torch.stack([row["ticks"] for row in rows]).float(),
        "queries": torch.stack([row["queries"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


class PosAttentionRead(nn.Module):
    def __init__(self, token_dim: int, classes: int, width: int = 64, positions: int = 21) -> None:
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
    results = {}
    for ticks, update in ((10, False), (20, False), (20, True)):
        key = f"t{ticks}{'_update' if update else ''}"
        train = build(TRAIN, seed, ticks, update)
        fit = build(FIT, seed + FIT_OFFSET, ticks, update)
        last = ticks - 1
        target_ticks = [0] if not update else [0, ticks // 2]
        arms = {}
        for arm in ("stateless_reset", "replay", "state_carry"):
            if arm == "stateless_reset":
                train_activity = collect_activity(
                    graph, train["ticks"], [last], reset=True
                ).detach()
                fit_activity = collect_activity(graph, fit["ticks"], [last], reset=True).detach()
            else:
                use = [*target_ticks, last] if arm == "replay" else list(range(ticks))
                train_activity = collect_activity(graph, train["ticks"], use).detach()
                fit_activity = collect_activity(graph, fit["ticks"], use).detach()
            head = PosAttentionRead(train_activity.shape[-1], 16, positions=train_activity.shape[1])
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
        results[key] = arms
    after = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    damage = {skill: max(0.0, baseline[skill] - after[skill]) for skill in baseline}
    return {"seed": seed, "results": results, "retention_damage": damage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {
        key: {
            arm: sum(row["results"][key][arm] for row in rows) / len(rows)
            for arm in rows[0]["results"][key]
        }
        for key in rows[0]["results"]
    }
    gates = {}
    for key in ("t10", "t20", "t20_update"):
        gates[f"{key}_learnable"] = means[key]["replay"] >= REPLAY_MIN
        gates[f"{key}_carry"] = (
            means[key]["state_carry"] >= CARRY_MIN
            and means[key]["state_carry"] >= means[key]["replay"] - TOL
        )
    gates["retention"] = all(max(row["retention_damage"].values()) <= 0.02 for row in rows)
    verdict = "cn_continuous_ext_verified" if all(gates.values()) else "cn_continuous_ext_partial"
    payload = {
        "schema": "taiji-u2-cn-continuous2-v1",
        "card": "Chinese continuous task at 10/20 ticks and with a mid-sequence update",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "conditions": ["t10", "t20", "t20_update"],
            "train": TRAIN,
            "fit": FIT,
            "fit_offset": FIT_OFFSET,
            "steps": STEPS,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "one high-density Chinese instruction per tick; both non-update and update variants share the positional-attention readout",
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
    return 0 if verdict == "cn_continuous_ext_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
