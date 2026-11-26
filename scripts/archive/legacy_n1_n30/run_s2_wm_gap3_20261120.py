"""Retention re-test with positional-attention readout (0 check).

Same clean fixture and disjoint fit seeds as the gap retest, but every arm uses the upgraded
readout (learned positional embeddings + two-step attention over per-tick unit activities).
Replay must reach 0.80 before carry is interpreted; the question is whether the retention
window extends beyond the previous 5 ticks.  Zero check/final.
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
from scripts.training.run_s2_23_0b_joint_20261120 import lr_at  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402
from scripts.training.run_s2_wm_gap2_20261120 import (  # noqa: E402
    FIT,
    FIT_OFFSET,
    GAPS,
    STEPS,
    TRAIN,
    build,
    collect_activity,
)

OUT = ROOT / "reports/taiji_u2_wm_gap3_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
REPLAY_MIN = 0.80
BATCH = 32
LR = 3e-3


class PosAttentionRead(nn.Module):
    def __init__(self, token_dim: int, width: int = 64, positions: int = 40) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(positions, token_dim))
        self.token = nn.Sequential(nn.Linear(token_dim, width), nn.Tanh())
        self.query1 = nn.Sequential(nn.Linear(width, width), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(width + 8, width), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 4))

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
        fit = build(FIT, seed + FIT_OFFSET, gap)
        last = gap - 1
        arms = {}
        for arm in ("stateless_reset", "replay", "state_carry"):
            if arm == "stateless_reset":
                train_activity = collect_activity(
                    graph, train["ticks"], [last], reset=True
                ).detach()
                fit_activity = collect_activity(graph, fit["ticks"], [last], reset=True).detach()
            else:
                use = [0, last] if arm == "replay" else list(range(gap))
                train_activity = collect_activity(graph, train["ticks"], use).detach()
                fit_activity = collect_activity(graph, fit["ticks"], use).detach()
            head = PosAttentionRead(train_activity.shape[-1], positions=train_activity.shape[1])
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
    means = {
        str(gap): {
            arm: sum(row["gaps"][str(gap)][arm] for row in rows) / len(rows)
            for arm in ("stateless_reset", "replay", "state_carry")
        }
        for gap in GAPS
    }
    carry = [means[str(gap)]["state_carry"] for gap in GAPS]
    replay = [means[str(gap)]["replay"] for gap in GAPS]
    boundary = 0
    for index, gap in enumerate(GAPS):
        if carry[index] >= 0.60 and carry[index] >= replay[index] - 0.10:
            boundary = gap
    gates = {
        "task_learnable": all(value >= REPLAY_MIN for value in replay),
        "retention": all(max(row["retention_damage"].values()) <= 0.02 for row in rows),
    }
    verdict = (
        "memory_gap3_verified" if all(gates.values()) and boundary > 0 else "memory_gap3_partial"
    )
    payload = {
        "schema": "taiji-u2-wm-gap3-v1",
        "card": "retention re-test with positional-attention readout",
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
            "fit_offset": FIT_OFFSET,
            "steps": STEPS,
        },
        "rows": rows,
        "means": means,
        "boundary": boundary,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "positional-attention readout on frozen per-tick unit activities; replay gets target + last tick",
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
    return 0 if verdict == "memory_gap3_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
