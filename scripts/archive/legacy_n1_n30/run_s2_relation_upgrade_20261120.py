"""Relation-read upgrade retest on the frozen 23.1 composition contract (0 check).

Replaces the original two-step attention readout of 23.2c with the positional-attention
readout (learned token-position embeddings + two-step attention) on the same frozen unit
activities.  The original Q3 was 0.406-0.422; the gate is Q3 >= 0.46 with old-skill damage
<= 0.02.  The data contract (train/fit partitions, oracle) is unchanged.  Zero check/final.
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
from scripts.training.run_s2_23_0b_joint_20261120 import lr_at  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import (  # noqa: E402
    label_for,
    tokens_for,
    train_joint_parent,
    unit_activity,
)
from taiji.native_composition import build_partition  # noqa: E402

OUT = ROOT / "reports/taiji_u2_relation_upgrade_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
STEPS = 2000
BATCH = 32
LR = 3e-3
WIDTH = 64
Q3_MIN = 0.46
RETENTION_MAX = 0.02


def relation_rows(partition: str) -> list[dict]:
    return [row for row in build_partition(partition) if row["kind"] in {"q2", "q3"}]


def encode(rows: list[dict]) -> dict[str, torch.Tensor]:
    tokens, queries, labels = [], [], []
    for row in rows:
        token, query = tokens_for(row)
        tokens.append(token)
        queries.append(query)
        labels.append(label_for(row))
    return {
        "tokens": torch.stack(tokens).float(),
        "queries": torch.stack(queries).float(),
        "labels": torch.tensor(labels),
        "q3": torch.tensor([row["kind"] == "q3" for row in rows]),
    }


class PosAttentionRead(nn.Module):
    def __init__(self, token_dim: int, classes: int = 4, positions: int = 3) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(positions, token_dim))
        self.query1 = nn.Sequential(nn.Linear(8, WIDTH), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(WIDTH + 8, WIDTH), nn.Tanh())
        self.token = nn.Sequential(nn.Linear(token_dim, WIDTH), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.Tanh(), nn.Linear(WIDTH, classes))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        embed = self.token(tokens + self.pos[: tokens.shape[1]])
        q1 = self.query1(query)
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
        index = torch.randint(0, batch["tokens"].shape[0], (BATCH,), generator=generator)
        loss = loss_fn(
            head(batch["tokens"][index], batch["queries"][index]), batch["labels"][index]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate(head: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    with torch.no_grad():
        logits = head(batch["tokens"], batch["queries"])
    return {
        "q2": float(
            (logits[~batch["q3"]].argmax(-1) == batch["labels"][~batch["q3"]]).float().mean()
        ),
        "q3": float(
            (logits[batch["q3"]].argmax(-1) == batch["labels"][batch["q3"]]).float().mean()
        ),
    }


def run_seed(seed: int) -> dict:
    graph, skill_data = train_joint_parent(seed)
    from scripts.training.run_s2_23_0b_joint_20261120 import accuracy as skill_accuracy

    baseline = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    train_batch = encode(relation_rows("train"))
    fit_batch = encode(relation_rows("fit"))
    train_activity = unit_activity(graph, train_batch["tokens"])
    fit_activity = unit_activity(graph, fit_batch["tokens"])
    head = PosAttentionRead(train_activity.shape[-1])
    train_head(head, {**train_batch, "tokens": train_activity}, seed)
    scores = evaluate(head, {**fit_batch, "tokens": fit_activity})
    after = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    damage = {skill: max(0.0, baseline[skill] - after[skill]) for skill in baseline}
    return {"seed": seed, "scores": scores, "retention_damage": damage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {
        "q2": sum(row["scores"]["q2"] for row in rows) / len(rows),
        "q3": sum(row["scores"]["q3"] for row in rows) / len(rows),
    }
    gates = {
        "q3": means["q3"] >= Q3_MIN,
        "retention": all(max(row["retention_damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = "relation_upgrade_verified" if all(gates.values()) else "relation_upgrade_partial"
    payload = {
        "schema": "taiji-u2-relation-upgrade-v1",
        "card": "relation-read upgrade retest on the frozen 23.1 composition contract",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "steps": STEPS,
            "q3_min": Q3_MIN,
            "partitions": ["train", "fit"],
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "same frozen activities as 23.2c, only the readout changed to positional attention; no check/final reads",
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
    return 0 if verdict == "relation_upgrade_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
