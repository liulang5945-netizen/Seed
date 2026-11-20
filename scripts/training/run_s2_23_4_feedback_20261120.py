"""23.4: feedback-permutation probe on the token interface (0 check).

The content head reads the frozen 23.0b parent.  A feedback field is appended to the
query: either no hint, or a one-hot content hint (the true answer).  Training mixes hinted
and unhinted rows; evaluation compares content accuracy with vs without the hint, which
measures whether the feedback reaches the responsible head (not whether memory persists).
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
from scripts.training.run_s2_23_0b_joint_20261120 import lr_at
from scripts.training.run_s2_23_2c_attention_20261120 import (  # noqa: E402
    label_for,
    tokens_for,
    train_joint_parent,
    unit_activity,
)
from taiji.native_composition import build_partition  # noqa: E402

OUT = ROOT / "reports/taiji_u2_23_4_feedback_20261120.json"
CAP_CPU = 1000.0
SEEDS = (20261120, 20261121, 20261122)
BATCH = 32
LR = 3e-3
STEPS = 1200
FEEDBACK_GAIN_MIN = 0.05
RETENTION_MAX = 0.02
HINT_RATE = 0.5


def rows_of(partition: str) -> list[dict]:
    return [row for row in build_partition(partition) if row["kind"] in {"q2", "q3"}]


def encode(rows: list[dict], hinted: bool, seed: int) -> dict[str, torch.Tensor]:
    tokens = []
    queries = []
    labels = []
    generator = torch.Generator().manual_seed(seed + 71)
    for row in rows:
        token, query = tokens_for(row)
        labels.append(label_for(row))
        hint = torch.zeros(5)
        if hinted and torch.rand((), generator=generator).item() < HINT_RATE:
            hint[int(labels[-1])] = 1.0
            hint[4] = 1.0
        queries.append(torch.cat([query, hint]))
        tokens.append(token)
    return {
        "tokens": torch.stack(tokens).float(),
        "queries": torch.stack(queries).float(),
        "labels": torch.tensor(labels),
    }


class FeedbackRead(nn.Module):
    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.query1 = nn.Sequential(nn.Linear(13, 16), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(29, 16), nn.Tanh())
        self.token = nn.Sequential(nn.Linear(token_dim, 16), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(32, 24), nn.Tanh(), nn.Linear(24, 4))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        embed = self.token(tokens)
        q1 = self.query1(query)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, q1), dim=-1)
        context1 = torch.einsum("bt,btd->bd", scores, embed)
        q2 = self.query2(torch.cat([context1, query], dim=-1))
        scores2 = torch.softmax(torch.einsum("btd,bd->bt", embed, q2), dim=-1)
        context2 = torch.einsum("bt,btd->bd", scores2, embed)
        return self.head(torch.cat([context2, q1], dim=-1))


def train(module: nn.Module, batch: dict[str, torch.Tensor], seed: int) -> None:
    optimizer = torch.optim.Adam(module.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 79)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, batch["tokens"].shape[0], (BATCH,), generator=generator)
        loss = loss_fn(
            module(batch["tokens"][index], batch["queries"][index]), batch["labels"][index]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate(module: nn.Module, batch: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        logits = module(batch["tokens"], batch["queries"])
    return float((logits.argmax(-1) == batch["labels"]).float().mean())


def run_seed(seed: int) -> dict:
    graph, skill_data = train_joint_parent(seed)
    baseline = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    train_rows = rows_of("train")
    fit_rows = rows_of("fit")
    train_batch = encode(train_rows, hinted=True, seed=seed)
    fit_plain = encode(fit_rows, hinted=False, seed=seed)
    fit_hint = encode(fit_rows, hinted=True, seed=seed + 100)
    cross = {}
    for name, batch in (("train", train_batch), ("plain", fit_plain), ("hint", fit_hint)):
        cross[name] = unit_activity(graph, batch["tokens"]).detach()
    module = FeedbackRead(cross["train"].shape[-1])
    train(module, {**train_batch, "tokens": cross["train"]}, seed)
    no_hint = evaluate(module, {**fit_plain, "tokens": cross["plain"]})
    with_hint = evaluate(module, {**fit_hint, "tokens": cross["hint"]})
    retention = {
        "baseline": baseline,
        "after": {
            "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
            "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
            "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
        },
    }
    retention["damage"] = {
        skill: max(0.0, baseline[skill] - retention["after"][skill]) for skill in baseline
    }
    return {
        "seed": seed,
        "no_hint": no_hint,
        "with_hint": with_hint,
        "feedback_gain": with_hint - no_hint,
        "retention": retention,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    gains = [row["feedback_gain"] for row in rows]
    mean_gain = sum(gains) / len(gains)
    generator = __import__("numpy").random.default_rng(20261120)
    samples = [
        float(generator.choice(gains, size=len(gains), replace=True).mean()) for _ in range(2000)
    ]
    lower = float(__import__("numpy").percentile(samples, 2.5))
    gates = {
        "feedback_gain": mean_gain >= FEEDBACK_GAIN_MIN and lower > 0.0,
        "retention": all(max(row["retention"]["damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = "feedback_effect_verified" if all(gates.values()) else "feedback_effect_partial"
    payload = {
        "schema": "taiji-u2-23-4-feedback-v1",
        "card": "23.4 feedback-permutation probe on the token interface",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "steps": STEPS,
            "batch": BATCH,
            "hint_rate": HINT_RATE,
            "feedback_gain_min": FEEDBACK_GAIN_MIN,
        },
        "rows": rows,
        "mean_gain": mean_gain,
        "gain_lb95": lower,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "rows": rows,
                "mean_gain": mean_gain,
                "lb95": lower,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "feedback_effect_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
