"""25.0 pressure audit: is the relation-read ceiling information/optimization or structure? (0 check).

Fixes the two-step attention readout (23.2c) on the frozen 23.0b parent and sweeps training
budget (1,500 / 3,000 / 6,000 steps) plus a wider hidden arm.  If the Q3 accuracy saturates
below 0.50 and adding steps/width buys < 0.02, the ceiling is information/optimization
limited -> structural pressure is NOT separable at this interface (``pressure_absent``).
If accuracy keeps rising, the ceiling is optimization-limited and pressure is separable.
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
from scripts.training.run_s2_23_0b_joint_20261120 import lr_at  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import (  # noqa: E402
    label_for,
    tokens_for,
    train_joint_parent,
    unit_activity,
)
from taiji.native_composition import build_partition  # noqa: E402

OUT = ROOT / "reports/taiji_u2_25_0_pressure_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
STEPS = (1500, 3000, 6000)
WIDTHS = (32, 64)
BATCH = 32
LR = 3e-3
SATURATION = 0.02
TARGET = 0.50


class TwoStepRead(nn.Module):
    def __init__(self, token_dim: int, width: int) -> None:
        super().__init__()
        self.query1 = nn.Sequential(nn.Linear(8, width), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(width + 8, width), nn.Tanh())
        self.token = nn.Sequential(nn.Linear(token_dim, width), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 4))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        embed = self.token(tokens)
        q1 = self.query1(query)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, q1), dim=-1)
        context1 = torch.einsum("bt,btd->bd", scores, embed)
        q2 = self.query2(torch.cat([context1, query], dim=-1))
        scores2 = torch.softmax(torch.einsum("btd,bd->bt", embed, q2), dim=-1)
        context2 = torch.einsum("bt,btd->bd", scores2, embed)
        return self.out(torch.cat([context2, q1], dim=-1))


def relation_data() -> tuple[dict, dict]:
    def build(partition: str) -> dict:
        rows = [row for row in build_partition(partition) if row["kind"] in {"q2", "q3"}]
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

    return build("train"), build("fit")


def train_head(head: nn.Module, batch: dict, steps: int, seed: int) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, steps)
        index = torch.randint(0, batch["tokens"].shape[0], (BATCH,), generator=generator)
        loss = loss_fn(
            head(batch["tokens"][index], batch["queries"][index]), batch["labels"][index]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate(head: nn.Module, fit: dict) -> dict[str, float]:
    with torch.no_grad():
        logits = head(fit["tokens"], fit["queries"])
    return {
        "q2": float((logits[~fit["q3"]].argmax(-1) == fit["labels"][~fit["q3"]]).float().mean()),
        "q3": float((logits[fit["q3"]].argmax(-1) == fit["labels"][fit["q3"]]).float().mean()),
    }


def run_seed(seed: int) -> dict:
    graph, _ = train_joint_parent(seed)
    train_raw, fit_raw = relation_data()
    train_activity = unit_activity(graph, train_raw["tokens"]).detach()
    fit_activity = unit_activity(graph, fit_raw["tokens"]).detach()
    train_batch = {**train_raw, "tokens": train_activity}
    fit_batch = {**fit_raw, "tokens": fit_activity}
    results = {}
    for width in WIDTHS:
        for steps in STEPS:
            head = TwoStepRead(train_activity.shape[-1], width)
            train_head(head, train_batch, steps, seed + steps + width)
            results[f"w{width}_s{steps}"] = evaluate(head, fit_batch)
    return {"seed": seed, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {}
    for key in rows[0]["results"]:
        means[key] = sum(row["results"][key]["q3"] for row in rows) / len(rows)
    base = means["w32_s1500"]
    long_run = means["w32_s6000"]
    wide_long = means["w64_s6000"]
    improvement = max(long_run, wide_long) - base
    saturated = improvement < SATURATION and max(long_run, wide_long) < TARGET
    gates = {
        "saturation_or_separable": saturated or improvement >= SATURATION,
        "plateau_below_target": saturated,
    }
    verdict = "pressure_absent" if saturated else "pressure_separable"
    payload = {
        "schema": "taiji-u2-25-0-pressure-v1",
        "card": "25.0 pressure audit: two-step attention readout budget sweep",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "steps": list(STEPS),
            "widths": list(WIDTHS),
            "saturation": SATURATION,
            "target": TARGET,
        },
        "rows": rows,
        "q3_means": means,
        "improvement_long_vs_base": improvement,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "fixed frozen 23.0b parent; two-step attention readout; no structural growth in this audit",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "q3_means": means,
                "improvement": improvement,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "pressure_separable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
