"""Pressure-task P1/P2: five-body counting task, oracle and frozen-readout sweep (0 check).

P1 checks the symbolic oracle and label spread.  P2 fixes the frozen 23.0b parent plus the
two-step attention readout and sweeps training budget (1,500/3,000/6,000 steps): if counting
stays below 0.50 and does not improve with budget, the ceiling is readout/information
limited rather than optimization limited.  No unit growth (P3 is left for owner approval).
Zero check/final.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import channels, lr_at  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402

OUT = ROOT / "reports/taiji_u2_pressure_counting_20261120.json"
CAP_CPU = 1000.0
SEEDS = (20261120, 20261121, 20261122)
STEPS = (1500, 3000, 6000)
BATCH = 32
LR = 3e-3
BODIES = 5
COLOURS = 6
DIRECTIONS = ("左边", "右边", "前边", "后边")
TRAIN = 512
FIT = 128


def make_rows(count: int, seed: int) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(count):
        colour = int(torch.randint(0, COLOURS, (1,), generator=generator))
        direction = DIRECTIONS[int(torch.randint(0, len(DIRECTIONS), (1,), generator=generator))]
        matches = int(torch.randint(0, BODIES + 1, (1,), generator=generator))
        xs = (torch.rand(BODIES, generator=generator) - 0.5) * 2.0
        ys = (torch.rand(BODIES, generator=generator) - 0.5) * 2.0
        other_colours = [index for index in range(COLOURS) if index != colour]
        colours = [colour] * matches
        colours += [
            other_colours[int(torch.randint(0, len(other_colours), (1,), generator=generator))]
            for _ in range(BODIES - matches)
        ]
        colours = colours[:BODIES]
        # Place the matching bodies on the queried side (and the rest on neither side of it).
        for index in range(BODIES):
            if index < matches:
                if direction == "左边":
                    xs[index] = -float(torch.rand((), generator=generator)) - 0.05
                elif direction == "右边":
                    xs[index] = float(torch.rand((), generator=generator)) + 0.05
                elif direction == "前边":
                    ys[index] = float(torch.rand((), generator=generator)) + 0.05
                else:
                    ys[index] = -float(torch.rand((), generator=generator)) - 0.05
            elif colours[index] == colour:
                colours[index] = other_colours[
                    int(torch.randint(0, len(other_colours), (1,), generator=generator))
                ]
        tokens = torch.zeros(BODIES, 8)
        for index, body_colour in enumerate(colours):
            tokens[index, 0] = float(xs[index])
            tokens[index, 1] = float(ys[index])
            tokens[index, 2 + body_colour] = 1.0
        query = torch.zeros(11)
        query[colour] = 1.0
        query[6 + DIRECTIONS.index(direction)] = 1.0
        rows.append({"tokens": tokens, "query": query, "label": matches, "oracle": matches})
    return rows


def build(rows: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.stack([row["tokens"] for row in rows]).float(),
        "queries": torch.stack([row["query"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


class TwoStepRead(nn.Module):
    def __init__(self, token_dim: int, width: int = 64) -> None:
        super().__init__()
        self.query1 = nn.Sequential(nn.Linear(11, width), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(width + 11, width), nn.Tanh())
        self.token = nn.Sequential(nn.Linear(token_dim, width), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 6))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        embed = self.token(tokens)
        q1 = self.query1(query)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, q1), dim=-1)
        context1 = torch.einsum("bt,btd->bd", scores, embed)
        q2 = self.query2(torch.cat([context1, query], dim=-1))
        scores2 = torch.softmax(torch.einsum("btd,bd->bt", embed, q2), dim=-1)
        context2 = torch.einsum("bt,btd->bd", scores2, embed)
        return self.out(torch.cat([context2, q1], dim=-1))


def train_head(head: nn.Module, batch: dict[str, torch.Tensor], steps: int, seed: int) -> None:
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


def accuracy(head: nn.Module, batch: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        logits = head(batch["tokens"], batch["queries"])
    return float((logits.argmax(-1) == batch["labels"]).float().mean())


def unit_activity(graph, tokens: torch.Tensor) -> torch.Tensor:
    flat = torch.zeros(tokens.shape[0] * tokens.shape[1], 48)
    flat[:, : tokens.shape[2]] = tokens.reshape(tokens.shape[0] * tokens.shape[1], tokens.shape[2])
    gates = torch.ones(flat.shape[0], graph.units.n_units)
    result = graph(channels(flat), gates, graph.initial_state())
    return result["unit_result"]["activity"].reshape(tokens.shape[0], tokens.shape[1], -1)


def run_seed(seed: int) -> dict:
    graph, _ = train_joint_parent(seed)
    train_rows = make_rows(TRAIN, seed)
    fit_rows = make_rows(FIT, seed + 1)
    labels = Counter(row["label"] for row in train_rows)
    majority = max(labels.values()) / sum(labels.values())
    train = build(train_rows)
    fit = build(fit_rows)
    train["tokens"] = unit_activity(graph, train["tokens"]).detach()
    fit["tokens"] = unit_activity(graph, fit["tokens"]).detach()
    results = {}
    for steps in STEPS:
        head = TwoStepRead(train["tokens"].shape[-1])
        train_head(head, train, steps, seed + steps)
        results[f"steps_{steps}"] = accuracy(head, fit)
    return {
        "seed": seed,
        "majority": majority,
        "oracle_ok": all(row["oracle"] == row["label"] for row in train_rows + fit_rows),
        "oracle_accuracy": (
            1.0 if all(row["oracle"] == row["label"] for row in train_rows + fit_rows) else 0.0
        ),
        "label_hist": dict(labels),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {
        key: sum(row["results"][key] for row in rows) / len(rows) for key in rows[0]["results"]
    }
    improvement = means["steps_6000"] - means["steps_1500"]
    gates = {
        "p1_oracle": all(row["oracle_ok"] for row in rows),
        "p1_balance": all(row["majority"] <= 0.40 for row in rows),
        "p2_ceiling": max(means.values()) < 0.50 and improvement < 0.02,
    }
    verdict = "pressure_measurable_pending_p3" if all(gates.values()) else "pressure_not_structural"
    payload = {
        "schema": "taiji-u2-pressure-counting-v1",
        "card": "P1/P2 counting pressure task: oracle + frozen two-step readout sweep",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "bodies": BODIES,
            "train": TRAIN,
            "fit": FIT,
            "steps": list(STEPS),
            "unit_growth": "not_run_pending_owner",
        },
        "rows": rows,
        "means": means,
        "improvement": improvement,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "P3 (30->48 unit diagnostic) intentionally NOT run; requires owner approval",
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
                "majority": [row["majority"] for row in rows],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "pressure_measurable_pending_p3" else 1


if __name__ == "__main__":
    raise SystemExit(main())
