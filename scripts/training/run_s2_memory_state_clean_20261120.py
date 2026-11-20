"""True state-only retention test with collision-free features (0 check).

Fixes two defects in the earlier memory probes: (1) the state_carry arm previously collected
per-tick activity snapshots into an external [B,T,30] history array (a KV-cache analogue)
instead of reading the unit state at the final tick; (2) the fixture wrote colour one-hot and
position one-hot into overlapping indices.  This probe uses non-overlapping slots and compares
stateless_reset, state_final (only the final tick activity, no history), history (all snapshots,
for contrast) and replay (target + last tick) over gaps 5/10/20/40.  Zero check/final.
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

OUT = ROOT / "reports/taiji_u2_memory_state_clean_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
GAPS = (5, 10, 20, 40)
TRAIN = 384
FIT = 128
FIT_OFFSET = 10_000_000
BODY = 3
BODY_STRIDE = 12
TICK_OFFSET = 36
TICK_SLOTS = 12
DIM = 48
STEPS = 1500
BATCH = 32
LR = 3e-3


def episode(seed: int, gap: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target_colour = int(torch.randint(0, 6, (1,), generator=generator))
    target_position = int(torch.randint(0, 3, (1,), generator=generator))
    ticks = []
    for tick_index in range(gap):
        features = torch.zeros(DIM)
        for body in range(BODY):
            if tick_index == 0 and body == 0:
                colour, position = target_colour, target_position
            else:
                colour = int(torch.randint(0, 6, (1,), generator=generator))
                position = int(torch.randint(0, 3, (1,), generator=generator))
            base = body * BODY_STRIDE
            features[base + colour] = 1.0
            features[base + 6 + position] = 1.0
        features[TICK_OFFSET + (tick_index % TICK_SLOTS)] = 1.0
        ticks.append(features)
    query = torch.zeros(8)
    query[7] = 1.0
    return {"ticks": torch.stack(ticks), "query": query, "label": target_position}


def build(count: int, seed: int, gap: int) -> dict[str, torch.Tensor]:
    rows = [episode(seed + index, gap) for index in range(count)]
    return {
        "ticks": torch.stack([row["ticks"] for row in rows]).float(),
        "queries": torch.stack([row["query"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


class PosAttentionRead(nn.Module):
    def __init__(self, token_dim: int, positions: int = 40) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(positions, token_dim))
        self.token = nn.Sequential(nn.Linear(token_dim, 64), nn.Tanh())
        self.query1 = nn.Sequential(nn.Linear(8, 64), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(64 + 8, 64), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 4))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        embed = self.token(tokens + self.pos[: tokens.shape[1]])
        q1 = self.query1(query)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, q1), dim=-1)
        context1 = torch.einsum("bt,btd->bd", scores, embed)
        q2 = self.query2(torch.cat([context1, query], dim=-1))
        scores2 = torch.softmax(torch.einsum("btd,bd->bt", embed, q2), dim=-1)
        context2 = torch.einsum("bt,btd->bd", scores2, embed)
        return self.out(torch.cat([context2, q1], dim=-1))


class FinalStateRead(nn.Module):
    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim + 8, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 4),
        )

    def forward(self, state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, query], dim=-1))


def run_ticks(graph, ticks: torch.Tensor, collect_ticks: list[int], reset: bool = False):
    """Return final activity plus optionally collected per-tick activities."""
    state = graph.initial_state()
    collected = []
    final = None
    for tick_index in range(ticks.shape[1]):
        if reset and tick_index != ticks.shape[1] - 1:
            continue
        gates = torch.ones(ticks.shape[0], graph.units.n_units)
        with torch.no_grad():
            result = graph(channels(ticks[:, tick_index]), gates, state)
        activity = result["unit_result"]["activity"]
        if tick_index in collect_ticks:
            collected.append(activity)
        state = result["next_state"]
        final = activity
    return final, torch.stack(collected, dim=1)


def train_head(head: nn.Module, batch: dict[str, torch.Tensor], kind: str, seed: int) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, batch["labels"].shape[0], (BATCH,), generator=generator)
        if kind == "final":
            logits = head(batch["state"][index], batch["queries"][index])
        else:
            logits = head(batch["activities"][index], batch["queries"][index])
        loss = loss_fn(logits, batch["labels"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def accuracy(head: nn.Module, batch: dict[str, torch.Tensor], kind: str) -> float:
    with torch.no_grad():
        if kind == "final":
            logits = head(batch["state"], batch["queries"])
        else:
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
        # stateless reset: only the last tick with a fresh state
        state, _ = run_ticks(graph, train["ticks"], [last], reset=True)
        fit_state, _ = run_ticks(graph, fit["ticks"], [last], reset=True)
        head = FinalStateRead(state.shape[-1])
        train_head(
            head,
            {"state": state, "queries": train["queries"], "labels": train["labels"]},
            "final",
            seed,
        )
        arms["stateless_reset"] = accuracy(
            head, {"state": fit_state, "queries": fit["queries"], "labels": fit["labels"]}, "final"
        )
        # true state-only: process all ticks, read only the final activity
        final_train, _ = run_ticks(graph, train["ticks"], [last])
        final_fit, _ = run_ticks(graph, fit["ticks"], [last])
        head = FinalStateRead(final_train.shape[-1])
        train_head(
            head,
            {"state": final_train, "queries": train["queries"], "labels": train["labels"]},
            "final",
            seed,
        )
        arms["state_final"] = accuracy(
            head, {"state": final_fit, "queries": fit["queries"], "labels": fit["labels"]}, "final"
        )
        # history (KV-cache analogue): all per-tick snapshots
        _, hist_train = run_ticks(graph, train["ticks"], list(range(gap)))
        _, hist_fit = run_ticks(graph, fit["ticks"], list(range(gap)))
        head = PosAttentionRead(hist_train.shape[-1], positions=hist_train.shape[1])
        train_head(
            head,
            {"activities": hist_train, "queries": train["queries"], "labels": train["labels"]},
            "seq",
            seed,
        )
        arms["history"] = accuracy(
            head,
            {"activities": hist_fit, "queries": fit["queries"], "labels": fit["labels"]},
            "seq",
        )
        # replay: target tick + last tick snapshots
        _, rep_train = run_ticks(graph, train["ticks"], [0, last])
        _, rep_fit = run_ticks(graph, fit["ticks"], [0, last])
        head = PosAttentionRead(rep_train.shape[-1], positions=rep_train.shape[1])
        train_head(
            head,
            {"activities": rep_train, "queries": train["queries"], "labels": train["labels"]},
            "seq",
            seed,
        )
        arms["replay"] = accuracy(
            head, {"activities": rep_fit, "queries": fit["queries"], "labels": fit["labels"]}, "seq"
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
            for arm in ("stateless_reset", "state_final", "history", "replay")
        }
        for gap in GAPS
    }
    gates = {
        "retention": all(max(row["retention_damage"].values()) <= 0.02 for row in rows),
    }
    payload = {
        "schema": "taiji-u2-memory-state-clean-v1",
        "card": "true state-only retention with collision-free features",
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
            "arms": ["stateless_reset", "state_final", "history", "replay"],
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": "memory_state_clean_diagnostic",
        "disclosure": "state_final reads only the final tick activity; history is an external snapshot array for contrast (KV-cache analogue), not unit-state retention",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "means": means,
                "gates": gates,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
