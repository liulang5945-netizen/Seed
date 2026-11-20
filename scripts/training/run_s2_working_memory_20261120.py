"""Working-memory probe: 5-tick target recall with carried unit state (0 check).

Each episode: t0 shows the target body, t1-t3 are distractor scenes, t4 asks where the
target was.  The frozen 23.0b parent is run tick by tick (state carried via the graph's
next_state) and a two-step attention readout reads the collected 30-dim activity vectors.
Arms: no_state (t4 only), replay (t0+t4, upper bound), state_carry (all 5 ticks).
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

OUT = ROOT / "reports/taiji_u2_working_memory_20261120.json"
CAP_CPU = 2000.0
SEEDS = (20261120, 20261121, 20261122)
TRAIN = 512
FIT = 128
TICKS = 5
BODY = 3
BATCH = 32
STEPS = 1500
LR = 3e-3
REPLAY_MIN = 0.80
CARRY_MIN = 0.60
GAIN_MIN = 0.05
RETENTION_MAX = 0.02


def episode(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target_colour = int(torch.randint(0, 6, (1,), generator=generator))
    target_position = int(torch.randint(0, 3, (1,), generator=generator))
    ticks = []
    for tick_index in range(TICKS):
        features = torch.zeros(48)
        for body in range(BODY):
            colour = (
                target_colour
                if (tick_index == 0 and body == 0)
                else int(torch.randint(0, 6, (1,), generator=generator))
            )
            position = (
                target_position
                if (tick_index == 0 and body == 0)
                else int(torch.randint(0, 3, (1,), generator=generator))
            )
            features[body * 8 + 2 + colour] = 1.0
            features[body * 8 + 6 + position] = 1.0
        features[24 + tick_index] = 1.0
        features[29 + (4 - tick_index)] = 1.0
        ticks.append(features)
    query = torch.zeros(8)
    query[7] = 1.0
    return {"ticks": torch.stack(ticks), "query": query, "label": target_position}


def build(count: int, seed: int) -> dict[str, torch.Tensor]:
    rows = [episode(seed + index) for index in range(count)]
    return {
        "ticks": torch.stack([row["ticks"] for row in rows]).float(),
        "queries": torch.stack([row["query"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


class TickRead(nn.Module):
    def __init__(self, token_dim: int, width: int = 64) -> None:
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


def collect_activity(graph, ticks: torch.Tensor, use_mask: torch.Tensor) -> torch.Tensor:
    """ticks [B,T,48], use_mask [T] bool -> activity [B,used,30] with carried state."""
    batch = ticks.shape[0]
    state = graph.initial_state()
    activities = []
    for tick_index in range(TICKS):
        if not bool(use_mask[tick_index]):
            continue
        gates = torch.ones(batch, graph.units.n_units)
        result = graph(channels(ticks[:, tick_index]), gates, state)
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


def mask_for(arm: str) -> torch.Tensor:
    mask = torch.zeros(TICKS, dtype=torch.bool)
    if arm == "no_state":
        mask[4] = True
    elif arm == "replay":
        mask[0] = True
        mask[4] = True
    else:
        mask[:] = True
    return mask


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
    train = build(TRAIN, seed)
    fit = build(FIT, seed + 1)
    arms = {}
    for arm in ("no_state", "replay", "state_carry"):
        mask = mask_for(arm)
        train_activity = collect_activity(graph, train["ticks"], mask).detach()
        fit_activity = collect_activity(graph, fit["ticks"], mask).detach()
        head = TickRead(train_activity.shape[-1])
        train_head(
            head,
            {"activities": train_activity, "queries": train["queries"], "labels": train["labels"]},
            seed + (0 if arm == "no_state" else 1),
        )
        arms[arm] = accuracy(
            head, {"activities": fit_activity, "queries": fit["queries"], "labels": fit["labels"]}
        )
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
    gains = [row["arms"]["state_carry"] - row["arms"]["no_state"] for row in rows]
    gates = {
        "g1_replay": means["replay"] >= REPLAY_MIN,
        "g2_carry": means["state_carry"] - means["no_state"] >= GAIN_MIN
        and means["state_carry"] >= CARRY_MIN
        and all(gain > 0 for gain in gains),
        "g3_retention": all(max(row["retention_damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = (
        "memory_carry_verified"
        if all(gates.values())
        else ("memory_probe_partial" if gates["g1_replay"] else "memory_probe_invalid")
    )
    payload = {
        "schema": "taiji-u2-working-memory-v1",
        "card": "working-memory 5-tick recall with carried unit state",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "ticks": TICKS,
            "train": TRAIN,
            "fit": FIT,
            "steps": STEPS,
            "arms": ["no_state", "replay", "state_carry"],
        },
        "rows": rows,
        "means": means,
        "gains": gains,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "frozen 23.0b parent, two-step attention over per-tick unit activity; state carried through graph next_state",
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
                "gains": gains,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "memory_carry_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
