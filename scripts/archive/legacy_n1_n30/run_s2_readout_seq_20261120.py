"""Readout-upgrade minimal check on the Chinese word-order task (0 check).

Compares two readouts over the frozen 23.0b parent's per-character-tick activities:
baseline = mean pool + MLP; upgrade = learned positional embeddings + two-step attention.
Replay (raw sequence, no units) is the upper bound and must reach >= 0.80 before the unit
readouts are interpreted.  Gate: upgrade beats baseline by >= 0.10 and reaches >= 0.60.
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
from scripts.training.run_s2_cn_order_20261120 import (  # noqa: E402
    FIT_OFFSET,
    FIT_VARIANTS,
    MAX_LEN,
    STEPS,
    TRAIN_VARIANTS,
    build,
    make_examples,
)

OUT = ROOT / "reports/taiji_u2_readout_seq_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
REPLAY_MIN = 0.80
UPGRADE_MIN = 0.60
MARGIN = 0.10


class PosAttentionRead(nn.Module):
    def __init__(self, token_dim: int, width: int = 64, positions: int = MAX_LEN) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(positions, token_dim))
        self.token = nn.Sequential(nn.Linear(token_dim, width), nn.Tanh())
        self.query = nn.Sequential(nn.Linear(width, width), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 4))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        embed = self.token(sequence + self.pos[: sequence.shape[1]])
        key = embed.mean(dim=1)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, self.query(key)), dim=-1)
        context = torch.einsum("bt,btd->bd", scores, embed)
        return self.out(torch.cat([context, key], dim=-1))


class PoolRead(nn.Module):
    def __init__(self, token_dim: int, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 4),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.net(sequence.mean(dim=1))


def train_readout(
    head: nn.Module, batch: dict[str, torch.Tensor], sequence_key: str, seed: int
) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, batch[sequence_key].shape[0], (64,), generator=generator)
        loss = loss_fn(head(batch[sequence_key][index]), batch["labels"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def eval_readout(head: nn.Module, batch: dict[str, torch.Tensor], key: str) -> float:
    with torch.no_grad():
        logits = head(batch[key])
    return float((logits.argmax(-1) == batch["labels"]).float().mean())


def run_seed(seed: int) -> dict:
    graph, _ = train_joint_parent(seed)
    skill_data = build_skill_data(seed)
    baseline_skills = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    for parameter in graph.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    train = build(make_examples(seed, range(3), TRAIN_VARIANTS))
    fit = build(make_examples(seed + FIT_OFFSET, range(3, 4), FIT_VARIANTS))

    def unit_seq(sequence: torch.Tensor) -> torch.Tensor:
        state = graph.initial_state()
        activities = []
        for tick_index in range(sequence.shape[1]):
            gates = torch.ones(sequence.shape[0], graph.units.n_units)
            with torch.no_grad():
                result = graph(channels(sequence[:, tick_index]), gates, state)
            activities.append(result["unit_result"]["activity"])
            state = result["next_state"]
        return torch.stack(activities, dim=1)

    arms = {}
    train_units = unit_seq(train["seq"])
    fit_units = unit_seq(fit["seq"])
    for name, head_class in (("baseline_mean", PoolRead), ("upgrade_posattn", PosAttentionRead)):
        head = head_class(train_units.shape[-1])
        train_readout(head, {"sequence": train_units, "labels": train["labels"]}, "sequence", seed)
        arms[name] = eval_readout(
            head, {"sequence": fit_units, "labels": fit["labels"]}, "sequence"
        )
    raw_train = train["seq"].reshape(train["seq"].shape[0], MAX_LEN, -1)
    raw_fit = fit["seq"].reshape(fit["seq"].shape[0], MAX_LEN, -1)
    replay = PosAttentionRead(raw_train.shape[-1])
    train_readout(replay, {"sequence": raw_train, "labels": train["labels"]}, "sequence", seed)
    arms["replay"] = eval_readout(
        replay, {"sequence": raw_fit, "labels": fit["labels"]}, "sequence"
    )
    after = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    damage = {skill: max(0.0, baseline_skills[skill] - after[skill]) for skill in baseline_skills}
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
        "replay_learnable": means["replay"] >= REPLAY_MIN,
        "upgrade_gain": means["upgrade_posattn"] >= means["baseline_mean"] + MARGIN,
        "upgrade_level": means["upgrade_posattn"] >= UPGRADE_MIN,
        "retention": all(max(row["retention_damage"].values()) <= 0.02 for row in rows),
    }
    verdict = "readout_seq_verified" if all(gates.values()) else "readout_seq_partial"
    payload = {
        "schema": "taiji-u2-readout-seq-v1",
        "card": "readout upgrade on Chinese word-order task: mean pool vs positional attention",
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
            "steps": STEPS,
            "margin": MARGIN,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "frozen parent per-tick activities; replay = same positional-attention readout on raw character features",
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
    return 0 if verdict == "readout_seq_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
