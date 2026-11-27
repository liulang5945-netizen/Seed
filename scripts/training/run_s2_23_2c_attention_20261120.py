"""23.2c: token/attention relation read - per-object tokens plus a direction query (0 check).

Cross-unit arm: each object token is fed through the frozen 23.0b parent and the attention
module reads the 30-dim unit activity vectors (query-conditioned, two attention steps for
two-hop).  Direct arm: the same attention module reads raw 8-dim object tokens (x, y,
colour one-hot) without units.  Held-out direction axis is the same as 23.2/23.2b.
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
from scripts.training.run_s2_23_0b_joint_20261120 import (
    STEPS as SKILL_STEPS,
)
from scripts.training.run_s2_23_0b_joint_20261120 import (  # noqa: E402
    accuracy as skill_accuracy,
)
from scripts.training.run_s2_23_0b_joint_20261120 import (
    build_data as build_skill_data,
)
from scripts.training.run_s2_23_0b_joint_20261120 import (
    build_graph,
    channels,
    head_logits,
    lr_at,
)
from taiji.native_composition import build_partition, world_scene  # noqa: E402
from taiji.u2_task_episode import NAMES  # noqa: E402

OUT = ROOT / "reports/taiji_u2_23_2c_attention_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
DIRECTION_INDEX = {"左边": 0, "右边": 1, "前边": 2, "后边": 3}
KIND_INDEX = {"q2": 0, "q3": 1}
BATCH = 32
LR = 3e-3
REL_STEPS = 1500
RETENTION_MAX = 0.02


def tokens_for(row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    bodies = world_scene(row["seed"])["bodies"]
    tokens = torch.zeros(3, 8)
    for index, body in enumerate(bodies):
        tokens[index, 0] = float(body["position"][0])
        tokens[index, 1] = float(body["position"][1])
        tokens[index, 2 + NAMES.index(body["name"])] = 1.0
    query = torch.zeros(8)
    direction = row["axes"]["direction"]
    query[DIRECTION_INDEX[direction]] = 1.0
    pair = row["axes"]["pair_order"]
    if pair is not None:
        query[4 + DIRECTION_INDEX[pair[1]]] = 1.0
    query[6 + KIND_INDEX[row["kind"]]] = 1.0
    return tokens, query


def label_for(row: dict) -> int:
    names = [body["name"] for body in world_scene(row["seed"])["bodies"]]
    return names.index(row["answer"]) if row["answer"] in names else 3


def relation_data() -> dict[str, torch.Tensor]:
    def build(partition: str) -> dict[str, torch.Tensor]:
        rows = [row for row in build_partition(partition) if row["kind"] in {"q2", "q3"}]
        tokens = []
        queries = []
        for row in rows:
            token, query = tokens_for(row)
            tokens.append(token)
            queries.append(query)
        return {
            "tokens": torch.stack(tokens).float(),
            "queries": torch.stack(queries).float(),
            "labels": torch.tensor([label_for(row) for row in rows]),
            "kinds": torch.tensor([row["kind"] == "q2" for row in rows]),
        }

    return {"train": build("train"), "fit": build("fit")}


class AttentionRead(nn.Module):
    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.query1 = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(24, 16), nn.Tanh())
        self.token = nn.Sequential(nn.Linear(token_dim, 16), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(32, 24), nn.Tanh(), nn.Linear(24, 4))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        embed = self.token(tokens)
        q1 = self.query1(query)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, q1), dim=-1)
        context1 = torch.einsum("bt,btd->bd", scores, embed)
        q2 = self.query2(torch.cat([context1, query], dim=-1))
        scores2 = torch.softmax(torch.einsum("btd,bd->bt", embed, q2), dim=-1)
        context2 = torch.einsum("bt,btd->bd", scores2, embed)
        return self.out(torch.cat([context2, q1], dim=-1))


def train_joint_parent(seed: int):
    skill_data = build_skill_data(seed)
    graph = build_graph(seed)
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 7)
    schedule = ["s1"] * SKILL_STEPS["s1"] + ["s2"] * SKILL_STEPS["s2"] + ["s3"] * SKILL_STEPS["s3"]
    shuffle = torch.randperm(len(schedule), generator=generator).tolist()
    features = {"s1": skill_data["x1"], "s2": skill_data["x2"], "s3": skill_data["x3"]}
    labels = {"s1": skill_data["y1"], "s2": skill_data["y2"], "s3": skill_data["y3"]}
    heads = {"s1": 0, "s2": 1, "s3": 2}
    for step, position in enumerate(shuffle):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, len(schedule))
        skill = schedule[position]
        index = torch.randint(0, features[skill].shape[0], (BATCH,), generator=generator)
        logits = head_logits(graph, features[skill][index], heads[skill])
        k = int(labels[skill].max()) + 1
        loss = loss_fn(logits[:, :k], labels[skill][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()
    for parameter in graph.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return graph, skill_data


def unit_activity(graph, tokens: torch.Tensor) -> torch.Tensor:
    flat = torch.zeros(tokens.shape[0] * tokens.shape[1], 48)
    flat[:, :8] = tokens.reshape(-1, 8)
    gates = torch.ones(flat.shape[0], graph.units.n_units)
    result = graph(channels(flat), gates, graph.initial_state())
    return result["unit_result"]["activity"].reshape(tokens.shape[0], tokens.shape[1], -1)


def train_module(
    module: nn.Module,
    tokens: torch.Tensor,
    queries: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    seed: int,
) -> None:
    optimizer = torch.optim.Adam(module.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 41)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, steps)
        index = torch.randint(0, tokens.shape[0], (BATCH,), generator=generator)
        loss = loss_fn(module(tokens[index], queries[index]), labels[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def accuracy(
    module: nn.Module,
    tokens: torch.Tensor,
    queries: torch.Tensor,
    labels: torch.Tensor,
    kinds: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        logits = module(tokens, queries)
    result = {}
    for name, mask in (("q2", kinds), ("q3", ~kinds)):
        result[name] = float((logits[mask].argmax(-1) == labels[mask]).float().mean())
    return result


def run_seed(seed: int) -> dict:
    graph, skill_data = train_joint_parent(seed)
    baseline = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    data = relation_data()
    train, fit = data["train"], data["fit"]
    cross_train = unit_activity(graph, train["tokens"]).detach()
    cross_fit = unit_activity(graph, fit["tokens"]).detach()
    cross_module = AttentionRead(cross_train.shape[-1])
    direct_module = AttentionRead(8)
    train_module(cross_module, cross_train, train["queries"], train["labels"], REL_STEPS, seed)
    train_module(
        direct_module, train["tokens"], train["queries"], train["labels"], REL_STEPS, seed + 1
    )
    cross = accuracy(cross_module, cross_fit, fit["queries"], fit["labels"], fit["kinds"])
    direct = accuracy(direct_module, fit["tokens"], fit["queries"], fit["labels"], fit["kinds"])
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
        "cross": cross,
        "direct": direct,
        "q3_gain": cross["q3"] - direct["q3"],
        "q2_gain": cross["q2"] - direct["q2"],
        "retention": retention,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    gains = [row["q3_gain"] for row in rows]
    mean_gain = sum(gains) / len(gains)
    gates = {
        "q3_accuracy": all(row["cross"]["q3"] >= 0.40 for row in rows),
        "q3_gain": mean_gain >= 0.05 and all(gain > 0 for gain in gains),
        "retention": all(max(row["retention"]["damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = "relation_read_verified" if all(gates.values()) else "relation_read_partial"
    payload = {
        "schema": "taiji-u2-23-2c-attention-v1",
        "card": "23.2c token/attention relation read with cross-unit vs direct ablation",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "relation_steps": REL_STEPS,
            "batch": BATCH,
            "freeze_parent": True,
            "two_hop_attention": True,
        },
        "rows": rows,
        "gates": gates,
        "verdict": verdict,
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
    return 0 if verdict == "relation_read_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
