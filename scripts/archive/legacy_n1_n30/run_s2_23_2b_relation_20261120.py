"""23.2b: two-hop relation reading with continuous-coordinate features (0 check).

Rebuilds the jointly trained three-skill parent (23.0b protocol), freezes its shared units
and old heads, then trains a relation read head on Q2/Q3 from the frozen composition
fixture; the ablation arm trains an equally sized head directly on the 48-dim feature
vector (no units).  Gates: two-hop accuracy and cross-unit gain over the direct arm,
plus old-skill retention.  Zero check/final.
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
from scripts.training.run_s2_23_0b_joint_20261120 import (
    accuracy as skill_accuracy,
)
from scripts.training.run_s2_23_0b_joint_20261120 import (  # noqa: E402
    build_data as build_skill_data,
)
from scripts.training.run_s2_23_0b_joint_20261120 import (
    build_graph,
    channels,  # noqa: E402
    head_logits,
    lr_at,
)
from taiji.native_composition import build_partition, world_scene  # noqa: E402
from taiji.u2_task_episode import NAMES  # noqa: E402

OUT = ROOT / "reports/taiji_u2_23_2b_relation_20261120.json"
CAP_CPU = 8000.0
SEEDS = (20261120, 20261121, 20261122)
DIRECTION_INDEX = {"左边": 0, "右边": 1, "前边": 2, "后边": 3}
KIND_INDEX = {"q2": 0, "q3": 1}
BATCH = 32
LR = 3e-3
REL_STEPS = 1500
RETENTION_MAX = 0.02


def feature_for(row: dict) -> torch.Tensor:
    scene = world_scene(row["seed"])
    bodies = scene["bodies"]
    vector = torch.zeros(48)
    for index, body in enumerate(bodies):
        colour = NAMES.index(body["name"])
        vector[index * 2] = float(body["position"][0])
        vector[index * 2 + 1] = float(body["position"][1])
        vector[6 + index * 6 + colour] = 1.0
    direction = row["axes"]["direction"]
    if direction in DIRECTION_INDEX:
        vector[30 + DIRECTION_INDEX[direction]] = 1.0
    pair = row["axes"]["pair_order"]
    if pair is not None:
        vector[34 + DIRECTION_INDEX[pair[1]]] = 1.0
    vector[38 + KIND_INDEX[row["kind"]]] = 1.0
    return vector


def label_for(row: dict) -> int:
    names = [body["name"] for body in world_scene(row["seed"])["bodies"]]
    return names.index(row["answer"]) if row["answer"] in names else 3


def relation_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    rows_train = [row for row in build_partition("train") if row["kind"] in {"q2", "q3"}]
    rows_fit = [row for row in build_partition("fit") if row["kind"] in {"q2", "q3"}]
    x_train = torch.stack([feature_for(row) for row in rows_train]).float()
    y_train = torch.tensor([label_for(row) for row in rows_train])
    x_fit = torch.stack([feature_for(row) for row in rows_fit]).float()
    y_fit = torch.tensor([label_for(row) for row in rows_fit])
    q2 = torch.tensor([row["kind"] == "q2" for row in rows_fit])
    q3 = torch.tensor([row["kind"] == "q3" for row in rows_fit])
    return x_train, y_train, x_fit, y_fit, {"q2": q2, "q3": q3}


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


def train_relation_head(
    head: nn.Module, features: torch.Tensor, labels: torch.Tensor, steps: int, seed: int
) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 31)
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, steps)
        index = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        loss = loss_fn(head(features[index]), labels[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate(
    head: nn.Module, features: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> float:
    with torch.no_grad():
        logits = head(features)
    return float((logits[mask].argmax(-1) == labels[mask]).float().mean())


def run_seed(seed: int) -> dict:
    graph, skill_data = train_joint_parent(seed)
    baseline = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    x_train, y_train, x_fit, y_fit, masks = relation_data()
    with torch.no_grad():
        gates = torch.ones(x_train.shape[0], graph.units.n_units)
        direct_train = graph(channels(x_train), gates, graph.initial_state())["direct_query"]
        gates_fit = torch.ones(x_fit.shape[0], graph.units.n_units)
        direct_fit = graph(channels(x_fit), gates_fit, graph.initial_state())["direct_query"]
    cross_head = nn.Sequential(nn.Linear(direct_train.shape[1], 32), nn.Tanh(), nn.Linear(32, 4))
    direct_head = nn.Sequential(nn.Linear(48, 32), nn.Tanh(), nn.Linear(32, 4))
    train_relation_head(cross_head, direct_train, y_train, REL_STEPS, seed)
    train_relation_head(direct_head, x_train, y_train, REL_STEPS, seed + 1)
    cross_q3 = evaluate(cross_head, direct_fit, y_fit, masks["q3"])
    cross_q2 = evaluate(cross_head, direct_fit, y_fit, masks["q2"])
    direct_q3 = evaluate(direct_head, x_fit, y_fit, masks["q3"])
    direct_q2 = evaluate(direct_head, x_fit, y_fit, masks["q2"])
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
        "cross_q2": cross_q2,
        "cross_q3": cross_q3,
        "direct_q2": direct_q2,
        "direct_q3": direct_q3,
        "q3_gain": cross_q3 - direct_q3,
        "q2_gain": cross_q2 - direct_q2,
        "retention": retention,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    q3_gains = [row["q3_gain"] for row in rows]
    mean_gain = sum(q3_gains) / len(q3_gains)
    same_direction = all(gain > 0 for gain in q3_gains)
    gates = {
        "q3_accuracy": all(row["cross_q3"] >= 0.40 for row in rows),
        "q3_gain": mean_gain >= 0.05 and same_direction,
        "retention": all(max(row["retention"]["damage"].values()) <= RETENTION_MAX for row in rows),
    }
    verdict = "relation_read_verified" if all(gates.values()) else "relation_read_partial"
    payload = {
        "schema": "taiji-u2-23-2-relation-v1",
        "card": "23.2b two-hop relation reading with continuous-coordinate features",
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
