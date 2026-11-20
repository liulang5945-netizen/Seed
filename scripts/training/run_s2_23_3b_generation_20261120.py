"""23.3b: joint content/source/time generation with provenance carried by object tokens (0 check).

Fix of 23.3: provenance values are encoded into each object token (source bit, time bit)
so the head must retrieve the provenance of the relation-answer object; the query carries
direction/kind only.  Content = Q2/Q3 answer (4 classes), source/time = that object's tag.
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
from scripts.training.run_s2_23_0b_joint_20261120 import (  # noqa: E402
    accuracy as skill_accuracy,
)
from scripts.training.run_s2_23_0b_joint_20261120 import (
    channels,  # noqa: E402
    lr_at,
)
from scripts.training.run_s2_23_2c_attention_20261120 import (  # noqa: E402
    label_for,
    train_joint_parent,
)
from taiji.native_composition import build_partition, world_scene  # noqa: E402
from taiji.u2_task_episode import NAMES  # noqa: E402

OUT = ROOT / "reports/taiji_u2_23_3b_generation_20261120.json"
CAP_CPU = 12000.0
SEEDS = (20261120, 20261121, 20261122)
BATCH = 32
LR = 3e-3
STEPS = 1500
CONTENT_MIN = 0.60
PROVENANCE_MIN = 0.80
AUX_GAP_MAX = 0.05
RETENTION_MAX = 0.02
DIRECTION_INDEX = {"左边": 0, "右边": 1, "前边": 2, "后边": 3}
KIND_INDEX = {"q2": 0, "q3": 1}


def token_feature(row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    bodies = world_scene(row["seed"])["bodies"]
    tokens = torch.zeros(3, 12)
    for index, body in enumerate(bodies):
        tokens[index, 0] = float(body["position"][0])
        tokens[index, 1] = float(body["position"][1])
        tokens[index, 2 + NAMES.index(body["name"])] = 1.0
        tokens[index, 8] = float((int(row["seed"]) + index) % 2)
        tokens[index, 9] = float((int(row["seed"]) // 2 + index) % 2)
    query = torch.zeros(12)
    direction = row["axes"]["direction"]
    query[DIRECTION_INDEX[direction]] = 1.0
    pair = row["axes"]["pair_order"]
    if pair is not None:
        query[4 + DIRECTION_INDEX[pair[1]]] = 1.0
    query[8 + KIND_INDEX[row["kind"]]] = 1.0
    return tokens, query


def unit_activity(graph, tokens: torch.Tensor) -> torch.Tensor:
    flat = torch.zeros(tokens.shape[0] * tokens.shape[1], 48)
    flat[:, : tokens.shape[-1]] = tokens.reshape(-1, tokens.shape[-1])
    gates = torch.ones(flat.shape[0], graph.units.n_units)
    result = graph(channels(flat), gates, graph.initial_state())
    return result["unit_result"]["activity"].reshape(tokens.shape[0], tokens.shape[1], -1)


def tag_of(row: dict, index: int) -> tuple[int, int]:
    return (int(row["seed"]) + index) % 2, (int(row["seed"]) // 2 + index) % 2


class GenerationRead(nn.Module):
    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.query1 = nn.Sequential(nn.Linear(12, 16), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(28, 16), nn.Tanh())
        self.token = nn.Sequential(nn.Linear(token_dim, 16), nn.Tanh())
        self.content = nn.Linear(32, 4)
        self.source = nn.Linear(32, 2)
        self.time = nn.Linear(32, 2)

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> dict[str, torch.Tensor]:
        embed = self.token(tokens)
        q1 = self.query1(query)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, q1), dim=-1)
        context1 = torch.einsum("bt,btd->bd", scores, embed)
        q2 = self.query2(torch.cat([context1, query], dim=-1))
        scores2 = torch.softmax(torch.einsum("btd,bd->bt", embed, q2), dim=-1)
        context2 = torch.einsum("bt,btd->bd", scores2, embed)
        hidden = torch.cat([context2, q1], dim=-1)
        return {
            "content": self.content(hidden),
            "source": self.source(hidden),
            "time": self.time(hidden),
            "hidden": hidden,
        }


class AuxGeneration(nn.Module):
    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.joint = GenerationRead(token_dim)
        self.aux_source = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 2))
        self.aux_time = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 2))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.joint(tokens, query)
        hidden = output["hidden"]
        output["aux_source"] = self.aux_source(hidden)
        output["aux_time"] = self.aux_time(hidden)
        return output


def build_generation_data() -> dict[str, dict[str, torch.Tensor]]:
    def build(partition: str) -> dict[str, torch.Tensor]:
        rows = [row for row in build_partition(partition) if row["kind"] in {"q2", "q3"}]
        tokens = []
        queries = []
        content = []
        source = []
        timestep = []
        for row in rows:
            token, query = token_feature(row)
            tokens.append(token)
            queries.append(query)
            label = label_for(row)
            content.append(label)
            src, tim = tag_of(row, label) if label < 3 else (0, 0)
            source.append(src)
            timestep.append(tim)
        return {
            "tokens": torch.stack(tokens).float(),
            "queries": torch.stack(queries).float(),
            "content": torch.tensor(content),
            "source": torch.tensor(source),
            "time": torch.tensor(timestep),
        }

    return {"train": build("train"), "fit": build("fit")}


def train_module(module: nn.Module, batch: dict[str, torch.Tensor], seed: int, aux: bool) -> None:
    optimizer = torch.optim.Adam(module.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(seed + 61)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, batch["tokens"].shape[0], (BATCH,), generator=generator)
        output = module(batch["tokens"][index], batch["queries"][index])
        loss = nn.functional.cross_entropy(output["content"], batch["content"][index])
        loss = loss + nn.functional.cross_entropy(output["source"], batch["source"][index])
        loss = loss + nn.functional.cross_entropy(output["time"], batch["time"][index])
        if aux:
            loss = loss + nn.functional.cross_entropy(output["aux_source"], batch["source"][index])
            loss = loss + nn.functional.cross_entropy(output["aux_time"], batch["time"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate(module: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    with torch.no_grad():
        output = module(batch["tokens"], batch["queries"])
    return {
        "content": float((output["content"].argmax(-1) == batch["content"]).float().mean()),
        "source": float((output["source"].argmax(-1) == batch["source"]).float().mean()),
        "time": float((output["time"].argmax(-1) == batch["time"]).float().mean()),
    }


def run_seed(seed: int) -> dict:
    graph, skill_data = train_joint_parent(seed)
    baseline = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    data = build_generation_data()
    train = data["train"]
    fit = data["fit"]
    cross_train = unit_activity(graph, train["tokens"]).detach()
    cross_fit = unit_activity(graph, fit["tokens"]).detach()
    dim = cross_train.shape[-1]
    joint = GenerationRead(dim)
    train_module(joint, {**train, "tokens": cross_train}, seed, aux=False)
    arm_a = evaluate(joint, {**fit, "tokens": cross_fit})
    aux_module = AuxGeneration(dim)
    train_module(aux_module, {**train, "tokens": cross_train}, seed + 1, aux=True)
    arm_b = evaluate(aux_module, {**fit, "tokens": cross_fit})
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
        "arm_a": arm_a,
        "arm_b": arm_b,
        "aux_gap": abs(arm_b["content"] - arm_a["content"]),
        "retention": retention,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    gates = {
        "g1_content": all(row["arm_a"]["content"] >= CONTENT_MIN for row in rows),
        "g2_provenance": all(
            row["arm_a"]["source"] >= PROVENANCE_MIN and row["arm_a"]["time"] >= PROVENANCE_MIN
            for row in rows
        ),
        "g3_aux_gap": all(row["aux_gap"] <= AUX_GAP_MAX for row in rows),
        "g4_retention": all(
            max(row["retention"]["damage"].values()) <= RETENTION_MAX for row in rows
        ),
    }
    if all(gates.values()):
        verdict = "generation_bounded_verified"
    elif gates["g1_content"]:
        verdict = "generation_bounded_partial"
    else:
        verdict = "generation_bounded_negative"
    payload = {
        "schema": "taiji-u2-23-3b-generation-v1",
        "card": "23.3b joint content/source/time generation with provenance-bearing tokens",
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
            "content_min": CONTENT_MIN,
            "provenance_min": PROVENANCE_MIN,
            "aux_gap_max": AUX_GAP_MAX,
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
    return 0 if verdict == "generation_bounded_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
