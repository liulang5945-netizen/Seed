"""23.3: joint content/source/time head on the token interface (0 check).

Inputs are the 23.2c token interface (3 object tokens + direction/kind query); the head
outputs content (4 classes: 3 body indices + unknown), source (2) and time (2) jointly.
Arm A trains the joint head directly; arm B trains with auxiliary per-field heads that
are then removed at evaluation.  Provenance labels are deterministic per seed (fixture
Q4 rows are evaluation-only, so this probe uses Q2/Q3 rows with a frozen provenance
assignment and discloses that provenance is a label, not a Chinese source phrase read).
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
    lr_at,
)
from scripts.training.run_s2_23_2c_attention_20261120 import (  # noqa: E402
    AttentionRead,
    label_for,
    tokens_for,
    train_joint_parent,
    unit_activity,
)
from taiji.native_composition import build_partition  # noqa: E402

OUT = ROOT / "reports/taiji_u2_23_3_generation_20261120.json"
CAP_CPU = 12000.0
SEEDS = (20261120, 20261121, 20261122)
BATCH = 32
LR = 3e-3
STEPS = 1500
CONTENT_MIN = 0.60
PROVENANCE_MIN = 0.80
AUX_GAP_MAX = 0.05
RETENTION_MAX = 0.02


def provenance(row: dict) -> tuple[int, int]:
    return (int(row["seed"]) // 2) % 2, (int(row["seed"]) // 3) % 2


def build_generation_data() -> dict[str, dict[str, torch.Tensor]]:
    def build(partition: str) -> dict[str, torch.Tensor]:
        rows = [row for row in build_partition(partition) if row["kind"] in {"q2", "q3"}]
        tokens = []
        queries = []
        content = []
        source = []
        timestep = []
        for row in rows:
            token, query = tokens_for(row)
            tokens.append(token)
            queries.append(query)
            content.append(label_for(row))
            src, tim = provenance(row)
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


class GenerationRead(AttentionRead):
    def __init__(self, token_dim: int) -> None:
        super().__init__(token_dim)
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
    """Joint head plus auxiliary per-field heads (aux removed at evaluation)."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.joint = GenerationRead(token_dim)
        self.aux_content = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 4))
        self.aux_source = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 2))
        self.aux_time = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 2))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.joint(tokens, query)
        hidden = output["hidden"]
        return {
            "content": output["content"],
            "source": output["source"],
            "time": output["time"],
            "aux_content": self.aux_content(hidden),
            "aux_source": self.aux_source(hidden),
            "aux_time": self.aux_time(hidden),
        }


def batch_step(
    module: nn.Module,
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
    aux: bool,
    step: int,
) -> float:
    index = torch.randint(0, batch["tokens"].shape[0], (BATCH,), generator=generator)
    tokens = batch["tokens"][index]
    queries = batch["queries"][index]
    output = module(tokens, queries)
    loss = nn.functional.cross_entropy(output["content"], batch["content"][index])
    loss = loss + nn.functional.cross_entropy(output["source"], batch["source"][index])
    loss = loss + nn.functional.cross_entropy(output["time"], batch["time"][index])
    if aux:
        loss = loss + nn.functional.cross_entropy(output["aux_content"], batch["content"][index])
        loss = loss + nn.functional.cross_entropy(output["aux_source"], batch["source"][index])
        loss = loss + nn.functional.cross_entropy(output["aux_time"], batch["time"][index])
    return loss


def train_module(module: nn.Module, batch: dict[str, torch.Tensor], seed: int, aux: bool) -> None:
    optimizer = torch.optim.Adam(module.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(seed + 53)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        loss = batch_step(module, batch, generator, aux, step)
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
    (ROOT / ".seed_test_tmp").mkdir(exist_ok=True)
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
        "schema": "taiji-u2-23-3-generation-v1",
        "card": "23.3 joint content/source/time generation on the token interface",
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
        "disclosure": "provenance labels are deterministic per seed (not read from a Chinese source phrase); Q4 rows are evaluation-only and unused for training",
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
