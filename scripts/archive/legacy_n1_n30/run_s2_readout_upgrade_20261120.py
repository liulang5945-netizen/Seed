"""Readout-upgrade card: per-token unit-activity readout vs 4-dim direct query (0 check).

S1 neutrality: adding the activity readout must not change the frozen units' state_dict
digest, the old three-skill readouts, or capsule round-trip.  S2: the three-skill parent is
rebuilt as in 23.0b.  S3: a wider activity readout (30-dim unit activity, hidden 64) is
compared with a 4-dim direct-query readout on Q2/Q3; the pre-registered gate is Q3 >= 0.46
and >= direct + 0.05.  Zero check/final.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
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
    channels,
    lr_at,
)
from scripts.training.run_s2_23_2c_attention_20261120 import (  # noqa: E402
    label_for,
    tokens_for,
    train_joint_parent,
    unit_activity,
)
from taiji.native_composition import build_partition  # noqa: E402
from taiji.u2_neural_units import (  # noqa: E402
    apply_capsule,
    build_capsule,
    load_capsule,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_readout_upgrade_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
STEPS = 2000
BATCH = 32
LR = 3e-3
HIDDEN = 64
Q3_MIN = 0.46
GAIN_MIN = 0.05
RETENTION_MAX = 0.02


class ActivityRead(nn.Module):
    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.query = nn.Sequential(nn.Linear(8, HIDDEN), nn.Tanh())
        self.token = nn.Sequential(nn.Linear(token_dim, HIDDEN), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 4))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        embed = self.token(tokens)
        q = self.query(query)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, q), dim=-1)
        context = torch.einsum("bt,btd->bd", scores, embed)
        return self.out(torch.cat([context, q], dim=-1))


class DirectRead(nn.Module):
    def __init__(self, direct_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(direct_dim + 8, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, 4),
        )

    def forward(self, direct: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([direct, query], dim=-1))


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


def train_module(
    module: nn.Module, x: torch.Tensor, queries: torch.Tensor, labels: torch.Tensor, seed: int
) -> None:
    optimizer = torch.optim.Adam(module.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 151)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, x.shape[0], (BATCH,), generator=generator)
        loss = loss_fn(module(x[index], queries[index]), labels[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def accuracy(
    module: nn.Module,
    x: torch.Tensor,
    queries: torch.Tensor,
    labels: torch.Tensor,
    q3_mask: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        logits = module(x, queries)
    return {
        "q2": float((logits[~q3_mask].argmax(-1) == labels[~q3_mask]).float().mean()),
        "q3": float((logits[q3_mask].argmax(-1) == labels[q3_mask]).float().mean()),
    }


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def run_seed(seed: int) -> dict:
    graph, skill_data = train_joint_parent(seed)
    skill = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    units_before = digest(
        {key: float(value.sum()) for key, value in graph.units.state_dict().items()}
    )
    capsule = build_capsule(graph)
    with tempfile.TemporaryDirectory(dir=ROOT / ".seed_test_tmp") as directory:
        path = save_capsule(Path(directory) / "parent.json", capsule)
        loaded = load_capsule(path)
    # Neutrality after adding the readout modules (units untouched).
    activity_module = ActivityRead(graph.units.n_units)
    direct_module = DirectRead(4)
    units_after = digest(
        {key: float(value.sum()) for key, value in graph.units.state_dict().items()}
    )
    readouts_after = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    fresh_graph, _ = train_joint_parent(seed)
    apply_capsule(fresh_graph, loaded)
    recovery_diff = max(
        abs(float(graph.state_dict()[key].sum()) - float(fresh_graph.state_dict()[key].sum()))
        for key in graph.state_dict()
    )
    train, fit = relation_data()
    train_activity = unit_activity(graph, train["tokens"]).detach()
    fit_activity = unit_activity(graph, fit["tokens"]).detach()

    def direct_query(tokens: torch.Tensor) -> torch.Tensor:
        flat = torch.zeros(tokens.shape[0], 48)
        flat[:, : tokens.shape[1] * tokens.shape[2]] = tokens.reshape(tokens.shape[0], -1)
        gates = torch.ones(tokens.shape[0], graph.units.n_units)
        with torch.no_grad():
            return graph(channels(flat), gates, graph.initial_state())["direct_query"]

    direct_train = direct_query(train["tokens"])
    direct_fit = direct_query(fit["tokens"])
    train_module(activity_module, train_activity, train["queries"], train["labels"], seed)
    activity_scores = accuracy(
        activity_module, fit_activity, fit["queries"], fit["labels"], fit["q3"]
    )
    train_module(direct_module, direct_train, train["queries"], train["labels"], seed + 1)
    direct_scores = accuracy(direct_module, direct_fit, fit["queries"], fit["labels"], fit["q3"])
    return {
        "seed": seed,
        "units_digest_before": units_before,
        "units_digest_after": units_after,
        "old_readouts_after": readouts_after,
        "skill": skill,
        "recovery_diff": recovery_diff,
        "activity": activity_scores,
        "direct": direct_scores,
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
        "s1_units_unchanged": all(
            row["units_digest_before"] == row["units_digest_after"] for row in rows
        ),
        "s1_old_readouts_unchanged": all(
            max(
                abs(row["skill"][skill] - row["old_readouts_after"][skill])
                for skill in ("s1", "s2", "s3")
            )
            <= 1e-6
            for row in rows
        ),
        "s2_three_skills": all(min(row["skill"].values()) >= 0.80 for row in rows),
        "s3_q3": all(row["activity"]["q3"] >= Q3_MIN for row in rows),
        "s3_gain": all(row["activity"]["q3"] - row["direct"]["q3"] >= GAIN_MIN for row in rows),
        "retention": all(
            max(0.0, row["skill"][skill] - row["old_readouts_after"][skill]) <= RETENTION_MAX
            for row in rows
            for skill in ("s1", "s2", "s3")
        ),
    }
    verdict = "readout_upgrade_verified" if all(gates.values()) else "readout_upgrade_partial"
    payload = {
        "schema": "taiji-u2-readout-upgrade-v1",
        "card": "readout upgrade: per-token unit activity vs 4-dim direct query",
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
            "hidden": HIDDEN,
            "q3_min": Q3_MIN,
            "gain_min": GAIN_MIN,
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
    return 0 if verdict == "readout_upgrade_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
