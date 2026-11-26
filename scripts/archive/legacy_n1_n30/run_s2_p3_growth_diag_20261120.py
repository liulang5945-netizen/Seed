"""P3 growth diagnostic: 30 vs 48 units on the counting pressure task (0 check, diagnostic only).

Builds two parents under the 23.0b joint protocol (30 = 6x5, 48 = 12x4), then trains the
same two-step attention readout on the balanced counting task for both parents.  D1 asks
whether 48 units beat 30 units by >=0.05 (3 seeds, same direction).  This is a diagnostic
only: it does not sign growth benefit or open any 25 growth arm.  Zero check/final.
Authorised under the standing recommendation-default (recorded in the decision ledger).
"""

from __future__ import annotations

import argparse
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
    channels,
    lr_at,
)
from scripts.training.run_s2_pressure_counting_20261120 import (  # noqa: E402
    FIT,
    TRAIN,
    TwoStepRead,
    build,
    make_rows,
)
from taiji.u2_multihead import MultiHeadGraph  # noqa: E402
from taiji.u2_neural_units import (  # noqa: E402
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    load_capsule,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_p3_growth_diag_20261120.json"
CAP_CPU = 1000.0
SEEDS = (20261120, 20261121, 20261122)
BATCH = 32
LR = 3e-3
COUNT_STEPS = 1500
GAIN_MIN = 0.05
RETENTION_MAX = 0.02
ARMS = {"units30": (6, 5), "units48": (12, 4)}


def build_graph(seed: int, groups: int, units_per_group: int) -> MultiHeadGraph:
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=groups,
        units_per_group=units_per_group,
        input_dim=8,
        hidden_dim=8,
        output_dim=24,
        seed=seed,
    )
    return MultiHeadGraph(spec, heads=3)


def head_logits(graph: MultiHeadGraph, features: torch.Tensor, head: int) -> torch.Tensor:
    gates = torch.ones(features.shape[0], graph.units.n_units)
    return graph(channels(features), gates, graph.initial_state(), head=head)["readout"]


def train_parent(seed: int, groups: int, units_per_group: int):
    data = build_skill_data(seed)
    graph = build_graph(seed, groups, units_per_group)
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 7)
    schedule = ["s1"] * SKILL_STEPS["s1"] + ["s2"] * SKILL_STEPS["s2"] + ["s3"] * SKILL_STEPS["s3"]
    shuffle = torch.randperm(len(schedule), generator=generator).tolist()
    features = {"s1": data["x1"], "s2": data["x2"], "s3": data["x3"]}
    labels = {"s1": data["y1"], "s2": data["y2"], "s3": data["y3"]}
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
    skills = {
        "s1": skill_accuracy(graph, data["x1f"], data["y1f"], 0),
        "s2": skill_accuracy(graph, data["x2f"], data["y2f"], 1),
        "s3": skill_accuracy(graph, data["x3f"], data["y3f"], 2),
    }
    return graph, data, skills


def unit_activity(graph, tokens: torch.Tensor) -> torch.Tensor:
    flat = torch.zeros(tokens.shape[0] * tokens.shape[1], 48)
    flat[:, : tokens.shape[2]] = tokens.reshape(tokens.shape[0] * tokens.shape[1], tokens.shape[2])
    gates = torch.ones(flat.shape[0], graph.units.n_units)
    result = graph(channels(flat), gates, graph.initial_state())
    return result["unit_result"]["activity"].reshape(tokens.shape[0], tokens.shape[1], -1)


def train_counting(head: nn.Module, batch: dict[str, torch.Tensor], seed: int) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 211)
    for step in range(COUNT_STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, COUNT_STEPS)
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


def run_seed(seed: int) -> dict:
    train_rows = make_rows(TRAIN, seed)
    fit_rows = make_rows(FIT, seed + 1)
    train = build(train_rows)
    fit = build(fit_rows)
    arms = {}
    for name, (groups, per_group) in ARMS.items():
        graph, skill_data, skills = train_parent(seed, groups, per_group)
        capsule = build_capsule(graph)
        with tempfile.TemporaryDirectory(dir=ROOT / ".seed_test_tmp") as directory:
            path = save_capsule(Path(directory) / "parent.json", capsule)
            loaded = load_capsule(path)
        fresh = build_graph(seed, groups, per_group)
        apply_capsule(fresh, loaded)
        recovery_diff = max(
            abs(float(graph.state_dict()[key].sum()) - float(fresh.state_dict()[key].sum()))
            for key in graph.state_dict()
        )
        activity_train = unit_activity(graph, train["tokens"]).detach()
        activity_fit = unit_activity(graph, fit["tokens"]).detach()
        head = TwoStepRead(activity_train.shape[-1])
        train_counting(head, {**train, "tokens": activity_train}, seed)
        arms[name] = {
            "fit": accuracy(head, {**fit, "tokens": activity_fit}),
            "skills": skills,
            "recovery_diff": recovery_diff,
        }
    return {"seed": seed, "arms": arms}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    (ROOT / ".seed_test_tmp").mkdir(exist_ok=True)
    rows = [run_seed(seed) for seed in SEEDS]
    gains = [row["arms"]["units48"]["fit"] - row["arms"]["units30"]["fit"] for row in rows]
    mean_gain = sum(gains) / len(gains)
    gates = {
        "d1_gain": mean_gain >= GAIN_MIN and all(gain > 0 for gain in gains),
        "d2_skills": all(
            min(row["arms"]["units48"]["skills"].values()) >= 0.80
            and row["arms"]["units48"]["recovery_diff"] <= 1e-10
            for row in rows
        ),
        "d3_retention": all(min(row["arms"]["units48"]["skills"].values()) >= 0.80 for row in rows),
    }
    verdict = "pressure_measurable" if all(gates.values()) else "pressure_not_structural"
    payload = {
        "schema": "taiji-u2-p3-growth-diag-v1",
        "card": "P3 growth diagnostic: 30 vs 48 units on balanced counting",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "arms": {name: list(spec) for name, spec in ARMS.items()},
            "count_steps": COUNT_STEPS,
            "gain_min": GAIN_MIN,
        },
        "rows": rows,
        "gains": gains,
        "mean_gain": mean_gain,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "diagnostic only under standing recommendation-default; does not sign growth benefit or open any 25 growth arm; 48-unit is a new candidate version",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "gains": gains,
                "mean_gain": mean_gain,
                "arms": [[row["arms"][name]["fit"] for name in ARMS] for row in rows],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "pressure_measurable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
