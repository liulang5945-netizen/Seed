"""Unit-level differentiation probe: 2 tasks, 2 task-bound newborn units (0 check).

Task A (joint24) trains the shared 30 units; then 2 newborn units are bound to task
B (action bins) and gated on only for B.  Checks the three D-20261119-87 conditions:
A retention exact, B learnable, ablating the bound units collapses only B, and the
bound units' activity is task-selective (modulation R >= 1.5).  3 seeds; cap 150 CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (
    adapter_v3_features,  # noqa: E402
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from scripts.training.probe_s2_action_consequence_20261120 import build_partition  # noqa: E402
from taiji.u2_neural_birth_multi import live_entry_birth_multi  # noqa: E402
from taiji.u2_neural_units import UnitGraphSpec  # noqa: E402
from taiji.u2_task_bound_units import TaskBoundGraph, masked_unit_step  # noqa: E402

OUT = ROOT / "reports/taiji_u2_unit_differentiation_20261120.json"
CAP_CPU = 150.0
OLD_WORLDS = 256
OLD_FIT = 32
NEW_WORLDS = 256
NEW_FIT = 32
OLD_STEPS = 1000
NEW_STEPS = 600
BATCH = 64
LR = 3e-3
WARMUP = 100
SEEDS = (0, 1, 2)
BOUND = 2
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
A_RETENTION_MAX = 1e-9
B_MIN = 0.433
ABLATE_DROP_MIN = 0.20
MODULATION_MIN = 1.5


def learning_rate(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def build_graph(seed: int) -> TaskBoundGraph:
    torch.manual_seed(20261120 + seed)
    spec = UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=24, seed=20261120 + seed
    )
    return TaskBoundGraph(spec, heads=2, head_hidden=16)


def train_task_a(graph: TaskBoundGraph, features, labels, seed: int) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(OLD_STEPS):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, OLD_STEPS)
        index = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        gates = torch.ones(BATCH, graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        logits = graph(channels(features[index]), gates, graph.initial_state(), head=0)["readout"]
        loss = loss_fn(logits, labels[index])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def accuracy(graph, features, labels, head: int, classes: int) -> float:
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        logits = graph(channels(features), gates, graph.initial_state(), head=head)["readout"]
    return float((logits[:, :classes].argmax(-1) == labels).float().mean())


def bound_activity(graph, features, head: int, bound: list[int]) -> torch.Tensor:
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        inputs = graph._encode(channels(features), head)
        active = graph.units.active_gate(gates * graph.gate_mask(head))
        result = graph.units(inputs, active, graph.initial_state())
    return result["activity"][:, bound].mean(dim=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    old_examples = generate_examples("train", OLD_WORLDS)
    old_fit_examples = generate_examples("fit", OLD_FIT)
    old_features = torch.stack([adapter_v3_features(ex) for ex in old_examples]).float()
    old_labels = torch.tensor([joint_label(ex) for ex in old_examples])
    old_fit_features = torch.stack([adapter_v3_features(ex) for ex in old_fit_examples]).float()
    old_fit_labels = torch.tensor([joint_label(ex) for ex in old_fit_examples])
    rows = build_partition("train", NEW_WORLDS + NEW_FIT, 20261120)
    new_features = torch.stack([row["features"] for row in rows[:NEW_WORLDS]])
    new_labels = torch.tensor([row["final_bin"] for row in rows[:NEW_WORLDS]])
    new_fit_features = torch.stack([row["features"] for row in rows[NEW_WORLDS:]])
    new_fit_labels = torch.tensor([row["final_bin"] for row in rows[NEW_WORLDS:]])
    results = []
    for seed in SEEDS:
        graph = build_graph(seed)
        train_task_a(graph, old_features, old_labels, seed)
        old_before = accuracy(graph, old_fit_features, old_fit_labels, head=0, classes=24)
        parents = [unit["unit_id"] for unit in graph.units.identities["units"][:1]]
        grown, _ = live_entry_birth_multi(graph.units, 0, parents, count=BOUND)
        graph.units = grown
        born = list(range(30, 30 + BOUND))
        graph.bind_born_units(1, born, list(range(30)))
        for unit_id in born:
            graph.units.set_lifecycle(f"u{unit_id:04d}", "mature")
        generator = torch.Generator().manual_seed(20261120 + seed + 3)
        masked_unit_step(graph, 1, new_features, new_labels, born, NEW_STEPS, LR, generator, BATCH)
        old_after = accuracy(graph, old_fit_features, old_fit_labels, head=0, classes=24)
        new_after = accuracy(graph, new_fit_features, new_fit_labels, head=1, classes=3)
        activity_a = bound_activity(graph, old_fit_features, 0, born)
        activity_b = bound_activity(graph, new_fit_features, 1, born)
        with torch.no_grad():
            within = (activity_a.var() + activity_b.var()) / 2 + 1e-9
            between = (activity_a.mean() - activity_b.mean()) ** 2
            modulation = float(between / within)
        saved = {
            "p": graph.units.p.detach().clone(),
            "r": graph.units.r.detach().clone(),
        }
        with torch.no_grad():
            graph.units.p[:, born] = 0.0
            graph.units.r[born, :] = 0.0
        new_ablated = accuracy(graph, new_fit_features, new_fit_labels, head=1, classes=3)
        old_ablated = accuracy(graph, old_fit_features, old_fit_labels, head=0, classes=24)
        with torch.no_grad():
            graph.units.p.copy_(saved["p"])
            graph.units.r.copy_(saved["r"])
        results.append(
            {
                "seed": seed,
                "old_before": round(old_before, 4),
                "old_after": round(old_after, 4),
                "old_retention_drop": round(old_before - old_after, 6),
                "new_accuracy": round(new_after, 4),
                "new_ablated": round(new_ablated, 4),
                "ablate_drop": round(new_after - new_ablated, 6),
                "old_under_ablation_drop": round(old_after - old_ablated, 6),
                "bound_modulation": round(modulation, 4),
            }
        )
    gates = {
        "old_retention": all(result["old_retention_drop"] <= A_RETENTION_MAX for result in results),
        "new_learnable": all(result["new_accuracy"] >= B_MIN for result in results),
        "necessary_bound_units": all(
            result["ablate_drop"] >= ABLATE_DROP_MIN for result in results
        ),
        "isolation_under_ablation": all(
            result["old_under_ablation_drop"] <= A_RETENTION_MAX for result in results
        ),
        "bound_units_selective": all(
            result["bound_modulation"] >= MODULATION_MIN for result in results
        ),
    }
    if all(gates.values()):
        verdict = "unit_differentiation_ready"
    elif (
        gates["old_retention"]
        and gates["new_learnable"]
        and not (gates["necessary_bound_units"] and gates["bound_units_selective"])
    ):
        verdict = "unit_differentiation_partial"
    else:
        verdict = "unit_differentiation_absent"
    payload = {
        "schema": "taiji-u2-unit-differentiation-v1",
        "card": "task-bound newborn units: necessity, selectivity, old retention",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "tasks": ["joint24", "action bins"],
            "bound_units_per_task": BOUND,
            "old_steps": OLD_STEPS,
            "new_steps": NEW_STEPS,
            "gates": "old drop<=1e-9; new>=0.433; ablate drop>=0.20; old under ablation drop<=1e-9; modulation>=1.5",
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": (OLD_STEPS + NEW_STEPS) * len(SEEDS),
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "results": results, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "unit_differentiation_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
