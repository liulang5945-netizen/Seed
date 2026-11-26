"""Shared-unit split/merge probe: manufacture and undo unit-level division of labour (0 check).

Trains 30-unit joint24, splits one unit into two (inherit + p-halving), fine-tunes only
the pair, checks old-task retention / differentiation / irreplaceability, then merges
the pair back (no retraining) and checks functional retention.  Five gates per
D-20261119-117.  3 seeds; cap 250 CPU.
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
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec  # noqa: E402
from taiji.u2_split_merge import merge_units, split_unit, train_pair  # noqa: E402

OUT = ROOT / "reports/taiji_u2_split_merge_20261120.json"
CAP_CPU = 250.0
TRAIN_WORLDS = 256
FIT_WORLDS = 32
BASE_STEPS = 1200
PAIR_STEPS = 200
BATCH = 64
LR = 3e-3
WARMUP = 100
SEEDS = (0, 1, 2)
RETENTION_MAX = 0.02
MODULATION_MIN = 1.5
CORRELATION_MAX = 0.5
ABLATE_DROP_MIN = 0.10
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")


def learning_rate(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def build_graph(seed: int) -> RouteBGraph:
    torch.manual_seed(seed)
    return RouteBGraph(
        UnitGraphSpec(
            groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=24, seed=seed
        )
    )


def accuracy(graph, features, labels) -> float:
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        logits = graph(channels(features), gates, graph.initial_state())["readout"]
    return float((logits.argmax(-1) == labels).float().mean())


def train_base(graph, features, labels, seed: int) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(BASE_STEPS):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, BASE_STEPS)
        index = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        gates = torch.ones(BATCH, graph.units.n_units)
        optimizer.zero_grad(set_to_none=True)
        logits = graph(channels(features[index]), gates, graph.initial_state())["readout"]
        loss = loss_fn(logits, labels[index])
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def activity_matrix(graph, features) -> torch.Tensor:
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        inputs = graph.encode_legal_inputs(channels(features))
        active = graph.units.active_gate(gates)
        result = graph.units(inputs, active, graph.initial_state())
    return result["activity"]


def class_modulation(activity: torch.Tensor, labels: torch.Tensor, unit: int) -> float:
    means = []
    for class_index in range(24):
        mask = labels == class_index
        if mask.any():
            means.append(float(activity[:, unit][mask].mean()))
    if len(means) < 2:
        return 0.0
    top = max(means)
    others = (sum(means) - top) / max(1, len(means) - 1)
    return (top - others) / (top + others + 1e-9)


def ablate_accuracy(graph, features, labels, unit_id: str) -> float:
    graph.units.set_lifecycle(unit_id, "dormant")
    value = accuracy(graph, features, labels)
    graph.units.set_lifecycle(unit_id, "mature")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    train_examples = generate_examples("train", TRAIN_WORLDS)
    fit_examples = generate_examples("fit", FIT_WORLDS)
    train_features = torch.stack([adapter_v3_features(ex) for ex in train_examples]).float()
    train_labels = torch.tensor([joint_label(ex) for ex in train_examples])
    fit_features = torch.stack([adapter_v3_features(ex) for ex in fit_examples]).float()
    fit_labels = torch.tensor([joint_label(ex) for ex in fit_examples])
    results = []
    for seed in SEEDS:
        graph = build_graph(20261120 + seed)
        train_base(graph, train_features, train_labels, seed)
        base_accuracy = accuracy(graph, fit_features, fit_labels)
        grown, split_info = split_unit(graph.units, 0)
        graph.units = grown
        pair = [split_info["parent"], split_info["child"]]
        generator = torch.Generator().manual_seed(20261120 + seed + 3)
        train_pair(graph, train_features, train_labels, pair, PAIR_STEPS, 3e-3, generator)
        split_accuracy = accuracy(graph, fit_features, fit_labels)
        activity = activity_matrix(graph, fit_features)
        correlation = float(
            torch.corrcoef(torch.stack([activity[:, pair[0]], activity[:, pair[1]]]))[0, 1]
        )
        modulation_parent = class_modulation(activity, fit_labels, pair[0])
        modulation_child = class_modulation(activity, fit_labels, pair[1])
        parent_id = graph.units.identities["units"][pair[0]]["unit_id"]
        child_id = graph.units.identities["units"][pair[1]]["unit_id"]
        parent_ablation = accuracy(graph, fit_features, fit_labels) - ablate_accuracy(
            graph, fit_features, fit_labels, parent_id
        )
        child_ablation = accuracy(graph, fit_features, fit_labels) - ablate_accuracy(
            graph, fit_features, fit_labels, child_id
        )
        merged, merge_info = merge_units(graph.units, pair[0], pair[1])
        graph.units = merged
        merged_accuracy = accuracy(graph, fit_features, fit_labels)
        results.append(
            {
                "seed": seed,
                "base_accuracy": round(base_accuracy, 4),
                "split_accuracy": round(split_accuracy, 4),
                "split_retention_drop": round(base_accuracy - split_accuracy, 6),
                "pair_correlation": round(correlation, 4),
                "modulation_parent": round(modulation_parent, 4),
                "modulation_child": round(modulation_child, 4),
                "parent_ablation_drop": round(parent_ablation, 4),
                "child_ablation_drop": round(child_ablation, 4),
                "merged_accuracy": round(merged_accuracy, 4),
                "merge_delta": round(split_accuracy - merged_accuracy, 6),
                "units": [30, graph.units.n_units + 1, graph.units.n_units],
                "ids": [parent_id, child_id, merge_info["kept_id"]],
            }
        )
    gates = {
        "old_task_retention": all(
            result["split_retention_drop"] <= RETENTION_MAX for result in results
        ),
        "differentiation": all(
            max(result["modulation_parent"], result["modulation_child"]) >= MODULATION_MIN
            or abs(result["pair_correlation"]) <= CORRELATION_MAX
            for result in results
        ),
        "irreplaceable": all(
            result["parent_ablation_drop"] >= ABLATE_DROP_MIN
            and result["child_ablation_drop"] >= ABLATE_DROP_MIN
            for result in results
        ),
        "merge_lossless": all(result["merge_delta"] <= RETENTION_MAX for result in results),
        "unit_count": all(
            result["units"][:2] == [30, 31] and result["units"][2] == 30 for result in results
        ),
    }
    if all(gates.values()):
        verdict = "unit_split_merge_ready"
    elif sum(1 for value in gates.values() if value) >= 3:
        verdict = "unit_split_merge_partial"
    else:
        verdict = "unit_split_merge_absent"
    payload = {
        "schema": "taiji-u2-split-merge-v1",
        "card": "shared-unit split into two (fine-tuned) then merged back without retraining",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "base_steps": BASE_STEPS,
            "pair_steps": PAIR_STEPS,
            "gates": "retention<=0.02; modulation>=1.5 or |corr|<=0.5; both ablations>=0.10; merge<=0.02; units 30->31->30",
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": (BASE_STEPS + PAIR_STEPS) * len(SEEDS),
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
    return 0 if verdict == "unit_split_merge_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
