"""Shared-unit task modulation probe on the five-task subject (zero training, 0 check).

Loads the five-pathway capsules, collects per-task shared-unit activity, and tests
whether units respond selectively across tasks (differentiation evidence at the
unit level):

* selectivity: for each unit, S = (max_task_mean - mean_of_other_tasks) /
  (max_task_mean + mean_of_other_tasks); count units with S >= 0.30;
* cross-task modulation: per unit, variance of task means / mean within-task
  variance (R); the mean R must be >= 1.5 for the population.

Gates: selective_units >= 5 (of 30) and mean_R >= 1.5; else record absent with the
measured numbers.  3 seeds; cap 60 CPU.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.probe_s2_multihead4_20261120 import (  # noqa: E402
    build_graph,
    channels,
    data_bundle,
)
from taiji.u2_neural_units import apply_capsule, load_capsule  # noqa: E402

OUT = ROOT / "reports/taiji_u2_unit_modulation_20261120.json"
CAPSULES = ROOT / "output/s2_multihead4"
CAP_CPU = 60.0
SEEDS = (0, 1, 2)
TASKS = (
    ("joint24", "old_fit", 0),
    ("action_bins", "new_fit", 1),
    ("direction", "direction_fit", 2),
    ("position", "position_fit", 3),
    ("color", "color_fit", 4),
)
SELECTIVITY_MIN = 0.30
SELECTIVE_UNITS_MIN = 5
MODULATION_MIN = 1.5


def task_activity(graph, features, head: int) -> torch.Tensor:
    with torch.no_grad():
        gates = torch.ones(features.shape[0], graph.units.n_units)
        inputs = graph._encode(channels(features), head)
        active = graph.units.active_gate(gates)
        result = graph.units(inputs, active, graph.initial_state())
    return result["activity"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    data = data_bundle()
    results = []
    for seed in SEEDS:
        graph = build_graph(seed)
        apply_capsule(graph, load_capsule(CAPSULES / f"seed{seed}_session5.pt"))
        activities = []
        for _, key, head in TASKS:
            activities.append(task_activity(graph, data[key][0], head))
        means = torch.stack([activity.mean(dim=0) for activity in activities])
        within = torch.stack([activity.var(dim=0, unbiased=False) for activity in activities])
        selectivity = []
        for unit in range(graph.units.n_units):
            column = means[:, unit]
            top = float(column.max())
            others = float((column.sum() - column.max()) / max(1, column.numel() - 1))
            selectivity.append((top - others) / (top + others + 1e-9))
        selectivity_tensor = torch.tensor(selectivity)
        between_variance = means.var(dim=0, unbiased=False)
        within_mean = within.mean(dim=0)
        modulation = between_variance / (within_mean + 1e-9)
        results.append(
            {
                "seed": seed,
                "selective_units": int((selectivity_tensor >= SELECTIVITY_MIN).sum()),
                "mean_selectivity": round(float(selectivity_tensor.mean()), 4),
                "max_selectivity": round(float(selectivity_tensor.max()), 4),
                "mean_modulation": round(float(modulation.mean()), 4),
                "task_means": [[round(float(value), 4) for value in row] for row in means.tolist()],
            }
        )
    gates = {
        "selective_units": all(
            result["selective_units"] >= SELECTIVE_UNITS_MIN for result in results
        ),
        "modulation": all(result["mean_modulation"] >= MODULATION_MIN for result in results),
    }
    verdict = "unit_modulation_present" if all(gates.values()) else "unit_modulation_absent"
    payload = {
        "schema": "taiji-u2-unit-modulation-v1",
        "card": "shared-unit task modulation on the five-task subject",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "selectivity_min": SELECTIVITY_MIN,
            "selective_units_min": SELECTIVE_UNITS_MIN,
            "modulation_min": MODULATION_MIN,
            "capsules": "output/s2_multihead4/seed*_session5.pt",
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "results": results, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "unit_modulation_present" else 1


if __name__ == "__main__":
    raise SystemExit(main())
