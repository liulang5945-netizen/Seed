"""Five-task pathway interference matrix + reversible disable probe (zero check).

Loads each seed's final five-pathway capsule, then for every task i: records all
five accuracies, disables task i's pathway (zeroes its head/query/encoders),
measures the whole matrix, reloads the untouched capsule (re-enable) and checks all
accuracies return exactly.  Expected: only the disabled task collapses (to the
constant-class baseline produced by zero logits); every other task is bit-identical
before/after.  3 seeds; cap 120 CPU.
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
from scripts.training.probe_s2_multihead8_20261120 import (  # noqa: E402
    accuracy,
    build_graph,
    data_bundle,
)
from taiji.u2_neural_units import apply_capsule, load_capsule  # noqa: E402

OUT = ROOT / "reports/taiji_u2_multihead9_interference_20261120.json"
CAPSULES = ROOT / "output/s2_multihead8"
CAP_CPU = 120.0
SEEDS = (0, 1, 2)
TASKS = (
    ("joint24", 0, 24),
    ("action_bins", 1, 3),
    ("direction", 2, 2),
    ("position", 3, 4),
    ("color", 4, 6),
    ("quadrant", 5, 4),
    ("cross", 6, 2),
    ("colormod", 7, 3),
    ("parity", 8, 2),
)
RETENTION_MAX = 1e-9
DISABLED_BASELINE_TOL = 0.05
DISABLED_DROP_MIN = 0.25


def task_accuracy(graph, data, index, classes):
    key = (
        "old_fit",
        "new_fit",
        "direction_fit",
        "position_fit",
        "color_fit",
        "quadrant_fit",
        "cross_fit",
        "colormod_fit",
        "parity_fit",
    )[index]
    return accuracy(graph, *data[key], head=index, classes=classes)


def disable(graph, index: int) -> None:
    with torch.no_grad():
        for parameter in graph.heads[index].parameters():
            parameter.zero_()
        if index > 0:
            graph.task_queries[index - 1].zero_()
            for parameter in graph.task_encoders[f"h{index}"].parameters():
                parameter.zero_()


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
        apply_capsule(graph, load_capsule(CAPSULES / f"seed{seed}_session9.pt"))
        baseline = [task_accuracy(graph, data, index, classes) for _, index, classes in TASKS]
        baseline_zero = []
        for _, index, _ in TASKS:
            key = (
                "old_fit",
                "new_fit",
                "direction_fit",
                "position_fit",
                "color_fit",
                "quadrant_fit",
                "cross_fit",
                "colormod_fit",
                "parity_fit",
            )[index]
            labels = data[key][1]
            baseline_zero.append(float((labels == 0).float().mean()))
        interference = []
        for _disabled, (name, index, _classes) in enumerate(TASKS):
            disable(graph, index)
            matrix = [task_accuracy(graph, data, other, cls) for _, other, cls in TASKS]
            interference.append(
                {
                    "disabled": name,
                    "matrix": [round(value, 6) for value in matrix],
                    "others_max_delta": round(
                        max(
                            abs(matrix[other] - baseline[other])
                            for other in range(len(TASKS))
                            if other != index
                        ),
                        9,
                    ),
                    "disabled_delta": round(matrix[index] - baseline[index], 6),
                }
            )
            apply_capsule(graph, load_capsule(CAPSULES / f"seed{seed}_session9.pt"))
        reenabled = [task_accuracy(graph, data, index, classes) for _, index, classes in TASKS]
        results.append(
            {
                "seed": seed,
                "baseline": [round(value, 6) for value in baseline],
                "zero_constant_baseline": [round(value, 6) for value in baseline_zero],
                "interference": interference,
                "reenabled": [round(value, 6) for value in reenabled],
            }
        )
    others_ok = all(
        entry["others_max_delta"] <= RETENTION_MAX
        for result in results
        for entry in result["interference"]
    )
    collapse_ok = True
    for result in results:
        for entry in result["interference"]:
            index = [task[1] for task in TASKS].index(
                {name: index for name, index, _ in TASKS}[entry["disabled"]]
            )
            drop = result["baseline"][index] - entry["matrix"][index]
            expected = result["zero_constant_baseline"][index]
            if not (
                entry["matrix"][index] <= expected + DISABLED_BASELINE_TOL
                and drop >= DISABLED_DROP_MIN
            ):
                collapse_ok = False
    reenabled_ok = all(
        max(
            abs(result["reenabled"][index] - result["baseline"][index])
            for index in range(len(TASKS))
        )
        <= RETENTION_MAX
        for result in results
    )
    gates = {
        "others_unchanged": others_ok,
        "disabled_collapses": collapse_ok,
        "reenabled_exact": reenabled_ok,
    }
    verdict = "pathway_interference_ready" if all(gates.values()) else "pathway_interference_absent"
    payload = {
        "schema": "taiji-u2-pathway-interference-v1",
        "card": "nine-task pathway interference matrix and reversible disable",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "tasks": [name for name, _, _ in TASKS],
            "capsules": "output/s2_multihead8/seed*_session5.pt",
            "retention_max": RETENTION_MAX,
            "disabled_baseline_tol": DISABLED_BASELINE_TOL,
            "disabled_drop_min": DISABLED_DROP_MIN,
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
    return 0 if verdict == "pathway_interference_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
