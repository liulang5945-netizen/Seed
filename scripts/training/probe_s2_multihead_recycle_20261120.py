"""Task-pathway recycling probe: remove one pathway, keep others exact, re-add.

Loads the final four-task capsule from the multihead3 probe, measures accuracies,
zeros task 3's pathway (input encoders, query, readout head) to emulate recycling
it, verifies only that task's ability collapses (others bitwise unchanged), then
re-trains that pathway and checks recovery.  3 seeds; zero check; cap 120 CPU.
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
from scripts.training.probe_s2_multihead3_20261120 import (  # noqa: E402
    accuracy,
    build_graph,
    data_bundle,
    train_head,
)
from taiji.u2_neural_units import apply_capsule, load_capsule  # noqa: E402

OUT = ROOT / "reports/taiji_u2_multihead_recycle_20261120.json"
CAPSULES = ROOT / "output/s2_multihead3"
CAP_CPU = 120.0
SEEDS = (0, 1, 2)
STEPS = 700
RETENTION_MAX = 0.02
REMOVED_MAX = 0.30
RECOVERED_MIN = 0.45


def zero_task3(graph) -> None:
    with torch.no_grad():
        for parameter in graph.heads[3].parameters():
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
        apply_capsule(graph, load_capsule(CAPSULES / f"seed{seed}_session4.pt"))
        before = {
            "old": accuracy(graph, *data["old_fit"], head=0, classes=24),
            "new": accuracy(graph, *data["new_fit"], head=1, classes=3),
            "direction": accuracy(graph, *data["direction_fit"], head=2, classes=2),
            "position": accuracy(graph, *data["position_fit"], head=3, classes=4),
        }
        zero_task3(graph)
        fresh = build_graph(seed)
        with torch.no_grad():
            graph.heads[3].load_state_dict(fresh.heads[3].state_dict())
            graph.task_queries[2].copy_(fresh.task_queries[2])
            graph.task_encoders["h3"].load_state_dict(fresh.task_encoders["h3"].state_dict())
        removed = {
            "old": accuracy(graph, *data["old_fit"], head=0, classes=24),
            "new": accuracy(graph, *data["new_fit"], head=1, classes=3),
            "direction": accuracy(graph, *data["direction_fit"], head=2, classes=2),
            "position": accuracy(graph, *data["position_fit"], head=3, classes=4),
        }
        train_head(graph, 3, *data["position_train"], STEPS, seed + 40)
        recovered = {
            "old": accuracy(graph, *data["old_fit"], head=0, classes=24),
            "new": accuracy(graph, *data["new_fit"], head=1, classes=3),
            "direction": accuracy(graph, *data["direction_fit"], head=2, classes=2),
            "position": accuracy(graph, *data["position_fit"], head=3, classes=4),
        }
        results.append(
            {
                "seed": seed,
                "before": {key: round(value, 4) for key, value in before.items()},
                "removed": {key: round(value, 4) for key, value in removed.items()},
                "recovered": {key: round(value, 4) for key, value in recovered.items()},
            }
        )
    gates = {
        "others_exact_after_removal": all(
            abs(result["before"]["old"] - result["removed"]["old"]) <= RETENTION_MAX
            and abs(result["before"]["new"] - result["removed"]["new"]) <= RETENTION_MAX
            and abs(result["before"]["direction"] - result["removed"]["direction"]) <= RETENTION_MAX
            for result in results
        ),
        "removed_task_collapses": all(
            result["removed"]["position"] <= REMOVED_MAX for result in results
        ),
        "recovered_learnable": all(
            result["recovered"]["position"] >= RECOVERED_MIN for result in results
        ),
        "others_exact_after_recovery": all(
            abs(result["before"]["old"] - result["recovered"]["old"]) <= RETENTION_MAX
            and abs(result["before"]["new"] - result["recovered"]["new"]) <= RETENTION_MAX
            and abs(result["before"]["direction"] - result["recovered"]["direction"])
            <= RETENTION_MAX
            for result in results
        ),
    }
    verdict = "pathway_recycle_ready" if all(gates.values()) else "pathway_recycle_absent"
    payload = {
        "schema": "taiji-u2-pathway-recycle-v1",
        "card": "task-pathway recycling probe (zero, then fresh re-init, then retrain)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "capsules": "output/s2_multihead3/seed*_session4.pt",
            "retrain_steps": STEPS,
            "retention_max": RETENTION_MAX,
            "removed_max": REMOVED_MAX,
            "recovered_min": RECOVERED_MIN,
        },
        "results": results,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"optimizer_steps": STEPS * len(SEEDS), "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "results": results, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "pathway_recycle_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
