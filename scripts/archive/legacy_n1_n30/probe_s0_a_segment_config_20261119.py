"""S0 A-segment config recovery probe (2026-11-19).

The S0 work table records ``600 updates, batch 4, 147.17 CPU`` and seed
20261111 ``fit 30/32`` but does not record the train-set size.  A re-run with
``train_count=64`` reached only 25/32 at the same measured CPU, so this probe
varies one recorded knob at a time (train-set size; then lr/warmup) on that
seed and reports the fit reading and the measured CPU for each variant.

It is diagnostic instrumentation, not a gate: no criterion is changed here.
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
from scripts.training.first_a_support_22_5 import generate_examples  # noqa: E402
from taiji.u2_22_5_pipeline import ATrainConfig, train_task_a  # noqa: E402

REPORT = ROOT / "reports/taiji_u2_s0_a_segment_config_probe_20261119.json"


def run(seed: int, train_count: int, lr: float, warmup: int, batch: int) -> dict:
    started = time.process_time()
    torch.set_num_threads(1)
    train_examples = generate_examples("train", train_count)
    fit_examples = generate_examples("fit", 32)
    _, history = train_task_a(
        train_examples,
        ATrainConfig(
            seed=seed,
            d=32,
            updates=600,
            batch_size=batch,
            lr=lr,
            warmup_steps=warmup,
        ),
        fit_examples=fit_examples,
        eval_every=150,
    )
    return {
        "seed": seed,
        "train_count": train_count,
        "lr": lr,
        "warmup": warmup,
        "batch": batch,
        "history": history,
        "fit": history[-1]["fit_accuracy"] if history else None,
        "cpu": round(time.process_time() - started, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args()
    variants = [
        {"train_count": 256, "lr": 4e-3, "warmup": 60},
        {"train_count": 64, "lr": 5e-3, "warmup": 200},
    ]
    rows = [run(args.seed, batch=4, **variant) for variant in variants]
    payload = {
        "schema": "taiji-u2-s0-a-segment-config-probe-v1",
        "card": "22.5.1/S0",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "purpose": (
            "recover the unrecorded train-set size of the S0 A segment from its "
            "published seed-20261111 fit reading (30/32)"
        ),
        "frozen_common": {"d": 32, "updates": 600, "batch": 4, "fit_examples": 32},
        "requested_claim": {"seed": 20261111, "fit": 0.9375},
        "rows": rows,
        "config_recovered": rows[0]["fit"] >= 0.9375,
        "verdict": (
            "train_count_256_reproduces_claim" if rows[0]["fit"] >= 0.9375 else "config_unresolved"
        ),
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "rows": [
                    {k: row[k] for k in ("train_count", "lr", "warmup", "fit", "cpu")}
                    for row in rows
                ],
                "verdict": payload["verdict"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
