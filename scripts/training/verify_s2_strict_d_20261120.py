"""Zero-training strict-D sensitivity analysis for the 22.5.1 joint gate (fit face).

The frozen joint-gate contract adjudicates D against the parent's same-condition
behaviour.  The result card disclosed that a stricter reference (the parent's
A-ground-truth accuracy on the same items) can flip the verdict.  This script
re-decodes the saved child artifacts and the accepted parent on the fit face and
reports BOTH references, with paired one-sided bootstrap bounds, as decision
support only: zero optimizer steps, zero check/final reads, no signing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples as generate_v2  # noqa: E402
from scripts.training.run_s2_joint_gate_20261120 import _records  # noqa: E402
from scripts.training.s0_parent_loader_20261119 import (  # noqa: E402
    CHECKPOINTS,
    load_historical,
)
from taiji.u2_22_5_pipeline import (  # noqa: E402
    DEFAULT_TEMPLATES,
    build_b_items,
    decode_episode,
)
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_5_1_strict_d_20261120.json"
ARTIFACTS = ROOT / "output/s2_joint_gate_20261120"
PARENT_SEED = 20261112
SEEDS = (20261110, 20261111, 20261112)


def upper_bound(values: list[float], resamples: int = 10000, seed: int = 20261120) -> float:
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    draws = rng.choice(data, size=(resamples, data.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.95))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    fit = generate_v2("fit", 32)
    items = build_b_items(fit, templates=DEFAULT_TEMPLATES[:2])
    parent, _payload, _audit = load_historical(CHECKPOINTS[PARENT_SEED])
    # Parent reference decodes (shared across child seeds).
    parent_rows = []
    with torch.no_grad():
        for index, _item in enumerate(items):
            item, rec_b, rec_d, _d2, _old, _new = _records(items, index, DEFAULT_TEMPLATES[:2])
            ex = item["example"]
            a_out, _ = decode_episode(parent, ex["public"], ex["id"], records=None)
            d_out, _ = decode_episode(parent, ex["public"], ex["id"], records=[rec_d])
            parent_rows.append(
                {
                    "id": ex["id"],
                    "a_truth": a_out == ex["target"],
                    "a_output": a_out.hex(),
                    "d_same_condition": d_out == a_out,
                    "d_truth": d_out == ex["target"],
                }
            )
    rows = []
    for seed in SEEDS:
        payload = torch.load(ARTIFACTS / f"child-{seed}.pt", map_location="cpu", weights_only=False)
        child = U2Subject(U2Config(**payload["config"]))
        child.load_state_dict(payload["state_dict"])
        child_rows = []
        with torch.no_grad():
            for index, _item in enumerate(items):
                item, _b, rec_d, _d2, _old, _new = _records(items, index, DEFAULT_TEMPLATES[:2])
                ex = item["example"]
                a_out, _ = decode_episode(child, ex["public"], ex["id"], records=None)
                d_out, _ = decode_episode(child, ex["public"], ex["id"], records=[rec_d])
                child_rows.append(
                    {
                        "id": ex["id"],
                        "a_truth": a_out == ex["target"],
                        "d_same_condition": d_out == bytes.fromhex(parent_rows[index]["a_output"]),
                        "d_truth": d_out == ex["target"],
                    }
                )
        same_condition_drop = [
            float(parent_rows[index]["d_same_condition"])
            - float(child_rows[index]["d_same_condition"])
            for index in range(len(items))
        ]
        truth_drop = [
            float(parent_rows[index]["a_truth"]) - float(child_rows[index]["d_truth"])
            for index in range(len(items))
        ]
        rows.append(
            {
                "seed": seed,
                "a_child_truth": sum(1 for r in child_rows if r["a_truth"]),
                "a_parent_truth": sum(1 for r in parent_rows if r["a_truth"]),
                "d_child_same_condition": sum(1 for r in child_rows if r["d_same_condition"]),
                "d_parent_same_condition": sum(1 for r in parent_rows if r["d_same_condition"]),
                "d_child_truth": sum(1 for r in child_rows if r["d_truth"]),
                "d_parent_truth": sum(1 for r in parent_rows if r["d_truth"]),
                "same_condition_drop_upper_bound": round(upper_bound(same_condition_drop), 4),
                "truth_reference_drop_upper_bound": round(upper_bound(truth_drop), 4),
                "n": len(items),
            }
        )
    thresholds = {"d_reference_upper_bound_max": 0.02}
    same_ok = all(
        row["same_condition_drop_upper_bound"] <= thresholds["d_reference_upper_bound_max"]
        for row in rows
    )
    truth_ok = all(
        row["truth_reference_drop_upper_bound"] <= thresholds["d_reference_upper_bound_max"]
        for row in rows
    )
    payload = {
        "schema": "taiji-u2-22-5-1-strict-d-v1",
        "card": "22.5.1 joint gate fit-face D-reference sensitivity",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "scope": (
            "re-decode saved children + accepted parent on fit 24 B items; "
            "zero optimizer steps, zero check/final reads, no signing"
        ),
        "parent": {
            "seed": PARENT_SEED,
            "checkpoint": str(CHECKPOINTS[PARENT_SEED].relative_to(ROOT)),
        },
        "rows": rows,
        "thresholds": thresholds,
        "verdicts": {
            "frozen_same_condition_reference": (
                "d_reference_pass" if same_ok else "d_reference_breach"
            ),
            "strict_parent_a_truth_reference": (
                "d_reference_pass" if truth_ok else "d_reference_breach"
            ),
        },
        "recommendation": (
            "both references pass at the 24-item fit denominator only one way: "
            "same-condition passes; the strict parent-A-truth reference breaches. "
            "Owner should pick the D reference before any formal check exposure."
        ),
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"rows": rows, "verdicts": payload["verdicts"], "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
