"""Multi-seed robustness check for the 22.6.3 recursive-correction feasibility.

The single-seed probe showed dependent > fixed only with 6 sparse stage-2 samples;
with 12 samples the effect vanished. This checks the 6-sample protocol across 5
seeds before any 22.6.3 card is pre-registered. Zero check; feasibility only.
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
from scripts.training.probe_s2_22_6_3_recursive_feasibility_20261120 import (  # noqa: E402
    BASE_STEPS,
    FIT_WORLDS,
    GROUP,
    REFINE,
    TRAIN_WORLDS,
    evaluate,
    labels_for,
    make_features,
    make_graph,
    rule_labels,
    train,
)

OUT = ROOT / "reports/taiji_u2_22_6_3_recursive_multiseed_20261120.json"
SEEDS = (0, 1, 2, 3, 4)
A_STEPS = 600
B_STEPS = 60
SPARSE_SAMPLES = 6


def run_seed(seed: int, data: tuple) -> dict[str, float]:
    train_features, train_colours, train_clean, fit_features, fit_colours, fit_clean = data
    rule1_train = rule_labels(train_colours, train_clean, amount=1)
    rule2_train = rule_labels(train_colours, train_clean, amount=2)
    rule2_fit = rule_labels(fit_colours, fit_clean, amount=2)
    group_mask = torch.tensor([colour in GROUP for colour in train_colours.tolist()])
    refine_mask = torch.tensor([colour in REFINE for colour in train_colours.tolist()])
    sparse_index = torch.nonzero(refine_mask).flatten()[:SPARSE_SAMPLES]
    fit_refine = torch.tensor([colour in REFINE for colour in fit_colours.tolist()])

    base = make_graph(seed)
    train(base, train_features, train_clean, BASE_STEPS, seed)
    base_refine = float(
        evaluate(base, fit_features[fit_refine], rule2_fit[fit_refine]).float().mean()
    )

    dependent = make_graph(seed)
    train(dependent, train_features, train_clean, BASE_STEPS, seed)
    train(dependent, train_features[group_mask], rule1_train[group_mask], A_STEPS, seed + 1)
    after_a = float(
        evaluate(dependent, fit_features[fit_refine], rule2_fit[fit_refine]).float().mean()
    )
    train(dependent, train_features[sparse_index], rule2_train[sparse_index], B_STEPS, seed + 2)
    dependent_refine = float(
        evaluate(dependent, fit_features[fit_refine], rule2_fit[fit_refine]).float().mean()
    )

    fixed_parent = make_graph(seed)
    train(fixed_parent, train_features, train_clean, BASE_STEPS, seed)
    train(fixed_parent, train_features[sparse_index], rule2_train[sparse_index], B_STEPS, seed + 2)
    fixed_refine = float(
        evaluate(fixed_parent, fit_features[fit_refine], rule2_fit[fit_refine]).float().mean()
    )
    return {
        "base_refine": base_refine,
        "after_a_refine": after_a,
        "dependent_refine": dependent_refine,
        "fixed_parent_refine": fixed_refine,
        "margin": dependent_refine - fixed_refine,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    train_examples = __import__(
        "scripts.training.first_a_support_22_5_v2", fromlist=["generate_examples"]
    ).generate_examples("train", TRAIN_WORLDS)
    fit_examples = __import__(
        "scripts.training.first_a_support_22_5_v2", fromlist=["generate_examples"]
    ).generate_examples("fit", FIT_WORLDS)
    train_colours, train_clean = labels_for(train_examples)
    fit_colours, fit_clean = labels_for(fit_examples)
    data = (
        make_features(train_examples),
        train_colours,
        train_clean,
        make_features(fit_examples),
        fit_colours,
        fit_clean,
    )
    results = []
    for seed in SEEDS:
        results.append({"seed": seed, **run_seed(seed, data)})
    margins = [entry["margin"] for entry in results]
    mean_margin = float(sum(margins) / len(margins))
    min_margin = min(margins)
    verdict = (
        "recursive_multiseed_ready"
        if min_margin >= 0.05 and mean_margin >= 0.10
        else "recursive_multiseed_absent"
    )
    payload = {
        "schema": "taiji-u2-22-6-3-recursive-multiseed-v1",
        "card": "22.6.3 two-dependent-corrections multi-seed feasibility",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "base_steps": BASE_STEPS,
            "correction_a_steps": A_STEPS,
            "correction_b_steps": B_STEPS,
            "correction_b_samples": SPARSE_SAMPLES,
            "train_worlds": TRAIN_WORLDS,
            "fit_worlds": FIT_WORLDS,
            "readiness": "min margin >= 0.05 and mean margin >= 0.10",
        },
        "results": results,
        "mean_margin": round(mean_margin, 4),
        "min_margin": round(min_margin, 4),
        "verdict": verdict,
        "discipline": {
            "optimizer_steps": (BASE_STEPS * 3 + A_STEPS + B_STEPS * 2) * len(SEEDS),
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "mean_margin": payload["mean_margin"],
                "min_margin": payload["min_margin"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "recursive_multiseed_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
