"""Repeat the K structured-loss diagnostic across three course seeds."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_b3_k_loss_diagnostic import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CANDIDATE_DIR,
    run_diagnostic,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-loss-stability-v1"
VERSION = 1
COURSE_SEEDS = (0, 1, 2)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_loss_stability_20260910.json"


def run_stability(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    candidate_root: Path = DEFAULT_CANDIDATE_DIR.parent / "stability",
    model_seed: int = 17,
    course_seeds: tuple[int, ...] = COURSE_SEEDS,
) -> dict[str, Any]:
    cells = []
    for course_seed in course_seeds:
        cell = run_diagnostic(
            artifact_dir=artifact_dir,
            candidate_dir=candidate_root / f"course_seed_{course_seed}" / f"model_{model_seed}",
            model_seed=model_seed,
            course_seed=course_seed,
        )
        cells.append({"course_seed": course_seed, "report": cell})

    reports = [item["report"] for item in cells]
    parent_digests = {report.get("parent_checkpoint_digest") for report in reports}
    delta_values = [
        float(report.get("holdout_structured_loss_delta", {}).get("combined_mse", 0.0))
        for report in reports
    ]
    technical_pass = all(report.get("status") == "passed" for report in reports)
    same_parent = len(parent_digests) == 1 and None not in parent_digests
    train_digests = [report.get("train_experience_digest") for report in reports]
    train_variant_indexes = [report.get("train_episode_index") for report in reports]
    distinct_course_variants = (
        len(train_digests) == len(set(train_digests))
        and None not in train_digests
        and len(train_variant_indexes) == len(set(train_variant_indexes))
        and None not in train_variant_indexes
    )
    all_improved = bool(delta_values) and all(value < 0.0 for value in delta_values)
    technical_gate_passed = technical_pass and same_parent and distinct_course_variants
    report = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "status": "passed" if technical_pass and same_parent else "failed",
        "run_kind": "learning-stability-diagnostic",
        "model_seed": model_seed,
        "course_seeds": list(course_seeds),
        "cell_count": len(cells),
        "same_parent": same_parent,
        "parent_checkpoint_digests": sorted(parent_digests),
        "train_experience_digests": train_digests,
        "train_episode_indexes": train_variant_indexes,
        "distinct_course_variants": distinct_course_variants,
        "combined_loss_delta_by_course_seed": {
            str(item["course_seed"]): float(
                item["report"].get("holdout_structured_loss_delta", {}).get(
                    "combined_mse", 0.0
                )
            )
            for item in cells
        },
        "mean_combined_loss_delta": statistics.fmean(delta_values) if delta_values else None,
        "worst_combined_loss_delta": max(delta_values) if delta_values else None,
        "all_course_seeds_improved": all_improved,
        "performance_gate_passed": all_improved,
        "stability_gate_passed": technical_gate_passed and all_improved,
        "technical_gate_passed": technical_gate_passed,
        "can_start_r6_formal": False,
        "can_promote": False,
        "candidate_promoted": False,
        "cells": cells,
        "blocking_reason": (
            None
            if technical_gate_passed
            else "K loss stability cells failed or did not consume distinct train variants"
        ),
        "promotion_blocking_reason": (
            None
            if all_improved
            else "course variant updates are not uniformly non-degrading on the fixed holdout"
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_DIR.parent / "stability")
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else PROJECT_ROOT / args.artifact_dir
    candidate_root = args.candidate_root if args.candidate_root.is_absolute() else PROJECT_ROOT / args.candidate_root
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report = run_stability(
        artifact_dir=artifact_dir,
        candidate_root=candidate_root,
        model_seed=args.model_seed,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
