"""Repeat the target-aware K diagnostic across real model seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_b3_k_target_aware import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    run_target_aware,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-target-aware-model-seeds-v1"
VERSION = 1
MODEL_SEEDS = (17, 23, 31)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_target_aware_model_seeds_20260910.json"
)
DEFAULT_CANDIDATE_ROOT = (
    PROJECT_ROOT / "output" / "taiji_m4v2_b3_k_target_aware_model_seeds_20260910"
)


def run_model_seed_stability(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_DIR.parent,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    model_seeds: tuple[int, ...] = MODEL_SEEDS,
) -> dict[str, Any]:
    model_reports = []
    for model_seed in model_seeds:
        artifact_dir = artifact_root / f"model_{model_seed}"
        model_report = run_target_aware(
            artifact_dir=artifact_dir,
            candidate_root=candidate_root / f"model_{model_seed}",
            model_seed=model_seed,
        )
        model_reports.append({"model_seed": model_seed, "report": model_report})

    reports = [item["report"] for item in model_reports]
    parent_by_model = {
        str(item["model_seed"]): item["report"].get("parent_checkpoint_digests", [])
        for item in model_reports
    }
    model_parent_values = [
        tuple(value) for value in parent_by_model.values()
    ]
    independent_model_parents = (
        len(model_parent_values) == len(set(model_parent_values))
        and all(len(value) == 1 for value in model_parent_values)
    )
    all_cells = [cell for report in reports for cell in report.get("cells", [])]
    all_deltas = [
        float(cell["report"]["holdout_structured_loss_delta"]["combined_mse"])
        for cell in all_cells
    ]
    model_technical = all(report.get("technical_gate_passed") for report in reports)
    model_performance = all(report.get("performance_gate_passed") for report in reports)
    model_candidate_distinct = all(
        report.get("candidate_updates_distinct") for report in reports
    )
    technical_gate_passed = (
        len(model_seeds) == 3
        and len(set(model_seeds)) == 3
        and independent_model_parents
        and model_technical
        and model_candidate_distinct
    )
    performance_gate_passed = bool(all_deltas) and all(delta < 0.0 for delta in all_deltas)
    report = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "status": "passed" if technical_gate_passed else "failed",
        "run_kind": "target-aware-model-seed-stability-diagnostic",
        "model_seeds": list(model_seeds),
        "model_count": len(model_seeds),
        "independent_model_parents": independent_model_parents,
        "parent_checkpoint_digests_by_model": parent_by_model,
        "model_reports": model_reports,
        "cell_count": len(all_cells),
        "all_combined_loss_deltas": all_deltas,
        "mean_combined_loss_delta": sum(all_deltas) / len(all_deltas)
        if all_deltas
        else None,
        "worst_combined_loss_delta": max(all_deltas) if all_deltas else None,
        "model_technical_gates_passed": model_technical,
        "model_performance_gates_passed": model_performance,
        "model_candidate_updates_distinct": model_candidate_distinct,
        "technical_gate_passed": technical_gate_passed,
        "performance_gate_passed": performance_gate_passed,
        "stability_gate_passed": technical_gate_passed and performance_gate_passed,
        "can_start_r6_formal": False,
        "can_promote": False,
        "candidate_promoted": False,
        "blocking_reason": (
            None
            if technical_gate_passed
            else "target-aware model-seed stability technical Gate failed"
        ),
        "promotion_blocking_reason": (
            None
            if technical_gate_passed and performance_gate_passed
            else "target-aware model-seed evidence is diagnostic only or has a non-negative cell"
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_DIR.parent)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    artifact_root = (
        args.artifact_root
        if args.artifact_root.is_absolute()
        else PROJECT_ROOT / args.artifact_root
    )
    candidate_root = (
        args.candidate_root
        if args.candidate_root.is_absolute()
        else PROJECT_ROOT / args.candidate_root
    )
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report = run_model_seed_stability(
        artifact_root=artifact_root,
        candidate_root=candidate_root,
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
