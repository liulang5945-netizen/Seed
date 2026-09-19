"""Aggregate calibration runs and apply the frozen selection rule (§4).

Contract: plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_FROZEN_20260919.md §A

Reads the four calibration run reports (arms A/B x cal_lr1/cal_lr3, seed
20260920), applies the per-arm selection rule across the two candidate
recipes -- among checkpoint evaluations meeting copy >= 0.90, unknown >= 0.90
and finite parameters, the greatest three-flip-class pairwise macro wins;
ties prefer the lower learning rate, then the earlier update -- and records
the selected recipe per arm (or the arm's stop when nothing qualifies).

Usage: python scripts/training/aggregate_taiji_r2_content_binding_calibration.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_taiji_r2_content_binding import (  # noqa: E402
    FLIP_GATE_CLASSES,
    RECIPES,
    SELECTION_COPY_MIN,
    SELECTION_UNKNOWN_MIN,
    select_recipe_candidate,
)

OUTPUT_ROOT = Path("reports/r2_content_binding_v1")
CALIBRATION_SEED = 20260920
ARMS = ("A", "B")
OUT_REPORT = Path("reports/r2_content_binding_v1/calibration_selection_20260920.json")
REPORT_FORMAT = "r2-content-binding-calibration-selection-v1"


def _load_run(arm: str, config: str) -> dict[str, Any] | None:
    path = PROJECT_ROOT / OUTPUT_ROOT / arm / str(CALIBRATION_SEED) / config / "run_report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=OUT_REPORT)
    args = parser.parse_args()

    per_arm: OrderedDict[str, Any] = OrderedDict()
    all_runs_present = True
    for arm in ARMS:
        runs: OrderedDict[str, Any] = OrderedDict()
        for config in sorted(RECIPES):
            report = _load_run(arm, config)
            runs[config] = report
            if report is None or report.get("status") != "completed":
                all_runs_present = False
        # candidate evaluations across BOTH configs of this arm
        candidates: list[dict[str, Any]] = []
        for config, report in runs.items():
            if report is None:
                continue
            for evaluation in report.get("calibration_evaluations", []):
                candidate = dict(evaluation)
                candidate["config"] = config
                candidate["arm"] = arm
                candidates.append(candidate)
        selected = select_recipe_candidate(candidates)
        per_arm[arm] = {
            "runs": {
                config: None
                if report is None
                else {
                    "status": report["status"],
                    "updates_done": report["updates_done"],
                    "elapsed_seconds": report["elapsed_seconds"],
                    "final_health": report["health"][-1] if report["health"] else None,
                    "calibration_evaluations": report.get("calibration_evaluations", []),
                    "group_exposure_per_class": report.get("group_exposure_per_class"),
                }
                for config, report in runs.items()
            },
            "eligible_candidates": [
                {
                    "config": c["config"],
                    "update": c["update"],
                    "flip_pairwise_macro": c["flip_pairwise_macro"],
                    "copy_rate": c["copy_rate"],
                    "unknown_rate": c["unknown_rate"],
                }
                for c in candidates
                if c["copy_rate"] is not None
                and c["copy_rate"] >= SELECTION_COPY_MIN
                and c["unknown_rate"] is not None
                and c["unknown_rate"] >= SELECTION_UNKNOWN_MIN
                and c["finite_parameters"]
            ],
            "selected": None
            if selected is None
            else {
                "config": selected["config"],
                "update": selected["update"],
                "learning_rate": selected["learning_rate"],
                "flip_pairwise_macro": selected["flip_pairwise_macro"],
                "copy_rate": selected["copy_rate"],
                "unknown_rate": selected["unknown_rate"],
                "checkpoint": selected["checkpoint"],
            },
            "arm_status": "selected" if selected is not None else "stopped_no_qualified_candidate",
        }

    gates = {
        "all_calibration_runs_completed": all_runs_present,
        "any_arm_selected": any(per_arm[arm]["selected"] is not None for arm in ARMS),
    }
    report = {
        "format": REPORT_FORMAT,
        "version": 1,
        "contract": "plans/reference/M5_R2_CONTENT_BINDING_CONTRACT_FROZEN_20260919.md",
        "calibration_seed": CALIBRATION_SEED,
        "selection_rule": "content-binding-selection-v1",
        "flip_gate_classes": list(FLIP_GATE_CLASSES),
        "per_arm": per_arm,
        "gates": gates,
        "next_step": (
            "formal runs (seeds 20260921/20260922/20260923) with each arm's selected recipe"
            if gates["any_arm_selected"]
            else "package stops: no arm produced a qualified candidate (contract §4)"
        ),
        "outcome": "passed" if all(gates.values()) else "failed",
    }
    out_path = PROJECT_ROOT / args.report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outcome": report["outcome"],
                "A": per_arm["A"]["selected"],
                "B": per_arm["B"]["selected"],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0 if report["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
