"""Run the bounded multi-seed/order R4 shadow-growth matrix.

The matrix reuses the exact five-arm canary and its corrected parent/owner
boundary.  It changes only the model seed and pre-registered course order so
the result can distinguish a noisy S/G cell from a systematic candidate
selection problem.  This report is evidence, not an R5 promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_r4_shadow import (  # noqa: E402
    R4Course,
    course_variant,
    run_canary,
)
from taiji.internalization import content_digest  # noqa: E402

FORMAL_FORMAT = "taiji-m4v2-r4-shadow-formal-v1"
FORMAL_VERSION = 1
DEFAULT_MODEL_SEEDS = (71, 83, 97)
DEFAULT_COURSE_SEEDS = (101, 202, 303)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_r4_shadow_formal_20260909.json"
ARM_NAMES = (
    "frozen-parent",
    "r3-fixed-capacity",
    "pressure-driven-growth",
    "random-growth",
    "fixed-large",
)
PHASES = ("S", "G")


def _arm_summary(arm: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = arm.get("training_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    return {
        "scores": {phase: float(arm["scores"][phase]) for phase in PHASES},
        "candidate_lesion_delta": (
            None
            if arm.get("candidate_lesion_delta") is None
            else {
                phase: float(arm["candidate_lesion_delta"][phase])
                for phase in PHASES
            }
        ),
        "candidate_digest": arm.get("candidate_digest"),
        "candidate_unit_id": arm.get("candidate_unit_id"),
        "birth_anchor_unit_id": arm.get("birth_anchor_unit_id"),
        "birth_anchor_unit_ids": arm.get("birth_anchor_unit_ids"),
        "birth_anchor_weights": arm.get("birth_anchor_weights"),
        "counts": dict(arm.get("counts", {})),
        "parent_frozen": arm.get("parent_frozen"),
        "parent_substrate_unchanged": arm.get("parent_substrate_unchanged"),
        "training_diagnostics": {
            key: value for key, value in diagnostics.items() if key != "trace"
        },
        "shadow_fresh_restore": arm.get("shadow_fresh_restore"),
        "shadow_rollback_matches_bare": arm.get("shadow_rollback_matches_bare"),
        "mature_f1_owners_unchanged": arm.get("mature_f1_owners_unchanged"),
    }


def _cell_summary(
    report: Mapping[str, Any],
    *,
    model_seed: int,
    course: R4Course,
) -> dict[str, Any]:
    arms = report.get("arms", {})
    return {
        "model_seed": int(model_seed),
        "course_seed": int(course.course_seed),
        "course_label": course.label,
        "course": dict(report["course"]),
        "report_digest": content_digest(report),
        "technical_gates": dict(report["technical_gates"]),
        "technical_passed": bool(all(report["technical_gates"].values())),
        "candidate_only_smoke": dict(report["candidate_only_smoke"]),
        "arms": {name: _arm_summary(arms[name]) for name in ARM_NAMES},
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _stats(values: Sequence[float], *, positive_is_better: bool = False) -> dict[str, Any]:
    values = tuple(float(value) for value in values)
    better = (lambda value: value > 0.0) if positive_is_better else (lambda value: value < 0.0)
    non_worse = (
        (lambda value: value >= 0.0)
        if positive_is_better
        else (lambda value: value <= 0.0)
    )
    return {
        "count": len(values),
        "mean": _mean(values),
        "min": min(values, default=0.0),
        "max": max(values, default=0.0),
        "better_count": sum(1 for value in values if better(value)),
        "non_worse_count": sum(1 for value in values if non_worse(value)),
    }


def _deltas(
    cells: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    baseline: str,
    phase: str,
) -> tuple[float, ...]:
    return tuple(
        float(cell["arms"][arm]["scores"][phase])
        - float(cell["arms"][baseline]["scores"][phase])
        for cell in cells
    )


def run_formal(
    *,
    model_seeds: Sequence[int] = DEFAULT_MODEL_SEEDS,
    course_seeds: Sequence[int] = DEFAULT_COURSE_SEEDS,
) -> dict[str, Any]:
    normalized_model_seeds = tuple(int(seed) for seed in model_seeds)
    normalized_course_seeds = tuple(int(seed) for seed in course_seeds)
    if not normalized_model_seeds or not normalized_course_seeds:
        raise ValueError("R4 formal matrix requires model and course seeds")
    cells: list[dict[str, Any]] = []
    for model_seed in normalized_model_seeds:
        for course_seed in normalized_course_seeds:
            course = course_variant(course_seed)
            canary = run_canary(model_seed=model_seed, course=course)
            cells.append(
                _cell_summary(
                    canary,
                    model_seed=model_seed,
                    course=course,
                )
            )

    pressure_vs_fixed = {
        phase: _stats(
            _deltas(
                cells,
                arm="pressure-driven-growth",
                baseline="r3-fixed-capacity",
                phase=phase,
            )
        )
        for phase in PHASES
    }
    pressure_vs_fixed_large = {
        phase: _stats(
            _deltas(
                cells,
                arm="pressure-driven-growth",
                baseline="fixed-large",
                phase=phase,
            )
        )
        for phase in PHASES
    }
    pressure_lesion = {
        phase: _stats(
            tuple(
                float(cell["arms"]["pressure-driven-growth"]["candidate_lesion_delta"][phase])
                for cell in cells
            ),
            positive_is_better=True,
        )
        for phase in PHASES
    }
    technical_passed = all(bool(cell["technical_passed"]) for cell in cells)
    return {
        "format": FORMAL_FORMAT,
        "version": FORMAL_VERSION,
        "status": "passed" if technical_passed else "failed",
        "can_promote": False,
        "promotion_reason": "R4 formal matrix is diagnostic evidence; it cannot promote structure",
        "matrix": {
            "model_seeds": list(normalized_model_seeds),
            "course_seeds": list(normalized_course_seeds),
            "cell_count": len(cells),
            "same_five_arm_canary": True,
            "same_parent_owner_boundary": True,
            "same_pressure_contract": True,
            "same_course_budget": True,
        },
        "aggregate": {
            "pressure_vs_fixed_capacity": pressure_vs_fixed,
            "pressure_vs_fixed_large": pressure_vs_fixed_large,
            "pressure_candidate_lesion": pressure_lesion,
            "lower_mean_surprise_is_better": True,
            "positive_lesion_delta_means_candidate_helped": True,
        },
        "technical_gates": {
            "all_cells_technical_passed": technical_passed,
            "all_cells_have_candidate_only_smoke": all(
                bool(cell["candidate_only_smoke"]["parent_substrate_unchanged"])
                for cell in cells
            ),
            "all_cells_have_shared_parent_boundary": all(
                bool(cell["technical_gates"].get("shared_efficacy_parent"))
                and bool(cell["technical_gates"].get("matched_parent_learning_boundary"))
                for cell in cells
            ),
        },
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_formal()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "format": report["format"],
                "status": report["status"],
                "can_promote": report["can_promote"],
                "matrix": report["matrix"],
                "technical_gates": report["technical_gates"],
                "aggregate": report["aggregate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
