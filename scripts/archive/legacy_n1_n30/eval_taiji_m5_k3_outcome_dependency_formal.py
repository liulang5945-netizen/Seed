"""M5.K3 formal: cross-seed robustness of outcome-dependent world feedback.

The formal contract is frozen in
``plans/reference/M5_K3_FORMAL_PREREGISTRATION_20260909.md``.  This runner
imports the passed single-cell canary and applies only the preregistered
matrix and aggregate gates; it does not duplicate projection, learning,
planner, or Workbench execution logic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m5_k3_outcome_dependency import (  # noqa: E402
    run_cell,
)

REPORT_FORMAT = "taiji-m5-k3-outcome-dependency-formal-v1"
VERSION = 1
TASK_SEEDS = (0, 1, 2)
LEARNER_SEEDS = (17, 23, 31)
HOLDOUT_FLOOR = 0.75
SEPARATION_FLOOR = 0.25
EPSILON = 1e-12
ARM_NAMES = ("A-full-feedback", "B-no-feedback", "C-outcome-lesion")


def _arm_rows_produced(arm: dict[str, Any]) -> bool:
    episodes = arm.get("episodes", [])
    required = {"step", "path", "step_success"}
    return (
        bool(episodes)
        and len(episodes) == 4
        and all(
            episode.get("rows") and all(required.issubset(row) for row in episode["rows"])
            for episode in episodes
        )
    )


def _a_rows_well_formed(arm: dict[str, Any]) -> bool:
    episodes = arm.get("episodes", [])
    common = {"step", "path", "transition_status", "step_success"}
    probe_fields = {
        "semantic_status",
        "outcome_admitted",
        "projection_reason",
        "projection_accepted",
    }
    dependent_fields = {
        "feedback_input_present",
        "dependency_projection_accepted",
        "dependency_gate",
        "lineage_matches",
    }
    if not episodes or len(episodes) != 4:
        return False
    for episode in episodes:
        rows = episode.get("rows", [])
        if len(rows) != 3:
            return False
        if not all(common.issubset(row) for row in rows):
            return False
        if not probe_fields.issubset(rows[0]):
            return False
        if not all(dependent_fields.issubset(row) for row in rows[1:]):
            return False
    return True


def _projection_contract(arm: dict[str, Any], expected_reason: str) -> bool:
    """Ensure the negative-control arm really disables the dependency path."""

    for episode in arm.get("episodes", []):
        rows = episode.get("rows", [])
        if not rows or rows[0].get("projection_reason") != expected_reason:
            return False
        if rows[0].get("projection_accepted", False):
            return False
        for row in rows[1:]:
            if row.get("dependency_gate", False):
                return False
    return True


def _cell_checks(cell: dict[str, Any]) -> dict[str, Any]:
    arms = cell["arms"]
    full = arms["A-full-feedback"]
    no_feedback = arms["B-no-feedback"]
    lesion = arms["C-outcome-lesion"]
    canary_checks = cell.get("checks", {})
    technical = {
        "canary_checks_all_passed": bool(cell.get("technical_gate_all_passed")),
        "all_arm_rows_produced": all(_arm_rows_produced(arms[name]) for name in ARM_NAMES),
        "a_rows_well_formed": _a_rows_well_formed(full),
        "b_projection_disabled": _projection_contract(no_feedback, "no_feedback"),
        "c_projection_lesioned": _projection_contract(lesion, "outcome_feedback_lesioned"),
        "prefit_checkpoint_gate": bool(canary_checks.get("prefit_checkpoint_gate")),
        "postfit_checkpoint_gate": bool(canary_checks.get("postfit_checkpoint_gate")),
    }
    primary = {
        "a_holdout_ge_floor": cell["metrics"]["a_holdout_success"] >= HOLDOUT_FLOOR,
        "a_minus_b_ge_floor": cell["metrics"]["a_minus_b"] >= SEPARATION_FLOOR,
        "a_minus_c_ge_floor": cell["metrics"]["a_minus_c"] >= SEPARATION_FLOOR,
        "a_train_success_is_1": cell["metrics"]["a_train_success"] == 1.0,
        "train_episode_count_at_least_6": len(cell["train_arm"]["episodes"]) >= 6,
        "a_probe_admission_rate_1p0": bool(
            full.get("all_probe_outcomes_admitted")
            and full.get("probe_outcome_admission_rate") == 1.0
        ),
        "a_feedback_lineage_rate_1p0": bool(
            full.get("all_feedback_lineages_admitted")
            and full.get("feedback_lineage_admission_rate") == 1.0
        ),
        "feedback_reward_variance_positive": (
            cell["metrics"]["feedback_reward_variance"] > EPSILON
        ),
        "a_model_feedback_facts_consumed": bool(
            canary_checks.get("a_model_feedback_facts_consumed")
        ),
    }
    return {
        "technical": technical,
        "primary": primary,
        "cell_passed": all(technical.values()) and all(primary.values()),
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    cells: list[dict[str, Any]] = []
    for task_seed in TASK_SEEDS:
        for learner_seed in LEARNER_SEEDS:
            cell = run_cell(task_seed=task_seed, learner_seed=learner_seed)
            checks = _cell_checks(cell)
            cells.append(
                {
                    "task_seed": task_seed,
                    "learner_seed": learner_seed,
                    "cell": cell,
                    "checks": checks,
                }
            )
            print(
                json.dumps(
                    {
                        "cell": f"{task_seed}x{learner_seed}",
                        "a_holdout": cell["metrics"]["a_holdout_success"],
                        "a_minus_b": cell["metrics"]["a_minus_b"],
                        "a_minus_c": cell["metrics"]["a_minus_c"],
                        "a_train": cell["metrics"]["a_train_success"],
                        "cell_passed": checks["cell_passed"],
                    }
                ),
                flush=True,
            )

    def series(metric: str) -> list[float]:
        return [float(entry["cell"]["metrics"][metric]) for entry in cells]

    a_holdout = series("a_holdout_success")
    a_minus_b = series("a_minus_b")
    a_minus_c = series("a_minus_c")
    a_train = series("a_train_success")
    feedback_variance = series("feedback_reward_variance")
    cells_passed = sum(1 for entry in cells if entry["checks"]["cell_passed"])
    technical_cells = sum(1 for entry in cells if all(entry["checks"]["technical"].values()))
    admission_cells = sum(
        1 for entry in cells if entry["checks"]["primary"]["a_probe_admission_rate_1p0"]
    )
    lineage_cells = sum(
        1 for entry in cells if entry["checks"]["primary"]["a_feedback_lineage_rate_1p0"]
    )
    variance_cells = sum(
        1 for entry in cells if entry["checks"]["primary"]["feedback_reward_variance_positive"]
    )
    robust = cells_passed == len(cells)
    aggregate_checks = {
        "technical_gates_all_cells": technical_cells == len(cells),
        "a_holdout_all_cells_ge_floor": all(value >= HOLDOUT_FLOOR for value in a_holdout),
        "a_minus_b_all_cells_ge_floor": all(value >= SEPARATION_FLOOR for value in a_minus_b),
        "a_minus_c_all_cells_ge_floor": all(value >= SEPARATION_FLOOR for value in a_minus_c),
        "probe_admission_all_cells": admission_cells == len(cells),
        "lineage_admission_all_cells": lineage_cells == len(cells),
        "feedback_variance_all_cells": variance_cells == len(cells),
        "all_cells_passed": cells_passed == len(cells),
    }
    payload = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if robust else "failed",
        "can_promote": False,
        "preregistration": "plans/reference/M5_K3_FORMAL_PREREGISTRATION_20260909.md",
        "matrix": {
            "task_seeds": list(TASK_SEEDS),
            "learner_seeds": list(LEARNER_SEEDS),
            "execution_order": "task_seed_outer_learner_seed_inner_serial",
        },
        "metric_contract": {
            "primary": "absolute three-step outcome-dependent holdout episode success rate",
            "comparison": "episode success deltas A-B and A-C",
            "success_definition": (
                "all three planner decisions accepted, real outcomes complete the "
                "pre-registered branch, and dependency lineage remains continuous"
            ),
            "parent_retention": None,
            "parent_retention_note": (
                "K3 uses standalone semantic/transition learners with learn=False; "
                "no F1, memory, or structural parent owner is written."
            ),
        },
        "cells": cells,
        "aggregate": {
            "a_holdout": {
                **_stats(a_holdout),
                "cells_ge_floor": sum(value >= HOLDOUT_FLOOR for value in a_holdout),
            },
            "a_minus_b": {
                **_stats(a_minus_b),
                "cells_ge_floor": sum(value >= SEPARATION_FLOOR for value in a_minus_b),
            },
            "a_minus_c": {
                **_stats(a_minus_c),
                "cells_ge_floor": sum(value >= SEPARATION_FLOOR for value in a_minus_c),
            },
            "a_train": _stats(a_train),
            "feedback_reward_variance": _stats(feedback_variance),
            "technical_cells_passed": f"{technical_cells}/{len(cells)}",
            "probe_admission_cells_1p0": f"{admission_cells}/{len(cells)}",
            "lineage_admission_cells_1p0": f"{lineage_cells}/{len(cells)}",
            "feedback_variance_positive_cells": f"{variance_cells}/{len(cells)}",
            "cells_passed": f"{cells_passed}/{len(cells)}",
        },
        "aggregate_checks": aggregate_checks,
        "verdict": {"robust": robust, "cells_passed": f"{cells_passed}/{len(cells)}"},
        "elapsed_seconds": time.perf_counter() - started,
        "boundary": (
            "M5.K3 formal only; real success/failure read outcomes in isolated "
            "temporary workspaces; no provider, network, client write, CUDA, "
            "default runtime integration, or structural promotion; can_promote "
            "stays false."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "robust": robust,
                "cells_passed": f"{cells_passed}/{len(cells)}",
                "failed_aggregate_checks": [
                    key for key, value in aggregate_checks.items() if not value
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if robust else 1


if __name__ == "__main__":
    raise SystemExit(main())
