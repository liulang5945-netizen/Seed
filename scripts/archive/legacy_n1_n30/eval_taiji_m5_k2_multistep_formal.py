"""M5.K2 formal: cross-seed robustness of the multistep composition canary.

The formal contract is frozen in
``plans/reference/M5_K2_FORMAL_PREREGISTRATION_20260909.md``.  This runner
imports the canary ``run_cell`` without copying its course logic, then applies
the preregistered per-cell gates and aggregate decision.
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

from scripts.training.eval_taiji_m5_k2_multistep_composition import (  # noqa: E402
    run_cell,
)

REPORT_FORMAT = "taiji-m5-k2-multistep-composition-formal-v1"
VERSION = 1
TASK_SEEDS = (0, 1, 2)
LEARNER_SEEDS = (17, 23, 31)
HOLDOUT_FLOOR = 0.75
SEPARATION_FLOOR = 0.25
EPSILON = 1e-12


def _success_read_admission(arm: dict[str, Any]) -> tuple[int, int]:
    """Return admitted/total over successful workspace.read rows."""

    rows = [
        row
        for episode in arm["episodes"]
        for row in episode["rows"]
        if row.get("intent_kind") == "workspace.read" and row.get("real_success")
    ]
    admitted = [row for row in rows if row.get("outcome_admitted")]
    return len(admitted), len(rows)


def _all_rows_well_formed(arm: dict[str, Any]) -> bool:
    episodes = arm.get("episodes", [])
    if len(episodes) != 4:
        return False
    required = {
        "tick",
        "path",
        "transition_status",
        "semantic_status",
        "accepted",
        "real_success",
        "outcome_admitted",
    }
    return all(
        len(episode.get("rows", [])) == 3 and all(required.issubset(row) for row in episode["rows"])
        for episode in episodes
    )


def _cell_checks(cell: dict[str, Any]) -> dict[str, Any]:
    arm = cell["arms"]["A-full-chain"]
    admitted, success_read_total = _success_read_admission(arm)
    technical = {
        "canary_technical_gate": bool(cell["technical_gate_all_passed"]),
        "all_holdout_rows_produced": _all_rows_well_formed(arm),
        "checkpoint_mask_gate": bool(cell["checkpoint_gate"]["passed"]),
    }
    primary = {
        "a_holdout_ge_floor": cell["metrics"]["a_holdout_success"] >= HOLDOUT_FLOOR,
        "a_minus_b_ge_floor": cell["metrics"]["a_minus_b"] >= SEPARATION_FLOOR,
        "a_minus_c_ge_floor": cell["metrics"]["a_minus_c"] >= SEPARATION_FLOOR,
        # Fail closed: no successful read cannot demonstrate admission.
        "success_read_admission_rate_1p0": (
            success_read_total > 0 and admitted == success_read_total
        ),
        "reward_variance_positive": cell["metrics"]["a_reward_variance"] > EPSILON,
    }
    return {
        "technical": technical,
        "primary": primary,
        "cell_passed": all(technical.values()) and all(primary.values()),
        "success_read_admission": {"admitted": admitted, "total": success_read_total},
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
    reward_variance = series("a_reward_variance")
    cells_passed = sum(1 for entry in cells if entry["checks"]["cell_passed"])
    technical_cells = sum(1 for entry in cells if all(entry["checks"]["technical"].values()))
    admission_cells = sum(
        1 for entry in cells if entry["checks"]["primary"]["success_read_admission_rate_1p0"]
    )
    variance_cells = sum(
        1 for entry in cells if entry["checks"]["primary"]["reward_variance_positive"]
    )
    robust = cells_passed == len(cells)
    aggregate_checks = {
        "technical_gates_all_cells": technical_cells == len(cells),
        "a_holdout_all_cells_ge_floor": all(value >= HOLDOUT_FLOOR for value in a_holdout),
        "a_minus_b_all_cells_ge_floor": all(value >= SEPARATION_FLOOR for value in a_minus_b),
        "a_minus_c_all_cells_ge_floor": all(value >= SEPARATION_FLOOR for value in a_minus_c),
        "success_read_admission_all_cells": admission_cells == len(cells),
        "reward_variance_all_cells": variance_cells == len(cells),
        "all_cells_passed": cells_passed == len(cells),
    }
    payload = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if robust else "failed",
        "can_promote": False,
        "preregistration": "plans/reference/M5_K2_FORMAL_PREREGISTRATION_20260909.md",
        "matrix": {
            "task_seeds": list(TASK_SEEDS),
            "learner_seeds": list(LEARNER_SEEDS),
        },
        "metric_contract": {
            "primary": "absolute three-step holdout episode success rate",
            "comparison": "episode success deltas A-B and A-C",
            "success_definition": "all three planner decisions accepted and real execution succeeded",
            "parent_retention": None,
            "parent_retention_note": (
                "K2 uses standalone semantic/transition learners with learn=False; "
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
            "a_reward_variance": _stats(reward_variance),
            "technical_cells_passed": f"{technical_cells}/{len(cells)}",
            "success_read_admission_cells_1p0": f"{admission_cells}/{len(cells)}",
            "reward_variance_positive_cells": f"{variance_cells}/{len(cells)}",
            "cells_passed": f"{cells_passed}/{len(cells)}",
        },
        "aggregate_checks": aggregate_checks,
        "verdict": {"robust": robust, "cells_passed": f"{cells_passed}/{len(cells)}"},
        "elapsed_seconds": time.perf_counter() - started,
        "boundary": (
            "M5.K2 formal only; real read-only executions in isolated temporary "
            "workspaces; no provider, network, client write, CUDA, or K3 feedback; "
            "can_promote stays false."
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
