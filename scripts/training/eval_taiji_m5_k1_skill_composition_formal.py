"""M5.K1 formal: multi-seed robustness of the skill-composition canary.

Preregistration: ``plans/reference/M5_K1_FORMAL_PREREGISTRATION
_20260909.md``.  A 9-cell matrix (3 task seeds x 3 learner seeds); every
cell reuses the canary ``run_cell`` unchanged, so the arms, the K1.1
typed fact->feature binding, and the per-cell criteria are frozen.

Metric contract (M4.V2.R0 semantics): the primary measure is the
absolute unseen-triple real-execution success rate, reported separately
from the arm deltas.  ``parent_retention`` is explicitly ``null``: the
K1 chain writes no F1/memory/structural owner (``learn=False``; the
semantic learner is standalone), so no parent baseline exists to
compare against, and none is fabricated.
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

from scripts.training.eval_taiji_m5_k1_skill_composition import run_cell  # noqa: E402

REPORT_FORMAT = "taiji-m5-k1-skill-composition-formal-v1"
VERSION = 1
TASK_SEEDS = (0, 1, 2)
LEARNER_SEEDS = (17, 23, 31)
UNSEEN_FLOOR = 0.8
UNSEEN_MEAN_FLOOR = 0.9
SEPARATION_FLOOR = 0.3
EPSILON = 1e-12


def _population_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _success_read_admission(arm: dict[str, Any]) -> tuple[int, int]:
    """Return (admitted, total) over successful workspace.read outcomes."""

    success_reads = [
        row
        for row in arm["rows"]
        if row.get("intent_kind") == "workspace.read" and row.get("real_success")
    ]
    admitted = [row for row in success_reads if row.get("outcome_admitted")]
    return len(admitted), len(success_reads)


def _cell_checks(cell: dict[str, Any]) -> dict[str, Any]:
    arm = cell["arms"]["A-full-chain"]
    rows = arm["rows"]
    admitted, success_read_total = _success_read_admission(arm)
    rewards = [float(row.get("real_reward", 0.0)) for row in rows]
    technical = {
        "canary_technical_gate": bool(cell["technical_gate_all_passed"]),
        "stage2_interface_ok": bool(
            cell["stage2_interface"]["all_worlds_wellformed"]
            and cell["stage2_interface"]["all_contracts_ok"]
        ),
        "all_rows_produced": (
            len(rows) == 6
            and all(
                "semantic_status" in row
                and "real_success" in row
                and "outcome_admitted" in row
                for row in rows
            )
        ),
    }
    primary = {
        "a_unseen_ge_floor": cell["metrics"]["a_unseen_success"] >= UNSEEN_FLOOR,
        "a_minus_b_ge_floor": cell["metrics"]["a_minus_b"] >= SEPARATION_FLOOR,
        "a_minus_c_ge_floor": cell["metrics"]["a_minus_c"] >= SEPARATION_FLOOR,
        # Fail-closed: a cell with zero successful reads cannot demonstrate
        # admission, so an empty set fails the gate instead of passing vacuously.
        "success_read_admission_rate_1p0": (
            success_read_total > 0 and admitted == success_read_total
        ),
        "reward_variance_positive": _population_variance(rewards) > EPSILON,
    }
    return {
        "technical": technical,
        "primary": primary,
        "cell_passed": all(technical.values()) and all(primary.values()),
        "success_read_admission": {"admitted": admitted, "total": success_read_total},
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
                        "a_unseen": cell["metrics"]["a_unseen_success"],
                        "a_minus_b": cell["metrics"]["a_minus_b"],
                        "a_minus_c": cell["metrics"]["a_minus_c"],
                        "cell_passed": checks["cell_passed"],
                    }
                ),
                flush=True,
            )

    def _series(metric: str) -> list[float]:
        return [float(entry["cell"]["metrics"][metric]) for entry in cells]

    def _stats(values: list[float]) -> dict[str, float]:
        return {
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }

    a_unseen = _series("a_unseen_success")
    a_minus_b = _series("a_minus_b")
    a_minus_c = _series("a_minus_c")
    cells_passed = sum(1 for entry in cells if entry["checks"]["cell_passed"])
    technical_cells = sum(
        1 for entry in cells if all(entry["checks"]["technical"].values())
    )
    admission_cells = sum(
        1
        for entry in cells
        if entry["checks"]["primary"]["success_read_admission_rate_1p0"]
    )
    variance_cells = sum(
        1 for entry in cells if entry["checks"]["primary"]["reward_variance_positive"]
    )
    robust = cells_passed == len(cells)

    aggregate = {
        "a_unseen": {**_stats(a_unseen), "cells_ge_floor": sum(1 for v in a_unseen if v >= UNSEEN_FLOOR)},
        "a_minus_b": {**_stats(a_minus_b), "cells_ge_floor": sum(1 for v in a_minus_b if v >= SEPARATION_FLOOR)},
        "a_minus_c": {**_stats(a_minus_c), "cells_ge_floor": sum(1 for v in a_minus_c if v >= SEPARATION_FLOOR)},
        "technical_cells_passed": f"{technical_cells}/{len(cells)}",
        "success_read_admission_rate_cells_1p0": f"{admission_cells}/{len(cells)}",
        "reward_variance_positive_cells": f"{variance_cells}/{len(cells)}",
        "cells_passed": f"{cells_passed}/{len(cells)}",
    }
    aggregate_checks = {
        "technical_gates_all_cells": technical_cells == len(cells),
        "a_unseen_all_cells_ge_floor": all(v >= UNSEEN_FLOOR for v in a_unseen),
        "a_unseen_mean_ge_floor": (sum(a_unseen) / len(a_unseen)) >= UNSEEN_MEAN_FLOOR,
        "a_minus_b_all_cells_ge_floor": all(v >= SEPARATION_FLOOR for v in a_minus_b),
        "a_minus_c_all_cells_ge_floor": all(v >= SEPARATION_FLOOR for v in a_minus_c),
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
        "preregistration": (
            "plans/reference/M5_K1_FORMAL_PREREGISTRATION_20260909.md"
        ),
        "matrix": {
            "task_seeds": list(TASK_SEEDS),
            "learner_seeds": list(LEARNER_SEEDS),
        },
        "metric_contract": {
            "primary": "absolute unseen-triple real-execution success rate (per cell, per arm)",
            "comparison": "arm deltas A-B and A-C (reported separately, never mixed with the absolute rate)",
            "parent_retention": None,
            "parent_retention_note": (
                "K1 chain writes no F1/memory/structural owner (learn=False; "
                "standalone semantic learner), so no parent baseline exists; "
                "retention is explicitly null rather than fabricated."
            ),
        },
        "cells": cells,
        "aggregate": aggregate,
        "aggregate_checks": aggregate_checks,
        "verdict": {"robust": robust, "cells_passed": f"{cells_passed}/{len(cells)}"},
        "elapsed_seconds": time.perf_counter() - started,
        "boundary": (
            "M5.K1 formal only; real read-only executions in process-owned "
            "temporary workspaces; no provider, network, or real client write; "
            "can_promote stays false until an aggregate scorecard computes it."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "robust": robust,
                "cells_passed": f"{cells_passed}/{len(cells)}",
                "a_unseen": aggregate["a_unseen"],
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
