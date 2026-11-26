"""M5.S6B formal: verify failure-admission lesion recovery across seeds.

Same preregistered read as S6 formal: 3x3 (task_seed x learner_seed), every
cell must clear the grounding-lesion floor and pass the full S1 causal gate.
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

from scripts.training.eval_taiji_m5_s6b_failure_admission import (  # noqa: E402
    GROUNDING_MARGIN_FLOOR,
    run_cell,
)

FORMAT = "taiji-m5-s6b-failure-admission-formal-v1"
VERSION = 1
TASK_SEEDS = (0, 1, 2)
LEARNER_SEEDS = (17, 23, 31)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    cells: list[dict[str, Any]] = []
    for task_seed in TASK_SEEDS:
        for learner_seed in LEARNER_SEEDS:
            cell = run_cell(task_seed=task_seed, learner_seed=learner_seed)
            cells.append(cell)
            print(
                json.dumps(
                    {
                        "task_seed": task_seed,
                        "learner_seed": learner_seed,
                        "technical": cell["technical_gate_all_passed"],
                        "margin": round(cell["grounding_lesion_margin"], 6),
                        "failure_admitted": cell["data"]["failure_admitted_count"],
                    }
                )
            )

    margins = [float(c["grounding_lesion_margin"]) for c in cells]
    cleared = sum(1 for m in margins if m >= GROUNDING_MARGIN_FLOOR)
    gates_passed = sum(1 for c in cells if c["technical_gate_all_passed"])
    total = len(cells)
    robust = bool(cleared == total and gates_passed == total)

    report = {
        "format": FORMAT,
        "version": VERSION,
        "generated_at_epoch": int(time.time()),
        "status": "supported" if robust else "not-robust",
        "can_promote": False,
        "margin_recovery_robust": robust,
        "verdict": (
            "failure-admission lesion recovery clears the preregistered floor "
            "in every cell: real success AND failure outcomes give the "
            "internalization probes a reliably meaningful target"
            if robust
            else "failure-admission lesion recovery is not robust across seeds"
        ),
        "grounding_margin_floor": GROUNDING_MARGIN_FLOOR,
        "aggregate": {
            "cells": total,
            "cleared_floor": cleared,
            "gates_passed": gates_passed,
            "margin_mean": sum(margins) / total,
            "margin_min": min(margins),
            "margin_max": max(margins),
        },
        "matrix": {"task_seeds": list(TASK_SEEDS), "learner_seeds": list(LEARNER_SEEDS)},
        "cells": cells,
        "boundary": "M5.S6B formal only; failures enter with non-positive reward only; no provider, network, or real client write",
        "resources": {"total_elapsed_seconds": time.perf_counter() - started},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "margin_recovery_robust": robust,
                "cleared_floor": f"{cleared}/{total}",
                "gates_passed": f"{gates_passed}/{total}",
                "margin_mean": round(sum(margins) / total, 6),
                "margin_min": round(min(margins), 6),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
