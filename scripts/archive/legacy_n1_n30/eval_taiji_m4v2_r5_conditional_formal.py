"""M4.V2.R5 conditional-modularity formal: preregistered 3x3 matrix.

Runs the pre-registered design from
``plans/reference/M4V2_R5_CONDITIONAL_MODULARITY_PREREGISTRATION_20260909.md``
over model seeds 71/83/97 x course seeds 101/202/303 with three arms
(fixed-capacity, fixed-large control, conditional module), the resource
normalization caps, and the two-exit stop line.  The primary read is the
conditional module versus fixed-large on unseen-combination G holdouts:
``non-worse >= 7/9`` cells with a better mean, S holdouts non-worse on all
cells, and route lesion observable.  ``can_promote=false`` is fixed; the
verdict is an architecture-level branch, not a capability promotion.
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

from scripts.training.eval_taiji_m4v2_r4_shadow import course_variant  # noqa: E402
from scripts.training.eval_taiji_m4v2_r5_conditional_canary import (  # noqa: E402
    run_cell,
)

FORMAT = "taiji-m4v2-r5-conditional-formal-v1"
VERSION = 1
MODEL_SEEDS = (71, 83, 97)
COURSE_SEEDS = (101, 202, 303)
WALL_CLOCK_MULTIPLIER_CAP = 1.5
WORKING_SET_MULTIPLIER_CAP = 1.25
PRIMARY_NON_WORSE_REQUIRED = 7


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    cells: list[dict[str, Any]] = []
    resource_violations: list[str] = []
    for model_seed in MODEL_SEEDS:
        for course_seed in COURSE_SEEDS:
            course = course_variant(course_seed)
            cell = run_cell(model_seed=model_seed, course=course)
            label = f"model{model_seed}-course{course_seed}"

            conditional_seconds = float(cell["resource_records"]["conditional_train_seconds"])
            fixed_large_seconds = float(cell["arms"]["fixed_large"]["resources"]["train_seconds"])
            wall_clock_cap = fixed_large_seconds * WALL_CLOCK_MULTIPLIER_CAP
            wall_clock_ok = conditional_seconds <= wall_clock_cap
            if not wall_clock_ok:
                resource_violations.append(
                    f"{label}: conditional {conditional_seconds:.1f}s > "
                    f"{wall_clock_cap:.1f}s (fixed-large {fixed_large_seconds:.1f}s x "
                    f"{WALL_CLOCK_MULTIPLIER_CAP})"
                )

            working_set_cap: float | None = None
            working_set_ok = True
            fixed_large_ws = cell["resource_records"].get("working_set_bytes_after_fixed_large")
            conditional_ws = cell["resource_records"]["working_set_bytes_after_conditional"]
            if fixed_large_ws:
                working_set_cap = float(fixed_large_ws) * WORKING_SET_MULTIPLIER_CAP
                working_set_ok = float(conditional_ws) <= working_set_cap
                if not working_set_ok:
                    resource_violations.append(
                        f"{label}: conditional working set {conditional_ws} > {working_set_cap:.0f}"
                    )

            cells.append(
                {
                    "label": label,
                    "model_seed": int(model_seed),
                    "course_seed": int(course_seed),
                    "course_label": course.label,
                    "technical_gate_all_passed": cell["technical_gate_all_passed"],
                    "resource_records": {
                        **cell["resource_records"],
                        "fixed_large_train_seconds": fixed_large_seconds,
                        "wall_clock_cap": wall_clock_cap,
                        "wall_clock_ok": wall_clock_ok,
                        "working_set_cap": working_set_cap,
                        "working_set_ok": working_set_ok,
                    },
                    "resource_violations": [v for v in resource_violations if v.startswith(label)],
                    "arms": cell["arms"],
                    "conditional_module": cell["conditional_module"],
                    "checks": cell["checks"],
                }
            )
            print(
                json.dumps(
                    {
                        "cell": label,
                        "technical": cell["technical_gate_all_passed"],
                        "route_lesion_delta": cell["conditional_module"]["route_lesion_delta"],
                    }
                )
            )

    valid_cells = [c for c in cells if c["technical_gate_all_passed"]]
    g_deltas = [
        float(c["conditional_module"]["scores"]["G"])
        - float(c["arms"]["fixed_large"]["scores"]["G"])
        for c in valid_cells
    ]
    s_deltas = [
        float(c["conditional_module"]["scores"]["S"])
        - float(c["arms"]["fixed_large"]["scores"]["S"])
        for c in valid_cells
    ]
    g_non_worse = sum(1 for v in g_deltas if v <= 0.0)
    s_non_worse = sum(1 for v in s_deltas if v <= 0.0)
    route_lesion_g_positive = sum(
        1 for c in valid_cells if float(c["conditional_module"]["route_lesion_delta"]["G"]) > 0.0
    )
    route_lesion_s_positive = sum(
        1 for c in valid_cells if float(c["conditional_module"]["route_lesion_delta"]["S"]) > 0.0
    )

    enough_valid_cells = len(valid_cells) >= PRIMARY_NON_WORSE_REQUIRED
    primary_passed = bool(
        enough_valid_cells
        and g_non_worse >= PRIMARY_NON_WORSE_REQUIRED
        and (sum(g_deltas) / max(1, len(g_deltas))) < 0.0
    )
    s_non_worse_passed = s_non_worse == len(valid_cells)
    route_lesion_passed = route_lesion_g_positive == len(
        valid_cells
    ) and route_lesion_s_positive == len(valid_cells)
    resource_clean = not resource_violations
    hypothesis_supported = bool(
        primary_passed and s_non_worse_passed and route_lesion_passed and resource_clean
    )

    aggregate: dict[str, Any] = {
        "g_delta_vs_fixed_large": {
            "values": g_deltas,
            "mean": sum(g_deltas) / max(1, len(g_deltas)),
            "non_worse_count": g_non_worse,
            "required_non_worse": PRIMARY_NON_WORSE_REQUIRED,
        },
        "s_delta_vs_fixed_large": {
            "values": s_deltas,
            "mean": sum(s_deltas) / max(1, len(s_deltas)),
            "non_worse_count": s_non_worse,
        },
        "route_lesion_g_positive_cells": route_lesion_g_positive,
        "route_lesion_s_positive_cells": route_lesion_s_positive,
    }

    exit_branch = (
        "conditional modularity outperforms equal-parameter preallocation "
        "under resource normalization; architecture discussion may proceed"
        if hypothesis_supported
        else "hypothesis rejected: structural-growth mainline closes and "
        "remaining routes move to the M5 periphery"
    )

    report = {
        "format": FORMAT,
        "version": VERSION,
        "generated_at_epoch": int(time.time()),
        "status": "supported" if hypothesis_supported else "rejected",
        "can_promote": False,
        "hypothesis_supported": hypothesis_supported,
        "verdict": exit_branch,
        "resource_caps": {
            "wall_clock_multiplier": WALL_CLOCK_MULTIPLIER_CAP,
            "working_set_multiplier": WORKING_SET_MULTIPLIER_CAP,
            "violations": resource_violations,
            "clean": resource_clean,
        },
        "retention_gate": {
            "g_non_worse_cells": g_non_worse,
            "g_required": PRIMARY_NON_WORSE_REQUIRED,
            "s_non_worse_cells": s_non_worse,
            "route_lesion_g_positive_cells": route_lesion_g_positive,
            "route_lesion_s_positive_cells": route_lesion_s_positive,
            "valid_cells": len(valid_cells),
        },
        "aggregate": aggregate,
        "cells": cells,
        "resources": {"total_elapsed_seconds": time.perf_counter() - started},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    g_delta_mean = aggregate["g_delta_vs_fixed_large"]["mean"]
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "hypothesis_supported": hypothesis_supported,
                "verdict": exit_branch,
                "g_delta_mean": g_delta_mean,
                "g_non_worse": g_non_worse,
                "s_non_worse": s_non_worse,
                "resource_violations": len(resource_violations),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
