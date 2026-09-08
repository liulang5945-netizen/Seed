"""R0.7: read-only re-evaluation of existing joint children with versioned verdicts.

Each completed child report already records parent_metrics, final_metrics and
checkpoint lineage.  This entry point re-judges those numbers through the R0.5
three-question contract (absolute / retention / incremental) without touching
the original report, so old JSON semantics stay green untouched.  A child whose
behaviour metrics are frozen relative to its parent is labelled explicitly:
retention may pass, but that is not new learning and the old B5 pass is not
re-declared from it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji.measurement_verdict import (  # noqa: E402
    CapabilityVerdict,
    judge_absolute,
    judge_incremental,
    judge_retention,
    verdict_report,
)

REEVAL_FORMAT = "taiji-r0-reevaluation-v1"
REEVAL_VERSION = 1
# Fixed, versioned absolute thresholds (pre-registered in the R0 contract).
ABSOLUTE_SPECS = {
    "sequence_holdout_bpb": {"direction": "lower_is_better", "threshold": 6.5, "metric": "bpb"},
    "memory_holdout_recall": {"direction": "higher_is_better", "threshold": 0.5, "metric": "recall"},
    "world_holdout_error": {"direction": "lower_is_better", "threshold": 0.1, "metric": "error"},
    "goal_holdout_success": {"direction": "higher_is_better", "threshold": 0.5, "metric": "success_rate"},
}
# Retention slack in metric units, and the incremental rule (no gain demanded
# from an already-saturated course, but a frozen course must be labelled).
RETENTION_TOLERANCE = {
    "sequence_holdout_bpb": 0.25,
    "memory_holdout_recall": 0.05,
    "world_holdout_error": 0.01,
    "goal_holdout_success": 0.05,
}
FROZEN_EPSILON = 1e-9


def reevaluate_report(path: Path) -> dict[str, Any]:
    """Return one re-evaluation entry for a completed child report."""

    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{path}: report must be a JSON object")
    parent_metrics = report.get("parent_metrics")
    final_metrics = report.get("final_metrics")
    if not isinstance(parent_metrics, dict) or not isinstance(final_metrics, dict):
        raise ValueError(f"{path}: parent_metrics and final_metrics are required")

    verdicts: dict[str, CapabilityVerdict] = {}
    frozen = True
    gains: list[str] = []
    for metric, spec in ABSOLUTE_SPECS.items():
        if metric not in final_metrics or metric not in parent_metrics:
            continue
        final = float(final_metrics[metric])
        parent = float(parent_metrics[metric])
        direction = spec["direction"]

        verdicts[f"{metric}:absolute"] = judge_absolute(
            ability_id=metric,
            metric=spec["metric"],
            value=final,
            threshold=spec["threshold"],
            direction=direction,
            detail="R0 fixed absolute threshold",
        )
        verdicts[f"{metric}:retention"] = judge_retention(
            ability_id=metric,
            metric=spec["metric"],
            child_value=final,
            parent_value=parent,
            tolerance=RETENTION_TOLERANCE[metric],
            direction=direction,
            detail="continued child vs direct parent",
        )
        delta = final - parent if direction == "higher_is_better" else parent - final
        if abs(final - parent) > FROZEN_EPSILON:
            frozen = False
        if delta > FROZEN_EPSILON:
            gains.append(f"{metric}:{delta:.6f}")
        verdicts[f"{metric}:incremental"] = judge_incremental(
            ability_id=metric,
            metric=spec["metric"],
            child_value=final,
            baseline_value=parent,
            target_delta=0.0,
            direction=direction,
            require_delta=False,
            detail=(
                "no gain demanded for saturated course"
                if abs(final - parent) <= FROZEN_EPSILON
                else f"observed_delta={delta:.6f}"
            ),
        )

    return {
        "report_path": str(path),
        "format": report.get("format"),
        "version": report.get("version"),
        "training_phases": report.get("training_phases"),
        "parent_checkpoint_digest": report.get("parent_checkpoint_digest"),
        "child_checkpoint_digest": report.get("child_checkpoint_digest"),
        "continuation_source_checkpoint_digest": report.get(
            "continuation_source_checkpoint_digest"
        ),
        "checkpoint_paths": report.get("checkpoint_paths"),
        "behavior_frozen": frozen,
        "behavior_gains": gains,
        "verdicts": verdict_report(verdicts)["verdicts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="completed child report JSON files")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / f"taiji_m2_r0_reevaluation_{datetime.now(timezone.utc):%Y%m%d}.json",
    )
    args = parser.parse_args(argv)

    entries = [reevaluate_report(path) for path in args.reports]
    payload = {
        "format": REEVAL_FORMAT,
        "version": REEVAL_VERSION,
        "count": len(entries),
        "entries": entries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for entry in entries:
        frozen = entry["behavior_frozen"]
        print(
            f"{Path(entry['report_path']).name:56s} phases={entry['training_phases']} "
            f"frozen={frozen} gains={entry['behavior_gains']}"
        )
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
