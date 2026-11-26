"""Run the deterministic R5C-S1 structural pressure canary."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    StructuralEvidenceLedger,
    StructuralRuntimeObservation,
    project_structural_growth_pressure,
)

REPORT_FORMAT = "taiji-w7-r5c-s1-structural-pressure-v1"


def _observation(
    tick: int,
    *,
    task_slice_id: str,
    partition: str,
    prediction_error: float = 0.7,
) -> StructuralRuntimeObservation:
    return StructuralRuntimeObservation(
        network_id="network:pressure-canary",
        region_id="region:pressure-canary",
        tick=tick,
        usage=0.5,
        resource_pressure=0.2,
        prediction_error=prediction_error,
        learning_gain=0.1,
        holdout_transfer=0.8 if partition == "holdout" else 0.0,
        evidence_id=f"pressure-canary:{partition}:{task_slice_id}:{tick}",
        task_slice_id=task_slice_id,
        partition=partition,
    )


def evaluate() -> dict[str, object]:
    ledger = StructuralEvidenceLedger(window_capacity=1)
    for observation in (
        _observation(1, task_slice_id="task-a", partition="train"),
        _observation(2, task_slice_id="task-b", partition="train", prediction_error=0.6),
        _observation(3, task_slice_id="task-a", partition="holdout"),
    ):
        ledger.append(observation)

    before = ledger.to_payload()
    projection = project_structural_growth_pressure(ledger.sealed_summaries)
    projection_roundtrip = (
        project_structural_growth_pressure(ledger.sealed_summaries).to_payload()
        == projection.to_payload()
    )
    single_slice_rejected = False
    try:
        project_structural_growth_pressure(
            ledger.sealed_summaries,
            minimum_train_task_slices=3,
        )
    except ValueError:
        single_slice_rejected = True

    metrics = {
        "cross_task_train_slices": len(projection.train_task_slice_ids) == 2,
        "separate_holdout": projection.holdout_window_count == 1,
        "train_error_isolated": math.isclose(
            projection.mean_prediction_error,
            0.65,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "resource_state_is_runtime_derived": math.isclose(
            projection.mean_resource_state,
            0.8,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "projection_roundtrip": projection_roundtrip,
        "single_slice_rejected": single_slice_rejected,
        "ledger_unchanged": ledger.to_payload() == before,
        "no_topology_candidate": True,
    }
    return {
        "format": REPORT_FORMAT,
        "projection": projection.to_payload(),
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "only sealed windows with independent train task slices may produce a "
                "non-mutating pressure projection; holdout/retention remain validation-only"
            ),
        },
        "boundary": (
            "This canary produces metrics only. It does not invoke the growth controller, "
            "create a proposal, admit topology, or claim open-domain autonomy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s1_structural_pressure_20260830.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
