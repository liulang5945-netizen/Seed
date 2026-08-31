"""Run the R5C-S19 provenance-aware pressure projection canary."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_structural_bridge import _build_model  # noqa: E402
from taiji import (  # noqa: E402
    StructuralEvidenceLedger,
    StructuralRuntimeObservation,
    TSKV8Adapter,
    project_structural_growth_pressure,
)

REPORT_FORMAT = "taiji-w7-r5c-s19-structural-provenance-projection-v1"


def _observation(
    tick: int,
    evidence_id: str,
    *,
    task_slice_id: str,
    partition: str,
) -> StructuralRuntimeObservation:
    return StructuralRuntimeObservation(
        network_id="standalone:adaptive.cortex",
        region_id="region:pressure-canary",
        tick=tick,
        usage=0.5,
        resource_pressure=0.2,
        prediction_error=0.7,
        learning_gain=0.1,
        holdout_transfer=0.8 if partition == "holdout" else 0.0,
        evidence_id=evidence_id,
        task_slice_id=task_slice_id,
        partition=partition,
    )


def _expect_rejection(factory: object) -> bool:
    try:
        factory()  # type: ignore[operator]
    except (KeyError, TypeError, ValueError):
        return True
    return False


def evaluate() -> dict[str, object]:
    model, region = _build_model()
    model._structural_evidence_ledger = StructuralEvidenceLedger(window_capacity=1)
    observations = (
        _observation(1, "projection:train:one", task_slice_id="task:one", partition="train"),
        _observation(2, "projection:train:two", task_slice_id="task:two", partition="train"),
        _observation(3, "projection:holdout", task_slice_id="task:holdout", partition="holdout"),
    )
    for observation in observations:
        model.record_structural_runtime_observation(observation)
    before = project_structural_growth_pressure(
        model.structural_evidence_summaries,
        minimum_train_task_slices=2,
        minimum_train_windows=2,
    )
    scheduled = model.schedule_structural_growth_from_evidence(
        network_id="standalone:adaptive.cortex",
        region_id="region:pressure-canary",
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": "u2"},
        minimum_train_task_slices=2,
        minimum_train_windows=2,
    )
    candidate_count = len(model.structural_proposal_candidates)
    compaction = model.compact_structural_evidence_history(keep_latest_per_stream=1)
    after = project_structural_growth_pressure(
        model.structural_evidence_summaries,
        minimum_train_task_slices=2,
        minimum_train_windows=2,
        historical_snapshots=model.structural_evidence_ledger.pressure_snapshots,
    )
    repeated = model.schedule_structural_growth_from_evidence(
        network_id="standalone:adaptive.cortex",
        region_id="region:pressure-canary",
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": "u2"},
        minimum_train_task_slices=2,
        minimum_train_windows=2,
    )
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    tampered = copy.deepcopy(checkpoint)
    snapshot_payload = tampered["components"]["structural_runtime"]["evidence_ledger"][
        "pressure_snapshots"
    ][0]
    snapshot_payload["train_window_count"] = int(snapshot_payload["train_window_count"]) + 1

    metrics = {
        "candidate_created_from_active_evidence": scheduled.status == "candidate_created",
        "projection_identity_survives_compaction": after == before,
        "compaction_keeps_consumed_lineage_out_of_active_windows": (
            compaction.status == "compacted"
            and len(compaction.compacted_window_digests) == 2
            and len(model.structural_evidence_summaries) == 1
            and set(compaction.compacted_window_digests).isdisjoint(
                {item.window_digest for item in model.structural_evidence_summaries}
            )
        ),
        "repeated_schedule_does_not_create_candidate": (
            repeated.status == "waiting"
            and repeated.reason == "no_new_sealed_window"
            and len(model.structural_proposal_candidates) == candidate_count
        ),
        "checkpoint_preserves_candidate_and_pressure_snapshot": (
            len(restored.structural_proposal_candidates) == candidate_count
            and restored.structural_evidence_ledger.pressure_snapshots
            == model.structural_evidence_ledger.pressure_snapshots
            and restored.structural_evidence_consumption_audit
            == model.structural_evidence_consumption_audit
        ),
        "tampered_pressure_snapshot_fails_closed": _expect_rejection(
            lambda: TSKV8Adapter.from_native_checkpoint(tampered)
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "projection_before": before.to_payload(),
        "projection_after": after.to_payload(),
        "schedule": scheduled.to_payload(),
        "repeated_schedule": repeated.to_payload(),
        "compaction": compaction.to_payload(),
        "checkpoint": {
            "candidate_count": len(restored.structural_proposal_candidates),
            "pressure_snapshot_count": len(restored.structural_evidence_ledger.pressure_snapshots),
        },
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "compacted history may restore the prior pressure identity as read-only provenance, "
                "but only unseen active windows can trigger a new schedule or candidate"
            ),
        },
        "boundary": (
            "This canary verifies provenance-aware projection and candidate deduplication; it does "
            "not claim open-domain quality, unlimited growth, CUDA, or CI completion."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s19_structural_provenance_projection_20260830.json",
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
