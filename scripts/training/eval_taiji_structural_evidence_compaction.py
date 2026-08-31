"""Run the R5C-S18 multi-round evidence compaction and consumption canary."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    StructuralEvidenceConsumptionAudit,
    StructuralEvidenceLedger,
    StructuralRuntimeObservation,
)
from taiji.structural_pressure import project_structural_growth_pressure  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s18-structural-evidence-compaction-v1"


def _observation(
    tick: int,
    evidence_id: str,
    *,
    network_id: str,
    region_id: str,
    task_slice_id: str = "task:train",
    partition: str = "train",
) -> StructuralRuntimeObservation:
    return StructuralRuntimeObservation(
        network_id=network_id,
        region_id=region_id,
        tick=tick,
        usage=0.4,
        resource_pressure=0.2,
        prediction_error=0.7,
        learning_gain=0.1,
        holdout_transfer=0.8,
        task_slice_id=task_slice_id,
        partition=partition,
        evidence_id=evidence_id,
    )


def _expect_rejection(factory: object) -> bool:
    try:
        factory()  # type: ignore[operator]
    except (KeyError, TypeError, ValueError):
        return True
    return False


def evaluate() -> dict[str, object]:
    ledger = StructuralEvidenceLedger(
        window_capacity=2,
        max_sealed_windows=8,
        max_evidence_index=8,
        max_compacted_windows=4,
    )
    for tick in range(1, 7):
        ledger.append(
            _observation(
                tick,
                f"demo:{tick}",
                network_id="network:demo",
                region_id="region:demo",
            )
        )
    for tick in range(1, 3):
        ledger.append(
            _observation(
                tick,
                f"other:{tick}",
                network_id="network:other",
                region_id="region:other",
            )
        )
    first, second, third, other = ledger.sealed_summaries
    evaluated = (first.window_digest, second.window_digest)
    before = ledger.audit_consumption(
        evaluated_window_digests=evaluated,
        scheduler_revision=2,
    )
    result = ledger.compact_consumed_windows(
        evaluated_window_digests=evaluated,
        scheduler_revision=2,
        keep_latest_per_stream=1,
    )
    compacted = ledger.compacted_windows[0]
    active_after_compaction = ledger.active_observed_count
    ledger.append(
        _observation(
            7,
            "demo:7",
            network_id="network:demo",
            region_id="region:demo",
        )
    )
    ledger.append(
        _observation(
            8,
            "demo:8",
            network_id="network:demo",
            region_id="region:demo",
        )
    )
    after = ledger.audit_consumption(
        evaluated_window_digests=evaluated,
        scheduler_revision=2,
    )
    checkpoint = ledger.to_payload()
    restored = StructuralEvidenceLedger.from_payload(checkpoint)
    tampered = copy.deepcopy(checkpoint)
    tampered["compacted_windows"][0]["provenance_digest"] = "1" * 64

    pressure_ledger = StructuralEvidenceLedger(window_capacity=1, max_compacted_windows=4)
    pressure_observations = (
        _observation(
            1,
            "pressure:train:one",
            network_id="network:pressure",
            region_id="region:pressure",
            task_slice_id="task:one",
        ),
        _observation(
            2,
            "pressure:train:two",
            network_id="network:pressure",
            region_id="region:pressure",
            task_slice_id="task:two",
        ),
        _observation(
            3,
            "pressure:holdout",
            network_id="network:pressure",
            region_id="region:pressure",
            task_slice_id="task:holdout",
            partition="holdout",
        ),
    )
    for observation in pressure_observations:
        pressure_ledger.append(observation)
    pressure_before = project_structural_growth_pressure(
        pressure_ledger.sealed_summaries,
        minimum_train_task_slices=2,
        minimum_train_windows=2,
    )
    pressure_ledger.compact_consumed_windows(
        evaluated_window_digests=tuple(
            item.window_digest for item in pressure_ledger.sealed_summaries[:2]
        ),
        keep_latest_per_stream=0,
    )
    pressure_after = project_structural_growth_pressure(
        pressure_ledger.sealed_summaries,
        minimum_train_task_slices=2,
        minimum_train_windows=2,
        historical_snapshots=pressure_ledger.pressure_snapshots,
    )

    metrics = {
        "consumption_audit_is_content_addressed": (
            StructuralEvidenceConsumptionAudit.from_payload(before.to_payload()) == before
        ),
        "compaction_only_moves_consumed_history": (
            result.status == "compacted"
            and result.compacted_window_digests == (first.window_digest,)
            and result.retained_window_digests == (
                second.window_digest,
                third.window_digest,
                other.window_digest,
            )
        ),
        "source_lineage_and_consumption_revision_survive": (
            compacted.window_id == first.window_id
            and compacted.network_id == first.network_id
            and compacted.region_id == first.region_id
            and compacted.task_slice_ids == first.task_slice_ids
            and compacted.partition_counts == first.partition_counts
            and compacted.evidence_ids == first.evidence_ids
            and compacted.window_digest == first.window_digest
            and compacted.consumed_scheduler_revision == 2
        ),
        "unconsumed_window_remains_active_after_compaction": (
            third.window_digest in after.unconsumed_window_digests
            and other.window_digest in after.unconsumed_window_digests
        ),
        "compaction_frees_active_evidence_capacity": (
            active_after_compaction == 6 and ledger.active_observed_count == 8
        ),
        "fresh_round_is_not_marked_consumed": (
            len(ledger.sealed_summaries) == 4
            and all(
                digest not in after.consumed_window_digests
                for digest in (
                    ledger.sealed_summaries[-1].window_digest,
                )
            )
        ),
        "pressure_identity_survives_compaction": pressure_after == pressure_before,
        "compacted_digest_is_not_a_new_pressure_trigger": (
            pressure_ledger.pressure_snapshots[0].window_digests[0]
            not in tuple(item.window_digest for item in pressure_ledger.sealed_summaries)
            and pressure_after.window_digests[0]
            == pressure_ledger.pressure_snapshots[0].window_digests[0]
        ),
        "duplicate_compacted_evidence_is_idempotent": (
            ledger.append(
                _observation(
                    1,
                    "demo:1",
                    network_id="network:demo",
                    region_id="region:demo",
                )
            ).status
            == "duplicate"
        ),
        "checkpoint_roundtrip_preserves_compacted_history": (
            restored.digest == ledger.digest
            and restored.compacted_windows == ledger.compacted_windows
            and restored.audit_consumption(
                evaluated_window_digests=evaluated,
                scheduler_revision=2,
            )
            == after
        ),
        "tampered_compacted_provenance_fails_closed": _expect_rejection(
            lambda: StructuralEvidenceLedger.from_payload(tampered)
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "rounds": {
            "evaluated_window_digests": list(evaluated),
            "before": before.to_payload(),
            "compaction": result.to_payload(),
            "after": after.to_payload(),
        },
        "pressure_projection": {
            "before": pressure_before.to_payload(),
            "after": pressure_after.to_payload(),
            "snapshot_digests": [item.snapshot_digest for item in pressure_ledger.pressure_snapshots],
        },
        "ledger": {
            "active_observed_count": ledger.active_observed_count,
            "observed_count": ledger.observed_count,
            "retained_window_digests": [item.window_digest for item in ledger.sealed_summaries],
            "compacted_window_digests": [item.window_digest for item in ledger.compacted_windows],
            "digest": ledger.digest,
        },
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "only scheduler-consumed older windows may be compacted; active projection keeps "
                "unconsumed windows, compacted provenance remains content-addressed, fresh evidence "
                "cannot be resurrected or double-consumed, and checkpoint restore is equivalent"
            ),
        },
        "boundary": (
            "This canary verifies bounded evidence history and cross-round consumption accounting; "
            "it does not claim open-domain quality, unlimited growth, CUDA, or CI completion."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s18_structural_evidence_compaction_20260830.json",
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
