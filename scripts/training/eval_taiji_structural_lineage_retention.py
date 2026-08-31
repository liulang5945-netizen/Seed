"""Run the R5C-S22 protected structural-lineage retention canary."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import (  # noqa: E402
    _record_real_evidence,
)
from taiji import StructuralCandidateBatch, TSKV8Adapter  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s22-structural-lineage-retention-v1"
RETENTION_LIMIT = 1


def _terminal_batch(batch: StructuralCandidateBatch, batch_id: str) -> StructuralCandidateBatch:
    return replace(
        batch,
        batch_id=batch_id,
        selected_candidate_ids=(),
        deferred_candidate_ids=(),
        rejected_candidate_ids=batch.candidate_ids,
        candidate_states=tuple((candidate_id, "rejected") for candidate_id in batch.candidate_ids),
        reserved_resource_cost=0,
        reservation_remaining=0,
        arbitration_digest=f"terminal:{batch_id}",
        revision=batch.revision + 1,
        status="completed",
    )


def _batch(model: TSKV8Adapter, batch_id: str) -> StructuralCandidateBatch:
    return next(item for item in model.structural_candidate_batches if item.batch_id == batch_id)


def evaluate() -> dict[str, object]:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    model = runtime.model.architecture
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"retention canary batch was not created: {schedule}")
    batch_id = str(schedule["batch_id"])
    active = _batch(model, batch_id)
    pending_ids = set(active.candidate_ids)
    model.config = replace(model.config, cognitive_lineage_history_limit=RETENTION_LIMIT)

    model._record_structural_candidate_batch(_terminal_batch(active, "batch:terminal"))
    after_terminal = {item.batch_id: item for item in model.structural_candidate_batches}
    retained_active = after_terminal.get(batch_id)
    evicted_terminal = "batch:terminal" not in after_terminal

    candidate = next(
        item for item in model.structural_proposal_candidates if item.candidate_id == active.candidate_ids[0]
    )
    extra = replace(
        candidate,
        candidate_id="candidate:extra-retention",
        specification=(
            ("region_id", dict(candidate.specification)["region_id"]),
            ("unit_id", "retention-extra"),
        ),
    )
    model._queue_structural_proposal_candidate(extra)
    retained_pending_ids = {item.candidate_id for item in model.structural_proposal_candidates}

    admitted = replace(
        retained_active,
        candidate_states=tuple((candidate_id, "admitted") for candidate_id in retained_active.candidate_ids),
        selected_candidate_ids=retained_active.candidate_ids,
        reserved_resource_cost=0,
        reservation_remaining=0,
        status="completed",
    )
    model._record_structural_candidate_batch(admitted)
    model._record_structural_candidate_batch(_terminal_batch(admitted, "batch:terminal-admitted"))
    after_admitted = {item.batch_id: item for item in model.structural_candidate_batches}
    admitted_retained = after_admitted.get(batch_id)
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_batches = {item.batch_id: item for item in restored.structural_candidate_batches}

    metrics = {
        "active_batch_survives_terminal_eviction": (
            retained_active is not None
            and retained_active.active_reservation
            and evicted_terminal
        ),
        "pending_candidates_survive_batch_retention": (
            pending_ids.issubset(retained_pending_ids)
            and "candidate:extra-retention" not in retained_pending_ids
        ),
        "rollbackable_admitted_batch_survives_terminal_eviction": (
            admitted_retained is not None
            and all(state == "admitted" for state in admitted_retained.state_by_candidate.values())
            and "batch:terminal-admitted" not in after_admitted
        ),
        "checkpoint_preserves_protected_lineage": (
            batch_id in restored_batches
            and restored_batches[batch_id].to_payload() == admitted_retained.to_payload()
        ),
        "no_unsafe_eviction_under_pressure": (
            len(model.structural_candidate_batches) >= 1
            and len(model.structural_proposal_candidates) >= len(pending_ids)
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "retention_limit": RETENTION_LIMIT,
        "schedule": schedule,
        "batch_ids": [item.batch_id for item in model.structural_candidate_batches],
        "pending_candidate_ids": [item.candidate_id for item in model.structural_proposal_candidates],
        "restored_batch_ids": [item.batch_id for item in restored.structural_candidate_batches],
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "bounded retention may evict only terminal structural records; active reservations, "
                "pending/deferred candidates, and rollbackable admissions must survive and checkpoint"
            ),
        },
        "boundary": (
            "This canary covers protected candidate/batch retention under a small configured limit; "
            "S20/S21 cover stale artifact isolation, evidence compaction, and long-sequence ledger stress. "
            "It does not claim open-domain quality, CUDA, CI, or physical deletion."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s22_structural_lineage_retention_20260831.json",
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
