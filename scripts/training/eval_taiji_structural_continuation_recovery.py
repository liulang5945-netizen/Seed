"""Run the deterministic R5C-S9 multi-round continuation and rollback canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_structural_arbitration import (  # noqa: E402
    _build_model,
    _candidate,
    _holdout_payload,
)
from taiji import StructuralProposalCandidate, TSKV8Adapter  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s9-structural-continuation-recovery-v1"


def evaluate() -> dict[str, object]:
    model = _build_model()
    c1 = _candidate(
        "c1",
        region_id="adaptive.cortex",
        unit_id="u2",
        priority=0.9,
        source_tick=5,
        conflict_key="region:adaptive.cortex",
    )
    c2 = _candidate(
        "c2",
        region_id="adaptive.cortex",
        unit_id="u3",
        priority=0.8,
        source_tick=6,
        conflict_key="region:adaptive.cortex",
    )
    c3 = _candidate(
        "c3",
        region_id="adaptive.memory",
        unit_id="m2",
        priority=0.7,
        source_tick=4,
        conflict_key="region:adaptive.memory",
    )
    for candidate in (c1, c2, c3):
        model._queue_structural_proposal_candidate(candidate)

    cortex_pressure_before = model.measure_structural_capacity_pressure(
        "adaptive.cortex",
        capacity_limit=4,
    )
    memory_pressure_before = model.measure_structural_capacity_pressure(
        "adaptive.memory",
        capacity_limit=4,
    )
    initial_topology = tuple(region.unit_ids for region in model.neuron_regions)
    initial_budget = model.cognitive_snapshot().development.structural_budget
    first_batch = model.arbitrate_structural_candidate_batch(("c1", "c2", "c3"))
    first_payload = _holdout_payload(model, "c1")
    first_round = model.continue_structural_candidate_batch(
        first_batch.batch_id,
        continuations_by_candidate={"c1": first_payload},
    )
    checkpoint_after_first = model.native_checkpoint()
    continued = TSKV8Adapter.from_native_checkpoint(checkpoint_after_first)
    memory_pressure_during_continuation = continued.measure_structural_capacity_pressure(
        "adaptive.memory",
        capacity_limit=4,
    )
    second_payload = _holdout_payload(continued, "c3")
    second_round = continued.continue_structural_candidate_batch(
        first_batch.batch_id,
        continuations_by_candidate={"c3": second_payload},
    )
    completed_checkpoint = continued.native_checkpoint()
    recovered = TSKV8Adapter.from_native_checkpoint(completed_checkpoint)
    memory_pressure_after_admission = recovered.measure_structural_capacity_pressure(
        "adaptive.memory",
        capacity_limit=4,
    )
    rollback = recovered.rollback_structural_candidate_batch(first_batch.batch_id, "c3")
    rollback_checkpoint = recovered.native_checkpoint()
    rollback_restored = TSKV8Adapter.from_native_checkpoint(rollback_checkpoint)
    memory_pressure_after_rollback = rollback_restored.measure_structural_capacity_pressure(
        "adaptive.memory",
        capacity_limit=4,
    )
    rollback_memory_units = rollback_restored.neuron_regions[1].unit_ids
    rollback_budget = rollback_restored.cognitive_snapshot().development.structural_budget

    fresh_candidate = StructuralProposalCandidate(
        candidate_id="c2:fresh-evidence",
        network_id="workbench",
        target_kind="neuron",
        operation="add",
        substrate_ids=("adaptive.cortex",),
        evidence_ids=("evidence:c2:fresh",),
        source_tick=10,
        priority=0.95,
        specification=(
            ("region_id", "adaptive.cortex"),
            ("unit_id", "u3"),
        ),
        resource_cost=1,
        conflict_keys=("region:adaptive.cortex",),
    )
    rollback_restored._queue_structural_proposal_candidate(fresh_candidate)
    second_batch = rollback_restored.arbitrate_structural_candidate_batch(
        ("c2", "c2:fresh-evidence")
    )
    fresh_payload = _holdout_payload(rollback_restored, "c2:fresh-evidence")
    fresh_round = rollback_restored.continue_structural_candidate_batch(
        second_batch.batch_id,
        continuations_by_candidate={"c2:fresh-evidence": fresh_payload},
    )
    final_restored = TSKV8Adapter.from_native_checkpoint(rollback_restored.native_checkpoint())
    repeated_rollback = final_restored.rollback_structural_candidate_batch(
        first_batch.batch_id,
        "c3",
    )
    metrics = {
        "capacity_pressure_is_cross_region_and_read_only": (
            cortex_pressure_before.region_id != memory_pressure_before.region_id
            and cortex_pressure_before.pressure_digest != memory_pressure_before.pressure_digest
            and initial_topology == (("u0", "u1"), ("m0", "m1"))
            and initial_budget == 2
        ),
        "first_round_keeps_unprocessed_reservation": (
            first_round["batch"]["reservation_remaining"] == 1
            and first_round["batch"]["status"] == "running"
        ),
        "cross_checkpoint_second_round_admitted": (
            second_round["results"]["c3"]["status"] == "admitted"
            and len(continued.structural_admission_results) == 2
        ),
        "capacity_pressure_tracks_admission": (
            memory_pressure_during_continuation.unit_count == 2
            and memory_pressure_during_continuation.reserved_resource_cost == 1
            and memory_pressure_after_admission.unit_count == 3
            and memory_pressure_after_admission.reserved_resource_cost == 0
        ),
        "rollback_reopens_budget_and_reverts_region": (
            rollback["status"] == "rolled_back"
            and rollback_memory_units == ("m0", "m1")
            and rollback_budget == 1
        ),
        "rollback_is_checkpointed": (
            len(rollback_restored.structural_candidate_rollbacks) == 1
            and rollback_restored.structural_candidate_rollbacks[0].status == "rolled_back"
            and memory_pressure_after_rollback.unit_count == 2
        ),
        "fresh_evidence_rearbitrates_deferred_candidate": (
            second_batch.selected_candidate_ids == ("c2:fresh-evidence",)
            and second_batch.deferred_candidate_ids == ("c2",)
            and second_batch.reason_by_candidate["c2"] == "conflict_with_selected:c2:fresh-evidence"
        ),
        "fresh_candidate_admitted_after_rollback": (
            fresh_round["results"]["c2:fresh-evidence"]["status"] == "admitted"
            and rollback_restored.neuron_regions[0].unit_ids == ("u0", "u1", "u2", "u3")
            and rollback_restored.cognitive_snapshot().development.structural_budget == 0
        ),
        "repeated_rollback_is_idempotent": (
            repeated_rollback == rollback
            and len(final_restored.structural_candidate_rollbacks) == 1
        ),
        "old_deferred_candidate_remains_auditable": (
            any(
                item.candidate_id == "c2"
                for item in final_restored.structural_proposal_candidates
            )
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "first_batch": first_batch.to_payload(),
        "first_round": first_round,
        "second_round": second_round,
        "rollback": rollback,
        "second_batch": second_batch.to_payload(),
        "fresh_round": fresh_round,
        "capacity": {
            "cortex_before": cortex_pressure_before.to_payload(),
            "memory_before": memory_pressure_before.to_payload(),
            "memory_during_continuation": memory_pressure_during_continuation.to_payload(),
            "memory_after_admission": memory_pressure_after_admission.to_payload(),
            "memory_after_rollback": memory_pressure_after_rollback.to_payload(),
        },
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "multi-round growth must preserve batch reservations across checkpoints, measure "
                "cross-region capacity pressure, support fresh-evidence re-arbitration, and reverse "
                "an admitted candidate with an auditable checkpointed rollback"
            ),
        },
        "boundary": (
            "This canary does not expand structural budget, parallelize admission, or revive the "
            "same deferred evidence without a fresh candidate identity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s9_structural_continuation_recovery_20260830.json",
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
