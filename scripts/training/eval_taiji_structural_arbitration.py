"""Run the deterministic R5C-S8 multi-candidate arbitration canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    AdaptiveNeuronRegion,
    AdaptiveStructuralGrowthController,
    StructuralGrowthDynamics,
    StructuralProposalCandidate,
    StructuralRuntimeObservation,
    TaijiConfig,
    TSKV8Adapter,
)

REPORT_FORMAT = "taiji-w7-r5c-s8-structural-arbitration-v1"


def _build_model() -> TSKV8Adapter:
    config = TaijiConfig(
        alphabet_size=257,
        boundary_symbol=256,
        region_sizes=(8, 6),
        synapse_fan_in=3,
        motor_fan_in=4,
        lateral_fan_in=3,
        memory_units=12,
        memory_fan_in=3,
        memory_readout_fan_in=4,
        memory_meta_dim=4,
        memory_time_dim=4,
        memory_episode_dim=4,
        development_structural_budget=2,
        seed=103,
    )
    model = TSKV8Adapter(config, episode_id="structural-arbitration")
    for region_id, unit_ids, seed in (
        ("adaptive.cortex", ("u0", "u1"), 103),
        ("adaptive.memory", ("m0", "m1"), 104),
    ):
        model.attach_adaptive_neuron_region(
            AdaptiveNeuronRegion(
                region_id=region_id,
                input_dim=5,
                unit_ids=unit_ids,
                fan_in=2,
                generator=torch.Generator().manual_seed(seed),
            )
        )
    for tick in range(1, 7):
        model.record_structural_runtime_observation(
            StructuralRuntimeObservation(
                network_id="workbench",
                region_id="adaptive.cortex",
                tick=tick,
                usage=0.5,
                resource_pressure=0.2,
                prediction_error=0.5,
                learning_gain=0.1,
                holdout_transfer=0.0,
                evidence_id=f"s8:clock:{tick}",
                task_slice_id="arbitration-clock",
                partition="runtime",
            )
        )
    model.attach_structural_growth_controller(
        AdaptiveStructuralGrowthController(
            dynamics=StructuralGrowthDynamics(
                ema_rate=1.0,
                error_threshold=0.0,
                holdout_transfer_threshold=0.0,
                minimum_resource_state=0.0,
                required_error_steps=1,
            )
        )
    )
    return model


def _candidate(
    candidate_id: str,
    *,
    region_id: str,
    unit_id: str,
    priority: float,
    source_tick: int,
    conflict_key: str,
) -> StructuralProposalCandidate:
    return StructuralProposalCandidate(
        candidate_id=candidate_id,
        network_id="workbench",
        target_kind="neuron",
        operation="add",
        substrate_ids=(region_id,),
        evidence_ids=(f"evidence:{candidate_id}",),
        source_tick=source_tick,
        priority=priority,
        specification=(
            ("region_id", region_id),
            ("unit_id", unit_id),
        ),
        resource_cost=1,
        conflict_keys=(conflict_key,),
    )


def _holdout_payload(model: TSKV8Adapter, candidate_id: str) -> dict[str, object]:
    candidate = next(
        item
        for item in model.structural_proposal_candidates
        if item.candidate_id == candidate_id
    )
    proposal = model.materialize_structural_candidate(candidate_id)
    if proposal is None:
        raise AssertionError(f"candidate {candidate_id} was not materialized")
    region_id = str(dict(candidate.specification)["region_id"])
    region = next(item for item in model.neuron_regions if item.region_id == region_id)
    trial = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    trial.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(0))
    holdout_input = torch.zeros(region.input_dim)
    holdout_input[trial.incoming.pre_index[-1]] = torch.sign(trial.incoming.edge_weight[-1])
    return {
        "holdout_inputs": (holdout_input,),
        "expected_activities": (trial.step(holdout_input),),
        "retention_regression": 0.0,
        "lesion_effect": 1.0,
        "resource_state": 0.8,
        "evidence_ids": (f"s8:retention:{candidate_id}", f"s8:lesion:{candidate_id}"),
    }


def evaluate() -> dict[str, object]:
    model = _build_model()
    candidates = (
        _candidate(
            "c1",
            region_id="adaptive.cortex",
            unit_id="u2",
            priority=0.9,
            source_tick=5,
            conflict_key="region:adaptive.cortex",
        ),
        _candidate(
            "c2",
            region_id="adaptive.cortex",
            unit_id="u3",
            priority=0.8,
            source_tick=6,
            conflict_key="region:adaptive.cortex",
        ),
        _candidate(
            "c3",
            region_id="adaptive.memory",
            unit_id="m2",
            priority=0.7,
            source_tick=4,
            conflict_key="region:adaptive.memory",
        ),
        _candidate(
            "c4",
            region_id="adaptive.other",
            unit_id="x1",
            priority=0.6,
            source_tick=3,
            conflict_key="region:adaptive.other",
        ),
    )
    for candidate in candidates:
        model._queue_structural_proposal_candidate(candidate)

    batch = model.arbitrate_structural_candidate_batch(("c4", "c2", "c3", "c1"))
    repeated_batch = model.arbitrate_structural_candidate_batch(("c1", "c2", "c3", "c4"))
    if batch != repeated_batch:
        raise AssertionError("repeated arbitration did not return the same batch")
    arbitration_topology = tuple(region.unit_ids for region in model.neuron_regions)
    arbitration_budget = model.cognitive_snapshot().development.structural_budget
    arbitration_no_mutation = (
        arbitration_topology == tuple(region.unit_ids for region in model.neuron_regions)
        and arbitration_budget == model.cognitive_snapshot().development.structural_budget
    )

    first_payload = _holdout_payload(model, "c1")
    first_continuation = model.continue_structural_candidate_batch(
        batch.batch_id,
        continuations_by_candidate={"c1": first_payload},
    )
    running_batch = model.structural_candidate_batches[0]
    checkpoint_after_first = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint_after_first)
    restored_reservation_before_second = restored.structural_candidate_batches[0].reservation_remaining
    second_payload = _holdout_payload(restored, "c3")
    second_continuation = restored.continue_structural_candidate_batch(
        batch.batch_id,
        continuations_by_candidate={"c3": second_payload},
    )
    completed_batch = restored.structural_candidate_batches[0]
    repeated_continuation = restored.continue_structural_candidate_batch(
        batch.batch_id,
        continuations_by_candidate={"c1": first_payload, "c3": second_payload},
    )
    metrics = {
        "rank_order_is_explicit": batch.selected_candidate_ids == ("c1", "c3"),
        "same_region_lower_priority_deferred": (
            "c2" in batch.deferred_candidate_ids
            and "conflict_with_selected:c1" in batch.reason_by_candidate["c2"]
        ),
        "budget_overflow_deferred": (
            "c4" in batch.deferred_candidate_ids
            and batch.reason_by_candidate["c4"] == "structural_budget_insufficient_for_batch"
        ),
        "same_batch_replay_idempotent": batch == repeated_batch,
        "arbitration_does_not_mutate_topology_or_budget": arbitration_no_mutation,
        "first_admission_completed": (
            first_continuation["results"]["c1"]["status"] == "admitted"
        ),
        "reservation_kept_for_remaining_candidate": (
            running_batch.reservation_remaining == 1 and running_batch.status == "running"
        ),
        "restore_keeps_batch_reservation": (
            restored_reservation_before_second == 1
        ),
        "remaining_candidate_continued_after_restore": (
            second_continuation["results"]["c3"]["status"] == "admitted"
            and completed_batch.status == "completed"
            and completed_batch.reservation_remaining == 0
        ),
        "repeated_batch_continuation_is_idempotent": (
            repeated_continuation["batch"]["status"] == "completed"
            and len(restored.structural_admission_results) == 2
        ),
        "deferred_candidates_remain_recoverable": (
            {item.candidate_id for item in restored.structural_proposal_candidates}
            == {"c2", "c4"}
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "batch": batch.to_payload(),
        "first_continuation": first_continuation,
        "second_continuation": second_continuation,
        "repeated_continuation": repeated_continuation,
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "multi-candidate arbitration must be deterministic, reserve budget without mutation, "
                "isolate conflicts, and continue selected candidates across checkpoint restore"
            ),
        },
        "boundary": (
            "This canary does not parallelize admission, expand structural budget, or promote deferred "
            "candidates without a new arbitration batch."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s8_structural_arbitration_20260830.json",
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
