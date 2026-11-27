"""Run the deterministic R5C-S3-A candidate shadow validation canary."""

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
    StructuralEvidenceLedger,
    StructuralGrowthDynamics,
    StructuralRuntimeObservation,
    TaijiConfig,
    TSKV8Adapter,
    project_structural_growth_pressure,
)

REPORT_FORMAT = "taiji-w7-r5c-s3a-structural-validation-v1"


def _observation(
    tick: int,
    *,
    task_slice_id: str,
    partition: str,
) -> StructuralRuntimeObservation:
    return StructuralRuntimeObservation(
        network_id="standalone:adaptive.cortex",
        region_id="region:validation-canary",
        tick=tick,
        usage=0.5,
        resource_pressure=0.2,
        prediction_error=0.7,
        learning_gain=0.1,
        holdout_transfer=0.8 if partition == "holdout" else 0.0,
        evidence_id=f"validation-canary:{partition}:{task_slice_id}:{tick}",
        task_slice_id=task_slice_id,
        partition=partition,
    )


def _projection():
    ledger = StructuralEvidenceLedger(window_capacity=1)
    for observation in (
        _observation(1, task_slice_id="task-a", partition="train"),
        _observation(2, task_slice_id="task-b", partition="train"),
        _observation(3, task_slice_id="task-a", partition="holdout"),
    ):
        ledger.append(observation)
    return project_structural_growth_pressure(ledger.sealed_summaries)


def _build_model(episode_id: str) -> tuple[TSKV8Adapter, AdaptiveNeuronRegion]:
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
        development_structural_budget=1,
        seed=71,
    )
    model = TSKV8Adapter(config, episode_id=episode_id)
    region = AdaptiveNeuronRegion(
        region_id="adaptive.cortex",
        input_dim=5,
        unit_ids=("u0", "u1"),
        fan_in=2,
        generator=torch.Generator().manual_seed(7),
    )
    model.attach_adaptive_neuron_region(region)
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
    return model, region


def _queue_candidate(model: TSKV8Adapter, projection) -> str:
    region = model.neuron_regions[0]
    candidate = model.propose_structural_candidate_from_pressure(
        projection,
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": "u2"},
    )
    if candidate is None:
        raise AssertionError("pressure bridge did not produce a candidate")
    return candidate.candidate_id


def _expected_activity(model: TSKV8Adapter, candidate_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    region = model.neuron_regions[0]
    proposal = model.materialize_structural_candidate(candidate_id)
    if proposal is None:
        raise AssertionError("candidate did not materialize")
    trial = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    trial.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(0))
    holdout_input = torch.zeros(5)
    holdout_input[trial.incoming.pre_index[-1]] = torch.sign(trial.incoming.edge_weight[-1])
    return holdout_input, trial.step(holdout_input)


def evaluate() -> dict[str, object]:
    projection = _projection()
    model, region = _build_model("structural-validation")
    candidate_id = _queue_candidate(model, projection)
    holdout_input, expected_activity = _expected_activity(model, candidate_id)
    before_units = region.unit_ids
    before_budget = model.cognitive_snapshot().development.structural_budget
    validation = model.validate_structural_candidate_shadow(
        candidate_id,
        holdout_inputs=(holdout_input,),
        expected_activities=(expected_activity,),
    )
    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())

    failed_model, failed_region = _build_model("structural-validation-failure")
    failed_candidate_id = _queue_candidate(failed_model, projection)
    failed_validation = failed_model.validate_structural_candidate_shadow(
        failed_candidate_id,
        holdout_inputs=(torch.zeros(5),),
        expected_activities=(torch.zeros(2),),
    )
    failed_restored = TSKV8Adapter.from_native_checkpoint(failed_model.native_checkpoint())
    failed_proposal = failed_restored.topology_proposals[-1]

    metrics = {
        "valid_candidate_shadowed": validation.status == "validated",
        "topology_digest_unchanged": (
            validation.topology_before_digest == validation.topology_after_digest
        ),
        "budget_unchanged": (
            validation.structural_budget_before
            == validation.structural_budget_after
            == before_budget
        ),
        "parent_structure_unchanged": region.unit_ids == before_units,
        "proposal_remains_pending": model.topology_proposals[-1].status == "pending",
        "validation_roundtrip": restored.structural_candidate_validations == (validation,),
        "rejected_malformed_holdout": failed_validation.status == "failed_closed",
        "rejected_candidate_not_pending_after_restore": (
            failed_proposal.status == "rejected"
            and failed_restored.structural_proposal_candidates == ()
        ),
        "rejection_keeps_parent_and_budget": (
            failed_region.unit_ids == ("u0", "u1")
            and failed_model.cognitive_snapshot().development.structural_budget == 1
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "projection_digest": projection.projection_digest,
        "validation": validation.to_payload(),
        "failed_validation": failed_validation.to_payload(),
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "candidate shadow validation must be checkpointable and topology-neutral; "
                "malformed holdout must fail closed without resurrecting a rejected candidate"
            ),
        },
        "boundary": (
            "This canary validates a candidate only. It does not admit topology, spend budget, "
            "or claim retention/lesion causality before the next validation slice."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s3a_structural_validation_20260830.json",
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
