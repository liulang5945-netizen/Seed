"""Run the deterministic R5C-S5 bounded multi-step growth canary."""

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

REPORT_FORMAT = "taiji-w7-r5c-s5-structural-continuation-v1"


def _projection(offset: int):
    ledger = StructuralEvidenceLedger(window_capacity=1)
    observations = (
        StructuralRuntimeObservation(
            network_id="standalone:adaptive.cortex",
            region_id="region:continuation",
            tick=offset + 1,
            usage=0.5,
            resource_pressure=0.2,
            prediction_error=0.7,
            learning_gain=0.1,
            holdout_transfer=0.0,
            evidence_id=f"continuation:train:a:{offset + 1}",
            task_slice_id="task-a",
            partition="train",
        ),
        StructuralRuntimeObservation(
            network_id="standalone:adaptive.cortex",
            region_id="region:continuation",
            tick=offset + 2,
            usage=0.5,
            resource_pressure=0.2,
            prediction_error=0.7,
            learning_gain=0.1,
            holdout_transfer=0.0,
            evidence_id=f"continuation:train:b:{offset + 2}",
            task_slice_id="task-b",
            partition="train",
        ),
        StructuralRuntimeObservation(
            network_id="standalone:adaptive.cortex",
            region_id="region:continuation",
            tick=offset + 3,
            usage=0.5,
            resource_pressure=0.2,
            prediction_error=0.7,
            learning_gain=0.1,
            holdout_transfer=0.8,
            evidence_id=f"continuation:holdout:a:{offset + 3}",
            task_slice_id="task-a",
            partition="holdout",
        ),
    )
    for observation in observations:
        ledger.append(observation)
    return project_structural_growth_pressure(ledger.sealed_summaries)


def _build_model() -> tuple[TSKV8Adapter, AdaptiveNeuronRegion]:
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
        seed=97,
    )
    model = TSKV8Adapter(config, episode_id="structural-continuation")
    region = AdaptiveNeuronRegion(
        region_id="adaptive.cortex",
        input_dim=5,
        unit_ids=("u0", "u1"),
        fan_in=2,
        generator=torch.Generator().manual_seed(97),
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


def _admit_one(
    model: TSKV8Adapter,
    projection,
    unit_id: str,
) -> tuple[object, object, object]:
    region = model.neuron_regions[0]
    candidate = model.propose_structural_candidate_from_pressure(
        projection,
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": unit_id},
    )
    if candidate is None:
        raise AssertionError(f"candidate {unit_id} was not produced")
    proposal = model.materialize_structural_candidate(candidate.candidate_id)
    if proposal is None:
        raise AssertionError(f"proposal {unit_id} was not materialized")
    trial = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    trial.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(0))
    holdout_input = torch.zeros(5)
    holdout_input[trial.incoming.pre_index[-1]] = torch.sign(trial.incoming.edge_weight[-1])
    validation = model.validate_structural_candidate_shadow(
        candidate.candidate_id,
        holdout_inputs=(holdout_input,),
        expected_activities=(trial.step(holdout_input),),
    )
    budget = model.cognitive_snapshot().development.structural_budget
    decision = model.evaluate_structural_candidate_gate(
        validation,
        retention_regression=0.0,
        lesion_effect=0.15,
        resource_state=min(1.0, budget / 2.0),
        evidence_ids=(f"continuation:retention:{unit_id}", f"continuation:lesion:{unit_id}"),
    )
    admission = None if not decision.passed else model.admit_structural_candidate(validation, decision)
    return validation, decision, admission


def evaluate() -> dict[str, object]:
    model, region = _build_model()
    first = _admit_one(model, _projection(0), "u2")
    checkpoint_after_first = model.native_checkpoint()
    continued = TSKV8Adapter.from_native_checkpoint(checkpoint_after_first)
    continued_units_before_second = continued.neuron_regions[0].unit_ids
    second = _admit_one(continued, _projection(3), "u3")
    checkpoint_after_second = continued.native_checkpoint()
    exhausted = TSKV8Adapter.from_native_checkpoint(checkpoint_after_second)
    third = _admit_one(exhausted, _projection(6), "u4")
    restored_exhausted = TSKV8Adapter.from_native_checkpoint(exhausted.native_checkpoint())

    first_validation, first_decision, first_admission = first
    second_validation, second_decision, second_admission = second
    third_validation, third_decision, third_admission = third
    metrics = {
        "first_step_admitted": first_admission is not None and first_admission.status == "admitted",
        "checkpoint_continuation_preserves_first_growth": continued_units_before_second == ("u0", "u1", "u2"),
        "second_step_admitted": second_admission is not None and second_admission.status == "admitted",
        "two_step_topology": continued.neuron_regions[0].unit_ids == ("u0", "u1", "u2", "u3"),
        "two_step_budget_exact": continued.cognitive_snapshot().development.structural_budget == 0,
        "first_second_lineage_persisted": (
            len(continued.structural_admission_results) == 2
            and continued.structural_admission_results[0] == first_admission
            and continued.structural_admission_results[1] == second_admission
        ),
        "third_policy_rejected_at_budget_zero": third_decision.passed is False,
        "third_not_admitted": third_admission is None,
        "third_rejection_is_resource_bound": "structural_budget_insufficient" in third_decision.reasons,
        "exhausted_topology_unchanged": exhausted.neuron_regions[0].unit_ids == ("u0", "u1", "u2", "u3"),
        "exhausted_budget_unchanged": exhausted.cognitive_snapshot().development.structural_budget == 0,
        "third_rejected_not_pending_after_restore": (
            restored_exhausted.topology_proposals[-1].status == "rejected"
            and restored_exhausted.structural_proposal_candidates == ()
        ),
        "validation_chain_present": (
            first_validation.status == "validated"
            and second_validation.status == "validated"
            and third_validation.status == "validated"
        ),
        "policy_chain_is_bound": first_decision.candidate_id != second_decision.candidate_id,
        "initial_model_region_bootstrap": region.unit_ids[:2] == ("u0", "u1"),
    }
    return {
        "format": REPORT_FORMAT,
        "first": {
            "validation": first_validation.to_payload(),
            "decision": first_decision.to_payload(),
            "admission": None if first_admission is None else first_admission.to_payload(),
        },
        "second": {
            "validation": second_validation.to_payload(),
            "decision": second_decision.to_payload(),
            "admission": None if second_admission is None else second_admission.to_payload(),
        },
        "third": {
            "validation": third_validation.to_payload(),
            "decision": third_decision.to_payload(),
            "admission": None if third_admission is None else third_admission.to_payload(),
        },
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "bounded structural growth may continue across checkpoints only while budget and "
                "validation gates pass; exhaustion must reject the next candidate without mutation"
            ),
        },
        "boundary": (
            "This canary allows two sequential bounded admissions with budget two. It does not "
            "enable unbounded loops, automatic budget expansion, or full retraining."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s5_structural_continuation_20260830.json",
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
