from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from taiji import (
    AdaptiveNeuronRegion,
    AdaptiveStructuralGrowthController,
    StructuralEvidenceLedger,
    StructuralGrowthDynamics,
    StructuralGrowthEvidenceProjection,
    StructuralRuntimeObservation,
    TaijiConfig,
    TSKV8Adapter,
    project_structural_growth_pressure,
)


def _observation(
    tick: int,
    *,
    task_slice_id: str,
    partition: str,
    error: float = 0.7,
) -> StructuralRuntimeObservation:
    return StructuralRuntimeObservation(
        network_id="network:pressure",
        region_id="region:pressure",
        tick=tick,
        usage=0.5,
        resource_pressure=0.2,
        prediction_error=error,
        learning_gain=0.1,
        holdout_transfer=0.8 if partition == "holdout" else 0.0,
        evidence_id=f"pressure:{partition}:{task_slice_id}:{tick}",
        task_slice_id=task_slice_id,
        partition=partition,
    )


def test_pressure_projection_requires_cross_task_train_and_separate_holdout() -> None:
    ledger = StructuralEvidenceLedger(window_capacity=1)
    ledger.append(_observation(1, task_slice_id="task-a", partition="train"))
    ledger.append(_observation(2, task_slice_id="task-b", partition="train", error=0.6))
    ledger.append(_observation(3, task_slice_id="task-a", partition="holdout"))

    projection = project_structural_growth_pressure(ledger.sealed_summaries)

    assert isinstance(projection, StructuralGrowthEvidenceProjection)
    assert projection.train_task_slice_ids == ("task-a", "task-b")
    assert projection.train_window_count == 2
    assert projection.holdout_window_count == 1
    assert projection.mean_prediction_error == pytest.approx(0.65)
    assert projection.mean_resource_state == pytest.approx(0.8)
    assert projection.mean_holdout_transfer == pytest.approx(0.8)
    assert StructuralGrowthEvidenceProjection.from_payload(projection.to_payload()) == projection


def test_pressure_projection_rejects_single_slice_and_runtime_only_evidence() -> None:
    ledger = StructuralEvidenceLedger(window_capacity=1)
    ledger.append(_observation(1, task_slice_id="task-a", partition="train"))
    ledger.append(_observation(2, task_slice_id="task-a", partition="train"))
    ledger.append(_observation(3, task_slice_id="task-a", partition="runtime"))

    with pytest.raises(ValueError, match="independent train task slices"):
        project_structural_growth_pressure(ledger.sealed_summaries, require_holdout=False)


def test_pressure_projection_never_mutates_ledger() -> None:
    ledger = StructuralEvidenceLedger(window_capacity=1)
    ledger.append(_observation(1, task_slice_id="task-a", partition="train"))
    ledger.append(_observation(2, task_slice_id="task-b", partition="train"))
    ledger.append(_observation(3, task_slice_id="task-a", partition="holdout"))
    before = ledger.to_payload()

    project_structural_growth_pressure(ledger.sealed_summaries)

    assert ledger.to_payload() == before


def test_pressure_bridge_is_candidate_only_deduplicated_and_checkpointable() -> None:
    ledger = StructuralEvidenceLedger(window_capacity=1)
    for observation in (
        _observation(
            1,
            task_slice_id="task-a",
            partition="train",
        ),
        _observation(
            2,
            task_slice_id="task-b",
            partition="train",
        ),
        _observation(
            3,
            task_slice_id="task-a",
            partition="holdout",
        ),
    ):
        ledger.append(replace(observation, network_id="standalone:adaptive.cortex"))
    projection = project_structural_growth_pressure(ledger.sealed_summaries)

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
    model = TSKV8Adapter(config, episode_id="pressure-bridge")
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

    candidate = model.propose_structural_candidate_from_pressure(
        projection,
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": "u2"},
    )
    assert candidate is not None
    assert model.neuron_regions[0].unit_ids == ("u0", "u1")
    assert model.cognitive_snapshot().development.structural_budget == 1
    assert candidate.parent_checkpoint_id is not None
    assert model.propose_structural_candidate_from_pressure(
        projection,
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": "u2"},
    ) is None

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.structural_pressure_projection_digests == (projection.projection_digest,)
    restored_candidate = restored.structural_proposal_candidates[0]
    proposal = restored.materialize_structural_candidate(restored_candidate.candidate_id)
    assert proposal is not None
    assert proposal.parent_checkpoint_id == restored_candidate.parent_checkpoint_id
    assert restored.neuron_regions[0].unit_ids == ("u0", "u1")


def test_candidate_shadow_validation_is_checkpointed_without_admission() -> None:
    ledger = StructuralEvidenceLedger(window_capacity=1)
    for observation in (
        _observation(1, task_slice_id="task-a", partition="train"),
        _observation(2, task_slice_id="task-b", partition="train"),
        _observation(3, task_slice_id="task-a", partition="holdout"),
    ):
        ledger.append(replace(observation, network_id="standalone:adaptive.cortex"))
    projection = project_structural_growth_pressure(ledger.sealed_summaries)

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
    model = TSKV8Adapter(config, episode_id="candidate-shadow")
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
    candidate = model.propose_structural_candidate_from_pressure(
        projection,
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": "u2"},
    )
    assert candidate is not None
    proposal = model.materialize_structural_candidate(candidate.candidate_id)
    assert proposal is not None
    trial = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    trial.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(0))
    holdout_input = torch.zeros(5)
    holdout_input[trial.incoming.pre_index[-1]] = torch.sign(trial.incoming.edge_weight[-1])
    expected_activity = trial.step(holdout_input)
    before_units = region.unit_ids
    before_budget = model.cognitive_snapshot().development.structural_budget

    result = model.validate_structural_candidate_shadow(
        candidate.candidate_id,
        holdout_inputs=(holdout_input,),
        expected_activities=(expected_activity,),
    )

    assert result.status == "validated"
    assert result.topology_before_digest == result.topology_after_digest
    assert result.structural_budget_before == result.structural_budget_after == before_budget
    assert model.neuron_regions[0].unit_ids == before_units
    assert model.topology_proposals[-1].status == "pending"
    assert model.structural_candidate_validations == (result,)
    decision = model.evaluate_structural_candidate_gate(
        result,
        retention_regression=0.02,
        lesion_effect=0.15,
        resource_state=0.80,
        evidence_ids=("retention:shadow", "lesion:shadow"),
    )
    assert decision.passed is True
    assert model.topology_proposals[-1].status == "pending"
    assert model.structural_validation_gate_decisions == (decision,)

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.structural_candidate_validations == (result,)
    assert restored.structural_validation_gate_decisions == (decision,)
    assert restored.neuron_regions[0].unit_ids == before_units
    assert restored.topology_proposals[-1].status == "pending"


def test_candidate_shadow_validation_fails_closed_without_holdout_mutation() -> None:
    ledger = StructuralEvidenceLedger(window_capacity=1)
    for observation in (
        _observation(1, task_slice_id="task-a", partition="train"),
        _observation(2, task_slice_id="task-b", partition="train"),
        _observation(3, task_slice_id="task-a", partition="holdout"),
    ):
        ledger.append(replace(observation, network_id="standalone:adaptive.cortex"))
    projection = project_structural_growth_pressure(ledger.sealed_summaries)

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
    model = TSKV8Adapter(config, episode_id="candidate-shadow-missing-holdout")
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
    candidate = model.propose_structural_candidate_from_pressure(
        projection,
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": "u2"},
    )
    assert candidate is not None

    result = model.validate_structural_candidate_shadow(
        candidate.candidate_id,
        holdout_inputs=(),
        expected_activities=(),
    )

    assert result.status == "failed_closed"
    assert model.structural_proposal_candidates == (candidate,)
    assert model.cognitive_snapshot().development.structural_budget == 1
    assert model.neuron_regions[0].unit_ids == ("u0", "u1")
