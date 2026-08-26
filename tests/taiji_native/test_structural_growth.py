from __future__ import annotations

import torch

from taiji import (
    AdaptiveNeuronRegion,
    AdaptiveStructuralGrowthController,
    StructuralGrowthDynamics,
    TaijiConfig,
    TSKV8Adapter,
)


def _controller() -> AdaptiveStructuralGrowthController:
    return AdaptiveStructuralGrowthController(
        dynamics=StructuralGrowthDynamics(
            ema_rate=1.0,
            error_threshold=0.6,
            holdout_transfer_threshold=0.7,
            minimum_resource_state=0.5,
            required_error_steps=2,
        )
    )


def _region() -> AdaptiveNeuronRegion:
    return AdaptiveNeuronRegion(
        region_id="adaptive.cortex",
        input_dim=4,
        unit_ids=("u0", "u1"),
        fan_in=2,
        generator=torch.Generator().manual_seed(71),
    )


def _config(*, budget: int) -> TaijiConfig:
    return TaijiConfig(
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
        development_structural_budget=budget,
        seed=71,
    )


def test_structural_growth_requires_persistent_error_transfer_and_resources() -> None:
    controller = _controller()
    first = controller.observe(
        "adaptive.cortex",
        prediction_error=0.9,
        resource_state=0.8,
        holdout_transfer=0.9,
        evidence_ids=("tick:1",),
    )
    second = controller.observe(
        "adaptive.cortex",
        prediction_error=0.9,
        resource_state=0.8,
        holdout_transfer=0.9,
        evidence_ids=("tick:2",),
    )

    assert first.should_grow is False
    assert second.should_grow is True
    assert controller.next_unit_id("adaptive.cortex", ("u0", "u1")) == (
        "adaptive.cortex.grown.1"
    )
    restored = AdaptiveStructuralGrowthController.from_payload(controller.to_payload())
    assert restored.total_observations == 2
    assert restored.regions[0].proposal_count == 1


def test_runtime_structural_growth_proposal_enters_ledger_and_rolls_back() -> None:
    model = TSKV8Adapter(_config(budget=1), episode_id="structural-growth")
    model.attach_adaptive_neuron_region(_region())
    model.attach_structural_growth_controller(_controller())

    assert model.propose_neuron_growth_from_error(
        region_id="adaptive.cortex",
        prediction_error=0.9,
        resource_state=0.8,
        holdout_transfer=0.9,
        evidence_ids=("tick:1",),
    ) is None
    proposal = model.propose_neuron_growth_from_error(
        region_id="adaptive.cortex",
        prediction_error=0.9,
        resource_state=0.8,
        holdout_transfer=0.9,
        evidence_ids=("tick:2",),
    )
    assert proposal is not None
    assert dict(proposal.specification)["unit_id"] == "adaptive.cortex.grown.1"
    assert model.commit_neuron_add(proposal) is True
    assert model.neuron_regions[0].unit_ids[-1] == "adaptive.cortex.grown.1"

    model.neuron_regions[0].lesion_topology_proposal(proposal)
    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.structural_growth_controller is not None
    assert restored.structural_growth_controller.total_observations == 2
    assert restored.neuron_regions[0].lesioned_unit_ids == ("adaptive.cortex.grown.1",)
    assert restored.rollback_neuron_add(proposal.proposal_id) is True
    assert restored.neuron_regions[0].unit_ids == ("u0", "u1")
    assert restored.cognitive_snapshot().development.structural_budget == 1
