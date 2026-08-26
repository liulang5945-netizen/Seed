from __future__ import annotations

import torch

from taiji import AdaptiveNeuronRegion, TaijiConfig, TSKV8Adapter


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


def _region() -> AdaptiveNeuronRegion:
    return AdaptiveNeuronRegion(
        region_id="adaptive.cortex",
        input_dim=5,
        unit_ids=("u0", "u1"),
        fan_in=2,
        generator=torch.Generator().manual_seed(7),
    )


def test_runtime_neuron_ledger_consumes_budget_and_rolls_back_native() -> None:
    model = TSKV8Adapter(_config(budget=1), episode_id="neuron-ledger")
    region = _region()
    model.attach_adaptive_neuron_region(region)
    proposal = model.propose_neuron_add(
        region_id=region.region_id,
        unit_id="u2",
        evidence_ids=("runtime:neuron-holdout",),
    )

    assert model.commit_neuron_add(proposal) is True
    assert model.neuron_regions[0].unit_ids == ("u0", "u1", "u2")
    assert model.topology_proposals[-1].status == "accepted"
    assert model.cognitive_snapshot().development.structural_budget == 0
    assert model.cognitive_snapshot().development.last_update_source == "neuron-topology-growth"

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.neuron_regions[0].unit_ids == ("u0", "u1", "u2")
    assert restored.topology_proposals[-1].status == "accepted"
    assert restored.rollback_neuron_add(proposal.proposal_id) is True
    assert restored.neuron_regions[0].unit_ids == ("u0", "u1")
    assert restored.cognitive_snapshot().development.structural_budget == 1
    assert restored.topology_proposals[-1].status == "rolled_back"


def test_runtime_neuron_ledger_rejects_growth_without_budget() -> None:
    model = TSKV8Adapter(_config(budget=0), episode_id="neuron-no-budget")
    region = _region()
    model.attach_adaptive_neuron_region(region)
    proposal = model.propose_neuron_add(
        region_id=region.region_id,
        unit_id="u2",
        evidence_ids=("runtime:neuron-holdout",),
    )

    assert model.commit_neuron_add(proposal) is False
    assert model.neuron_regions[0].unit_ids == ("u0", "u1")
    assert model.topology_proposals[-1].status == "rejected"
    assert model.cognitive_snapshot().development.structural_budget == 0
