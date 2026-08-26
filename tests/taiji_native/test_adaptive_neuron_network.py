from __future__ import annotations

import torch

from taiji import (
    AdaptiveNeuronNetwork,
    AdaptiveNeuronRegion,
    NeuronRegionDynamics,
)


def _region(region_id: str) -> AdaptiveNeuronRegion:
    return AdaptiveNeuronRegion(
        region_id=region_id,
        input_dim=3,
        unit_ids=(f"{region_id}.u0", f"{region_id}.u1"),
        fan_in=2,
        dynamics=NeuronRegionDynamics(membrane_decay=0.0, recurrent_gain=0.0),
        generator=torch.Generator().manual_seed(len(region_id)),
    )


def _network() -> AdaptiveNeuronNetwork:
    return AdaptiveNeuronNetwork(
        (_region("source"), _region("target")),
        execution_order=("source", "target"),
    )


def _connection_proposal(network: AdaptiveNeuronNetwork):
    return network.propose_connection_add(
        source_region_id="source",
        target_region_id="target",
        evidence_ids=("holdout:cross-region",),
        fan_in=1,
        parent_checkpoint_id="parent:network",
    )


def test_cross_region_projection_drives_target_and_lesion_is_causal() -> None:
    network = _network()
    proposal = _connection_proposal(network)
    assert network.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(99),
    ) is True
    connection = network.connections[0][3]
    connection.edge_weight.fill_(1.0)
    source = network.regions[0]
    source.incoming.edge_weight.fill_(1.0)

    active = network.step({"source": torch.ones(3)})
    driven = float(active["target"].sum().item())
    assert driven > 0.0

    assert network.lesion_topology_proposal(proposal) is True
    silent = network.step({"source": torch.ones(3)})
    assert float(silent["target"].sum().item()) == 0.0


def test_region_growth_migrates_adjacent_connection_without_redrawing_old_support() -> None:
    network = _network()
    connection_proposal = _connection_proposal(network)
    network.apply_topology_proposal(
        connection_proposal,
        generator=torch.Generator().manual_seed(99),
    )
    connection = network.connections[0][3]
    old_index = connection.pre_index.clone()
    old_weight = connection.edge_weight.clone()
    source = network.regions[0]
    neuron_proposal = source.propose_unit_add(
        unit_id="source.u2",
        evidence_ids=("holdout:source-growth",),
    )
    network.apply_neuron_proposal(
        neuron_proposal,
        generator=torch.Generator().manual_seed(100),
    )

    migrated = network.connections[0][3]
    assert migrated.in_features == 3
    assert torch.equal(migrated.pre_index, old_index)
    assert torch.equal(migrated.edge_weight, old_weight)
    assert network.regions[0].unit_ids == ("source.u0", "source.u1", "source.u2")


def test_cross_region_network_checkpoint_roundtrip_preserves_contract() -> None:
    network = _network()
    proposal = _connection_proposal(network)
    network.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(99),
    )
    restored = AdaptiveNeuronNetwork.from_payload(
        network.to_payload(),
        generator=torch.Generator().manual_seed(123),
    )

    assert restored.region_ids == network.region_ids
    assert restored.execution_order == network.execution_order
    assert restored.connection_ids == network.connection_ids
    assert torch.equal(
        restored.connections[0][3].pre_index,
        network.connections[0][3].pre_index,
    )
