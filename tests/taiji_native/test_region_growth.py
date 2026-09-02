from __future__ import annotations

import pytest
import torch

from taiji import (
    AdaptiveNeuronNetwork,
    AdaptiveNeuronRegion,
    AdaptiveStructuralGrowthController,
    AdaptiveStructuralPruningController,
    CrossRegionCooperationLearner,
    CrossRegionLearningDynamics,
    StructuralGrowthDynamics,
    StructuralPruningDynamics,
    TaijiConfig,
    TSKV8Adapter,
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


def _network() -> AdaptiveNeuronNetwork:
    def make(region_id: str) -> AdaptiveNeuronRegion:
        return AdaptiveNeuronRegion(
            region_id=region_id,
            input_dim=3,
            unit_ids=(f"{region_id}.u0", f"{region_id}.u1"),
            fan_in=2,
            generator=torch.Generator().manual_seed(len(region_id)),
        )

    return AdaptiveNeuronNetwork(
        (make("source"), make("bottleneck")),
        execution_order=("source", "bottleneck"),
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


def _pruning_controller() -> AdaptiveStructuralPruningController:
    return AdaptiveStructuralPruningController(
        dynamics=StructuralPruningDynamics(
            ema_rate=1.0,
            maximum_usage=0.2,
            minimum_resource_pressure=0.7,
            maximum_learning_gain=0.1,
            required_underuse_steps=2,
            maximum_holdout_regression=0.05,
        )
    )


def test_region_growth_is_explicit_connected_checkpointed_and_reversible() -> None:
    model = TSKV8Adapter(_config(budget=2), episode_id="region-growth")
    model.attach_adaptive_neuron_network("cortex", _network())
    model.attach_structural_growth_controller(_controller())

    first = model.propose_region_growth_from_error(
        network_id="cortex",
        bottleneck_region_id="bottleneck",
        input_dim=3,
        unit_count=2,
        fan_in=2,
        prediction_error=0.9,
        resource_state=0.8,
        holdout_transfer=0.9,
        evidence_ids=("region:tick:1",),
    )
    proposal = model.propose_region_growth_from_error(
        network_id="cortex",
        bottleneck_region_id="bottleneck",
        input_dim=3,
        unit_count=2,
        fan_in=2,
        prediction_error=0.9,
        resource_state=0.8,
        holdout_transfer=0.9,
        evidence_ids=("region:tick:2",),
    )

    assert first is None
    assert proposal is not None
    assert proposal.substrate_id == "bottleneck.region.1"
    assert dict(proposal.specification)["topology_role"] == "region"
    assert model.commit_region_add("cortex", proposal) is True
    network = model.neuron_networks[0]
    assert network.region_ids == ("source", "bottleneck", "bottleneck.region.1")
    assert network.execution_order == network.region_ids
    assert network.regions[-1].unit_ids == (
        "bottleneck.region.1.u0",
        "bottleneck.region.1.u1",
    )
    child = network.regions[-1]
    holdout_input = torch.ones(3)
    expected_activity = torch.full((child.unit_count,), 0.8)
    child.incoming.edge_weight.zero_()
    for _ in range(32):
        child.learn(holdout_input, expected_activity - child.activity)
    assert (
        model.validate_region_growth_holdout(
            network_id="cortex",
            proposal_id=proposal.proposal_id,
            holdout_inputs=(
                {proposal.substrate_id: holdout_input},
                {proposal.substrate_id: holdout_input},
            ),
            expected_activities=(
                {proposal.substrate_id: expected_activity},
                {proposal.substrate_id: expected_activity},
            ),
        )
        is True
    )
    assert model.topology_proposals[-1].validation_score > 0.05
    assert model.cognitive_snapshot().development.last_validation_status == "validated"

    connection = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="bottleneck.region.1",
        evidence_ids=("region:connection:holdout",),
        fan_in=1,
    )
    assert model.commit_cross_region_connection("cortex", connection) is True
    assert network.connection_ids == ("connection:source->bottleneck.region.1",)

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    restored_network = restored.neuron_networks[0]
    assert restored_network.region_ids == network.region_ids
    assert restored_network.execution_order == network.execution_order
    assert restored_network.connection_ids == network.connection_ids
    restored_network.lesion_region("bottleneck.region.1")
    assert restored.select_cross_region_connections("cortex") == ()

    assert restored.rollback_cross_region_connection(connection.proposal_id) is True
    assert restored.rollback_region_add(proposal.proposal_id) is True
    assert restored.neuron_networks[0].region_ids == ("source", "bottleneck")
    assert restored.cognitive_snapshot().development.structural_budget == 2


def test_region_growth_rejects_without_budget_and_does_not_mutate_network() -> None:
    model = TSKV8Adapter(_config(budget=0), episode_id="region-growth-no-budget")
    model.attach_adaptive_neuron_network("cortex", _network())
    model.attach_structural_growth_controller(_controller())
    proposal = None
    for index in range(1, 3):
        proposal = model.propose_region_growth_from_error(
            network_id="cortex",
            bottleneck_region_id="bottleneck",
            input_dim=3,
            unit_count=2,
            fan_in=2,
            prediction_error=0.9,
            resource_state=0.8,
            holdout_transfer=0.9,
            evidence_ids=(f"region:no-budget:{index}",),
        )
    assert proposal is not None
    assert model.commit_region_add("cortex", proposal) is False
    assert model.neuron_networks[0].region_ids == ("source", "bottleneck")
    assert model.topology_proposals[-1].status == "rejected"


def test_region_connection_is_blocked_until_holdout_gain_clears_threshold() -> None:
    model = TSKV8Adapter(_config(budget=2), episode_id="region-growth-holdout-block")
    model.attach_adaptive_neuron_network("cortex", _network())
    model.attach_structural_growth_controller(_controller())
    proposal = None
    for index in range(1, 3):
        proposal = model.propose_region_growth_from_error(
            network_id="cortex",
            bottleneck_region_id="bottleneck",
            input_dim=3,
            unit_count=2,
            fan_in=2,
            prediction_error=0.9,
            resource_state=0.8,
            holdout_transfer=0.9,
            evidence_ids=(f"region:blocked:{index}",),
        )
    assert proposal is not None
    assert model.commit_region_add("cortex", proposal) is True
    # The mechanism under test is the connection gate, not the newborn region's
    # random draw.  Silencing the child makes it provably indistinguishable from
    # the lesioned baseline, so the holdout gain is exactly zero by construction
    # instead of by luck; a freshly initialised region can legitimately beat
    # silence by a few percent on a single holdout sample, which used to make
    # this precondition hostage to any unrelated shift in RNG consumption order.
    child = next(
        region
        for region in model.neuron_networks[0].regions
        if region.region_id == proposal.substrate_id
    )
    child.incoming.edge_weight.zero_()
    assert (
        model.validate_region_growth_holdout(
            network_id="cortex",
            proposal_id=proposal.proposal_id,
            holdout_inputs=({proposal.substrate_id: torch.ones(3)},),
            expected_activities=({proposal.substrate_id: torch.ones(2)},),
        )
        is False
    )
    assert model.topology_proposals[-1].validation_score == 0.0
    assert model.cognitive_snapshot().development.last_validation_status == "rejected"
    connection = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id=proposal.substrate_id,
        evidence_ids=("region:blocked:connection",),
        fan_in=1,
    )
    with pytest.raises(ValueError, match="holdout validation"):
        model.commit_cross_region_connection("cortex", connection)


def test_region_pruning_is_resource_aware_checkpointed_and_reversible() -> None:
    model = TSKV8Adapter(_config(budget=2), episode_id="region-pruning")
    model.attach_adaptive_neuron_network("cortex", _network())
    connection = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="bottleneck",
        evidence_ids=("prune:connection",),
        fan_in=1,
    )
    assert model.commit_cross_region_connection("cortex", connection) is True
    model.attach_structural_pruning_controller(_pruning_controller())

    first = model.propose_region_prune_from_underuse(
        network_id="cortex",
        region_id="bottleneck",
        usage=0.05,
        resource_pressure=0.9,
        learning_gain=0.0,
        evidence_ids=("prune:tick:1",),
    )
    proposal = model.propose_region_prune_from_underuse(
        network_id="cortex",
        region_id="bottleneck",
        usage=0.05,
        resource_pressure=0.9,
        learning_gain=0.0,
        evidence_ids=("prune:tick:2",),
    )
    assert first is None
    assert proposal is not None
    assert (
        model.validate_region_prune_holdout(
            network_id="cortex",
            proposal_id=proposal.proposal_id,
            holdout_inputs=({"source": torch.ones(3)}, {"source": torch.ones(3)}),
            expected_activities=(
                {"source": torch.zeros(2)},
                {"source": torch.zeros(2)},
            ),
        )
        is True
    )
    assert model.commit_region_prune("cortex", proposal) is True
    assert model.neuron_networks[0].region_ids == ("source",)
    assert model.neuron_networks[0].connection_ids == ()
    assert model.cognitive_snapshot().development.prune_count == 1

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.structural_pruning_controller is not None
    assert restored.structural_pruning_controller.total_observations == 2
    assert restored.rollback_region_prune(proposal.proposal_id) is True
    assert restored.neuron_networks[0].region_ids == ("source", "bottleneck")
    assert restored.neuron_networks[0].connection_ids == (connection.substrate_id,)
    assert restored.rollback_cross_region_connection(connection.proposal_id) is True
    assert restored.cognitive_snapshot().development.structural_budget == 2


def test_region_pruning_requires_learning_stagnation() -> None:
    model = TSKV8Adapter(_config(budget=2), episode_id="region-pruning-gain")
    model.attach_adaptive_neuron_network("cortex", _network())
    model.attach_structural_pruning_controller(_pruning_controller())

    assert (
        model.propose_region_prune_from_underuse(
            network_id="cortex",
            region_id="bottleneck",
            usage=0.05,
            resource_pressure=0.9,
            learning_gain=1.0,
            evidence_ids=("prune:gain:tick:1",),
        )
        is None
    )
    assert (
        model.propose_region_prune_from_underuse(
            network_id="cortex",
            region_id="bottleneck",
            usage=0.05,
            resource_pressure=0.9,
            learning_gain=1.0,
            evidence_ids=("prune:gain:tick:2",),
        )
        is None
    )
    state = model.structural_pruning_controller.regions[0]
    assert state.learning_gain_ema == 1.0
    assert state.proposal_count == 0


def test_cross_region_connection_pruning_is_checkpointed_and_reversible() -> None:
    model = TSKV8Adapter(_config(budget=2), episode_id="connection-pruning")
    model.attach_adaptive_neuron_network("cortex", _network())
    connection = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="bottleneck",
        evidence_ids=("route:connection",),
        fan_in=1,
    )
    assert model.commit_cross_region_connection("cortex", connection) is True
    network = model.neuron_networks[0]
    network.attach_cooperation_learner(
        CrossRegionCooperationLearner(
            dynamics=CrossRegionLearningDynamics(ema_rate=1.0),
        )
    )
    network.observe_connection(
        connection.substrate_id,
        prediction_error=1.0,
        holdout_transfer=0.0,
        resource_state=0.1,
        selected=False,
    )
    network.observe_connection(
        connection.substrate_id,
        prediction_error=1.0,
        holdout_transfer=0.0,
        resource_state=0.1,
        selected=False,
    )
    for region in network.regions:
        region.incoming.edge_weight.zero_()
        region.activity.zero_()
    network.connections[0][3].edge_weight.zero_()
    model.attach_structural_pruning_controller(_pruning_controller())

    first = model.propose_cross_region_connection_prune_from_route(
        network_id="cortex",
        connection_id=connection.substrate_id,
        evidence_ids=("route:tick:1",),
    )
    proposal = model.propose_cross_region_connection_prune_from_route(
        network_id="cortex",
        connection_id=connection.substrate_id,
        evidence_ids=("route:tick:2",),
    )
    assert first is None
    assert proposal is not None
    assert (
        model.validate_cross_region_connection_prune_holdout(
            network_id="cortex",
            proposal_id=proposal.proposal_id,
            holdout_inputs=(
                {"source": torch.ones(3)},
                {"source": torch.ones(3)},
            ),
            expected_activities=(
                {"source": torch.zeros(2), "bottleneck": torch.zeros(2)},
                {"source": torch.zeros(2), "bottleneck": torch.zeros(2)},
            ),
        )
        is True
    )
    assert model.commit_cross_region_connection_prune("cortex", proposal) is True
    assert model.neuron_networks[0].region_ids == ("source", "bottleneck")
    assert model.neuron_networks[0].connection_ids == ()
    assert model.cognitive_snapshot().development.prune_count == 1

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.structural_pruning_controller is not None
    assert restored.structural_pruning_controller.total_observations == 2
    assert restored.rollback_cross_region_connection_prune(proposal.proposal_id) is True
    assert restored.neuron_networks[0].connection_ids == (connection.substrate_id,)
    assert restored.cognitive_snapshot().development.structural_budget == 1
    assert restored.rollback_cross_region_connection(connection.proposal_id) is True
    assert restored.neuron_networks[0].connection_ids == ()
    assert restored.cognitive_snapshot().development.structural_budget == 2


def test_region_split_preserves_unit_lineage_and_is_reversible() -> None:
    model = TSKV8Adapter(_config(budget=2), episode_id="region-split")
    model.attach_adaptive_neuron_network("cortex", _network())
    model.attach_structural_growth_controller(_controller())
    network = model.neuron_networks[0]
    network.regions[1].incoming.edge_weight.zero_()
    network.regions[1].activity.zero_()

    first = model.propose_region_split_from_error(
        network_id="cortex",
        region_id="bottleneck",
        first_unit_count=1,
        prediction_error=0.9,
        resource_state=0.8,
        holdout_transfer=0.9,
        evidence_ids=("split:tick:1",),
    )
    proposal = model.propose_region_split_from_error(
        network_id="cortex",
        region_id="bottleneck",
        first_unit_count=1,
        prediction_error=0.9,
        resource_state=0.8,
        holdout_transfer=0.9,
        evidence_ids=("split:tick:2",),
    )
    assert first is None
    assert proposal is not None
    assert (
        model.validate_region_split_holdout(
            network_id="cortex",
            proposal_id=proposal.proposal_id,
            holdout_inputs=(
                {"bottleneck": torch.ones(3)},
                {"bottleneck": torch.ones(3)},
            ),
            expected_activities=(
                {"bottleneck": torch.zeros(2)},
                {"bottleneck": torch.zeros(2)},
            ),
        )
        is True
    )
    assert model.commit_region_split("cortex", proposal) is True
    assert network.region_ids == ("source", "bottleneck", "bottleneck.split.1")
    assert network.execution_order == network.region_ids
    assert network.regions[1].unit_ids == ("bottleneck.u0",)
    assert network.regions[2].unit_ids == ("bottleneck.u1",)
    assert model.cognitive_snapshot().development.split_merge_count == 1

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.rollback_region_split(proposal.proposal_id) is True
    assert restored.neuron_networks[0].region_ids == ("source", "bottleneck")
    assert restored.neuron_networks[0].regions[1].unit_ids == (
        "bottleneck.u0",
        "bottleneck.u1",
    )
    assert restored.cognitive_snapshot().development.structural_budget == 2


def test_connected_region_split_migrates_connections_and_route_lineage() -> None:
    model = TSKV8Adapter(_config(budget=2), episode_id="connected-region-split")
    model.attach_adaptive_neuron_network("cortex", _network())
    connection = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="bottleneck",
        evidence_ids=("split:route",),
        fan_in=1,
    )
    assert model.commit_cross_region_connection("cortex", connection) is True
    network = model.neuron_networks[0]
    network.attach_cooperation_learner(
        CrossRegionCooperationLearner(
            dynamics=CrossRegionLearningDynamics(ema_rate=1.0),
        )
    )
    network.observe_connection(
        connection.substrate_id,
        prediction_error=0.5,
        holdout_transfer=0.5,
        resource_state=0.8,
        selected=True,
    )
    for region in network.regions:
        region.incoming.edge_weight.zero_()
        region.activity.zero_()
    network.connections[0][3].edge_weight.zero_()
    model.attach_structural_growth_controller(_controller())

    first = model.propose_region_split_from_error(
        network_id="cortex",
        region_id="bottleneck",
        first_unit_count=1,
        prediction_error=0.9,
        resource_state=0.8,
        holdout_transfer=0.9,
        evidence_ids=("split:connected:tick:1",),
    )
    proposal = model.propose_region_split_from_error(
        network_id="cortex",
        region_id="bottleneck",
        first_unit_count=1,
        prediction_error=0.9,
        resource_state=0.8,
        holdout_transfer=0.9,
        evidence_ids=("split:connected:tick:2",),
    )
    assert first is None
    assert proposal is not None
    assert (
        model.validate_region_split_holdout(
            network_id="cortex",
            proposal_id=proposal.proposal_id,
            holdout_inputs=({"source": torch.ones(3), "bottleneck": torch.zeros(3)},),
            expected_activities=({"bottleneck": torch.zeros(2)},),
        )
        is True
    )
    assert model.commit_region_split("cortex", proposal) is True
    assert network.region_ids == ("source", "bottleneck", "bottleneck.split.1")
    assert network.execution_order == network.region_ids
    assert network.connection_ids == (
        "connection:source->bottleneck",
        "connection:source->bottleneck.split.1",
    )
    assert network.cooperation_learner is not None
    assert network.cooperation_learner.route_ids == network.connection_ids
    assert all(route.evidence_count == 1 for route in network.cooperation_learner.routes)

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.rollback_region_split(proposal.proposal_id) is True
    assert restored.neuron_networks[0].region_ids == ("source", "bottleneck")
    assert restored.neuron_networks[0].connection_ids == (connection.substrate_id,)
    assert restored.neuron_networks[0].cooperation_learner is not None
    assert restored.neuron_networks[0].cooperation_learner.route_ids == (connection.substrate_id,)


def test_region_merge_aggregates_external_connections_and_is_reversible() -> None:
    model = TSKV8Adapter(_config(budget=3), episode_id="region-merge")

    def make(region_id: str) -> AdaptiveNeuronRegion:
        return AdaptiveNeuronRegion(
            region_id=region_id,
            input_dim=3,
            unit_ids=(f"{region_id}.u0", f"{region_id}.u1"),
            fan_in=2,
            generator=torch.Generator().manual_seed(len(region_id)),
        )

    model.attach_adaptive_neuron_network(
        "cortex",
        AdaptiveNeuronNetwork(
            (make("source"), make("bottleneck"), make("sink")),
            execution_order=("source", "bottleneck", "sink"),
        ),
    )
    first_connection = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="sink",
        evidence_ids=("merge:first",),
        fan_in=1,
    )
    second_connection = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="bottleneck",
        target_region_id="sink",
        evidence_ids=("merge:second",),
        fan_in=1,
    )
    assert model.commit_cross_region_connection("cortex", first_connection) is True
    assert model.commit_cross_region_connection("cortex", second_connection) is True
    network = model.neuron_networks[0]
    network.attach_cooperation_learner(
        CrossRegionCooperationLearner(
            dynamics=CrossRegionLearningDynamics(ema_rate=1.0),
        )
    )
    for connection_id in network.connection_ids:
        network.observe_connection(
            connection_id,
            prediction_error=0.5,
            holdout_transfer=0.5,
            resource_state=0.8,
            selected=True,
        )
    for region in network.regions:
        region.incoming.edge_weight.zero_()
        region.activity.zero_()
    for _, _, _, connection in network.connections:
        connection.edge_weight.zero_()
    model.attach_structural_pruning_controller(_pruning_controller())

    first = model.propose_region_merge_from_redundancy(
        network_id="cortex",
        region_ids=("source", "bottleneck"),
        usage=0.05,
        resource_pressure=0.9,
        learning_gain=0.0,
        evidence_ids=("merge:tick:1",),
    )
    proposal = model.propose_region_merge_from_redundancy(
        network_id="cortex",
        region_ids=("source", "bottleneck"),
        usage=0.05,
        resource_pressure=0.9,
        learning_gain=0.0,
        evidence_ids=("merge:tick:2",),
    )
    assert first is None
    assert proposal is not None
    assert (
        model.validate_region_merge_holdout(
            network_id="cortex",
            proposal_id=proposal.proposal_id,
            holdout_inputs=(
                {
                    "source": torch.ones(3),
                    "bottleneck": torch.ones(3),
                },
            ),
            expected_activities=({"source": torch.zeros(4)},),
        )
        is True
    )
    assert model.commit_region_merge("cortex", proposal) is True
    assert network.region_ids == ("source", "sink")
    assert network.execution_order == network.region_ids
    assert network.regions[0].unit_ids == (
        "source.u0",
        "source.u1",
        "bottleneck.u0",
        "bottleneck.u1",
    )
    assert network.connection_ids == ("connection:source->sink",)
    assert network.cooperation_learner is not None
    assert network.cooperation_learner.route_ids == network.connection_ids
    assert network.cooperation_learner.routes[0].evidence_count == 2
    assert model.cognitive_snapshot().development.split_merge_count == 1

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.rollback_region_merge(proposal.proposal_id) is True
    assert restored.neuron_networks[0].region_ids == (
        "source",
        "bottleneck",
        "sink",
    )
    assert restored.neuron_networks[0].connection_ids == (
        first_connection.substrate_id,
        second_connection.substrate_id,
    )
    assert restored.rollback_cross_region_connection(second_connection.proposal_id) is True
    assert restored.rollback_cross_region_connection(first_connection.proposal_id) is True
    assert restored.cognitive_snapshot().development.structural_budget == 3
