from __future__ import annotations

import torch

from taiji import (
    AdaptiveNeuronNetwork,
    AdaptiveNeuronRegion,
    AdaptiveStructuralGrowthController,
    AdaptiveStructuralPruningController,
    CrossRegionCooperationLearner,
    StructuralGrowthDynamics,
    StructuralProposalCandidate,
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


def _region() -> AdaptiveNeuronRegion:
    return AdaptiveNeuronRegion(
        region_id="adaptive.cortex",
        input_dim=5,
        unit_ids=("u0", "u1"),
        fan_in=2,
        generator=torch.Generator().manual_seed(7),
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
        (make("source"), make("target")),
        execution_order=("source", "target"),
    )


def _three_region_network() -> AdaptiveNeuronNetwork:
    def make(region_id: str) -> AdaptiveNeuronRegion:
        return AdaptiveNeuronRegion(
            region_id=region_id,
            input_dim=3,
            unit_ids=(f"{region_id}.u0", f"{region_id}.u1"),
            fan_in=2,
            generator=torch.Generator().manual_seed(len(region_id) + 11),
        )

    return AdaptiveNeuronNetwork(
        (make("source"), make("relay"), make("target")),
        execution_order=("source", "relay", "target"),
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


def test_runtime_standalone_tick_owns_neuron_birth_candidate() -> None:
    model = TSKV8Adapter(_config(budget=1), episode_id="standalone-runtime")
    region = _region()
    model.attach_adaptive_neuron_region(region)
    model.attach_structural_growth_controller(
        AdaptiveStructuralGrowthController(
            dynamics=StructuralGrowthDynamics(
                ema_rate=1.0,
                error_threshold=0.0,
                holdout_transfer_threshold=0.0,
                minimum_resource_state=0.0,
                minimum_holdout_gain=0.05,
                required_error_steps=1,
            )
        )
    )

    first = model.step_adaptive_neuron_region(region.region_id, torch.ones(5))
    model.step_adaptive_neuron_region(
        region.region_id,
        torch.ones(5),
        expected_activity=first,
        holdout=True,
    )
    observations = model.structural_runtime_observations
    assert len(observations) == 2
    assert observations[0].prediction_error is None
    assert observations[1].prediction_error is not None
    candidates = model.structural_proposal_candidates
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.target_kind == "neuron"
    assert candidate.operation == "add"
    assert candidate.substrate_ids == (region.region_id,)
    original_unit_ids = region.unit_ids

    proposal = model.materialize_structural_candidate(candidate.candidate_id)
    assert proposal is not None
    assert model.structural_proposal_candidates == ()
    trial = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    trial.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(0),
    )
    holdout_input = torch.zeros(5)
    holdout_input[trial.incoming.pre_index[-1]] = torch.sign(
        trial.incoming.edge_weight[-1]
    )
    expected = trial.step(holdout_input)
    assert model.validate_structural_candidate_holdout(
        candidate.candidate_id,
        holdout_inputs=(holdout_input,),
        expected_activities=(expected,),
    ) is True
    assert model.commit_structural_candidate(candidate.candidate_id) is True
    assert model.neuron_regions[0].unit_count == 3

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.neuron_regions[0].unit_ids == original_unit_ids + (
        "adaptive.cortex.grown.1",
    )
    assert restored.structural_runtime_observations == observations
    assert restored.topology_proposals[-1].status == "accepted"
    assert restored.rollback_structural_candidate(candidate.candidate_id) is True
    assert restored.neuron_regions[0].unit_ids == original_unit_ids
    assert restored.cognitive_snapshot().development.structural_budget == 1


def test_neuron_birth_candidates_can_grow_one_population_in_dependency_order() -> None:
    model = TSKV8Adapter(_config(budget=2), episode_id="sequential-neuron-birth")
    region = _region()
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
    first_proposal = model.propose_neuron_add(
        region_id=region.region_id,
        unit_id="adaptive.cortex.first",
        evidence_ids=("runtime:first-birth",),
    )
    second_proposal = StructuralProposalCandidate(
        candidate_id="candidate:second-birth",
        network_id="standalone:adaptive.cortex",
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        evidence_ids=("runtime:second-birth",),
        source_tick=2,
        priority=0.8,
        specification=(
            ("region_id", region.region_id),
            ("unit_id", "adaptive.cortex.second"),
        ),
        depends_on_candidate_ids=("candidate:first-birth",),
    )
    first_candidate = StructuralProposalCandidate(
        candidate_id="candidate:first-birth",
        network_id="standalone:adaptive.cortex",
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        evidence_ids=("runtime:first-birth",),
        source_tick=1,
        priority=0.9,
        specification=(
            ("region_id", region.region_id),
            ("unit_id", "adaptive.cortex.first"),
        ),
    )
    model._queue_structural_proposal_candidate(first_candidate)
    model._queue_structural_proposal_candidate(second_proposal)

    first_trial = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    first_trial.apply_topology_proposal(
        first_proposal,
        generator=torch.Generator().manual_seed(0),
    )
    first_input = torch.zeros(5)
    first_input[first_trial.incoming.pre_index[-1]] = torch.sign(
        first_trial.incoming.edge_weight[-1]
    )
    first_expected = first_trial.step(first_input)

    second_parent = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    second_parent.apply_topology_proposal(
        first_proposal,
        generator=torch.Generator().manual_seed(0),
    )
    second_proposal_for_trial = second_parent.propose_unit_add(
        unit_id="adaptive.cortex.second",
        evidence_ids=("runtime:second-birth",),
    )
    second_trial = AdaptiveNeuronRegion.from_payload(
        second_parent.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    second_trial.apply_topology_proposal(
        second_proposal_for_trial,
        generator=torch.Generator().manual_seed(0),
    )
    second_input = torch.zeros(5)
    second_input[second_trial.incoming.pre_index[-1]] = torch.sign(
        second_trial.incoming.edge_weight[-1]
    )
    second_expected = second_trial.step(second_input)

    results = model.run_structural_maintenance_cycle(
        candidate_ids=(second_proposal.candidate_id, first_candidate.candidate_id),
        holdout_inputs_by_candidate={
            first_candidate.candidate_id: (first_input,),
            second_proposal.candidate_id: (second_input,),
        },
        expected_activities_by_candidate={
            first_candidate.candidate_id: (first_expected,),
            second_proposal.candidate_id: (second_expected,),
        },
    )
    assert tuple(item.candidate_id for item in results) == (
        first_candidate.candidate_id,
        second_proposal.candidate_id,
    )
    assert all(item.status == "committed" for item in results)
    assert region.unit_ids == (
        "u0",
        "u1",
        "adaptive.cortex.first",
        "adaptive.cortex.second",
    )
    assert model.cognitive_snapshot().development.structural_budget == 0
    assert model.rollback_structural_candidate(second_proposal.candidate_id) is True
    assert model.rollback_structural_candidate(first_candidate.candidate_id) is True
    assert model.neuron_regions[0].unit_ids == ("u0", "u1")


def test_three_region_maintenance_preserves_routes_across_mixed_growth() -> None:
    model = TSKV8Adapter(_config(budget=6), episode_id="three-region-maintenance")
    network = _three_region_network()
    model.attach_adaptive_neuron_network("cortex", network)
    source_relay = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="relay",
        evidence_ids=("runtime:source-relay",),
        fan_in=1,
    )
    relay_target = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="relay",
        target_region_id="target",
        evidence_ids=("runtime:relay-target",),
        fan_in=1,
    )
    assert model.commit_cross_region_connection("cortex", source_relay) is True
    assert model.commit_cross_region_connection("cortex", relay_target) is True
    original_region_ids = network.region_ids
    original_connection_ids = network.connection_ids
    model.attach_cross_region_cooperation("cortex", CrossRegionCooperationLearner())
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
    first = model.step_cross_region_network(
        "cortex",
        {"source": torch.ones(3)},
        max_connections=2,
    )
    model.step_cross_region_network(
        "cortex",
        {"source": torch.ones(3)},
        expected_activities=first,
        holdout=True,
        max_connections=2,
    )

    standalone = _region()
    model.attach_adaptive_neuron_region(standalone)
    standalone_first = model.step_adaptive_neuron_region(
        standalone.region_id,
        torch.ones(5),
    )
    model.step_adaptive_neuron_region(
        standalone.region_id,
        torch.ones(5),
        expected_activity=standalone_first,
        holdout=True,
    )
    split_candidate = next(
        item
        for item in model.structural_proposal_candidates
        if item.network_id == "cortex" and item.operation == "split"
    )
    add_candidate = next(
        item
        for item in model.structural_proposal_candidates
        if item.network_id == "standalone:adaptive.cortex"
        and item.target_kind == "neuron"
    )
    add_proposal = model.propose_neuron_add(
        region_id=standalone.region_id,
        unit_id=dict(add_candidate.specification)["unit_id"],
        evidence_ids=add_candidate.evidence_ids,
    )
    add_trial = AdaptiveNeuronRegion.from_payload(
        standalone.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    add_trial.apply_topology_proposal(
        add_proposal,
        generator=torch.Generator().manual_seed(0),
    )
    add_input = torch.zeros(5)
    add_input[add_trial.incoming.pre_index[-1]] = torch.sign(
        add_trial.incoming.edge_weight[-1]
    )
    add_expected = add_trial.step(add_input)

    results = model.run_structural_maintenance_cycle(
        candidate_ids=(add_candidate.candidate_id, split_candidate.candidate_id),
        holdout_inputs_by_candidate={
            add_candidate.candidate_id: (add_input,),
            split_candidate.candidate_id: ({"source": torch.ones(3)},),
        },
        expected_activities_by_candidate={
            add_candidate.candidate_id: (add_expected,),
            split_candidate.candidate_id: (first,),
        },
    )
    assert all(item.status == "committed" for item in results)
    assert model.neuron_regions[0].unit_count == 3
    assert len(model.neuron_networks[0].region_ids) == 4
    assert model.neuron_networks[0].connection_ids == (
        "connection:relay->target",
        "connection:source->relay",
        "connection:source.split.1->relay",
    )

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.neuron_regions[0].unit_count == 3
    assert len(restored.neuron_networks[0].region_ids) == 4
    assert restored.neuron_networks[0].connection_ids == model.neuron_networks[0].connection_ids
    assert restored.rollback_structural_candidate(split_candidate.candidate_id) is True
    assert restored.rollback_structural_candidate(add_candidate.candidate_id) is True
    assert restored.neuron_regions[0].unit_count == 2
    assert restored.neuron_networks[0].region_ids == original_region_ids
    assert restored.neuron_networks[0].connection_ids == original_connection_ids


def test_structural_maintenance_cycle_fails_closed_on_dependency_and_conflict() -> None:
    model = TSKV8Adapter(_config(budget=1), episode_id="maintenance-guards")
    region = _region()
    model.attach_adaptive_neuron_region(region)
    dependency = StructuralProposalCandidate(
        candidate_id="candidate:dependency",
        network_id="standalone:adaptive.cortex",
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        evidence_ids=("runtime:dependency",),
        source_tick=1,
        priority=0.8,
        specification=(
            ("region_id", region.region_id),
            ("unit_id", "adaptive.cortex.dep"),
        ),
        conflict_keys=("dependency-domain",),
    )
    dependent = StructuralProposalCandidate(
        candidate_id="candidate:dependent",
        network_id="standalone:adaptive.cortex",
        target_kind="region",
        operation="split",
        substrate_ids=("adaptive.other",),
        evidence_ids=("runtime:dependent",),
        source_tick=2,
        priority=0.7,
        specification=(
            ("region_id", "adaptive.other"),
            ("unit_id", "adaptive.cortex.child"),
        ),
        depends_on_candidate_ids=(dependency.candidate_id,),
        conflict_keys=("dependent-domain",),
    )
    model._queue_structural_proposal_candidate(dependency)
    model._queue_structural_proposal_candidate(dependent)

    results = model.run_structural_maintenance_cycle(
        candidate_ids=(
            dependent.candidate_id,
            dependency.candidate_id,
        ),
        holdout_inputs_by_candidate={
            dependent.candidate_id: (torch.ones(5),),
        },
        expected_activities_by_candidate={
            dependent.candidate_id: (torch.zeros(3),),
        },
    )
    assert tuple(item.candidate_id for item in results) == (
        dependency.candidate_id,
        dependent.candidate_id,
    )
    assert results[0].status == "missing_holdout"
    assert "dependency" in (results[1].error or "")
    assert results[1].status == "failed_closed"
    assert region.unit_ids == ("u0", "u1")
    assert model.topology_proposals == ()

    conflict_model = TSKV8Adapter(_config(budget=1), episode_id="maintenance-conflict")
    conflict_region = _region()
    conflict_model.attach_adaptive_neuron_region(conflict_region)
    conflict_add = StructuralProposalCandidate(
        candidate_id="candidate:conflict-add",
        network_id="standalone:adaptive.cortex",
        target_kind="neuron",
        operation="add",
        substrate_ids=(conflict_region.region_id,),
        evidence_ids=("runtime:conflict-add",),
        source_tick=1,
        priority=0.8,
        specification=(
            ("region_id", conflict_region.region_id),
            ("unit_id", "adaptive.cortex.conflict"),
        ),
    )
    conflict_prune = StructuralProposalCandidate(
        candidate_id="candidate:conflict-prune",
        network_id="standalone:adaptive.cortex",
        target_kind="region",
        operation="prune",
        substrate_ids=(conflict_region.region_id,),
        evidence_ids=("runtime:conflict-prune",),
        source_tick=2,
        priority=0.7,
        specification=(("region_id", conflict_region.region_id),),
    )
    conflict_model._queue_structural_proposal_candidate(conflict_add)
    conflict_model._queue_structural_proposal_candidate(conflict_prune)
    conflict_results = conflict_model.run_structural_maintenance_cycle(
        candidate_ids=(conflict_add.candidate_id, conflict_prune.candidate_id),
        holdout_inputs_by_candidate={
            conflict_add.candidate_id: (torch.ones(5),),
            conflict_prune.candidate_id: (torch.ones(5),),
        },
        expected_activities_by_candidate={
            conflict_add.candidate_id: (torch.zeros(3),),
            conflict_prune.candidate_id: (torch.zeros(3),),
        },
    )
    assert all(item.status == "failed_closed" for item in conflict_results)
    assert all("conflict" in (item.error or "") for item in conflict_results)
    assert conflict_region.unit_ids == ("u0", "u1")
    assert conflict_model.topology_proposals == ()


def test_runtime_cross_region_ledger_owns_connection_and_rolls_back() -> None:
    model = TSKV8Adapter(_config(budget=1), episode_id="cross-region-ledger")
    network = _network()
    model.attach_adaptive_neuron_network("cortex", network)
    proposal = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="target",
        evidence_ids=("runtime:cross-region-holdout",),
        fan_in=1,
    )

    assert model.commit_cross_region_connection("cortex", proposal) is True
    assert model.neuron_networks[0].connection_ids == (
        "connection:source->target",
    )
    assert model.cognitive_snapshot().development.structural_budget == 0
    assert model.cognitive_snapshot().development.last_update_source == (
        "cross-region-topology-growth"
    )
    model.attach_cross_region_cooperation("cortex", CrossRegionCooperationLearner())
    first = model.step_cross_region_network(
        "cortex",
        {"source": torch.ones(3)},
    )
    model.step_cross_region_network(
        "cortex",
        {"source": torch.ones(3)},
        expected_activities={"target": first["target"]},
        holdout=True,
    )
    assert model.select_cross_region_connections("cortex") == (proposal.substrate_id,)

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.neuron_networks[0].connection_ids == (
        "connection:source->target",
    )
    assert restored.select_cross_region_connections("cortex") == (proposal.substrate_id,)
    assert restored.rollback_cross_region_connection(proposal.proposal_id) is True
    assert restored.neuron_networks[0].connection_ids == ()
    assert restored.cognitive_snapshot().development.structural_budget == 1


def test_runtime_tick_feeds_structural_organs_and_checkpoint_continues() -> None:
    model = TSKV8Adapter(_config(budget=1), episode_id="runtime-structure")
    model.attach_adaptive_neuron_network("cortex", _network())
    model.attach_cross_region_cooperation("cortex", CrossRegionCooperationLearner())
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
    model.attach_structural_pruning_controller(
        AdaptiveStructuralPruningController(
            dynamics=StructuralPruningDynamics(ema_rate=1.0)
        )
    )

    first = model.step_cross_region_network(
        "cortex",
        {"source": torch.ones(3)},
    )
    observations = model.structural_runtime_observations
    assert len(observations) == 2
    assert observations[0].tick == 1
    assert observations[0].prediction_error is None
    assert model.structural_growth_controller is not None
    assert model.structural_growth_controller.total_observations == 0
    assert model.structural_pruning_controller is not None
    assert model.structural_pruning_controller.total_observations == 2

    model.step_cross_region_network(
        "cortex",
        {"source": torch.ones(3)},
        expected_activities=first,
        holdout=True,
    )
    assert len(model.structural_runtime_observations) == 4
    assert all(item.prediction_error is not None for item in model.structural_runtime_observations[2:])
    assert model.structural_growth_controller.total_observations == 2
    assert model.structural_pruning_controller.total_observations == 5
    assert {item.operation for item in model.structural_proposal_candidates} == {"split"}
    assert {item.substrate_ids for item in model.structural_proposal_candidates} == {
        ("source",),
        ("target",),
    }
    candidate = model.structural_proposal_candidates[0]
    materialized = model.materialize_structural_candidate(candidate.candidate_id)
    assert materialized is not None
    assert materialized.status == "pending"
    assert model.neuron_networks[0].region_ids == ("source", "target")
    assert candidate not in model.structural_proposal_candidates
    assert model.validate_structural_candidate_holdout(
        candidate.candidate_id,
        holdout_inputs=({"source": torch.ones(3)},),
        expected_activities=(first,),
    ) is True
    materialized = next(
        item for item in model.topology_proposals if item.proposal_id == materialized.proposal_id
    )
    assert model.cognitive_snapshot().development.last_update_source == (
        "region-split-holdout-validation"
    )
    assert model.commit_structural_candidate(candidate.candidate_id) is True
    assert model.neuron_networks[0].region_ids == (
        "source",
        "target",
        "source.split.1",
    )
    assert model.rollback_structural_candidate(candidate.candidate_id) is True
    assert model.neuron_networks[0].region_ids == ("source", "target")
    materialized = next(
        item for item in model.topology_proposals if item.proposal_id == materialized.proposal_id
    )
    assert materialized.status == "rolled_back"
    remaining = model.structural_proposal_candidates[0]
    cycle = model.run_structural_maintenance_cycle(
        candidate_ids=(remaining.candidate_id,),
        holdout_inputs_by_candidate={
            remaining.candidate_id: ({"target": torch.ones(3)},),
        },
        expected_activities_by_candidate={
            remaining.candidate_id: (first,),
        },
    )
    assert len(cycle) == 1
    assert cycle[0].status == "committed"
    assert model.neuron_networks[0].region_ids == (
        "source",
        "target",
        "target.split.1",
    )
    assert model.rollback_structural_candidate(remaining.candidate_id) is True
    assert model.neuron_networks[0].region_ids == ("source", "target")
    assert model.structural_maintenance_results[-1].status == "committed"

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    assert restored.structural_runtime_observations == model.structural_runtime_observations
    assert restored.structural_proposal_candidates == model.structural_proposal_candidates
    assert restored.structural_maintenance_results == model.structural_maintenance_results
    assert restored.materialize_structural_candidate(candidate.candidate_id) == materialized
    assert restored.structural_growth_controller is not None
    assert restored.structural_growth_controller.total_observations == 2
    restored.step_cross_region_network(
        "cortex",
        {"source": torch.ones(3)},
        expected_activities=first,
        holdout=True,
    )
    assert restored.structural_runtime_observations[-1].tick == 3
    assert restored.structural_growth_controller.total_observations == 4
