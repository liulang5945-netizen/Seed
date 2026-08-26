from __future__ import annotations

import torch

from taiji import (
    AdaptiveNeuronNetwork,
    AdaptiveNeuronRegion,
    CrossRegionCooperationLearner,
    CrossRegionLearningDynamics,
    NeuronRegionDynamics,
)


def _region(region_id: str) -> AdaptiveNeuronRegion:
    return AdaptiveNeuronRegion(
        region_id=region_id,
        input_dim=3,
        unit_ids=(f"{region_id}.u0", f"{region_id}.u1"),
        fan_in=2,
        dynamics=NeuronRegionDynamics(
            membrane_decay=0.0,
            recurrent_gain=0.0,
        ),
        generator=torch.Generator().manual_seed(len(region_id)),
    )


def _network() -> AdaptiveNeuronNetwork:
    return AdaptiveNeuronNetwork(
        (_region("good"), _region("distractor"), _region("target")),
        execution_order=("good", "distractor", "target"),
    )


def _add_connection(
    network: AdaptiveNeuronNetwork,
    source_region_id: str,
    target_region_id: str,
):
    proposal = network.propose_connection_add(
        source_region_id=source_region_id,
        target_region_id=target_region_id,
        evidence_ids=(f"holdout:{source_region_id}",),
        fan_in=1,
        parent_checkpoint_id="parent:network",
    )
    network.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(99),
    )
    return proposal


def test_route_learner_prefers_holdout_transfer_and_resource_state() -> None:
    learner = CrossRegionCooperationLearner()
    learner.register_connection("good", resource_cost=1.0)
    learner.register_connection("distractor", resource_cost=1.0)
    learner.observe(
        "good",
        prediction_error=0.10,
        holdout_transfer=0.95,
        resource_state=0.90,
    )
    learner.observe(
        "distractor",
        prediction_error=0.80,
        holdout_transfer=0.15,
        resource_state=0.90,
    )

    assert learner.select(("distractor", "good")) == ("good",)
    assert learner.score("good") > learner.score("distractor")

    resource_only = CrossRegionCooperationLearner(
        dynamics=CrossRegionLearningDynamics(
            quality_weight=0.0,
            transfer_weight=0.0,
            resource_weight=1.0,
            cost_weight=0.0,
            exploration_weight=0.0,
        )
    )
    resource_only.register_connection("low", resource_cost=1.0)
    resource_only.register_connection("high", resource_cost=1.0)
    resource_only.observe(
        "low",
        prediction_error=0.2,
        holdout_transfer=0.8,
        resource_state=0.1,
    )
    resource_only.observe(
        "high",
        prediction_error=0.2,
        holdout_transfer=0.8,
        resource_state=0.9,
    )
    assert resource_only.select() == ("high",)


def test_network_routes_learned_path_and_preserves_it_through_checkpoint_and_lesions() -> None:
    network = _network()
    good = _add_connection(network, "good", "target")
    distractor = _add_connection(network, "distractor", "target")
    network.attach_cooperation_learner(CrossRegionCooperationLearner())
    network.observe_connection(
        good.substrate_id,
        prediction_error=0.10,
        holdout_transfer=0.95,
        resource_state=0.90,
    )
    network.observe_connection(
        distractor.substrate_id,
        prediction_error=0.80,
        holdout_transfer=0.15,
        resource_state=0.90,
    )

    assert network.selected_connection_ids() == (good.substrate_id,)
    restored = AdaptiveNeuronNetwork.from_payload(
        network.to_payload(),
        generator=torch.Generator().manual_seed(123),
    )
    assert restored.selected_connection_ids() == (good.substrate_id,)

    restored.lesion_topology_proposal(good)
    assert restored.selected_connection_ids() == (distractor.substrate_id,)
    restored.lesion_region("target")
    assert restored.selected_connection_ids() == ()
    activities = restored.step({"good": torch.ones(3), "distractor": torch.ones(3)})
    assert float(activities["target"].sum().item()) == 0.0


def test_network_step_derives_online_route_credit_from_actual_target_outcome() -> None:
    network = _network()
    proposal = _add_connection(network, "good", "target")
    network.attach_cooperation_learner(CrossRegionCooperationLearner())
    first = network.step(
        {"good": torch.ones(3)},
        connection_ids=(proposal.substrate_id,),
    )
    network.step(
        {"good": torch.ones(3)},
        connection_ids=(proposal.substrate_id,),
        expected_activities={"target": first["target"]},
        holdout=True,
    )

    route = network.cooperation_learner
    assert route is not None
    assert route.total_evidence == 1
    assert route.route_state(proposal.substrate_id).evidence_count == 1
    assert route.route_state(proposal.substrate_id).holdout_transfer > 0.0
    restored = AdaptiveNeuronNetwork.from_payload(
        network.to_payload(),
        generator=torch.Generator().manual_seed(123),
    )
    assert restored.cooperation_learner is not None
    assert restored.cooperation_learner.total_evidence == 1
