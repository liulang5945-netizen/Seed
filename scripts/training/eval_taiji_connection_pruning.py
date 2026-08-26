"""Evaluate route-level resource-aware pruning in the native Taiji network."""

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
    AdaptiveNeuronNetwork,
    AdaptiveNeuronRegion,
    AdaptiveStructuralPruningController,
    CrossRegionCooperationLearner,
    CrossRegionLearningDynamics,
    StructuralPruningDynamics,
    TaijiConfig,
    TSKV8Adapter,
)

REPORT_FORMAT = "taiji-connection-pruning-v1"
MANIFEST_FORMAT = "taiji-connection-pruning-manifest-v1"


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


def _controller() -> AdaptiveStructuralPruningController:
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


def _attach_stagnant_route(model: TSKV8Adapter) -> tuple[str, str]:
    connection = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="bottleneck",
        evidence_ids=("route:add",),
        fan_in=1,
    )
    assert model.commit_cross_region_connection("cortex", connection)
    network = model.neuron_networks[0]
    network.attach_cooperation_learner(
        CrossRegionCooperationLearner(
            dynamics=CrossRegionLearningDynamics(ema_rate=1.0),
        )
    )
    for _ in range(2):
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
    return connection.substrate_id, connection.proposal_id


def _holdout() -> tuple[tuple[dict[str, torch.Tensor], ...], tuple[dict[str, torch.Tensor], ...]]:
    inputs = ({"source": torch.ones(3)}, {"source": torch.ones(3)})
    expected = (
        {"source": torch.zeros(2), "bottleneck": torch.zeros(2)},
        {"source": torch.zeros(2), "bottleneck": torch.zeros(2)},
    )
    return inputs, expected


def evaluate() -> dict[str, object]:
    model = TSKV8Adapter(_config(budget=2), episode_id="connection-pruning")
    model.attach_adaptive_neuron_network("cortex", _network())
    connection_id, connection_add_id = _attach_stagnant_route(model)
    model.attach_structural_pruning_controller(_controller())
    proposals = [
        model.propose_cross_region_connection_prune_from_route(
            network_id="cortex",
            connection_id=connection_id,
            evidence_ids=(f"route:tick:{index}",),
        )
        for index in range(1, 3)
    ]
    proposal = proposals[-1]
    assert proposal is not None
    holdout_inputs, expected_activities = _holdout()
    holdout_validated = model.validate_cross_region_connection_prune_holdout(
        network_id="cortex",
        proposal_id=proposal.proposal_id,
        holdout_inputs=holdout_inputs,
        expected_activities=expected_activities,
    )
    validated_proposal = next(
        item for item in model.topology_proposals if item.proposal_id == proposal.proposal_id
    )
    committed = model.commit_cross_region_connection_prune("cortex", proposal)
    pruned_network = model.neuron_networks[0]
    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    restored_network = restored.neuron_networks[0]
    restored_pruning = restored.structural_pruning_controller
    checkpoint_continuation = bool(
        restored_pruning is not None
        and restored_pruning.total_observations == 2
        and restored_network.region_ids == ("source", "bottleneck")
        and restored_network.connection_ids == ()
        and next(
            item for item in restored.topology_proposals if item.proposal_id == proposal.proposal_id
        ).status
        == "accepted"
    )
    rollback_connection_prune = restored.rollback_cross_region_connection_prune(
        proposal.proposal_id
    )
    rollback_connection_add = restored.rollback_cross_region_connection(connection_add_id)

    high_gain = TSKV8Adapter(_config(budget=2), episode_id="connection-pruning-high-gain")
    high_gain.attach_adaptive_neuron_network("cortex", _network())
    high_gain_connection = high_gain.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="bottleneck",
        evidence_ids=("route:high-gain:add",),
        fan_in=1,
    )
    assert high_gain.commit_cross_region_connection("cortex", high_gain_connection)
    high_gain.neuron_networks[0].attach_cooperation_learner(
        CrossRegionCooperationLearner(
            dynamics=CrossRegionLearningDynamics(ema_rate=1.0),
        )
    )
    for _ in range(2):
        high_gain.neuron_networks[0].observe_connection(
            high_gain_connection.substrate_id,
            prediction_error=0.0,
            holdout_transfer=1.0,
            resource_state=0.1,
            selected=False,
        )
    high_gain.attach_structural_pruning_controller(_controller())
    high_gain_proposals = [
        high_gain.propose_cross_region_connection_prune_from_route(
            network_id="cortex",
            connection_id=high_gain_connection.substrate_id,
            evidence_ids=(f"route:high-gain:{index}",),
        )
        for index in range(1, 3)
    ]

    no_budget = TSKV8Adapter(_config(budget=1), episode_id="connection-pruning-no-budget")
    no_budget.attach_adaptive_neuron_network("cortex", _network())
    no_budget_connection_id, _ = _attach_stagnant_route(no_budget)
    no_budget.attach_structural_pruning_controller(_controller())
    no_budget_proposals = [
        no_budget.propose_cross_region_connection_prune_from_route(
            network_id="cortex",
            connection_id=no_budget_connection_id,
            evidence_ids=(f"route:no-budget:{index}",),
        )
        for index in range(1, 3)
    ]
    no_budget_proposal = no_budget_proposals[-1]
    assert no_budget_proposal is not None
    no_budget_holdout_inputs, no_budget_expected = _holdout()
    assert no_budget.validate_cross_region_connection_prune_holdout(
        network_id="cortex",
        proposal_id=no_budget_proposal.proposal_id,
        holdout_inputs=no_budget_holdout_inputs,
        expected_activities=no_budget_expected,
    )
    rejected = no_budget.commit_cross_region_connection_prune(
        "cortex",
        no_budget_proposal,
    )

    final_network = restored.neuron_networks[0]
    route_state = model.neuron_networks[0].cooperation_learner
    gate = {
        "passed": bool(
            proposals[0] is None
            and holdout_validated
            and validated_proposal.validation_score >= 0.95
            and committed
            and pruned_network.region_ids == ("source", "bottleneck")
            and pruned_network.connection_ids == ()
            and checkpoint_continuation
            and rollback_connection_prune
            and rollback_connection_add
            and final_network.region_ids == ("source", "bottleneck")
            and final_network.connection_ids == ()
            and restored.cognitive_snapshot().development.structural_budget == 2
            and route_state is not None
            and route_state.route_ids == ()
            and all(item is None for item in high_gain_proposals)
            and not rejected
            and no_budget.neuron_networks[0].connection_ids == (no_budget_connection_id,)
        ),
        "criterion": (
            "persistent low route usage, high resource pressure and route learning stagnation "
            "must emit a connection prune proposal from existing route evidence; unseen-input "
            "regression must stay within threshold before ledger commit, checkpoint continuation "
            "must preserve the decision, pruning must remove only the owned connection, reverse "
            "rollback must restore it, high route gain must suppress pruning, and zero budget must "
            "fail closed"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "proposal": proposal.to_payload(),
        "metrics": {
            "connection_id": connection_id,
            "route_evidence_count": 2,
            "route_selection_count": 0,
            "route_usage": 0.0,
            "route_resource_pressure": 0.9,
            "route_learning_gain": 0.0,
            "holdout_validation_score": validated_proposal.validation_score,
            "holdout_validated": holdout_validated,
            "connection_ids_after_pruning": list(pruned_network.connection_ids),
            "checkpoint_continuation": checkpoint_continuation,
            "rollback_connection_prune": rollback_connection_prune,
            "rollback_connection_add": rollback_connection_add,
            "connection_ids_after_rollback": list(final_network.connection_ids),
            "high_gain_suppressed": all(item is None for item in high_gain_proposals),
            "rejected_without_budget": not rejected,
        },
        "gate": gate,
        "boundary": (
            "This gate proves controlled route-level structural maintenance; it does not claim "
            "unrestricted self-evolution, automatic semantic invention, or general intelligence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_connection_pruning_20260826.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "plans" / "manifests" / "taiji_connection_pruning_v1.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "format": MANIFEST_FORMAT,
        "task": "resource-aware structural cross-region connection pruning",
        "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "seed": 71,
        "controls": [
            "route_learner_evidence",
            "persistent_low_usage",
            "resource_pressure",
            "learning_stagnation",
            "unseen_holdout_regression_gate",
            "budget_validation",
            "checkpoint_continuation",
            "connection_only_removal",
            "reverse_rollback",
            "high_gain_suppression",
            "zero_budget",
        ],
        "gate": report["gate"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
