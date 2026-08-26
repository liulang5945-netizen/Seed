"""Evaluate identity-preserving isolated-region split in the native Taiji network."""

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
    AdaptiveStructuralGrowthController,
    CrossRegionCooperationLearner,
    CrossRegionLearningDynamics,
    StructuralGrowthDynamics,
    TaijiConfig,
    TSKV8Adapter,
)

REPORT_FORMAT = "taiji-region-split-v1"
MANIFEST_FORMAT = "taiji-region-split-manifest-v1"


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
            maximum_restructure_holdout_regression=0.05,
            required_error_steps=2,
        )
    )


def evaluate() -> dict[str, object]:
    model = TSKV8Adapter(_config(budget=2), episode_id="region-split")
    model.attach_adaptive_neuron_network("cortex", _network())
    model.attach_structural_growth_controller(_controller())
    network = model.neuron_networks[0]
    network.regions[1].incoming.edge_weight.zero_()
    network.regions[1].activity.zero_()
    proposals = [
        model.propose_region_split_from_error(
            network_id="cortex",
            region_id="bottleneck",
            first_unit_count=1,
            prediction_error=0.9,
            resource_state=0.8,
            holdout_transfer=0.9,
            evidence_ids=(f"split:tick:{index}",),
        )
        for index in range(1, 3)
    ]
    proposal = proposals[-1]
    assert proposal is not None
    holdout_inputs = ({"bottleneck": torch.ones(3)}, {"bottleneck": torch.ones(3)})
    expected_activities = (
        {"bottleneck": torch.zeros(2)},
        {"bottleneck": torch.zeros(2)},
    )
    holdout_validated = model.validate_region_split_holdout(
        network_id="cortex",
        proposal_id=proposal.proposal_id,
        holdout_inputs=holdout_inputs,
        expected_activities=expected_activities,
    )
    validated_proposal = next(
        item for item in model.topology_proposals if item.proposal_id == proposal.proposal_id
    )
    committed = model.commit_region_split("cortex", proposal)
    split_network = model.neuron_networks[0]
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_network = restored.neuron_networks[0]
    checkpoint_continuation = bool(
        restored_network.region_ids == ("source", "bottleneck", "bottleneck.split.1")
        and restored_network.execution_order == restored_network.region_ids
        and next(
            item for item in restored.topology_proposals if item.proposal_id == proposal.proposal_id
        ).status
        == "accepted"
    )
    rollback = restored.rollback_region_split(proposal.proposal_id)

    connected = TSKV8Adapter(_config(budget=2), episode_id="region-split-connected")
    connected.attach_adaptive_neuron_network("cortex", _network())
    connection = connected.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="bottleneck",
        evidence_ids=("split:connected",),
        fan_in=1,
    )
    assert connected.commit_cross_region_connection("cortex", connection)
    connected_network = connected.neuron_networks[0]
    connected_network.attach_cooperation_learner(
        CrossRegionCooperationLearner(
            dynamics=CrossRegionLearningDynamics(ema_rate=1.0),
        )
    )
    connected_network.observe_connection(
        connection.substrate_id,
        prediction_error=0.5,
        holdout_transfer=0.5,
        resource_state=0.8,
        selected=True,
    )
    for region in connected_network.regions:
        region.incoming.edge_weight.zero_()
        region.activity.zero_()
    connected_network.connections[0][3].edge_weight.zero_()
    connected.attach_structural_growth_controller(_controller())
    connected_proposals = [
        connected.propose_region_split_from_error(
            network_id="cortex",
            region_id="bottleneck",
            first_unit_count=1,
            prediction_error=0.9,
            resource_state=0.8,
            holdout_transfer=0.9,
            evidence_ids=(f"split:connected:{index}",),
        )
        for index in range(1, 3)
    ]
    connected_proposal = connected_proposals[-1]
    assert connected_proposal is not None
    connected_holdout = connected.validate_region_split_holdout(
        network_id="cortex",
        proposal_id=connected_proposal.proposal_id,
        holdout_inputs=({"source": torch.ones(3), "bottleneck": torch.zeros(3)},),
        expected_activities=({"bottleneck": torch.zeros(2)},),
    )
    connected_committed = connected.commit_region_split("cortex", connected_proposal)
    connected_after = connected.neuron_networks[0]
    connected_migrated = bool(
        connected_holdout
        and connected_committed
        and connected_after.connection_ids
        == (
            "connection:source->bottleneck",
            "connection:source->bottleneck.split.1",
        )
        and connected_after.cooperation_learner is not None
        and connected_after.cooperation_learner.route_ids == connected_after.connection_ids
        and all(
            route.evidence_count == 1
            for route in connected_after.cooperation_learner.routes
        )
        and connected.rollback_region_split(connected_proposal.proposal_id)
        and connected.neuron_networks[0].connection_ids == (connection.substrate_id,)
    )

    no_budget = TSKV8Adapter(_config(budget=0), episode_id="region-split-no-budget")
    no_budget.attach_adaptive_neuron_network("cortex", _network())
    no_budget.attach_structural_growth_controller(_controller())
    no_budget_proposals = [
        no_budget.propose_region_split_from_error(
            network_id="cortex",
            region_id="bottleneck",
            first_unit_count=1,
            prediction_error=0.9,
            resource_state=0.8,
            holdout_transfer=0.9,
            evidence_ids=(f"split:no-budget:{index}",),
        )
        for index in range(1, 3)
    ]
    no_budget_proposal = no_budget_proposals[-1]
    assert no_budget_proposal is not None
    assert no_budget.validate_region_split_holdout(
        network_id="cortex",
        proposal_id=no_budget_proposal.proposal_id,
        holdout_inputs=holdout_inputs,
        expected_activities=expected_activities,
    )
    rejected = no_budget.commit_region_split("cortex", no_budget_proposal)

    final_network = restored.neuron_networks[0]
    gate = {
        "passed": bool(
            proposals[0] is None
            and holdout_validated
            and validated_proposal.validation_score >= 0.95
            and committed
            and split_network.region_ids == ("source", "bottleneck", "bottleneck.split.1")
            and split_network.execution_order == split_network.region_ids
            and split_network.regions[1].unit_ids == ("bottleneck.u0",)
            and split_network.regions[2].unit_ids == ("bottleneck.u1",)
            and checkpoint_continuation
            and rollback
            and final_network.region_ids == ("source", "bottleneck")
            and final_network.regions[1].unit_ids == ("bottleneck.u0", "bottleneck.u1")
            and restored.cognitive_snapshot().development.structural_budget == 2
            and connected_migrated
            and not rejected
            and no_budget.neuron_networks[0].region_ids == ("source", "bottleneck")
        ),
        "criterion": (
            "persistent substrate pressure must emit a region split proposal that preserves the "
            "parent region identity, partitions unit identities and local state, passes unseen "
            "holdout validation, survives checkpoint continuation, consumes the ledger budget, "
            "and reverses cleanly; connected regions must migrate their routes explicitly and "
            "zero-budget splits must fail closed"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "proposal": proposal.to_payload(),
        "metrics": {
            "required_error_steps": 2,
            "proposal_emitted_after_steps": 2,
            "parent_region_id": "bottleneck",
            "child_region_id": "bottleneck.split.1",
            "holdout_validation_score": validated_proposal.validation_score,
            "holdout_validated": holdout_validated,
            "region_ids_after_split": list(split_network.region_ids),
            "execution_order_after_split": list(split_network.execution_order),
            "checkpoint_continuation": checkpoint_continuation,
            "rollback": rollback,
            "region_ids_after_rollback": list(final_network.region_ids),
            "connected_split_migrated": connected_migrated,
            "rejected_without_budget": not rejected,
        },
        "gate": gate,
        "boundary": (
            "This gate proves identity-preserving split for isolated and connected regions with "
            "explicit sparse connection migration; it makes no claim of unrestricted "
            "self-evolution or general intelligence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_region_split_20260826.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "plans" / "manifests" / "taiji_region_split_v1.json",
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
        "task": "identity-preserving region split with connection migration",
        "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "seed": 71,
        "controls": [
            "persistent_substrate_error",
            "identity_preserving_unit_partition",
            "local_state_migration",
            "unseen_holdout_regression_gate",
            "budget_validation",
            "checkpoint_continuation",
            "reverse_rollback",
            "connected_connection_migration",
            "route_learner_lineage",
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
