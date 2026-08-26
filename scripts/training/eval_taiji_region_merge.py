"""Evaluate resource-governed region merge and external route aggregation."""

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

REPORT_FORMAT = "taiji-region-merge-v1"
MANIFEST_FORMAT = "taiji-region-merge-manifest-v1"


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
        (make("source"), make("bottleneck"), make("sink")),
        execution_order=("source", "bottleneck", "sink"),
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


def _attach_redundant_routes(model: TSKV8Adapter) -> tuple[str, str]:
    first = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="sink",
        evidence_ids=("merge:first",),
        fan_in=1,
    )
    second = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="bottleneck",
        target_region_id="sink",
        evidence_ids=("merge:second",),
        fan_in=1,
    )
    assert model.commit_cross_region_connection("cortex", first)
    assert model.commit_cross_region_connection("cortex", second)
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
    return first.proposal_id, second.proposal_id


def _holdout() -> tuple[tuple[dict[str, torch.Tensor], ...], tuple[dict[str, torch.Tensor], ...]]:
    inputs = (
        {"source": torch.ones(3), "bottleneck": torch.ones(3)},
        {"source": torch.ones(3), "bottleneck": torch.ones(3)},
    )
    expected = (
        {"source": torch.zeros(4)},
        {"source": torch.zeros(4)},
    )
    return inputs, expected


def evaluate() -> dict[str, object]:
    model = TSKV8Adapter(_config(budget=3), episode_id="region-merge")
    model.attach_adaptive_neuron_network("cortex", _network())
    first_add_id, second_add_id = _attach_redundant_routes(model)
    model.attach_structural_pruning_controller(_controller())
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
    assert proposal is not None
    holdout_inputs, expected_activities = _holdout()
    holdout_validated = model.validate_region_merge_holdout(
        network_id="cortex",
        proposal_id=proposal.proposal_id,
        holdout_inputs=holdout_inputs,
        expected_activities=expected_activities,
    )
    validated_proposal = next(
        item for item in model.topology_proposals if item.proposal_id == proposal.proposal_id
    )
    committed = model.commit_region_merge("cortex", proposal)
    merged_network = model.neuron_networks[0]
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_network = restored.neuron_networks[0]
    checkpoint_continuation = bool(
        restored_network.region_ids == ("source", "sink")
        and restored_network.connection_ids == ("connection:source->sink",)
        and restored_network.cooperation_learner is not None
        and restored_network.cooperation_learner.routes[0].evidence_count == 2
    )
    rollback_merge = restored.rollback_region_merge(proposal.proposal_id)
    rollback_second = restored.rollback_cross_region_connection(second_add_id)
    rollback_first = restored.rollback_cross_region_connection(first_add_id)

    internal = TSKV8Adapter(_config(budget=2), episode_id="region-merge-internal")
    internal.attach_adaptive_neuron_network("cortex", _network())
    internal_connection = internal.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="bottleneck",
        evidence_ids=("merge:internal",),
        fan_in=1,
    )
    assert internal.commit_cross_region_connection("cortex", internal_connection)
    internal_rejected = False
    try:
        internal.neuron_networks[0].propose_region_merge(
            region_ids=("source", "bottleneck"),
            evidence_ids=("merge:internal:reject",),
        )
    except ValueError as exc:
        internal_rejected = "internal" in str(exc)

    no_budget = TSKV8Adapter(_config(budget=2), episode_id="region-merge-no-budget")
    no_budget.attach_adaptive_neuron_network("cortex", _network())
    _attach_redundant_routes(no_budget)
    no_budget.attach_structural_pruning_controller(_controller())
    no_budget_proposals = [
        no_budget.propose_region_merge_from_redundancy(
            network_id="cortex",
            region_ids=("source", "bottleneck"),
            usage=0.05,
            resource_pressure=0.9,
            learning_gain=0.0,
            evidence_ids=(f"merge:no-budget:{index}",),
        )
        for index in range(1, 3)
    ]
    no_budget_proposal = no_budget_proposals[-1]
    assert no_budget_proposal is not None
    no_budget_inputs, no_budget_expected = _holdout()
    assert no_budget.validate_region_merge_holdout(
        network_id="cortex",
        proposal_id=no_budget_proposal.proposal_id,
        holdout_inputs=no_budget_inputs,
        expected_activities=no_budget_expected,
    )
    rejected = no_budget.commit_region_merge("cortex", no_budget_proposal)

    final_network = restored.neuron_networks[0]
    gate = {
        "passed": bool(
            first is None
            and holdout_validated
            and validated_proposal.validation_score >= 0.95
            and committed
            and merged_network.region_ids == ("source", "sink")
            and merged_network.execution_order == merged_network.region_ids
            and merged_network.regions[0].unit_ids
            == ("source.u0", "source.u1", "bottleneck.u0", "bottleneck.u1")
            and merged_network.connection_ids == ("connection:source->sink",)
            and checkpoint_continuation
            and rollback_merge
            and rollback_second
            and rollback_first
            and final_network.region_ids == ("source", "bottleneck", "sink")
            and final_network.connection_ids == ()
            and restored.cognitive_snapshot().development.structural_budget == 3
            and internal_rejected
            and not rejected
            and no_budget.neuron_networks[0].region_ids == ("source", "bottleneck", "sink")
        ),
        "criterion": (
            "persistent redundancy evidence must emit a region merge proposal; compatible unit "
            "identities and local state must be combined, external sparse connections and route "
            "learner evidence must aggregate explicitly, unseen holdout regression must stay within "
            "threshold, checkpoint and reverse rollback must work, internal connections and zero "
            "budget must fail closed"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "proposal": proposal.to_payload(),
        "metrics": {
            "required_redundancy_steps": 2,
            "proposal_emitted_after_steps": 2,
            "retained_region_id": "source",
            "absorbed_region_id": "bottleneck",
            "holdout_validation_score": validated_proposal.validation_score,
            "holdout_validated": holdout_validated,
            "region_ids_after_merge": list(merged_network.region_ids),
            "connection_ids_after_merge": list(merged_network.connection_ids),
            "merged_route_evidence_count": 2,
            "checkpoint_continuation": checkpoint_continuation,
            "rollback_merge": rollback_merge,
            "rollback_connections": rollback_second and rollback_first,
            "region_ids_after_rollback": list(final_network.region_ids),
            "internal_connection_rejected": internal_rejected,
            "rejected_without_budget": not rejected,
        },
        "gate": gate,
        "boundary": (
            "This gate proves controlled compatible-region merge with explicit external-route "
            "aggregation; it does not claim unrestricted self-evolution, automatic semantic "
            "invention, or general intelligence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_region_merge_20260826.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "plans" / "manifests" / "taiji_region_merge_v1.json",
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
        "task": "resource-governed compatible region merge",
        "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "seed": 71,
        "controls": [
            "persistent_redundancy_evidence",
            "compatible_unit_state_merge",
            "external_sparse_connection_aggregation",
            "route_learner_lineage",
            "unseen_holdout_regression_gate",
            "budget_validation",
            "checkpoint_continuation",
            "reverse_rollback",
            "internal_connection_fail_closed",
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
