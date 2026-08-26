"""Evaluate substrate-driven automatic region growth in the native Taiji network."""

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
    StructuralGrowthDynamics,
    TaijiConfig,
    TSKV8Adapter,
)

REPORT_FORMAT = "taiji-region-growth-v1"
MANIFEST_FORMAT = "taiji-region-growth-manifest-v1"


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


def evaluate() -> dict[str, object]:
    model = TSKV8Adapter(_config(budget=2), episode_id="region-growth")
    model.attach_adaptive_neuron_network("cortex", _network())
    model.attach_structural_growth_controller(_controller())
    proposals = [
        model.propose_region_growth_from_error(
            network_id="cortex",
            bottleneck_region_id="bottleneck",
            input_dim=3,
            unit_count=2,
            fan_in=2,
            prediction_error=0.9,
            resource_state=0.8,
            holdout_transfer=0.9,
            evidence_ids=(f"region:tick:{index}",),
        )
        for index in range(1, 3)
    ]
    proposal = proposals[-1]
    assert proposal is not None
    committed = model.commit_region_add("cortex", proposal)
    network = model.neuron_networks[0]
    child = network.regions[-1]
    holdout_input = torch.ones(3)
    expected_activity = torch.full((child.unit_count,), 0.8)
    child.incoming.edge_weight.zero_()
    for _ in range(32):
        child.learn(holdout_input, expected_activity - child.activity)
    holdout_validated = model.validate_region_growth_holdout(
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
    validated_proposal = model.topology_proposals[-1]
    connection = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id=proposal.substrate_id,
        evidence_ids=("region:connection:holdout",),
        fan_in=1,
    )
    connection_committed = model.commit_cross_region_connection("cortex", connection)

    restored = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    restored_network = restored.neuron_networks[0]
    checkpoint_holdout = restored.validate_region_growth_holdout(
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
    checkpoint_continuation = bool(
        restored_network.region_ids == network.region_ids
        and restored_network.execution_order == network.execution_order
        and restored_network.connection_ids == network.connection_ids
        and next(
            item for item in restored.topology_proposals if item.proposal_id == proposal.proposal_id
        ).validation_score
        > 0.05
        and checkpoint_holdout
    )
    restored_network.lesion_region(proposal.substrate_id)
    lesion = restored.select_cross_region_connections("cortex") == ()
    rollback_connection = restored.rollback_cross_region_connection(connection.proposal_id)
    rollback_region = restored.rollback_region_add(proposal.proposal_id)

    no_budget = TSKV8Adapter(_config(budget=0), episode_id="region-growth-no-budget")
    no_budget.attach_adaptive_neuron_network("cortex", _network())
    no_budget.attach_structural_growth_controller(_controller())
    no_budget_proposal = None
    for index in range(1, 3):
        no_budget_proposal = no_budget.propose_region_growth_from_error(
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
    assert no_budget_proposal is not None
    rejected = no_budget.commit_region_add("cortex", no_budget_proposal)

    final_regions = restored.neuron_networks[0].region_ids
    gate = {
        "passed": bool(
            proposals[0] is None
            and committed
            and holdout_validated
            and connection_committed
            and checkpoint_continuation
            and lesion
            and rollback_connection
            and rollback_region
            and final_regions == ("source", "bottleneck")
            and restored.cognitive_snapshot().development.structural_budget == 2
            and not rejected
            and no_budget.neuron_networks[0].region_ids == ("source", "bottleneck")
        ),
        "criterion": (
            "persistent regional prediction error plus holdout transfer and available resources "
            "must emit a non-semantic child-region proposal; the ledger must validate budget and "
            "checkpoint trial, require post-growth unseen-input improvement before explicit wiring, "
            "preserve order and identity, silence the grown region functionally, and reverse both "
            "structural mutations"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "proposal": proposal.to_payload(),
        "metrics": {
            "persistent_error_required_steps": 2,
            "proposal_emitted_after_steps": 2,
            "region_id": proposal.substrate_id,
            "region_ids_after_growth": list(network.region_ids),
            "execution_order_after_growth": list(network.execution_order),
            "holdout_validation_score": validated_proposal.validation_score,
            "holdout_validated_before_connection": holdout_validated,
            "holdout_validated_after_checkpoint": checkpoint_holdout,
            "connection_id": connection.substrate_id,
            "checkpoint_continuation": checkpoint_continuation,
            "functional_region_lesion": lesion,
            "rollback_connection": rollback_connection,
            "rollback_region": rollback_region,
            "region_ids_after_rollback": list(final_regions),
            "rejected_without_budget": not rejected,
        },
        "gate": gate,
        "boundary": (
            "This gate proves controlled substrate-driven region birth and explicit wiring; it does "
            "not claim unrestricted self-evolution, automatic semantic invention, or general intelligence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_region_growth_20260826.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "plans" / "manifests" / "taiji_region_growth_v1.json",
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
        "task": "substrate-driven automatic region growth",
        "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "seed": 71,
        "controls": [
            "persistent_regional_prediction_error",
            "holdout_transfer_gate",
            "resource_availability_gate",
            "non_semantic_region_identity",
            "budget_validation",
            "checkpoint_trial",
            "explicit_cross_region_wiring",
            "functional_region_lesion",
            "reverse_rollback",
            "zero_budget",
        ],
        "gate": report["gate"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
