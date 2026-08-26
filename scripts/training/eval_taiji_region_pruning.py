"""Evaluate resource-aware structural region pruning in the native Taiji network."""

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
    StructuralPruningDynamics,
    TaijiConfig,
    TSKV8Adapter,
)

REPORT_FORMAT = "taiji-region-pruning-v1"
MANIFEST_FORMAT = "taiji-region-pruning-manifest-v1"


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


def evaluate() -> dict[str, object]:
    model = TSKV8Adapter(_config(budget=2), episode_id="region-pruning")
    model.attach_adaptive_neuron_network("cortex", _network())
    connection = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="bottleneck",
        evidence_ids=("prune:connection",),
        fan_in=1,
    )
    connection_committed = model.commit_cross_region_connection("cortex", connection)
    model.attach_structural_pruning_controller(_controller())

    proposals = [
        model.propose_region_prune_from_underuse(
            network_id="cortex",
            region_id="bottleneck",
            usage=0.05,
            resource_pressure=0.9,
            learning_gain=0.0,
            evidence_ids=(f"prune:tick:{index}",),
        )
        for index in range(1, 3)
    ]
    proposal = proposals[-1]
    assert proposal is not None
    holdout_inputs = ({"source": torch.ones(3)}, {"source": torch.ones(3)})
    expected_activities = (
        {"source": torch.zeros(2)},
        {"source": torch.zeros(2)},
    )
    holdout_validated = model.validate_region_prune_holdout(
        network_id="cortex",
        proposal_id=proposal.proposal_id,
        holdout_inputs=holdout_inputs,
        expected_activities=expected_activities,
    )
    validated_proposal = next(
        item for item in model.topology_proposals if item.proposal_id == proposal.proposal_id
    )
    committed = model.commit_region_prune("cortex", proposal)
    pruned_network = model.neuron_networks[0]
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_network = restored.neuron_networks[0]
    checkpoint_continuation = bool(
        restored.structural_pruning_controller is not None
        and restored.structural_pruning_controller.total_observations == 2
        and restored_network.region_ids == ("source",)
        and restored_network.connection_ids == ()
        and next(
            item for item in restored.topology_proposals if item.proposal_id == proposal.proposal_id
        ).status
        == "accepted"
    )
    rollback_prune = restored.rollback_region_prune(proposal.proposal_id)
    rollback_connection = restored.rollback_cross_region_connection(connection.proposal_id)

    high_gain = TSKV8Adapter(_config(budget=2), episode_id="region-pruning-high-gain")
    high_gain.attach_adaptive_neuron_network("cortex", _network())
    high_gain.attach_structural_pruning_controller(_controller())
    high_gain_proposals = [
        high_gain.propose_region_prune_from_underuse(
            network_id="cortex",
            region_id="bottleneck",
            usage=0.05,
            resource_pressure=0.9,
            learning_gain=1.0,
            evidence_ids=(f"prune:high-gain:{index}",),
        )
        for index in range(1, 3)
    ]

    no_budget = TSKV8Adapter(_config(budget=0), episode_id="region-pruning-no-budget")
    no_budget.attach_adaptive_neuron_network("cortex", _network())
    no_budget.attach_structural_pruning_controller(_controller())
    no_budget_proposals = [
        no_budget.propose_region_prune_from_underuse(
            network_id="cortex",
            region_id="bottleneck",
            usage=0.05,
            resource_pressure=0.9,
            learning_gain=0.0,
            evidence_ids=(f"prune:no-budget:{index}",),
        )
        for index in range(1, 3)
    ]
    no_budget_proposal = no_budget_proposals[-1]
    assert no_budget_proposal is not None
    rejected = no_budget.commit_region_prune("cortex", no_budget_proposal)

    final_network = restored.neuron_networks[0]
    pruning_state = restored.structural_pruning_controller.regions[0]
    gate = {
        "passed": bool(
            proposals[0] is None
            and connection_committed
            and holdout_validated
            and validated_proposal.validation_score >= 0.95
            and committed
            and pruned_network.region_ids == ("source",)
            and pruned_network.connection_ids == ()
            and checkpoint_continuation
            and rollback_prune
            and rollback_connection
            and final_network.region_ids == ("source", "bottleneck")
            and final_network.connection_ids == ()
            and restored.cognitive_snapshot().development.structural_budget == 2
            and all(item is None for item in high_gain_proposals)
            and not rejected
            and no_budget.neuron_networks[0].region_ids == ("source", "bottleneck")
        ),
        "criterion": (
            "persistent low usage, high resource pressure and long-term learning stagnation "
            "must emit a non-semantic region prune proposal; unseen-input regression must stay "
            "within threshold before ledger commit, checkpoint continuation must preserve the "
            "decision, pruning must remove owned connections, reverse rollback must restore the "
            "parent topology, high learning gain must suppress pruning, and zero budget must "
            "fail closed"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "proposal": proposal.to_payload(),
        "metrics": {
            "required_underuse_steps": 2,
            "proposal_emitted_after_steps": 2,
            "region_id": proposal.substrate_id,
            "usage_ema": pruning_state.usage_ema,
            "resource_pressure_ema": pruning_state.resource_pressure_ema,
            "learning_gain_ema": pruning_state.learning_gain_ema,
            "holdout_validation_score": validated_proposal.validation_score,
            "holdout_validated": holdout_validated,
            "region_ids_after_pruning": list(pruned_network.region_ids),
            "connection_ids_after_pruning": list(pruned_network.connection_ids),
            "checkpoint_continuation": checkpoint_continuation,
            "rollback_region": rollback_prune,
            "rollback_connection": rollback_connection,
            "region_ids_after_rollback": list(final_network.region_ids),
            "connection_ids_after_rollback": list(final_network.connection_ids),
            "high_gain_suppressed": all(item is None for item in high_gain_proposals),
            "rejected_without_budget": not rejected,
        },
        "gate": gate,
        "boundary": (
            "This gate proves controlled resource-aware region retention and pruning; it does "
            "not claim unrestricted self-evolution, automatic semantic invention, or general intelligence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_region_pruning_20260826.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "plans" / "manifests" / "taiji_region_pruning_v1.json",
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
        "task": "resource-aware structural region pruning",
        "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "seed": 71,
        "controls": [
            "persistent_low_usage",
            "resource_pressure",
            "learning_stagnation",
            "unseen_holdout_regression_gate",
            "budget_validation",
            "checkpoint_continuation",
            "owned_connection_removal",
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
