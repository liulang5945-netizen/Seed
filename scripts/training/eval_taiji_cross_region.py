"""Evaluate explicit cross-region wiring and runtime ownership."""

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
    NeuronRegionDynamics,
    TaijiConfig,
    TSKV8Adapter,
)

REPORT_FORMAT = "taiji-cross-region-v1"
MANIFEST_FORMAT = "taiji-cross-region-manifest-v1"


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


def _proposal(network: AdaptiveNeuronNetwork):
    return network.propose_connection_add(
        source_region_id="source",
        target_region_id="target",
        evidence_ids=("holdout:cross-region",),
        fan_in=1,
        parent_checkpoint_id="parent:network",
    )


def evaluate() -> dict[str, object]:
    network = _network()
    proposal = _proposal(network)
    network.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(99),
    )
    connection = network.connections[0][3]
    connection.edge_weight.fill_(1.0)
    source = network.regions[0]
    source.incoming.edge_weight.fill_(1.0)
    active = network.step({"source": torch.ones(3)})
    driven = float(active["target"].sum().item())
    network.lesion_topology_proposal(proposal)
    silent = network.step({"source": torch.ones(3)})
    lesioned = float(silent["target"].sum().item())

    migrated_before = network.connections[0][3].pre_index.clone()
    migrated_weights = network.connections[0][3].edge_weight.clone()
    growth = source.propose_unit_add(
        unit_id="source.u2",
        evidence_ids=("holdout:source-growth",),
    )
    network.apply_neuron_proposal(
        growth,
        generator=torch.Generator().manual_seed(100),
    )
    migrated = network.connections[0][3]
    support_migrated = bool(
        migrated.in_features == 3
        and torch.equal(migrated.pre_index, migrated_before)
        and torch.equal(migrated.edge_weight, migrated_weights)
    )
    restored_network = AdaptiveNeuronNetwork.from_payload(
        network.to_payload(),
        generator=torch.Generator().manual_seed(123),
    )
    network_checkpoint = bool(
        restored_network.region_ids == network.region_ids
        and restored_network.connection_ids == network.connection_ids
        and restored_network.execution_order == network.execution_order
    )

    model = TSKV8Adapter(_config(budget=1), episode_id="cross-region-ledger")
    runtime_network = _network()
    model.attach_adaptive_neuron_network("cortex", runtime_network)
    runtime_proposal = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="target",
        evidence_ids=("runtime:cross-region-holdout",),
        fan_in=1,
    )
    accepted = model.commit_cross_region_connection("cortex", runtime_proposal)
    budget_after_accept = model.cognitive_snapshot().development.structural_budget
    restored_model = TSKV8Adapter.from_native_checkpoint(model.native_checkpoint())
    checkpoint_status = restored_model.topology_proposals[-1].status
    rollback = restored_model.rollback_cross_region_connection(runtime_proposal.proposal_id)
    budget_after_rollback = restored_model.cognitive_snapshot().development.structural_budget
    no_budget = TSKV8Adapter(_config(budget=0), episode_id="cross-region-no-budget")
    no_budget_network = _network()
    no_budget.attach_adaptive_neuron_network("cortex", no_budget_network)
    rejected_proposal = no_budget.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="target",
        evidence_ids=("runtime:cross-region-holdout",),
        fan_in=1,
    )
    rejected = no_budget.commit_cross_region_connection("cortex", rejected_proposal)
    gate = {
        "passed": bool(
            driven > 0.0
            and lesioned == 0.0
            and support_migrated
            and network_checkpoint
            and accepted
            and budget_after_accept == 0
            and checkpoint_status == "accepted"
            and rollback
            and budget_after_rollback == 1
            and not rejected
            and no_budget.topology_proposals[-1].status == "rejected"
        ),
        "criterion": (
            "cross-region wiring must be explicit, drive a downstream region from upstream "
            "activity, fail under connection lesion, migrate across region growth without "
            "redrawing old support, survive checkpoint continuation, and obey ledger budget/rollback"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "proposal": proposal.to_payload(),
        "metrics": {
            "driven_target_activity": driven,
            "lesioned_target_activity": lesioned,
            "support_migrated": support_migrated,
            "network_checkpoint": network_checkpoint,
            "runtime_accepted": accepted,
            "runtime_budget_after_accept": budget_after_accept,
            "runtime_checkpoint_status": checkpoint_status,
            "runtime_rollback": rollback,
            "runtime_budget_after_rollback": budget_after_rollback,
            "rejected_without_budget": not rejected,
        },
        "gate": gate,
        "boundary": (
            "This gate proves explicit wiring between two adaptive regions; it does not yet "
            "claim learned multi-region routing, open-domain self-evolution, or general intelligence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_cross_region_20260826.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "plans" / "manifests" / "taiji_cross_region_v1.json",
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
        "task": "explicit cross-region adaptive wiring",
        "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "seed": 71,
        "controls": [
            "upstream_drive",
            "connection_lesion",
            "region_growth_migration",
            "network_checkpoint",
            "budget_consumption",
            "native_checkpoint",
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
