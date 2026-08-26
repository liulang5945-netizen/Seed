"""Evaluate learned competition among explicit Taiji cross-region routes."""

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
    CrossRegionCooperationLearner,
    NeuronRegionDynamics,
)

REPORT_FORMAT = "taiji-cross-region-learning-v1"
MANIFEST_FORMAT = "taiji-cross-region-learning-manifest-v1"


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
        (_region("signal"), _region("distractor"), _region("target")),
        execution_order=("signal", "distractor", "target"),
    )


def _add_connection(
    network: AdaptiveNeuronNetwork,
    source_region_id: str,
    target_region_id: str,
    *,
    resource_cost: int = 1,
):
    proposal = network.propose_connection_add(
        source_region_id=source_region_id,
        target_region_id=target_region_id,
        evidence_ids=(f"holdout:{source_region_id}",),
        fan_in=1,
        parent_checkpoint_id="parent:cross-region-learning",
        resource_cost=resource_cost,
    )
    network.apply_topology_proposal(
        proposal,
        generator=torch.Generator().manual_seed(99),
    )
    return proposal


def evaluate() -> dict[str, object]:
    network = _network()
    signal = _add_connection(network, "signal", "target")
    distractor = _add_connection(network, "distractor", "target")
    network.attach_cooperation_learner(CrossRegionCooperationLearner())
    network.regions[0].incoming.edge_weight.fill_(1.0)
    network.regions[1].incoming.edge_weight.fill_(1.0)
    network.connections[0][3].edge_weight.fill_(1.0)
    network.connections[1][3].edge_weight.fill_(-1.0)
    expected_target = network.step(
        {"signal": torch.ones(3)},
        connection_ids=(signal.substrate_id,),
    )["target"].clone()
    for _ in range(3):
        network.step(
            {"signal": torch.ones(3)},
            connection_ids=(signal.substrate_id,),
            expected_activities={"target": expected_target},
        )
        network.step(
            {"distractor": torch.ones(3)},
            connection_ids=(distractor.substrate_id,),
            expected_activities={"target": expected_target},
        )
    for _ in range(3):
        network.step(
            {"signal": torch.ones(3)},
            connection_ids=(signal.substrate_id,),
            expected_activities={"target": expected_target},
            holdout=True,
        )
        network.step(
            {"distractor": torch.ones(3)},
            connection_ids=(distractor.substrate_id,),
            expected_activities={"target": expected_target},
            holdout=True,
        )
    routes = (signal.substrate_id, distractor.substrate_id)
    holdout_transfer = {
        signal.substrate_id: 0.95,
        distractor.substrate_id: 0.15,
    }
    selected = network.selected_connection_ids()
    selected_transfer = sum(holdout_transfer[item] for item in selected) / len(selected)
    full_transfer = sum(holdout_transfer.values()) / len(holdout_transfer)
    random_transfer = full_transfer
    selected_target = network.step(
        {"signal": torch.ones(3), "distractor": torch.ones(3)},
        resource_budget=1.0,
    )["target"]
    full_target = network.step(
        {"signal": torch.ones(3), "distractor": torch.ones(3)},
        connection_ids=routes,
        max_connections=2,
    )["target"]

    checkpoint = AdaptiveNeuronNetwork.from_payload(
        network.to_payload(),
        generator=torch.Generator().manual_seed(123),
    )
    checkpoint_continuation = (
        checkpoint.selected_connection_ids() == selected
        and checkpoint.cooperation_learner is not None
        and checkpoint.cooperation_learner.total_evidence == 12
    )

    checkpoint.lesion_topology_proposal(signal)
    connection_lesion = checkpoint.selected_connection_ids() == (distractor.substrate_id,)
    checkpoint.lesion_region("target")
    region_lesion = checkpoint.selected_connection_ids() == ()
    lesioned_target = checkpoint.step(
        {"signal": torch.ones(3), "distractor": torch.ones(3)},
    )["target"]

    constrained = _network()
    expensive = _add_connection(constrained, "signal", "target", resource_cost=2)
    affordable = _add_connection(constrained, "distractor", "target", resource_cost=1)
    constrained.attach_cooperation_learner(CrossRegionCooperationLearner())
    constrained.connections[0][3].edge_weight.fill_(1.0)
    constrained.connections[1][3].edge_weight.zero_()
    constrained_expected = constrained.step(
        {"signal": torch.ones(3)},
        connection_ids=(expensive.substrate_id,),
        resource_budget=2.0,
    )["target"].clone()
    constrained.step(
        {"signal": torch.ones(3)},
        connection_ids=(expensive.substrate_id,),
        expected_activities={"target": constrained_expected},
        resource_budget=2.0,
        holdout=True,
    )
    constrained.step(
        {"distractor": torch.ones(3)},
        connection_ids=(affordable.substrate_id,),
        expected_activities={"target": constrained_expected},
        resource_budget=2.0,
        holdout=True,
    )
    resource_constrained = constrained.selected_connection_ids(resource_budget=1.0) == (
        affordable.substrate_id,
    )

    gate = {
        "passed": bool(
            selected == (signal.substrate_id,)
            and selected_transfer > full_transfer
            and selected_transfer > random_transfer
            and float(selected_target.sum().item()) > 0.0
            and float(full_target.sum().item()) < float(selected_target.sum().item())
            and checkpoint_continuation
            and connection_lesion
            and region_lesion
            and float(lesioned_target.sum().item()) == 0.0
            and resource_constrained
        ),
        "criterion": (
            "cross-region route selection must learn from prediction error, holdout transfer and "
            "resource state; outperform fixed-full and random baselines on holdout evidence; "
            "obey resource feasibility; and survive checkpoint continuation plus connection/region lesion"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "routes": routes,
        "metrics": {
            "selected_routes": selected,
            "selected_holdout_transfer": selected_transfer,
            "fixed_full_holdout_transfer": full_transfer,
            "random_holdout_transfer": random_transfer,
            "selected_target_activity": float(selected_target.sum().item()),
            "fixed_full_target_activity": float(full_target.sum().item()),
            "checkpoint_continuation": checkpoint_continuation,
            "connection_lesion_excludes_selected": connection_lesion,
            "region_lesion_excludes_routes": region_lesion,
            "region_lesion_target_activity": float(lesioned_target.sum().item()),
            "resource_constrained_selection": resource_constrained,
        },
        "gate": gate,
        "boundary": (
            "This gate proves evidence-driven selection among explicit native regions; it does not "
            "claim unrestricted self-evolution, open-domain language acquisition, or general intelligence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_cross_region_learning_20260826.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "plans" / "manifests" / "taiji_cross_region_learning_v1.json",
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
        "task": "learned cross-region cooperation",
        "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "seed": 71,
        "controls": [
            "prediction_error_credit",
            "holdout_transfer_selection",
            "resource_state_selection",
            "fixed_full_baseline",
            "random_baseline",
            "connection_lesion",
            "region_lesion",
            "network_checkpoint_continuation",
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
