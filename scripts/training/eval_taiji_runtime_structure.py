"""Evaluate structural evidence emitted by real native network runtime ticks."""

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
    AdaptiveStructuralPruningController,
    CrossRegionCooperationLearner,
    StructuralGrowthDynamics,
    StructuralPruningDynamics,
    TaijiConfig,
    TSKV8Adapter,
)

REPORT_FORMAT = "taiji-runtime-structure-v1"
MANIFEST_FORMAT = "taiji-runtime-structure-manifest-v1"


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
        (make("source"), make("target")),
        execution_order=("source", "target"),
    )


def _growth_controller() -> AdaptiveStructuralGrowthController:
    return AdaptiveStructuralGrowthController(
        dynamics=StructuralGrowthDynamics(
            ema_rate=1.0,
            error_threshold=0.0,
            holdout_transfer_threshold=0.0,
            minimum_resource_state=0.0,
            required_error_steps=1,
        )
    )


def _pruning_controller() -> AdaptiveStructuralPruningController:
    return AdaptiveStructuralPruningController(
        dynamics=StructuralPruningDynamics(ema_rate=1.0)
    )


def evaluate() -> dict[str, object]:
    model = TSKV8Adapter(_config(budget=1), episode_id="runtime-structure")
    model.attach_adaptive_neuron_network("cortex", _network())
    route = model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="target",
        evidence_ids=("runtime:route:add",),
        fan_in=1,
    )
    assert model.commit_cross_region_connection("cortex", route)
    model.attach_cross_region_cooperation("cortex", CrossRegionCooperationLearner())
    model.attach_structural_growth_controller(_growth_controller())
    model.attach_structural_pruning_controller(_pruning_controller())

    first = model.step_cross_region_network(
        "cortex",
        {"source": torch.ones(3)},
    )
    model.step_cross_region_network(
        "cortex",
        {"source": torch.ones(3)},
        expected_activities=first,
        holdout=True,
    )
    before_checkpoint = tuple(model.structural_runtime_observations)
    before_topology = model.neuron_networks[0].region_ids, model.neuron_networks[0].connection_ids
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored.step_cross_region_network(
        "cortex",
        {"source": torch.ones(3)},
        expected_activities=first,
        holdout=True,
    )
    restored_network = restored.neuron_networks[0]
    checkpoint_continuation = bool(
        restored.structural_runtime_observations[:4] == before_checkpoint
        and restored.structural_runtime_observations[-1].tick == 3
        and restored.structural_growth_controller is not None
        and restored.structural_growth_controller.total_observations == 4
        and restored.structural_pruning_controller is not None
        and restored.structural_pruning_controller.total_observations == 6
    )
    route_state = restored_network.cooperation_learner.route_state(route.substrate_id)

    gate = {
        "passed": bool(
            len(before_checkpoint) == 4
            and before_checkpoint[0].prediction_error is None
            and all(item.prediction_error is not None for item in before_checkpoint[2:])
            and model.structural_growth_controller is not None
            and model.structural_growth_controller.total_observations == 2
            and model.structural_pruning_controller is not None
            and model.structural_pruning_controller.total_observations == 4
            and route_state.evidence_count == 2
            and before_topology == (restored_network.region_ids, restored_network.connection_ids)
            and checkpoint_continuation
        ),
        "criterion": (
            "real native network ticks must emit checkpointable activity, prediction-error, "
            "learning-gain and resource observations; attached structural organs and route "
            "credit must continue across checkpoint, while topology remains unchanged until "
            "a separate holdout/budget/trial/rollback transaction commits it"
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "runtime_ticks": 3,
            "observations_before_checkpoint": len(before_checkpoint),
            "observations_after_checkpoint": len(restored.structural_runtime_observations),
            "growth_observations": model.structural_growth_controller.total_observations,
            "pruning_observations": model.structural_pruning_controller.total_observations,
            "route_evidence_count": route_state.evidence_count,
            "checkpoint_continuation": checkpoint_continuation,
            "topology_unchanged": before_topology
            == (restored_network.region_ids, restored_network.connection_ids),
        },
        "gate": gate,
        "boundary": (
            "This gate proves runtime evidence ownership and checkpoint continuation. It does not "
            "claim that a tick may bypass holdout validation, resource budget, trial checkpoint "
            "or reverse rollback to mutate live topology."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_runtime_structure_20260826.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "plans" / "manifests" / "taiji_runtime_structure_v1.json",
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
        "task": "runtime-owned structural evidence",
        "report": str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "seed": 71,
        "controls": [
            "real_network_tick",
            "activity_observation",
            "prediction_error_observation",
            "resource_observation",
            "growth_controller_runtime_ownership",
            "pruning_controller_runtime_ownership",
            "route_credit",
            "checkpoint_continuation",
            "topology_no_implicit_mutation",
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
