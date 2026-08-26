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
    StructuralProposalCandidate,
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


def _region() -> AdaptiveNeuronRegion:
    return AdaptiveNeuronRegion(
        region_id="adaptive.cortex",
        input_dim=5,
        unit_ids=("u0", "u1"),
        fan_in=2,
        generator=torch.Generator().manual_seed(7),
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
    model = TSKV8Adapter(_config(budget=2), episode_id="runtime-structure")
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
    candidate = model.structural_proposal_candidates[0]
    materialized = model.materialize_structural_candidate(candidate.candidate_id)
    assert materialized is not None
    assert materialized.status == "pending"
    holdout_validated = model.validate_structural_candidate_holdout(
        candidate.candidate_id,
        holdout_inputs=({"source": torch.ones(3)},),
        expected_activities=(first,),
    )
    materialized = next(
        item for item in model.topology_proposals if item.proposal_id == materialized.proposal_id
    )
    committed = model.commit_structural_candidate(candidate.candidate_id)
    topology_after_commit = (
        model.neuron_networks[0].region_ids,
        model.neuron_networks[0].connection_ids,
    )
    rolled_back = model.rollback_structural_candidate(candidate.candidate_id)
    materialized = next(
        item for item in model.topology_proposals if item.proposal_id == materialized.proposal_id
    )
    remaining = model.structural_proposal_candidates[0]
    cycle_results = model.run_structural_maintenance_cycle(
        candidate_ids=(remaining.candidate_id,),
        holdout_inputs_by_candidate={
            remaining.candidate_id: ({"target": torch.ones(3)},),
        },
        expected_activities_by_candidate={
            remaining.candidate_id: (first,),
        },
    )
    cycle_committed = cycle_results[0].status == "committed"
    cycle_topology = (
        model.neuron_networks[0].region_ids,
        model.neuron_networks[0].connection_ids,
    )
    cycle_rolled_back = model.rollback_structural_candidate(remaining.candidate_id)
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
    restored_candidate = restored.materialize_structural_candidate(candidate.candidate_id)
    checkpoint_checks = {
        "observations": restored.structural_runtime_observations[:4] == before_checkpoint,
        "tick": restored.structural_runtime_observations[-1].tick == 3,
        "growth": (
            restored.structural_growth_controller is not None
            and restored.structural_growth_controller.total_observations == 4
        ),
        "pruning": (
            restored.structural_pruning_controller is not None
            and restored.structural_pruning_controller.total_observations == 11
        ),
        "candidate": restored_candidate == materialized,
        "maintenance_results": len(restored.structural_maintenance_results) == 1,
    }
    checkpoint_continuation = bool(
        all(checkpoint_checks.values())
    )
    route_state = restored_network.cooperation_learner.route_state(route.substrate_id)

    direct_model = TSKV8Adapter(_config(budget=1), episode_id="runtime-standalone-neuron")
    direct_region = _region()
    direct_model.attach_adaptive_neuron_region(direct_region)
    direct_model.attach_structural_growth_controller(_growth_controller())
    direct_first = direct_model.step_adaptive_neuron_region(
        direct_region.region_id,
        torch.ones(5),
    )
    direct_model.step_adaptive_neuron_region(
        direct_region.region_id,
        torch.ones(5),
        expected_activity=direct_first,
        holdout=True,
    )
    direct_candidate = direct_model.structural_proposal_candidates[0]
    direct_proposal = direct_model.materialize_structural_candidate(
        direct_candidate.candidate_id
    )
    assert direct_proposal is not None
    direct_trial = AdaptiveNeuronRegion.from_payload(
        direct_region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    direct_trial.apply_topology_proposal(
        direct_proposal,
        generator=torch.Generator().manual_seed(0),
    )
    direct_holdout_input = torch.zeros(5)
    direct_holdout_input[direct_trial.incoming.pre_index[-1]] = torch.sign(
        direct_trial.incoming.edge_weight[-1]
    )
    direct_expected = direct_trial.step(direct_holdout_input)
    direct_holdout_validated = direct_model.validate_structural_candidate_holdout(
        direct_candidate.candidate_id,
        holdout_inputs=(direct_holdout_input,),
        expected_activities=(direct_expected,),
    )
    direct_committed = direct_model.commit_structural_candidate(
        direct_candidate.candidate_id
    )
    direct_checkpoint_model = TSKV8Adapter.from_native_checkpoint(
        direct_model.native_checkpoint()
    )
    direct_checkpoint = bool(
        direct_checkpoint_model.neuron_regions[0].unit_ids
        == ("u0", "u1", "adaptive.cortex.grown.1")
        and direct_checkpoint_model.structural_runtime_observations
        == direct_model.structural_runtime_observations
    )
    direct_rolled_back = direct_checkpoint_model.rollback_structural_candidate(
        direct_candidate.candidate_id
    )

    guard_model = TSKV8Adapter(_config(budget=1), episode_id="runtime-maintenance-guards")
    guard_region = _region()
    guard_model.attach_adaptive_neuron_region(guard_region)
    dependency = StructuralProposalCandidate(
        candidate_id="candidate:guard-dependency",
        network_id="standalone:adaptive.cortex",
        target_kind="neuron",
        operation="add",
        substrate_ids=(guard_region.region_id,),
        evidence_ids=("runtime:guard-dependency",),
        source_tick=1,
        priority=0.8,
        specification=(
            ("region_id", guard_region.region_id),
            ("unit_id", "adaptive.cortex.guard"),
        ),
        conflict_keys=("guard-dependency",),
    )
    dependent = StructuralProposalCandidate(
        candidate_id="candidate:guard-dependent",
        network_id="standalone:adaptive.cortex",
        target_kind="region",
        operation="split",
        substrate_ids=("adaptive.third",),
        evidence_ids=("runtime:guard-dependent",),
        source_tick=2,
        priority=0.7,
        specification=(
            ("region_id", "adaptive.third"),
            ("first_unit_count", 1),
        ),
        depends_on_candidate_ids=(dependency.candidate_id,),
        conflict_keys=("guard-dependent",),
    )
    guard_model._queue_structural_proposal_candidate(dependency)
    guard_model._queue_structural_proposal_candidate(dependent)
    dependency_results = guard_model.run_structural_maintenance_cycle(
        candidate_ids=(dependent.candidate_id, dependency.candidate_id),
        holdout_inputs_by_candidate={dependent.candidate_id: (torch.ones(5),)},
        expected_activities_by_candidate={dependent.candidate_id: (torch.zeros(3),)},
    )
    conflict_split = StructuralProposalCandidate(
        candidate_id="candidate:guard-conflict-split",
        network_id="standalone:adaptive.cortex",
        target_kind="region",
        operation="split",
        substrate_ids=("adaptive.other",),
        evidence_ids=("runtime:guard-conflict-split",),
        source_tick=3,
        priority=0.6,
        specification=(
            ("region_id", "adaptive.other"),
            ("first_unit_count", 1),
        ),
        conflict_keys=("guard-conflict-domain",),
    )
    conflict_prune = StructuralProposalCandidate(
        candidate_id="candidate:guard-conflict-prune",
        network_id="standalone:adaptive.cortex",
        target_kind="region",
        operation="prune",
        substrate_ids=(guard_region.region_id,),
        evidence_ids=("runtime:guard-conflict-prune",),
        source_tick=4,
        priority=0.5,
        specification=(("region_id", guard_region.region_id),),
        conflict_keys=("guard-conflict-domain",),
    )
    guard_model._queue_structural_proposal_candidate(conflict_split)
    guard_model._queue_structural_proposal_candidate(conflict_prune)
    conflict_results = guard_model.run_structural_maintenance_cycle(
        candidate_ids=(conflict_split.candidate_id, conflict_prune.candidate_id),
        holdout_inputs_by_candidate={
            conflict_split.candidate_id: (torch.ones(5),),
            conflict_prune.candidate_id: (torch.ones(5),),
        },
        expected_activities_by_candidate={
            conflict_split.candidate_id: (torch.zeros(3),),
            conflict_prune.candidate_id: (torch.zeros(3),),
        },
    )
    dependency_guard = bool(
        tuple(item.candidate_id for item in dependency_results)
        == (dependency.candidate_id, dependent.candidate_id)
        and dependency_results[0].status == "missing_holdout"
        and dependency_results[1].status == "failed_closed"
        and "dependency" in (dependency_results[1].error or "")
    )
    conflict_guard = bool(
        len(conflict_results) == 2
        and all(item.status == "failed_closed" for item in conflict_results)
        and all("conflict" in (item.error or "") for item in conflict_results)
        and guard_region.unit_ids == ("u0", "u1")
    )

    scale_model = TSKV8Adapter(_config(budget=6), episode_id="runtime-three-region")
    scale_network = _three_region_network()
    scale_model.attach_adaptive_neuron_network("cortex", scale_network)
    scale_source_relay = scale_model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="source",
        target_region_id="relay",
        evidence_ids=("runtime:scale-source-relay",),
        fan_in=1,
    )
    scale_relay_target = scale_model.propose_cross_region_connection(
        network_id="cortex",
        source_region_id="relay",
        target_region_id="target",
        evidence_ids=("runtime:scale-relay-target",),
        fan_in=1,
    )
    assert scale_model.commit_cross_region_connection("cortex", scale_source_relay)
    assert scale_model.commit_cross_region_connection("cortex", scale_relay_target)
    scale_original_regions = scale_network.region_ids
    scale_original_connections = scale_network.connection_ids
    scale_model.attach_cross_region_cooperation("cortex", CrossRegionCooperationLearner())
    scale_model.attach_structural_growth_controller(_growth_controller())
    scale_first = scale_model.step_cross_region_network(
        "cortex",
        {"source": torch.ones(3)},
        max_connections=2,
    )
    scale_model.step_cross_region_network(
        "cortex",
        {"source": torch.ones(3)},
        expected_activities=scale_first,
        holdout=True,
        max_connections=2,
    )
    scale_standalone = _region()
    scale_model.attach_adaptive_neuron_region(scale_standalone)
    scale_standalone_first = scale_model.step_adaptive_neuron_region(
        scale_standalone.region_id,
        torch.ones(5),
    )
    scale_model.step_adaptive_neuron_region(
        scale_standalone.region_id,
        torch.ones(5),
        expected_activity=scale_standalone_first,
        holdout=True,
    )
    scale_split_candidate = next(
        item
        for item in scale_model.structural_proposal_candidates
        if item.network_id == "cortex" and item.operation == "split"
    )
    scale_add_candidate = next(
        item
        for item in scale_model.structural_proposal_candidates
        if item.network_id == "standalone:adaptive.cortex"
        and item.target_kind == "neuron"
    )
    scale_add_proposal = scale_model.propose_neuron_add(
        region_id=scale_standalone.region_id,
        unit_id=dict(scale_add_candidate.specification)["unit_id"],
        evidence_ids=scale_add_candidate.evidence_ids,
    )
    scale_add_trial = AdaptiveNeuronRegion.from_payload(
        scale_standalone.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    scale_add_trial.apply_topology_proposal(
        scale_add_proposal,
        generator=torch.Generator().manual_seed(0),
    )
    scale_add_input = torch.zeros(5)
    scale_add_input[scale_add_trial.incoming.pre_index[-1]] = torch.sign(
        scale_add_trial.incoming.edge_weight[-1]
    )
    scale_add_expected = scale_add_trial.step(scale_add_input)
    scale_results = scale_model.run_structural_maintenance_cycle(
        candidate_ids=(scale_add_candidate.candidate_id, scale_split_candidate.candidate_id),
        holdout_inputs_by_candidate={
            scale_add_candidate.candidate_id: (scale_add_input,),
            scale_split_candidate.candidate_id: ({"source": torch.ones(3)},),
        },
        expected_activities_by_candidate={
            scale_add_candidate.candidate_id: (scale_add_expected,),
            scale_split_candidate.candidate_id: (scale_first,),
        },
    )
    scale_topology_after_commit = (
        scale_model.neuron_networks[0].region_ids,
        scale_model.neuron_networks[0].connection_ids,
    )
    scale_checkpoint_model = TSKV8Adapter.from_native_checkpoint(
        scale_model.native_checkpoint()
    )
    scale_checkpoint = bool(
        scale_checkpoint_model.neuron_regions[0].unit_count == 3
        and scale_checkpoint_model.neuron_networks[0].region_ids
        == scale_topology_after_commit[0]
        and scale_checkpoint_model.neuron_networks[0].connection_ids
        == scale_topology_after_commit[1]
    )
    scale_rollback = bool(
        scale_checkpoint_model.rollback_structural_candidate(
            scale_split_candidate.candidate_id
        )
        and scale_checkpoint_model.rollback_structural_candidate(
            scale_add_candidate.candidate_id
        )
        and scale_checkpoint_model.neuron_networks[0].region_ids
        == scale_original_regions
        and scale_checkpoint_model.neuron_networks[0].connection_ids
        == scale_original_connections
        and scale_checkpoint_model.neuron_regions[0].unit_count == 2
    )
    scale_gate = bool(
        len(scale_results) == 2
        and all(item.status == "committed" for item in scale_results)
        and len(scale_topology_after_commit[0]) == 4
        and "connection:relay->target" in scale_topology_after_commit[1]
        and len(scale_topology_after_commit[1]) == 3
        and scale_checkpoint
        and scale_rollback
    )

    gate_checks = {
        "observation_count": len(before_checkpoint) == 4,
        "first_without_error": before_checkpoint[0].prediction_error is None,
        "holdout_errors": all(item.prediction_error is not None for item in before_checkpoint[2:]),
        "growth_count": (
            model.structural_growth_controller is not None
            and model.structural_growth_controller.total_observations == 2
        ),
        "pruning_count": (
            model.structural_pruning_controller is not None
            and model.structural_pruning_controller.total_observations == 7
        ),
        "queue_drained": len(model.structural_proposal_candidates) == 0,
        "first_rolled_back": materialized.status == "rolled_back",
        "first_holdout": holdout_validated,
        "first_commit": committed,
        "first_rollback": rolled_back,
        "first_topology": topology_after_commit[0] == ("source", "target", "source.split.1"),
        "cycle_commit": cycle_committed,
        "cycle_rollback": cycle_rolled_back,
        "cycle_topology": cycle_topology[0] == ("source", "target", "target.split.1"),
        "route_evidence": route_state.evidence_count == 2,
        "restored_topology": before_topology
        == (restored_network.region_ids, restored_network.connection_ids),
        "checkpoint": checkpoint_continuation,
        "direct_add_candidate": (
            direct_candidate.target_kind == "neuron"
            and direct_candidate.operation == "add"
        ),
        "direct_add_holdout": direct_holdout_validated,
        "direct_add_commit": direct_committed,
        "direct_add_rollback": direct_rolled_back,
        "direct_add_checkpoint": direct_checkpoint,
        "dependency_fail_closed": dependency_guard,
        "conflict_fail_closed": conflict_guard,
        "multi_region_mixed_maintenance": scale_gate,
    }

    gate = {
        "passed": all(gate_checks.values()),
        "checks": gate_checks,
        "criterion": (
            "real native network ticks must emit checkpointable activity, prediction-error, "
            "learning-gain and resource observations; attached structural organs and route "
            "credit must continue across checkpoint, while topology remains unchanged until "
            "a separate holdout/budget/trial/rollback transaction commits it; standalone "
            "native neuron ticks must use the same governed birth path; candidate dependencies "
            "must execute in order and candidate conflicts must fail closed; a three-region "
            "network must preserve unaffected routes while connected split and standalone add "
            "are maintained together"
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
            "proposal_candidates": len(model.structural_proposal_candidates),
            "candidate_holdout_validated": holdout_validated,
            "candidate_committed": committed,
            "candidate_rolled_back": rolled_back,
            "topology_after_candidate_commit": list(topology_after_commit[0]),
            "maintenance_cycle_status": cycle_results[0].status,
            "maintenance_cycle_rolled_back": cycle_rolled_back,
            "direct_add_candidate": direct_candidate.candidate_id,
            "direct_add_holdout_validated": direct_holdout_validated,
            "direct_add_committed": direct_committed,
            "direct_add_rolled_back": direct_rolled_back,
            "direct_add_checkpoint": direct_checkpoint,
            "dependency_fail_closed": dependency_guard,
            "conflict_fail_closed": conflict_guard,
            "multi_region_mixed_maintenance": scale_gate,
            "multi_region_topology_after_commit": list(scale_topology_after_commit[0]),
            "multi_region_connections_after_commit": list(scale_topology_after_commit[1]),
            "multi_region_checkpoint": scale_checkpoint,
            "multi_region_rollback": scale_rollback,
            "route_evidence_count": route_state.evidence_count,
            "checkpoint_continuation": checkpoint_continuation,
            "checkpoint_checks": checkpoint_checks,
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
            "standalone_neuron_tick",
            "direct_neuron_birth_candidate",
            "direct_neuron_birth_holdout",
            "candidate_dependency_order",
            "candidate_conflict_fail_closed",
            "multi_region_route_migration",
            "mixed_add_split_maintenance",
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
