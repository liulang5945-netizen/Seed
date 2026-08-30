"""Run the deterministic R5C-S4 cross-seed structural stability canary."""

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
    AdaptiveNeuronRegion,
    AdaptiveStructuralGrowthController,
    StructuralEvidenceLedger,
    StructuralGrowthDynamics,
    StructuralRuntimeObservation,
    TaijiConfig,
    TSKV8Adapter,
    project_structural_growth_pressure,
)

REPORT_FORMAT = "taiji-w7-r5c-s4-structural-stability-v1"


def _observation(
    tick: int,
    *,
    task_slice_id: str,
    partition: str,
    seed: int,
) -> StructuralRuntimeObservation:
    return StructuralRuntimeObservation(
        network_id="standalone:adaptive.cortex",
        region_id=f"region:stability:{seed}",
        tick=tick,
        usage=0.5,
        resource_pressure=0.2,
        prediction_error=0.7,
        learning_gain=0.1,
        holdout_transfer=0.8 if partition == "holdout" else 0.0,
        evidence_id=f"stability:{seed}:{partition}:{task_slice_id}:{tick}",
        task_slice_id=task_slice_id,
        partition=partition,
    )


def _projection(seed: int):
    ledger = StructuralEvidenceLedger(window_capacity=1)
    for observation in (
        _observation(1, task_slice_id="task-a", partition="train", seed=seed),
        _observation(2, task_slice_id="task-b", partition="train", seed=seed),
        _observation(3, task_slice_id="task-a", partition="holdout", seed=seed),
    ):
        ledger.append(observation)
    return project_structural_growth_pressure(ledger.sealed_summaries)


def _build_model(seed: int) -> tuple[TSKV8Adapter, AdaptiveNeuronRegion]:
    config = TaijiConfig(
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
        development_structural_budget=1,
        seed=seed,
    )
    model = TSKV8Adapter(config, episode_id=f"stability-{seed}")
    region = AdaptiveNeuronRegion(
        region_id="adaptive.cortex",
        input_dim=5,
        unit_ids=("u0", "u1"),
        fan_in=2,
        generator=torch.Generator().manual_seed(seed),
    )
    model.attach_adaptive_neuron_region(region)
    model.attach_structural_growth_controller(
        AdaptiveStructuralGrowthController(
            dynamics=StructuralGrowthDynamics(
                ema_rate=1.0,
                error_threshold=0.0,
                holdout_transfer_threshold=0.0,
                minimum_resource_state=0.0,
                required_error_steps=1,
            )
        )
    )
    return model, region


def _trial_metrics(
    region: AdaptiveNeuronRegion,
    proposal,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    parent = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    trial = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    trial.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(0))

    retention_input = torch.zeros(5)
    retention_input[parent.incoming.pre_index[0]] = torch.sign(parent.incoming.edge_weight[0])
    parent_activity = parent.step(retention_input)
    trial_activity = trial.step(retention_input)[: parent.unit_count]
    retention_regression = float(
        torch.mean(torch.abs(parent_activity - trial_activity)).clamp(0.0, 1.0).item()
    )

    lesion_input = torch.zeros(5)
    lesion_input[trial.incoming.pre_index[-1]] = torch.sign(trial.incoming.edge_weight[-1])
    trial_activity = trial.step(lesion_input)
    lesioned = AdaptiveNeuronRegion.from_payload(
        trial.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    lesioned.lesion_topology_proposal(proposal)
    lesioned_activity = lesioned.step(lesion_input)
    lesion_effect = float(
        torch.mean(torch.abs(trial_activity - lesioned_activity)).clamp(0.0, 1.0).item()
    )
    return retention_regression, lesion_effect, retention_input, lesion_input


def _run_seed(seed: int) -> dict[str, object]:
    projection = _projection(seed)
    model, region = _build_model(seed)
    candidate = model.propose_structural_candidate_from_pressure(
        projection,
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": f"u2-seed-{seed}"},
    )
    if candidate is None:
        raise AssertionError(f"seed {seed} did not produce a candidate")
    proposal = model.materialize_structural_candidate(candidate.candidate_id)
    if proposal is None:
        raise AssertionError(f"seed {seed} did not materialize a proposal")
    trial = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    trial.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(0))
    holdout_input = torch.zeros(5)
    holdout_input[trial.incoming.pre_index[-1]] = torch.sign(trial.incoming.edge_weight[-1])
    expected_activity = trial.step(holdout_input)
    validation = model.validate_structural_candidate_shadow(
        candidate.candidate_id,
        holdout_inputs=(holdout_input,),
        expected_activities=(expected_activity,),
    )
    retention_regression, lesion_effect, _, _ = _trial_metrics(region, proposal)
    decision = model.evaluate_structural_candidate_gate(
        validation,
        retention_regression=retention_regression,
        lesion_effect=lesion_effect,
        resource_state=0.80,
        evidence_ids=(
            f"retention:seed:{seed}",
            f"lesion:seed:{seed}",
        ),
    )
    before_units = region.unit_ids
    admission = model.admit_structural_candidate(validation, decision)
    admitted_units = region.unit_ids
    admitted_checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(admitted_checkpoint)
    rollback_ok = restored.rollback_structural_candidate(candidate.candidate_id)

    return {
        "seed": seed,
        "projection_digest": projection.projection_digest,
        "candidate_id": candidate.candidate_id,
        "validation": validation.to_payload(),
        "decision": decision.to_payload(),
        "admission": admission.to_payload(),
        "observed_retention_regression": retention_regression,
        "observed_lesion_effect": lesion_effect,
        "before_units": before_units,
        "admitted_units": admitted_units,
        "rollback": {
            "succeeded": rollback_ok,
            "restored_units": restored.neuron_regions[0].unit_ids,
            "restored_budget": restored.cognitive_snapshot().development.structural_budget,
        },
        "checks": {
            "policy_passed": decision.passed,
            "admitted_once": admission.status == "admitted",
            "retention_within_limit": retention_regression <= 0.05,
            "lesion_effect_is_causal": lesion_effect >= 0.05,
            "one_unit_growth": admitted_units == before_units + (f"u2-seed-{seed}",),
            "rollback_restores_parent": (
                rollback_ok and restored.neuron_regions[0].unit_ids == before_units
            ),
            "rollback_restores_budget": (
                rollback_ok
                and restored.cognitive_snapshot().development.structural_budget == 1
            ),
        },
    }


def evaluate() -> dict[str, object]:
    trials = [_run_seed(seed) for seed in (11, 29)]
    checks = {
        f"seed_{trial['seed']}": trial["checks"] for trial in trials
    }
    flattened = [value for trial in trials for value in trial["checks"].values()]
    return {
        "format": REPORT_FORMAT,
        "trials": trials,
        "metrics": {
            "independent_seeds": len(trials) == 2,
            "all_trial_checks_passed": all(flattened),
            "cross_seed_retention_stable": all(
                trial["checks"]["retention_within_limit"] for trial in trials
            ),
            "cross_seed_lesion_stable": all(
                trial["checks"]["lesion_effect_is_causal"] for trial in trials
            ),
            "cross_seed_rollback_stable": all(
                trial["checks"]["rollback_restores_parent"]
                and trial["checks"]["rollback_restores_budget"]
                for trial in trials
            ),
        },
        "checks_by_seed": checks,
        "gate": {
            "passed": all(flattened),
            "criterion": (
                "the same bounded structural admission must reproduce holdout validation, "
                "old-task retention, lesion causality, and parent rollback across independent seeds"
            ),
        },
        "boundary": (
            "This is a two-seed canary, not a proof of open-domain intelligence or permission for "
            "unbounded structural scaling."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s4_structural_stability_20260830.json",
    )
    args = parser.parse_args()
    report = evaluate()
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
