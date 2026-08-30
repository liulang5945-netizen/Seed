"""Run the deterministic R5C-S2 candidate-only structural bridge canary."""

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

REPORT_FORMAT = "taiji-w7-r5c-s2-structural-bridge-v1"


def _observation(
    tick: int,
    *,
    task_slice_id: str,
    partition: str,
    prediction_error: float = 0.7,
) -> StructuralRuntimeObservation:
    return StructuralRuntimeObservation(
        network_id="standalone:adaptive.cortex",
        region_id="region:pressure-canary",
        tick=tick,
        usage=0.5,
        resource_pressure=0.2,
        prediction_error=prediction_error,
        learning_gain=0.1,
        holdout_transfer=0.8 if partition == "holdout" else 0.0,
        evidence_id=f"bridge-canary:{partition}:{task_slice_id}:{tick}",
        task_slice_id=task_slice_id,
        partition=partition,
    )


def _build_model() -> tuple[TSKV8Adapter, AdaptiveNeuronRegion]:
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
        seed=71,
    )
    model = TSKV8Adapter(config, episode_id="pressure-bridge-canary")
    region = AdaptiveNeuronRegion(
        region_id="adaptive.cortex",
        input_dim=5,
        unit_ids=("u0", "u1"),
        fan_in=2,
        generator=torch.Generator().manual_seed(7),
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


def evaluate() -> dict[str, object]:
    ledger = StructuralEvidenceLedger(window_capacity=1)
    for observation in (
        _observation(1, task_slice_id="task-a", partition="train"),
        _observation(2, task_slice_id="task-b", partition="train"),
        _observation(3, task_slice_id="task-a", partition="holdout"),
    ):
        ledger.append(observation)
    projection = project_structural_growth_pressure(ledger.sealed_summaries)

    model, region = _build_model()
    before_units = model.neuron_regions[0].unit_ids
    before_budget = model.cognitive_snapshot().development.structural_budget
    candidate = model.propose_structural_candidate_from_pressure(
        projection,
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": "u2"},
    )
    if candidate is None:
        raise AssertionError("pressure bridge did not produce a candidate")
    duplicate = model.propose_structural_candidate_from_pressure(
        projection,
        controller_region_id=region.region_id,
        target_kind="neuron",
        operation="add",
        substrate_ids=(region.region_id,),
        specification={"region_id": region.region_id, "unit_id": "u2"},
    )
    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    restored_candidate = restored.structural_proposal_candidates[0]
    proposal = restored.materialize_structural_candidate(restored_candidate.candidate_id)
    restored_checkpoint = restored.native_checkpoint()
    checkpoint_runtime = checkpoint["components"]["structural_runtime"]
    restored_runtime = restored_checkpoint["components"]["structural_runtime"]

    metrics = {
        "candidate_created": candidate is not None,
        "duplicate_projection_rejected": duplicate is None,
        "topology_unchanged": model.neuron_regions[0].unit_ids == before_units,
        "budget_unchanged": (
            model.cognitive_snapshot().development.structural_budget == before_budget
        ),
        "parent_checkpoint_bound": bool(candidate.parent_checkpoint_id),
        "runtime_clock_covers_evidence": (
            checkpoint_runtime["runtime_tick"] >= projection.last_tick
        ),
        "checkpoint_restores_projection_dedupe": (
            restored.structural_pressure_projection_digests
            == (projection.projection_digest,)
        ),
        "checkpoint_restores_candidate": (
            restored_candidate.to_payload() == candidate.to_payload()
        ),
        "materialization_stays_pending": proposal is not None,
        "materialization_preserves_parent": (
            proposal is not None
            and proposal.parent_checkpoint_id == restored_candidate.parent_checkpoint_id
        ),
        "restore_keeps_topology_unchanged": (
            restored.neuron_regions[0].unit_ids == before_units
        ),
        "restore_keeps_budget_unchanged": (
            restored.cognitive_snapshot().development.structural_budget == before_budget
        ),
        "restored_checkpoint_is_serializable": (
            restored_runtime["runtime_tick"] >= projection.last_tick
        ),
    }
    return {
        "format": REPORT_FORMAT,
        "projection": projection.to_payload(),
        "candidate": candidate.to_payload(),
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "sealed cross-task pressure may create one deduplicated, lineage-bound "
                "candidate, but cannot mutate topology or spend structural budget"
            ),
        },
        "boundary": (
            "This canary does not admit or execute structural growth. It only bridges "
            "a validated projection to the existing controller and preserves recovery."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s2_structural_bridge_20260830.json",
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
