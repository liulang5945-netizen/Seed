"""Run the R5C-S31 restart candidate-admission and rollback canary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training.eval_taiji_structural_lineage_compaction import (  # noqa: E402
    _record_terminal_subgraph,
)
from scripts.training.eval_taiji_structural_lineage_restart_continuation import (  # noqa: E402
    _continuation_requests,
    _record_continuation_evidence,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import (  # noqa: E402
    _record_real_evidence,
)
from taiji import AdaptiveNeuronRegion, StructuralLineageRetentionPolicy  # noqa: E402
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s31-structural-lineage-restart-admission-v1"


def _holdout_payload(runtime: SeedRuntime, candidate_id: str) -> dict[str, object]:
    model = runtime.model.architecture
    candidate = next(
        item for item in model.structural_proposal_candidates if item.candidate_id == candidate_id
    )
    proposal = model.materialize_structural_candidate(candidate_id)
    if proposal is None:
        raise AssertionError(f"candidate {candidate_id} was not materialized")
    region_id = str(dict(candidate.specification)["region_id"])
    region = next(item for item in model.neuron_regions if item.region_id == region_id)
    trial = AdaptiveNeuronRegion.from_payload(
        region.to_payload(),
        generator=torch.Generator().manual_seed(0),
    )
    trial.apply_topology_proposal(proposal, generator=torch.Generator().manual_seed(0))
    holdout_input = torch.zeros(region.input_dim)
    holdout_input[trial.incoming.pre_index[-1]] = torch.sign(trial.incoming.edge_weight[-1])
    return {
        "holdout_inputs": (holdout_input,),
        "expected_activities": (trial.step(holdout_input),),
        "retention_regression": 0.0,
        "lesion_effect": 1.0,
        "resource_state": 0.8,
        "evidence_ids": (f"s31:retention:{candidate_id}", f"s31:lesion:{candidate_id}"),
    }


def _build_migrated_runtime() -> SeedRuntime:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S31 initial batch was not created: {schedule}")
    active = next(
        item
        for item in runtime.model.architecture.structural_candidate_batches
        if item.batch_id == schedule["batch_id"]
    )
    _record_terminal_subgraph(runtime.model.architecture, active)
    source = StructuralLineageRetentionPolicy.create(1)
    runtime.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_policy=source,
    )
    runtime.migrate_structural_lineage_retention_policy(source.migrate_to_latest())
    _record_continuation_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _continuation_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S31 continuation batch was not created: {schedule}")
    return runtime


def evaluate() -> dict[str, object]:
    runtime = _build_migrated_runtime()
    model = runtime.model.architecture
    batch = model.structural_candidate_batches[-1]
    first_candidate, second_candidate = batch.selected_candidate_ids
    first_spec = next(
        item for item in model.structural_proposal_candidates if item.candidate_id == first_candidate
    )
    first_region_id = str(dict(first_spec.specification)["region_id"])
    first_unit_id = str(dict(first_spec.specification)["unit_id"])
    second_spec = next(
        item for item in model.structural_proposal_candidates if item.candidate_id == second_candidate
    )
    second_region_id = str(dict(second_spec.specification)["region_id"])
    second_unit_id = str(dict(second_spec.specification)["unit_id"])

    checkpoint_root = PROJECT_ROOT / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    before_path = checkpoint_root / f"s31-before-admission-{suffix}.pt"
    after_first_path = checkpoint_root / f"s31-after-first-{suffix}.pt"
    after_rollback_path = checkpoint_root / f"s31-after-rollback-{suffix}.pt"
    try:
        runtime.save(before_path)
        restored = SeedRuntime.load(before_path)
        foreign_candidate = next(
            candidate_id
            for other_batch in restored.model.architecture.structural_candidate_batches
            if other_batch.batch_id != batch.batch_id
            for candidate_id in other_batch.selected_candidate_ids
        )
        before_cross_batch = _checkpoint_digest(restored.model.architecture.native_checkpoint())
        try:
            restored.continue_structural_candidate_batch(
                batch.batch_id,
                continuations_by_candidate={foreign_candidate: {}},
            )
        except ValueError as exc:
            cross_batch_rejected = "outside the selected batch" in str(exc)
        else:
            cross_batch_rejected = False
        cross_batch_atomic = (
            _checkpoint_digest(restored.model.architecture.native_checkpoint()) == before_cross_batch
        )
        first = restored.continue_structural_candidate_batch(
            batch.batch_id,
            continuations_by_candidate={
                first_candidate: _holdout_payload(restored, first_candidate),
            },
        )
        restored.save(after_first_path)
        resumed = SeedRuntime.load(after_first_path)
        second = resumed.continue_structural_candidate_batch(
            batch.batch_id,
            continuations_by_candidate={
                second_candidate: _holdout_payload(resumed, second_candidate),
            },
        )
        second_region_after_admission = next(
            item
            for item in resumed.model.architecture.neuron_regions
            if item.region_id == second_region_id
        )
        admitted_budget = resumed.model.architecture.cognitive_snapshot().development.structural_budget
        rollback = resumed.model.architecture.rollback_structural_candidate_batch(
            batch.batch_id,
            second_candidate,
        )
        rollback_budget = resumed.model.architecture.cognitive_snapshot().development.structural_budget
        resumed.save(after_rollback_path)
        final = SeedRuntime.load(after_rollback_path)
        final_model = final.model.architecture
        first_region = next(item for item in final_model.neuron_regions if item.region_id == first_region_id)
        second_region = next(item for item in final_model.neuron_regions if item.region_id == second_region_id)
        final_status = final.structural_maintenance_status()
        metrics = {
            "first_candidate_admits_after_restart": (
                first["results"][first_candidate]["status"] == "admitted"
                and first_unit_id in next(
                    item
                    for item in restored.model.architecture.neuron_regions
                    if item.region_id == first_region_id
                ).unit_ids
            ),
            "second_candidate_continues_from_intermediate_checkpoint": (
                second["results"][second_candidate]["status"] == "admitted"
                and second_unit_id in second_region_after_admission.unit_ids
            ),
            "rollback_restores_parent_budget_and_structure": (
                rollback["status"] == "rolled_back"
                and rollback_budget == admitted_budget + 1
                and first_unit_id in first_region.unit_ids
                and second_unit_id not in second_region.unit_ids
            ),
            "migration_policy_and_rollback_lineage_survive_final_restore": (
                final_status["last_retention_policy"]["revision"] == 2
                and final_status["last_retention_policy_migration"]["status"] == "committed"
                and final_model.structural_candidate_rollbacks[-1].candidate_id == second_candidate
            ),
            "candidate_and_batch_records_remain_checkpointable": (
                any(item.candidate_id == first_candidate for item in final_model.structural_admission_results)
                and any(item.candidate_id == second_candidate for item in final_model.structural_candidate_rollbacks)
            ),
            "cross_batch_continuation_fails_closed": cross_batch_rejected and cross_batch_atomic,
        }
        return {
            "format": REPORT_FORMAT,
            "batch_id": batch.batch_id,
            "candidate_ids": [first_candidate, second_candidate],
            "first_admission": first,
            "second_admission": second,
            "rollback": rollback,
            "metrics": metrics,
            "gate": {
                "passed": all(metrics.values()),
                "criterion": (
                    "after restart, a new candidate batch must continue through candidate-only replay, "
                    "atomic admission and rollback across checkpoints while preserving policy lineage and "
                    "topology/budget invariants"
                ),
            },
            "boundary": (
                "This canary covers native CPU restart candidate lifecycle. "
                "It does not claim open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
            ),
        }
    finally:
        before_path.unlink(missing_ok=True)
        after_first_path.unlink(missing_ok=True)
        after_rollback_path.unlink(missing_ok=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s31_structural_lineage_restart_admission_20260831.json",
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
