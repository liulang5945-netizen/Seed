from __future__ import annotations

import os
from pathlib import Path

import torch

from api.seed_runtime import SeedRuntime
from scripts.training.eval_taiji_structural_lineage_compaction import _record_terminal_subgraph
from scripts.training.eval_taiji_structural_lineage_restart_continuation import (
    _continuation_requests,
    _record_continuation_evidence,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import _record_real_evidence
from taiji import AdaptiveNeuronRegion, StructuralLineageRetentionPolicy
from taiji.adapter import _checkpoint_digest


def _holdout_payload(runtime: SeedRuntime, candidate_id: str) -> dict[str, object]:
    model = runtime.model.architecture
    candidate = next(
        item for item in model.structural_proposal_candidates if item.candidate_id == candidate_id
    )
    proposal = model.materialize_structural_candidate(candidate_id)
    assert proposal is not None
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
    assert schedule.get("status") == "batch_created"
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
    assert schedule.get("status") == "batch_created"
    return runtime


def test_restart_candidate_admission_and_rollback_continue_from_checkpoint() -> None:
    runtime = _build_migrated_runtime()
    checkpoint_root = Path(__file__).resolve().parents[2] / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    before_path = checkpoint_root / f"s31-before-admission-{suffix}.pt"
    after_first_path = checkpoint_root / f"s31-after-first-{suffix}.pt"
    after_rollback_path = checkpoint_root / f"s31-after-rollback-{suffix}.pt"
    try:
        batch = runtime.model.architecture.structural_candidate_batches[-1]
        first_candidate, second_candidate = batch.selected_candidate_ids
        first_spec = next(
            item
            for item in runtime.model.architecture.structural_proposal_candidates
            if item.candidate_id == first_candidate
        )
        first_region_id = str(dict(first_spec.specification)["region_id"])
        first_unit_id = str(dict(first_spec.specification)["unit_id"])
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
            assert "outside the selected batch" in str(exc)
        else:
            raise AssertionError("cross-batch candidate continuation unexpectedly succeeded")
        assert _checkpoint_digest(restored.model.architecture.native_checkpoint()) == before_cross_batch

        first = restored.continue_structural_candidate_batch(
            batch.batch_id,
            continuations_by_candidate={
                first_candidate: _holdout_payload(restored, first_candidate),
            },
        )
        assert first["results"][first_candidate]["status"] == "admitted"
        runtime_after_first = restored.model.architecture
        assert any(
            result.candidate_id == first_candidate and result.status == "admitted"
            for result in runtime_after_first.structural_admission_results
        )
        first_region = next(
            item for item in runtime_after_first.neuron_regions if item.region_id == first_region_id
        )
        assert first_unit_id in first_region.unit_ids
        restored.save(after_first_path)
        resumed = SeedRuntime.load(after_first_path)

        second = resumed.continue_structural_candidate_batch(
            batch.batch_id,
            continuations_by_candidate={
                second_candidate: _holdout_payload(resumed, second_candidate),
            },
        )
        assert second["results"][second_candidate]["status"] == "admitted"
        after_admission_budget = resumed.model.architecture.cognitive_snapshot().development.structural_budget
        rollback = resumed.model.architecture.rollback_structural_candidate_batch(
            batch.batch_id,
            second_candidate,
        )
        assert rollback["status"] == "rolled_back"
        after_rollback_budget = resumed.model.architecture.cognitive_snapshot().development.structural_budget
        assert after_rollback_budget == after_admission_budget + 1
        first_region = next(
            item for item in resumed.model.architecture.neuron_regions if item.region_id == first_region_id
        )
        assert first_unit_id in first_region.unit_ids
        resumed.save(after_rollback_path)
        final = SeedRuntime.load(after_rollback_path)
        final_status = final.structural_maintenance_status()
        assert final_status["last_retention_policy"]["revision"] == 2
        assert final_status["last_retention_policy_migration"]["status"] == "committed"
        assert final.model.architecture.structural_candidate_rollbacks[-1].candidate_id == second_candidate
        assert final.model.architecture.cognitive_snapshot().development.structural_budget == (
            after_rollback_budget
        )
    finally:
        before_path.unlink(missing_ok=True)
        after_first_path.unlink(missing_ok=True)
        after_rollback_path.unlink(missing_ok=True)
