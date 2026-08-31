from __future__ import annotations

import os
from pathlib import Path

import torch

from api.seed_runtime import SeedRuntime
from scripts.training.eval_taiji_structural_lineage_compaction import _record_terminal_subgraph
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import _record_real_evidence
from taiji import StructuralLineageRetentionPolicy


def _runtime_with_migrated_policy() -> tuple[SeedRuntime, dict[str, object]]:
    print("s29:build_runtime", flush=True)
    runtime = _build_runtime()
    print("s29:record_evidence", flush=True)
    _record_real_evidence(runtime)
    print("s29:schedule", flush=True)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    assert schedule.get("status") == "batch_created"
    model = runtime.model.architecture
    active = next(
        item
        for item in model.structural_candidate_batches
        if item.batch_id == str(schedule["batch_id"])
    )
    _record_terminal_subgraph(model, active)
    print("s29:maintenance", flush=True)
    source = StructuralLineageRetentionPolicy.create(1)
    runtime.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_policy=source,
    )
    print("s29:migrate", flush=True)
    target = source.migrate_to_latest()
    migration = runtime.migrate_structural_lineage_retention_policy(target)
    print("s29:ready", flush=True)
    return runtime, migration


def test_seed_runtime_disk_checkpoint_preserves_migration_and_rollback() -> None:
    runtime, migration = _runtime_with_migrated_policy()
    expected_status = runtime.structural_maintenance_status()
    expected_result = runtime.model.architecture.structural_lineage_retention_result
    terminal_batch_id = "batch:terminal-lineage"
    assert terminal_batch_id not in {
        item.batch_id for item in runtime.model.architecture.structural_candidate_batches
    }

    checkpoint_root = Path(__file__).resolve().parents[2] / "output" / "manual-r5-canary"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suffix = os.getpid()
    checkpoint = checkpoint_root / f"s29-migrated-{suffix}.pt"
    rolled_back_checkpoint = checkpoint_root / f"s29-rolled-back-{suffix}.pt"
    tampered_checkpoint = checkpoint_root / f"s29-tampered-{suffix}.pt"
    missing_field_checkpoint = checkpoint_root / f"s29-missing-field-{suffix}.pt"
    try:
        runtime.save(checkpoint)
        original_bytes = checkpoint.read_bytes()
        restored = SeedRuntime.load(checkpoint)

        assert restored.structural_maintenance_status() == expected_status
        assert (
            restored.model.architecture.structural_lineage_retention_result == expected_result
        )
        assert terminal_batch_id not in {
            item.batch_id for item in restored.model.architecture.structural_candidate_batches
        }
        rolled_back = restored.rollback_structural_lineage_retention_policy_migration(migration)
        assert rolled_back["status"] == "rolled_back"
        assert restored.model.architecture.structural_lineage_retention_policy.revision == 1
        assert restored.model.architecture.structural_lineage_retention_result == expected_result

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        taiji_payload = dict(payload["taiji"])
        components = dict(taiji_payload["components"])
        structural_runtime = dict(components["structural_runtime"])
        migration_payload = dict(structural_runtime["lineage_retention_policy_migration"])
        migration_payload["migration_digest"] = "0" * 64
        structural_runtime["lineage_retention_policy_migration"] = migration_payload
        components["structural_runtime"] = structural_runtime
        taiji_payload["components"] = components
        payload["taiji"] = taiji_payload
        torch.save(payload, tampered_checkpoint)
        try:
            SeedRuntime.load(tampered_checkpoint)
        except ValueError as exc:
            assert "migration digest mismatch" in str(exc)
        else:
            raise AssertionError("tampered migration checkpoint unexpectedly loaded")
        assert checkpoint.read_bytes() == original_bytes
        assert runtime.structural_maintenance_status() == expected_status

        missing_payload = dict(payload)
        missing_payload["config"] = dict(missing_payload["config"])
        missing_payload["config"].pop("taiji", None)
        torch.save(missing_payload, missing_field_checkpoint)
        try:
            SeedRuntime.load(missing_field_checkpoint)
        except (KeyError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("incomplete checkpoint unexpectedly loaded")
        assert checkpoint.read_bytes() == original_bytes
        assert runtime.structural_maintenance_status() == expected_status

        restored.save(rolled_back_checkpoint)
        resumed = SeedRuntime.load(rolled_back_checkpoint)
        resumed_status = resumed.structural_maintenance_status()
        assert resumed_status["last_retention_policy"]["revision"] == 1
        assert resumed_status["last_retention_policy_migration"]["status"] == "rolled_back"
        assert resumed.model.architecture.structural_lineage_retention_result == expected_result
        assert terminal_batch_id not in {
            item.batch_id for item in resumed.model.architecture.structural_candidate_batches
        }
    finally:
        checkpoint.unlink(missing_ok=True)
        rolled_back_checkpoint.unlink(missing_ok=True)
        tampered_checkpoint.unlink(missing_ok=True)
        missing_field_checkpoint.unlink(missing_ok=True)
