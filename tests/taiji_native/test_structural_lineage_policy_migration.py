from __future__ import annotations

import pytest

from api.seed_runtime import SeedRuntime
from scripts.training.eval_taiji_structural_lineage_compaction import _record_terminal_subgraph
from scripts.training.eval_taiji_workbench_multi_region_batch import (
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import _record_real_evidence
from taiji import (
    STRUCTURAL_LINEAGE_RETENTION_POLICY_LATEST_REVISION,
    StructuralLineageRetentionPolicy,
    StructuralLineageRetentionPolicyMigration,
    TSKV8Adapter,
)
from taiji.adapter import _checkpoint_digest


def _runtime_with_terminal_lineage() -> SeedRuntime:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
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
    return runtime


def _maintain(
    runtime: SeedRuntime,
    *,
    policy: StructuralLineageRetentionPolicy | None = None,
) -> dict[str, object]:
    return runtime.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_policy=policy,
    )


def test_policy_migration_is_adjacent_and_keeps_safety_semantics() -> None:
    source = StructuralLineageRetentionPolicy.create(1)
    target = source.migrate_to_latest()
    migration = StructuralLineageRetentionPolicyMigration.create(
        source,
        target,
        status="committed",
    )
    restored = StructuralLineageRetentionPolicyMigration.from_payload(migration.to_payload())

    assert target.revision == STRUCTURAL_LINEAGE_RETENTION_POLICY_LATEST_REVISION
    assert target.max_batches == source.max_batches
    assert target.protection_rules == source.protection_rules
    assert restored == migration
    with pytest.raises(ValueError, match="safety semantics"):
        StructuralLineageRetentionPolicyMigration.create(
            source,
            StructuralLineageRetentionPolicy.create(2, revision=2),
        )


def test_runtime_policy_migration_and_rollback_preserve_lineage_and_audit() -> None:
    runtime = _runtime_with_terminal_lineage()
    source = StructuralLineageRetentionPolicy.create(1)
    _maintain(runtime, policy=source)
    model = runtime.model.architecture
    before_result = model.structural_lineage_retention_result
    topology = tuple((item.region_id, item.unit_ids) for item in model.neuron_regions)
    budget = model.cognitive_snapshot().development.structural_budget

    target = source.migrate_to_latest()
    committed_payload = runtime.migrate_structural_lineage_retention_policy(target)
    committed = StructuralLineageRetentionPolicyMigration.from_payload(committed_payload)
    assert committed.status == "committed"
    assert model.structural_lineage_retention_policy == target
    assert model.structural_lineage_retention_result == before_result

    checkpoint = model.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    assert restored.structural_lineage_retention_policy == target
    assert restored.structural_lineage_retention_policy_migration == committed
    assert restored.structural_lineage_retention_result == before_result

    rolled_back_payload = runtime.rollback_structural_lineage_retention_policy_migration(
        committed_payload
    )
    rolled_back = StructuralLineageRetentionPolicyMigration.from_payload(rolled_back_payload)
    assert rolled_back.status == "rolled_back"
    assert model.structural_lineage_retention_policy == source
    assert model.structural_lineage_retention_result == before_result
    assert tuple((item.region_id, item.unit_ids) for item in model.neuron_regions) == topology
    assert model.cognitive_snapshot().development.structural_budget == budget
    assert model.structural_lineage_retention_policy_migration == rolled_back


def test_invalid_migration_and_missing_request_are_atomic_and_non_implicit() -> None:
    runtime = _runtime_with_terminal_lineage()
    source = StructuralLineageRetentionPolicy.create(1)
    _maintain(runtime, policy=source)
    model = runtime.model.architecture
    before = _checkpoint_digest(model.native_checkpoint())
    with pytest.raises(ValueError, match="safety semantics"):
        runtime.migrate_structural_lineage_retention_policy(
            StructuralLineageRetentionPolicy.create(2, revision=2)
        )
    assert _checkpoint_digest(model.native_checkpoint()) == before
    assert model.structural_lineage_retention_policy == source
    assert model.structural_lineage_retention_policy_migration is None

    _maintain(runtime)
    assert model.structural_lineage_retention_policy == source
    assert model.structural_lineage_retention_policy_migration is None
