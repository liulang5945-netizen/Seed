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
    STRUCTURAL_LINEAGE_RETENTION_PROTECTION_RULES,
    StructuralLineageRetentionPolicy,
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


def _empty_maintenance(
    runtime: SeedRuntime,
    *,
    max_batches: int | None = None,
    policy: StructuralLineageRetentionPolicy | None = None,
) -> dict[str, object]:
    return runtime.run_structural_maintenance_cycle(
        candidate_ids=(),
        holdout_inputs_by_candidate={},
        expected_activities_by_candidate={},
        lineage_retention_max_batches=max_batches,
        lineage_retention_policy=policy,
    )


def test_retention_policy_is_canonical_and_content_addressed() -> None:
    policy = StructuralLineageRetentionPolicy.create(
        3,
        protection_rules=tuple(reversed(STRUCTURAL_LINEAGE_RETENTION_PROTECTION_RULES)),
    )
    restored = StructuralLineageRetentionPolicy.from_payload(policy.to_payload())

    assert restored == policy
    assert policy.protection_rules == tuple(sorted(STRUCTURAL_LINEAGE_RETENTION_PROTECTION_RULES))
    with pytest.raises(ValueError, match="revision"):
        StructuralLineageRetentionPolicy.from_payload(
            {**policy.to_payload(), "revision": 99}
        )
    with pytest.raises(ValueError, match="digest mismatch"):
        StructuralLineageRetentionPolicy.from_payload(
            {**policy.to_payload(), "policy_digest": "0" * 64}
        )


def test_legacy_integer_and_policy_entry_use_the_same_retention_semantics() -> None:
    legacy_runtime = _runtime_with_terminal_lineage()
    policy_runtime = _runtime_with_terminal_lineage()
    policy = StructuralLineageRetentionPolicy.create(1)

    legacy = _empty_maintenance(legacy_runtime, max_batches=1)
    explicit = _empty_maintenance(policy_runtime, policy=policy)

    assert legacy["lineage_retention"] is not None
    assert explicit["lineage_retention"] is not None
    assert legacy["lineage_retention"]["removed_batch_ids"] == explicit["lineage_retention"][
        "removed_batch_ids"
    ]
    assert explicit["retention_policy"] == policy.to_payload()
    assert legacy["retention_policy"] == policy.to_payload()


def test_policy_checkpoint_restore_and_switch_are_explicit() -> None:
    runtime = _runtime_with_terminal_lineage()
    first_policy = StructuralLineageRetentionPolicy.create(1)
    first = _empty_maintenance(runtime, policy=first_policy)
    first_result = runtime.model.architecture.structural_lineage_retention_result
    checkpoint = runtime.model.architecture.native_checkpoint()
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)

    assert restored.structural_lineage_retention_policy == first_policy
    assert restored.structural_lineage_retention_result == first_result
    assert first["retention_policy"] == first_policy.to_payload()

    second_policy = StructuralLineageRetentionPolicy.create(2)
    second = _empty_maintenance(runtime, policy=second_policy)
    assert second["retention_policy"] == second_policy.to_payload()
    assert runtime.model.architecture.structural_lineage_retention_policy == second_policy
    assert first_result is not None
    assert first_result.max_batches == 1


def test_invalid_policy_combination_is_rejected_before_any_maintenance_mutation() -> None:
    runtime = _runtime_with_terminal_lineage()
    before = _checkpoint_digest(runtime.model.architecture.native_checkpoint())
    with pytest.raises(ValueError, match="max_batches or retention_policy"):
        _empty_maintenance(
            runtime,
            max_batches=1,
            policy=StructuralLineageRetentionPolicy.create(1),
        )
    assert _checkpoint_digest(runtime.model.architecture.native_checkpoint()) == before
