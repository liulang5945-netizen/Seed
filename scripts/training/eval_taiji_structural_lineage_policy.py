"""Run the R5C-S27 versioned structural-lineage policy canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training.eval_taiji_structural_lineage_compaction import (  # noqa: E402
    _record_terminal_subgraph,
)
from scripts.training.eval_taiji_workbench_multi_region_batch import (  # noqa: E402
    _build_runtime,
    _schedule_requests,
)
from scripts.training.eval_taiji_workbench_multi_region_lifecycle import (  # noqa: E402
    _record_real_evidence,
)
from seed import Seed  # noqa: E402
from taiji import (  # noqa: E402
    STRUCTURAL_LINEAGE_RETENTION_POLICY_REVISION,
    StructuralLineageRetentionPolicy,
    TSKV8Adapter,
)
from taiji.adapter import _checkpoint_digest  # noqa: E402

REPORT_FORMAT = "taiji-w7-r5c-s27-structural-lineage-policy-v1"
RETENTION_LIMIT = 1


def _runtime_with_terminal_lineage() -> SeedRuntime:
    runtime = _build_runtime()
    _record_real_evidence(runtime)
    schedule = runtime.schedule_structural_candidate_batch_from_workbench_evidence(
        _schedule_requests()
    )
    if schedule.get("status") != "batch_created":
        raise AssertionError(f"S27 active batch was not created: {schedule}")
    model = runtime.model.architecture
    active = next(item for item in model.structural_candidate_batches if item.batch_id == schedule["batch_id"])
    _record_terminal_subgraph(model, active)
    return runtime


def _maintenance(
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


def evaluate() -> dict[str, object]:
    policy = StructuralLineageRetentionPolicy.create(RETENTION_LIMIT)
    policy_roundtrip = StructuralLineageRetentionPolicy.from_payload(policy.to_payload())
    canonical_rules = policy.protection_rules == tuple(sorted(policy.protection_rules))

    legacy_runtime = _runtime_with_terminal_lineage()
    policy_runtime = _runtime_with_terminal_lineage()
    legacy = _maintenance(legacy_runtime, max_batches=RETENTION_LIMIT)
    explicit = _maintenance(policy_runtime, policy=policy)
    legacy_retention = legacy["lineage_retention"]
    explicit_retention = explicit["lineage_retention"]

    restored_runtime = SeedRuntime(Seed.from_checkpoint(policy_runtime.model.checkpoint()))
    restored_status = restored_runtime.structural_maintenance_status()
    expected_status = policy_runtime.structural_maintenance_status()

    switch_model = _runtime_with_terminal_lineage()
    first = _maintenance(switch_model, policy=policy)
    first_result = switch_model.model.architecture.structural_lineage_retention_result
    second_policy = StructuralLineageRetentionPolicy.create(2)
    second = _maintenance(switch_model, policy=second_policy)
    switched_result = switch_model.model.architecture.structural_lineage_retention_result

    invalid_before = _checkpoint_digest(policy_runtime.model.architecture.native_checkpoint())
    try:
        _maintenance(
            policy_runtime,
            max_batches=RETENTION_LIMIT,
            policy=policy,
        )
    except ValueError as exc:
        invalid_failed_closed = "max_batches or retention_policy" in str(exc)
    else:
        invalid_failed_closed = False
    invalid_after = _checkpoint_digest(policy_runtime.model.architecture.native_checkpoint())

    tampered_policy = {**policy.to_payload(), "policy_digest": "0" * 64}
    try:
        StructuralLineageRetentionPolicy.from_payload(tampered_policy)
    except ValueError as exc:
        tamper_failed_closed = "digest mismatch" in str(exc)
    else:
        tamper_failed_closed = False

    inconsistent = policy_runtime.model.architecture.native_checkpoint()
    runtime_component = dict(inconsistent["components"]["structural_runtime"])
    inconsistent_policy = StructuralLineageRetentionPolicy.create(2).to_payload()
    runtime_component["lineage_retention_policy"] = inconsistent_policy
    inconsistent["components"] = {
        **inconsistent["components"],
        "structural_runtime": runtime_component,
    }
    try:
        TSKV8Adapter.from_native_checkpoint(inconsistent)
    except ValueError as exc:
        inconsistent_failed_closed = "does not match result" in str(exc)
    else:
        inconsistent_failed_closed = False

    metrics = {
        "policy_roundtrip_is_canonical": (
            policy_roundtrip == policy
            and canonical_rules
            and policy.revision == STRUCTURAL_LINEAGE_RETENTION_POLICY_REVISION
        ),
        "legacy_and_policy_entries_share_semantics": (
            legacy_retention is not None
            and explicit_retention is not None
            and legacy_retention["max_batches"] == explicit_retention["max_batches"]
            and legacy_retention["removed_batch_ids"] == explicit_retention["removed_batch_ids"]
            and legacy["retention_policy"] == policy.to_payload()
            and explicit["retention_policy"] == policy.to_payload()
        ),
        "policy_checkpoint_restore_is_consistent": restored_status == expected_status,
        "policy_switch_affects_only_subsequent_maintenance": (
            first["retention_policy"] == policy.to_payload()
            and second["retention_policy"] == second_policy.to_payload()
            and first_result is not None
            and first_result.max_batches == RETENTION_LIMIT
            and switched_result is not None
            and switched_result.max_batches == 2
        ),
        "invalid_policy_combination_is_atomic": invalid_failed_closed and invalid_before == invalid_after,
        "tampered_policy_fails_closed": tamper_failed_closed,
        "inconsistent_checkpoint_policy_fails_closed": inconsistent_failed_closed,
    }
    return {
        "format": REPORT_FORMAT,
        "retention_limit": RETENTION_LIMIT,
        "policy": policy.to_payload(),
        "metrics": metrics,
        "gate": {
            "passed": all(metrics.values()),
            "criterion": (
                "retention is controlled by one versioned content-addressed safe policy; "
                "the legacy integer is only a compatibility conversion, policy state checkpoints "
                "with its result, and malformed or inconsistent policy inputs fail closed"
            ),
        },
        "boundary": (
            "This canary covers native CPU retention policy materialization. "
            "It does not claim policy-driven growth, background cleanup, open-domain quality, unlimited growth, CUDA, frontend behavior, or CI."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5c_s27_structural_lineage_policy_20260831.json",
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
