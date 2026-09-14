"""Preflight the same-parent K adapter checkpoint and rollback boundary.

This is an R6 entry preflight only.  It does not train a learner, attach the
adapter to the default runtime, or promote a candidate.  The contract is
frozen in ``plans/reference/M4V2_R6_A8_PROMOTION_PREREGISTRATION_20260909.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    KContinualAdapter,
    OutcomeDependencyProjector,
    OutcomeDependencySpec,
    WorldEvent,
    WorldState,
    content_digest,
)

REPORT_FORMAT = "taiji-m4v2-r6-k-adapter-preflight-v1"
VERSION = 1


def _projection() -> Any:
    world = WorldState(tick=0, entities=("workbench",), uncertainty=0.0)
    event = WorldEvent(
        event_id="outcome-0",
        kind="workbench.evidence",
        tick=0,
        subject_id="workspace.read",
        attributes=(
            ("capability_id", "workspace.read"),
            ("success", True),
        ),
        provenance="workbench-observed",
    )
    spec = OutcomeDependencySpec(
        dependency_id="dep-0",
        next_task_id="task-follow-up",
        capability_id="workspace.read",
        required_outcome="success",
    )
    return OutcomeDependencyProjector("r6-scope").project(world, event, spec)


def _adapter(parent: dict[str, Any]) -> KContinualAdapter:
    return KContinualAdapter(
        parent_checkpoint_digest=content_digest(parent),
        owner_graph_digest="1" * 64,
        source_manifest_digest="2" * 64,
        resource_manifest_digest="3" * 64,
        dependency_scope_id="r6-scope",
    )


def run_preflight() -> dict[str, Any]:
    parent = {
        "format": "r6-parent-v1",
        "owner": "taiji",
        "revision": 0,
    }
    adapter = _adapter(parent)
    projection = _projection()
    checks: dict[str, bool] = {
        "parent_checkpoint_match": adapter.parent_checkpoint_matches(parent),
        "dependency_projection_accepted": bool(projection.accepted),
        "dependency_projection_lineage_complete": len(projection.lineage) == 4,
        "training_steps_zero_before_trial": adapter.training_steps == 0,
    }
    adapter.bind_dependency_projection(projection)
    preflight_checkpoint = adapter.checkpoint()
    restored_preflight = KContinualAdapter.from_checkpoint(preflight_checkpoint)
    checks["checkpoint_prefit_roundtrip"] = restored_preflight.checkpoint() == preflight_checkpoint

    rollback_token = adapter.stage_candidate(
        candidate_checkpoint_digest="4" * 64,
        candidate_owner_graph_digest="5" * 64,
        candidate_source_manifest_digest="6" * 64,
        candidate_parent_checkpoint_digest=content_digest(parent),
    )
    staged_checkpoint = adapter.checkpoint()
    staged_restored = KContinualAdapter.from_checkpoint(staged_checkpoint)
    checks["candidate_parent_is_same_parent"] = (
        staged_restored.parent_checkpoint_digest == content_digest(parent)
    )
    checks["candidate_checkpoint_roundtrip"] = staged_restored.checkpoint() == staged_checkpoint
    checks["candidate_namespace_is_isolated"] = (
        staged_restored.active_namespace == staged_restored.candidate_namespace
        and staged_restored.candidate_namespace != staged_restored.parent_namespace
    )
    candidate_trial_id = str(staged_checkpoint["staged_trial_id"])
    rollback_record = adapter.rollback(rollback_token)
    rollback_checkpoint = adapter.checkpoint()
    restored_rollback = KContinualAdapter.from_checkpoint(rollback_checkpoint)
    checks["rollback_record_is_explicit"] = (
        rollback_record.status == "rolled_back"
        and rollback_record.reason == "explicit_parent_restore"
        and rollback_record.trial_id == candidate_trial_id
    )
    checks["rollback_restores_parent_namespace"] = (
        restored_rollback.active_namespace == restored_rollback.parent_namespace
    )
    checks["rollback_preserves_dependency_projection"] = (
        restored_rollback.dependency_projection == projection
    )
    checks["rollback_checkpoint_roundtrip"] = restored_rollback.checkpoint() == rollback_checkpoint
    checks["training_steps_zero_after_trial"] = adapter.training_steps == 0

    return {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "artifacts": {
            "parent_checkpoint_digest": content_digest(parent),
            "dependency_projection_digest": projection.projection_digest,
            "candidate_trial_id": candidate_trial_id,
            "rollback_record_digest": rollback_record.record_digest,
            "preflight_checkpoint_digest": preflight_checkpoint["checkpoint_digest"],
            "staged_checkpoint_digest": staged_checkpoint["checkpoint_digest"],
            "rollback_checkpoint_digest": rollback_checkpoint["checkpoint_digest"],
        },
        "boundary": {
            "training_performed": False,
            "default_runtime_attached": False,
            "candidate_promoted": False,
            "cuda_required": False,
            "purpose": "R6 same-parent checkpoint/lineage/rollback entry preflight",
        },
        "next_gate": (
            "Do not train or attach the adapter until the preregistered R4/R5/K "
            "entry gates and resource/rollback/old-capability retention course "
            "are satisfied."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = run_preflight()
    payload["generated_at_epoch"] = int(time.time())
    payload["elapsed_seconds"] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": payload["status"],
                "checks_passed": sum(payload["checks"].values()),
                "checks_total": len(payload["checks"]),
                "training_performed": payload["boundary"]["training_performed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
