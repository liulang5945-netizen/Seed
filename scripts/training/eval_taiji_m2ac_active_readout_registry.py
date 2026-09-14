"""Canary for a checkpointed, isolated active predictive readout branch."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    Taiji,
    WorkbenchBoundaryAuthorization,
    WorkbenchTaskBoundary,
)
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-active-readout-registry-canary-v1"


def _load_native_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    for candidate in (
        payload.get("model"),
        payload.get("substrate"),
        payload,
    ):
        if isinstance(candidate, dict) and str(candidate.get("format", "")).startswith(
            "taiji-native-"
        ):
            return copy.deepcopy(candidate)
    raise ValueError("checkpoint does not contain a native Taiji payload")


def _protected_owner_digest(model: Taiji) -> str:
    owners: dict[str, Any] = {
        "fabric": model.fabric.to_payload(),
        "motor": model.motor.to_payload(),
        "predictive_context": model.predictive_context.to_payload(),
        "predictive_readout": model.predictive_readout.to_payload(),
        "memory": model.memory.to_payload(),
    }
    if model.identity_organ is not None:
        owners["identity"] = model.identity_organ.to_payload(
            parent_checkpoint_digest="m2ac-active-readout-canary"
        )
    return content_digest(owners)


def _authorization(boundary: WorkbenchTaskBoundary) -> WorkbenchBoundaryAuthorization:
    return WorkbenchBoundaryAuthorization(
        project_id=boundary.project_id,
        task_id=boundary.task_id,
        session_id=boundary.session_id,
        capability_snapshot_id=boundary.capability_snapshot_id,
        authorized_capability_ids=("workspace.read",),
        active_boundary_digest=boundary.token_digest,
        current_tick=12,
        usage="execute",
    )


def run_canary(checkpoint: Path) -> dict[str, Any]:
    source = _load_native_checkpoint(checkpoint)
    model = Taiji.from_checkpoint(source)
    # This canary always starts from a protected-only parent.  An input model
    # produced by a previous experiment must not become an implicit branch
    # parent for the new registry.
    model.clear_active_predictive_readout()
    parent = model.checkpoint()
    protected = WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id="task:protected",
        session_id="session:one",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope="protected",
        issued_tick=10,
        ttl_ticks=20,
    )
    active = protected.successor(
        task_id="task:active",
        generation_scope="active",
        issued_tick=12,
        ttl_ticks=20,
    )
    authorization = _authorization(active)

    protected_before = _protected_owner_digest(model)
    protected_readout_before = model.readout_registry_status()["protected"]["readout_digest"]
    model.clone_protected_predictive_readout_as_active(
        boundary_digest=active.token_digest,
    )
    registered = model.active_predictive_readout_metadata
    if registered is None:
        raise RuntimeError("active readout registration produced no metadata")
    active_before = registered["readout_digest"]

    model.learn_bytes(
        b"Taiji active branch. Taiji active branch.\n",
        epochs=2,
        learn_fabric=False,
        learn_predictive_context=False,
        boundary=active,
        authorization=authorization,
    )
    active_after = model.active_predictive_readout_metadata
    if active_after is None:
        raise RuntimeError("active readout disappeared after training")
    protected_after = _protected_owner_digest(model)
    active_checkpoint = model.checkpoint()
    active_checkpoint_digest = content_digest(active_checkpoint)

    restored = Taiji.from_checkpoint(active_checkpoint)
    restored_metadata = restored.active_predictive_readout_metadata
    restored_checkpoint_digest = content_digest(restored.checkpoint())
    active_output = model.generate(
        b"Taiji",
        12,
        boundary=active,
        authorization=authorization,
    )
    restored_output = restored.generate(
        b"Taiji",
        12,
        boundary=active,
        authorization=authorization,
    )

    protected_only = Taiji.from_checkpoint(parent)
    try:
        protected_only.generate(
            b"Taiji",
            4,
            boundary=active,
            authorization=authorization,
        )
    except RuntimeError as exc:
        missing_registry_rejection = str(exc)
    else:
        missing_registry_rejection = ""

    checks = {
        "registry_metadata_has_owner_parent_boundary_and_digest": all(
            bool(registered.get(key))
            for key in (
                "owner",
                "parent_checkpoint_digest",
                "boundary_digest",
                "readout_digest",
            )
        ),
        "active_branch_starts_from_protected_readout": active_before == protected_readout_before,
        "active_training_changes_only_active_readout": (
            active_before != active_after["readout_digest"] and protected_before == protected_after
        ),
        "active_registry_is_checkpointed": "predictive_readout_registry" in active_checkpoint,
        "active_registry_fresh_process_round_trip": (
            restored_metadata == active_after
            and restored_checkpoint_digest == active_checkpoint_digest
        ),
        "active_generation_uses_active_owner": (
            model.last_generation_route is not None
            and model.last_generation_route["generation_scope"] == "active"
            and model.last_generation_route["readout_owner"] == "predictive_readout.active"
        ),
        "active_generation_output_round_trips": active_output == restored_output,
        "missing_active_registry_fails_closed": "not attached" in missing_registry_rejection,
    }
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "can_promote": False,
        "checkpoint": str(checkpoint),
        "parent_checkpoint_digest": content_digest(parent),
        "active_checkpoint_digest": active_checkpoint_digest,
        "checks": checks,
        "registry": {
            "registered": registered,
            "after_training": active_after,
            "restored": restored_metadata,
        },
        "owner_audit": {
            "protected_owner_digest_before": protected_before,
            "protected_owner_digest_after_active_training": protected_after,
            "protected_owners_unchanged": protected_before == protected_after,
            "missing_registry_rejection": missing_registry_rejection,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = run_canary(args.checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    json.loads(args.report.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": result["status"],
                "checks_passed": sum(int(value) for value in result["checks"].values()),
                "checks_total": len(result["checks"]),
                "protected_owners_unchanged": result["owner_audit"]["protected_owners_unchanged"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
