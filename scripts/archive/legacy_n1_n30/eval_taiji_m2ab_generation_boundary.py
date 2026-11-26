"""Canary Taiji generation consumption of a Workbench boundary token."""

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

FORMAT = "taiji-workbench-generation-boundary-canary-v1"


def _load_model(path: Path) -> tuple[Taiji, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    model_payload = payload.get("model", payload)
    if not isinstance(model_payload, dict):
        raise ValueError("checkpoint model payload must contain a mapping")
    model = Taiji.from_checkpoint(copy.deepcopy(model_payload))
    return model, content_digest(model.checkpoint())


def _owner_digest(model: Taiji) -> str:
    owners: dict[str, Any] = {
        "fabric": model.fabric.to_payload(),
        "motor": model.motor.to_payload(),
        "memory": model.memory.to_payload(),
        "predictive_context": model.predictive_context.to_payload(),
        "predictive_readout": model.predictive_readout.to_payload(),
    }
    if model.identity_organ is not None:
        owners["identity"] = model.identity_organ.to_payload(
            parent_checkpoint_digest="generation-boundary-canary"
        )
    return content_digest(owners)


def _authorization(
    boundary: WorkbenchTaskBoundary,
    *,
    usage: str = "execute",
    active_boundary_digest: str | None = None,
) -> WorkbenchBoundaryAuthorization:
    return WorkbenchBoundaryAuthorization(
        project_id=boundary.project_id,
        task_id=boundary.task_id,
        session_id=boundary.session_id,
        capability_snapshot_id=boundary.capability_snapshot_id,
        authorized_capability_ids=("workspace.read",),
        active_boundary_digest=(
            boundary.token_digest if active_boundary_digest is None else active_boundary_digest
        ),
        current_tick=10,
        usage=usage,
    )


def run_canary(checkpoint: Path) -> dict[str, Any]:
    model, checkpoint_before = _load_model(checkpoint)
    protected = WorkbenchTaskBoundary.issue(
        project_id="project:seed",
        task_id="task:protected",
        session_id="session:one",
        language_id="python",
        capability_snapshot_id="capability:snapshot:1",
        capability_ids=("workspace.read",),
        generation_scope="protected",
        issued_tick=0,
        ttl_ticks=20,
    )
    active = protected.successor(
        task_id="task:active",
        generation_scope="active",
        issued_tick=10,
        ttl_ticks=20,
    )
    closed = protected.close(closed_tick=10)

    owners_before = _owner_digest(model)
    protected_output = model.generate(
        b"boundary",
        2,
        boundary=protected,
        authorization=_authorization(protected),
    )
    owners_after_protected = _owner_digest(model)
    protected_route = model.last_generation_route

    try:
        model.generate(
            b"boundary",
            2,
            boundary=active,
            authorization=_authorization(active),
        )
    except RuntimeError as exc:
        active_rejection = str(exc)
    else:
        active_rejection = ""
    owners_after_active_rejection = _owner_digest(model)

    replay_output = model.generate(
        b"boundary",
        2,
        boundary=closed,
        authorization=_authorization(closed, usage="read_only_replay"),
    )
    owners_after_replay = _owner_digest(model)
    replay_route = model.last_generation_route
    checkpoint_after = content_digest(model.checkpoint())

    checks = {
        "protected_generation_succeeded": bool(protected_output),
        "protected_route_recorded": protected_route
        == {
            "boundary_digest": protected.token_digest,
            "generation_scope": "protected",
            "readout_owner": "predictive_readout",
            "read_only_replay": False,
        },
        "active_generation_missing_readout_fails_closed": "not attached" in active_rejection,
        "active_rejection_does_not_change_persistent_owners": (
            owners_after_active_rejection == owners_after_protected
        ),
        "closed_generation_replay_succeeded": bool(replay_output),
        "closed_generation_replay_is_read_only": (
            replay_route is not None and replay_route["read_only_replay"] is True
        ),
        "persistent_owners_unchanged": (
            owners_before
            == owners_after_protected
            == owners_after_active_rejection
            == owners_after_replay
        ),
    }
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "can_promote": False,
        "checkpoint": str(checkpoint),
        "checkpoint_digest_before": checkpoint_before,
        "checkpoint_digest_after_generation": checkpoint_after,
        "checkpoint_dynamics_changed": checkpoint_before != checkpoint_after,
        "checks": checks,
        "routes": {
            "protected": protected_route,
            "closed_replay": replay_route,
            "active_rejection": active_rejection,
        },
        "owner_audit": {
            "persistent_owner_digest_before": owners_before,
            "persistent_owner_digest_after_protected": owners_after_protected,
            "persistent_owner_digest_after_active_rejection": owners_after_active_rejection,
            "persistent_owner_digest_after_replay": owners_after_replay,
            "persistent_owners_read_only": checks["persistent_owners_unchanged"],
            "note": "generation advances dynamic state; no learning owner is modified",
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
                "persistent_owners_read_only": result["owner_audit"]["persistent_owners_read_only"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
