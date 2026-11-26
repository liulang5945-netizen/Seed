"""Canary the explicit Workbench task boundary/readout-generation contract."""

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
    select_readout_generation,
)
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-workbench-task-boundary-canary-v1"


def _load_model(path: Path) -> tuple[Taiji, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    model_payload = payload.get("model", payload)
    if not isinstance(model_payload, dict):
        raise ValueError("checkpoint model payload must contain a mapping")
    model = Taiji.from_checkpoint(copy.deepcopy(model_payload))
    return model, content_digest(model.checkpoint())


def _authorization(
    boundary: WorkbenchTaskBoundary,
    *,
    current_tick: int,
    usage: str = "execute",
    project_id: str | None = None,
    active_boundary_digest: str | None = None,
    authorized_capability_ids: tuple[str, ...] = ("workspace.read",),
) -> WorkbenchBoundaryAuthorization:
    return WorkbenchBoundaryAuthorization(
        project_id=boundary.project_id if project_id is None else project_id,
        task_id=boundary.task_id,
        session_id=boundary.session_id,
        capability_snapshot_id=boundary.capability_snapshot_id,
        authorized_capability_ids=authorized_capability_ids,
        active_boundary_digest=(
            boundary.token_digest if active_boundary_digest is None else active_boundary_digest
        ),
        current_tick=current_tick,
        usage=usage,
    )


def _check(accepted: bool, reason: str = "") -> dict[str, Any]:
    return {"passed": bool(accepted), "reason": reason}


def run_canary(checkpoint: Path) -> dict[str, Any]:
    model, checkpoint_before = _load_model(checkpoint)
    base_kwargs = {
        "project_id": "project:seed",
        "task_id": "task:alpha",
        "session_id": "session:one",
        "language_id": "python",
        "capability_snapshot_id": "capability:snapshot:1",
        "capability_ids": ("workspace.read",),
        "generation_scope": "protected",
        "issued_tick": 10,
        "ttl_ticks": 20,
    }
    old = WorkbenchTaskBoundary.issue(**base_kwargs)
    same_task = WorkbenchTaskBoundary.issue(**base_kwargs)
    old_closed = old.close(closed_tick=12)
    active = old_closed.successor(
        task_id="task:beta",
        generation_scope="active",
        issued_tick=12,
        ttl_ticks=20,
    )
    unauthorized = WorkbenchTaskBoundary.issue(
        **{**base_kwargs, "capability_ids": ("workspace.apply_patch",)}
    )

    checks = {
        "same_task_is_stable": _check(same_task.token_digest == old.token_digest),
        "task_switch_creates_new_generation": _check(
            active.token_digest != old.token_digest
            and active.parent_token_digest == old_closed.token_digest
            and active.generation_scope == "active"
        ),
        "active_generation_is_authorized": _check(
            select_readout_generation(
                active,
                _authorization(active, current_tick=12),
            )
            == "active"
        ),
        "closed_old_task_is_read_only_replay": _check(
            select_readout_generation(
                old_closed,
                _authorization(old_closed, current_tick=12, usage="read_only_replay"),
            )
            == "protected"
        ),
        "closed_old_task_cannot_execute": _check(
            not old_closed.authorize(_authorization(old_closed, current_tick=12)).accepted
        ),
        "old_generation_is_stale_after_switch": _check(
            old.authorize(
                _authorization(
                    old,
                    current_tick=12,
                    active_boundary_digest=active.token_digest,
                )
            ).reason_code
            == "stale_task_generation"
        ),
        "unauthorized_capability_fails_closed": _check(
            unauthorized.authorize(_authorization(unauthorized, current_tick=12)).reason_code
            == "unauthorized_capability"
        ),
        "expired_boundary_fails_closed": _check(
            old.authorize(_authorization(old, current_tick=31)).reason_code == "expired_boundary"
        ),
        "cross_project_boundary_fails_closed": _check(
            old.authorize(
                _authorization(old, current_tick=12, project_id="project:other")
            ).reason_code
            == "cross_project_boundary"
        ),
    }

    covert_payload = old.to_payload()
    covert_payload.update({"phase": "phase-B", "target_bytes": "answer"})
    try:
        WorkbenchTaskBoundary.from_payload(covert_payload)
    except ValueError as exc:
        checks["covert_teacher_fields_are_rejected"] = _check(
            True,
            reason=str(exc),
        )
    else:
        checks["covert_teacher_fields_are_rejected"] = _check(False)

    checkpoint_after = content_digest(model.checkpoint())
    all_passed = all(item["passed"] for item in checks.values())
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if all_passed else "failed",
        "can_promote": False,
        "contract": {
            "boundary_format": old.format,
            "allowed_context": [
                "project_id",
                "task_id",
                "session_id",
                "language_id",
                "capability_snapshot_id",
                "capability_ids",
                "lifecycle",
                "generation_scope",
                "issued_tick",
                "expires_tick",
                "parent_token_digest",
            ],
            "forbidden_teacher_fields": ["phase", "target_bytes", "answer_map", "prompt"],
            "default_model_routing_changed": False,
        },
        "checks": checks,
        "routes": {
            "active_generation": active.generation_scope,
            "closed_old_replay": old_closed.generation_scope,
            "active_boundary_digest": active.token_digest,
            "closed_old_boundary_digest": old_closed.token_digest,
        },
        "owner_audit": {
            "boundary_owner_digest": content_digest(
                {
                    "old": old.to_payload(),
                    "closed_old": old_closed.to_payload(),
                    "active": active.to_payload(),
                }
            ),
            "checkpoint_before_digest": checkpoint_before,
            "checkpoint_after_digest": checkpoint_after,
            "checkpoint_read_only": checkpoint_before == checkpoint_after,
            "model_state_mutated": False,
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
                "checkpoint_read_only": result["owner_audit"]["checkpoint_read_only"],
                "checks_passed": sum(int(item["passed"]) for item in result["checks"].values()),
                "checks_total": len(result["checks"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
