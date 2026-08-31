"""Taiji-owned boundary between planning and Workbench execution.

Grounding and cognition happen before this module.  This boundary only binds
Taiji ActionIntents to the current Workbench snapshot, performs preflight or
the explicit approval-plan preparation, and publishes the execution outcome.
It deliberately depends on a small runtime protocol instead of importing the
large ``SeedRuntime`` implementation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from seed_platform.workbench import WorkbenchActionRequest


class WorkbenchExecutionBoundary:
    """Own request binding, preflight, approval preparation, and execution."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def run(
        self,
        *,
        base: dict[str, Any],
        planning_steps: Sequence[Mapping[str, Any]],
        intents: Sequence[Any],
        environment: Any,
        loop_id: str,
        max_steps: int,
        max_budget_units: float,
        learn: bool,
        prepare_only: bool,
    ) -> dict[str, Any]:
        """Bind the plan and either prepare approval or execute it."""

        requests = self._build_requests(intents, environment)
        base["planning"] = {
            "status": "planned",
            "steps": list(planning_steps),
            "action_intents": [intent.to_payload() for intent in intents],
        }
        if prepare_only:
            return self._prepare_plan(
                base=base,
                requests=requests,
                intents=intents,
                environment=environment,
                loop_id=loop_id,
                max_steps=max_steps,
                max_budget_units=max_budget_units,
                learn=learn,
            )
        return self._execute(
            base=base,
            intents=intents,
            requests=requests,
            environment=environment,
            loop_id=loop_id,
            max_steps=max_steps,
            max_budget_units=max_budget_units,
            learn=learn,
        )

    @staticmethod
    def _build_requests(
        intents: Sequence[Any],
        environment: Any,
    ) -> tuple[WorkbenchActionRequest, ...]:
        """Bind each Taiji intent to current capability and MCP snapshots."""

        return tuple(
            WorkbenchActionRequest.from_action_intent(
                intent,
                snapshot_id=environment.capability_snapshot.snapshot_id,
                mcp_registry_snapshot_id=(
                    environment.mcp_registry.snapshot_id
                    if str(intent.kind).startswith("mcp.")
                    else ""
                ),
                capability_registry_snapshot_id=environment.capability_registry.snapshot_id,
            )
            for intent in intents
        )

    def _prepare_plan(
        self,
        *,
        base: dict[str, Any],
        requests: Sequence[WorkbenchActionRequest],
        intents: Sequence[Any],
        environment: Any,
        loop_id: str,
        max_steps: int,
        max_budget_units: float,
        learn: bool,
    ) -> dict[str, Any]:
        runtime = self.runtime
        plan_id = "taiji-plan:" + hashlib.sha256(
            (
                f"{loop_id}|{runtime.model.tick}|{environment.capability_snapshot.snapshot_id}|"
                + "|".join(request.request_id for request in requests)
            ).encode("utf-8")
        ).hexdigest()[:24]
        approval_requirements: list[dict[str, Any]] = []
        for index, request in enumerate(requests):
            policy = environment.policy_for(request)
            if policy.decision == "deny":
                base["status"] = "rejected"
                base["reason_code"] = policy.reason_code
                base["preflight"] = {
                    "accepted": False,
                    "error_code": policy.reason_code,
                }
                return base
            if policy.reason_code == "capability_requires_approval":
                approval_requirements.append(
                    {
                        "index": index,
                        "request_id": request.request_id,
                        "capability_id": request.capability_id,
                        "policy": policy.to_payload(),
                        "preview": environment.preview_tool(
                            request.capability_id,
                            request.parameters,
                        ),
                    }
                )
        runtime._pending_workbench_plans[plan_id] = {
            "plan_id": plan_id,
            "snapshot_id": environment.capability_snapshot.snapshot_id,
            "tick": int(runtime.model.tick),
            "loop_id": str(loop_id),
            "max_steps": int(max_steps),
            "max_budget_units": float(max_budget_units),
            "learn": bool(learn),
            "base": dict(base),
            "intents": tuple(intents),
            "requests": tuple(requests),
            "approvals": {},
        }
        base["plan_id"] = plan_id
        base["approval_requirements"] = approval_requirements
        base["status"] = "needs_approval" if approval_requirements else "planned"
        base["reason_code"] = (
            "workbench_approval_required" if approval_requirements else "taiji_plan_ready"
        )
        return base

    def _execute(
        self,
        *,
        base: dict[str, Any],
        intents: Sequence[Any],
        requests: Sequence[WorkbenchActionRequest],
        environment: Any,
        loop_id: str,
        max_steps: int,
        max_budget_units: float,
        learn: bool,
    ) -> dict[str, Any]:
        runtime = self.runtime
        preflight = runtime.preflight_workbench_loop(
            requests,
            loop_id=loop_id,
            max_steps=max_steps,
            max_budget_units=max_budget_units,
        )
        base["preflight"] = preflight
        if not preflight.get("accepted"):
            base["status"] = "rejected"
            base["reason_code"] = str(preflight.get("error_code", "preflight_rejected"))
            return base
        execution = runtime.execute_preflighted_workbench_loop(
            intents,
            requests,
            loop_id=loop_id,
            preflight_id=str(preflight["preflight_id"]),
            max_steps=max_steps,
            max_budget_units=max_budget_units,
            learn=learn,
        )
        side_effects = any(
            bool(step.get("success"))
            and (
                (descriptor := environment.capability_snapshot.get(
                    str(step.get("capability_id", ""))
                ))
                is not None
                and descriptor.risk != "read_only"
            )
            for step in execution.get("steps", ())
        )
        base["execution"] = {
            **execution,
            "side_effects": side_effects,
        }
        base["status"] = str(execution.get("status", "rejected"))
        base["reason_code"] = str(execution.get("error_code", ""))
        return base
