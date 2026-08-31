"""Taiji-owned natural-language Workbench protocol orchestration.

This module owns the product-facing plan/approval/execute boundary.  The
runtime remains the cognitive owner of grounding and execution, while this
orchestrator keeps transport lifecycle concerns out of the large runtime
facade.  Plans are intentionally process-local and are not executable after
restart; checkpoint recovery belongs to the runtime's normal action outcome
path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any


class NaturalLanguageWorkbenchOrchestrator:
    """Own the two-phase Taiji Workbench task protocol."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def execute(
        self,
        prompt: str,
        semantic_evidence: Any,
        *,
        snapshot_id: str,
        parameter_bindings: Any = None,
        loop_id: str,
        max_steps: int = 1,
        max_budget_units: float = 1.0,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
        learn: bool = False,
    ) -> dict[str, Any]:
        """Keep the legacy direct facade while routing through the protocol owner."""

        kwargs = {
            "snapshot_id": snapshot_id,
            "loop_id": loop_id,
            "max_steps": max_steps,
            "max_budget_units": max_budget_units,
            "novelty": novelty,
            "resource_budget": resource_budget,
            "learn": learn,
        }
        if parameter_bindings is not None:
            kwargs["parameter_bindings"] = parameter_bindings
        return self.runtime._execute_natural_language_workbench_task_impl(
            prompt,
            semantic_evidence,
            **kwargs,
        )

    def plan(
        self,
        prompt: str,
        semantic_evidence: Any,
        *,
        snapshot_id: str,
        loop_id: str,
        max_steps: int = 1,
        max_budget_units: float = 1.0,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> dict[str, Any]:
        """Create a Taiji-owned plan without executing or auto-approving it."""

        return self.runtime._execute_natural_language_workbench_task_impl(
            prompt,
            semantic_evidence,
            snapshot_id=snapshot_id,
            loop_id=loop_id,
            max_steps=max_steps,
            max_budget_units=max_budget_units,
            novelty=novelty,
            resource_budget=resource_budget,
            prepare_only=True,
        )

    def approve(self, plan_id: str, request_id: str) -> dict[str, Any]:
        """Issue one idempotent exact approval token for a plan request."""

        runtime = self.runtime
        with runtime._lock:
            plan = runtime._pending_workbench_plans.get(str(plan_id))
            if plan is None:
                raise ValueError("natural-language Workbench plan is unknown or expired")
            environment = runtime._sync_workbench_root()
            if int(plan["tick"]) != int(runtime.model.tick):
                raise ValueError("natural-language Workbench plan is stale")
            request = next(
                (
                    item
                    for item in plan["requests"]
                    if item.request_id == str(request_id)
                ),
                None,
            )
            if request is None:
                raise ValueError("natural-language Workbench request is not in the plan")
            existing = plan["approvals"].get(request.request_id)
            if existing is not None:
                return dict(existing)
            policy = environment.policy_for(request)
            approval = environment.issue_approval(request)
            result = {
                "format": "seed-natural-language-workbench-approval-v1",
                "plan_id": str(plan_id),
                "request_id": request.request_id,
                "capability_id": request.capability_id,
                "policy": policy.to_payload(),
                "preview": approval["preview"],
                "approval_token": approval["approval_token"],
                "expires_in_seconds": approval["expires_in_seconds"],
            }
            plan["approvals"][request.request_id] = dict(result)
            return result

    def execute_planned(
        self,
        plan_id: str,
        approval_tokens: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute only a current plan after exact request approvals."""

        runtime = self.runtime
        with runtime._lock:
            plan = runtime._pending_workbench_plans.get(str(plan_id))
            if plan is None:
                raise ValueError("natural-language Workbench plan is unknown or expired")
            if int(plan["tick"]) != int(runtime.model.tick):
                raise ValueError("natural-language Workbench plan is stale")
            if approval_tokens is None:
                approval_tokens = {}
            if not isinstance(approval_tokens, Mapping):
                raise TypeError("planned Workbench approval tokens must be a mapping")
            from seed_platform.workbench import WorkbenchActionRequest

            environment = runtime._sync_workbench_root()
            if str(plan["snapshot_id"]) != environment.capability_snapshot.snapshot_id:
                raise ValueError("natural-language Workbench plan capability snapshot drifted")
            requests = tuple(
                replace(
                    request,
                    approval_token=str(approval_tokens.get(request.request_id, "") or ""),
                )
                for request in plan["requests"]
            )
            if any(not isinstance(request, WorkbenchActionRequest) for request in requests):
                raise TypeError("natural-language Workbench plan contains invalid requests")
            preflight = runtime.preflight_workbench_loop(
                requests,
                loop_id=str(plan["loop_id"]),
                max_steps=int(plan["max_steps"]),
                max_budget_units=float(plan["max_budget_units"]),
            )
            base = dict(plan["base"])
            base["plan_id"] = str(plan_id)
            base["preflight"] = preflight
            if not preflight.get("accepted"):
                base["status"] = "rejected"
                base["reason_code"] = str(preflight.get("error_code", "preflight_rejected"))
                return base
            runtime._pending_workbench_plans.pop(str(plan_id), None)
            execution = runtime.execute_preflighted_workbench_loop(
                plan["intents"],
                requests,
                loop_id=str(plan["loop_id"]),
                preflight_id=str(preflight["preflight_id"]),
                max_steps=int(plan["max_steps"]),
                max_budget_units=float(plan["max_budget_units"]),
                learn=bool(plan["learn"]),
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
