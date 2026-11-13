"""Typed active-request conditioning of the existing Native recurrence."""

from __future__ import annotations

import copy
from typing import ClassVar

import torch
from torch import nn

from .u2_evidence_runtime import EvidenceRuntime
from .u2_evidence_subject import SCHEMA as EVIDENCE_SCHEMA
from .u2_evidence_subject import EvidenceSubject
from .u2_history import value_digest
from .u2_request_state import active_request
from .u2_subject import U2Subject


class _TaskGoalSubject(EvidenceSubject):
    projection_mode = "abstract"

    def __init__(self, config):
        if self.projection_mode not in {"active-query", "persistent-control"}:
            raise ValueError("task_goal_profile_required")
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            self.native_task_projection = nn.Linear(config.d, config.d, bias=False)
            nn.init.zeros_(self.native_task_projection.weight)
        self.projection_parent: str | None = None

    @classmethod
    def from_parent(cls, parent):
        if type(parent) is not EvidenceSubject:
            raise ValueError("task_goal_parent_profile_mismatch")
        candidate = cls(parent.c).to(dtype=parent.native_w.dtype, device=parent.native_w.device)
        missing, unexpected = candidate.load_state_dict(parent.state_dict(), strict=False)
        if missing != ["native_task_projection.weight"] or unexpected:
            raise ValueError("task_goal_parent_parameter_mismatch")
        names = list(dict(candidate.named_parameters()))
        if names[:-1] != list(dict(parent.named_parameters())) or names[-1] != missing[0]:
            raise ValueError("task_goal_parameter_order_mismatch")
        candidate.parameter_revision = parent.parameter_revision + 1
        candidate.generation = parent.generation
        candidate.unit_ids = copy.deepcopy(parent.unit_ids)
        candidate.unit_branches = copy.deepcopy(parent.unit_branches)
        candidate.projection_parent = value_digest(parent.state_dict())
        return candidate

    def native(self, state, goal, action, event):
        if active_request(state):
            source = (
                self.goal(state["answer_request"]["raw"])
                if self.projection_mode == "active-query"
                else goal
            )
            goal = goal + self.native_task_projection(source)
        return super().native(state, goal, action, event)

    def export_bundle(self, state, optimizer_state=None):
        if not bool(torch.isfinite(self.native_task_projection.weight).all()):
            raise ValueError("nonfinite_task_goal_projection")
        bundle = super().export_bundle(state, optimizer_state)
        return {
            **bundle,
            "schema": f"taiji-u2-task-goal-{self.projection_mode}-v1",
            "goal_projection": {
                "mode": self.projection_mode,
                "parent_parameter_digest": self.projection_parent,
                "parameter": "native_task_projection.weight",
                "owner": "Theta",
            },
        }

    @classmethod
    def from_bundle(cls, bundle):
        if bundle["schema"] != f"taiji-u2-task-goal-{cls.projection_mode}-v1":
            raise ValueError("task_goal_schema_mismatch")
        metadata = bundle["goal_projection"]
        if (
            set(metadata) != {"mode", "parent_parameter_digest", "parameter", "owner"}
            or metadata["mode"] != cls.projection_mode
            or metadata["owner"] != "Theta"
            or metadata["parameter"] != "native_task_projection.weight"
        ):
            raise ValueError("task_goal_registry_mismatch")
        parent = metadata["parent_parameter_digest"]
        if parent is not None and (not isinstance(parent, str) or len(parent) != 64):
            raise ValueError("task_goal_ancestry_mismatch")
        inner = {key: value for key, value in bundle.items() if key != "goal_projection"}
        model, state = super().from_bundle({**inner, "schema": EVIDENCE_SCHEMA})
        model.projection_parent = parent
        model.export_bundle(state)
        return model, state


class ActiveQuerySubject(_TaskGoalSubject):
    projection_mode = "active-query"


class PersistentGoalSubject(_TaskGoalSubject):
    projection_mode = "persistent-control"


class ActiveQueryRuntime(EvidenceRuntime):
    subject_class: ClassVar[type[U2Subject]] = ActiveQuerySubject

    def __init__(self, model, *args, **kwargs):
        if not isinstance(model, ActiveQuerySubject):
            raise ValueError("active_query_runtime_profile_required")
        super().__init__(model, *args, **kwargs)


class PersistentGoalRuntime(EvidenceRuntime):
    subject_class: ClassVar[type[U2Subject]] = PersistentGoalSubject

    def __init__(self, model, *args, **kwargs):
        if not isinstance(model, PersistentGoalSubject):
            raise ValueError("persistent_goal_runtime_profile_required")
        super().__init__(model, *args, **kwargs)
