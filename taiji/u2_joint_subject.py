"""Typed shared token fusion in the normal Native transition and expression paths."""

from __future__ import annotations

import copy
from typing import ClassVar

import torch

from .u2_evidence_runtime import EvidenceRuntime
from .u2_history import value_digest
from .u2_joint_tokens import JointTokenFusion
from .u2_request_state import active_request
from .u2_subject import U2Subject
from .u2_task_goal import ActiveQuerySubject


class _JointSubject(ActiveQuerySubject):
    joint_mode = "abstract"

    def __init__(self, config):
        if self.joint_mode not in {"query", "goal-control"}:
            raise ValueError("joint_subject_profile_required")
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026111307)
            self.joint_fusion = JointTokenFusion(config.d)
        self.fusion_parent: str | None = None

    @classmethod
    def from_parent(cls, parent):
        if type(parent) is not ActiveQuerySubject:
            raise ValueError("joint_subject_parent_profile_mismatch")
        candidate = cls(parent.c).to(parent.native_w)
        missing, unexpected = candidate.load_state_dict(parent.state_dict(), strict=False)
        expected = {f"joint_fusion.{name}" for name in candidate.joint_fusion.state_dict()}
        if set(missing) != expected or unexpected:
            raise ValueError("joint_subject_parent_layout_mismatch")
        if list(dict(candidate.named_parameters()))[: len(list(parent.parameters()))] != list(
            dict(parent.named_parameters())
        ):
            raise ValueError("joint_subject_parameter_order_mismatch")
        candidate.parameter_revision = parent.parameter_revision + 1
        candidate.generation = parent.generation
        candidate.unit_ids = copy.deepcopy(parent.unit_ids)
        candidate.unit_branches = copy.deepcopy(parent.unit_branches)
        candidate.projection_parent = parent.projection_parent
        candidate.fusion_parent = value_digest(parent.state_dict())
        return candidate

    def joint_feature(self, state):
        if "joint_tokens" in getattr(self, "_verification_cuts", ()):
            return torch.zeros_like(state["H"][:, 0])
        request = active_request(state)
        raw = (
            state["answer_request"]["raw"]
            if request and self.joint_mode == "query"
            else state["goal_raw"]
        )
        query = self.text_units(raw)
        evidence, metadata = self.evidence_tokens(state)
        valid = torch.ones(evidence.shape[0], dtype=torch.bool, device=evidence.device)
        return self.joint_fusion(query, evidence, metadata, valid, state["H"])

    def native(self, state, goal, action, event):
        if "joint_native" not in getattr(self, "_verification_cuts", ()):
            event = event + self.joint_feature(state)
        return super().native(state, goal, action, event)

    def branch_context(self, state, goal, action):
        original = super().branch_context(state, goal, action)
        if "joint_expression" in getattr(self, "_verification_cuts", ()):
            return original
        return original + self.joint_feature(state)

    def export_bundle(self, state, optimizer_state=None):
        bundle = super().export_bundle(state, optimizer_state)
        return {
            **bundle,
            "schema": f"taiji-u2-joint-token-{self.joint_mode}-v1",
            "joint_tokens": {
                "mode": self.joint_mode,
                "parent_parameter_digest": self.fusion_parent,
                "owner": "Theta",
            },
        }

    @classmethod
    def from_bundle(cls, bundle):
        if bundle["schema"] != f"taiji-u2-joint-token-{cls.joint_mode}-v1":
            raise ValueError("joint_subject_schema_mismatch")
        metadata = bundle["joint_tokens"]
        if (
            set(metadata) != {"mode", "parent_parameter_digest", "owner"}
            or metadata["mode"] != cls.joint_mode
            or metadata["owner"] != "Theta"
        ):
            raise ValueError("joint_subject_registry_mismatch")
        parent = metadata["parent_parameter_digest"]
        if parent is not None and (not isinstance(parent, str) or len(parent) != 64):
            raise ValueError("joint_subject_ancestry_mismatch")
        inner = {key: value for key, value in bundle.items() if key != "joint_tokens"}
        model, state = super().from_bundle(
            {**inner, "schema": "taiji-u2-task-goal-active-query-v1"}
        )
        model.fusion_parent = parent
        model.export_bundle(state)
        return model, state


class JointQuerySubject(_JointSubject):
    joint_mode = "query"


class JointGoalSubject(_JointSubject):
    joint_mode = "goal-control"


class JointQueryRuntime(EvidenceRuntime):
    subject_class: ClassVar[type[U2Subject]] = JointQuerySubject

    def __init__(self, model, *args, **kwargs):
        if type(model) is not JointQuerySubject:
            raise ValueError("joint_query_runtime_profile_required")
        super().__init__(model, *args, **kwargs)


class JointGoalRuntime(EvidenceRuntime):
    subject_class: ClassVar[type[U2Subject]] = JointGoalSubject

    def __init__(self, model, *args, **kwargs):
        if type(model) is not JointGoalSubject:
            raise ValueError("joint_goal_runtime_profile_required")
        super().__init__(model, *args, **kwargs)
