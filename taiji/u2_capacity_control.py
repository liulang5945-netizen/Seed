"""Parameter-matched request/H residual control for the direct evidence reader."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from .u2_evidence_runtime import EvidenceRuntime
from .u2_evidence_subject import SCHEMA as EVIDENCE_SCHEMA
from .u2_evidence_subject import EvidenceSubject
from .u2_subject import U2Subject

SCHEMA = "taiji-u2-capacity-control-subject-v1"


class CapacityResidual(nn.Module):
    """584 active scalars at width eight, conditioned on the existing H reading."""

    def __init__(self, width: int):
        super().__init__()
        if width != 8:
            raise ValueError("capacity_control_is_width_eight_only")
        self.input = nn.Linear(2 * width, 23)
        self.output = nn.Linear(23, width)
        self.scale_logit = nn.Parameter(torch.zeros(()))

    def forward(self, request, h, context, encoded, metadata, valid):
        if request.shape != (8,) or h.ndim != 2 or h.shape[-1] != 8 or context.shape != h.shape:
            raise ValueError("capacity_control_input_shape")
        joined = torch.cat([request.expand(h.shape[0], -1), h], -1)
        change = self.output(self.input(joined).tanh())
        result = context + 0.2 * self.scale_logit.sigmoid() * change
        # The shared subject performs evidence admission. This control reads request/H.
        return result, context.new_empty(0), bool(valid.any())


class CapacitySubject(EvidenceSubject):
    def __init__(self, config):
        super().__init__(config)
        self.add_module("evidence_reader", CapacityResidual(config.d))

    def export_bundle(self, state, optimizer_state=None):
        return {**super().export_bundle(state, optimizer_state), "schema": SCHEMA}

    @classmethod
    def from_bundle(cls, bundle):
        if bundle["schema"] != SCHEMA:
            raise ValueError("capacity_subject_schema_mismatch")
        return super().from_bundle({**bundle, "schema": EVIDENCE_SCHEMA})


class CapacityRuntime(EvidenceRuntime):
    subject_class: ClassVar[type[U2Subject]] = CapacitySubject

    def __init__(self, model, *args, **kwargs):
        if not isinstance(model, CapacitySubject):
            raise ValueError("capacity_runtime_requires_capacity_subject")
        super().__init__(model, *args, **kwargs)
