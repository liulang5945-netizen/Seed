"""Request-conditioned public evidence reading for the U2 P3 candidate.

This module defines the reader. Subject history admission and recovery migration
are separate integration responsibilities specified in the P3 contract.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .u2_subject import bounded


class QueryEvidenceReader(nn.Module):
    """Two shared cross-attention steps over provenance-admitted evidence tokens."""

    def __init__(self, width: int, heads: int = 2) -> None:
        super().__init__()
        if width < 1 or heads < 1 or width % heads:
            raise ValueError("evidence_reader_head_shape")
        self.width = width
        self.heads = heads
        self.input_query = nn.Linear(2 * width, width)
        self.key = nn.Linear(width + 8, width, bias=False)
        self.value = nn.Linear(width + 8, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)
        self.message = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(
        self,
        request: torch.Tensor,
        h: torch.Tensor,
        context: torch.Tensor,
        encoded: torch.Tensor,
        metadata: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        width, heads = self.width, self.heads
        if (
            request.shape != (width,)
            or h.ndim != 2
            or h.shape[-1] != width
            or context.shape != h.shape
            or encoded.ndim != 2
            or encoded.shape[-1] != width
            or metadata.shape != (encoded.shape[0], 8)
            or valid.shape != (encoded.shape[0],)
            or valid.dtype != torch.bool
        ):
            raise ValueError("evidence_reader_input_shape")
        if encoded.shape[0] > 128:
            raise ValueError("evidence_reader_token_budget_stop")
        tensors = (request, h, context, encoded, metadata)
        if any(t.dtype != context.dtype or t.device != context.device for t in tensors):
            raise ValueError("evidence_reader_precision_mismatch")
        if valid.device != context.device:
            raise ValueError("evidence_reader_mask_device")
        if not all(bool(torch.isfinite(t).all()) for t in (request, h, context)):
            raise ValueError("evidence_reader_nonfinite_query")
        if not bool(torch.isfinite(encoded[valid]).all() and torch.isfinite(metadata[valid]).all()):
            raise ValueError("evidence_reader_nonfinite_evidence")
        branches, count = h.shape[0], encoded.shape[0]
        if not bool(valid.any()):
            return context, context.new_zeros(2, branches, heads, count), False
        # Remove invalid values before projection, including NaNs in absent tokens.
        source = torch.cat([encoded, metadata], -1)
        source = torch.where(valid[:, None], source, torch.zeros_like(source))
        head_width = width // heads
        keys = self.key(source).reshape(count, heads, head_width)
        values = self.value(source).reshape(count, heads, head_width)
        initial = self.input_query(torch.cat([request.expand(branches, -1), h], -1))
        current = initial
        attentions = []
        for _ in range(2):
            query = self.query(current).reshape(branches, heads, head_width)
            score = torch.einsum("bhd,nhd->bhn", query, keys) / math.sqrt(head_width)
            attention = score.masked_fill(~valid[None, None], -torch.inf).softmax(-1)
            message = torch.einsum("bhn,nhd->bhd", attention, values).reshape(branches, width)
            current = current + bounded(self.message(message))
            attentions.append(attention)
        return context + 0.1 * self.output(current - initial), torch.stack(attentions), True
