"""Bounded query/evidence/native-token fusion for an explicit research profile."""

from __future__ import annotations

import math

import torch
from torch import nn


def rms(value: torch.Tensor) -> torch.Tensor:
    return value / (value.square().mean(-1, keepdim=True) + 1e-4).sqrt()


class _FusionBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.width, self.heads = width, heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.up = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        branches, count, width = tokens.shape
        shape = (branches, count, self.heads, width // self.heads)
        query, key, value = self.qkv(rms(tokens)).chunk(3, -1)
        query, key, value = [item.reshape(shape).transpose(1, 2) for item in (query, key, value)]
        score = query @ key.transpose(-1, -2) / math.sqrt(width // self.heads)
        attention = score.masked_fill(~valid[None, None, None], -torch.inf).softmax(-1)
        message = (attention @ value).transpose(1, 2).reshape(branches, count, width)
        tokens = tokens + self.output(message)
        result: torch.Tensor = tokens + self.down(torch.nn.functional.gelu(self.up(rms(tokens))))
        return result


class JointTokenFusion(nn.Module):
    """Fuse local query, public evidence and branch-local H before pooling."""

    def __init__(self, width: int, heads: int = 2) -> None:
        super().__init__()
        if width < 1 or heads < 1 or width % heads:
            raise ValueError("joint_token_head_shape")
        self.width, self.heads = width, heads
        self.roles = nn.Parameter(torch.randn(3, width) * 0.02)
        self.metadata = nn.Linear(8, width, bias=False)
        self.blocks = nn.ModuleList([_FusionBlock(width, heads) for _ in range(2)])
        self.pool_score = nn.Linear(width, 1, bias=False)
        self.pool_value = nn.Linear(width, width, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        query: torch.Tensor,
        evidence: torch.Tensor,
        metadata: torch.Tensor,
        valid: torch.Tensor,
        native_h: torch.Tensor,
    ) -> torch.Tensor:
        width = self.width
        if (
            query.ndim != 2
            or query.shape[-1] != width
            or evidence.ndim != 2
            or evidence.shape[-1] != width
            or metadata.shape != (evidence.shape[0], 8)
            or valid.shape != (evidence.shape[0],)
            or valid.dtype != torch.bool
            or native_h.ndim != 3
            or native_h.shape[-1] != width
        ):
            raise ValueError("joint_token_input_shape")
        branches, groups = native_h.shape[:2]
        if not 1 <= branches <= 8 or not 1 <= groups <= 16:
            raise ValueError("joint_token_native_budget_stop")
        if query.shape[0] > 512 or evidence.shape[0] > 128:
            raise ValueError("joint_token_sequence_budget_stop")
        count = query.shape[0] + evidence.shape[0] + groups
        if branches * self.heads * count * count > 8_388_608:
            raise ValueError("joint_token_attention_budget_stop")
        reference = self.roles
        if (
            any(
                item.dtype != reference.dtype or item.device != reference.device
                for item in (query, evidence, metadata, native_h)
            )
            or valid.device != reference.device
        ):
            raise ValueError("joint_token_precision_mismatch")
        if not all(
            bool(torch.isfinite(item).all())
            for item in (query, native_h, evidence[valid], metadata[valid])
        ):
            raise ValueError("joint_token_nonfinite_input")
        if query.shape[0] == 0:
            return native_h.new_zeros(branches, width)
        clean_evidence = torch.where(valid[:, None], evidence, torch.zeros_like(evidence))
        clean_metadata = torch.where(valid[:, None], metadata, torch.zeros_like(metadata))
        query_tokens = (query + self.roles[0])[None].expand(branches, -1, -1)
        evidence_tokens = (clean_evidence + self.metadata(clean_metadata) + self.roles[1])[
            None
        ].expand(branches, -1, -1)
        tokens = torch.cat([query_tokens, evidence_tokens, native_h + self.roles[2]], 1)
        key_valid = torch.cat([valid.new_ones(query.shape[0]), valid, valid.new_ones(groups)])
        for block in self.blocks:
            tokens = block(tokens, key_valid)
            if not bool(torch.isfinite(tokens).all()):
                raise ValueError("joint_token_nonfinite_activity")
        question_tokens = rms(tokens[:, : query.shape[0]])
        attention = self.pool_score(question_tokens).softmax(1)
        pooled: torch.Tensor = (attention * self.pool_value(question_tokens)).sum(1)
        if not bool(torch.isfinite(pooled).all()):
            raise ValueError("joint_token_nonfinite_readout")
        return pooled / (1 + pooled.square().mean(-1, keepdim=True)).sqrt()
