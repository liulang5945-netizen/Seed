"""Byte-level value encoder for the N177 F candidate.

The encoder sees only the UTF-8 bytes of a producer-extracted value span. The
answer path may use its output, but never the raw value bytes directly; the
copy-only arm is the separate lower-bound control.
"""

from __future__ import annotations

import torch
from torch import nn

from .joint_state_consumption_course import E1_COLORS

VALUE_ENCODER_FORMAT = "taiji-n177-f-value-byte-encoder-v1"


class ValueByteEncoderV1(nn.Module):
    """Embedding + GRU over value-span bytes; output is a fixed continuous vector."""

    def __init__(self, *, byte_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        if byte_dim <= 0 or hidden_dim <= 0:
            raise ValueError("value encoder dimensions must be positive")
        self.byte_dim = byte_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(256, byte_dim)
        self.gru = nn.GRU(byte_dim, hidden_dim, batch_first=True)

    def forward(self, value_bytes: torch.Tensor) -> torch.Tensor:
        if value_bytes.ndim == 1:
            rows = value_bytes.reshape(1, -1)
            embedded = self.embedding(rows.long())
            _, hidden = self.gru(embedded)
            last: torch.Tensor = hidden[-1]
            return last.reshape(-1)
        if value_bytes.ndim != 2:
            raise ValueError("value bytes must be a 1-D span or a 2-D batch of spans")
        embedded = self.embedding(value_bytes.long())
        _, hidden = self.gru(embedded)
        last_batch: torch.Tensor = hidden[-1]
        return last_batch

    def encode_text(self, value: str) -> torch.Tensor:
        if value not in E1_COLORS:
            raise ValueError("value encoder received a value outside the frozen colour set")
        raw = torch.tensor(list(value.encode("utf-8")), dtype=torch.long)
        return self.forward(raw)


class FeatureByteBiasV1(nn.Module):
    """Map a selected content feature to a 256-byte next-logit bias."""

    def __init__(self, *, feature_dim: int = 64, byte_vocab: int = 256) -> None:
        super().__init__()
        if feature_dim <= 0 or byte_vocab <= 0:
            raise ValueError("feature byte bias dimensions must be positive")
        self.feature_dim = feature_dim
        self.byte_vocab = byte_vocab
        self.linear = nn.Linear(feature_dim, byte_vocab)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.numel() != self.feature_dim:
            raise ValueError("feature byte bias input width differs from its declaration")
        bias: torch.Tensor = self.linear(feature.reshape(-1))
        return bias
