"""A record-reading organ with a sentinel gate (Pointer Sentinel, Merity et al. 2017).

The earlier version decided relevance with a separate scalar head trained by BCE and
collapsed to "always copy" or "never copy".  The sentinel construction removes that
degree of freedom: the sentinel is one more candidate inside the same softmax as the
record's windows, so mass given to the sentinel is mass taken from copying, and the
pointer-sum loss lets either side satisfy the target.

Every tensor here is new, so attaching the organ leaves the parent bit-identical.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def spans_of(record: bytes) -> list[int]:
    out, index = [], 0
    while index < len(record):
        out.append(index)
        lead = record[index]
        index += 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
    return out


def windows_of(record: bytes) -> list[bytes]:
    """Two-character windows: the answer unit in this world is a two-character word."""
    spans = spans_of(record)
    out = []
    for i in range(len(spans) - 1):
        end = spans[i + 2] if i + 2 < len(spans) else len(record)
        out.append(record[spans[i] : end])
    return out


class RecordOrgan(nn.Module):
    def __init__(self, emb: int = 16, width: int = 32) -> None:
        super().__init__()
        self.byte = nn.Embedding(256, emb)
        self.query = nn.Linear(2 * emb, width)
        self.key = nn.Linear(emb, width)
        self.sentinel = nn.Parameter(torch.zeros(width))
        # The gate must see how confident the pointer is, not only the query.
        self.gain = nn.Parameter(torch.zeros(()))

    def attention(self, request: bytes, record: bytes) -> torch.Tensor:
        """Softmax over [window scores; sentinel]; the last entry is the gate."""
        windows = windows_of(record)
        if not windows:
            return torch.tensor([1.0])
        text = self.byte(torch.tensor(list(request), dtype=torch.long))
        pooled = torch.cat([text.mean(0), text.amax(0)], -1)
        query = self.query(pooled)
        scale = math.sqrt(query.shape[-1])
        keys = self.key(
            torch.stack(
                [
                    self.byte(torch.tensor(list(window), dtype=torch.long)).mean(0)
                    for window in windows
                ]
            )
        )
        scores = keys @ query / scale
        sentinel = (query @ self.sentinel) / scale + self.gain * scores.max()
        return torch.cat([scores, sentinel.reshape(1)]).softmax(-1)

    def route(self, request: bytes, record: bytes) -> bytes | None:
        """Return the bound window, or None when the sentinel wins (back off)."""
        attention = self.attention(request, record)
        windows = windows_of(record)
        if float(attention[-1]) > 0.5 or not windows:
            return None
        return windows[int(attention[:-1].argmax())]

    def log_distribution(self, window: bytes, offset: int) -> torch.Tensor:
        logits = torch.full((257,), -1e4)
        if offset < len(window):
            logits[window[offset]] = 0.0
        else:
            logits[256] = 0.0
        return logits.log_softmax(-1)
