"""A record-reading organ that lives inside the subject.

The word channel proved (E5.6) that a correction can be answered from a released
record without touching the parent, but it lived outside the model.  This organ puts
the same binding inside the subject: it reads the record's characters, binds one
two-character window, and emits that window's bytes in order.

Two properties are deliberate:

* every tensor here is new, so attaching the organ leaves the parent's weights
  bit-identical (the runner checks a digest of the parent keys);
* the organ answers only when the episode carries a record, and the routing lives in
  ``_WorldKernel.output`` rather than in the harness.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def spans_of(record: bytes) -> list[int]:
    """Byte offsets of character starts."""
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
    """Bind one word from the record and hand its bytes out one at a time."""

    def __init__(self, emb: int = 16, width: int = 32) -> None:
        super().__init__()
        self.byte = nn.Embedding(256, emb)
        self.query = nn.Linear(2 * emb, width)
        self.key = nn.Linear(emb, width)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.proj = nn.Linear(emb, emb)

    def select(self, request: bytes, record: bytes) -> bytes:
        windows = windows_of(record)
        if not windows:
            return b""
        pieces = [torch.tensor(list(window), dtype=torch.long) for window in windows]
        keys = self.key(torch.stack([self.byte(piece).mean(0) for piece in pieces]))
        text = self.byte(torch.tensor(list(request), dtype=torch.long))
        pooled = torch.cat([text.mean(0), text.amax(0)], -1)
        query = self.query(pooled)
        scores = keys @ query / math.sqrt(query.shape[-1])
        return windows[int(scores.argmax())]

    def relevance_logits(self, request: bytes, record: bytes) -> torch.Tensor:
        """Containment score: the best match of the asked entity against the record.

        Pooling the whole record averages the entity away against the template, so
        the score is the maximum over record positions of the entity vector.
        """
        # Every character of the asked entity must appear somewhere in the record:
        # min over entity characters of max over record positions.  Averaging the
        # entity (or taking a single max) cannot tell 绿色圆片 from 紫色圆片, which
        # differ in one character only.
        body = self.byte(torch.tensor(list(record), dtype=torch.long))
        spans = spans_of(request)
        pieces = [
            request[start : (spans[i + 1] if i + 1 < len(spans) else len(request))]
            for i, start in enumerate(spans)
        ]
        per_char = [
            self.byte(torch.tensor(list(piece), dtype=torch.long)).mean(0) for piece in pieces
        ]
        projected = self.proj(body)
        matched = torch.stack([(projected @ self.proj(vector)).max() for vector in per_char])
        return (self.scale * matched.min()).reshape(1)

    def relevant(self, request: bytes, record: bytes) -> float:
        return float(self.relevance_logits(request, record).sigmoid())

    def log_distribution(self, window: bytes, offset: int) -> torch.Tensor:
        """One-hot log distribution: the organ decides, the parent head is bypassed."""
        logits = torch.full((257,), -1e4)
        if offset < len(window):
            logits[window[offset]] = 0.0
        else:
            logits[256] = 0.0
        return logits.log_softmax(-1)
