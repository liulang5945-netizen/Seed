"""A zero-start bilinear readout over the subject's own goal and action features.

The head is deliberately tiny: ``score = (gᵀU)·(aᵀV)`` with ``U, V`` of shape
``[width, rank]``. One factor starts at zero, so every score is exactly 0 and the
candidate behaves bit-identically until the head is trained; the other factor
starts small-random, otherwise the zero factor would receive a zero gradient
(``∂score/∂V = (gᵀU)a`` vanishes when both factors are zero) and nothing could
ever move.

Only these two parameters are meant to be trained; the subject is frozen.
"""

from __future__ import annotations

import torch
from torch import nn


class BilinearReadout(nn.Module):
    def __init__(self, width: int = 128, rank: int = 4, seed: int = 0) -> None:
        super().__init__()
        self.width = width
        self.rank = rank
        generator = torch.Generator().manual_seed(seed)
        left = torch.empty(width, rank, dtype=torch.float64)
        left.normal_(mean=0.0, std=0.01, generator=generator)
        self.left = nn.Parameter(left)
        self.right = nn.Parameter(torch.zeros(width, rank, dtype=torch.float64))

    def score(self, goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return (goal @ self.left) @ (action @ self.right)

    def parameter_count(self) -> int:
        return self.left.numel() + self.right.numel()
