"""A zero-start readout that scores an action from the goal **and** the state.

``score(goal, state, action) = ([g; s]ᵀU)·(aᵀV)`` with ``U`` of shape
``[2*width, rank]`` and ``V`` of shape ``[width, rank]``. ``V`` starts at zero, so
every action ties at the start; ``U`` starts small-random so the zero factor still
receives a non-zero gradient.

Only these 1,536 parameters are meant to be trained; the subject is frozen.
"""

from __future__ import annotations

import torch
from torch import nn


class StateGoalActionReadout(nn.Module):
    def __init__(self, width: int = 128, rank: int = 4, seed: int = 0) -> None:
        super().__init__()
        self.width = width
        self.rank = rank
        generator = torch.Generator().manual_seed(seed)
        left = torch.empty(2 * width, rank, dtype=torch.float64)
        left.normal_(mean=0.0, std=0.01, generator=generator)
        self.left = nn.Parameter(left)
        self.right = nn.Parameter(torch.zeros(width, rank, dtype=torch.float64))

    def score(self, goal: torch.Tensor, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return (torch.cat([goal, state]) @ self.left) @ (action @ self.right)

    def parameter_count(self) -> int:
        return self.left.numel() + self.right.numel()
