"""A readout that scores an action from the subject's recurrent state.

``score(h, action) = (h @ U)·(action @ V)`` with ``U`` of shape ``[state_width, rank]``
and ``V`` of shape ``[action_width, rank]``. ``V`` starts at zero, so every action ties
at the start; ``U`` starts small-random so the zero factor still gets a gradient.
"""

from __future__ import annotations

import torch
from torch import nn


class StateActionReadout(nn.Module):
    def __init__(
        self, state_width: int, action_width: int = 128, rank: int = 4, seed: int = 0
    ) -> None:
        super().__init__()
        self.state_width = state_width
        self.action_width = action_width
        self.rank = rank
        generator = torch.Generator().manual_seed(seed)
        left = torch.empty(state_width, rank, dtype=torch.float64)
        left.normal_(mean=0.0, std=0.01, generator=generator)
        self.left = nn.Parameter(left)
        self.right = nn.Parameter(torch.zeros(action_width, rank, dtype=torch.float64))

    def score(self, state_feature: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return (state_feature @ self.left) @ (action @ self.right)

    def parameter_count(self) -> int:
        return self.left.numel() + self.right.numel()
