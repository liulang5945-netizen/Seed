"""Two slots read through the subject's own episodic read, then a factorised score.

``slot`` is ``EpisodicRead(state.h, goal_vector(descriptor), view(records))`` -- the
model's own retrieval, not a new one. The head only adds two low-rank projections that
map a slot onto an object or a place representation:

    score(action) = bilinear(slot_obj, action_obj) + bilinear(slot_place, action_place)

Nothing here is trained yet; the first job is the channel check: the slots must move
when the glossary mapping changes, otherwise they are reading a string and not a table.
"""

from __future__ import annotations

import torch
from torch import nn


class SlotCompositionHead(nn.Module):
    def __init__(self, width: int, rank: int = 4, seed: int = 0) -> None:
        super().__init__()
        self.width = width
        self.rank = rank
        generator = torch.Generator().manual_seed(seed)
        left = torch.empty(width, rank, dtype=torch.float64)
        left.normal_(mean=0.0, std=0.01, generator=generator)
        self.left = nn.Parameter(left)
        self.right = nn.Parameter(torch.zeros(width, rank, dtype=torch.float64))

    def score(self, slot: torch.Tensor, action_part: torch.Tensor) -> torch.Tensor:
        return (slot @ self.left) @ (action_part @ self.right)

    def parameter_count(self) -> int:
        return self.left.numel() + self.right.numel()


def read_slot(model, state, descriptor: str, records):
    """One slot: the subject's own episodic read keyed by one descriptor."""
    view = model.source_view(tuple(records), "owner", "scope", 0)
    goal = model.encoder.goal_vector(descriptor.encode())
    with torch.no_grad():
        recalled, attention = model.episodic(state.h, goal, view)
    return recalled.detach(), attention.detach()


def relative_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    scale = float(first.norm()) + float(second.norm())
    if scale == 0.0:
        return 0.0
    return 2.0 * float((first - second).norm()) / scale
