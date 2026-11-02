"""Contract guard for 10.13: the state readout starts tied and counts its parameters."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taiji.state_action_readout import StateActionReadout  # noqa: E402


def test_starts_tied_and_counts_parameters():
    head = StateActionReadout(state_width=8, action_width=4, rank=2, seed=0)
    assert head.parameter_count() == 8 * 2 + 4 * 2
    state = torch.ones(8, dtype=torch.float64)
    actions = [torch.full((4,), float(index), dtype=torch.float64) for index in range(6)]
    scores = [float(head.score(state, action).detach()) for action in actions]
    assert scores == [0.0] * 6
