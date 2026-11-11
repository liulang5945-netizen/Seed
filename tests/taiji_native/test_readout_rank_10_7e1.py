"""Contract guard for 10.7e-1: the all-actions ranking helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "probe_readout_rank_10_7e1.py"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROBE = _load("probe_readout_rank_10_7e1", SCRIPT)


class _OneHotHead:
    def score(self, goal, action):
        return torch.dot(goal, action)


class _OneHotEncoder:
    """Goal ``g`` maps to e0; action ``把o放到p`` maps to a vector keyed by (o, p)."""

    def __init__(self, table):
        self.table = table

    def goal_vector(self, raw):
        return self.table[raw]

    def context(self, raw):
        vector = self.table[raw]
        return torch.stack([vector])


def test_rank_marks_the_matching_action_first():
    table = {}
    probe = PROBE
    objects = probe.OBJECTS
    places = probe.PLACES
    goal = "目标"
    table[goal.encode()] = torch.zeros(6).index_fill(0, torch.tensor([0]), 1.0)
    for index, (obj, place) in enumerate((o, p) for o in objects for p in places):
        vector = torch.zeros(6)
        vector[index] = 1.0
        table[f"把{obj}放到{place}".encode()] = vector
    encoder = _OneHotEncoder(table)
    correct = (objects[0], places[0])
    result = probe.rank_goal(_OneHotHead(), encoder, goal, correct)
    assert result["rank"] == 1
    assert result["top1"] is True
    assert result["top_move"] == list(correct)


def test_rank_reports_a_lower_position():
    table = {}
    probe = PROBE
    goal = "目标"
    table[goal.encode()] = torch.zeros(6).index_fill(0, torch.tensor([0]), 1.0)
    for index, (obj, place) in enumerate((o, p) for o in probe.OBJECTS for p in probe.PLACES):
        vector = torch.zeros(6)
        if index == 1:
            vector[0] = 1.0
        table[f"把{obj}放到{place}".encode()] = vector
    encoder = _OneHotEncoder(table)
    correct = (probe.OBJECTS[0], probe.PLACES[0])
    result = probe.rank_goal(_OneHotHead(), encoder, goal, correct)
    assert result["rank"] == 2
    assert result["top1"] is False
