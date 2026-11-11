"""Contract guard for 10.7f: the state+goal readout starts tied, has 1,536 params,
can learn a toy three-way choice, and the verdict table names every branch."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "verify_taiji_selection_10_7f.py"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taiji.goal_state_readout import StateGoalActionReadout  # noqa: E402


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SELECTION = _load("verify_taiji_selection_10_7f", SCRIPT)


def test_starts_tied_with_1536_parameters():
    head = StateGoalActionReadout(seed=0)
    assert head.parameter_count() == 1536
    goal = torch.ones(128, dtype=torch.float64)
    state = torch.full((128,), 0.5, dtype=torch.float64)
    actions = [torch.full((128,), float(index), dtype=torch.float64) for index in range(6)]
    scores = [float(head.score(goal, state, action).detach()) for action in actions]
    assert scores == [0.0] * 6


def _toy_rows():
    first = torch.zeros(4, dtype=torch.float64)
    first[0] = 1.0
    second = torch.zeros(4, dtype=torch.float64)
    second[1] = 1.0
    blank = torch.zeros(4, dtype=torch.float64)
    rows = []
    for goal, label in ((first, 0), (second, 1)):
        actions = [first, second, blank]
        rows.append(
            {
                "goal": goal,
                "state": blank,
                "actions": actions,
                "label": label,
                "moves": [["a", "x"], ["b", "x"], ["c", "x"]],
                "goal_text": "toy",
                "correct_move": ["a", "x"] if label == 0 else ["b", "x"],
            }
        )
    return rows


def test_toy_choice_is_learnable_from_one_sided_zero_init():
    head, _ = SELECTION.fit(_toy_rows(), mismatch=False, steps=400, lr=0.1, seed=0, width=4, rank=2)
    with torch.no_grad():
        picks = []
        for row in _toy_rows():
            logits = torch.stack(
                [head.score(row["goal"], row["state"], action) for action in row["actions"]]
            )
            picks.append(int(torch.argmax(logits)))
    assert picks == [0, 1], picks


def test_status_table():
    status = SELECTION.status_from
    assert (
        status(budget_ok=True, frozen=True, control_top1=2, top1=6) == "readout_selection_supported"
    )
    assert (
        status(budget_ok=True, frozen=True, control_top1=2, top1=5) == "readout_selection_supported"
    )
    assert status(budget_ok=True, frozen=True, control_top1=2, top1=2) == "readout_selection_absent"
    assert (
        status(budget_ok=True, frozen=True, control_top1=2, top1=4)
        == "readout_selection_inconclusive"
    )
    assert status(budget_ok=True, frozen=True, control_top1=3, top1=6) == "control_not_null"
    assert status(budget_ok=True, frozen=False, control_top1=2, top1=6) == "backbone_mutated"
    assert status(budget_ok=False, frozen=False, control_top1=6, top1=6) == "budget_stop"
