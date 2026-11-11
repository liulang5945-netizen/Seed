"""Contract guard for 10.7e: the readout starts silent, has 1,024 params, can learn a toy
binding, and the verdict table names every branch."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "verify_taiji_readout_10_7e.py"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taiji.goal_action_readout import BilinearReadout  # noqa: E402


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


READOUT = _load("verify_taiji_readout_10_7e", SCRIPT)


def test_starts_silent_and_counts_1024_parameters():
    head = BilinearReadout(seed=0)
    assert head.parameter_count() == 1024
    goal = torch.ones(128, dtype=torch.float64)
    action = torch.full((128,), 0.5, dtype=torch.float64)
    assert float(head.score(goal, action).detach()) == 0.0
    other = torch.full((128,), -2.0, dtype=torch.float64)
    assert float(head.score(goal, other).detach()) == 0.0


def _toy_rows():
    first = torch.zeros(8, dtype=torch.float64)
    first[0] = 1.0
    second = torch.zeros(8, dtype=torch.float64)
    second[1] = 1.0
    rows = []
    for index in range(4):
        goal, correct, other = (first, first, second) if index % 2 == 0 else (second, second, first)
        rows.append({"goal": goal, "correct": correct, "other": other})
    return rows


def test_one_sided_zero_init_is_not_a_saddle():
    head, _ = READOUT.fit(_toy_rows(), flip=False, steps=300, lr=0.1, width=8, rank=2)
    with torch.no_grad():
        margins = [
            float(head.score(row["goal"], row["correct"]) - head.score(row["goal"], row["other"]))
            for row in _toy_rows()
        ]
    assert all(margin > 0 for margin in margins), margins


def test_status_table():
    status = READOUT.status_from
    assert (
        status(budget_ok=True, frozen=True, control_hits=3, arm_hits=6)
        == "readout_binding_supported"
    )
    assert (
        status(budget_ok=True, frozen=True, control_hits=3, arm_hits=5)
        == "readout_binding_supported"
    )
    assert (
        status(budget_ok=True, frozen=True, control_hits=3, arm_hits=3) == "readout_binding_absent"
    )
    assert status(budget_ok=True, frozen=True, control_hits=3, arm_hits=4) == "readout_inconclusive"
    assert status(budget_ok=True, frozen=True, control_hits=5, arm_hits=6) == "control_not_null"
    assert status(budget_ok=True, frozen=False, control_hits=3, arm_hits=6) == "backbone_mutated"
    assert status(budget_ok=False, frozen=False, control_hits=6, arm_hits=6) == "budget_stop"
