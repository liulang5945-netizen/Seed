"""Contract guard for 10.12: the permutation really changes every mapping."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "verify_taiji_records_state_10_12.py"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STATE = _load("verify_taiji_records_state_10_12", SCRIPT)


def test_permutation_changes_every_pairing():
    real = {source.raw[:-1] for source in STATE.glossary_records()}
    wrong = {source.raw[:-1] for source in STATE.permuted_records()}
    assert len(real) == len(wrong)
    shared = real & wrong
    assert len(shared) < len(real)


def test_relative_distance_edges():
    vector = torch.tensor([1.0, 0.0])
    other = torch.tensor([0.0, 1.0])
    assert STATE.relative_distance(vector, vector) == 0.0
    assert abs(STATE.relative_distance(vector, other) - 2**0.5) < 1e-6
    assert STATE.relative_distance(torch.zeros(2), torch.zeros(2)) == 0.0
