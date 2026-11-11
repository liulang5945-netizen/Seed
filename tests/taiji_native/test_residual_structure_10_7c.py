"""Contract guard for the 10.7c residual-structure diagnostic: its three pure helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "diagnose_residual_structure_10_7c.py"


def _load(name: str, path: Path):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RESIDUAL = _load("diagnose_residual_structure_10_7c", SCRIPT)


def test_explained_fraction_full_and_zero():
    vector = torch.tensor([1.0, 0.0, 0.0])
    residuals = [vector, vector, vector, -vector, -vector, -vector]
    labels = ["a", "a", "a", "b", "b", "b"]
    assert RESIDUAL.explained_fraction(residuals, labels) == 1.0
    assert RESIDUAL.explained_fraction([torch.zeros(3)] * 6, labels) == 0.0


def test_explained_fraction_drops_when_groups_mix():
    vector = torch.tensor([1.0, 0.0])
    mixed = [vector, -vector, vector, -vector, vector, -vector]
    labels = ["a", "a", "a", "b", "b", "b"]
    fraction = RESIDUAL.explained_fraction(mixed, labels)
    assert 0.0 <= fraction < 0.5


def test_within_cosine():
    vector = torch.tensor([1.0, 2.0])
    other = torch.tensor([2.0, -1.0])
    coherence = RESIDUAL.within_cosine([vector, vector, other, other], ["a", "a", "b", "b"])
    assert abs(coherence["a"] - 1.0) < 1e-9
    assert abs(coherence["b"] - 1.0) < 1e-9


def test_verdict_table():
    classify = RESIDUAL.classify
    assert classify(0.8, {"a": 0.9, "b": 0.8}) == "residual_group_structure"
    assert classify(0.8, {"a": 0.9, "b": 0.1}) == "residual_mixed"
    assert classify(0.2, {"a": 0.99, "b": 0.99}) == "residual_noise_like"
    assert classify(None, {}) == "no_structure"
    assert classify(0.6, {"a": None}) == "no_structure"
