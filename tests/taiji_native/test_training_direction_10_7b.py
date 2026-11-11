"""Contract guard for the 10.7b direction diagnostic: cosine and the verdict table."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "diagnose_training_direction_10_7b.py"


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


DIAGNOSTIC = _load("diagnose_training_direction_10_7b", SCRIPT)


def test_cosine_edges():
    vector = torch.tensor([1.0, 2.0, 3.0])
    assert abs(DIAGNOSTIC.cosine(vector, vector) - 1.0) < 1e-9
    assert abs(DIAGNOSTIC.cosine(vector, -vector) + 1.0) < 1e-9
    assert abs(DIAGNOSTIC.cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))) < 1e-12
    assert DIAGNOSTIC.cosine(torch.zeros(3), vector) is None


def test_verdict_table():
    classify = DIAGNOSTIC.classify
    assert classify(0.99) == "groups_pull_together"
    assert classify(0.90) == "groups_pull_together"
    assert classify(-0.99) == "groups_oppose"
    assert classify(-0.90) == "groups_oppose"
    assert classify(0.30) == "groups_mixed"
    assert classify(-0.30) == "groups_mixed"
    assert classify(None) == "no_alignment"
