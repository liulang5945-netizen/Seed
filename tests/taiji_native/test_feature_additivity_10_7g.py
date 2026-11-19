"""Contract guard for 10.7g: the two-way decomposition and its verdict table."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "probe_feature_additivity_10_7g.py"

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


ADDITIVITY = _load("probe_feature_additivity_10_7g", SCRIPT)


def _table(values):
    out = {}
    for obj in ADDITIVITY.OBJECTS:
        for place in ADDITIVITY.PLACES:
            vector = torch.zeros(4, dtype=torch.float64)
            vector[0] = values(obj, place)
            out[(obj, place)] = vector
    return out


def test_pure_main_effects_are_additive():
    objects = ADDITIVITY.OBJECTS
    places = ADDITIVITY.PLACES

    def value(obj, place):
        return 1.0 * objects.index(obj) + 3.0 * places.index(place)

    stats = ADDITIVITY.decompose(_table(value))
    assert stats["additive_share"] == 1.0
    assert ADDITIVITY.classify(stats["additive_share"]) == "features_additive"


def test_pure_interaction_is_entangled():
    objects = ADDITIVITY.OBJECTS
    places = ADDITIVITY.PLACES
    diagonal = {(0, 0), (1, 1)}
    off = {(0, 1), (1, 0)}

    def value(obj, place):
        cell = (objects.index(obj), places.index(place))
        if cell in diagonal:
            return 1.0
        if cell in off:
            return -1.0
        return 0.0

    stats = ADDITIVITY.decompose(_table(value))
    assert stats["additive_share"] == 0.0
    assert ADDITIVITY.classify(stats["additive_share"]) == "features_entangled"


def test_verdict_table():
    classify = ADDITIVITY.classify
    assert classify(0.9) == "features_additive"
    assert classify(0.8) == "features_additive"
    assert classify(0.6) == "features_mixed"
    assert classify(0.45) == "features_entangled"
    assert classify(None) == "features_undefined"
