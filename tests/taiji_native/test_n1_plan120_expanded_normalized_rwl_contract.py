from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "_taiji_n1_expanded_normalized_coupled_rwl_adam.py"


def _module():
    spec = importlib.util.spec_from_file_location("n1_plan120_expanded_normalized_rwl", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture():
    groups = {
        "reader": [("reader.weight", torch.tensor([3.0]))],
        "writer": [("writer.weight", torch.tensor([4.0]))],
        "local": [("local.weight", torch.tensor([0.0]))],
        "context": [("context.weight", torch.tensor([10.0]))],
        "relation": [("relation.weight", torch.tensor([4.0]))],
    }
    before = {name: parameter for values in groups.values() for name, parameter in values}
    proposed = {
        "reader.weight": torch.tensor([0.008]),
        "writer.weight": torch.tensor([0.006]),
        "local.weight": torch.tensor([0.0]),
        "context.weight": torch.tensor([0.02]),
        "relation.weight": torch.tensor([0.002]),
    }
    return groups, before, proposed


def test_shared_rwl_radius_is_two_sqrt_three() -> None:
    module = _module()
    groups, before, proposed = _fixture()

    projected, metrics = module.project_deltas(groups, before, proposed)

    radius = 2 * math.sqrt(3.0)
    scales = [metrics[group]["shared_normalized_scale"] for group in module.RWL]
    assert max(scales) == pytest.approx(min(scales))
    assert metrics["reader"]["shared_normalized_radius"] == pytest.approx(radius)
    normalized = [metrics[group]["group_normalized_proposal"] * scales[0] for group in module.RWL]
    assert sum(value**2 for value in normalized) ** 0.5 <= radius
    assert projected["reader.weight"].item() == pytest.approx(0.008 * scales[0])
    assert projected["writer.weight"].item() == pytest.approx(0.006 * scales[0])


def test_context_relation_keep_independent_relative_caps() -> None:
    module = _module()
    groups, before, proposed = _fixture()

    projected, metrics = module.project_deltas(groups, before, proposed)

    assert metrics["context"]["cap_scope"] == "per_group"
    assert metrics["relation"]["cap_scope"] == "per_group"
    assert metrics["context"]["scale"] == pytest.approx(0.5)
    assert metrics["relation"]["scale"] == pytest.approx(1.0)
    assert projected["context.weight"].item() == pytest.approx(0.01)
