from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "_taiji_n1_coupled_rwl_group_adam.py"
RUNNER = REPO / "scripts" / "training" / "run_taiji_n1_coupled_rwl_length_balanced_training.py"


def _module():
    spec = importlib.util.spec_from_file_location("n1_plan116_coupled_rwl", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture():
    module = _module()
    groups = {
        "reader": [("reader.weight", torch.tensor([3.0]))],
        "writer": [("writer.weight", torch.tensor([4.0]))],
        "local": [("local.weight", torch.tensor([0.0]))],
        "context": [("context.weight", torch.tensor([10.0]))],
        "relation": [("relation.weight", torch.tensor([4.0]))],
    }
    before = {name: values[0][1] for values in groups.values() for name, _ in values}
    proposed = {
        "reader.weight": torch.tensor([0.008]),
        "writer.weight": torch.tensor([0.006]),
        "local.weight": torch.tensor([0.0]),
        "context.weight": torch.tensor([0.02]),
        "relation.weight": torch.tensor([0.002]),
    }
    return module, groups, before, proposed


def test_reader_writer_local_share_one_scale_and_one_combined_cap() -> None:
    module, groups, before, proposed = _fixture()

    projected, metrics = module.project_deltas(groups, before, proposed)

    assert metrics["reader"]["shared_block_scale"] == pytest.approx(0.5)
    assert metrics["writer"]["shared_block_scale"] == pytest.approx(0.5)
    assert metrics["local"]["shared_block_scale"] == pytest.approx(0.5)
    assert projected["reader.weight"].item() == pytest.approx(0.004)
    assert projected["writer.weight"].item() == pytest.approx(0.003)
    assert projected["local.weight"].item() == pytest.approx(0.0)
    assert (
        sum(
            float(torch.linalg.vector_norm(projected[name])) ** 2
            for name in ("reader.weight", "writer.weight", "local.weight")
        )
        ** 0.5
        <= metrics["reader"]["shared_block_cap"]
    )


def test_context_and_relation_keep_independent_relative_caps() -> None:
    module, groups, before, proposed = _fixture()

    projected, metrics = module.project_deltas(groups, before, proposed)

    assert metrics["context"]["cap_scope"] == "per_group"
    assert metrics["context"]["scale"] == pytest.approx(0.5)
    assert projected["context.weight"].item() == pytest.approx(0.01)
    assert metrics["relation"]["cap_scope"] == "per_group"
    assert metrics["relation"]["scale"] == pytest.approx(1.0)
    assert projected["relation.weight"].item() == pytest.approx(0.002)


def test_projection_refuses_invalid_limit_or_incomplete_group_partition() -> None:
    module, groups, before, proposed = _fixture()

    with pytest.raises(ValueError, match="relative limit"):
        module.project_deltas(groups, before, proposed, relative_limit=0)
    with pytest.raises(ValueError, match="five frozen parameter groups"):
        module.project_deltas({**groups, "extra": []}, before, proposed)


def test_runner_keeps_the_frozen_length_schedule_and_does_not_query_holdout() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "configure_candidate_runner" in source
    assert "run_checkpoint_preflight" in source
    assert "run_training" in source
    assert not calls.intersection({"torch.save", "generate_confirmation", "query_holdout"})
