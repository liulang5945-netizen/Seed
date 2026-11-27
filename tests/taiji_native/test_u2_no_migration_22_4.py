"""Negative controls for the zero-update recovery comparison."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from scripts.training.verify_taiji_u2_no_migration_22_4 import (
    compare_measurements,
    validate_unchanged_width,
)
from taiji.u2_subject import U2Config, U2Subject


def test_dense_metadata_graph_accepts_no_migration() -> None:
    model = U2Subject(U2Config.small()).metadata_candidate()
    validate_unchanged_width(SimpleNamespace(model=model))
    model.c = replace(model.c, base_units=12, group_units=(12, 12))
    with pytest.raises(ValueError, match="grown layout"):
        validate_unchanged_width(SimpleNamespace(model=model))


def test_original_graph_cannot_substitute_metadata_graph() -> None:
    with pytest.raises(ValueError, match="metadata-enabled"):
        validate_unchanged_width(SimpleNamespace(model=U2Subject(U2Config.small())))


def test_comparison_includes_unused_and_zero_parameters() -> None:
    source = {"gradients": {"task": {"unused": None, "zero": torch.zeros(2)}}}
    assert [row["status"] for row in compare_measurements(source, source)] == ["unused", "zero"]


def test_comparison_rejects_small_parameter_credit_change() -> None:
    source = {"gradients": {"task": {"weight": torch.tensor([1.0])}}}
    changed = {"gradients": {"task": {"weight": torch.tensor([1.000001])}}}
    with pytest.raises(AssertionError):
        compare_measurements(source, changed)
