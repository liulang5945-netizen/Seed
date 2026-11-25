"""Tests for the N34.2 growth-benefit task v2 (zero training)."""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")
import torch

from taiji.continual_growth_task_v2 import IN_DIM, NUM_CLASSES, TASK_VERSION, make_split


def test_shapes_and_label_ranges() -> None:
    data = make_split(7, n_train=128, n_val=64)
    assert data["phase1"]["train"]["x"].shape == (128, IN_DIM)
    assert data["phase1"]["val"]["y"].shape == (64,)
    assert data["phase2"]["train"]["x"].shape == (128, IN_DIM)
    assert int(data["phase1"]["train"]["y"].min()) >= 0
    assert int(data["phase1"]["train"]["y"].max()) < 6
    assert int(data["phase2"]["train"]["y"].min()) >= 6
    assert int(data["phase2"]["train"]["y"].max()) < NUM_CLASSES


def test_deterministic() -> None:
    a = make_split(11, n_train=64, n_val=32)
    b = make_split(11, n_train=64, n_val=32)
    for phase in ("phase1", "phase2"):
        for split in ("train", "val"):
            assert torch.equal(a[phase][split]["x"], b[phase][split]["x"])
            assert torch.equal(a[phase][split]["y"], b[phase][split]["y"])


def test_no_task_id_routing() -> None:
    assert TASK_VERSION == "taiji-continual-growth-v2"
    data = make_split(3, n_train=16, n_val=8)
    for phase in data.values():
        for split in phase.values():
            assert set(split) == {"x", "y"}
