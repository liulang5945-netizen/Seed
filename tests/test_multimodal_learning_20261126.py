"""Tests for learned multimodal consequence prediction (zero heavy training)."""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")
import torch

from taiji.multimodal_learning import (
    IN_DIM,
    N_OUTCOMES,
    ConsequenceNet,
    accuracy,
    collect,
    train_net,
)


def test_dataset_shapes_and_labels() -> None:
    x, y = collect(1, "incline", 3)
    assert x.dim() == 2 and x.shape[1] == IN_DIM
    assert x.shape[0] == y.shape[0] > 0
    assert int(y.min()) >= 0 and int(y.max()) < N_OUTCOMES


def test_dataset_deterministic() -> None:
    a = collect(5, "collision", 2)
    b = collect(5, "collision", 2)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_net_learns_in_family() -> None:
    x, y = collect(3, "incline", 12)
    net = train_net(x, y, seed=3, steps=300)
    assert accuracy(net, x, y) >= 0.90


def test_model_shape_and_determinism() -> None:
    x, _ = collect(2, "incline", 1)
    a = ConsequenceNet(7)
    b = ConsequenceNet(7)
    out_a = a(x)
    out_b = b(x)
    assert out_a.shape == (x.shape[0], N_OUTCOMES)
    assert torch.allclose(out_a, out_b)
