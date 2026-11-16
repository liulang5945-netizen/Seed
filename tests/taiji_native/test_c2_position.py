"""Zero-training contract tests for the C.2 position-focus objective."""

import torch

from scripts.training import run_c2_position_20261116 as c2


def test_position_focus_loss_rewards_concentration_on_label():
    raw = "红色圆片现在在左边。".encode()
    label = "左边"
    lo, hi = c2.label_span(raw, label)
    n = len(raw)
    valid = torch.ones(n, dtype=torch.bool)
    uniform = torch.ones(n)
    concentrated = torch.full((n,), 1e-6)
    concentrated[lo:hi] = 1.0
    loss_uniform = float(c2.position_focus_loss(uniform, valid, raw, label))
    loss_concentrated = float(c2.position_focus_loss(concentrated, valid, raw, label))
    assert loss_concentrated < loss_uniform
    assert loss_concentrated < 0.1


def test_position_focus_loss_gradient_reaches_position_weights():
    raw = "红色圆片现在在左边。".encode()
    weights = torch.ones(len(raw), requires_grad=True)
    valid = torch.ones(len(raw), dtype=torch.bool)
    loss = c2.position_focus_loss(weights, valid, raw, "左边")
    loss.backward()
    assert weights.grad is not None and bool(weights.grad.abs().sum() > 0)
