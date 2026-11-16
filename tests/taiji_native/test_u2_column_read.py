"""Contract tests for the in-body column read (``column-read-v1``)."""

from __future__ import annotations

import torch
from torch import nn

from taiji.u2_column_read import (
    CLASSES,
    COLUMNS,
    FEATURES,
    IDENTITY,
    INJECT,
    READOUT_WIDTH,
    ColumnRead,
    column_positions,
    inject_read,
    parameter_census,
)


class _Stub(nn.Module):
    """Minimal stand-in with the joint readout the hook attaches to."""

    def __init__(self) -> None:
        super().__init__()
        # The read enters the input of the joint readout: 3 * d = 24 values.
        self.joint_readout = nn.Linear(READOUT_WIDTH, 8)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.joint_readout(value)


def test_identity_is_declared() -> None:
    assert IDENTITY == "column-read-v1"


def test_read_has_four_classes_and_no_free_slot() -> None:
    read = ColumnRead(8)
    tokens = torch.randn(COLUMNS, 8)
    out = read(tokens, torch.randn(8))
    assert out["logits"].shape == (4,)
    assert len(CLASSES) == 4
    assert "未知" in CLASSES
    names = [name for name, _ in read.named_parameters()]
    assert not any("null" in name or "abstain" in name for name in names)


def test_parameter_census_matches_the_contract() -> None:
    census = parameter_census(ColumnRead(8))
    assert census["scalars"] == 476
    assert census["tensors"] == len(census["names"])
    assert "out.weight" in census["names"]


def test_injection_is_a_no_op_at_initialisation() -> None:
    """Zero-initialised output projection: the body must not move before training."""
    stub = _Stub()
    read = ColumnRead(8)
    value = torch.randn(3, READOUT_WIDTH)
    before = stub(value)
    with inject_read(stub, read, lambda: torch.zeros(INJECT)):
        after = stub(value)
    assert torch.equal(before, after)


def test_position_feature_follows_the_attended_column() -> None:
    read = ColumnRead(8)
    with torch.no_grad():
        read.key.weight.zero_()
        read.key.weight[0, 0] = 1.0  # score depends only on token dimension 0
        read.query.weight.zero_()
        read.query.weight[0, 0] = 1.0
        tokens = torch.zeros(COLUMNS, 8)
        tokens[5, 0] = 200.0  # column 5 dominates the key
        out = read(tokens, torch.ones(8))
    assert int(out["attention"].argmax()) == 5
    assert abs(float(out["position"]) - float(column_positions()[5])) < 1e-3


def test_features_carry_position_strength_and_entropy() -> None:
    read = ColumnRead(8)
    out = read(torch.randn(COLUMNS, 8), torch.randn(8))
    assert out["features"].shape == (FEATURES,)
    assert -1.0 <= float(out["position"]) <= 1.0
    assert 0.0 <= float(out["strength"]) <= 1.0
    assert float(out["entropy"]) >= 0.0


def test_decision_point_carries_the_read() -> None:
    """A forward hook must see the read inside the readout's input.

    The read has to reach the decision point, because that is what a reader
    probes; a read added to the output would leave the probe untouched.
    """
    stub = _Stub()
    read = ColumnRead(8)
    with torch.no_grad():
        read.out.weight.normal_(0.0, 0.1)
    out = read(torch.randn(COLUMNS, 8), torch.randn(8))
    captured: list[torch.Tensor] = []

    def capture(module, inputs, result):  # noqa: ANN001 - torch hook signature
        captured.append(inputs[0].detach().clone())
        return None

    handle = stub.joint_readout.register_forward_hook(capture)
    try:
        with inject_read(stub, read, lambda: out["injection"]):
            value = torch.randn(3, READOUT_WIDTH)
            stub(value)
    finally:
        handle.remove()
    assert captured
    delta = (captured[0][..., -8:] - value[..., -8:]).abs().sum()
    assert float(delta) > 0.0


def test_class_head_is_on_the_byte_path() -> None:
    """The four class probabilities are injected, so the byte loss trains the head.

    A first version injected only the eleven features, which left the head
    outside the gradient path entirely; this test is what caught it.
    """
    stub = _Stub()
    read = ColumnRead(8)
    out = read(torch.randn(COLUMNS, 8), torch.randn(8))
    with inject_read(stub, read, lambda: out["injection"]):
        stub(torch.randn(3, READOUT_WIDTH)).sum().backward()
    assert read.head[0].weight.grad is not None
    # Zero-initialised residual: on the first step only the projection can move.
    assert float(read.head[0].weight.grad.abs().sum()) == 0.0
    assert float(read.out.weight.grad.abs().sum()) > 0.0


def test_head_trains_once_the_residual_is_live() -> None:
    """After the first update the class head receives the byte loss gradient."""
    stub = _Stub()
    read = ColumnRead(8)
    with torch.no_grad():
        read.out.weight.normal_(0.0, 0.1)
    out = read(torch.randn(COLUMNS, 8), torch.randn(8))
    with inject_read(stub, read, lambda: out["injection"]):
        stub(torch.randn(3, READOUT_WIDTH)).sum().backward()
    assert float(read.head[0].weight.grad.abs().sum()) > 0.0
