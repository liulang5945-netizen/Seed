"""Contract tests for the F byte-level value encoder."""

from __future__ import annotations

import pytest
import torch

from taiji.joint_state_consumption_course import E1_COLORS
from taiji.joint_state_value_encoder import ValueByteEncoderV1


def test_single_and_batched_shapes() -> None:
    encoder = ValueByteEncoderV1()
    raw = torch.tensor(list("红色".encode()), dtype=torch.long)
    single = encoder(raw)
    batch = encoder(torch.stack([raw, raw]))
    assert single.shape == (encoder.hidden_dim,)
    assert batch.shape == (2, encoder.hidden_dim)


def test_encode_text_rejects_unknown_value() -> None:
    encoder = ValueByteEncoderV1()
    with pytest.raises(ValueError):
        encoder.encode_text("灰色")


def test_encoder_parameters_receive_gradients() -> None:
    encoder = ValueByteEncoderV1()
    features = torch.stack([encoder.encode_text(colour) for colour in E1_COLORS])
    loss = features.pow(2).sum()
    loss.backward()
    assert encoder.embedding.weight.grad is not None
    assert encoder.gru.weight_ih_l0.grad is not None
    assert float(encoder.embedding.weight.grad.abs().sum()) > 0
