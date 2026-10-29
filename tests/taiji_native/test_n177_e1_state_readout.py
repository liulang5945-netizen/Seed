"""Contract tests for the E1 closed-form value readout."""

from __future__ import annotations

import pytest
import torch

from taiji.joint_state_readout import FeatureValueReadoutV1


def _prototypes() -> tuple[torch.Tensor, list[int]]:
    features = torch.eye(8, dtype=torch.float32)
    return features, list(range(8))


def test_closed_form_readout_separates_all_prototypes() -> None:
    features, labels = _prototypes()
    readout = FeatureValueReadoutV1.fit_closed_form(features, labels)
    for index, feature in enumerate(features):
        result = readout.read(feature)
        assert result.selected_index == index


def test_relevant_and_irrelevant_feature_swap() -> None:
    features, labels = _prototypes()
    readout = FeatureValueReadoutV1.fit_closed_form(features, labels)
    first = readout.read(features[0])
    same = readout.read(features[0])
    assert first.logits == same.logits
    changed = readout.read(features[3])
    assert first.selected_value != changed.selected_value


def test_readout_with_query_hidden_input() -> None:
    features, labels = _prototypes()
    hidden = [torch.zeros(2, dtype=torch.float32) for _ in labels]
    readout = FeatureValueReadoutV1.fit_closed_form(features, labels, query_hiddens=hidden)
    for index, feature in enumerate(features):
        result = readout.read(feature, hidden[index])
        assert result.selected_index == index


def test_readout_parameters_receive_gradients() -> None:
    features, labels = _prototypes()
    readout = FeatureValueReadoutV1.fit_closed_form(features, labels)
    logits = readout(features[0])
    loss = (logits**2).sum()
    loss.backward()
    assert readout.linear.weight.grad is not None
    assert readout.linear.bias.grad is not None
    assert float(readout.linear.weight.grad.abs().sum()) > 0


def test_readout_rejects_unknown_label() -> None:
    features, _ = _prototypes()
    with pytest.raises(ValueError):
        FeatureValueReadoutV1.fit_closed_form(features, [0, 1, 2, 3, 4, 5, 6, 8])
