from __future__ import annotations

import importlib

import pytest
import torch

from scripts.training.run_taiji_n1_c2_padding_placement import front_padding, prefix_padding
from taiji.stepwise_consumer import StepwiseConsumerRequest, select_and_prepare, vector_digest


def test_front_padding_preserves_order_content_and_dose_with_honest_digest() -> None:
    pool = {"short": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "long": torch.ones(5, 2)}
    request = StepwiseConsumerRequest("oracle-correct", ("short", "long"), "short")
    query = torch.tensor([1.0, 0.0])
    original, old = select_and_prepare(request, query, pool)
    actual, audit = prefix_padding(request, query, pool)
    assert audit["replay_steps"] == old["replay_steps"] == 5
    assert torch.equal(actual[:3], torch.zeros(3, 2))
    assert torch.equal(actual[3:], pool["short"])
    assert audit["replay_sha256"] == vector_digest(actual)
    assert audit["tail_replay_sha256"] == vector_digest(original)
    assert audit["replay_sha256"] != audit["tail_replay_sha256"]
    for mode, control in (("wrong-episode", "long"), ("zero-replay", None)):
        request = StepwiseConsumerRequest(mode, ("short", "long"), control)
        expected, _ = select_and_prepare(request, query, pool)
        actual, _ = prefix_padding(request, query, pool)
        assert torch.equal(actual, expected)


def test_diagnostic_selector_binding_is_restored_even_on_abort() -> None:
    module = importlib.import_module("taiji.model")
    original = module.select_and_prepare
    with pytest.raises(RuntimeError, match="abort"), front_padding():
        assert module.select_and_prepare is prefix_padding
        raise RuntimeError("abort")
    assert module.select_and_prepare is original
