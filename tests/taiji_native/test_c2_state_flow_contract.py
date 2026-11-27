from __future__ import annotations

import pytest
import torch

from scripts.training.run_taiji_n1_c2_state_flow import (
    assert_noninterference,
    observe_state_flow,
)
from taiji import InputFrame, TaijiConfig, TSKV8Adapter
from taiji.stepwise_consumer import StepwiseConsumerRequest


def prepared():
    model = TSKV8Adapter(
        TaijiConfig(region_sizes=(24, 16), synapse_fan_in=6, motor_fan_in=8, seed=41)
    )
    model.persistent_stepwise_carrier = True
    for index, length in enumerate((3, 5)):
        model.generate_input(
            InputFrame(
                input_id=f"support:{index}",
                modality="text",
                payload=b"fact",
                source="test",
                timestamp=index,
                provenance="external",
                confidence=1.0,
            ),
            length,
            reset=True,
            stop_at_boundary=False,
        )
    model.persistent_stepwise_carrier = False
    ids = model._persistent_stepwise_carrier.episode_ids
    model.configure_stepwise_consumer(StepwiseConsumerRequest("oracle-correct", ids, ids[0]))
    return model


def test_observers_preserve_forward_and_cover_replay_and_actual_emissions() -> None:
    plain, traced = prepared(), prepared()
    expected = plain.generate(b"question", 7, stop_at_boundary=False)
    weights = [tensor.clone() for tensor in traced.parameter_tensors()]
    with observe_state_flow(traced) as trace:
        actual = traced.generate(b"question", 7, stop_at_boundary=False)
    assert expected == actual
    assert plain.last_stepwise_consumer_audit == traced.last_stepwise_consumer_audit
    assert len(trace["replay"]) == 5
    assert set(map(int, trace["snapshots"])) == {1, 2, 3, 4, 5}
    assert trace["after_readout"]["tick"] - trace["query"]["tick"] == 5
    assert len(trace["generation"]) == 7
    assert bytes(row["input_symbol"] for row in trace["generation"]) == actual
    assert trace["encoded_context"] == trace["after_readout"]["context"]
    assert all(torch.equal(a, b) for a, b in zip(weights, traced.parameter_tensors(), strict=True))
    for owner, name in (
        (traced, "observe"),
        (traced.fabric, "step"),
        (traced.predictive_context, "encode"),
        (traced, "_consume_stepwise_before_answer"),
    ):
        assert name not in owner.__dict__


def test_observer_restores_existing_instance_binding_on_exception() -> None:
    model = prepared()
    original = model.observe
    model.observe = original
    with pytest.raises(RuntimeError, match="abort"), observe_state_flow(model):
        raise RuntimeError("abort")
    assert model.__dict__["observe"] is original
    assert "step" not in model.fabric.__dict__


def test_noninterference_rejects_raw_and_first_distribution_changes() -> None:
    baseline = dict(
        raw_hex="61",
        output="a",
        consumer={"first_probabilities_after": [0.5, 0.5]},
        emitted_steps=1,
        natural_stop=True,
    )
    assert_noninterference(baseline, dict(baseline))
    with pytest.raises(RuntimeError, match="raw_hex"):
        assert_noninterference(baseline, dict(baseline, raw_hex="62"))
    with pytest.raises(RuntimeError, match="consumer"):
        assert_noninterference(
            baseline, dict(baseline, consumer={"first_probabilities_after": [0.6, 0.4]})
        )
