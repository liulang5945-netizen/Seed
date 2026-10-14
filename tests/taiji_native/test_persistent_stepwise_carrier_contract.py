from __future__ import annotations

import pytest
import torch

from taiji import InputFrame, TaijiConfig, TSKV8Adapter
from taiji.internalization import content_digest
from taiji.memory import PersistentStepwiseCarrier
from taiji.model import Taiji


def _config(seed: int = 41) -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(24, 16),
        synapse_fan_in=6,
        motor_fan_in=8,
        seed=seed,
    )


def _frame(input_id: str = "client-1") -> InputFrame:
    return InputFrame(
        input_id=input_id,
        modality="text",
        payload=b"prompt",
        source="test.client",
        timestamp=7,
        provenance="external",
        confidence=0.8,
    )


def test_default_off_keeps_generation_and_checkpoint_payload_unchanged() -> None:
    first = TSKV8Adapter(_config())
    second = TSKV8Adapter(_config())
    frame = _frame()

    first_output = first.generate_input(frame, 4, reset=True)
    second_output = second.generate_input(frame, 4, reset=True)

    assert first.persistent_stepwise_carrier is False
    assert first_output == second_output
    assert content_digest(first.checkpoint()) == content_digest(second.checkpoint())
    assert Taiji.PERSISTENT_STEPWISE_CARRIER_KEY not in first.checkpoint()


def test_open_carrier_records_each_pre_observe_context_and_source(monkeypatch) -> None:
    model = TSKV8Adapter(_config())
    model.persistent_stepwise_carrier = True
    frame = _frame()
    parameters_before = tuple(value.detach().clone() for value in model.parameter_tensors())
    observed_cues: list[torch.Tensor] = []
    cortical_cue = model.cortical_cue

    def capture_cue() -> torch.Tensor:
        cue = cortical_cue()
        observed_cues.append(cue)
        return cue

    monkeypatch.setattr(model, "cortical_cue", capture_cue)
    generated = model.generate_input(frame, 4, reset=True, stop_at_boundary=False)
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameter_tensors(), strict=True)
    )

    carrier = model._persistent_stepwise_carrier
    assert carrier.episode_count == 1
    assert len(generated) == 4
    episode_id = carrier.episode_ids[0]
    trace = model.read_persistent_stepwise_episode(episode_id)
    assert trace.dtype == torch.float32
    assert tuple(trace.shape) == (len(generated), model.config.cortical_context_dim)
    assert torch.equal(trace, torch.stack(observed_cues))
    assert carrier.episode_manifest()[0]["source"] == {
        "input_id": frame.input_id,
        "modality": frame.modality,
        "source": frame.source,
        "timestamp": frame.timestamp,
        "provenance": frame.provenance,
        "confidence": frame.confidence,
    }

    trace[0].zero_()
    assert torch.equal(
        model.read_persistent_stepwise_episode(episode_id), torch.stack(observed_cues)
    )
    assert tuple(model.read_persistent_stepwise_episode("unknown").shape) == (
        0,
        model.config.cortical_context_dim,
    )


def test_input_ids_may_repeat_because_episode_serial_is_unique() -> None:
    model = TSKV8Adapter(_config())
    model.persistent_stepwise_carrier = True
    model.generate_input(_frame("reused"), 1, reset=True)
    model.generate_input(_frame("reused"), 1, reset=True)

    episodes = model._persistent_stepwise_carrier.episode_manifest()
    assert [episode["serial"] for episode in episodes] == [0, 1]
    assert episodes[0]["episode_id"] != episodes[1]["episode_id"]
    assert [episode["source"]["input_id"] for episode in episodes] == ["reused", "reused"]


def test_checkpoint_roundtrip_restores_trace_but_leaves_recording_disabled() -> None:
    original = TSKV8Adapter(_config())
    original.persistent_stepwise_carrier = True
    original.generate_input(_frame(), 3, reset=True)
    checkpoint = original.checkpoint()
    episode_id = original._persistent_stepwise_carrier.episode_ids[0]

    restored = TSKV8Adapter(_config())
    restored.restore(checkpoint)

    assert restored.persistent_stepwise_carrier is False
    assert torch.equal(
        restored.read_persistent_stepwise_episode(episode_id),
        original.read_persistent_stepwise_episode(episode_id),
    )


def test_missing_carrier_key_restores_empty_and_default_off() -> None:
    checkpoint = TSKV8Adapter(_config()).checkpoint()
    assert Taiji.PERSISTENT_STEPWISE_CARRIER_KEY not in checkpoint

    restored = TSKV8Adapter(_config())
    restored.restore(checkpoint)

    assert restored.persistent_stepwise_carrier is False
    assert restored._persistent_stepwise_carrier.episode_count == 0
    assert Taiji.PERSISTENT_STEPWISE_CARRIER_KEY not in restored.checkpoint()


def test_open_direct_generation_requires_an_input_frame_source() -> None:
    model = TSKV8Adapter(_config())
    model.persistent_stepwise_carrier = True

    with pytest.raises(ValueError, match="requires source metadata"):
        model.generate(b"prompt", 1)


def test_carrier_fails_closed_on_episode_and_step_capacity() -> None:
    source = _frame().to_payload()
    carrier = PersistentStepwiseCarrier(
        context_dim=3,
        max_episodes=1,
        max_steps_per_episode=2,
    )
    episode_id = carrier.begin_episode(source)

    with pytest.raises(RuntimeError, match="step limit"):
        carrier.commit_episode(episode_id, torch.ones((3, 3)))
    carrier.abort_episode(episode_id)
    assert carrier.to_payload() is None

    episode_id = carrier.begin_episode(source)
    carrier.commit_episode(episode_id, torch.ones((2, 3)))
    with pytest.raises(RuntimeError, match="episode capacity"):
        carrier.begin_episode(source)


def test_carrier_payload_rejects_wrong_dtype_and_context_dimension() -> None:
    carrier = PersistentStepwiseCarrier(context_dim=3)
    episode_id = carrier.begin_episode(_frame().to_payload())
    carrier.commit_episode(episode_id, torch.ones((1, 3), dtype=torch.float32))
    payload = carrier.to_payload()
    assert payload is not None

    bad_dtype = dict(payload)
    bad_dtype["episodes"] = [
        dict(payload["episodes"][0], vectors=torch.ones((1, 3), dtype=torch.float64))
    ]
    with pytest.raises(ValueError, match="must be float32"):
        PersistentStepwiseCarrier(context_dim=3).load_payload(bad_dtype)

    with pytest.raises(ValueError, match="dimension does not match"):
        PersistentStepwiseCarrier(context_dim=4).load_payload(payload)
