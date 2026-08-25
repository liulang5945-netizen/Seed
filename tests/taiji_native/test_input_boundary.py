from __future__ import annotations

import pytest

from taiji import InputFrame, InputTrace, TaijiConfig, TSKV8Adapter


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(24, 16),
        synapse_fan_in=6,
        motor_fan_in=8,
        seed=41,
    )


def test_input_frame_is_a_raw_byte_transport_contract() -> None:
    frame = InputFrame(
        input_id="client-1",
        modality="text",
        payload="你好".encode(),
        source="test.client",
        timestamp=7,
        provenance="external",
        confidence=0.8,
    )

    restored = InputFrame.from_payload(frame.to_payload())

    assert restored == frame
    assert restored.payload == "你好".encode()


def test_input_frame_validates_text_without_creating_token_ids() -> None:
    with pytest.raises(ValueError, match="valid UTF-8"):
        InputFrame(
            input_id="invalid",
            modality="text",
            payload=b"\xff",
            source="test.client",
        )

    raw = InputFrame(
        input_id="raw-1",
        modality="text-byte",
        payload=b"\xff",
        source="test.sensor",
    )
    assert raw.payload == b"\xff"


def test_ingest_input_emits_observation_and_percept_without_intent_mapping() -> None:
    adapter = TSKV8Adapter(_config())
    frame = InputFrame(
        input_id="client-2",
        modality="text",
        payload=b"ab",
        source="test.client",
        timestamp=12,
        provenance="external",
        confidence=0.9,
    )

    trace = adapter.ingest_input(frame, learn=False)
    restored = InputTrace.from_payload(trace.to_payload())

    assert trace.input_id == "client-2"
    assert trace.modality == "text"
    assert tuple(item.value for item in trace.observations) == (97, 98)
    assert all(item.modality == "text-byte" for item in trace.observations)
    assert all(item.source == "test.client" for item in trace.observations)
    assert len(trace.percepts) == 2
    assert all(item.modality == "text-byte" for item in trace.percepts)
    assert trace.action_intent is None
    assert restored.input_id == trace.input_id
    assert restored.observations == trace.observations
    assert len(restored.percepts) == len(trace.percepts)


def test_generate_input_uses_the_same_byte_effector_path() -> None:
    adapter = TSKV8Adapter(_config())
    frame = InputFrame(
        input_id="generation-1",
        modality="text",
        payload=b"a",
        source="test.client",
    )

    generated = adapter.generate_input(frame, 4, reset=True)

    assert isinstance(generated, bytes)
    assert len(generated) == 4

    with pytest.raises(ValueError, match="unsupported input modality"):
        adapter.generate_input(
            InputFrame(
                input_id="image-1",
                modality="image",
                payload=b"raw",
                source="test.client",
            ),
            1,
        )
