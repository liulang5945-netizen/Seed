from __future__ import annotations

from pathlib import Path

import pytest

from scripts.training.eval_taiji_m2r1_phase_c_canary import _train_stream_in_chunks
from taiji import Taiji, TaijiConfig
from taiji.internalization import content_digest


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(8,),
        synapse_fan_in=2,
        motor_fan_in=4,
        predictive_context_fan_in=2,
        memory_units=16,
        memory_fan_in=2,
        memory_readout_fan_in=2,
        memory_meta_dim=4,
        memory_time_dim=2,
        memory_episode_dim=2,
        lateral_fan_in=2,
        identity_organ_capacity=8,
        concept_capacity=8,
        seed=101,
    )


def test_split_byte_stream_preserves_single_stream_learning_semantics() -> None:
    data = b"abcd" * 6
    direct = Taiji(_config())
    split = Taiji(_config())

    direct_metrics = direct.learn_bytes(
        data,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=False,
    )

    cut = 11
    first_metrics = split.learn_bytes(
        data[:cut],
        include_boundary=False,
        include_start_boundary=True,
        include_end_boundary=False,
        reset=True,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=False,
    )
    second_metrics = split.learn_bytes(
        data[cut:],
        include_boundary=False,
        include_start_boundary=False,
        include_end_boundary=True,
        reset=False,
        use_memory=False,
        learn_fabric=False,
        learn_predictive_context=False,
    )

    assert content_digest(split.checkpoint()) == content_digest(direct.checkpoint())
    assert direct_metrics["observations"] == (
        first_metrics["observations"] + second_metrics["observations"]
    )
    assert direct_metrics["online_accuracy"] == (
        first_metrics["online_accuracy"] * first_metrics["observations"]
        + second_metrics["online_accuracy"] * second_metrics["observations"]
    ) / direct_metrics["observations"]


def test_stream_progress_checkpoint_can_restore_completed_chunk_run() -> None:
    data = b"abcd" * 4
    progress = Path(".seed_test_tmp") / "m2r1-stream-progress.pt"
    progress.parent.mkdir(parents=True, exist_ok=True)
    try:
        direct = Taiji(_config())
        direct.learn_bytes(data)

        chunked, result = _train_stream_in_chunks(
            Taiji(_config()),
            data,
            epochs=1,
            chunk_bytes=5,
            checkpoint_interval=1,
            progress_path=progress,
            resume=False,
            run_key="test:stream:completed",
        )

        assert content_digest(chunked.checkpoint()) == content_digest(direct.checkpoint())
        assert result["resource"]["chunk_count"] == 4
        assert result["resource"]["resume_cursor"] == 0
        assert result["resource"]["resume_epoch"] == 1

        restored, resumed = _train_stream_in_chunks(
            Taiji(_config()),
            data,
            epochs=1,
            chunk_bytes=5,
            checkpoint_interval=1,
            progress_path=progress,
            resume=True,
            run_key="test:stream:completed",
        )
        assert content_digest(restored.checkpoint()) == content_digest(chunked.checkpoint())
        assert resumed["resource"]["resumed_from_checkpoint"] is True
    finally:
        progress.unlink(missing_ok=True)


def test_stream_progress_resumes_after_a_chunk_boundary_interruption() -> None:
    data = b"abcd" * 4
    progress = Path(".seed_test_tmp") / "m2r1-stream-progress-interrupted.pt"
    progress.parent.mkdir(parents=True, exist_ok=True)
    try:
        direct = Taiji(_config())
        direct.learn_bytes(data)

        interrupted = Taiji(_config())
        original = interrupted.learn_bytes
        calls = 0

        def fail_on_second_chunk(chunk: bytes, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption after one committed chunk")
            return original(chunk, **kwargs)

        interrupted.learn_bytes = fail_on_second_chunk  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="simulated interruption"):
            _train_stream_in_chunks(
                interrupted,
                data,
                epochs=1,
                chunk_bytes=5,
                checkpoint_interval=1,
                progress_path=progress,
                resume=False,
                run_key="test:stream:interrupted",
            )

        resumed, result = _train_stream_in_chunks(
            Taiji(_config()),
            data,
            epochs=1,
            chunk_bytes=5,
            checkpoint_interval=1,
            progress_path=progress,
            resume=True,
            run_key="test:stream:interrupted",
        )
        assert content_digest(resumed.checkpoint()) == content_digest(direct.checkpoint())
        assert result["resource"]["resumed_from_checkpoint"] is True
        assert result["resource"]["chunk_count"] == 4
    finally:
        progress.unlink(missing_ok=True)
