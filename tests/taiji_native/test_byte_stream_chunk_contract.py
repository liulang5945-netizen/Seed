from __future__ import annotations

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
