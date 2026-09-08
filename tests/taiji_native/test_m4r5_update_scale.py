from __future__ import annotations

import math

import pytest

from taiji import Taiji, TaijiConfig
from taiji.internalization import content_digest


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(32,),
        synapse_fan_in=8,
        motor_fan_in=16,
        memory_units=32,
        memory_fan_in=8,
        memory_readout_fan_in=16,
        memory_meta_dim=16,
        seed=59,
    )


def _owners(model: Taiji) -> tuple[str, str, str, str]:
    return (
        content_digest(model.fabric.to_payload()),
        content_digest(model.predictive_context.to_payload()),
        content_digest(model.predictive_readout.to_payload()),
        content_digest(model.memory.to_payload()),
    )


def test_update_scale_one_matches_default_and_zero_freezes_predictive_owners() -> None:
    data = b"abcd" * 32
    baseline = Taiji(_config())
    explicit_one = Taiji(_config())
    baseline.learn_bytes(data, epochs=2, learn_fabric=False)
    explicit_one.learn_bytes(
        data,
        epochs=2,
        learn_fabric=False,
        predictive_update_scale=1.0,
    )
    assert _owners(baseline) == _owners(explicit_one)

    frozen = Taiji(_config())
    before = _owners(frozen)
    frozen.learn_bytes(
        data,
        epochs=2,
        learn_fabric=False,
        predictive_update_scale=0.0,
    )
    after = _owners(frozen)
    assert after == before


def test_positive_subunit_update_scale_changes_both_predictive_owners() -> None:
    model = Taiji(_config())
    before = _owners(model)
    model.learn_bytes(
        b"abcd" * 32,
        epochs=2,
        learn_fabric=False,
        predictive_update_scale=0.5,
    )
    after = _owners(model)
    assert after[0] == before[0]
    assert after[1] != before[1]
    assert after[2] != before[2]
    assert after[3] == before[3]


@pytest.mark.parametrize("value", (-0.1, math.nan, math.inf))
def test_update_scale_rejects_non_finite_or_negative_values(value: float) -> None:
    model = Taiji(_config())
    with pytest.raises(ValueError, match="predictive_update_scale"):
        model.learn_bytes(b"abcd", predictive_update_scale=value)
