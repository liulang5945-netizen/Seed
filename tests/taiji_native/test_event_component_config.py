import pytest

from taiji import TaijiConfig


def test_event_component_gains_roundtrip_and_default() -> None:
    config = TaijiConfig()
    assert config.memory_event_component_gains == (1.0,) * 6
    restored = TaijiConfig.from_dict(config.to_dict())
    assert restored == config

    candidate = TaijiConfig(
        memory_event_component_gains=(1.0, 1.0, 1.0, 1.0, 0.05, 1.0)
    )
    assert TaijiConfig.from_dict(candidate.to_dict()) == candidate


def test_event_component_gains_require_six_non_negative_values() -> None:
    with pytest.raises(ValueError, match="one gain per"):
        TaijiConfig(memory_event_component_gains=(1.0, 1.0))
    with pytest.raises(ValueError, match="non-negative"):
        TaijiConfig(memory_event_component_gains=(1.0, 1.0, 1.0, 1.0, -0.1, 1.0))
    with pytest.raises(ValueError, match="at least one"):
        TaijiConfig(memory_event_component_gains=(0.0,) * 6)
