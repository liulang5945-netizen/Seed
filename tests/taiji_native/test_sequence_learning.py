from taiji import Taiji, TaijiConfig


def test_native_taiji_learns_a_raw_byte_cycle_online() -> None:
    data = b"abcdabcdabcdabcd"
    model = Taiji(
        TaijiConfig(
            region_sizes=(64, 48),
            synapse_fan_in=16,
            motor_fan_in=48,
            seed=7,
        )
    )

    before = model.score_bytes(data)
    model.learn_bytes(data, epochs=200)
    after = model.score_bytes(data)

    assert after["mean_surprise"] < before["mean_surprise"]
    assert after["accuracy"] >= 0.75
    assert model.generate(b"a", 8) == b"bcdabcda"


def test_byte_interfaces_isolate_long_term_memory_by_default(monkeypatch) -> None:
    """F1 byte prediction must not silently inherit F2 episodic feedback."""

    model = Taiji(
        TaijiConfig(
            region_sizes=(16,),
            synapse_fan_in=4,
            motor_fan_in=8,
            memory_units=16,
            memory_fan_in=4,
            memory_readout_fan_in=8,
            memory_meta_dim=8,
            seed=17,
        )
    )
    observed = model.observe
    memory_flags: list[bool] = []

    def observe_spy(symbol: int, **kwargs):
        memory_flags.append(bool(kwargs.get("use_memory", True)))
        return observed(symbol, **kwargs)

    monkeypatch.setattr(model, "observe", observe_spy)

    model.learn_bytes(b"abcd")
    model.score_bytes(b"abcd")
    model.generate(b"a", 2)

    assert memory_flags
    assert not any(memory_flags)

    memory_flags.clear()
    model.score_bytes(b"abcd", use_memory=True)
    assert memory_flags
    assert all(memory_flags)
