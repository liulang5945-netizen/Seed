"""Controlled physical witnesses and transactional sensor regressions."""

import numpy as np
import pytest
import torch

from taiji.u2_history import assert_values_equal, value_digest
from taiji.u2_task_world import PhysicalWorld


def command(world, operation=0, duration=0.04):
    action = torch.zeros(15, dtype=torch.float64)
    action[operation] = 1
    if operation != 3:
        action[9:11] = torch.tensor(world.snapshot()["bodies"][0]["position"])
        action[13], action[14] = 1, duration
        if operation == 0:
            action[11] = 0.8
    return action


def fixture(mass=1.0, drag=1.0, dtype="float32"):
    world = PhysicalWorld.reset(120, {"body_count": 1, "dtype": dtype})
    state = world.snapshot()
    state["bodies"][0].update(position=[-0.5, 0.0], velocity=[0.0, 0.0], mass=mass, drag=drag)
    return PhysicalWorld.restore(state)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reproducible_raw_and_codec_roundtrip(dtype):
    first = PhysicalWorld.reset(20, {"dtype": dtype})
    second = PhysicalWorld.reset(20, {"dtype": dtype})
    assert_values_equal(first.initial_observation("initial"), second.initial_observation("initial"))
    for index in range(3):
        action = command(first)
        a, b = first.step(str(index), action), second.step(str(index), action)
        assert_values_equal(a, b)
        assert a["observations"]["rgb"].shape == (16, 48)
        assert a["observations"]["audio"].shape == (7, 33)
        assert a["observations"]["proprio"].shape == (1, 16)
        assert a["execution_outcome"].dtype == getattr(torch, dtype)
    restored = PhysicalWorld.restore(first.snapshot())
    assert_values_equal(first.elapse("after-restore"), restored.elapse("after-restore"))


def test_duplicate_returns_original_without_rng_or_effect():
    world = fixture()
    action = command(world)
    result = world.step("same", action)
    saved = value_digest(world.snapshot())
    result["raw_media"]["rgb"].zero_()
    again = world.step("same", action)
    assert bool(again["raw_media"]["rgb"].any())
    assert value_digest(world.snapshot()) == saved
    action[11] = -0.5
    with pytest.raises(ValueError, match="conflict"):
        world.step("same", action)
    assert value_digest(world.snapshot()) == saved


def test_failed_physics_is_atomic(monkeypatch):
    from taiji import u2_task_world

    world = fixture()
    before = value_digest(world.snapshot())

    def broken(*_args):
        raise ValueError("injected_physics_failure")

    monkeypatch.setattr(u2_task_world, "advance", broken)
    with pytest.raises(ValueError, match="injected"):
        world.step("failed", command(world))
    assert value_digest(world.snapshot()) == before
    assert world.query("failed")["status"] == "not_seen"


def test_inspect_changes_view_and_has_no_drive():
    world = fixture()
    result = world.step("look", command(world, 2))
    assert world.snapshot()["bodies"][0]["position"] == [-0.5, 0.0]
    assert float(result["coordinates"]["rgb"][:, 0].max()) < 0.5
    silent = world.step("zero-exposure", command(world, 2, 0))
    assert "rgb" not in silent["observations"]
    assert "audio" in silent["observations"]


def test_ask_does_not_advance_time_or_rng():
    world = fixture()
    before = world.snapshot()
    result = world.step("ask", command(world, 3))
    after = world.snapshot()
    assert result["t_start"] == result["t_end"] == 0
    assert result["observations"] == {}
    assert_values_equal(before["rng_states"], after["rng_states"])


def test_missing_sensor_schedule_does_not_change_rng_consumption():
    first = PhysicalWorld.reset(9, {"modalities": []})
    second = PhysicalWorld.reset(9)
    assert first.elapse("tick")["observations"] == {}
    second.elapse("tick")
    assert_values_equal(first.snapshot()["rng_states"], second.snapshot()["rng_states"])


def test_same_action_different_mass_changes_real_motion():
    light, heavy = fixture(0.5), fixture(2.0)
    a = command(light)
    light.step("push", a)
    heavy.step("push", a)
    assert (
        light.snapshot()["bodies"][0]["position"][0] > heavy.snapshot()["bodies"][0]["position"][0]
    )


def test_collision_dissipates_and_produces_audio():
    world = PhysicalWorld.reset(1, {"body_count": 2})
    state = world.snapshot()
    for index, body in enumerate(state["bodies"]):
        body.update(
            position=[-0.2 if index == 0 else 0.2, 0.0],
            velocity=[1.0 if index == 0 else -1.0, 0.0],
            radius=0.18,
            mass=1.0,
            drag=0.5,
        )
    world = PhysicalWorld.restore(state)
    result = world.elapse("collision")
    bodies = world.snapshot()["bodies"]
    energy = sum(0.5 * b["mass"] * np.dot(b["velocity"], b["velocity"]) for b in bodies)
    assert energy < 1
    assert np.linalg.norm(np.array(bodies[0]["position"]) - bodies[1]["position"]) >= 0.36 - 1e-9
    assert float(result["raw_media"]["audio"].abs().max()) > 0.01


@pytest.mark.parametrize("mass", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("drag", [0.5, 1.0, 2.0])
def test_sixteen_round_physical_reachability_witness(mass, drag):
    """Privileged PD controller checks mechanics only, independently of legal oracle."""
    world = fixture(mass, drag)
    for index in range(16):
        body = world.snapshot()["bodies"][0]
        action = command(world)
        requested = (
            mass
            * (64 * (0.65 - body["position"][0]) - (16 - drag) * body["velocity"][0])
            / (80 * 0.8)
        )
        action[11] = max(-0.95, min(0.95, requested))
        world.step(str(index), action)
    assert 0.4 <= world.snapshot()["bodies"][0]["position"][0] <= 0.85


@pytest.mark.parametrize("field", ["rng_states", "audio_ring", "source_pins"])
def test_incomplete_snapshot_rejected(field):
    state = fixture().snapshot()
    del state[field]
    with pytest.raises(ValueError):
        PhysicalWorld.restore(state)


def test_random_legal_trajectories_remain_finite_and_recoverable():
    rng = torch.Generator().manual_seed(20261112)
    for seed in range(24):
        world = PhysicalWorld.reset(seed)
        for index in range(8):
            action = torch.zeros(15, dtype=torch.float64)
            action[0] = 1
            action[9:11] = torch.rand(2, generator=rng) * 2 - 1
            direction = torch.randn(2, generator=rng, dtype=torch.float64)
            action[11:13] = direction / (1 + direction.square().sum()).sqrt()
            action[13] = torch.rand((), generator=rng)
            action[14] = 0.05 * torch.rand((), generator=rng)
            result = world.step(str(index), action)
            assert all(
                bool(torch.isfinite(value).all()) for value in result["observations"].values()
            )
        restored = PhysicalWorld.restore(world.snapshot())
        assert_values_equal(world.elapse("resume"), restored.elapse("resume"))


def test_quota_rejection_preserves_world_and_cached_retry():
    world = fixture()
    world.contract["interface"]["max_transactions"] = 1
    action = command(world)
    first = world.step("one", action)
    digest = value_digest(world.snapshot())
    with pytest.raises(ValueError, match="budget_stop"):
        world.elapse("two")
    assert value_digest(world.snapshot()) == digest
    assert_values_equal(first, world.step("one", action))


def test_cache_size_survives_serialization_and_detects_tampering(tmp_path):
    from taiji.u2_history import atomic_load, atomic_save
    from taiji.u2_task_world import cache_bytes

    world = fixture()
    world.step("one", command(world))
    state = world.snapshot()
    atomic_save(state, tmp_path / "world.bin")
    loaded = atomic_load(tmp_path / "world.bin")
    assert cache_bytes(loaded["transactions"]) == state["byte_count"]
    restored = PhysicalWorld.restore(loaded)
    assert_values_equal(world.elapse("next"), restored.elapse("next"))
    loaded["byte_count"] += 1
    with pytest.raises(ValueError, match="cache_size_mismatch"):
        PhysicalWorld.restore(loaded)
