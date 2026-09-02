"""阶段 1 放大配置的失败测试：训练用放大画像必须存在且可构造。

计划要求：在 ``taiji/config.py`` 增加训练用放大配置（区域数/维度/边密度）。
放大画像不是新的动力学——它只把基底做大，全部校验规则与默认画像共用，
因此任何画像都必须原地通过 ``TaijiConfig`` 的全部 ``__post_init__`` 约束。
"""

import pytest
import torch

from taiji import CapacityPolicy, Taiji, TaijiConfig


def test_training_profile_scales_regions_dimensions_and_edge_density() -> None:
    profile = TaijiConfig.training_profile()
    default = TaijiConfig()

    assert profile.region_sizes > default.region_sizes
    assert profile.synapse_fan_in > default.synapse_fan_in
    assert profile.motor_fan_in > default.motor_fan_in
    assert profile.memory_units > default.memory_units
    assert profile.memory_fan_in > default.memory_fan_in
    assert profile.memory_time_dim > default.memory_time_dim
    assert profile.memory_episode_dim > default.memory_episode_dim
    assert profile.cortical_context_dim > default.cortical_context_dim


def test_training_profile_builds_and_steps() -> None:
    model = Taiji(TaijiConfig.training_profile(seed=311))
    assert model.parameter_count() > Taiji(TaijiConfig(seed=311)).parameter_count()

    model.observe(97, learn=True)
    step = model.observe(98, learn=True)
    assert step.probabilities.shape == (model.config.alphabet_size,)


def test_training_profile_rejects_non_positive_scale() -> None:
    try:
        TaijiConfig.training_profile(scale=0)
    except ValueError:
        pass
    else:
        raise AssertionError("scale must be positive")


def test_planned_parameter_count_matches_constructed_substrate() -> None:
    for config in (TaijiConfig(seed=311), TaijiConfig.training_profile(scale=2, seed=311)):
        assert config.planned_active_parameter_count == Taiji(config).parameter_count()


def test_capacity_profile_fits_a_parameter_budget_deterministically() -> None:
    # The invariant is "a budget above the default substrate yields a strictly
    # larger substrate", so the budget is derived from the default rather than
    # hardcoded: M1-63 promoted the identity organ onto the default path, and a
    # fixed literal would silently start describing a *smaller* target.
    default = TaijiConfig(seed=311)
    budget = int(default.planned_active_parameter_count * 1.5)
    profile = TaijiConfig.capacity_profile(budget, seed=311)

    assert profile.planned_active_parameter_count <= budget
    assert profile.planned_active_parameter_count >= int(budget * 0.75)
    assert profile.region_sizes > default.region_sizes
    assert profile == TaijiConfig.capacity_profile(budget, seed=311)
    assert Taiji(profile).parameter_count() == profile.planned_active_parameter_count


def test_capacity_profile_rejects_a_budget_below_the_smallest_valid_fabric() -> None:
    try:
        TaijiConfig.capacity_profile(1, seed=311)
    except ValueError:
        pass
    else:
        raise AssertionError("an impossible parameter budget must be rejected")


def test_capacity_policy_can_change_depth_and_structural_proportions() -> None:
    policy = CapacityPolicy(
        region_ratios=(1.0, 0.60, 0.35, 0.20),
        synapse_fan_in_ratio=0.15,
        motor_fan_in_ratio=0.30,
        memory_units_ratio=1.25,
        memory_fan_in_ratio=0.20,
        memory_meta_ratio=0.30,
        memory_readout_fan_in_ratio=0.25,
        memory_time_ratio=0.20,
        memory_episode_ratio=0.25,
        lateral_fan_in_ratio=0.10,
        alignment=8,
    )
    profile = TaijiConfig.capacity_profile(300_000, policy=policy, seed=311)

    assert len(profile.region_sizes) == 4
    assert profile.region_sizes[1] < profile.region_sizes[0]
    assert profile.region_sizes[-1] < profile.region_sizes[1]
    default_profile = TaijiConfig.capacity_profile(300_000, seed=311)
    assert profile.memory_time_dim > default_profile.memory_time_dim
    assert profile.memory_episode_dim > default_profile.memory_episode_dim
    assert profile.planned_active_parameter_count <= 300_000
    assert Taiji(profile).parameter_count() == profile.planned_active_parameter_count


def test_capacity_policy_rejects_invalid_ratios() -> None:
    with pytest.raises(ValueError, match="region_ratios"):
        CapacityPolicy(region_ratios=())
    with pytest.raises(ValueError, match="positive"):
        CapacityPolicy(memory_units_ratio=0.0)


def test_capacity_policy_roundtrips_as_external_search_input() -> None:
    policy = CapacityPolicy(region_ratios=(1.0, 0.65, 0.40), alignment=16)

    assert CapacityPolicy.from_dict(policy.to_dict()) == policy


def test_capacity_policy_loads_older_json_without_memory_dimension_ratios() -> None:
    policy = CapacityPolicy(region_ratios=(1.0, 0.65, 0.40), alignment=16)
    payload = policy.to_dict()
    payload.pop("memory_time_ratio")
    payload.pop("memory_episode_ratio")

    assert CapacityPolicy.from_dict(payload) == CapacityPolicy(
        region_ratios=(1.0, 0.65, 0.40), alignment=16
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is not available")
def test_cuda_substrate_builds_steps_and_keeps_learned_tensors_on_device() -> None:
    config = TaijiConfig.capacity_profile(100_000, seed=311)
    model = Taiji(config, device="cuda")

    model.observe(97, learn=True)
    step = model.observe(98, learn=True)

    assert step.probabilities.device.type == "cuda"
    assert all(tensor.device.type == "cuda" for tensor in model.parameter_tensors())
