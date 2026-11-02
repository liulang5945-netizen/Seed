"""Contract guard for 10.21.2: the declared graph, counts, cache and restore boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taiji.query_binding import (  # noqa: E402
    PARAMETER_GROUPS,
    S_TRAINABLE_GROUPS,
    BindingConfig,
    BindingFeatures,
    binding_loss,
    build_binding,
    expected_parameter_count,
    make_optimizer,
    module_bundle_payload,
    module_fixture_state,
    save_module_bundle,
    set_trainable,
    trainable_parameters,
)


def small_config() -> BindingConfig:
    return BindingConfig(width=8, query_width=8)


def small_features() -> BindingFeatures:
    generator = torch.Generator().manual_seed(17021)
    dtype = torch.float64
    return BindingFeatures(
        goal_bytes=b"module-fixture",
        records=(),
        state=module_fixture_state(),
        goal_features=torch.randn(6, 8, dtype=dtype, generator=generator),
        s=torch.randn(8, dtype=dtype, generator=generator),
        source_positions=torch.randn(3, 8, dtype=dtype, generator=generator),
        source_metadata=torch.randn(3, 4, dtype=dtype, generator=generator),
        source_mask=torch.ones(3, dtype=torch.bool),
        action_objects=torch.randn(3, 8, dtype=dtype, generator=generator),
        action_places=torch.randn(3, 8, dtype=dtype, generator=generator),
        cache_key={},
    )


def test_parameter_matrix_and_group_counts():
    module = build_binding(BindingConfig())
    assert module.parameter_count() == 105760
    assert expected_parameter_count(128, 32, 2, 4) == 105760
    counts = {
        group: sum(getattr(module, name).numel() for name in PARAMETER_GROUPS[group])
        for group in PARAMETER_GROUPS
    }
    assert counts == {
        "K_goal": 4096,
        "V_goal": 16512,
        "role": 64,
        "W_state": 4128,
        "Q_source": 9248,
        "K_source": 4224,
        "V_source": 17024,
        "source_null": 160,
        "W_bind": 49280,
        "U": 512,
        "V": 512,
    }
    small = build_binding(small_config())
    assert small.parameter_count() == 904
    assert expected_parameter_count(8, 8, 2, 4) == 904


def test_s_arm_trainable_count_and_initial_forward():
    config = BindingConfig()
    stop = build_binding(config)
    set_trainable(stop, S_TRAINABLE_GROUPS)
    params = trainable_parameters(stop, S_TRAINABLE_GROUPS)
    assert sum(parameter.numel() for parameter in params) == 50304
    assert sum(parameter.numel() for parameter in params if parameter.requires_grad) == 50304
    for group in PARAMETER_GROUPS:
        if group in S_TRAINABLE_GROUPS:
            continue
        assert not any(getattr(stop, name).requires_grad for name in PARAMETER_GROUPS[group])
    features = small_features()
    small_joint = build_binding(small_config())
    small_stop = build_binding(small_config())
    set_trainable(small_stop, S_TRAINABLE_GROUPS)
    joint_goal, joint_actions, _ = small_joint.forward_binding(features)
    stop_goal, stop_actions, _ = small_stop.forward_binding(features)
    joint_scores = small_joint.score(joint_goal, joint_actions)
    stop_scores = small_stop.score(stop_goal, stop_actions)
    assert torch.equal(joint_goal, stop_goal)
    assert torch.equal(joint_actions, stop_actions)
    assert torch.equal(joint_scores, stop_scores)


def test_forward_shapes_probabilities_and_finite():
    module = build_binding(small_config())
    features = small_features()
    b_goal, b_actions, content = module.forward_binding(features)
    assert b_goal.shape == (8,)
    assert b_actions.shape == (3, 8)
    assert content.shape == (2, 8)
    scores = module.score(b_goal, b_actions)
    probabilities = torch.softmax(scores, dim=0)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    assert abs(float(probabilities.detach().sum()) - 1.0) < 1e-12
    loss = binding_loss(scores, 1)
    assert torch.isfinite(loss)


def test_gradients_reach_every_group_and_input():
    module = build_binding(small_config())
    features = small_features()
    b_goal, b_actions, _ = module.forward_binding(features)
    loss = -torch.log_softmax(module.score(b_goal, b_actions), dim=0)[1]
    loss.backward()
    for group in PARAMETER_GROUPS:
        gradients = [
            getattr(module, name).grad
            for name in PARAMETER_GROUPS[group]
            if getattr(module, name).grad is not None
        ]
        assert gradients, group
        assert all(torch.isfinite(gradient).all() for gradient in gradients), group
    leaves = {
        "goal": features.goal_features.detach().clone().requires_grad_(True),
        "positions": features.source_positions.detach().clone().requires_grad_(True),
        "s": features.s.detach().clone().requires_grad_(True),
    }
    input_features = BindingFeatures(
        goal_bytes=features.goal_bytes,
        records=features.records,
        state=features.state,
        goal_features=leaves["goal"],
        s=leaves["s"],
        source_positions=leaves["positions"],
        source_metadata=features.source_metadata,
        source_mask=features.source_mask,
        action_objects=features.action_objects,
        action_places=features.action_places,
        cache_key={},
    )
    input_goal, input_actions, _ = module.forward_binding(input_features)
    input_scores = module.score(input_goal, input_actions)
    for leaf in leaves.values():
        gradient = torch.autograd.grad(input_scores.sum(), leaf, retain_graph=True)[0]
        assert float(gradient.norm()) > 0.0


def test_rank_four_score_equation():
    module = build_binding(small_config())
    z_bind = torch.randn(8, dtype=torch.float64)
    b_actions = torch.randn(3, 8, dtype=torch.float64)
    left = z_bind @ module.U
    right = b_actions @ module.V
    assert torch.allclose(module.score(z_bind, b_actions), right @ left)


def test_module_bundle_roundtrip_preserves_scores(tmp_path):
    config = small_config()
    module = build_binding(config)
    features = small_features()
    b_goal, b_actions, _ = module.forward_binding(features)
    scores = module.score(b_goal, b_actions)
    optimizer = make_optimizer(module, tuple(PARAMETER_GROUPS))
    payload = module_bundle_payload(
        "small-J-float64",
        config,
        module,
        optimizer,
        features,
        None,
        1,
        tuple(PARAMETER_GROUPS),
        {"fixture_index": 0, "position": 0},
    )
    path = tmp_path / "small.pt"
    save_module_bundle(path, payload)
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    restored = build_binding(BindingConfig(**bundle["config"]))
    restored.load_state_dict(bundle["parameters"], strict=True)
    restored_goal, restored_actions, _ = restored.forward_binding(
        BindingFeatures(
            goal_bytes=bundle["features"]["goal"],
            records=(),
            state=module_fixture_state(),
            goal_features=bundle["features"]["goal_features"],
            s=bundle["features"]["s"],
            source_positions=bundle["features"]["source_positions"],
            source_metadata=bundle["features"]["source_metadata"],
            source_mask=bundle["features"]["source_mask"],
            action_objects=bundle["features"]["action_objects"],
            action_places=bundle["features"]["action_places"],
            cache_key={},
        )
    )
    restored_scores = restored.score(restored_goal, restored_actions)
    assert torch.equal(scores.detach(), restored_scores.detach())
