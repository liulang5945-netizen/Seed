"""Full public task input, role distinction, credit and baseline recovery."""

import pytest
import torch

from scripts.training.task_baselines_22_4_a import TaskSequenceBaseline
from taiji.u2_history import assert_values_equal
from taiji.u2_subject import U2Config, synthetic_fixture


@pytest.mark.parametrize("architecture", ["gru", "transformer"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_question_scene_credit_and_restore(architecture, dtype, tmp_path):
    torch.set_num_threads(1)
    model = TaskSequenceBaseline(architecture).to(dtype=dtype)
    fixture = synthetic_fixture(U2Config.small(), dtype)
    state = model.set_goal(model.initial(), "按观测可靠回答当前请求。")
    state = model.observe(state, fixture["events"][0]["observations"], fixture["coordinates"])
    first = model.set_request(state, "红色圆片现在在哪里？")
    second = model.set_request(state, "红色圆片上一次在哪里？")
    assert first["count"] > 8 and second["count"] > 8
    assert not torch.equal(model.output(first), model.output(second))
    loss = -model.output(first)[ord("a")]
    gradients = torch.autograd.grad(
        loss, [model.frontend.encoder["rgb"].weight, model.frontend.text_input.weight]
    )
    assert all(bool(g.any()) and bool(torch.isfinite(g).all()) for g in gradients)
    first = model.accept_byte(first, 0xE5)
    model.save(tmp_path / "baseline.bin", first)
    restored, other = TaskSequenceBaseline.load(tmp_path / "baseline.bin")
    assert_values_equal(first, other)
    for byte in (0xB7, 0xA6, 256):
        first = model.accept_byte(first, byte)
        other = restored.accept_byte(other, byte)
    assert_values_equal(first, other)
    assert not first["active"]
    assert abs(sum(p.numel() for p in model.parameters()) / 22742 - 1) < 0.10


@pytest.mark.parametrize("architecture", ["gru", "transformer"])
def test_missing_modalities_and_parameter_epoch(architecture):
    model = TaskSequenceBaseline(architecture)
    state = model.set_goal(model.initial(), "当前任务。")
    assert model.observe(state, {}, {})["count"] == state["count"]
    state = model.set_request(state, "完整的问题。")
    model.parameter_revision += 1
    with pytest.raises(ValueError, match="parameter_window"):
        model.output(state)
    rebuilt = model.reencode(state["ops"])
    assert rebuilt["revision"] == model.parameter_revision


@pytest.mark.parametrize("architecture", ["gru", "transformer"])
def test_sequence_limit_is_explicit(architecture):
    model = TaskSequenceBaseline(architecture, max_tokens=8)
    with pytest.raises(ValueError, match="sequence_budget_stop"):
        model.set_goal(model.initial(), "一个超过八字节的完整目标。")
