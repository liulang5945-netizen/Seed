"""B-4 覆盖率第三刀：`neuroplex/resonance/continuous.py`（C25-E 连续时间共振的纯数学单元）。

断言按模块 docstring 声称的契约逐条钉：激活值域/门限、时间积分权重、收敛判据含
**单元素绑定的无偏方差坑**（unbiased=False 那处修复）、theta 嵌套的零回归契约
（`theta_omega=0` 时包络必须恒等于 1.0）、记忆 entrain 的峰值对齐与复位、
环境变量免改码开关。phasor 用鸭子替身（`step` 只调 `evolve`/`binding_tensor` 两个方法）。
"""

from __future__ import annotations

import math

import pytest
import torch

from neuroplex.resonance.continuous import ContinuousResonance


class _PhasorStub:
    """只提供 ContinuousResonance 用到的两个方法，并记录调用参数。"""

    def __init__(self, bindings: torch.Tensor):
        self._bindings = bindings
        self.evolve_calls: list[tuple] = []

    def evolve(self, ids, coactivation=None, dt=None):
        self.evolve_calls.append((tuple(ids), dt))
        return torch.zeros(len(ids), 2)

    def binding_tensor(self, ids, coactivation=None, phasors=None):
        return self._bindings


def test_activation_is_sigmoid_with_offset_as_half_point() -> None:
    cr = ContinuousResonance(act_temp=1.5, act_offset=0.3)
    a = cr.activation(torch.tensor([0.3]))
    assert float(a.item()) == pytest.approx(0.5, abs=1e-6), "binding=act_offset 必须落在 0.5"

    grid = torch.linspace(-1.0, 1.0, 9)
    acts = cr.activation(grid)
    assert bool(((acts > 0) & (acts < 1)).all()), "σ 必须严格落在开区间"
    assert bool((acts.diff() > 0).all()), "激活必须随 binding 单调上升"

    sharp = ContinuousResonance(act_temp=50.0, act_offset=0.0)
    assert float(sharp.activation(torch.tensor([1.0])).item()) > 0.99
    assert float(sharp.activation(torch.tensor([-1.0])).item()) < 0.01


def test_weights_accum_is_time_integral() -> None:
    cr = ContinuousResonance(steps=4, dt=0.25)
    weights = torch.zeros(3)
    activ = torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 0.5]])
    conf = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    for step in range(2):
        weights = cr.weights_accum(weights, activ[step], conf[step], cr.dt)
    assert torch.allclose(
        weights, torch.tensor([0.5 * 1 + 1.0 * 2, 0.5 * 1 + 0.0 * 2, 0.5 * 1 + 0.5 * 2]) * 0.25
    )
    # dt 翻倍 ⇒ 权重翻倍（纯线性积分，没有隐藏归一化）
    doubled = cr.weights_accum(torch.zeros(3), activ[1], conf[1], cr.dt * 2)
    assert torch.allclose(doubled, cr.weights_accum(torch.zeros(3), activ[1], conf[1], cr.dt) * 2)


def test_converged_needs_two_points_and_handles_singleton_binding() -> None:
    cr = ContinuousResonance(conv_tol=0.02)
    assert cr.converged([]) is False and cr.converged([torch.tensor([0.1])]) is False
    assert cr.converged([torch.tensor([]), torch.tensor([])]) is False, "空绑定不能判收敛"

    # 单元素绑定：无偏标准差是 NaN ⇒ 实现用 unbiased=False，这里必须判得出稳定
    singleton = [torch.tensor([0.4]), torch.tensor([0.8])]
    assert cr.converged(singleton) is True, "单神经元基线应判稳定（NaN 会静默阻断收敛）"

    stable = [torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.51])]
    assert cr.converged(stable) is True, "绑定方差的相邻步变化 < tol ⇒ 判锁定"
    # 判定看的是**方差的相邻步变化**，不是绑定本身变了多少
    moving = [torch.tensor([0.1, 0.9, 0.2]), torch.tensor([0.9, 0.1, 0.95])]
    assert cr.converged(moving) is False
    assert cr.converged(moving, tol=1.0) is True, "tol 必须真的参与判定"


def test_lock_degree_boundaries() -> None:
    cr = ContinuousResonance()
    assert cr.lock_degree(torch.tensor([])) == 0.0
    assert cr.lock_degree(torch.ones(5)) == pytest.approx(1.0)
    assert cr.lock_degree(-torch.ones(5)) == pytest.approx(-1.0)
    # ⚠️ 已知含糊：docstring 说"均值高 + 方差低 = 强锁定"，实现只返回均值
    #    （第 153 行算出 std 后丢弃）⇒ 同相与散乱但均值相同的两种绑定给出同一读数。
    same_mean = torch.tensor([0.9, 0.1])
    assert cr.lock_degree(same_mean) == pytest.approx(cr.lock_degree(torch.full((2,), 0.5)))


def test_theta_nesting_is_identity_when_disabled() -> None:
    """零回归契约：theta_omega=0 且无记忆时，包络必须恒等于 1.0。"""

    cr = ContinuousResonance(theta_omega=0.0, theta_amp=0.5)
    for t in (0.0, 0.3, 1.0, 7.5):
        assert cr.theta_envelope(t) == 1.0
        assert cr.theta_phase_at(t) == 0.0
    activ = torch.rand(4)
    assert torch.equal(cr.theta_modulate(activ, 1.25), activ), "未启用嵌套时调制必须是恒等"


def test_theta_nesting_envelope_bounds_period_and_peak() -> None:
    omega, amp = 0.5, 0.2
    cr = ContinuousResonance(theta_omega=omega, theta_amp=amp)
    ts = [0.0, 1.0, 2.5, 6.0, 13.0]
    envs = [cr.theta_envelope(t) for t in ts]
    assert all(1.0 - amp - 1e-9 <= e <= 1.0 + amp + 1e-9 for e in envs), envs
    assert cr.theta_envelope(0.0) == pytest.approx(1.0 + amp), "相位 0 必须是包络峰值"
    period = 2 * math.pi / omega
    assert cr.theta_envelope(1.7) == pytest.approx(cr.theta_envelope(1.7 + period), abs=1e-9)
    assert cr.theta_envelope(period / 2) == pytest.approx(1.0 - amp, abs=1e-9), "半周期必须是谷"

    modulated = cr.theta_modulate(torch.full((3,), 0.4), 0.0)
    assert torch.allclose(modulated, torch.full((3,), 0.4 * (1.0 + amp)))


def test_memory_entrain_pins_peak_and_reset_restores_identity() -> None:
    cr = ContinuousResonance(theta_omega=0.0, theta_amp=0.3)
    assert cr.theta_envelope(5.0) == 1.0

    cr.entrain_memory()  # 默认对齐峰值相位 0
    assert cr.theta_phase_at(99.0) == 0.0, "entrain 后相位必须钉住，不再随 t 走"
    assert cr.theta_envelope(99.0) == pytest.approx(1.3)

    cr.entrain_memory(target_phase=math.pi)
    assert cr.theta_envelope(1.0) == pytest.approx(1.0 - 0.3), "KoPE：相位归属到 π ⇒ 取谷值"

    cr.reset_entrain()
    assert cr.theta_envelope(1.0) == 1.0 and cr.theta_phase_at(1.0) == 0.0


def test_theta_nesting_env_switch(monkeypatch) -> None:
    """TAIJI_THETA_NESTING=1 免改码开嵌套（标定值 ω=0.5）。"""

    monkeypatch.delenv("TAIJI_THETA_NESTING", raising=False)
    assert ContinuousResonance().theta_omega == 0.0

    monkeypatch.setenv("TAIJI_THETA_NESTING", "1")
    cr = ContinuousResonance()
    assert cr.theta_omega == 0.5 and cr.theta_amp == 0.2
    assert cr.theta_envelope(0.0) == pytest.approx(1.2)

    monkeypatch.setenv("TAIJI_THETA_NESTING", "false")
    assert ContinuousResonance().theta_omega == 0.0
    # 显式传入的 omega 优先于 env
    assert ContinuousResonance(theta_omega=1.25).theta_omega == 1.25


def test_step_and_activations_from_phasors_delegate_correctly() -> None:
    cr = ContinuousResonance(act_temp=1.5, act_offset=0.0)
    bindings = torch.tensor([1.0, -1.0, 0.0])
    phasor = _PhasorStub(bindings)
    ids = ["a", "b", "c"]

    new_p, activ = cr.step(phasor, ids, coactivation=None, dt=0.125)
    assert new_p.shape == (3, 2)
    assert phasor.evolve_calls == [(tuple(ids), 0.125)], "step 必须把 dt 透传给 evolve"
    assert torch.allclose(activ, cr.activation(bindings))
    assert float(activ[0]) > float(activ[2]) > float(activ[1]), "同相 > 正交 > 反相"

    direct = cr.activations_from_phasors(phasor, ids)
    assert torch.allclose(direct, activ), "一步到位路径与 step 必须给同一激活"
