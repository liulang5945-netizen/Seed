"""PLAN-R2-01 守卫：读出侧的 UTF-8 位置输入（4 维 one-hot，默认关）。

三条守卫直接对应提案 §5：

1. **关闭时逐位可载旧档**：默认配置下读出的输入维数、权重形状、payload 与 digest 全不动，
   旧 payload 载回来张量逐位相同。
2. **零初始化 ⇒ 开启但未训时逐位等于关闭**：位置列存在但全零，前向证据与关闭时逐位相同。
3. **"被走到"计数**：开启后前向（预测）与后向（学习）都真被喂到，权重真离开零；
   开关开了却没喂 ⇒ 响亮失败，不静默空转。

另附两条钉语义：位置类严格等于共享状态机 `utf8_state.advance_utf8` 的 DFA 余量推进；
payload 往返（可载 / 关闭却拿到带位置列的档 ⇒ 响亮失败）。
"""

from __future__ import annotations

import pytest
import torch

from taiji import Taiji, TaijiConfig, content_digest
from taiji.organs import BytePredictiveReadout
from taiji.utf8_state import UTF8_POSITION_DIM, advance_utf8, remaining_after

EPISODE = "r2-plan-01-guard"

#: 与 `train_taiji_langfloor.py` 的 arm A 配方一致：写面唯一＝predictive_readout。
OBSERVE_KWARGS = {
    "learn": True,
    "readout": "predictive",
    "learn_motor": False,
    "learn_fabric": False,
    "learn_predictive_context": False,
    "learn_predictive_readout": True,
}

#: 一段含 1/2/3 字节 UTF-8 序列与 ASCII 的字节流（覆盖 remaining 的四种取值）。
SAMPLE = "a中bé€c".encode("utf-8")


def _config(*, position: bool = False) -> TaijiConfig:
    base = TaijiConfig.capacity_profile(300_000, seed=20260927)
    if not position:
        return base
    return TaijiConfig.from_dict({**base.to_dict(), "readout_utf8_position_input": True})


def _readout(config: TaijiConfig, *, seed: int = 20260927) -> BytePredictiveReadout:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return BytePredictiveReadout(config, generator=generator, device="cpu")


def _model(config: TaijiConfig) -> Taiji:
    return Taiji(config, episode_id=EPISODE)


def _feed(model: Taiji, data: bytes) -> None:
    for symbol in data:
        model.observe(int(symbol), **OBSERVE_KWARGS)


# ---------------------------------------------------------------- 守卫 1：默认关

def test_default_config_keeps_payload_and_shape_unchanged() -> None:
    config = _config()
    assert config.readout_utf8_position_input is False
    readout = _readout(config)
    assert readout.position_weight is None
    assert readout.position_input_enabled is False
    payload = readout.to_payload()
    assert BytePredictiveReadout.POSITION_PAYLOAD_KEY not in payload
    #: 旧档（无位置键）载回来必须逐位相同。
    restored = _readout(config, seed=1)  # 不同的初始化流，靠载入覆盖
    restored.load_payload(payload)
    torch.testing.assert_close(
        restored.synapses.edge_weight, readout.synapses.edge_weight, rtol=0, atol=0
    )
    torch.testing.assert_close(restored.bias, readout.bias, rtol=0, atol=0)
    assert content_digest(restored.to_payload()) == content_digest(payload)


def test_disabled_readout_ignores_position_state() -> None:
    config = _config()
    readout = _readout(config)
    context = torch.randn(config.motor_context_dim, generator=torch.Generator().manual_seed(7))
    baseline = readout.probabilities(context)
    torch.testing.assert_close(
        readout.probabilities(context, position_state=2), baseline, rtol=0, atol=0
    )


# ---------------------------------------------------------------- 守卫 2：零初始化惰性

def test_enabled_position_input_is_bit_inert_until_trained() -> None:
    plain = _model(_config())
    enabled = _model(_config(position=True))
    enabled_readout = enabled.predictive_readout
    assert enabled_readout.position_input_enabled
    assert enabled_readout.position_weight is not None
    assert enabled_readout.position_weight.shape == (
        enabled.config.alphabet_size,
        UTF8_POSITION_DIM,
    )
    #: 位置列零初始化：权重全零，且不改变 `SparseSynapses` 的拓扑与初值。
    assert float(enabled_readout.position_weight.abs().sum().item()) == 0.0
    torch.testing.assert_close(
        enabled_readout.synapses.edge_weight, plain.predictive_readout.synapses.edge_weight,
        rtol=0, atol=0,
    )
    #: 只读推理（learn=False）下，开启与关闭必须逐位同 —— 零列不移动任何 logit。
    for symbol in SAMPLE:
        left = plain.observe(int(symbol), learn=False, readout="predictive")
        right = enabled.observe(int(symbol), learn=False, readout="predictive")
        torch.testing.assert_close(left.probabilities, right.probabilities, rtol=0, atol=0)


# ---------------------------------------------------------------- 守卫 3：被走到

def test_enabled_position_input_is_wired_for_prediction_and_learning() -> None:
    model = _model(_config(position=True))
    _feed(model, SAMPLE * 4)
    readout = model.predictive_readout
    assert readout.position_probability_steps == len(SAMPLE) * 4
    assert readout.position_learn_steps > 0
    assert float(readout.position_weight.abs().sum().item()) > 0.0


def test_enabled_readout_refuses_missing_position_state() -> None:
    config = _config(position=True)
    readout = _readout(config)
    context = torch.randn(config.motor_context_dim, generator=torch.Generator().manual_seed(11))
    with pytest.raises(ValueError, match="no utf-8 position state"):
        readout.probabilities(context)
    target = torch.zeros(config.alphabet_size)
    target[65] = 1.0
    with pytest.raises(ValueError, match="no utf-8 position state"):
        readout.learn(context, target, 65)


def test_position_state_outside_dfa_range_is_rejected() -> None:
    readout = _readout(_config(position=True))
    context = torch.randn(
        _config().motor_context_dim, generator=torch.Generator().manual_seed(13)
    )
    with pytest.raises(ValueError, match="outside the 0..3"):
        readout.probabilities(context, position_state=UTF8_POSITION_DIM)


# ---------------------------------------------------------------- 语义钉

def test_position_class_tracks_the_shared_utf8_state_machine() -> None:
    model = _model(_config(position=True))
    expected = 0
    for symbol in SAMPLE:
        model.observe(int(symbol), **OBSERVE_KWARGS)
        expected = remaining_after(expected, int(symbol))
        assert model.snapshot().motor_position_class == expected
    #: 参考走法：直接用共享状态机独立重算一遍，两条路必须同值。
    reference = 0
    for symbol in SAMPLE:
        reference = advance_utf8(reference, 0, int(symbol))[0]
    assert model.snapshot().motor_position_class == reference


def test_disabled_config_leaves_state_payload_without_position_key() -> None:
    model = _model(_config())
    _feed(model, SAMPLE)
    state_payload = model.snapshot().to_payload()
    assert "motor_position_class" not in state_payload
    enabled = _model(_config(position=True))
    _feed(enabled, SAMPLE)
    assert "motor_position_class" in enabled.snapshot().to_payload()


# ---------------------------------------------------------------- payload 往返

def test_position_columns_round_trip_and_refuse_cross_config_load() -> None:
    enabled = _readout(_config(position=True))
    with torch.no_grad():
        enabled.position_weight.normal_(generator=torch.Generator().manual_seed(17))
    payload = enabled.to_payload()
    assert BytePredictiveReadout.POSITION_PAYLOAD_KEY in payload
    restored = _readout(_config(position=True), seed=2)
    restored.load_payload(payload)
    torch.testing.assert_close(
        restored.position_weight, enabled.position_weight, rtol=0, atol=0
    )
    #: 开启的档载进关闭的读出器 ⇒ 响亮失败（不许静默丢列）。
    disabled = _readout(_config())
    with pytest.raises(ValueError, match="position columns"):
        disabled.load_payload(payload)
    #: 关闭的档载进开启的读出器 ⇒ 位置列保持零（这正是"开启但未训＝关闭"那条）。
    legacy = _readout(_config()).to_payload()
    fresh = _readout(_config(position=True), seed=3)
    fresh.load_payload(legacy)
    assert float(fresh.position_weight.abs().sum().item()) == 0.0


def test_position_input_rejected_on_developmental_f1_path() -> None:
    model = _model(_config(position=True))
    with pytest.raises(ValueError, match="not wired to the developmental F1"):
        model.migrate_f1_to_developmental_synapses()


def test_parameter_accounting_counts_position_columns_when_enabled() -> None:
    plain = _config()
    enabled = _config(position=True)
    assert (
        enabled.planned_active_parameter_count - plain.planned_active_parameter_count
        == plain.alphabet_size * UTF8_POSITION_DIM
    )
    model = _model(enabled)
    assert model.parameter_count() - _model(plain).parameter_count() == (
        plain.alphabet_size * UTF8_POSITION_DIM
    )