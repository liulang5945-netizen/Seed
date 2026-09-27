"""PLAN-A-25 守卫：复制回路的加性证据按 UTF-8 位置状态门控（默认关）。

由来（`PLAN-A-24` rev18 实测）：位置输入保住的字节合法性只在裸读出那条链上成立；
挂上 A2 电路后输出又变乱码（v3 可解码率 1.0000 → 0.0128），因为电路的证据把整条事件内容的
字节码池化成一个 257 维向量、**完全不知道当前位置能接什么字节**。本件让它只提议合法后继字节。

三条守卫：

1. **默认关逐位不变**：配置关时状态 payload 不含新键，且电路拿到的 `utf8_state` 恒为 `None`；
2. **掩码判定来自共享状态机**（不另写一份），且边界符不在合法集里；
3. **门开只滤非法字节**：非法后继字节上的证据被置零，合法字节上的证据原样保留。
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from taiji import CopyCircuit, Taiji, TaijiConfig
from taiji.utf8_state import UTF8_POSITION_DIM, utf8_allowed

TELL = "我叫阿岩。".encode()
ASK = "我的名字是什么？".encode()


def _config(**overrides: Any) -> TaijiConfig:
    values: dict[str, Any] = {
        "region_sizes": (64, 48),
        "synapse_fan_in": 16,
        "motor_fan_in": 48,
        "seed": 11,
    }
    values.update(overrides)
    return TaijiConfig(**values)


def _model(**overrides: Any) -> Taiji:
    return Taiji(_config(**overrides))


def _feed(model: Taiji, symbols: bytes) -> torch.Tensor:
    model.reset_dynamics(episode_id="a25-gate")
    step = model.observe(model.config.boundary_symbol, learn=False, readout="predictive")
    last = step
    for symbol in symbols:
        last = model.observe(int(symbol), learn=False, readout="predictive")
    return last.probabilities.detach().cpu().clone()


# ---------------------------------------------------------------- 守卫 1：默认关

def test_default_gate_is_off_and_state_payload_has_no_new_keys() -> None:
    config = _config()
    assert config.copy_evidence_utf8_gate is False
    model = _model()
    _feed(model, TELL)
    payload = model.snapshot().to_payload()
    assert "motor_utf8_lead" not in payload
    assert "motor_position_class" not in payload


def test_gate_off_never_feeds_utf8_state_to_the_circuit() -> None:
    """配置关 ⇒ 电路收到的 `utf8_state` 必须是 `None`（否则"默认关"只是名义上的）。"""

    seen: list[Any] = []
    model = _model()
    model.mount_copy_circuit()
    circuit = model.copy_circuit
    assert circuit is not None
    _feed(model, TELL)
    model.record_told_content(TELL)
    original = circuit.evidence

    def spy(**kwargs: Any) -> torch.Tensor:
        seen.append(kwargs.get("utf8_state"))
        return original(**kwargs)

    circuit.evidence = spy  # type: ignore[method-assign]
    _feed(model, ASK)
    assert seen, "证据必须被调用过"
    assert all(state is None for state in seen)


def test_gate_on_tracks_the_lead_byte_in_state() -> None:
    """门开 ⇒ 状态里带上 (余量, 首字节)；对齐后首字节是**真首字节**。"""

    model = _model(copy_evidence_utf8_gate=True)
    #: 先喂一个完整汉字让跟踪对齐，再喂一个汉字的首字节停在序列中间。
    _feed(model, "我".encode() + bytes([0xE6]))
    state = model.snapshot()
    assert state.motor_position_class == 2  # 还剩 2 个续字节
    assert state.motor_utf8_lead == 0xE6  # 「我」的首字节
    payload = state.to_payload()
    assert payload["motor_utf8_lead"] == 0xE6
    assert payload["motor_position_class"] == 2


def test_boundary_symbol_quirk_in_the_shared_state_machine_is_pinned() -> None:
    """**已知缺陷（钉住现状，不在本件修）**：边界符 256 在共享状态机里被当成
    "4 字节序列的引导字节"（`advance_utf8` 只看 `symbol < 0x80 / < 0xE0 / < 0xF0 / else`），
    所以刚喂过边界符那一步的 (余量, 首字节) 是 `(3, 256)` 而不是 `(0, 0)`。

    实测后果有限：纯 CJK 文本在一个汉字之内就重新对齐（3 个字节正好把幻影计数抵消），
    但**第一个汉字内部**的首字节是错的。修它必须改 `advance_utf8` 的语义，
    而两臂（F0 过线那两臂）的训练与评测都用的是现状语义 ⇒ **只能与下一轮训练一起修**。
    本守卫的作用是把它钉在明处，而不是让它悄悄漂走。
    """

    model = _model(copy_evidence_utf8_gate=True)
    _feed(model, bytes([0xE6]))  # 边界符之后直接喂一个汉字的首字节
    state = model.snapshot()
    assert (state.motor_position_class, state.motor_utf8_lead) == (2, 256)


def test_utf8_tracking_stays_off_when_both_switches_are_off() -> None:
    model = _model()
    _feed(model, "我".encode() + bytes([0xE6]))
    state = model.snapshot()
    assert state.motor_position_class is None
    assert state.motor_utf8_lead is None


# ---------------------------------------------------------------- 守卫 2：掩码来自共享状态机

@pytest.mark.parametrize(
    "utf8_state",
    [(0, 0), (1, 0xE4), (2, 0xE0), (2, 0xED), (3, 0xF0), (3, 0xF4)],
)
def test_legal_suffix_mask_matches_the_shared_state_machine(utf8_state) -> None:
    circuit = CopyCircuit(_config(), max_events=4)
    mask = circuit.legal_suffix_mask(utf8_state)
    allowed = {int(index) for index in mask.nonzero().flatten().tolist()}
    assert allowed == set(utf8_allowed(*utf8_state))
    #: 边界符（`alphabet_size`>256）不进合法集：掩码只约束字面字节，不碰停止语义。
    assert circuit.config.boundary_symbol not in allowed
    assert len(allowed) > 0


def test_position_class_dimension_is_still_four() -> None:
    assert UTF8_POSITION_DIM == 4


# ---------------------------------------------------------------- 守卫 3：门开只滤非法字节

def _populated_circuit(model: Taiji) -> CopyCircuit:
    model.mount_copy_circuit()
    circuit = model.copy_circuit
    assert circuit is not None
    circuit.store.record(TELL, torch.ones(model.config.cortical_context_dim))
    weights = torch.ones(len(TELL))
    circuit.address_override = weights
    circuit.gate_override = 20.0
    return circuit


def test_gated_evidence_zeroes_illegal_successor_bytes_only() -> None:
    model = _model()
    circuit = _populated_circuit(model)
    cue = torch.ones(model.config.cortical_context_dim)
    f1_context = torch.zeros(model.config.motor_context_dim)

    #: 处在一个 3 字节汉字的第 2 个续字节位置（remaining=1）⇒ 只允许 0x80..0xBF。
    utf8_state = (1, 0xE6)
    ungated = circuit.evidence(cue=cue, f1_context=f1_context, utf8_state=None)
    gated = circuit.evidence(cue=cue, f1_context=f1_context, utf8_state=utf8_state)

    legal = set(utf8_allowed(*utf8_state))
    assert float(ungated[0xE6]) > 0.0, "未门控时内容里的引导字节是有证据的"
    assert float(gated[0xE6]) == 0.0, "门开后引导字节必须被滤掉"
    for code in range(circuit.config.alphabet_size):
        if code in legal:
            assert float(gated[code]) == float(ungated[code]), code
        else:
            assert float(gated[code]) == 0.0, code
            assert float(ungated[code]) >= 0.0, code


def test_gate_on_keeps_mount_bitwise_inert_when_store_is_empty() -> None:
    """门开但库空/证据为零 ⇒ 与不挂电路逐位相同（不破坏"零初始化即惰性"这条既有规矩）。"""

    plain = _model()
    gated = _model(copy_evidence_utf8_gate=True)
    gated.mount_copy_circuit()
    assert torch.equal(_feed(plain, ASK), _feed(gated, ASK))