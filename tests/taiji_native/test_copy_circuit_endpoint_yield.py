"""PLAN-N1-00（S5 终点检测器档）的守卫测试——预注册 §4 的 G3 在单测层的钉子。

钉死三件事：

1. **默认关＝生产路径不可达**：`endpoint_yield_override` 缺省 False 时，即使 prev_byte
   恰好等于所选事件末字节，输出也走既有池化路径（边界符质量为 0）；
2. **G3 非末字节分支逐位不变**：开关开着但轨迹未到末字节时，输出与开关关着**逐位相同**
   ——封死"改动等价于整体关证据"的 §7.5 否决形态；
3. **终点提议的形态**：轨迹到末字节 ⇒ 内容字节质量归零（让位）、边界符拿到 gate 全额
   质量、合法后继掩码不消掉终点提议（边界符不是 UTF-8 字节）。

gate 全部零初始化 ⇒ 新电路的证据恒零（位级惰性设计），行为测试统一用诊断专用的
`gate_override` 给出非零 gate 质量，否则上面三条全是 0=0 的空转。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from taiji import CopyCircuit, Taiji, TaijiConfig

TELL = "我叫阿岩。".encode()
#: 。= E3 80 82，末字节 0x82。
LAST_BYTE = TELL[-1]
GATE = 0.7

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (REPO_ROOT, REPO_ROOT / "scripts" / "training"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))


def _circuit(*, endpoint: bool) -> CopyCircuit:
    values: dict[str, object] = {
        "region_sizes": (64, 48),
        "synapse_fan_in": 16,
        "motor_fan_in": 48,
        "seed": 11,
    }
    model = Taiji(TaijiConfig(**values))
    circuit = CopyCircuit(model.config, max_events=4)
    circuit.store.record(TELL, torch.ones(model.config.cortical_context_dim))
    circuit.gate_override = GATE
    circuit.endpoint_yield_override = endpoint
    return circuit


def _inputs(circuit: CopyCircuit) -> tuple[torch.Tensor, torch.Tensor]:
    cue = torch.zeros(circuit.config.cortical_context_dim)
    f1 = torch.zeros(circuit.config.motor_context_dim)
    f1[5] = 0.5
    return cue, f1


def test_default_off_keeps_the_pooled_path_even_at_last_byte() -> None:
    circuit = _circuit(endpoint=False)
    cue, f1 = _inputs(circuit)
    out = circuit.evidence(cue=cue, f1_context=f1, prev_byte=int(LAST_BYTE))
    boundary = int(circuit.config.boundary_symbol)
    #: 池化路径的特征：边界符零质量、内容字节带 gate×位置质量。
    assert float(out[boundary]) == 0.0
    assert float(out.abs().sum()) == pytest.approx(GATE, abs=1e-6)


def test_g3_non_endpoint_steps_bitwise_unchanged() -> None:
    cue, f1 = _inputs(_circuit(endpoint=False))
    #: 轨迹未到末字节的三种形态：无前字节／前字节是内容中段字节／前字节在内容之外。
    for prev in (None, int(TELL[0]), 0x41):
        off = _circuit(endpoint=False).evidence(
            cue=cue, f1_context=f1, prev_byte=None if prev is None else int(prev)
        )
        on = _circuit(endpoint=True).evidence(
            cue=cue, f1_context=f1, prev_byte=None if prev is None else int(prev)
        )
        assert bool(torch.equal(off, on)), f"endpoint branch leaked at prev_byte={prev!r}"


def test_endpoint_proposes_boundary_at_full_gate_mass() -> None:
    circuit = _circuit(endpoint=True)
    cue, f1 = _inputs(circuit)
    out = circuit.evidence(cue=cue, f1_context=f1, prev_byte=int(LAST_BYTE))
    boundary = int(circuit.config.boundary_symbol)
    assert float(out[boundary]) == pytest.approx(GATE, abs=1e-7)
    #: 让位：内容字节（与除边界符外的全部格位）质量归零。
    out[boundary] = 0.0
    assert float(out.abs().sum()) == 0.0


def test_endpoint_survives_the_utf8_legal_suffix_mask() -> None:
    circuit = _circuit(endpoint=True)
    cue, f1 = _inputs(circuit)
    #: 末字节 0x82 是 UTF-8 续字节，合法后继集很小；边界符不在字面字节集合里，
    #: 掩码若被乘上来会把终点提议整个消掉——这正是终点分支必须绕开掩码的原因。
    utf8_state = (1, 0x82)
    out = circuit.evidence(cue=cue, f1_context=f1, prev_byte=int(LAST_BYTE), utf8_state=utf8_state)
    assert float(out[int(circuit.config.boundary_symbol)]) == pytest.approx(GATE, abs=1e-7)


def test_switch_is_not_a_parameter_and_not_in_the_payload() -> None:
    circuit = _circuit(endpoint=True)
    assert "endpoint_yield_override" not in circuit.PARAMETER_ORDER
    payload = circuit.to_payload()
    assert "endpoint_yield_override" not in payload
