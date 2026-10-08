"""PLAN-N1-01（S1a 硬序贯读取档）的守卫测试——预注册 §3.1 的 G-S1-4 在单测层的钉子。

钉死四件事：

1. **默认关＝生产路径不可达**：`successor_restricted_addressing` 缺省 False 时 `_successor_positions`
   返回 None（掩码根本不下发），且证据里能看到**非后继位置**的字节质量；
2. **开关开着时确实限制寻址**（否则这档是空转）：中段前字节的场合，质量全部落在"前驱＝该字节"的
   那些后继位置 ⇒ 证据的唯一非零内容字节就是那个后继字节；
3. **本轮第一步的落点是位置 0**（`prev_byte` 为 None＝从存储事件的开头读起），这是设计选择不是巧合；
4. **空后继集＝回退到不限制并计数**：输出与开关关着**逐位相同**（回退不许悄悄改变被测路径），
   且回退次数被记下来——不然 J-S1a 会有一半在另一条路径上被量。

gate 零初始化 ⇒ 新电路证据恒零，故统一用诊断专用的 `gate_override` 给非零 gate 质量。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from taiji import CopyCircuit, Taiji, TaijiConfig

TELL = "我叫阿岩。".encode()
#: 中段前字节＝「我」的末字节；它的唯一后继位置是 3，那里的字节就是 SUCCESSOR_BYTE（「叫」的首字节）。
PREV_MID_BYTE = TELL[2]
SUCCESSOR_BYTE = TELL[3]
FIRST_BYTE = TELL[0]
LAST_BYTE = TELL[-1]
#: 不在这条告知里的字节：事件内找不到它的后继 ⇒ 走回退支。
ABSENT_BYTE = 0x41
GATE = 0.7

REPO_ROOT = Path(__file__).resolve().parents[2]
for _entry in (REPO_ROOT, REPO_ROOT / "scripts" / "training"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))


def _circuit(*, restricted: bool) -> CopyCircuit:
    model = Taiji(
        TaijiConfig(
            **{
                "region_sizes": (64, 48),
                "synapse_fan_in": 16,
                "motor_fan_in": 48,
                "seed": 11,
            }
        )
    )
    circuit = CopyCircuit(model.config, max_events=4)
    circuit.store.record(TELL, torch.ones(model.config.cortical_context_dim))
    circuit.gate_override = GATE
    circuit.successor_restricted_addressing = restricted
    return circuit


def _evidence(circuit: CopyCircuit, prev: int | None) -> torch.Tensor:
    cue = torch.zeros(circuit.config.cortical_context_dim)
    f1 = torch.zeros(circuit.config.motor_context_dim)
    f1[5] = 0.5
    return circuit.evidence(cue=cue, f1_context=f1, prev_byte=prev)


def _content_mass(out: torch.Tensor, circuit: CopyCircuit) -> torch.Tensor:
    out = out.clone()
    out[int(circuit.config.boundary_symbol)] = 0.0
    return out


def test_default_off_does_not_emit_a_mask() -> None:
    circuit = _circuit(restricted=False)
    codes = torch.tensor(list(TELL), device=circuit.device, dtype=torch.long)
    assert circuit._successor_positions(codes, int(PREV_MID_BYTE)) is None
    assert circuit._successor_positions(codes, None) is None
    #: 默认关着也不该留下回退计数——它只在开关开着时才有意义。
    assert circuit.successor_restriction_fallbacks == 0


def test_off_leaves_non_successor_bytes_in_play() -> None:
    circuit = _circuit(restricted=False)
    out = _content_mass(_evidence(circuit, int(PREV_MID_BYTE)), circuit)
    nonzero = int((out > 0.0).sum())
    #: 池化档的特征：一条 15 字节的告知里有 12 个不同字节，质量会散到多个字节上（不是只跟着前驱）。
    assert nonzero > 1, f"默认档应当把质量散到多个字节上，实测只有 {nonzero} 格"
    assert float(out.abs().sum()) == pytest.approx(GATE, abs=1e-6)


def test_on_concentrates_mass_on_the_successor_byte() -> None:
    circuit = _circuit(restricted=True)
    out = _content_mass(_evidence(circuit, int(PREV_MID_BYTE)), circuit)
    nonzero = torch.nonzero(out > 0.0).flatten().tolist()
    assert nonzero == [int(SUCCESSOR_BYTE)], f"质量必须只落在唯一后继字节上，实测 {nonzero}"
    assert float(out[int(SUCCESSOR_BYTE)]) == pytest.approx(GATE, abs=1e-6)


def test_first_step_of_the_turn_reads_position_zero() -> None:
    circuit = _circuit(restricted=True)
    out = _content_mass(_evidence(circuit, None), circuit)
    nonzero = torch.nonzero(out > 0.0).flatten().tolist()
    #: prev=None ⇒ 允许集＝位置 0 ⇒ 唯一非零字节是告知的首字节（这是 S1a 的设计选择）。
    assert nonzero == [int(FIRST_BYTE)], f"第一步应落在事件起点，实测 {nonzero}"
    assert circuit.successor_restriction_fallbacks == 0


def test_empty_successor_set_falls_back_bitwise_identical_and_counts() -> None:
    off = _circuit(restricted=False)
    on = _circuit(restricted=True)
    before = int(on.successor_restriction_fallbacks)
    a = _evidence(off, int(ABSENT_BYTE))
    b = _evidence(on, int(ABSENT_BYTE))
    assert bool(torch.equal(a, b)), "回退支不许改变输出——否则判据量的是另一条路径"
    assert int(on.successor_restriction_fallbacks) == before + 1, "回退必须被计数，不许静默"


def test_switch_is_not_a_parameter_and_not_in_the_payload() -> None:
    circuit = _circuit(restricted=True)
    assert "successor_restricted_addressing" not in circuit.PARAMETER_ORDER
    payload = circuit.to_payload()
    assert "successor_restricted_addressing" not in payload
    assert "successor_restriction_fallbacks" not in payload
