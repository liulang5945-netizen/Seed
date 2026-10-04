"""A-30 仪器 v8 的旗标守卫：`--no-copy-evidence-gate` 必须**被走到**，且能为 false。

为什么要有这一支（不是形式主义）：v7 之前把"挂载回路"当成一个变量，但 `enable_copy_circuit` 一次开**两样**东西
——回路自己的 prompt／加性证据通道，以及那条按 UTF-8 位置状态门控证据的门（`copy_evidence_utf8_gate`）。
不拆开就只能写"是挂载回路这一动作造成"，不能写"是回路的通道造成"。

两条断言的分工：

1. `test_flag_and_guard_are_declared_in_the_instrument` **无条件跑**（只读源码文本）：旗标、自述字段与
   守卫三样缺一就红——这样即使本机的 `*.pt` 不在（第二支会 skip），这一支仍然在守。
2. `test_the_flag_actually_disarms_the_evidence_gate` 跑真实装配：挂回路 ⇒ 门有效值为真；
   再按旗标置 False ⇒ 有效值必须变假。**这条就是"能为 false"的实物**：置了没生效就是红。
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT = PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a30_stop_failure.py"
CIRCUIT = PROJECT_ROOT / "output" / "taiji_r2_copy_circuit_chat" / "judge" / "circuit-final.pt"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "seed_a31self_with_circuit.pt"


def _effective_gate(substrate) -> bool:
    """读产品公开出口三键里的 effective（DEBT-G30 迁移；`config-or-override` 那条式子只住在产品里）。"""

    return bool(substrate.copy_evidence_utf8_gate_state()["effective"])


def test_flag_and_guard_are_declared_in_the_instrument() -> None:
    source = INSTRUMENT.read_text(encoding="utf-8")

    assert '"--no-copy-evidence-gate"' in source, "v8 的旗标不见了 ⇒ 那一格中间档又只能靠命令行猜"
    assert "copy_evidence_gate_off_requested" in source, "旗标没被自述 ⇒ 事后读件分不出哪一档用过它"
    assert "evidence_gate_flag_honored" in source, "缺『被走到』守卫 ⇒ 旗标静默失效也不会红"


@pytest.mark.skipif(
    not (CIRCUIT.is_file() and CHECKPOINT.is_file()),
    reason="needs gitignored *.pt: " + str(CHECKPOINT.relative_to(PROJECT_ROOT)),
)
def test_the_flag_actually_disarms_the_evidence_gate() -> None:
    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(CHECKPOINT)
    runtime.enable_copy_circuit(CIRCUIT)
    armed = _effective_gate(runtime.model.substrate)
    assert armed is True, "挂上回路后门本应被武装 ⇒ 这条前提变了，v8 的理由要重估"

    runtime.model.substrate.set_copy_evidence_utf8_gate(False)
    assert (
        _effective_gate(runtime.model.substrate) is False
    ), "旗标算式置了却读回真 ⇒ `--no-copy-evidence-gate` 那一档其实没关掉门，读数会假"
