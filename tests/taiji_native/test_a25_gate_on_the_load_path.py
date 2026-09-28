"""owner 裁定 (b)（2026-09-28，PLAN-A-25）在**产品挂载入口**上的守卫。

裁定原文是"**挂载复制回路即开 UTF-8 证据门**"。落地时它只接在
`SeedRuntime.enable_copy_circuit`（显式 opt-in、探针/评测用的那一个入口）上，而产品
自己**唯一**能挂载回路的路径是"档里带回路 ⇒ `Taiji.restore` 自动挂载"
（`model.py` 的 `COPY_CIRCUIT_KEY` 分支；`SeedRuntime.load` → `Seed.from_checkpoint`
→ `Seed.restore` → 适配器 → 这里）。2026-09-28 实测：那条路上门是**关**的
（覆写 `None` ＋ config `False`），于是"带电路的基底一出厂，合法性就被电路的加性证据
打回原形"（`PLAN-A-24` rev18 读数：v3 可解码率 1.0000 → 0.0128）会**静默**发生。

守卫钉四件事：

1. 从带回路的档 restore ⇒ 门**开**，而且是**行为上**开着（电路真的收到 `utf8_state`）；
2. 逃生口：restore 之后显式 `set_copy_evidence_utf8_gate(False)` 仍能关（同基底开/关对照要用）；
3. **不越权**：档里没有回路时覆写保持 `None`（门跟随 config），既有读数一位不动；
4. **语义边界**：`mount_copy_circuit` 直连挂载仍是"门跟随 config"——那是全部既有探针脚本
   用的入口，它们的读数必须逐位不变（本件只把裁定补到产品那条路，不顺手改全体）。
"""

from __future__ import annotations

from typing import Any

import torch

from taiji import Taiji, TaijiConfig

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


def _effective_gate(model: Taiji) -> bool:
    override = model._copy_evidence_utf8_gate_override
    return bool(model.config.copy_evidence_utf8_gate) if override is None else bool(override)


def _feed(model: Taiji, symbols: bytes) -> None:
    model.reset_dynamics(episode_id="a25-load-path")
    model.observe(model.config.boundary_symbol, learn=False, readout="predictive")
    for symbol in symbols:
        model.observe(int(symbol), learn=False, readout="predictive")


def _envelope_with_saved_circuit() -> dict[str, Any]:
    """造一份"档里带已存回路"的信封（先挂、再写内容、再取 `checkpoint()`）。"""

    model = Taiji(_config())
    model.mount_copy_circuit(max_events=4)
    _feed(model, TELL)
    model.record_told_content(TELL)
    return model.checkpoint()


def _seen_states(model: Taiji) -> list[Any]:
    """跑一趟提问，收回电路每次被调用时拿到的 `utf8_state`。"""

    circuit = model.copy_circuit
    assert circuit is not None
    seen: list[Any] = []
    original = circuit.evidence

    def spy(**kwargs: Any) -> torch.Tensor:
        seen.append(kwargs.get("utf8_state"))
        return original(**kwargs)

    circuit.evidence = spy  # type: ignore[method-assign]
    _feed(model, ASK)
    return seen


def test_the_envelope_really_carries_a_circuit() -> None:
    """前置自检：信封里必须真有 `copy_circuit` 键，否则后面几条断言的是"没挂载"。"""

    assert Taiji.COPY_CIRCUIT_KEY in _envelope_with_saved_circuit()


def test_restoring_a_saved_circuit_opens_the_evidence_gate() -> None:
    model = Taiji(_config())
    model.restore(_envelope_with_saved_circuit())
    assert model.copy_circuit is not None, "档里带回路 ⇒ 必须自动挂载"
    assert _effective_gate(model) is True, "裁定 (b)：挂载回路即开证据门"


def test_the_gate_is_live_after_restore_not_just_flagged() -> None:
    """行为面：restore 之后电路真的收到 `utf8_state`（只翻旗标而不被消费＝零生效）。"""

    model = Taiji(_config())
    model.restore(_envelope_with_saved_circuit())
    seen = _seen_states(model)
    assert seen, "提问那趟必须走到电路的证据"
    assert all(state is not None for state in seen), seen


def test_the_escape_hatch_still_closes_the_gate_after_restore() -> None:
    model = Taiji(_config())
    model.restore(_envelope_with_saved_circuit())
    model.set_copy_evidence_utf8_gate(False)
    assert _effective_gate(model) is False
    assert all(state is None for state in _seen_states(model))


def test_restore_without_a_circuit_leaves_the_gate_following_config() -> None:
    """不越权：没有回路的档一位都不动（产品默认基座就是这一档）。"""

    envelope = Taiji(_config()).checkpoint()
    assert Taiji.COPY_CIRCUIT_KEY not in envelope
    restored = Taiji(_config())
    restored.restore(envelope)
    assert restored.copy_circuit is None
    assert restored._copy_evidence_utf8_gate_override is None
    assert _effective_gate(restored) is False


def test_a_direct_mount_still_follows_config() -> None:
    """边界：`mount_copy_circuit` 直连语义不变（全部既有探针脚本走的是它）。"""

    model = Taiji(_config())
    model.mount_copy_circuit(max_events=4)
    assert model.copy_circuit is not None
    assert model._copy_evidence_utf8_gate_override is None
    assert _effective_gate(model) is False
