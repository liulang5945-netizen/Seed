"""SPEC-A-26 形状甲的守卫：注入形状开关——默认逐位不变、开了必须被走到、竞争式真的在竞争。

来历（2026-10-04，owner 弹窗未及时点、按在手的"未及时决策按推荐推进"授权立项）：§125 三判据
A/B/C 各自不成立后，"从不发 LF 那一群"只剩注入侧这条路；预注册冻在 `plans/reference/SPEC-A-26_…md`，
判据（L1/L2/L3）与否证分支先于本代码存在。本文件钉三件事，**不测能力**（能力面归仪器与冻结判据）：

1. 默认 `"additive"` ＝ 现行行为（逐位不变的 config 侧前提）；
2. 开关的验证与 `None` 语义（回到跟随 config，不是"关"）；
3. 竞争式真的在竞争——零证据档与加性逐位相同（电路零初始化 ⇒ max(p, p+0)=p），
   非零证据档与加性可分，且"被走到"计数只在竞争式应用时前进。
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import torch

from taiji import Taiji, TaijiConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT = "我叫阿岩。我的名字是什么？".encode()


def _config(**overrides: Any) -> TaijiConfig:
    values: dict[str, Any] = {
        "region_sizes": (64, 48),
        "synapse_fan_in": 16,
        "motor_fan_in": 48,
        "seed": 11,
    }
    values.update(overrides)
    return TaijiConfig(**values)


def _answer_with_mode(mode: str | None, *, config_mode: str = "additive") -> tuple[bytes, Taiji]:
    model = Taiji(_config(copy_evidence_injection_mode=config_mode))
    model.mount_copy_circuit(max_events=4)
    if mode is not None:
        model.set_copy_evidence_injection_mode(mode)
    answer = model.generate(PROMPT, 24)
    return answer, model


def test_default_mode_is_additive() -> None:
    assert TaijiConfig().copy_evidence_injection_mode == "additive"
    assert Taiji(_config()).copy_evidence_injection_state() == {
        "mode": "additive",
        "competitive_steps": 0,
        "gate_min_overlap": 0.0,
        "gated_armed_answers": 0,
        "gated_silenced_answers": 0,
    }


def test_setter_validates_and_none_restores_config_following() -> None:
    model = Taiji(_config(copy_evidence_injection_mode="competitive"))
    assert model.copy_evidence_injection_state()["mode"] == "competitive"
    model.set_copy_evidence_injection_mode("additive")
    assert model.copy_evidence_injection_state()["mode"] == "additive"
    #: `None` 不是"关"，是"回到跟随 config"——与 set_copy_evidence_utf8_gate 同一语义。
    model.set_copy_evidence_injection_mode(None)
    assert model.copy_evidence_injection_state()["mode"] == "competitive"
    with pytest.raises(ValueError, match="additive"):
        model.set_copy_evidence_injection_mode("greedy")
    with pytest.raises(TypeError):
        model.set_copy_evidence_injection_mode(True)  # type: ignore[arg-type]


def test_untrained_circuit_competitive_matches_additive_bit_for_bit() -> None:
    """电路零初始化 ⇒ 证据是精确零向量 ⇒ max(p, p+0)=p ⇒ 两条形状必须逐位相同。

    这是"默认关闭 ⇒ 逐位不变"在**开着** competitive 时的退化版自证：零证据档不构成分岔。
    同时 competitive_steps > 0 证明这条路真的被走到了（不是被静默跳过）；
    步数＝边界＋prompt＋生成（prompt 步也走注入点，`Taiji.generate` 内的既定事实）。
    """

    answers: dict[str, bytes] = {}
    for mode in ("additive", "competitive"):
        answer, model = _answer_with_mode(mode)
        answers[mode] = answer
        expected_steps = 0 if mode == "additive" else len(PROMPT) + 1 + 24
        state = model.copy_evidence_injection_state()
        assert state["mode"] == mode and state["competitive_steps"] == expected_steps, state
        assert state["gated_armed_answers"] == 0 and state["gated_silenced_answers"] == 0, state
    assert answers["additive"] == answers["competitive"], (
        answers["additive"],
        answers["competitive"],
    )


def test_competitive_semantics_equals_elementwise_max_of_both_branches() -> None:
    """竞争式的语义钉：最终分布＝归一化(max(无证据臂, 加性臂))——逐格，不重实现。

    加性臂的分布恰好就是竞争式里"复制分支"那一路（同一条读出头、同一份证据），
    零证据臂就是"基线分支"——所以用本模型自己的两条 observe 臂当分支真值，
    断言竞争臂 == renorm(max(两臂))。任何一处实现漂移（漏归一、换成求和、
    竞争改在 logits 上做）都会当场红。
    """

    boost_byte = 0x41

    def _probs_with_evidence(value: float, mode: str) -> torch.Tensor:
        model = Taiji(_config())
        model.mount_copy_circuit(max_events=4)
        evidence = torch.zeros(model.config.alphabet_size)
        evidence[boost_byte] = value
        model.copy_circuit.evidence = lambda **kwargs: evidence  # type: ignore[method-assign]
        model.set_copy_evidence_injection_mode(mode)
        model.reset_dynamics(episode_id="a26-competitive-semantics")
        model.observe(model.config.boundary_symbol, learn=False, readout="predictive")
        model.observe(int(PROMPT[0]), learn=False, readout="predictive")
        #: 本步算出的完整分布住在状态里（TaijiStep 只带标量 prior_probability）。
        return model._state.motor_probabilities

    p_zero = _probs_with_evidence(0.0, "additive")
    p_add = _probs_with_evidence(0.35, "additive")
    p_comp = _probs_with_evidence(0.35, "competitive")
    expected = torch.maximum(p_zero, p_add)
    expected = expected / expected.sum()
    assert torch.allclose(p_comp, expected, atol=1e-6), (p_comp - expected).abs().max()
    #: 竞争式不等于加性：零证据臂赢过的那些格保住了自己的概率（这正是"不抬非边界分"的形状）。
    assert not torch.allclose(p_comp, p_add, atol=1e-6)


def test_gated_mode_k_setter_and_byte_overlap_requirement() -> None:
    """形状乙的 K 设定器（验证／None 回 0.0）与"量名相符"的响亮失败约束（§147 冻结⑤）。"""

    model = Taiji(_config())
    model.set_copy_evidence_injection_mode("gated")
    assert model.copy_evidence_injection_state()["mode"] == "gated"
    assert model.copy_evidence_injection_state()["gate_min_overlap"] == 0.0
    model.set_copy_evidence_structural_min_overlap(0.2)
    assert model.copy_evidence_injection_state()["gate_min_overlap"] == 0.2
    model.set_copy_evidence_structural_min_overlap(None)
    assert model.copy_evidence_injection_state()["gate_min_overlap"] == 0.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.set_copy_evidence_structural_min_overlap(1.5)
    with pytest.raises(TypeError):
        model.set_copy_evidence_structural_min_overlap(True)  # type: ignore[arg-type]
    #: gated 的量是 byte_overlap 锁规则的记分列——配 cue_only 时 scores 名不符实，当场抛错。
    strict = Taiji(_config(lock_selection_rule="cue_only"))
    strict.set_copy_evidence_injection_mode("gated")
    with pytest.raises(ValueError, match="byte_overlap"):
        strict.generate(PROMPT, 4)


def test_gated_silences_and_arms_per_answer() -> None:
    """按答案资格门控的端到端语义（事件与提问共享 1 字 ⇒ overlap 1/7≈0.143）：

    * K=0.2 ⇒ 资格未过 ⇒ 整答静音 ⇒ 答复与"无回路"逐位相同，计数记 silenced；
    * K=0.0 ⇒ 资格过 ⇒ 通道照加性武装 ⇒ 答复与加性档逐位相同，计数记 armed。
    门槛是**按答**一次判定：同一答内不换（锁事件时刻定死）。
    """

    question = "我的名字是什么？".encode()

    def _gated_run(k: float | None) -> tuple[bytes, dict[str, Any]]:
        model = Taiji(_config(copy_evidence_injection_mode="gated"))
        model.mount_copy_circuit(max_events=4)
        model.copy_circuit.store.record(
            "我叫阿岩，今年九岁。".encode(), torch.zeros(model.copy_circuit.store.cue_dim)
        )
        evidence = torch.zeros(model.config.alphabet_size)
        evidence[0x41] = 0.35
        model.copy_circuit.evidence = lambda **kwargs: evidence  # type: ignore[method-assign]
        if k is not None:
            model.set_copy_evidence_structural_min_overlap(k)
        return model.generate(question, 24), model.copy_evidence_injection_state()

    gated_strict, state_strict = _gated_run(0.2)
    gated_loose, state_loose = _gated_run(0.0)

    #: 参照臂：同 stub 的加性档与无回路档。
    additive_model = Taiji(_config())
    additive_model.mount_copy_circuit(max_events=4)
    additive_model.copy_circuit.store.record(
        "我叫阿岩，今年九岁。".encode(), torch.zeros(additive_model.copy_circuit.store.cue_dim)
    )
    evidence = torch.zeros(additive_model.config.alphabet_size)
    evidence[0x41] = 0.35
    additive_model.copy_circuit.evidence = (  # type: ignore[method-assign]
        lambda **kwargs: evidence
    )
    additive_answer = additive_model.generate(question, 24)
    nocircuit_answer = Taiji(_config()).generate(question, 24)

    assert gated_strict == nocircuit_answer, (gated_strict, nocircuit_answer)
    assert state_strict["gated_silenced_answers"] == 1 and state_strict["gated_armed_answers"] == 0
    assert gated_loose == additive_answer, (gated_loose, additive_answer)
    assert state_loose["gated_armed_answers"] == 1 and state_loose["gated_silenced_answers"] == 0


def test_product_entry_forwards_injection_mode() -> None:
    """产品入口必须能把注入形状传进去（照 window_steps 那条的形状：kw-only、默认 None）。"""

    from api.seed_runtime import SeedRuntime

    parameter = inspect.signature(SeedRuntime.enable_copy_circuit).parameters["injection_mode"]
    assert parameter.default is None, parameter
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, parameter
    source = (PROJECT_ROOT / "api" / "seed_runtime.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    guarded = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "if injection_mode is not None:"
    ]
    assert guarded, "no conditional guard found"
    assert any(
        "substrate.set_copy_evidence_injection_mode(injection_mode)" in lines[index + offset]
        for index in guarded
        for offset in range(1, 7)
        if index + offset < len(lines)
    ), "forward not inside the guard"
    #: 形状乙的 K 走同一条入口（kw-only、默认 None、条件转发）。
    parameter = inspect.signature(SeedRuntime.enable_copy_circuit).parameters["gate_min_overlap"]
    assert parameter.default is None, parameter
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, parameter
    guarded_k = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "if gate_min_overlap is not None:"
    ]
    assert guarded_k, "no conditional guard for gate_min_overlap"
    assert any(
        "substrate.set_copy_evidence_structural_min_overlap(gate_min_overlap)"
        in lines[index + offset]
        for index in guarded_k
        for offset in range(1, 7)
        if index + offset < len(lines)
    ), "K forward not inside the guard"


def test_the_model_has_exactly_one_injection_site() -> None:
    """注入只许住一处（SPEC-A-26 §6②）；竞争计数与模式解析也各只写一份。"""

    source = (PROJECT_ROOT / "taiji" / "model.py").read_text(encoding="utf-8")
    assert source.count("self._copy_circuit.evidence(") == 1, source.count(
        "self._copy_circuit.evidence("
    )
    assert source.count("self._copy_evidence_competitive_steps += 1") == 1
    #: 模式解析（config-or-override）只住一处（私有 helper，DEBT-G30 的教训当场应用）；
    #: 状态读数／注入点／generate 锁块三处都必须走它，内联旧形状必须绝迹。
    assert source.count("def _copy_evidence_injection_mode_effective") == 1
    assert source.count("self._copy_evidence_injection_mode_effective()") >= 3
    assert source.count("if self._copy_evidence_injection_mode_override is None") == 0
    config_source = (PROJECT_ROOT / "taiji" / "config.py").read_text(encoding="utf-8")
    assert 'copy_evidence_injection_mode: str = "additive"' in config_source
