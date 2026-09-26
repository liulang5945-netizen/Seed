"""B-4 第十刀：链式生成的编排契约（generate_task_chain / generate_staged / 场记忆沉淀）。

这段是"三重传递 + 阶段质量门"的调度本体。用脚本化的 `self.generate` 替身喂入预设输出，
断言的是编排规则本身（不依赖模型能不能生成）：

* 空 prompt 阶段 ⇒ 产出 `""`、gates 记 `empty_prompt`，且**不消耗一次前向**；
* `{prev}` 模板 vs 无模板时的换行拼接，两种传文本方式必须各自成立；
* 上一阶段的场状态 ⇒ 下一阶段的 `memory_vectors=[(fs, 0.8)]`（第一阶没有）；
* 阶段抛异常 ⇒ 该阶产出 `""`、gate 记 error 且**截断到 80 字符**，后续阶段继续跑；
* 质量门：退化 ⇒ 高温重试（`+0.15`，上限 1.2），仍退化记 `degenerate`、修好记 `retried`、
  本来正常记 `ok`；`quality_gate=False` 的阶段**不得**重试；
* `record_memory` 走 SleepEngine：在场且文本非空才记，异常记 `skip:`，标签缺省取模板前 40 字符；
* `generate_staged` 的 dict→TaskSet 升级要保住每个字段（含 mode 默认 continuous、
  quality_gate 默认 True），并把参数转交给 v2 调度器。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from neuroplex.brain.cortex import TaskSet

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_capture_module():
    path = PROJECT_ROOT / "scripts" / "training" / "capture_taiji_cortex_rolling_nll_golden.py"
    spec = importlib.util.spec_from_file_location("c1_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = _load_capture_module()


@pytest.fixture()
def cortex(tmp_path):
    import os

    built, _hub, _general_sp, _zh_sp = capture.build_cortex()
    built.neurons_dir = str(tmp_path / "neurons")
    os.makedirs(built.neurons_dir, exist_ok=True)
    return built


class _ScriptedGenerate:
    """按脚本返回预设文本；记录每次调用 kwargs；可按条目抛异常。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def temperatures(self) -> list:
        return [c.get("temperature") for c in self.calls]


def _stage(prompt, **kw):
    return TaskSet(prompt=prompt, **kw)


def test_empty_stage_short_circuits_without_a_forward(cortex, monkeypatch) -> None:
    # 脚本第一项必须落到**第二阶**：空阶段不消耗前向，也就不消费脚本
    gen = _ScriptedGenerate(["第二段的内容。", "不该被取到"])
    monkeypatch.setattr(cortex, "generate", gen)
    monkeypatch.setattr(cortex, "get_last_field_state", lambda: None)

    out = cortex.generate_task_chain([_stage(""), _stage("继续", quality_gate=False)])
    assert out["outputs"][0] == ""
    assert out["gates"][0] == {"error": "empty_prompt"}, "空阶段直接 continue，不该再走质量门判定"
    assert len(gen.calls) == 1, "空阶段不得消耗一次生成前向"
    assert out["outputs"][1] == "第二段的内容。"
    assert len(out["field_states"]) == 2


def test_prev_text_and_field_state_threading(cortex, monkeypatch) -> None:
    fs1 = torch.ones(8)
    gen = _ScriptedGenerate(["甲", "乙", "丙"])
    monkeypatch.setattr(cortex, "generate", gen)
    monkeypatch.setattr(cortex, "get_last_field_state", lambda: fs1)
    monkeypatch.setattr(cortex, "get_last_phase", lambda: 0.25)

    cortex.generate_task_chain(
        [
            _stage("问题一", quality_gate=False),
            _stage("承接：{prev} 然后回答", quality_gate=False),
            _stage("问题三", quality_gate=False),
        ]
    )
    assert gen.calls[0]["prompt"] == "问题一"
    assert gen.calls[0]["memory_vectors"] is None, "第一阶没有上游场状态"
    assert gen.calls[1]["prompt"] == "承接：甲 然后回答", "{prev} 模板必须被上一阶文本填充"
    assert gen.calls[2]["prompt"] == "问题三\n乙", "无模板时应换行拼接上一阶文本"

    memory = gen.calls[1]["memory_vectors"]
    assert memory is not None and len(memory) == 1
    injected_vec, weight = memory[0]
    assert weight == pytest.approx(0.8) and torch.equal(
        injected_vec, fs1
    ), "上一阶段的场状态必须以 0.8 权重进下一阶的 seed_memories"


def test_stage_exception_is_recorded_truncated_and_does_not_stop_chain(cortex, monkeypatch) -> None:
    long_message = "boom " + "x" * 200
    gen = _ScriptedGenerate([RuntimeError(long_message), "后续阶段仍然执行"])
    monkeypatch.setattr(cortex, "generate", gen)
    monkeypatch.setattr(cortex, "get_last_field_state", lambda: None)

    out = cortex.generate_task_chain(
        [_stage("阶一", quality_gate=False), _stage("阶二", quality_gate=False)]
    )
    assert out["outputs"][0] == ""
    assert len(out["gates"][0]["error"]) <= 80, "错误串必须截断，否则诊断面板会被撑爆"
    assert out["outputs"][1] == "后续阶段仍然执行", "一阶失败不得中断整条链"


def test_quality_gate_retry_paths(cortex, monkeypatch) -> None:
    degenerate = "重复重复重复"

    def _degenerate(text: str) -> bool:
        return text == degenerate

    monkeypatch.setattr(cortex, "_is_degenerate_text", staticmethod(_degenerate))

    # 退化 → 高温重试 → 修好：temp 从 0.5 抬到 0.65
    gen = _ScriptedGenerate([degenerate, "正常回答"])
    monkeypatch.setattr(cortex, "generate", gen)
    monkeypatch.setattr(cortex, "get_last_field_state", lambda: None)
    out = cortex.generate_task_chain([_stage(degenerate)], temperature=0.5)
    assert out["gates"][0]["quality"] == "retried"
    assert gen.temperatures == [0.5, 0.65], gen.temperatures

    # 重试仍退化 ⇒ degenerate，且不再二次重试（最多两次前向）
    gen2 = _ScriptedGenerate([degenerate, degenerate, "不该被调用"])
    monkeypatch.setattr(cortex, "generate", gen2)
    out2 = cortex.generate_task_chain([_stage(degenerate)], temperature=0.5)
    assert out2["gates"][0]["quality"] == "degenerate"
    assert len(gen2.calls) == 2

    # 重试温度上限 1.2
    gen3 = _ScriptedGenerate([degenerate, "正常"])
    monkeypatch.setattr(cortex, "generate", gen3)
    cortex.generate_task_chain([_stage(degenerate)], temperature=1.15)
    assert gen3.temperatures[1] == pytest.approx(1.2), "重试温度必须封顶在 1.2"

    # quality_gate=False ⇒ 一次前向、不做退化判定
    gen4 = _ScriptedGenerate([degenerate])
    monkeypatch.setattr(cortex, "generate", gen4)
    out4 = cortex.generate_task_chain([_stage(degenerate, quality_gate=False)])
    assert len(gen4.calls) == 1 and out4["gates"][0]["quality"] == "ok"


def test_memory_recording_uses_sleep_engine_and_falls_back_to_skip(cortex, monkeypatch) -> None:
    recorded: list[tuple] = []

    class _Engine:
        def record_field_memory(self, fs, label, text=None, phase=None):
            recorded.append((fs, label, text, phase))

    import neuroplex.life.sleep_engine as sleep_engine

    monkeypatch.setattr(sleep_engine, "get_sleep_engine", lambda: _Engine())
    gen = _ScriptedGenerate(["阶段产出内容"])
    monkeypatch.setattr(cortex, "generate", gen)
    monkeypatch.setattr(cortex, "get_last_field_state", lambda: torch.ones(4))
    monkeypatch.setattr(cortex, "get_last_phase", lambda: 0.5)

    out = cortex.generate_task_chain(
        [_stage("问题", record_memory=True, memory_label="我的标签", quality_gate=False)]
    )
    assert out["gates"][0]["memory"] == "recorded"
    assert recorded and recorded[0][1] == "我的标签" and recorded[0][3] == 0.5

    # 标签缺省 ⇒ 取模板前 40 字符
    recorded.clear()
    cortex.generate_task_chain([_stage("问" * 60, record_memory=True, quality_gate=False)])
    assert recorded[0][1] == "问" * 40

    # 沉淀抛异常 ⇒ 记 skip，不影响产出
    class _Boom:
        def record_field_memory(self, *a, **k):
            raise RuntimeError("睡眠引擎不可用")

    monkeypatch.setattr(sleep_engine, "get_sleep_engine", lambda: _Boom())
    out2 = cortex.generate_task_chain([_stage("问题", record_memory=True, quality_gate=False)])
    assert out2["gates"][0]["memory"].startswith("skip:")
    assert out2["outputs"] == ["阶段产出内容"], "记忆沉淀失败不得吃掉阶段产出"


def test_capture_field_memory_records_only_when_a_field_state_exists(cortex, monkeypatch) -> None:
    import neuroplex.life.sleep_engine as sleep_engine

    recorded: list[tuple] = []

    class _Engine:
        def record_field_memory(self, fs, label, text=None, phase=None):
            recorded.append((fs, label, text, phase))

    monkeypatch.setattr(sleep_engine, "get_sleep_engine", lambda: _Engine())
    monkeypatch.setattr(cortex, "get_last_field_state", lambda: None)
    cortex._capture_field_memory("问题", "回答")
    assert recorded == [], "没有真实场状态时不该沉淀"

    monkeypatch.setattr(cortex, "get_last_field_state", lambda: torch.ones(4))
    monkeypatch.setattr(cortex, "get_last_phase", lambda: 0.125)
    cortex._capture_field_memory("  带空白的提问  ", "回答")
    assert recorded[0][1] == "带空白的提问" and recorded[0][3] == 0.125
    cortex._capture_field_memory("", "回答")
    assert recorded[-1][1] == "auto", "空 prompt 时标签退回 auto"


def test_generate_staged_upgrades_dicts_and_forwards_params(cortex, monkeypatch) -> None:
    seen: dict = {}

    def fake_chain(task_sets, **kw):
        seen["task_sets"] = task_sets
        seen["kw"] = kw
        return {"outputs": ["a", "b"], "field_states": [None, None], "gates": [{}, {}]}

    monkeypatch.setattr(cortex, "generate_task_chain", fake_chain)
    result = cortex.generate_staged(
        [
            {"prompt": "一", "domain": "zh", "mode": "fusion", "max_tokens": 7, "temperature": 0.9},
            {"prompt": "二"},
        ],
        max_tokens_per_stage=21,
        temperature=0.3,
        top_k=9,
    )
    assert result == ["a", "b"], "兼容层返回各阶段文本，不是 v2 的 dict"
    first, second = seen["task_sets"]
    assert first.prompt == "一" and first.domain == "zh" and first.mode == "fusion"
    assert first.max_tokens == 7 and first.temperature == 0.9
    assert second.mode == "continuous", "缺省 mode 必须是 continuous"
    assert second.quality_gate is True, "缺省开质量门"
    assert seen["kw"]["max_tokens_per_stage"] == 21 and seen["kw"]["top_k"] == 9
