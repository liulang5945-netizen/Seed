"""PLAN-A-27 §2.6.5（owner 裁定 (d)）守卫：锁规则 byte_overlap 产品化。

1. **默认规则**＝`byte_overlap`（config 字段默认翻转——产品默认变更，owner 2026-09-28 批）；
2. **分派正确**：`byte_overlap` 档 `picked`＝"与提问共享字符"列的 argmax；
   `cue_only` 档保持旧实现（cue 余弦＋学习头）逐位不变；
3. **响亮失败**：未知规则名不许静默回退；
4. **口径修正**：`last_question_bytes` 从 `_serialize` 铺出的整段文本里取**最后一个提问轮**
   （无标记时原样返回——旧行为的逃生口）。
"""

from __future__ import annotations

import pytest
import torch

from taiji import Taiji, TaijiConfig
from taiji.copy_circuit import LOCK_RULES, last_question_bytes


def _config(**overrides):
    values = {
        "region_sizes": (64, 48),
        "synapse_fan_in": 16,
        "motor_fan_in": 48,
        "seed": 11,
    }
    values.update(overrides)
    return TaijiConfig(**values)


def _two_event_model(rule: str | None) -> Taiji:
    overrides = {} if rule is None else {"lock_selection_rule": rule}
    model = Taiji(_config(**overrides))
    model.mount_copy_circuit(max_events=4)
    cue = torch.zeros(model.config.cortical_context_dim)
    cue[1] = 1.0
    #: 两条告知：旧的那条含"雨桐"，新的那条含"明轩"——提问"我叫明轩"按 byte_overlap
    #: 必须选中新的（与提问共享字符多），按 cue_only 的位置尺子倾向选旧的。
    model.copy_circuit.store.record("我表哥叫雨桐。".encode("utf-8"), cue.clone())
    model.copy_circuit.store.record("我叫明轩。".encode("utf-8"), cue.clone())
    return model


ASK = "我叫明轩。我的名字是什么？"
SERIALIZED = f"问：我表哥叫雨桐。\n答：我表哥叫雨桐。\n问：{ASK}\n答：".encode("utf-8")


def _observe_then_select(model: Taiji, prompt: bytes):
    model.reset_dynamics(episode_id="lock-rule-guard")
    model.observe(model.config.boundary_symbol, learn=False, readout="predictive")
    for symbol in prompt:
        model.observe(int(symbol), learn=False, readout="predictive", use_memory=False)
    return model.copy_circuit.selection(
        cue=model.fabric.cortical_context(model._state.regions),
        f1_context=model._state.motor_context,
        query_bytes=last_question_bytes(prompt),
    )


def test_default_rule_is_byte_overlap() -> None:
    assert TaijiConfig().lock_selection_rule == "byte_overlap"
    assert "cue_only" in LOCK_RULES


def test_byte_overlap_rule_picks_the_question_overlapping_event() -> None:
    model = _two_event_model(None)  # 默认＝byte_overlap
    state = _observe_then_select(model, SERIALIZED)
    assert state["rule"] == "byte_overlap"
    events = [event.content.decode("utf-8") for event in model.copy_circuit.store.events()]
    assert "明轩" in events[state["picked"]]
    assert "雨桐" not in events[state["picked"]]


def test_cue_only_rule_keeps_the_legacy_scoring() -> None:
    model = _two_event_model("cue_only")
    state = _observe_then_select(model, SERIALIZED)
    assert state["rule"] == "cue_only"
    #: 旧实现逐位不变：picked ＝ argmax(cue 余弦列 + 学习头)。
    scores = state["features"][:, 0] + state["head_values"]
    legacy_picked = int(scores.argmax().item())
    assert state["picked"] == legacy_picked


def test_unknown_rule_fails_loudly() -> None:
    model = _two_event_model("recency_only")
    model.reset_dynamics(episode_id="lock-rule-guard")
    with pytest.raises(ValueError, match="unknown lock_selection_rule"):
        model.copy_circuit.selection(
            cue=torch.zeros(model.config.cortical_context_dim),
            f1_context=torch.zeros(model.config.motor_context_dim),
            query_bytes=ASK.encode("utf-8"),
        )


def test_last_question_bytes_extracts_the_final_question_turn() -> None:
    assert last_question_bytes(SERIALIZED) == ASK.encode("utf-8")
    #: 单轮
    assert last_question_bytes("问：你好？\n答：".encode("utf-8")) == "你好？".encode("utf-8")
    #: 无标记（旧调用方直喂原文）⇒ 原样返回
    assert last_question_bytes("就叫明轩".encode("utf-8")) == "就叫明轩".encode("utf-8")
    assert last_question_bytes(b"") == b""
