"""产品出口默认位的守卫：`SeedRuntime.chat()` 的重复惩罚默认必须是 2.0。

owner 裁定（2026-09-28）"取 2.0 为产品默认"，依据是 `PLAN-A-30` §2f 两段独立题面
（成句 12/72→71/72 与 7/72→69/72；同字节连写均值 5.44→1.00 与 6.19→1.01；命中 0→0 未见代价）。

默认位这件事在本仓已经栽过两次同型的：**裁定只落在一个入口，另一条路静默走旧行为**
（`PLAN-A-28` §0 的证据门、`PLAN-A-26` §6.2 的位置输入）。所以这里钉的不是"常量等于 2.0"，
而是**产品出口实际把什么值交给解码器**——用桩件把 `generate_input` 拦下来看参数。
同时钉住"旧面仍可钉"：评测/仪器要复现裁定前的读数必须能显式传 0.0（`eval_taiji_cap0_*` 等已就地钉住）。
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from api.seed_runtime import PRODUCT_REPETITION_PENALTY, SeedRuntime


class _Emission:
    def __init__(self, text: bytes) -> None:
        self.text_bytes = text
        self.backend_id = "stub-organ"


class _Organ:
    backend_id = "stub-organ"

    def emit(self, expression: Any) -> _Emission:
        return _Emission("好。".encode())


class _Substrate:
    copy_circuit = None
    tick = 0


class _Model:
    """只记录解码参数的桩：产品出口传下去什么，这条守卫就看得见什么。"""

    def __init__(self) -> None:
        self.substrate = _Substrate()
        self.tick = 0
        self.seen: list[float | None] = []
        self.seen_kwargs: dict[str, Any] = {}

    def generate_input(self, frame: Any, length: int, **kwargs: Any) -> bytes:
        self.seen.append(kwargs.get("repetition_penalty"))
        self.seen_kwargs = dict(kwargs)
        return "好。".encode()


def _runtime() -> tuple[SeedRuntime, _Model]:
    runtime = SeedRuntime.__new__(SeedRuntime)
    model = _Model()
    runtime.model = model  # type: ignore[attr-defined]
    runtime._lock = threading.Lock()  # type: ignore[attr-defined]
    runtime._chat_organ = _Organ()  # type: ignore[attr-defined]
    runtime._observe_provider_health = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    runtime._record_told_history = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    return runtime, model


def test_the_ruling_is_recorded_in_code_as_two() -> None:
    assert PRODUCT_REPETITION_PENALTY == 2.0


def test_chat_default_hands_the_product_penalty_to_the_decoder() -> None:
    runtime, model = _runtime()
    runtime.chat("我的名字是什么？", history=[("我叫阿岩。", "好。")], learn=False)
    assert model.seen == [PRODUCT_REPETITION_PENALTY], model.seen


def test_explicit_zero_still_reproduces_the_pre_ruling_face() -> None:
    """旧面必须**可达**：全部既有封存读数是在 0.0 上取的，仪器要复现就得能显式钉住。"""

    runtime, model = _runtime()
    runtime.chat("我的名字是什么？", history=[], learn=False, repetition_penalty=0.0)
    assert model.seen == [0.0]


def test_mask_and_stop_semantics_are_untouched_by_the_default() -> None:
    """默认位换了，SPEC-R2-02 那把掩码与"停在边界"的权力不许跟着变。"""

    runtime, model = _runtime()
    runtime.chat("我家住在哪里？", history=[], learn=False)
    assert model.seen == [2.0]
    assert model.seen_kwargs.get("utf8_strict") is True, model.seen_kwargs
    assert model.seen_kwargs.get("stop_at_boundary") is True, model.seen_kwargs
    assert model.seen_kwargs.get("sample") is False, model.seen_kwargs


def test_negative_penalty_is_rejected_loudly() -> None:
    runtime, _model = _runtime()
    with pytest.raises(ValueError, match="cannot be negative"):
        runtime.chat("我的名字是什么？", history=[], learn=False, repetition_penalty=-0.5)
