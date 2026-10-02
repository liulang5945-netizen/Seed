"""产品入口必须能把复制回路的生命周期门传进去（PLAN-A30，owner 2026-10-02 裁「立项进产品」）。

来历：门与门控计数都落在 `Taiji` 里，但挂载入口 `SeedRuntime.enable_copy_circuit` 只转发兄弟开关
`utf8_gate` ⇒ 从产品入口**够不到**这条门，等于"进了代码、没进产品"。本文件钉入口的形状（签名＋条件转发），
而不是重新跑一次能力面——能力面的数已在 PLAN-A-30 §第五十二~五十六次停靠按件入库。
"""

from __future__ import annotations

import inspect
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_enable_copy_circuit_accepts_the_window_steps_knob() -> None:
    from api.seed_runtime import SeedRuntime

    parameter = inspect.signature(SeedRuntime.enable_copy_circuit).parameters["window_steps"]
    assert parameter.default is None, parameter
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, parameter


def test_the_forward_is_conditional_so_the_default_path_is_untouched() -> None:
    """`None` 必须**完全不触发**覆写——"默认关闭 ⇒ 逐位不变"是靠这条成立的，不是靠约定。"""
    source = (PROJECT_ROOT / "api" / "seed_runtime.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    guarded = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "if window_steps is not None:"
    ]
    assert guarded, "no conditional guard found"
    assert any(
        "substrate.set_copy_evidence_window_steps(window_steps)" in lines[index + offset]
        for index in guarded
        for offset in range(1, 7)
        if index + offset < len(lines)
    ), "forward not inside the guard"


def test_the_guard_can_fail() -> None:
    """反例自检：把转发删掉，上一条断言必须为假（守卫能为 false）。"""
    source = (PROJECT_ROOT / "api" / "seed_runtime.py").read_text(encoding="utf-8")
    stripped = source.replace("substrate.set_copy_evidence_window_steps(window_steps)", "pass")
    assert "substrate.set_copy_evidence_window_steps(window_steps)" not in stripped
