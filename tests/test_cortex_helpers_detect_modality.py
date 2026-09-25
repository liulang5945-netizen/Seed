"""`detect_modality` 从 `Cortex` 方法抽离为纯函数后的**逐位等价**守卫（P2 神对象拆分）。

来历（2026-09-25）：审计 P2「Cortex 神对象拆分」推进——按 `_cortex_helpers.py` 已建立的
"纯函数 + staticmethod 委托" 模式，把 `detect_modality`（该簇里**唯一无 self 状态**的成员）
抽为纯函数。另两个目标成员（`_infer_domain` / `_reencode_domain_generation_context`）
**依赖 self**（`self.neurons`），属真重构、需负责人签字，**不在本件范围**。

钉住两件事：

1. `Cortex.detect_modality(x)`（staticmethod 委托）与
   `_cortex_helpers.detect_modality(x)` 对每个输入**逐位相同**；
2. 该函数保持**纯函数**（可经 `Cortex` 类直接调用、无需实例），
   证明它没有依赖任何 self 状态（否则委托就名不副实）。
"""

from __future__ import annotations

import torch

from neuroplex.brain import _cortex_helpers
from neuroplex.brain.cortex import Cortex

CASES = [
    {"modality": "image", "data": "x"},
    {"data": "x"},
    torch.zeros((2, 4, 8), dtype=torch.float32),
    torch.zeros((2, 4, 8), dtype=torch.long),
    torch.zeros((2, 4), dtype=torch.long),
    "这是一段中文文本",
    "some english text",
    123,
    None,
]


def test_delegate_and_helper_are_bit_identical() -> None:
    for case in CASES:
        assert Cortex.detect_modality(case) == _cortex_helpers.detect_modality(case), repr(case)


def test_the_function_is_pure_and_needs_no_instance() -> None:
    """staticmethod ⇒ 经 `Cortex` 类直接调用即可，证明不依赖任何 self 状态。"""

    results = {_cortex_helpers.detect_modality(case) for case in CASES}
    assert results <= {"text", "image", "audio", "video"}
    assert _cortex_helpers.detect_modality({"modality": "video"}) == "video"
    assert _cortex_helpers.detect_modality(torch.zeros((1, 1, 1))) == "image"
    assert _cortex_helpers.detect_modality("hi") == "text"


def test_degenerate_text_still_matches_helper() -> None:
    """顺带钉住既有委托（防本次改动误伤）。"""

    assert Cortex._is_degenerate_text("1.\n") == _cortex_helpers.is_degenerate_text("1.\n")
    assert Cortex._is_degenerate_text("正常中文句子。") == _cortex_helpers.is_degenerate_text(
        "正常中文句子。"
    )
