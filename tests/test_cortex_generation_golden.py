"""B-3 C-4 黄金等价测试：`_generate_p7` 的**单步解码算法**抽离到
`neuroplex/brain/_cortex_generation.decode_step` 后，`Cortex.generate()` 对同一装配
（同种子）复现**迁移前**端到端输出（15 组：5 case × 3 prompt）。

夹具：`assemble_cortex()` + `torch.manual_seed(SEED)`（与采集时同种子 ⇒ fallback 神经元
随机初始化确定）。**断言纪律**：每次 `generate()` 前都重设 `torch.manual_seed(SEED)`——
`generate` 内 `torch.multinomial` 逐 token 吃全局 RNG，不重设则跨用例污染、黄金必红。

⚠️ 覆盖说明：fallback general 神经元无 embed_adapter ⇒ 相似度分支（路由层）未被覆盖；
本测只覆盖**生成解码**链路（重复惩罚 / no-repeat-ngram / EOS bias / 熵停止 / top-k 采样）。

CASE kwargs 必须与 `scripts/training/capture_cortex_generation_golden.py::CASES` 保持同步，
否则"复现"是对着一个被改过的基准——既证明等价又反过来钉死迁移前语义。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neuroplex.brain._cortex_generation import decode_step

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads(
    (PROJECT_ROOT / "reports" / "cortex_generation_golden_20260926.json").read_text(encoding="utf-8")
)
FIX_SEED = int(GOLDEN["seed"])

# 必须与 capture_cortex_generation_golden.py::CASES 逐项一致（仅 max_tokens 固定 24）。
CASES = [
    {"tag": "default", "kwargs": {"max_tokens": 24}},
    {"tag": "topk0", "kwargs": {"max_tokens": 24, "top_k": 0}},
    {"tag": "norep1", "kwargs": {"max_tokens": 24, "repetition_penalty": 1.0}},
    {"tag": "domain_zh", "kwargs": {"max_tokens": 24, "domain": "zh"}},
    {"tag": "temp1", "kwargs": {"max_tokens": 24, "temperature": 1.0}},
]
CASE_KWARGS = {c["tag"]: c["kwargs"] for c in CASES}


def _reproduce(prompt: str, kwargs: dict) -> str:
    # 与捕获脚本同协议（capture_cortex_generation_golden.py::_run）：
    #   torch.manual_seed(SEED) → assemble_cortex() → generate()
    # ⚠️ 关键：**不在 generate 前重设种子**——assembly 会消耗全局 RNG，generate 看到的
    #   正是"种子被 assembly 吃掉部分后"的状态；重设会抹掉该消耗 ⇒ forward/multinomial 起点
    #   偏移 ⇒ 首个采样即偏离。逐行重装配（每次 seed→assemble→generate）才能逐位复现。
    torch.manual_seed(FIX_SEED)
    from neuroplex.loader import assemble_cortex

    cortex_obj, _tok, _extra = assemble_cortex()
    call = dict(kwargs)
    call.setdefault("auto_capture", False)
    call.setdefault("auto_memory", False)
    return cortex_obj.generate(prompt, **call)


def test_generate_reproduces_golden() -> None:
    """主契约：迁移后 `Cortex.generate()` 逐组等于迁移前黄金（证明 decode_step 抽离逐位等价）。"""
    for row in GOLDEN["rows"]:
        assert not row["is_error"], f"黄金用例本身为错误串：{row['case']} / {row['prompt']!r}"
        assert row["deterministic"], f"黄金用例非确定，不可作基准：{row['case']} / {row['prompt']!r}"
        got = _reproduce(row["prompt"], CASE_KWARGS[row["case"]])
        assert got == row["out"], (
            f"case={row['case']} prompt={row['prompt']!r}\n"
            f"  期望={row['out']!r}\n  实际={got!r}"
        )


def test_decode_step_repetition_penalty() -> None:
    """重复惩罚：已生成 token 的 logit 原地除 penalty（>0 时）。"""
    torch.manual_seed(0)
    logits = torch.zeros(1, 10)
    logits[0, 3] = 3.0
    decode_step(
        logits=logits,
        generated_token_ids={3},
        generated_token_list=[],
        generated_ids_ordered=[],
        repetition_penalty=1.4,
        no_repeat_ngram_size=0,
        eos_id=None,
        top_k=10,
    )
    assert logits[0, 3].item() == pytest.approx(3.0 / 1.4)


def test_decode_step_ngram_ban() -> None:
    """no-repeat-ngram：会出现在已生成 n-gram 末尾的 token，其 logit 原地置 -inf。"""
    torch.manual_seed(0)
    logits = torch.zeros(1, 16)
    logits[0, 7] = 10.0  # 本应被采到，但属 n-gram (7,7,7) 的封禁末尾
    decode_step(
        logits=logits,
        generated_token_ids=set(),  # 跳过重复惩罚分支，专测 ngram
        generated_token_list=[7, 7, 7],
        generated_ids_ordered=[],
        repetition_penalty=1.0,
        no_repeat_ngram_size=3,
        eos_id=None,
        top_k=10,
    )
    assert torch.isinf(logits[0, 7]) and logits[0, 7].item() < 0


def test_decode_step_entropy_stop_returns_none() -> None:
    """无 EOS 且软max 熵 > 8.0（且已生成 >=8 token）⇒ 返回 None（熵停止，调用方 break）。"""
    torch.manual_seed(0)
    logits = torch.zeros(1, 10000)  # 全 0 ⇒ 均匀 ⇒ 熵 = log(10000) ≈ 9.21 > 8
    out = decode_step(
        logits=logits,
        generated_token_ids=set(),
        generated_token_list=[],
        generated_ids_ordered=list(range(8)),  # len >= 8 满足早停门限
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        eos_id=None,
        top_k=10,
    )
    assert out is None
