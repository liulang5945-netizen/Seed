"""B-3 C-1 等价守卫：质量簇成员抽离后，**委托 / helper / 迁移前黄金** 三者同值。

夹具：`scripts/training/capture_taiji_cortex_rolling_nll_golden.py` 的 `build_cortex()`
= 生产构造路径 `neuroplex.loader.create_cortex()` + 自装 TokenizerHub/general SP。
（B-2 曾以"taiji 适配器没有 `_general_sp`/`_tokenizer_hub`"为由把 nll 等价挂起为 skip；
现改按 neuroplex Cortex 取真夹具 ⇒ skip 结清，且夹具秒级、不需加载基座。）

黄金件：`reports/cortex_rolling_nll_golden_20260926.json`（迁移前的 `Cortex._rolling_nll_quality` 读数）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neuroplex.brain import _cortex_quality
from neuroplex.brain.cortex import Cortex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads(
    (PROJECT_ROOT / "reports" / "cortex_rolling_nll_golden_20260926.json").read_text(
        encoding="utf-8"
    )
)


def _load_capture_module():
    """按路径加载采集脚本，与守卫共用夹具构造代码（两处各写一遍必漂）。"""

    path = PROJECT_ROOT / "scripts" / "training" / "capture_taiji_cortex_rolling_nll_golden.py"
    spec = importlib.util.spec_from_file_location("c1_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = _load_capture_module()

CANDIDATE_SETS = [
    [],
    ["只有一个候选"],
    ["短", "也很短"],
    [
        "这是一条中等长度的中文回答，包含一些具体的内容与细节描述。",
        "另一条回答，长度略短。",
        "第三条候选回答，长度中等且与第二条有部分重叠的词。",
    ],
    ["重复重复重复重复重复重复", "正常长度且有信息量的回答文本，不重复。"],
]


@pytest.fixture(scope="module")
def cortex_sp():
    cortex, hub, general_sp, zh_sp = capture.build_cortex()
    return cortex, hub, general_sp, zh_sp


def test_select_best_candidate_delegate_matches_helper() -> None:
    for candidates in CANDIDATE_SETS:
        probe = Cortex.__new__(Cortex)  # 该成员不依赖任何 self 状态
        assert probe._select_best_candidate(candidates) == (
            _cortex_quality.select_best_candidate(candidates)
        ), repr(candidates)


def test_to_ngrams_hoisted_matches_original_semantics() -> None:
    """内嵌闭包上提后语义不变：<n 返回 token 集合，>=n 返回 4-gram 元组集合。"""

    assert _cortex_quality.to_ngrams("a b", 4) == {"a", "b"}
    assert _cortex_quality.to_ngrams("a b c d e", 4) == {
        ("a", "b", "c", "d"),
        ("b", "c", "d", "e"),
    }


def test_nll_quality_delegate_matches_helper(cortex_sp) -> None:
    """nll 质量：委托与 helper 在同一输入下逐位相同，且读数非空（夹具有效性）。"""

    cortex, hub, general_sp, zh_sp = cortex_sp
    result = capture.synth_logits(zh_sp)
    prompt = "我叫阿岩。我的名字是什么？"
    ours = cortex._nll_quality_from_round1_logits(result, prompt, "zh")
    theirs = _cortex_quality.nll_quality_from_round1_logits(
        hub, general_sp, cortex.device, result, prompt, "zh"
    )
    assert ours == theirs, "委托与 helper 的 NLL 质量读数不一致"
    assert ours, "NLL 质量为空 ⇒ 夹具无效（应能取到非空质量）"
    # 词表不符的那支必须被跳过，不能混进质量集
    assert set(ours) == {"zh_unit_a", "zh_unit_b", "zh_unit_short"}, sorted(ours)
    # hub 缺失 ⇒ 早退 {}（迁移面包含这条守卫）
    bare = Cortex.__new__(Cortex)
    bare._tokenizer_hub = None
    bare._general_sp = general_sp
    bare.device = torch.device("cpu")
    assert bare._nll_quality_from_round1_logits(result, prompt, "zh") == {}


def test_rolling_nll_reproduces_pre_migration_golden(cortex_sp) -> None:
    """滚动后验：委托与 helper 都逐值复现**迁移前**黄金（8 个参数格 + 2 条早退分支）。

    window 三格（1/4/8）读数互不相同 ⇒ 该参数真参与取位，不是恒等式；
    `length_bound_short_neuron` 钉 `min(lens)-1` 的长度上限；词表不符支不参与。
    """

    cortex, _hub, _general_sp, zh_sp = cortex_sp
    golden_grid = GOLDEN["grid"]
    cases = [k for k, v in golden_grid.items() if "gen_text" in v]
    assert len(cases) == 8, f"黄金应含 8 个可复放参数格，实得 {len(cases)}"
    for name in cases:
        case = golden_grid[name]
        result = capture.synth_logits(zh_sp, include_short=case["include_short"])
        delegate = cortex._rolling_nll_quality(
            result, case["gen_text"], case["domain"], case["window"]
        )
        helper = _cortex_quality.rolling_nll_quality(
            cortex._tokenizer_hub,
            cortex.device,
            result,
            case["gen_text"],
            case["domain"],
            case["window"],
        )
        expected = case["out"]
        assert delegate == expected, f"{name}: 委托 {delegate} != 迁移前 {expected}"
        assert helper == expected, f"{name}: helper {helper} != 迁移前 {expected}"
    # 迁移前后 window 必须真的改变读数（否则上面 8 格等价是空洞的）
    reads = [golden_grid[c]["out"]["zh_unit_a"] for c in ("zh_window1", "zh_window4", "zh_window8")]
    assert len(set(reads)) == 3, f"window 未参与读数（{reads}）⇒ 该等价测试无分辨力"
    # 两条早退分支
    assert cortex._rolling_nll_quality({"round1_logits": {}}, "阿岩。", "zh", 4) == (
        golden_grid["empty_round1"]["out"]
    )
    bare = Cortex.__new__(Cortex)
    bare._tokenizer_hub = None
    bare.device = torch.device("cpu")
    assert (
        Cortex._rolling_nll_quality(bare, capture.synth_logits(zh_sp), "阿岩。", "zh", 4)
        == golden_grid["hub_missing"]["out"]
    )


def test_rolling_nll_helper_is_not_a_stub(cortex_sp) -> None:
    """反向锚：helper 在读一片拟合更好的 logits 时必须给出更高的质量分（防"恒返回 {}"式抽离）。"""

    cortex, hub, _general_sp, zh_sp = cortex_sp
    vocab = int(zh_sp.GetPieceSize())
    gen_text = "阿岩。我的名字是阿岩。"
    ids = zh_sp.encode(gen_text)
    n = 4
    length = len(ids) + 4
    flat = torch.zeros(1, length, vocab)
    # 滚动窗口读的是 **尾部** 位置：pos = length-n-1+k 预测 gen_text 倒数第 n-k 个 token
    for k in range(n):
        target = ids[len(ids) - n + k]
        if target in (0, 1):  # 该 token 会被 mask 掉，换一个非停用位
            target = ids[len(ids) - n + k - 4]
        flat[0, length - n - 1 + k, target] = 30.0
    result = {"round1_logits": {"perfect_neuron": flat}}
    score = _cortex_quality.rolling_nll_quality(hub, cortex.device, result, gen_text, "zh", n)
    # 读数是 -NLL ⇒ 上界为 0（完美拟合趋近 0，不是正数）
    assert score and score["perfect_neuron"] > -0.5, score
    # 同长度的一片零 logits（均匀分布）必须显著更差 ⇒ 读数确实在拟合并随拟合变号
    uniform = _cortex_quality.rolling_nll_quality(
        hub,
        cortex.device,
        {"round1_logits": {"flat_neuron": torch.zeros(1, length, vocab)}},
        gen_text,
        "zh",
        n,
    )
    assert score["perfect_neuron"] - uniform["flat_neuron"] > 5.0, (score, uniform)
