"""B-3 C-1 等价守卫：`_select_best_candidate` / `to_ngrams` / `nll_quality_from_round1_logits`
抽离后，**委托与 helper 在同一输入下逐位相同**（活体对照：同一模型实例、同一 think 结果）。

`nll_quality_from_round1_logits` 的等价需要真实的 round1_logits ⇒ 用基座对固定 prompt
跑一次 `think()`（模块级 fixture，一次加载），委托与 helper 各算一遍比对。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuroplex.brain import _cortex_quality
from neuroplex.brain.cortex import Cortex

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_SETS = [
    [],
    ["只有一个候选"],
    ["短", "也很短"],
    ["这是一条中等长度的中文回答，包含一些具体的内容与细节描述。", "另一条回答，长度略短。", "第三条候选回答，长度中等且与第二条有部分重叠的词。"],
    ["重复重复重复重复重复重复", "正常长度且有信息量的回答文本，不重复。"],
]


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


@pytest.fixture(scope="module")
def think_result() -> tuple[object, object, dict]:
    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(PROJECT_ROOT / "checkpoints" / "seed_beta.pt")
    taiji = runtime.model.substrate
    taiji.reset_dynamics(episode_id="c1-equiv")
    prompt = "我叫阿岩。我的名字是什么？"
    # think 需要shared embedding（与 _generate_p7 同款接法，2026-09-25 逐行对照）
    ids = taiji._general_sp.encode(prompt)
    ids_tensor = torch.tensor([ids], dtype=torch.long, device=taiji.device)
    result = None
    if getattr(taiji, "_neuron_shared_embeddings", None):
        neuron_embeddings = {
            nid: emb(ids_tensor) for nid, emb in taiji._neuron_shared_embeddings.items()
        }
        result = taiji.think(
            active_nids=None, fusion_mode="soft", neuron_embeddings=neuron_embeddings
        )
    else:
        shared_emb = taiji._shared_embedding(ids_tensor)
        result = taiji.think(shared_emb, active_nids=None, fusion_mode="soft")
    assert result is not None and (result.round1_logits or {}), "think 结果缺 round1_logits"
    return taiji, prompt, result


def test_nll_quality_delegate_matches_helper(think_result) -> None:
    taiji, prompt, result = think_result
    ours = Cortex._nll_quality_from_round1_logits(taiji, result, prompt, "zh")
    theirs = _cortex_quality.nll_quality_from_round1_logits(
        taiji._tokenizer_hub, taiji._general_sp, taiji.device, result, prompt, "zh"
    )
    assert ours == theirs, "委托与 helper 的 NLL 质量读数不一致"
    assert ours, "NLL 质量为空 ⇒ 夹具无效（应能取到非空质量）"
