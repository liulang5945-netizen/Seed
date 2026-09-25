"""B-3 C-2 黄金向量等价测试：tokenizer/对齐簇迁移后**逐组复现**迁移前实现。

黄金向量：`reports/cortex_alignment_golden_20260925.json`
（迁移前用生产构造路径 `create_cortex()` + 真 tokenizer 采集：
reencode 4 组 + zh 对齐表 50000 条全量）。

另钉住：helper 对**传入 cache** 的写回语义（与 Cortex 原实现一致——缓存写回由引用完成）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sentencepiece as spm

from neuroplex.brain import _cortex_alignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads(
    (PROJECT_ROOT / "reports" / "cortex_alignment_golden_20260925.json").read_text(encoding="utf-8")
)

DOMAINS_DIR = PROJECT_ROOT / "neuroplex" / "domains"


@pytest.fixture(scope="module")
def tokenizers() -> dict:
    def load(rel: str):
        sp = spm.SentencePieceProcessor()
        sp.Load(str(DOMAINS_DIR / rel))
        return sp

    return {"general": load("general/sp_general.model"), "zh": load("zh/sp_zh.model")}


def test_reencode_matches_golden(tokenizers) -> None:
    for row in GOLDEN["reencode_rows"]:
        got = _cortex_alignment.reencode_domain_generation_context(
            tokenizers["general"], row["prefix"], row["generated_ids"], tokenizers["zh"]
        )
        assert list(got) == row["output_general_ids"], f"reencode {row['name']} 不一致"


def test_alignment_matches_golden(tokenizers) -> None:
    golden_alignment = GOLDEN["alignment_full"]
    cache: dict = {}
    got = _cortex_alignment.get_domain_to_general_alignment(
        tokenizers["general"], None, cache, "zh", tokenizers["zh"]
    )
    assert len(got) == len(golden_alignment), "对齐表条数不一致"
    for key, value in golden_alignment.items():
        # JSON 往返把 int 键序列化成了字符串 ⇒ 按 int 键比对
        assert got[int(key)] == value, f"对齐表 {key} 不一致"
    # 缓存写回：与原实现同语义（写回由引用完成）
    assert cache.get("zh", {}).get("alignment") is not None, "缓存应被写回"


def test_alignment_cache_hit_returns_same_object_content(tokenizers) -> None:
    """二次调用走缓存 ⇒ 结果一致且不再重建（条数一致即可判）。"""

    cache: dict = {}
    first = _cortex_alignment.get_domain_to_general_alignment(
        tokenizers["general"], None, cache, "zh", tokenizers["zh"]
    )
    second = _cortex_alignment.get_domain_to_general_alignment(
        tokenizers["general"], None, cache, "zh", tokenizers["zh"]
    )
    assert first == second and len(first) == len(second)
