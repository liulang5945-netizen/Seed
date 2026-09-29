"""A30 出厂污染门槛：判据、随包工件加载与两条门槛的纯函数。

owner 裁定（2026-09-29，PLAN-A-30 §7-1）：回路随出厂基座装的同时必须上两条门槛——
`learn=True` 只回写通过 ``well_formed ∧ 长度上限`` 的答复，且坏答复不进下一轮 prompt 历史。
本模块只放判据与纯函数；产品链的接线在 ``api/seed_runtime.py`` 的 ``chat()``。

**判据来源与逐位性**：``well_formed``／``mean_nll`` 与三个常数
（``MIN_LEN``／``MAX_SINGLE_SHARE``／``NLL_THRESHOLD``）是
``scripts/training/diag_taiji_r2_surface_decode.py`` 同名定义的**逐位拷贝**——
仪器是权威；产品侧拷贝由 tests 的逐位对照守卫钉住（漂移即红）。
无 n 元模型时 ``well_formed`` 沿用仪器的响亮拒绝（不得声称"成句"）。

**随包工件**：NLL 支路需要语料 n 元模型，语料原件 1.4 GB 不可随包；随包的是
200k 行模型的压缩工件（``build_taiji_a30_surface_ngram_artifact.py`` 产出，
+5.5 MB，owner 2026-09-29 同批批准）。工件是随包可信资产（与检查点信封同一
provenance 与分发渠道），因此用 lzma+pickle；不把它当不可信输入对待。
"""

from __future__ import annotations

import lzma
import pickle
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

#: 与仪器 diag_taiji_r2_surface_decode.py 同源的三个常数（漂移守卫钉住）。
MIN_LEN = 8
MAX_SINGLE_SHARE = 0.35
NLL_THRESHOLD = -7.8  # 每字符平均 n 元对数似然下限；由仪器 §controls 实测标定

#: 随包工件名（解析在检查点同一目录；打包链把它放进 dist 的 checkpoints/）。
SURFACE_NGRAM_ARTIFACT_NAME = "seed_surface_ngram.lzma"


def mean_nll(text: str, model: tuple) -> float:
    uni, bi, vocab, total = model
    s = "".join(text.split())
    if len(s) < 2:
        return 0.0
    import math as _m

    alpha = 1.0
    score = 0.0
    for a, b in zip(s, s[1:]):
        p = (bi.get((a, b), 0) + alpha) / (uni.get(a, 0) + alpha * (vocab + 1))
        score += _m.log(max(p, 1e-12))
    return score / (len(s) - 1)


def well_formed(text: str, model: tuple | None = None) -> bool:
    try:
        text.encode("utf-8").decode("utf-8")
    except UnicodeDecodeError:
        return False
    stripped = "".join(text.split())
    if len(stripped) < MIN_LEN:
        return False
    if Counter(stripped).most_common(1)[0][1] / len(stripped) > MAX_SINGLE_SHARE:
        return False
    if model is None:  # 没有 n 元模型时不得声称"成句"——那是被否证的旧口径
        raise RuntimeError("成句判据必须带语料 n 元模型；拒绝退回已被否证的四条件合取")
    return mean_nll(stripped, model) >= NLL_THRESHOLD


def load_surface_ngram(path: Path | str) -> tuple[dict, dict, int, int]:
    """载入随包 n 元工件，返回与仪器 ``build_ngram_model`` 同形的 (uni, bi, vocab, total)。"""

    payload = pickle.loads(lzma.decompress(Path(path).read_bytes()))
    return payload["uni"], payload["bi"], int(payload["vocab"]), int(payload["total"])


def filter_history(
    history: Sequence[tuple[str, str]] | None, model: tuple
) -> list[tuple[str, str]]:
    """门槛②：坏答复不进下一轮 prompt 历史。

    判定对象是**答复文本**（``entry[1]``），判据同回写门槛的 ``well_formed`` 支路——
    历史里的答复没有"是否吃满预算"可考（生成时的 budget 信息不在场），只按内容判。
    """

    if not history:
        return []
    kept: list[tuple[str, str]] = []
    for user, assistant in history:
        if user and assistant and well_formed(assistant, model):
            kept.append((user, assistant))
    return kept


def ended_naturally(raw: bytes, *, turn_markers: Sequence[str], budget: int) -> bool:
    """门槛①的"长度上限"支路：答复须在预算内自然收口。

    自然收口＝模型自己发出了轮界标记（下一轮的开头），或没吃满预算就停了
    （边界符胜出／任何其它停止条件）。被预算硬截＝未收口＝这条答复是"停不下来"
    的产物，不回写也不进历史。
    """

    if len(raw) < budget:
        return True
    return any(marker.encode("utf-8") in raw for marker in turn_markers)


def write_back_allowed(
    answer: str, raw: bytes, model: tuple, *, turn_markers: Sequence[str], budget: int
) -> tuple[bool, str]:
    """门槛①完整判定：``well_formed ∧ 长度上限``。返回 (是否放行, 原因码)。"""

    if not ended_naturally(raw, turn_markers=turn_markers, budget=budget):
        return False, "not_ended_naturally"
    if not well_formed(answer, model):
        return False, "not_well_formed"
    return True, "passed"


def find_artifact(checkpoint_path: Path | str | None) -> Path | None:
    """随包工件的解析规则：与检查点同一目录（打包链把它放进 dist 的 checkpoints/）。"""

    if checkpoint_path is None:
        return None
    candidate = Path(checkpoint_path).parent / SURFACE_NGRAM_ARTIFACT_NAME
    return candidate if candidate.is_file() else None


def build_artifact_blob(uni: Counter, bi: Counter, vocab: int, total: int) -> bytes:
    """把 (uni, bi, vocab, total) 压成随包工件字节（与 load_surface_ngram 互逆）。"""

    payload = {"uni": uni, "bi": bi, "vocab": vocab, "total": total}
    return lzma.compress(pickle.dumps(payload, protocol=5), preset=9)
