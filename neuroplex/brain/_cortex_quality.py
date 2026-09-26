"""Cortex 质量/评分纯函数（B-3 C-1 簇，从 `neuroplex/brain/cortex.py` 抽离，2026-09-25）。

设计约束（冻结基线）：**逐行搬移、只把 self 依赖提升为参数**，不改任何语句——
等价性由 `tests/test_cortex_quality_extraction.py` 的活体对照保证
（委托与 helper 在同一输入下逐位相同）。

成员：
* `nll_quality_from_round1_logits`（self 依赖：`_tokenizer_hub`/`_general_sp`/`device` ⇒ 参数）
* `rolling_nll_quality`（self 依赖：`_tokenizer_hub`/`device` ⇒ 参数；C27 增量一的滚动后验）
* `to_ngrams` / `select_best_candidate`（原为 `_select_best_candidate` 内的局部闭包+方法体，**无 self 依赖**）

留 Cortex 的成员（状态/编排耦合，见 PLAN-B-03 §5 停止线）：
`_probe_inactive_fused`（依赖 `self.ensemble.forward` + `self._shared_embedding`）、
`_capture_field_memory`（读 `self.get_last_field_state()` 并写全局 SleepEngine 单例）。
"""

from __future__ import annotations

import torch

from neuroplex.resonance.translator import build_position_alignment

__all__ = [
    "nll_quality_from_round1_logits",
    "rolling_nll_quality",
    "select_best_candidate",
    "to_ngrams",
]


def to_ngrams(text: str, n: int = 4) -> set:
    tokens = text.split()
    if len(tokens) < n:
        return set(tokens)
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def select_best_candidate(candidates: list[str]) -> str:
    """SMCS EPE 混合后验评分选最优候选。

    评分维度：
    1. Intra-response 置信度：候选长度（太短=低置信，太长=可能跑偏）
    2. Inter-response 一致性：与其他候选的 n-gram 重叠度（高一致=多采样收敛）
    3. 重复率惩罚：单候选内部 token 重复率（越低越好）

    综合分 = 一致性 + 长度置信 - 重复率
    """
    if not candidates:
        return ""
    n = len(candidates)
    if n == 1:
        return candidates[0]

    # 1. 计算 4-gram 集合（用于 inter-response 一致性）
    ngram_sets = [to_ngrams(c) for c in candidates]

    scores = []
    for i, text in enumerate(candidates):
        # Intra: 长度置信度（对数尺度，中等长度最优）
        length = len(text.split())
        if length == 0:
            scores.append(-1e9)
            continue
        length_score = -abs((length - 30) / max(length, 1)) * 0.3

        # Inter: 与其他候选的平均 n-gram 重叠
        if ngram_sets[i] and n > 1:
            overlaps = []
            for j in range(n):
                if j != i and ngram_sets[j]:
                    overlap = len(ngram_sets[i] & ngram_sets[j]) / max(
                        len(ngram_sets[i] | ngram_sets[j]), 1
                    )
                    overlaps.append(overlap)
            inter_score = sum(overlaps) / max(len(overlaps), 1)
        else:
            inter_score = 0.0

        # 重复率：单候选内部重复 token 比例
        tokens = text.split()
        unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
        repeat_penalty = (1 - unique_ratio) * 0.5

        total = inter_score + length_score - repeat_penalty
        scores.append(total)

    best_idx = scores.index(max(scores))
    return candidates[best_idx]


def nll_quality_from_round1_logits(
    tokenizer_hub: object,
    general_sp: object,
    device: torch.device,
    result: dict,
    prompt: str,
    domain: str,
) -> dict:
    """从 round1 logits 计算各 neuron 对 prompt 的 next-token NLL 质量。

    continuous leader 融合信号（C25-E 遗留）：质量 = 该 neuron 对 prompt
    的拟合度（域头在 general→domain 位置对齐空间的 next-token NLL，越低
    越贴合该域训练分布）。用 round1 独立 logits（leader 生成同源），零额外
    前向。返回 {nid: -NLL}（越大质量越好）；失败返回 {}（调用方回退）。

    Args:
        tokenizer_hub: 域/general tokenizer 的 hub（原 self._tokenizer_hub）
        general_sp: general SentencePiece（原 self._general_sp）
        device: 计算设备（原 self.device）
        result: think() 返回（含 round1_logits: {nid: [1, L, V]}）
        prompt: 生成输入（质量信号针对初始 prompt，不随生成增长）
        domain: 域（用于取对应 tokenizer；zh 50K 与 dialogue lm_head 对齐）
    """
    r1_logits = result.get("round1_logits") or {}
    if not r1_logits:
        return {}
    hub = tokenizer_hub
    if hub is None or not hasattr(hub, "get_tokenizer"):
        return {}
    try:
        tok = hub.get_tokenizer(domain) or hub.get_tokenizer("general")
        if tok is None or general_sp is None:
            return {}
        # 生成前向的序列位置来自 general tokenizer；域头目标来自 domain
        # tokenizer。两者词元数通常不同，必须按字符 span 对齐后再计算
        # next-token NLL，不能直接把两套 tokenizer 的 id 按位置硬配。
        _, aligned_targets = build_position_alignment(prompt, tok, general_sp)
        aligned_targets = aligned_targets.to(device)
    except Exception:
        return {}
    if aligned_targets.numel() < 2:
        return {}
    vocab = int(tok.GetPieceSize()) if hasattr(tok, "GetPieceSize") else None
    if not vocab:
        return {}
    out: dict = {}
    for nid, lg in r1_logits.items():
        if lg.shape[-1] != vocab:
            continue
        try:
            lg = lg.detach()  # 推理质量信号：仅前向，不携带梯度
            n = min(lg.shape[1] - 1, aligned_targets.numel() - 1)
            if n < 1:
                continue
            logp = torch.log_softmax(lg[:, :n, :], dim=-1)  # [1, n, V]
            tgt = aligned_targets[1 : n + 1]
            mask = (tgt >= 0) & (tgt != 1) & (tgt != 0)
            safe_tgt = tgt.clamp_min(0).unsqueeze(0).unsqueeze(-1)
            nll_tok = -logp.gather(-1, safe_tgt).squeeze(-1)  # [1, n]
            if mask.sum() == 0:
                continue
            out[nid] = -float((nll_tok * mask).sum() / mask.sum().float())
        except Exception:
            continue
    return out


def rolling_nll_quality(
    tokenizer_hub: object,
    device: torch.device,
    result: dict,
    gen_text: str,
    domain: str,
    window: int,
) -> dict:
    """滚动后验（C27 增量一）：对已生成文本窗口的 next-token NLL 质量。

    与 nll_quality_from_round1_logits（prompt 一次性，C25-E）不同：本函数
    取 round1_logits 尾部窗口（已生成文本区段），衡量各 neuron 对"最近
    生成内容"的续写拟合度——随生成演化，捕获实例内漂移。零额外前向
    （round1_logits 由生成主循环 think 产出）。返回 {nid: -NLL}；失败 {}。

    原 `Cortex._rolling_nll_quality`（B-3 C-1 第二刀，2026-09-26 逐行搬移）：
    `self._tokenizer_hub`/`self.device` 提升为参数，`self._rolling_nll_quality` 留委托。
    """

    r1 = result.get("round1_logits") or {}
    if not r1:
        return {}
    hub = tokenizer_hub
    if hub is None or not hasattr(hub, "get_tokenizer"):
        return {}
    try:
        tok = hub.get_tokenizer(domain) or hub.get_tokenizer("general")
        if tok is None:
            return {}
        zids = torch.tensor([tok.encode(gen_text)], dtype=torch.long, device=device)
    except Exception:
        return {}
    if zids.numel() < 1:
        return {}
    vocab = int(tok.GetPieceSize()) if hasattr(tok, "GetPieceSize") else None
    if not vocab:
        return {}
    lens = [int(lg.shape[1]) for lg in r1.values() if lg.shape[-1] == vocab]
    if not lens:
        return {}
    # 对齐（与 C25-E 同口径）：round1_logits 位置 t 预测上下文 t+1。
    # 取 logits 倒数 n+1 个位置中的前 n 个，target = 已生成文本最后 n 个
    # token（续写 NLL；软信号，尽力对齐即可，不追求逐 token 严格映射）。
    n = min(int(window), int(zids.numel()), min(lens) - 1)
    if n < 1:
        return {}
    tgt = zids[0][-n:].unsqueeze(0).unsqueeze(-1)  # [1, n, 1]
    out: dict = {}
    for nid, lg in r1.items():
        if lg.shape[-1] != vocab:
            continue
        try:
            lg_win = lg.detach()[:, -(n + 1) : -1, :]  # [1, n, V]
            logp = torch.log_softmax(lg_win, dim=-1)
            nll_tok = -logp.gather(-1, tgt).squeeze(-1)  # [1, n]
            mask = (tgt.squeeze(-1) != 1) & (tgt.squeeze(-1) != 0)
            if mask.sum() == 0:
                continue
            out[nid] = -float((nll_tok * mask).sum() / mask.sum().float())
        except Exception:
            continue
    return out
