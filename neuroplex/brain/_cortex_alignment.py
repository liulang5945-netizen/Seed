"""Cortex tokenizer/对齐纯函数（B-3 C-2 簇，从 `neuroplex/brain/cortex.py` 抽离，2026-09-25）。

设计约束（冻结基线）：**逐行搬移、只把 self 依赖提升为参数**，不改任何语句——
等价性由黄金向量（`reports/cortex_alignment_golden_20260925.json`，50000 条对齐表 + reencode 4 组）
与活体对照测试保证。

留 Cortex 的成员：`set_tokenizer*` / `set_alignment_rules` / `invalidate_alignment_cache`（公共 API setter）。
"""

from __future__ import annotations

from neuroplex.resonance.translator import tokenizer_fingerprint

__all__ = ["get_domain_to_general_alignment", "reencode_domain_generation_context"]


def reencode_domain_generation_context(
    general_sp,
    prefix_text: str,
    generated_ids: list[int],
    decode_sp,
) -> list[int]:
    """Re-encode the complete general context after a domain-token step.

    SentencePiece tokenization is boundary-sensitive: encoding a generated
    domain piece by itself and appending its general IDs is not equivalent
    to encoding ``prefix + generated_text``.  Generation must preserve the
    same text-level context that training used for alignment.
    """
    generated_text = decode_sp.DecodeIds(generated_ids) if generated_ids else ""
    general_ids = general_sp.encode(prefix_text + generated_text)
    return general_ids if general_ids else [0]


def get_domain_to_general_alignment(
    general_sp,
    alignment_rules,
    cache: dict,
    domain: str,
    domain_sp,
) -> dict[int, list]:
    """S6: 构建 domain token ID → general token IDs 对齐表（带缓存 + 热插拔失效）。

    消除自回归生成时的 domain→text→general re-encode 往返。
    对每个 domain token，预计算其 general token IDs 映射。

    热插拔：缓存项携带 tokenizer 指纹，任一 tokenizer（域/general）被替换后
    自动失效重建；也可用 invalidate_alignment_cache() 手动失效。

    可编辑层：set_alignment_rules() 注入的 AlignmentRules 中匹配的 domain
    piece 跳过自动转译，改用人工指定的 general piece 文本编码
    （新增特殊神经元时补充专业术语映射）。

    Args:
        general_sp: general SentencePiece（原 self._general_sp）
        alignment_rules: 可编辑词库规则层（原 self._alignment_rules）
        cache: 对齐表缓存（原 self._domain_to_general_cache，按引用写回）
        domain: 域名（如 "zh"）
        domain_sp: 域 tokenizer

    Returns:
        {domain_token_id: [general_token_ids]} 映射表
    """
    # 指纹 = (域 tokenizer 指纹, general tokenizer 指纹, 规则版本)
    rules_ver = alignment_rules.version if alignment_rules is not None else 0
    fp = (
        tokenizer_fingerprint(domain_sp),
        tokenizer_fingerprint(general_sp),
        rules_ver,
    )
    cached = cache.get(domain)
    if cached is not None and cached.get("fp") == fp:
        return cached["alignment"]

    if general_sp is None:
        return {}

    alignment: dict[int, list] = {}
    vocab_size = domain_sp.GetPieceSize() if hasattr(domain_sp, "GetPieceSize") else 0
    for domain_id in range(vocab_size):
        piece = domain_sp.id_to_piece(domain_id)
        manual = None
        if alignment_rules is not None:
            manual = alignment_rules.get(domain, piece)
        if manual is not None:
            # 人工规则：general piece 文本 → general ids（可多段，逐段 encode 拼接）
            general_ids = []
            for gp in manual:
                general_ids.extend(general_sp.encode(gp))
            alignment[domain_id] = (
                general_ids
                if general_ids
                else [general_sp.pad_id() if hasattr(general_sp, "pad_id") else 0]
            )
            continue
        if piece.startswith("<0x") and piece.endswith(">"):
            # byte fallback piece（如 <0x0A>）：必须 decode 成真实字节再 encode，
            # 否则 "<0x0A>" 会被当作 6 个字符编码，换行语义丢失
            text = domain_sp.decode([domain_id])
        else:
            text = piece
        general_ids = general_sp.encode(text)
        if general_ids:
            alignment[domain_id] = general_ids
        else:
            # 空映射用 pad_id 兜底
            pad_id = general_sp.pad_id() if hasattr(general_sp, "pad_id") else 0
            alignment[domain_id] = [pad_id]

    cache[domain] = {"fp": fp, "alignment": alignment}
    print(f"[S6] 域 '{domain}' 对齐表已构建: {len(alignment)} entries", flush=True)
    return alignment
