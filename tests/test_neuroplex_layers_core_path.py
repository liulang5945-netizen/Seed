"""B-4 覆盖率第一刀：`neuroplex/layers.py`（审计 S3/S9/S11 所在的核心推理路径）。

这块此前**全仓零测试**（只有 `scripts/training/train_tinystories*.py` 引用过这些类），
而审计的三条架构级发现正落在这里：S3 生成主循环接 KV cache、S9 RoPE 位置一致性、
S11 attention sink + 滑动窗口。故断言全部钉**语义**而非形状：

* RoPE 的相对位置不变性（RoPE 的定义性质，写错相位必红）；
* 逐 token 增量解码 **等于** 整段前向的对应位置（S3 接入 KV cache 的正确性契约）；
* 驱逐模式 cache 长度封顶但**绝对位置继续增长**（S9 的修复点，退化成"按 cache 长度推位置"必红）；
* sink 段在多次驱逐后**逐位不变**（锚点不被重旋）。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuroplex.layers import (
    GroupedQueryAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
    apply_rotary_emb,
)

torch.manual_seed(11)

HIDDEN, HEADS, KV_HEADS = 32, 4, 2  # head_dim = 8，GQA 每组 2 个 query 共享 1 组 KV
HEAD_DIM = HIDDEN // HEADS


@pytest.fixture(autouse=True)
def _inference_only():
    """本件测的是推理语义（KV cache / 驱逐 / 相位），不需要 autograd。"""

    with torch.no_grad():
        yield


def _causal_mask(length: int) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"))
    return mask.triu(1).unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]


def _attn(sink: int = 0, window: int = 0) -> GroupedQueryAttention:
    layer = GroupedQueryAttention(
        HIDDEN,
        HEADS,
        KV_HEADS,
        dropout=0.0,
        attention_sink_size=sink,
        sliding_window_size=window,
    )
    layer.eval()
    return layer


def _x(batch: int, length: int) -> torch.Tensor:
    return torch.randn(batch, length, HIDDEN)


# ── RMSNorm / SwiGLU ──────────────────────────────────────────────────────────


def test_rmsnorm_gives_unit_rms_and_scales_with_weight() -> None:
    norm = RMSNorm(HIDDEN)
    out = norm(torch.randn(2, 5, HIDDEN) * 7.3)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), rms

    weight2 = RMSNorm(HIDDEN)
    weight2.weight = nn.Parameter(torch.full((HIDDEN,), 2.0))
    raw = torch.randn(1, 3, HIDDEN)
    scaled = weight2(raw)
    assert torch.allclose(scaled, norm(raw) * 2.0, atol=1e-5), "weight 必须逐维线性作用于归一结果"
    assert not torch.allclose(scaled, norm(raw), atol=1e-5), "weight 没生效的写法必须被这条抓到"


def test_swiglu_gain_scales_output_only() -> None:
    ffn = SwiGLU(HIDDEN, HIDDEN * 2)
    ffn.eval()
    raw = _x(1, 4)
    base = ffn(raw)
    assert torch.allclose(ffn(raw, gain=2.0), base * 2.0, atol=1e-6)
    assert ffn(raw).shape == raw.shape


# ── RoPE ──────────────────────────────────────────────────────────────────────


def test_rope_dot_product_depends_only_on_relative_position() -> None:
    """RoPE 的定义性质：⟨R_i q, R_j k⟩ 只与 i−j 有关。相位算错必红。"""

    rope = RotaryEmbedding(HEAD_DIM)
    q = torch.randn(1, 1, 1, HEAD_DIM)
    k = torch.randn(1, 1, 1, HEAD_DIM)
    sin, cos = rope(torch.zeros(1, 9, 1, HEAD_DIM), 9)

    def rot(vec: torch.Tensor, pos: int) -> torch.Tensor:
        return apply_rotary_emb(vec, vec, sin[pos : pos + 1], cos[pos : pos + 1])[0]

    def dot(i: int, j: int) -> float:
        return float((rot(q, i)[0, 0, 0] * rot(k, j)[0, 0, 0]).sum())

    assert abs(dot(0, 3) - dot(5, 8)) < 1e-4, "平移 i,j 同量后点积必须不变"
    assert abs(dot(0, 3) - dot(0, 4)) > 1e-4, "i−j 变了点积必须变（否则相位是常数）"


def test_rope_position_zero_is_identity_and_cache_is_bucketed_prefix() -> None:
    rope = RotaryEmbedding(HEAD_DIM)
    probe = torch.zeros(1, 300, 1, HEAD_DIM)
    sin, _cos = rope(probe, 300)
    assert torch.allclose(sin[0], torch.zeros(HEAD_DIM // 2), atol=1e-6), "位置 0 必须不旋转"

    q = torch.randn(1, 1, 1, HEAD_DIM)
    rotated = apply_rotary_emb(q, q, sin[:1], _cos[:1])[0]
    assert torch.allclose(rotated, q, atol=1e-6)

    # 桶化缓存：同桶内的短序列必须拿到与前缀切片逐位相同的结果
    sin_small, cos_small = rope(probe, 200)
    assert torch.equal(sin_small, sin[:200]) and torch.equal(cos_small, _cos[:200])
    # LRU 上限 4（原实现注释声称的防内存泄漏约束）
    for length in (128, 256, 512, 1024, 2048, 4096):
        rope(probe, length)
    assert len(rope._cache) <= 4, sorted(rope._cache)


# ── GroupedQueryAttention ─────────────────────────────────────────────────────


def test_gqa_causal_mask_blocks_future_tokens() -> None:
    attn = _attn()
    length = 6
    x = _x(1, length)
    weights = attn(x, mask=_causal_mask(length), return_attn_weights=True)[2]
    assert weights is not None and weights.shape == (1, HEADS, length, length)
    assert torch.allclose(weights.sum(-1), torch.ones(1, HEADS, length), atol=1e-5), "每行必须归一"
    upper = weights.triu(diagonal=1)
    assert float(upper.abs().max()) < 1e-6, "上三角（未来 token）权重必须为 0"


def test_gqa_incremental_decode_matches_full_forward() -> None:
    """S3 契约：逐 token + KV cache 的结果 == 整段前向的对应位置。"""

    attn = _attn()  # 无驱逐 ⇒ 位置由 cache 长度推断即正确
    length = 7
    x = _x(1, length)
    full = attn(x, mask=_causal_mask(length))[0]

    cache = None
    last = None
    for step in range(length):
        last, cache = attn(x[:, step : step + 1], kv_cache=cache, use_cache=True)[:2]
        assert cache is not None and len(cache) == 2
        assert cache[0].shape[1] == step + 1
    assert torch.allclose(full[:, -1], last[:, -1], atol=2e-5), float(
        (full[:, -1] - last[:, -1]).abs().max()
    )


def test_gqa_eviction_caps_length_but_keeps_absolute_position_growing() -> None:
    """S11 + S9：长度封顶 = sink+window，绝对位置继续按已见 token 数走。"""

    sink, window = 2, 3
    attn = _attn(sink=sink, window=window)
    assert attn.kv_cache_max_len == sink + window

    x = _x(1, 10)
    cache = None
    for step in range(10):
        _out, cache = attn(x[:, step : step + 1], kv_cache=cache, use_cache=True)[:2]
        assert len(cache) == 3, "驱逐模式必须返回 3 元组携带绝对长度"
        assert cache[0].shape[1] <= sink + window
        assert cache[2] == step + 1, "绝对位置不能塌回 cache 长度（S9 的错法）"


def test_gqa_eviction_preserves_sink_and_slides_window() -> None:
    """驱逐语义：sink 锚点逐位不变，window 每次丢最旧、追加最新。

    ⚠️ 顺带钉住当前契约：驱逐只发生在**增量路径**（`kv_cache is not None`），
    整段前向 + use_cache 不触发驱逐（cache 长度 = 序列长度）。改这条契约的人会被本条提醒。
    """

    sink, window = 2, 3
    attn = _attn(sink=sink, window=window)
    max_len = sink + window

    x = _x(1, 10)
    full_cache = attn(x, use_cache=True)[1]
    assert full_cache[0].shape[1] == 10, "整段前向不做驱逐（当前契约）"

    cache = None
    prev = None
    evicted_from = None
    for step in range(10):
        _out, cache = attn(x[:, step : step + 1], kv_cache=cache, use_cache=True)[:2]
        cur_len = cache[0].shape[1]
        assert cur_len <= max_len, (step, cur_len)
        if prev is not None:
            # sink 锚点必须逐位不变（被重旋/重写即实现有副作用）；前几步 cache 还不足 sink 长度
            anchor = min(sink, prev[0].shape[1])
            assert torch.equal(
                cache[0][:, :anchor], prev[0][:, :anchor]
            ), f"step {step} sink 被改写"
        if cur_len == max_len and prev is not None and prev[0].shape[1] == max_len:
            # 已处于驱逐态：window 段丢最旧 ⇒ 本步 window 前段 = 上步 window 后段
            assert torch.equal(
                cache[0][:, sink:-1], prev[0][:, sink + 1 :]
            ), f"step {step} window 未滑动"
            evicted_from = step if evicted_from is None else evicted_from
        assert cache[2] == step + 1
        prev = cache
    assert (
        evicted_from is not None and evicted_from <= max_len
    ), "10 步里从没触发驱逐 ⇒ 上限形同虚设"


def test_evicted_cache_must_carry_absolute_length() -> None:
    """S9 否证：驱逐后按 **cache 长度** 推位置（旧 2 元组口径）会算错相位 ⇒ 读数必须不同。

    现实现用 3 元组携带绝对长度；把每步返回的 cache 降级成 2 元组即复现旧口径。
    两条口径共用**同一个 attn 实例**（否则比的是随机权重），差异只应来自相位。
    """

    sink, window = 2, 3
    x = _x(1, 9)
    attn = _attn(sink=sink, window=window)

    def decode(strip_abs: bool):
        cache = None
        out = None
        for step in range(9):
            out, cache = attn(x[:, step : step + 1], kv_cache=cache, use_cache=True)[:2]
            if strip_abs and len(cache) == 3:
                cache = (cache[0], cache[1])  # 丢绝对长度 ⇒ start_pos 回退成 cache 长度
        return out, cache

    correct, c_correct = decode(False)
    legacy, c_legacy = decode(True)
    assert len(c_correct) == 3 and len(c_legacy) == 2
    assert c_correct[2] == 9 and c_correct[0].shape[1] == sink + window
    assert c_legacy[0].shape[1] == sink + window, "两条口径 cache 长度必须一致，差异只在相位"
    assert c_legacy[0].shape[1] < c_correct[2], "没驱逐就没相位问题 ⇒ 本条退化成空测"
    gap = float((correct - legacy).abs().max())
    assert gap > 1e-2, f"相位差异仅 {gap} ⇒ 本条未测到 S9 的错法"


def test_gqa_temp_gain_sharpens_attention() -> None:
    """S9 神经调质门控：temp_gain>1 让分布更尖（等价抬高 logits 温度）。"""

    attn = _attn()
    x = _x(1, 5)
    mask = _causal_mask(5)
    base = attn(x, mask=mask, return_attn_weights=True)[2]
    sharp = attn(x, mask=mask, return_attn_weights=True, temp_gain=6.0)[2]
    ent_base = -(base.clamp_min(1e-9) * base.clamp_min(1e-9).log()).sum(-1).mean()
    ent_sharp = -(sharp.clamp_min(1e-9) * sharp.clamp_min(1e-9).log()).sum(-1).mean()
    assert ent_sharp < ent_base, (float(ent_base), float(ent_sharp))
    flat = attn(x, mask=mask, return_attn_weights=True, temp_gain=0.05)[2]
    ent_flat = -(flat.clamp_min(1e-9) * flat.clamp_min(1e-9).log()).sum(-1).mean()
    assert ent_flat > ent_base, "temp_gain<1 必须让分布更平"


# ── TransformerBlock ──────────────────────────────────────────────────────────


def test_transformer_block_standard_and_dendritic_paths() -> None:
    block = TransformerBlock(HIDDEN, HEADS, KV_HEADS, HIDDEN * 2)
    block.eval()
    x = _x(2, 5)
    out = block(x, mask=_causal_mask(5))[0]
    assert out.shape == x.shape and torch.isfinite(out).all()

    dend = TransformerBlock(
        HIDDEN, HEADS, KV_HEADS, HIDDEN * 2, dendritic=True, apical_kv_dim=HIDDEN
    )
    dend.eval()
    field = torch.randn(2, 3, HIDDEN)
    d_out, d_cache, _ = dend(x, mask=_causal_mask(5), field_state=field)
    assert d_out.shape == x.shape and torch.isfinite(d_out).all()
    assert d_cache is None
    # apical 路径必须真的改变结果（field_state 在场 vs 缺席）
    without_field = dend(x, mask=_causal_mask(5), field_state=None)[0]
    assert not torch.allclose(d_out, without_field, atol=1e-6), "树突化 apical 路径未生效"


def test_transformer_block_gains_are_not_noops() -> None:
    block = TransformerBlock(HIDDEN, HEADS, KV_HEADS, HIDDEN * 2)
    block.eval()
    x = _x(1, 4)
    base = block(x, mask=_causal_mask(4))[0]
    hotter = block(x, mask=_causal_mask(4), temp_gain=4.0)[0]
    boosted = block(x, mask=_causal_mask(4), ffn_gain=1.5)[0]
    assert not torch.allclose(base, hotter, atol=1e-6)
    assert not torch.allclose(base, boosted, atol=1e-6)


def test_use_cache_off_returns_no_cache() -> None:
    attn = _attn(sink=1, window=2)
    out, cache, weights = attn(_x(1, 3), use_cache=False)
    assert cache is None and weights is None and out.shape == (1, 3, HIDDEN)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_gqa_head_grouping_shapes(kv_heads: int) -> None:
    attn = GroupedQueryAttention(HIDDEN, HEADS, kv_heads).eval()
    x = _x(1, 4)
    out = attn(x, mask=_causal_mask(4))[0]
    assert out.shape == x.shape
