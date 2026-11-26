"""Open-vocab UTF-8 codepoint-block pointer-generator subject (`taiji-open-chinese-subject-v5`).

Processes raw UTF-8 byte streams across multi-turn Chinese episodes with ZERO closed-word
IDs, ZERO dictionary lookups, and ZERO environment state leakage. Segments raw UTF-8 streams
deterministically via UTF-8 lead-byte length into 4-byte codepoint blocks (`[b0, b1, b2, b3]`),
combining:
1. Pure-byte codepoint-block embedding + within-turn positional embedding + 3-layer multi-scale
   dilated Conv1d local encoder (d=1, 2, 4);
2. Sandwiched cross-turn architecture (`SDPA1 -> Turn-Tracker -> SDPA2`) with `dropout=0.0`
   for fast CPU SDPA kernels and 100% bit-exact determinism;
3. Hierarchical turn-level episodic state tracker (4-head pooling + causal Turn-GRU + Turn-BiGRU
   + Turn-SDPA) broadcast back to tokens;
4. Bidirectional 13-dim pure-byte episodic & bigram anchor features;
5. OOV-invariant location-shifted hybrid pointer-generator decoder with vectorized span-exact
   copy supervision and strict `taiji.utf8_state` constraints;
6. Native persistence via `SessionStore` / `SessionRecord.capsule`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from taiji.open_chinese_task_env import INTENTS
from taiji.session_store import SessionRecord, SessionStore
from taiji.utf8_state import advance_utf8, trim_partial_tail, utf8_allowed

SCHEMA = "taiji-open-chinese-subject-v5"
PAD_BYTE = 256
BOS_BYTE = 257
EOS_BYTE = 258
BYTE_VOCAB = 259

SEP_BLOCK: tuple[int, int, int, int] = (0x0A, PAD_BYTE, PAD_BYTE, PAD_BYTE)
BOS_BLOCK: tuple[int, int, int, int] = (BOS_BYTE, PAD_BYTE, PAD_BYTE, PAD_BYTE)
EOS_BLOCK: tuple[int, int, int, int] = (EOS_BYTE, PAD_BYTE, PAD_BYTE, PAD_BYTE)
PAD_BLOCK: tuple[int, int, int, int] = (PAD_BYTE, PAD_BYTE, PAD_BYTE, PAD_BYTE)
MAX_DECODE_CHARS = 8
MAX_TURN_POS = 64


def bytes_to_utf8_blocks(raw_bytes: list[int]) -> list[list[int]]:
    """Segment raw UTF-8 bytes deterministically into 4-byte padded codepoint blocks."""
    blocks: list[list[int]] = []
    i = 0
    n = len(raw_bytes)
    while i < n:
        b = int(raw_bytes[i])
        if b < 0x80:
            sz = 1
        elif b < 0xE0:
            sz = 2
        elif b < 0xF0:
            sz = 3
        else:
            sz = 4
        blk = [int(raw_bytes[k]) for k in range(i, min(n, i + sz))]
        while len(blk) < 4:
            blk.append(PAD_BYTE)
        blocks.append(blk)
        i += sz
    return blocks


def _is_cjk_content_block(blk: tuple[int, int, int, int]) -> bool:
    """Identify standard 3-byte CJK ideograph UTF-8 blocks (lead 0xE4..0xE9, excluding punct)."""
    return 0xE4 <= blk[0] <= 0xE9 and blk[3] == PAD_BYTE


def _cjk_bigrams(
    blks: list[tuple[int, int, int, int]],
) -> set[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    """Extract consecutive CJK 2-char UTF-8 block bigrams from a segmented block list."""
    return {
        (blks[i], blks[i + 1])
        for i in range(len(blks) - 1)
        if _is_cjk_content_block(blks[i]) and _is_cjk_content_block(blks[i + 1])
    }


def build_turn_memory_tensors(
    turns_bytes: list[list[int]],
    *,
    use_history: bool = True,
) -> tuple[list[list[int]], list[list[float]]]:
    """Flatten observed turns into UTF-8 codepoint blocks and 13 pure-byte episodic features."""
    if not turns_bytes:
        return [list(SEP_BLOCK)], [[1.0, 0.0, 1.0, *([0.0] * 10)]]

    cur_idx = len(turns_bytes) - 1
    selected = turns_bytes if use_history else [turns_bytes[cur_idx]]
    base_idx = 0 if use_history else cur_idx

    turn_blks = [[tuple(b) for b in bytes_to_utf8_blocks(tb)] for tb in selected]
    t0_bg = _cjk_bigrams(turn_blks[0]) if (use_history and base_idx == 0) else set()
    t1_bg = (
        _cjk_bigrams(turn_blks[1])
        if (use_history and base_idx == 0 and len(turn_blks) > 1)
        else set()
    )

    cur_blocks = turn_blks[-1]
    cur_1 = {b for b in cur_blocks if _is_cjk_content_block(b)}
    cur_2 = _cjk_bigrams(cur_blocks)
    seen_earlier_1: set[tuple[int, int, int, int]] = set()
    seen_earlier_2: set[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = set()

    flat_blocks: list[list[int]] = []
    flat_feats: list[list[float]] = []

    for offset, t_blks in enumerate(turn_blks):
        tid = base_idx + offset
        is_cur = 1.0 if tid == cur_idx else 0.0
        recency = min(1.0, max(0.0, float(cur_idx - tid) / 6.0))
        turn_oh = [1.0 if min(tid, 5) == k else 0.0 for k in range(6)]
        n = len(t_blks)
        m1 = [0.0] * n
        m2 = [0.0] * n
        mt0 = [0.0] * n
        mt1 = [0.0] * n
        mrec = [0.0] * n

        if tid < cur_idx:
            for j, blk in enumerate(t_blks):
                if blk in cur_1:
                    m1[j] = 1.0
                if (
                    j + 1 < n
                    and _is_cjk_content_block(blk)
                    and _is_cjk_content_block(t_blks[j + 1])
                ):
                    bg = (blk, t_blks[j + 1])
                    if bg in cur_2:
                        m2[j] = m2[j + 1] = 1.0
                    if tid > 0 and bg in t0_bg:
                        mt0[j] = mt0[j + 1] = 1.0
                    if tid > 1 and bg in t1_bg:
                        mt1[j] = mt1[j + 1] = 1.0
                    if bg in seen_earlier_2:
                        mrec[j] = mrec[j + 1] = 1.0
            seen_earlier_1 |= {b for b in t_blks if _is_cjk_content_block(b)}
            seen_earlier_2 |= _cjk_bigrams(t_blks)
        else:
            for j, blk in enumerate(t_blks):
                if blk in seen_earlier_1:
                    m1[j] = 1.0
                if (
                    j + 1 < n
                    and _is_cjk_content_block(blk)
                    and _is_cjk_content_block(t_blks[j + 1])
                ):
                    bg = (blk, t_blks[j + 1])
                    if bg in seen_earlier_2:
                        m2[j] = m2[j + 1] = 1.0
                        mrec[j] = mrec[j + 1] = 1.0
                    if tid > 0 and bg in t0_bg:
                        mt0[j] = mt0[j + 1] = 1.0
                    if tid > 1 and bg in t1_bg:
                        mt1[j] = mt1[j + 1] = 1.0

        for j, blk in enumerate(t_blks):
            flat_blocks.append(list(blk))
            flat_feats.append([is_cur, recency, *turn_oh, m1[j], m2[j], mt0[j], mt1[j], mrec[j]])

        if offset + 1 < len(turn_blks):
            flat_blocks.append(list(SEP_BLOCK))
            flat_feats.append([is_cur, recency, *turn_oh, 0.0, 0.0, 0.0, 0.0, 0.0])

    return flat_blocks, flat_feats


def encode_target_blocks(answer: str) -> list[list[int]]:
    """Encode target string as UTF-8 codepoint blocks followed by `EOS_BLOCK`."""
    blocks = bytes_to_utf8_blocks(list(answer.encode("utf-8")))
    target = [*blocks, list(EOS_BLOCK)]
    if len(target) > MAX_DECODE_CHARS:
        raise ValueError(f"target exceeds MAX_DECODE_CHARS={MAX_DECODE_CHARS}: {answer!r}")
    return target


def collate_memory_batch(
    samples: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad batch of codepoint-block samples for `forward_loss`."""
    bsz = len(samples)
    max_m = max(len(s["mem_blocks"]) for s in samples)
    max_k = max(len(s["target_blocks"]) for s in samples)

    pad_blk = torch.tensor(PAD_BLOCK, dtype=torch.long)
    mem_blocks = pad_blk.view(1, 1, 4).expand(bsz, max_m, 4).clone()
    mem_feats = torch.zeros((bsz, max_m, 13), dtype=torch.float32)
    mem_mask = torch.zeros((bsz, max_m), dtype=torch.bool)
    tgt_intent = torch.zeros((bsz,), dtype=torch.long)
    tgt_blocks = pad_blk.view(1, 1, 4).expand(bsz, max_k, 4).clone()

    for b, s in enumerate(samples):
        m_len = len(s["mem_blocks"])
        k_len = len(s["target_blocks"])
        mem_blocks[b, :m_len] = torch.tensor(s["mem_blocks"], dtype=torch.long)
        mem_feats[b, :m_len] = torch.tensor(s["mem_feats"], dtype=torch.float32)
        mem_mask[b, :m_len] = True
        tgt_intent[b] = int(s["target_intent"])
        tgt_blocks[b, :k_len] = torch.tensor(s["target_blocks"], dtype=torch.long)

    return mem_blocks, mem_feats, mem_mask, tgt_intent, tgt_blocks


class _SDPABlock(nn.Module):
    """Deterministic, fast-kernel scaled_dot_product_attention block (`dropout=0.0`)."""

    def __init__(self, d_model: int, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mem_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mem_mask[:, None, None, :],
            dropout_p=0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, self.d_model)
        h = self.norm1(x + self.out_proj(attn_out))
        return self.norm2(h + self.ffn(h))


def _within_turn_positions(turn_oh: torch.Tensor) -> torch.Tensor:
    """Compute within-turn token positions `[0..MAX_TURN_POS-1]` from `turn_oh [B, L, 6]`."""
    bsz, seq_len, _ = turn_oh.shape
    turn_id = turn_oh.argmax(dim=-1)
    is_new = torch.zeros((bsz, seq_len), dtype=torch.bool, device=turn_oh.device)
    is_new[:, 0] = True
    if seq_len > 1:
        is_new[:, 1:] = turn_id[:, 1:] != turn_id[:, :-1]
    idx = torch.arange(seq_len, device=turn_oh.device).unsqueeze(0).expand(bsz, seq_len)
    reset_idx = torch.where(is_new, idx, torch.zeros_like(idx)).cummax(dim=1).values
    return (idx - reset_idx).clamp(max=MAX_TURN_POS - 1)


class OpenChineseCognitiveSubject(nn.Module):
    """Pure UTF-8 Conv1d(d=1,2,4) + Sandwiched SDPA1 -> Turn-Tracker -> SDPA2 subject."""

    def __init__(
        self,
        seed: int,
        *,
        use_history: bool = True,
        use_copy: bool = True,
        d_byte: int = 12,
        d_char: int = 40,
        d_mem: int = 48,
    ) -> None:
        super().__init__()
        self.init_seed = int(seed)
        self.use_history = bool(use_history)
        self.use_copy = bool(use_copy)
        self.d_byte = int(d_byte)
        self.d_char = int(d_char)
        self.d_mem = int(d_mem)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.byte_emb = nn.Embedding(BYTE_VOCAB, self.d_byte, padding_idx=PAD_BYTE)
            self.char_proj = nn.Linear(4 * self.d_byte, self.d_char)
            self.in_proj = nn.Linear(self.d_char + 13, self.d_mem)
            self.pos_emb = nn.Embedding(MAX_TURN_POS, self.d_mem)

            self.conv1 = nn.Conv1d(self.d_mem, self.d_mem, kernel_size=5, padding=2)
            self.cnorm1 = nn.LayerNorm(self.d_mem)
            self.conv2 = nn.Conv1d(self.d_mem, self.d_mem, kernel_size=5, padding=4, dilation=2)
            self.cnorm2 = nn.LayerNorm(self.d_mem)
            self.conv3 = nn.Conv1d(self.d_mem, self.d_mem, kernel_size=5, padding=8, dilation=4)
            self.cnorm3 = nn.LayerNorm(self.d_mem)

            self.sdpa1 = _SDPABlock(self.d_mem, num_heads=4)

            self.turn_pool = nn.Linear(self.d_mem, 4)
            self.turn_in_proj = nn.Linear(self.d_mem * 4 + 4, self.d_mem)
            self.turn_causal_gru = nn.GRU(self.d_mem, self.d_mem, batch_first=True)
            self.tnorm1 = nn.LayerNorm(self.d_mem)
            self.turn_bi_gru = nn.GRU(
                self.d_mem, self.d_mem // 2, batch_first=True, bidirectional=True
            )
            self.tnorm2 = nn.LayerNorm(self.d_mem)
            self.turn_sdpa = _SDPABlock(self.d_mem, num_heads=4)
            self.turn_to_tok = nn.Linear(self.d_mem, self.d_mem)
            self.fuse_norm = nn.LayerNorm(self.d_mem)

            self.sdpa2 = _SDPABlock(self.d_mem, num_heads=4)

            self.pool_q_cur = nn.Linear(self.d_mem, 1)
            self.pool_q_all = nn.Linear(self.d_mem, 1)

            rep_dim = self.d_mem * 4 + 2
            self.intent_head = nn.Sequential(
                nn.Linear(rep_dim, self.d_mem),
                nn.Tanh(),
                nn.Linear(self.d_mem, len(INTENTS)),
            )

            self.dec_init = nn.Linear(rep_dim, self.d_mem)
            self.dec_gru = nn.GRUCell(self.d_mem + 3 + MAX_DECODE_CHARS, self.d_mem)
            self.k_proj = nn.Linear(self.d_mem, self.d_mem)
            self.q_proj = nn.Linear(self.d_mem, self.d_mem)
            self.shift_gate = nn.Linear(self.d_mem * 2, 1)
            self.mode_head = nn.Linear(self.d_mem * 2 + 3, 3)
            self.byte_head = nn.Linear(self.d_mem * 2, 4 * BYTE_VOCAB)

    def _embed_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        shape = blocks.shape[:-1]
        emb = self.byte_emb(blocks.reshape(-1, 4)).reshape(*shape, 4 * self.d_byte)
        return F.gelu(self.char_proj(emb))

    def encode_memory(
        self,
        mem_blocks: torch.Tensor,
        mem_feats: torch.Tensor,
        mem_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode codepoint blocks via Conv1d(d=1,2,4) + SDPA1 -> Turn-Tracker -> SDPA2."""
        turn_oh = mem_feats[:, :, 2:8]  # [B, L, 6]
        pos_ids = _within_turn_positions(turn_oh)

        ch = self._embed_blocks(mem_blocks)
        h = F.gelu(self.in_proj(torch.cat([ch, mem_feats], dim=-1))) + self.pos_emb(pos_ids)
        h = self.cnorm1(h + F.gelu(self.conv1(h.transpose(1, 2)).transpose(1, 2)))
        h = self.cnorm2(h + F.gelu(self.conv2(h.transpose(1, 2)).transpose(1, 2)))
        h = self.cnorm3(h + F.gelu(self.conv3(h.transpose(1, 2)).transpose(1, 2)))
        h = h * mem_mask.unsqueeze(-1).float()

        h1 = self.sdpa1(h, mem_mask)

        t_mask = (turn_oh > 0.5) & mem_mask.unsqueeze(-1)  # [B, L, 6]
        t_scores = self.turn_pool(h1)  # [B, L, 4]
        overlap_feats = mem_feats[:, :, 9:13]  # [B, L, 4]

        turn_vecs: list[torch.Tensor] = []
        for t in range(6):
            tm = t_mask[:, :, t]
            has_t = tm.any(dim=-1, keepdim=True).float()
            sc = t_scores.masked_fill(~tm.unsqueeze(-1), -1e9)
            w = F.softmax(sc, dim=1) * has_t.unsqueeze(1)
            pooled_t = torch.bmm(w.transpose(1, 2), h1).reshape(h1.shape[0], -1)
            ov_t = overlap_feats.masked_fill(~tm.unsqueeze(-1), 0.0).max(dim=1).values * has_t
            turn_vecs.append(torch.cat([pooled_t, ov_t], dim=-1))

        u_turn = F.gelu(self.turn_in_proj(torch.stack(turn_vecs, dim=1)))  # [B, 6, d_mem]
        th_c, _ = self.turn_causal_gru(u_turn)
        th_c = self.tnorm1(u_turn + th_c)
        th_b, _ = self.turn_bi_gru(th_c)
        th = self.tnorm2(th_c + th_b)
        has_turn_mask = t_mask.any(dim=1)  # [B, 6]
        th = self.turn_sdpa(th, has_turn_mask)

        tok_turn_ctx = torch.bmm(turn_oh, th)
        mh = self.fuse_norm(h1 + self.turn_to_tok(tok_turn_ctx))
        mh = self.sdpa2(mh, mem_mask)

        cur_mask = (mem_feats[:, :, 0] > 0.5) & mem_mask
        cur_scores = self.pool_q_cur(mh).squeeze(-1).masked_fill(~cur_mask, -1e9)
        all_scores = self.pool_q_all(mh).squeeze(-1).masked_fill(~mem_mask, -1e9)
        w_cur = F.softmax(cur_scores, dim=-1)
        w_all = F.softmax(all_scores, dim=-1)
        rep_cur = torch.bmm(w_cur.unsqueeze(1), mh).squeeze(1)
        rep_all = torch.bmm(w_all.unsqueeze(1), mh).squeeze(1)

        mask_f = mem_mask.float().unsqueeze(-1)
        rep_mean = (mh * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        cur_turn_oh = (turn_oh * cur_mask.unsqueeze(-1).float()).sum(dim=1)
        cur_turn_oh = cur_turn_oh / cur_turn_oh.sum(dim=-1, keepdim=True).clamp(min=1.0)
        th_cur = torch.bmm(cur_turn_oh.unsqueeze(1), th).squeeze(1)

        hist_m2 = mem_feats[:, :, 9] * ((mem_feats[:, :, 0] < 0.5) & mem_mask).float()
        hist_has_m2 = hist_m2.max(dim=1, keepdim=True).values
        hist_m2_count = (hist_m2.sum(dim=1, keepdim=True) / 8.0).clamp(max=1.0)
        rep = torch.cat([rep_cur, rep_all, rep_mean, th_cur, hist_has_m2, hist_m2_count], dim=-1)
        keys = self.k_proj(mh)
        return mh, keys, rep

    def _decode_step(
        self,
        step_k: int,
        prev_mode_oh: torch.Tensor,
        state: torch.Tensor,
        ctx: torch.Tensor,
        prev_alpha: torch.Tensor | None,
        mh: torch.Tensor,
        keys: torch.Tensor,
        mem_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one OOV-invariant decoder step returning logits, state, ctx, and alpha."""
        bsz = state.shape[0]
        step_idx = min(step_k, MAX_DECODE_CHARS - 1)
        step_oh = F.one_hot(
            torch.full((bsz,), step_idx, dtype=torch.long, device=state.device),
            num_classes=MAX_DECODE_CHARS,
        ).float()
        new_state = self.dec_gru(torch.cat([ctx, prev_mode_oh, step_oh], dim=-1), state)

        q = self.q_proj(new_state).unsqueeze(-1)
        attn_logits = torch.bmm(keys, q).squeeze(-1) / math.sqrt(self.d_mem)
        attn_logits = attn_logits.masked_fill(~mem_mask, -1e9)
        a_content = F.softmax(attn_logits, dim=-1)

        if prev_alpha is not None:
            shifted = torch.cat([torch.zeros_like(prev_alpha[:, :1]), prev_alpha[:, :-1]], dim=-1)
            shifted = shifted * mem_mask.float()
            shifted = shifted / shifted.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            g_shift = torch.sigmoid(self.shift_gate(torch.cat([new_state, ctx], dim=-1)))
            alpha = (1.0 - g_shift) * a_content + g_shift * shifted
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        else:
            alpha = a_content

        new_ctx = torch.bmm(alpha.unsqueeze(1), mh).squeeze(1)
        dec_feat = torch.cat([new_state, new_ctx], dim=-1)
        raw_mode = self.mode_head(torch.cat([dec_feat, prev_mode_oh], dim=-1))
        if not self.use_copy:
            mode_logits = torch.cat(
                [raw_mode[:, :2], torch.full_like(raw_mode[:, 2:3], -1e9)], dim=-1
            )
        else:
            mode_logits = raw_mode
        byte_logits = self.byte_head(dec_feat).view(bsz, 4, BYTE_VOCAB)
        return mode_logits, byte_logits, new_state, new_ctx, alpha

    def forward_loss(
        self,
        mem_blocks: torch.Tensor,
        mem_feats: torch.Tensor,
        mem_mask: torch.Tensor,
        target_intent: torch.Tensor,
        target_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute joint `(total_loss, intent_loss, block_loss)` with span-exact copy."""
        bsz, max_k, _ = target_blocks.shape
        max_m = mem_blocks.shape[1]
        mh, keys, rep = self.encode_memory(mem_blocks, mem_feats, mem_mask)

        intent_loss = F.cross_entropy(self.intent_head(rep), target_intent)

        span_match = torch.zeros((bsz, max_k, max_m), dtype=torch.bool, device=mem_blocks.device)
        has_span_copy = torch.zeros((bsz,), dtype=torch.bool, device=mem_blocks.device)
        if self.use_copy:
            mem_id = (
                (mem_blocks[:, :, 0] << 24)
                | (mem_blocks[:, :, 1] << 16)
                | (mem_blocks[:, :, 2] << 8)
                | mem_blocks[:, :, 3]
            )
            tgt_id = (
                (target_blocks[:, :, 0] << 24)
                | (target_blocks[:, :, 1] << 16)
                | (target_blocks[:, :, 2] << 8)
                | target_blocks[:, :, 3]
            )
            is_ans_tok = (target_blocks[:, :, 0] != EOS_BYTE) & (target_blocks[:, :, 0] != PAD_BYTE)
            ans_lens = is_ans_tok.sum(dim=-1).tolist()
            by_len: dict[int, list[int]] = {}
            for b, ans_len in enumerate(ans_lens):
                if 0 < ans_len <= max_m:
                    by_len.setdefault(ans_len, []).append(b)
            for ans_len, idxs in by_len.items():
                idx_t = torch.tensor(idxs, dtype=torch.long, device=mem_blocks.device)
                win = mem_id[idx_t].unfold(1, ans_len, 1)
                win_m = mem_mask[idx_t].unfold(1, ans_len, 1).all(dim=-1)
                starts = win_m & (win == tgt_id[idx_t, None, :ans_len]).all(dim=-1)
                has_span_copy[idx_t] = starts.any(dim=-1)
                n_pos = starts.shape[1]
                for k in range(ans_len):
                    span_match[idx_t, k, k : k + n_pos] = starts

        state = torch.tanh(self.dec_init(rep))
        ctx = rep[:, : self.d_mem]
        prev_mode = F.one_hot(
            torch.zeros((bsz,), dtype=torch.long, device=mem_blocks.device), num_classes=3
        ).float()
        prev_alpha: torch.Tensor | None = None

        step_losses: list[torch.Tensor] = []
        active_counts: list[torch.Tensor] = []

        for k in range(max_k):
            mode_logits, byte_logits, state, ctx, prev_alpha = self._decode_step(
                k, prev_mode, state, ctx, prev_alpha, mh, keys, mem_mask
            )
            tgt_k = target_blocks[:, k, :]
            is_active = tgt_k[:, 0] != PAD_BYTE
            is_eos = tgt_k[:, 0] == EOS_BYTE

            log_mode = F.log_softmax(mode_logits, dim=-1)
            log_byte = F.log_softmax(byte_logits, dim=-1)
            safe_tgt = tgt_k.clamp(0, BYTE_VOCAB - 1)
            log_p_gen = log_byte.gather(2, safe_tgt.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

            match_k = span_match[:, k, :]
            has_copy = self.use_copy & has_span_copy & (~is_eos) & is_active
            p_copy = (prev_alpha * match_k.float()).sum(dim=-1).clamp(min=1e-12)

            log_char = torch.where(
                has_copy,
                log_mode[:, 2] + torch.log(p_copy),
                log_mode[:, 1] + log_p_gen,
            )
            log_prob_k = torch.where(is_eos, log_mode[:, 0], log_char)

            step_losses.append((-log_prob_k * is_active.float()).sum())
            active_counts.append(is_active.float().sum())
            next_mode_id = torch.where(
                has_copy,
                torch.full((bsz,), 2, dtype=torch.long, device=mem_blocks.device),
                torch.ones((bsz,), dtype=torch.long, device=mem_blocks.device),
            )
            prev_mode = F.one_hot(next_mode_id, num_classes=3).float()

        total_active = torch.stack(active_counts).sum().clamp(min=1.0)
        block_loss = torch.stack(step_losses).sum() / total_active
        total_loss = 1.5 * intent_loss + block_loss
        return total_loss, intent_loss, block_loss

    @staticmethod
    def _best_valid_utf8_block(byte_logits_row: torch.Tensor) -> tuple[list[int], float]:
        """Decode highest-scoring valid 4-byte UTF-8 block using `taiji.utf8_state`."""
        log_p = F.log_softmax(byte_logits_row, dim=-1)
        allowed0 = torch.tensor(utf8_allowed(0, 0), dtype=torch.long)
        b0 = int(allowed0[int(torch.argmax(log_p[0, allowed0]).item())].item())
        lp = float(log_p[0, b0].item())

        if b0 < 0x80:
            blk = [b0, PAD_BYTE, PAD_BYTE, PAD_BYTE]
        else:
            sz = 2 if b0 < 0xE0 else (3 if b0 < 0xF0 else 4)
            rem, lead = advance_utf8(0, 0, b0)
            blk = [b0]
            for slot in range(1, sz):
                allowed = torch.tensor(utf8_allowed(rem, lead), dtype=torch.long)
                b = int(allowed[int(torch.argmax(log_p[slot, allowed]).item())].item())
                lp += float(log_p[slot, b].item())
                blk.append(b)
                rem, lead = advance_utf8(rem, lead, b)
            while len(blk) < 4:
                blk.append(PAD_BYTE)

        for slot in range(4):
            if blk[slot] == PAD_BYTE:
                lp += float(log_p[slot, PAD_BYTE].item())
        return blk, lp

    @torch.no_grad()
    def predict_turn(
        self,
        turns_bytes: list[list[int]],
        *,
        ablate_history_gen: torch.Generator | None = None,
        max_chars: int = MAX_DECODE_CHARS,
    ) -> tuple[int, str, bool]:
        """Predict `(intent_id, decoded_utf8_text, utf8_valid)` using strict `utf8_state`."""
        self.eval()
        used_turns = [list(tb) for tb in turns_bytes]
        if ablate_history_gen is not None and len(used_turns) > 1:
            for idx in range(len(used_turns) - 1):
                blks = bytes_to_utf8_blocks(used_turns[idx])
                if len(blks) > 1:
                    perm = torch.randperm(len(blks), generator=ablate_history_gen).tolist()
                    used_turns[idx] = [b for p in perm for b in blks[p] if b != PAD_BYTE]

        flat_blocks, flat_feats = build_turn_memory_tensors(
            used_turns, use_history=self.use_history
        )
        mem_blocks = torch.tensor([flat_blocks], dtype=torch.long)
        mem_feats = torch.tensor([flat_feats], dtype=torch.float32)
        mem_mask = torch.ones((1, len(flat_blocks)), dtype=torch.bool)

        mh, keys, rep = self.encode_memory(mem_blocks, mem_feats, mem_mask)
        pred_intent = int(torch.argmax(self.intent_head(rep), dim=-1).item())

        state = torch.tanh(self.dec_init(rep))
        ctx = rep[:, : self.d_mem]
        prev_mode = F.one_hot(torch.zeros((1,), dtype=torch.long), num_classes=3).float()
        prev_alpha: torch.Tensor | None = None

        out_bytes: list[int] = []
        for k in range(max_chars):
            mode_logits, byte_logits, state, ctx, prev_alpha = self._decode_step(
                k, prev_mode, state, ctx, prev_alpha, mh, keys, mem_mask
            )
            mode_probs = F.softmax(mode_logits[0], dim=-1)
            p_eos = float(mode_probs[0].item())
            p_gen = float(mode_probs[1].item())
            p_copy = float(mode_probs[2].item())

            gen_blk, gen_lp = self._best_valid_utf8_block(byte_logits[0])
            score_gen = p_gen * math.exp(gen_lp)

            best_copy_blk: list[int] | None = None
            score_copy = -1.0
            if self.use_copy:
                copy_scores: dict[tuple[int, int, int, int], float] = {}
                alphas = prev_alpha[0].tolist()
                for blk_list, a_val in zip(flat_blocks, alphas, strict=True):
                    blk_t = (blk_list[0], blk_list[1], blk_list[2], blk_list[3])
                    if blk_t in (SEP_BLOCK, PAD_BLOCK):
                        continue
                    copy_scores[blk_t] = copy_scores.get(blk_t, 0.0) + float(a_val)
                if copy_scores:
                    top_blk, top_a = max(copy_scores.items(), key=lambda kv: kv[1])
                    best_copy_blk = list(top_blk)
                    score_copy = p_copy * top_a

            if p_eos >= score_gen and p_eos >= score_copy:
                break
            if best_copy_blk is not None and score_copy > score_gen:
                chosen_blk = best_copy_blk
                nxt_mode = 2
            else:
                chosen_blk = gen_blk
                nxt_mode = 1
            for b in chosen_blk:
                if b != PAD_BYTE:
                    out_bytes.append(b)
            prev_mode = F.one_hot(torch.tensor([nxt_mode], dtype=torch.long), num_classes=3).float()

        trimmed = trim_partial_tail(bytes(out_bytes))
        try:
            decoded = trimmed.decode("utf-8")
            valid = len(trimmed) == len(out_bytes)
        except UnicodeDecodeError:
            decoded = ""
            valid = False
        return pred_intent, decoded, valid

    def save_to_session_store(
        self,
        store: SessionStore,
        subject_id: str,
        session_id: int = 0,
    ) -> SessionRecord:
        """Persist model capsule and architecture metadata to disk via `SessionStore`."""
        record = SessionRecord(subject_id=subject_id)
        for name, tensor in self.state_dict().items():
            record.capsule[name] = tensor.detach().cpu().clone()
        record.append_experience(
            "open_chinese_subject_snapshot",
            {
                "schema": SCHEMA,
                "session_id": int(session_id),
                "init_seed": self.init_seed,
                "use_history": self.use_history,
                "use_copy": self.use_copy,
                "d_byte": self.d_byte,
                "d_char": self.d_char,
                "d_mem": self.d_mem,
            },
            source="open_chinese_subject",
        )
        store.save(record)
        return record

    @classmethod
    def load_from_session_store(
        cls,
        store: SessionStore,
        subject_id: str,
    ) -> OpenChineseCognitiveSubject:
        """Restore subject weights and architecture flags bit-identically from `SessionStore`."""
        record = store.load(subject_id)
        meta: dict[str, Any] = {}
        for item in reversed(record.experience):
            if item.get("kind") == "open_chinese_subject_snapshot":
                meta = dict(item["payload"])
                break
        if not meta:
            raise ValueError(f"missing open_chinese_subject_snapshot metadata for {subject_id!r}")
        subject = cls(
            seed=int(meta["init_seed"]),
            use_history=bool(meta["use_history"]),
            use_copy=bool(meta["use_copy"]),
            d_byte=int(meta["d_byte"]),
            d_char=int(meta["d_char"]),
            d_mem=int(meta["d_mem"]),
        )
        missing, unexpected = subject.load_state_dict(record.capsule, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"state mismatch missing={missing} unexpected={unexpected}")
        return subject
