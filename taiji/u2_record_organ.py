"""Differentiable matched-filter record-reading organ for U2Subject.

This organ replaces the earlier pooled-projection + hard-threshold prototype
with an end-to-end differentiable, template-agnostic, multi-record mechanism:

1. **Character-level matched-filter relevance** (r3 class, template-agnostic):
   computes the vectorized character cosine similarity matrix between the full
   request and each candidate record, extracting both multi-scale contiguous
   diagonal match statistics (deterministic 1.0 on verbatim entity matches at
   birth, ~0.23 on near-miss 3/4-char mismatches, with zero hardcoded suffix)
   and a learned request-character salience conjunctive match.
2. **Multi-record competition with recency tie-breaking & backoff sentinel**:
   competes all records in ``state["records"]`` against a null sentinel slot,
   breaking ties among matching records in favor of newer versions/entries.
3. **End-to-end differentiable variable-length span copy**:
   contextualizes each record around its matched-entity peak via a lightweight
   1D Conv, predicts a joint distribution over ``(start_char, span_len)`` spans,
   and scatters byte/EOS probabilities directly into the 257-way vocabulary.
4. **Exact birth neutrality**:
   ``organ_mix`` starts at 0.0, so attaching the organ leaves the parent
   subject bit-for-bit identical (delta == 0.0) while preserving a clean,
   non-zero gradient on the very first feedback update.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional

MAX_REQUEST_CHARS = 64
MAX_SPAN_CHARS = 4


def spans_of(record: bytes) -> list[int]:
    """Return the byte start offset of every UTF-8 character in ``record``."""
    out, index = [], 0
    while index < len(record):
        out.append(index)
        lead = record[index]
        step = 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
        index += min(step, len(record) - index)
    return out


def char_bounds(record: bytes) -> tuple[list[int], list[int]]:
    """Return parallel ``(starts, ends)`` byte offsets for every UTF-8 character."""
    starts = spans_of(record)
    ends = [*starts[1:], len(record)]
    return starts, ends


def windows_of(record: bytes) -> list[bytes]:
    """Two-character windows kept for backward compatibility with E5 scripts."""
    spans = spans_of(record)
    out = []
    for i in range(len(spans) - 1):
        end = spans[i + 2] if i + 2 < len(spans) else len(record)
        out.append(record[spans[i] : end])
    return out


def _sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    half = dim // 2
    freq = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(100.0) / max(half, 1)))
    angles = positions * freq[None, :]
    basis = torch.zeros(length, dim, dtype=torch.float32)
    basis[:, 0 : 2 * half : 2] = angles.sin()
    basis[:, 1 : 2 * half : 2] = angles.cos()
    return basis * 0.25


def mix_log_probabilities(
    parent_logp: torch.Tensor,
    route_prob: torch.Tensor,
    organ_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Differentiable UTF-8-safe log-mixture, bit-identical when ``route_prob == 0``.

    Sanitizes masked-out entries before computing ratios/logaddexp so backward
    passes never produce NaN gradients on illegal UTF-8 byte slots.
    """
    neg_inf = parent_logp.new_full((), -torch.inf)
    safe_parent_logp = torch.where(mask, parent_logp, torch.zeros_like(parent_logp))
    if float(route_prob.detach()) <= 1e-6:
        parent_p = safe_parent_logp.exp()
        ratio = organ_probs / parent_p.clamp_min(1e-12)
        mixed = safe_parent_logp + torch.log1p(route_prob * (ratio - 1.0))
        return torch.where(mask, mixed, neg_inf)
    safe_route = route_prob.clamp(min=1e-12, max=1.0 - 1e-7)
    log_parent_term = torch.log1p(-safe_route) + safe_parent_logp
    positive = mask & (organ_probs > 0)
    safe_organ_p = torch.where(positive, organ_probs, torch.ones_like(organ_probs))
    organ_logp = torch.where(positive, safe_organ_p.log(), neg_inf)
    log_organ_term = safe_route.log() + organ_logp
    mixed = torch.logaddexp(log_parent_term, log_organ_term)
    return torch.where(mask, mixed, neg_inf)


class RecordOrgan(nn.Module):
    """Multi-record matched-filter routing & differentiable span-copy organ."""

    req_pos_basis: torch.Tensor
    match_basis: torch.Tensor

    def __init__(
        self, emb: int = 16, width: int = 32, max_span_chars: int = MAX_SPAN_CHARS
    ) -> None:
        super().__init__()
        self.emb = emb
        self.width = width
        self.max_span_chars = max_span_chars
        self.byte = nn.Embedding(256, emb)
        # Fixed orthogonal-ish byte basis anchors match cosine similarity against embedding collapse.
        gen = torch.Generator().manual_seed(20261119)
        basis = functional.normalize(torch.randn(256, emb, generator=gen), dim=-1, eps=1e-6)
        self.register_buffer("match_basis", basis, persistent=False)
        # Legacy projection layers retained for backward compatibility with E5.9/10 scripts.
        self.query = nn.Linear(2 * emb, width)
        self.key = nn.Linear(emb, width)
        self.sentinel = nn.Parameter(torch.zeros(width))
        self.gain = nn.Parameter(torch.zeros(()))
        # Template-agnostic character-salience & matched-filter relevance head.
        self.register_buffer(
            "req_pos_basis", _sinusoidal_positions(MAX_REQUEST_CHARS, emb), persistent=False
        )
        self.salience = nn.Linear(emb, 1)
        nn.init.zeros_(self.salience.weight)
        nn.init.zeros_(self.salience.bias)
        self.route_head = nn.Linear(4, 1)
        with torch.no_grad():
            self.route_head.weight.copy_(torch.tensor([[4.0, 1.0, 4.0, 0.25]]))
            self.route_head.bias.fill_(-5.0)
        self.recency_logit = nn.Parameter(torch.tensor(0.0))
        # Born neutral: 0.0 leaves the parent distribution bit-identical at birth,
        # while passing an exact non-zero gradient on the first feedback update.
        self.organ_mix = nn.Parameter(torch.tensor(0.0))
        # Contextual span extractor over character embeddings + match/position channels.
        self.span_conv1 = nn.Conv1d(emb + 3, width, kernel_size=3, padding=1)
        self.span_conv2 = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.start_head = nn.Linear(width, 1)
        self.len_head = nn.Linear(width, max_span_chars)

    def active_strength(self) -> torch.Tensor:
        """Return birth-neutral activation in ``[0, 1)`` (0.0 at birth, -> 1.0 once trained)."""
        mix_mag = torch.where(self.organ_mix >= 0, self.organ_mix, -self.organ_mix)
        return torch.tanh(2.0 * mix_mag)

    def _char_mean(
        self, raw: bytes, *, with_match_basis: bool = False
    ) -> tuple[torch.Tensor, list[int], list[int]]:
        """Average byte embeddings (or match-anchored vectors) over UTF-8 characters."""
        device = self.byte.weight.device
        dtype = self.byte.weight.dtype
        starts, ends = char_bounds(raw)
        n_chars = len(starts)
        if n_chars == 0:
            return torch.zeros((0, self.emb), dtype=dtype, device=device), starts, ends
        byte_ids = torch.tensor(list(raw), dtype=torch.long, device=device)
        byte_embs = self.byte(byte_ids)
        if with_match_basis:
            byte_embs = self.match_basis[byte_ids].to(dtype=dtype) + 0.15 * torch.tanh(byte_embs)
        char_index = torch.empty(len(raw), dtype=torch.long, device=device)
        counts = torch.empty(n_chars, dtype=dtype, device=device)
        for idx, (s, e) in enumerate(zip(starts, ends, strict=True)):
            char_index[s:e] = idx
            counts[idx] = float(e - s)
        summed = torch.zeros((n_chars, self.emb), dtype=dtype, device=device)
        summed.index_add_(0, char_index, byte_embs)
        return summed / counts[:, None].clamp_min(1.0), starts, ends

    def char_vectors(self, raw: bytes) -> tuple[torch.Tensor, list[int], list[int]]:
        """Vectorized UTF-8 character embeddings (mean of each character's byte embeddings)."""
        return self._char_mean(raw, with_match_basis=False)

    def match_char_vectors(self, raw: bytes) -> torch.Tensor:
        """Anchored UTF-8 character vectors preventing embedding collapse during matching."""
        vecs, _, _ = self._char_mean(raw, with_match_basis=True)
        return vecs

    def match_record(
        self,
        q_emb: torch.Tensor,
        uq: torch.Tensor,
        w_sal: torch.Tensor,
        record: bytes,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
        """Compute matched-filter relevance features and per-char match localization."""
        r_emb, r_starts, r_ends = self.char_vectors(record)
        n_req = q_emb.shape[0]
        n_rec = r_emb.shape[0]
        dtype = self.byte.weight.dtype
        device = self.byte.weight.device
        if n_req == 0 or n_rec == 0:
            neg = torch.full((4,), -1.0, dtype=dtype, device=device)
            return neg, torch.zeros((n_rec,), dtype=dtype, device=device), r_emb, r_starts, r_ends
        r_match_emb = self.match_char_vectors(record)
        ur = functional.normalize(r_match_emb, dim=-1, eps=1e-6)
        sim = uq @ ur.T  # [L, N]

        w = min(4, n_req, n_rec)
        slices = [sim[t : n_req - w + 1 + t, t : n_rec - w + 1 + t] for t in range(w)]
        diag_min = torch.stack(slices, dim=0).amin(dim=0)  # [L-w+1, N-w+1]
        w_slices = torch.stack([w_sal[t : n_req - w + 1 + t] for t in range(w)], dim=0)
        win_sal = 0.5 * w_slices.mean(dim=0) + 0.5 * w_slices.amin(dim=0)
        win_centered = win_sal - win_sal.mean()
        # Smoothly bounded in (0, 1] and equal to 1.0 at birth (win_centered == 0).
        win_gate = torch.minimum(
            torch.ones_like(win_centered),
            2.0 * torch.sigmoid(14.0 * win_centered),
        )
        win_prob = (14.0 * win_centered).softmax(dim=0)

        gated_diag = diag_min * win_gate[:, None]
        s_diag4 = gated_diag.amax()
        q_win_match = diag_min.amax(dim=1)
        s_pool4 = (win_prob * q_win_match).sum()

        col_peak = gated_diag.amax(dim=0)  # [N-w+1]
        span_match = torch.full((n_rec,), -1.0, dtype=dtype, device=device)
        for t in range(w):
            padded = torch.full((n_rec,), -1.0, dtype=dtype, device=device)
            padded[t : t + n_rec - w + 1] = col_peak
            span_match = torch.maximum(span_match, padded)

        char_best = sim.amax(dim=1)
        char_prob = (10.0 * (w_sal - w_sal.mean())).softmax(dim=0)
        mu_sal = (char_prob * char_best).sum()
        s_cubed = s_diag4.clamp_min(0.0).pow(3)

        features = torch.stack([s_diag4, s_pool4, s_cubed, mu_sal * s_diag4])
        return features, span_match, r_emb, r_starts, r_ends

    def _span_distribution(
        self,
        r_emb: torch.Tensor,
        span_match: torch.Tensor,
        r_starts: list[int],
        r_ends: list[int],
    ) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
        """Predict joint probabilities over valid ``(start_char, span_len)`` byte spans."""
        n_rec = r_emb.shape[0]
        dtype = r_emb.dtype
        device = r_emb.device
        if n_rec == 0:
            return torch.zeros((0,), dtype=dtype, device=device), (), ()
        positions = torch.arange(n_rec, dtype=dtype, device=device)
        match_weights = (span_match * 12.0).softmax(dim=0)
        entity_center = (match_weights * positions).sum()
        denom = float(max(n_rec, 1))
        rel_offset = (positions - entity_center) / denom
        abs_pos = positions / denom
        node_in = torch.cat(
            [r_emb, span_match[:, None], rel_offset[:, None], abs_pos[:, None]], dim=-1
        )  # [N, emb + 3]
        hidden = functional.gelu(self.span_conv1(node_in.T.unsqueeze(0)))
        hidden = functional.gelu(self.span_conv2(hidden)).squeeze(0).T  # [N, width]
        start_logits = self.start_head(hidden).squeeze(-1)  # [N]
        len_logits = self.len_head(hidden)  # [N, max_span_chars]
        joint = start_logits[:, None] + len_logits  # [N, max_span_chars]

        valid_logits = []
        span_starts: list[int] = []
        span_ends: list[int] = []
        max_len = min(self.max_span_chars, n_rec)
        for i in range(n_rec):
            for length in range(1, max_len + 1):
                if i + length <= n_rec:
                    valid_logits.append(joint[i, length - 1])
                    span_starts.append(r_starts[i])
                    span_ends.append(r_ends[i + length - 1])
        span_probs = torch.stack(valid_logits).softmax(dim=0)
        return span_probs, tuple(span_starts), tuple(span_ends)

    def _query_salience(self, q_match: torch.Tensor) -> torch.Tensor:
        """Compute per-character salience weights in ``(0, 1)`` from anchored query vectors."""
        n_req = q_match.shape[0]
        dtype = q_match.dtype
        device = q_match.device
        if n_req == 0:
            return torch.zeros((0,), dtype=dtype, device=device)
        pos = self.req_pos_basis[: min(n_req, MAX_REQUEST_CHARS)].to(dtype=dtype, device=device)
        if n_req > MAX_REQUEST_CHARS:
            pad = torch.zeros((n_req - MAX_REQUEST_CHARS, self.emb), dtype=dtype, device=device)
            pos = torch.cat([pos, pad], dim=0)
        return torch.sigmoid(self.salience(q_match + pos).squeeze(-1))

    def _legacy_override_plan(
        self, request: bytes, records: Sequence[Any], route: Any
    ) -> dict[str, Any] | None:
        """Honour a frozen E5-style instance-level ``route`` override, or return ``None``.

        The E5.10r3 card (D-20261119-08) was earned with a closure patch that
        installs ``organ.route`` per instance and trains the two scalars it
        closes over.  The S0 organ internalised that matched filter, so the
        forward path is ``plan``/``emit``; this explicit compatibility branch
        keeps the frozen runner reproducible without letting a patched attribute
        alter the native path.  It is single-record only, exactly like the
        legacy call site it mirrors.
        """
        records = tuple(records)
        if route is None or len(records) != 1:
            return None
        dtype = self.byte.weight.dtype
        device = self.byte.weight.device
        rec = records[0]
        raw = bytes(rec) if isinstance(rec, (bytes, bytearray)) else bytes(rec.raw)
        window = route(request, raw)
        zero = torch.zeros((), dtype=dtype, device=device)
        one = torch.ones((), dtype=dtype, device=device)
        if window is None:
            return {
                "route_prob": zero,
                "match_prob": zero,
                "record_weights": torch.zeros((1,), dtype=dtype, device=device),
                "entries": (),
                "legacy_hard": True,
            }
        start = raw.find(bytes(window))
        if start < 0:
            # A patched route must return a verbatim record span; otherwise the
            # compatibility contract is violated and the native path decides.
            return None
        entry = {
            "raw": raw,
            "span_probs": one.reshape(1),
            "span_starts": [start],
            "span_ends": [start + len(window)],
            "match_feature": one,
        }
        return {
            "route_prob": one,
            "match_prob": one,
            "record_weights": one.reshape(1),
            "entries": (entry,),
            "legacy_hard": True,
        }

    def plan(self, request: bytes, records: Sequence[Any]) -> dict[str, Any]:
        """Score all records against ``request`` and precompute differentiable span weights."""
        dtype = self.byte.weight.dtype
        device = self.byte.weight.device
        legacy = self._legacy_override_plan(request, records, self.__dict__.get("route"))
        if legacy is not None:
            return legacy
        if not records:
            return {
                "route_prob": torch.zeros((), dtype=dtype, device=device),
                "match_prob": torch.zeros((), dtype=dtype, device=device),
                "record_weights": torch.zeros((0,), dtype=dtype, device=device),
                "entries": (),
            }
        q_emb, _, _ = self.char_vectors(request)
        q_match = self.match_char_vectors(request)
        n_req = q_emb.shape[0]
        uq = functional.normalize(q_match, dim=-1, eps=1e-6) if n_req > 0 else q_match
        w_sal = self._query_salience(q_match)

        raw_list: list[bytes] = []
        versions: list[int] = []
        for idx, rec in enumerate(records):
            if isinstance(rec, (bytes, bytearray)):
                raw_list.append(bytes(rec))
                versions.append(idx)
            else:
                raw_list.append(bytes(rec.raw))
                versions.append(int(getattr(rec, "version", idx)))

        # Recency rank breaks ties when two records match the same entity.
        n_records = len(raw_list)
        v_min, v_max = min(versions), max(versions)
        if v_max > v_min:
            recency_ranks = [(v - v_min) / float(v_max - v_min) for v in versions]
        elif n_records > 1:
            recency_ranks = [i / float(n_records - 1) for i in range(n_records)]
        else:
            recency_ranks = [0.0]

        recency_scale = 2.5 * self.recency_logit.sigmoid()
        route_logits = []
        entries = []
        for raw, rank in zip(raw_list, recency_ranks, strict=True):
            features, span_match, r_emb, r_starts, r_ends = self.match_record(q_emb, uq, w_sal, raw)
            base_logit = self.route_head(features).squeeze(-1)
            # Only matching records compete on recency/version; distractors get ~0 boost.
            logit = base_logit + recency_scale * rank * base_logit.sigmoid()
            route_logits.append(logit)
            span_probs, span_starts, span_ends = self._span_distribution(
                r_emb, span_match, r_starts, r_ends
            )
            entries.append(
                {
                    "raw": raw,
                    "span_probs": span_probs,
                    "span_starts": span_starts,
                    "span_ends": span_ends,
                    "match_feature": features[0],
                }
            )

        stacked = torch.stack(route_logits)
        probs = torch.cat([stacked, stacked.new_zeros((1,))]).softmax(dim=0)
        rec_probs = probs[:-1]
        match_prob = 1.0 - probs[-1]
        route_gate = torch.sigmoid(12.0 * (match_prob - 0.5))
        route_prob = route_gate * self.active_strength()
        record_weights = rec_probs / rec_probs.sum().clamp_min(1e-9)
        return {
            "route_prob": route_prob,
            "match_prob": match_prob,
            "record_weights": record_weights,
            "entries": tuple(entries),
        }

    def emit(
        self,
        plan: dict[str, Any],
        offset: int,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute differentiable ``(route_prob, byte_probs[257])`` at utterance byte ``offset``."""
        dtype = self.byte.weight.dtype
        device = self.byte.weight.device
        route_prob: torch.Tensor = plan["route_prob"]
        entries = plan["entries"]
        if not entries:
            uniform = torch.full((257,), 1.0 / 257.0, dtype=dtype, device=device)
            return route_prob, uniform

        record_weights: torch.Tensor = plan["record_weights"]
        organ_probs = torch.zeros((257,), dtype=dtype, device=device)
        for rec_weight, entry in zip(record_weights, entries, strict=True):
            span_probs: torch.Tensor = entry["span_probs"]
            if span_probs.numel() == 0:
                continue
            raw: bytes = entry["raw"]
            symbols = [
                raw[s + offset] if s + offset < e else 256
                for s, e in zip(entry["span_starts"], entry["span_ends"], strict=True)
            ]
            symbol_ids = torch.tensor(symbols, dtype=torch.long, device=device)
            organ_probs = organ_probs.index_add(0, symbol_ids, rec_weight * span_probs)

        total = organ_probs.sum()
        if float(total.detach()) > 1e-12:
            organ_probs = organ_probs / total

        if mask is not None:
            mask_f = mask.to(dtype=dtype)
            masked_probs = organ_probs * mask_f
            legal_mass = masked_probs.sum()
            if float(legal_mass.detach()) <= 1e-9:
                fallback = mask_f / mask_f.sum().clamp_min(1.0)
                return route_prob * 0.0, fallback
            organ_probs = masked_probs / legal_mass
            route_prob = route_prob * legal_mass.clamp(max=1.0)
        return route_prob, organ_probs

    def match_statistic(self, request: bytes, record: bytes) -> torch.Tensor:
        """Return the birth-ready template-agnostic matched-filter score for ``(request, record)``."""
        q_match = self.match_char_vectors(request)
        if q_match.shape[0] == 0:
            return self.byte.weight.new_tensor(-1.0)
        uq = functional.normalize(q_match, dim=-1, eps=1e-6)
        w_sal = self._query_salience(q_match)
        features, *_ = self.match_record(q_match, uq, w_sal, record)
        return features[0]

    # --- Legacy single-record API preserved for backward compatibility ---

    def attention(self, request: bytes, record: bytes) -> torch.Tensor:
        """Softmax over [window scores; sentinel]; kept for legacy E5 scripts."""
        windows = windows_of(record)
        if not windows:
            return torch.tensor([1.0], dtype=self.byte.weight.dtype, device=self.byte.weight.device)
        device = self.byte.weight.device
        text = self.byte(torch.tensor(list(request), dtype=torch.long, device=device))
        pooled = torch.cat([text.mean(0), text.amax(0)], -1)
        query = self.query(pooled)
        scale = math.sqrt(query.shape[-1])
        keys = self.key(
            torch.stack(
                [
                    self.byte(torch.tensor(list(window), dtype=torch.long, device=device)).mean(0)
                    for window in windows
                ]
            )
        )
        scores = keys @ query / scale
        sentinel = (query @ self.sentinel) / scale + self.gain * scores.max()
        return torch.cat([scores, sentinel.reshape(1)]).softmax(-1)

    def route(self, request: bytes, record: bytes) -> bytes | None:
        """Return the highest-probability span bytes, or None when the sentinel wins."""
        plan = self.plan(request, [record])
        if float(plan["match_prob"].detach()) < 0.5 or not plan["entries"]:
            return None
        entry = plan["entries"][0]
        if entry["span_probs"].numel() == 0:
            return None
        best = int(entry["span_probs"].detach().argmax())
        return bytes(entry["raw"][entry["span_starts"][best] : entry["span_ends"][best]])

    def log_distribution(self, window: bytes, offset: int) -> torch.Tensor:
        logits = torch.full(
            (257,), -1e4, dtype=self.byte.weight.dtype, device=self.byte.weight.device
        )
        if offset < len(window):
            logits[window[offset]] = 0.0
        else:
            logits[256] = 0.0
        return logits.log_softmax(-1)
