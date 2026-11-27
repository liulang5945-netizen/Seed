"""Dynamic Byte Vocab Growth & Entropy-Boundary Chunker (`taiji-dynamic-byte-chunker-v1`).

Implements `TAIJI_NATIVE_ARCHITECTURE_V1 §2.2 / §5.2 / §5.10`:
  * Permanently reserves raw UTF-8 byte primitives (`0..255` + EOS/BOS/PAD = 259),
    guaranteeing 0% OOV for any text/domain forever;
  * Allows dynamic growth of multi-byte macro-symbols with greedy longest-match
    encoding and exact lossless decoding back to raw bytes;
  * Function-neutral birth (`grow_macro_symbol_neutral_`) initializes each macro's
    residual delta to `0.0` atop the pooled byte encoder so forward representation
    perturbation at birth is identically `0.0`;
  * `EntropyBoundaryChunker` groups byte streams causally using predictive entropy,
    explicitly marked as a compute-allocation signal rather than concept ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

BYTE_VOCAB = 259  # 0..255 raw UTF-8 bytes, 256=EOS, 257=BOS, 258=PAD
EOS_BYTE = 256

SCHEMA = "taiji-dynamic-byte-chunker-v1"
BOUNDARY_ROLE = "compute_allocation_signal_not_concept_truth"


@dataclass(frozen=True)
class MacroSymbol:
    symbol_id: int
    raw_bytes: bytes
    birth_step: int = 0
    lineage_tag: str = "chunk_growth"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_id": self.symbol_id,
            "raw_hex": self.raw_bytes.hex(),
            "birth_step": self.birth_step,
            "lineage_tag": self.lineage_tag,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MacroSymbol:
        return cls(
            symbol_id=int(data["symbol_id"]),
            raw_bytes=bytes.fromhex(str(data["raw_hex"])),
            birth_step=int(data.get("birth_step", 0)),
            lineage_tag=str(data.get("lineage_tag", "chunk_growth")),
        )


class DynamicByteVocab:
    """Lossless byte-fallback vocabulary supporting neutral macro-symbol growth."""

    def __init__(self) -> None:
        self._macros: list[MacroSymbol] = []
        self._bytes_to_id: dict[bytes, int] = {}
        self._id_to_bytes: dict[int, bytes] = {}
        self._max_len = 1

    @property
    def base_vocab_size(self) -> int:
        return BYTE_VOCAB

    @property
    def vocab_size(self) -> int:
        return BYTE_VOCAB + len(self._macros)

    @property
    def macros(self) -> tuple[MacroSymbol, ...]:
        return tuple(self._macros)

    def register_macro(
        self, raw_bytes: bytes, *, birth_step: int = 0, lineage_tag: str = "chunk_growth"
    ) -> int:
        if not isinstance(raw_bytes, bytes) or len(raw_bytes) < 2:
            raise ValueError("Macro-symbol must span at least 2 raw bytes")
        if raw_bytes in self._bytes_to_id:
            raise ValueError(f"Duplicate macro-symbol for bytes {raw_bytes!r}")
        sym_id = BYTE_VOCAB + len(self._macros)
        sym = MacroSymbol(
            symbol_id=sym_id,
            raw_bytes=raw_bytes,
            birth_step=int(birth_step),
            lineage_tag=str(lineage_tag),
        )
        self._macros.append(sym)
        self._bytes_to_id[raw_bytes] = sym_id
        self._id_to_bytes[sym_id] = raw_bytes
        self._max_len = max(self._max_len, len(raw_bytes))
        return sym_id

    def encode_bytes(self, raw: bytes) -> list[int]:
        """Encode arbitrary bytes via longest macro match with lossless 0..255 fallback."""
        out: list[int] = []
        n = len(raw)
        i = 0
        while i < n:
            matched_id: int | None = None
            matched_len = 1
            if self._max_len >= 2:
                upper = min(self._max_len, n - i)
                for length in range(upper, 1, -1):
                    cand = raw[i : i + length]
                    sid = self._bytes_to_id.get(cand)
                    if sid is not None:
                        matched_id = sid
                        matched_len = length
                        break
            if matched_id is not None:
                out.append(matched_id)
                i += matched_len
            else:
                out.append(int(raw[i]))
                i += 1
        return out

    def decode_to_bytes(self, ids: Sequence[int]) -> bytes:
        """Decode symbol IDs back to exact UTF-8/raw bytes (ignoring control IDs >= EOS)."""
        buf = bytearray()
        for sid in ids:
            val = int(sid)
            if 0 <= val <= 255:
                buf.append(val)
            elif val in self._id_to_bytes:
                buf.extend(self._id_to_bytes[val])
            elif val in (EOS_BYTE, BYTE_VOCAB - 2, BYTE_VOCAB - 1):
                continue
            else:
                raise ValueError(f"Unknown symbol_id {val}")
        return bytes(buf)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "base_vocab_size": BYTE_VOCAB,
            "macros": [m.to_dict() for m in self._macros],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DynamicByteVocab:
        v = cls()
        for entry in data.get("macros", []):
            m = MacroSymbol.from_dict(entry)
            v.register_macro(m.raw_bytes, birth_step=m.birth_step, lineage_tag=m.lineage_tag)
        return v


class DynamicByteChunkEncoder(nn.Module):
    """Causal byte encoder with neutral multi-byte macro-symbol growth."""

    def __init__(self, d_model: int = 128, byte_emb_dim: int = 64) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.byte_emb_dim = int(byte_emb_dim)
        self.vocab = DynamicByteVocab()

        self.byte_emb = nn.Embedding(BYTE_VOCAB, self.byte_emb_dim)
        nn.init.normal_(self.byte_emb.weight, std=0.02)

        self.conv1 = nn.Conv1d(self.byte_emb_dim, self.d_model, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, dilation=2)
        self.conv3 = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, dilation=4)

        # Residual deltas for dynamically grown macro-symbols (0.0 at birth)
        self.macro_deltas = nn.Parameter(torch.zeros(0, self.d_model))

    def encode_raw_sequence(self, raw_ids: torch.Tensor) -> torch.Tensor:
        """Causal multi-scale Conv1d encoding for base byte IDs `(B, T)` -> `(B, T, D)`."""
        x = self.byte_emb(raw_ids).transpose(1, 2)
        h1 = torch.tanh(self.conv1(F.pad(x, (2, 0))))
        h2 = h1 + torch.tanh(self.conv2(F.pad(h1, (4, 0))))
        h3 = h2 + torch.tanh(self.conv3(F.pad(h2, (8, 0))))
        return cast(torch.Tensor, h3.transpose(1, 2))

    @torch.no_grad()
    def grow_macro_symbol_neutral_(
        self, raw_bytes: bytes, *, birth_step: int = 0, lineage_tag: str = "chunk_growth"
    ) -> dict[str, Any]:
        """Register a multi-byte macro-symbol with a zero residual delta (0.0 birth shift)."""
        sym_id = self.vocab.register_macro(
            raw_bytes, birth_step=birth_step, lineage_tag=lineage_tag
        )
        old = self.macro_deltas.detach()
        new_row = torch.zeros(1, self.d_model, device=old.device, dtype=old.dtype)
        self.macro_deltas = nn.Parameter(torch.cat([old, new_row], dim=0))
        return {
            "op": "grow_macro_symbol_neutral",
            "symbol_id": sym_id,
            "n_bytes": len(raw_bytes),
            "vocab_size": self.vocab.vocab_size,
        }

    def encode_symbol_stream(self, symbol_ids: Sequence[int]) -> torch.Tensor:
        """Encode mixed raw-byte + macro-symbol IDs, preserving exact byte equivalence at birth.

        Every symbol expands deterministically to its underlying byte spans `(L,)` encoded by the
        shared causal byte front-end; each macro-symbol adds its learnable `macro_deltas[idx]`
        residual (identically `0.0` at birth, specialized via gradients thereafter) onto its span.
        """
        expanded: list[int] = []
        delta_positions: list[tuple[int, int, int]] = []
        for sid in symbol_ids:
            val = int(sid)
            start = len(expanded)
            if val >= BYTE_VOCAB:
                raw_b = self.vocab.decode_to_bytes([val])
                expanded.extend(raw_b)
                end = len(expanded)
                delta_positions.append((start, end, val - BYTE_VOCAB))
            else:
                expanded.append(val)

        dev = self.byte_emb.weight.device
        raw_tensor = torch.tensor([expanded], dtype=torch.long, device=dev)
        feats = self.encode_raw_sequence(raw_tensor)  # (1, T_bytes, D)
        if not delta_positions:
            return feats

        delta_overlay = torch.zeros_like(feats)
        for start, end, m_idx in delta_positions:
            delta_overlay[0, start:end, :] = self.macro_deltas[m_idx].unsqueeze(0)
        return feats + delta_overlay


class EntropyBoundaryChunker:
    """Strictly causal variable-length byte chunker driven by per-byte entropy."""

    def __init__(
        self,
        *,
        entropy_threshold: float = 2.0,
        min_chunk: int = 2,
        max_chunk: int = 16,
    ) -> None:
        if min_chunk < 1 or max_chunk < min_chunk:
            raise ValueError("Require 1 <= min_chunk <= max_chunk")
        self.entropy_threshold = float(entropy_threshold)
        self.min_chunk = int(min_chunk)
        self.max_chunk = int(max_chunk)

    def segment(self, entropies: Sequence[float]) -> list[tuple[int, int]]:
        """Causally partition length-T entropies into non-empty spans `[start, end)`."""
        n = len(entropies)
        if n == 0:
            return []
        spans: list[tuple[int, int]] = []
        start = 0
        for idx in range(n):
            cur_len = idx - start + 1
            is_last = idx == n - 1
            trigger = cur_len >= self.min_chunk and float(entropies[idx]) >= self.entropy_threshold
            if is_last or trigger or cur_len >= self.max_chunk:
                spans.append((start, idx + 1))
                start = idx + 1
        return spans

    def pool_features(
        self, byte_features: torch.Tensor, entropies: Sequence[float]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Pool `(1, T, D)` byte features across causal variable-length spans."""
        if byte_features.ndim != 3 or byte_features.shape[0] != 1:
            raise ValueError("pool_features expects shape (1, T, D)")
        spans = self.segment(entropies)
        if not spans:
            raise ValueError("Cannot pool empty sequence")
        pooled = [byte_features[:, s:e, :].mean(dim=1, keepdim=True) for s, e in spans]
        meta = {
            "boundary_role": BOUNDARY_ROLE,
            "n_chunks": len(spans),
            "spans": spans,
        }
        return torch.cat(pooled, dim=1), meta
