"""Byte-level match route: selection readout for the zero-training assembly.

The gate asked whether a *learnable* pathway can express "read a query span, find
where the same byte string occurs inside one of N record lines, and score that
line".  This module is that pathway.  It is registered in
``candidate_identity.GRAPH_MODULES`` as of package 19.2t (zero-training assembly);
registration changes ``candidate_sha_v2`` by design, while the module's own
parameters are initialised at identity and are never trained.

Shape of the computation:

* one embedding table for query bytes and record bytes (tied), so a matched byte
  contributes ``<e_b, e_b>`` and a mismatched byte ``<e_b, e_b'>``;
* a learned weight per alignment slot (relative position), so the score is
  ``sum_p w_p * <e_{q_p}, e_{r_{o+p}}>`` for a window starting at offset ``o``;
* the line score is the best window over all offsets (the same operation is
  applied at every relative position), and ``argmax`` over the N lines is the
  selection.  The winning offset is returned too, because a second hop must read
  the name bytes adjacent to the matched description.

Two calling modes:

* ``align_offsets=None`` - the pathway searches every offset by itself;
* ``align_offsets`` given - the caller injects the alignment start per line
  (``-1`` means "this line does not contain the span").  The constructive probe
  uses this mode to prove expressibility without training, not to replace the
  pathway with a hard-coded ``str.find``: the matching arithmetic is still the
  module's own.

Parameters: ``byte_embed`` (257, 256) + ``position_weight`` (15,) = 65,807.
"""

from __future__ import annotations

import torch
from torch import nn

PAD_ID = 256
VOCAB = 257
IDENTITY_DIM = VOCAB - 1


class ByteMatchRoute(nn.Module):
    """Byte-exact matching over N record lines, parameterised and trainable."""

    def __init__(self, *, dim: int = IDENTITY_DIM, span: int = 15):
        super().__init__()
        if dim < IDENTITY_DIM:
            raise ValueError("identity initialisation needs dim >= vocab - 1")
        self.dim = dim
        self.span = span
        self.byte_embed = nn.Parameter(torch.zeros(VOCAB, dim))
        self.position_weight = nn.Parameter(torch.ones(span))
        with torch.no_grad():
            self.byte_embed.copy_(torch.eye(VOCAB, dim))

    def forward(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        record_ids: torch.Tensor,
        record_mask: torch.Tensor,
        align_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(scores, offsets, matched_value_bytes)``.

        ``query_ids`` is ``(B, S)``, ``record_ids`` is ``(B, N, L)``; both masks
        are boolean with the same shape.  ``align_offsets`` is ``(B, N)`` long.
        """
        batch, span = query_ids.shape
        lines = record_ids.shape[1]
        length = record_ids.shape[2]
        if span > self.span:
            raise ValueError(f"query span {span} exceeds declared span {self.span}")

        query = self.byte_embed[query_ids]
        record = self.byte_embed[record_ids]
        pair = torch.einsum("bsd,bnld->bsnl", query, record)
        pair = pair * self.position_weight[:span].view(1, span, 1, 1)
        pair = pair * query_mask.to(pair.dtype).view(batch, span, 1, 1)

        fill = torch.finfo(pair.dtype).min
        lengths = record_mask.sum(dim=2)
        row_alive = record_mask.any(dim=2)

        if align_offsets is not None:
            offset = align_offsets.to(torch.long)
            valid = (offset >= 0) & (offset + span <= lengths)
            index = offset.view(batch, lines, 1) + torch.arange(span, device=query_ids.device).view(
                1, 1, span
            )
            index = index.clamp(0, length - 1)
            window = record.gather(2, index.unsqueeze(-1).expand(batch, lines, span, self.dim))
            gathered = pair.gather(
                3, index.view(batch, 1, lines, span).expand(batch, span, lines, span)
            )
            aligned = torch.diagonal(gathered, dim1=1, dim2=3)
            score = (aligned * query_mask.view(batch, 1, span)).sum(dim=2)
            score = torch.where(valid & row_alive, score, fill)
            chosen = torch.where(valid, offset, torch.full_like(offset, -1))
            return score, chosen, window

        accumulator = torch.zeros(batch, lines, length, dtype=pair.dtype, device=pair.device)
        for position in range(span):
            accumulator[:, :, : length - position] += pair[:, position, :, position:length]
        offsets = torch.arange(length, device=query_ids.device).view(1, 1, length)
        window_valid = (offsets + span <= lengths.view(batch, lines, 1)) & row_alive.view(
            batch, lines, 1
        )
        accumulator = torch.where(window_valid, accumulator, fill)
        score, chosen = accumulator.max(dim=2)
        index = chosen.view(batch, lines, 1) + torch.arange(span, device=query_ids.device).view(
            1, 1, span
        )
        index = index.clamp(0, length - 1)
        window = record.gather(2, index.unsqueeze(-1).expand(batch, lines, span, self.dim))
        return score, chosen, window
