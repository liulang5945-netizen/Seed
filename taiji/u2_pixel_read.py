"""Pixel-salience readout unit: query-conditioned salience over the 16x16 raster.

Born neutral: the projection into the readout space is zero-initialised, so a
model that grows this unit reproduces its parent bit for bit until the unit is
credited by the task loss.  Nothing here imports model state; wiring lives in
``U2Subject`` behind the ``readout`` config switch.

Two frozen identities live here and must not be conflated:

* ``pixel-salience`` (v1) is the operator that earned ``a_parent_earned`` in
  the 22.5.0 A-parent package.  Its code, parameter names, shapes and
  initialisation are bit-for-bit the pre-S0 revision, so the accepted parent's
  generation chain can be rebuilt and its bundle loaded (S0 / S1 discipline).
* ``pixel-salience-v2`` is the S0 architecture revision: four-head
  query-conditioned significance with a sinusoidal position basis, learnable
  query gate, LayerNorm text pooling, centre-surround chroma features, a
  learnable null slot and RMS feature normalisation.  It is a *new identity*:
  it shares no checkpoint with v1 and must earn its own capability evidence.

Both operators follow the same measured principle: a multiplicative match
between a query self-encoded from the request bytes and each pixel's visual
description, followed by a linear read of the weighted coordinate moments.
The concrete parameterisation is replaceable; the identity name is part of the
capability ledger, so it is never renamed in place.
"""

from __future__ import annotations

import math

import torch
from torch import nn

SIDE = 16
MAX_BYTES = 64
READOUT_V1 = "pixel-salience"
READOUT_V2 = "pixel-salience-v2"
READOUT_IDS = (READOUT_V1, READOUT_V2)
UNIT_IDS = ("pixel-salience:substrate", "pixel-salience:query", "pixel-salience:match")
UNIT_IDS_V2 = (
    "pixel-salience-v2:substrate",
    "pixel-salience-v2:query",
    "pixel-salience-v2:match",
)


def pixel_grid(coordinates: torch.Tensor) -> torch.Tensor:
    """Expand 16 patch centres to the 16x16 pixel grid with the fixed sub-offsets.

    Vectorised into one broadcast; patch order and the in-patch (i x j) sub-grid
    match how ``encode`` reshaped the raster.
    """
    centres = coordinates[:, :2]
    offsets = torch.tensor([-3 / 32, -1 / 32, 1 / 32, 3 / 32], dtype=coordinates.dtype)
    # [patch, row, col]: x changes with the column, y with the row.
    span_x = centres[:, 0, None, None] + offsets[None, None, :]
    span_y = centres[:, 1, None, None] + offsets[None, :, None]
    grid = torch.stack([span_x.expand(-1, 4, 4), span_y.expand(-1, 4, 4)], -1)
    return grid.reshape(4, 4, 4, 4, 2).permute(0, 2, 1, 3, 4).reshape(SIDE, SIDE, 2)


def raster(blocks: torch.Tensor) -> torch.Tensor:
    """Expand ``(16,48)`` or ``(B,16,48)`` patch blocks into a raster."""
    squeeze = blocks.dim() == 2
    if squeeze:
        if tuple(blocks.shape) != (16, 48):
            raise ValueError("pixel_blocks_shape")
        blocks = blocks[None]
    elif blocks.dim() != 3 or tuple(blocks.shape[1:]) != (16, 48):
        raise ValueError("pixel_blocks_shape")
    count = blocks.shape[0]
    view = blocks.reshape(count, 4, 4, 4, 4, 3)
    image = view.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(count, SIDE, SIDE, 3)
    return image[0] if squeeze else image


class PixelSalienceReadout(nn.Module):
    """Three-unit family: substrate (pixels), query (bytes), match (read).

    Frozen v1 identity: the exact operator behind the accepted 22.5.0 A parent.
    Do not edit its parameters, shapes or initialisation; the S1 rebuild check
    and any historical checkpoint depend on this byte layout.
    """

    def __init__(self, width: int = 16, text: int = 16, out_dim: int = 8) -> None:
        super().__init__()
        self.byte = nn.Embedding(256, text)
        self.position = nn.Parameter(torch.zeros(1, MAX_BYTES, text))
        self.text = nn.Linear(2 * text, width)
        self.pixel = nn.Linear(3, width)
        self.null = nn.Parameter(torch.zeros(width))
        self.log_temp = nn.Parameter(torch.zeros(()))
        self.out = nn.Linear(6, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, blocks: torch.Tensor, pixel_xy: torch.Tensor, request: torch.Tensor
    ) -> torch.Tensor:
        if request.ndim != 1:
            raise ValueError("pixel_request_vector_required")
        if request.numel() == 0:
            raise ValueError("pixel_readout_empty_request")
        if request.numel() > MAX_BYTES:
            raise ValueError("pixel_readout_request_too_long")
        image = raster(blocks)
        if image.dim() == 3:
            image = image[None]
        flat = image.reshape(image.shape[0], SIDE * SIDE, 3)
        key = nn.functional.normalize(self.pixel(flat), dim=-1)
        text = self.byte(request) + self.position[0, : request.numel()]
        question = nn.functional.normalize(
            self.text(torch.cat([text.mean(0), text.amax(0)], -1)), dim=-1
        )
        scale = math.sqrt(key.shape[-1]) * self.log_temp.clamp(0.0, 2.7726).exp()
        score = torch.cat(
            [
                torch.einsum("d,nkd->nk", question, key) * scale,
                (question @ nn.functional.normalize(self.null, dim=-1))
                .reshape(1, 1)
                .expand(image.shape[0], -1),
            ],
            -1,
        )
        weight = score.softmax(-1)[:, :-1]
        mass = weight.sum(-1)
        x = pixel_xy[..., 0].reshape(1, -1)
        y = pixel_xy[..., 1].reshape(1, -1)
        moment_x = (weight * x).sum(-1)
        moment_y = (weight * y).sum(-1)
        features = torch.stack(
            [
                moment_x / mass.clamp(min=1e-6),
                moment_y / mass.clamp(min=1e-6),
                mass,
                weight.amax(-1),
                moment_x,
                moment_y,
            ],
            -1,
        )
        result: torch.Tensor = self.out(features)
        return result


def _sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    half = dim // 2
    freq = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(100.0) / max(half, 1)))
    angles = positions * freq[None, :]
    basis = torch.zeros(length, dim, dtype=torch.float32)
    basis[:, 0 : 2 * half : 2] = angles.sin()
    basis[:, 1 : 2 * half : 2] = angles.cos()
    return basis * 0.25


class PixelSalienceReadoutV2(nn.Module):
    """S0 revision: four-head query-conditioned salience with general visual features.

    New identity ``pixel-salience-v2``.  Born neutral through the zero-initialised
    ``out`` projection; the rest of the operator is a direct multiplicative match
    (query x pixel) with a learned null slot, so salience can be credited by the
    task loss without labels, teachers or hardcoded palette colours.
    """

    pos_basis: torch.Tensor

    def __init__(self, width: int = 16, text: int = 16, out_dim: int = 8, heads: int = 4) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.byte = nn.Embedding(256, text)
        self.register_buffer("pos_basis", _sinusoidal_positions(MAX_BYTES, text), persistent=False)
        self.position = nn.Parameter(torch.zeros(1, MAX_BYTES, text))
        self.query_gate = nn.Linear(text, heads)
        self.text_norm = nn.LayerNorm(3 * text)
        self.text = nn.Linear(3 * text, heads * width)
        self.pixel = nn.Linear(8, heads * width)
        self.pixel_prior = nn.Linear(8, heads)
        nn.init.zeros_(self.pixel_prior.weight)
        nn.init.zeros_(self.pixel_prior.bias)
        self.null = nn.Parameter(torch.randn(heads, width) / math.sqrt(width))
        self.null_bias = nn.Parameter(torch.zeros(heads))
        self.log_temp = nn.Parameter(torch.zeros(heads))
        self.feat_scale = nn.Parameter(torch.ones(10 * heads))
        self.out = nn.Linear(10 * heads, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, blocks: torch.Tensor, pixel_xy: torch.Tensor, request: torch.Tensor
    ) -> torch.Tensor:
        if request.ndim != 1:
            raise ValueError("pixel_request_vector_required")
        if request.numel() == 0:
            raise ValueError("pixel_readout_empty_request")
        if request.numel() > MAX_BYTES:
            raise ValueError("pixel_readout_request_too_long")
        image = raster(blocks)
        if image.dim() == 3:
            image = image[None]
        batch_n = image.shape[0]
        flat = image.reshape(batch_n, SIDE * SIDE, 3)
        # Center-surround color contrast + chroma + luminance: general visual
        # features so foreground objects are immediately distinguishable from
        # any uniform background without hardcoding palette colors.
        centered = flat - flat.mean(dim=1, keepdim=True)
        chroma = flat.amax(dim=-1, keepdim=True) - flat.amin(dim=-1, keepdim=True)
        lum = flat.mean(dim=-1, keepdim=True)
        pix_feat = torch.cat([flat, centered, chroma, lum], dim=-1)
        key = nn.functional.normalize(
            self.pixel(pix_feat).reshape(batch_n, SIDE * SIDE, self.heads, self.width),
            dim=-1,
            eps=1e-6,
        )  # [N, 256, H, W]
        prior = self.pixel_prior(pix_feat).permute(0, 2, 1)  # [N, H, 256]

        embedded = self.byte(request)
        pos = self.pos_basis[: request.numel()].to(dtype=embedded.dtype, device=embedded.device)
        text = embedded + pos + self.position[0, : request.numel()]  # [L, text]
        gate = self.query_gate(text).softmax(dim=0)  # [L, H]
        attended = torch.einsum("lh,lt->ht", gate, text).mean(dim=0)  # [text]
        pooled = torch.cat([attended, text.mean(0), text.amax(0)], -1)
        question = nn.functional.normalize(
            self.text(self.text_norm(pooled)).reshape(self.heads, self.width),
            dim=-1,
            eps=1e-6,
        )  # [H, W]

        scale = math.sqrt(self.width) * self.log_temp.clamp(0.0, 2.7726).exp()  # [H]
        pixel_cos = torch.einsum("hw,nkhw->nhk", question, key)  # [N, H, 256]
        unit_null = nn.functional.normalize(self.null, dim=-1, eps=1e-6)
        null_cos = (question * unit_null).sum(dim=-1) + self.null_bias  # [H]
        score = torch.cat(
            [
                pixel_cos * scale[None, :, None] + prior,
                (null_cos * scale)[None, :, None].expand(batch_n, -1, 1),
            ],
            dim=-1,
        )  # [N, H, 257]
        weight = score.softmax(dim=-1)[:, :, :-1]  # [N, H, 256]
        mass = weight.sum(dim=-1)  # [N, H]
        x = pixel_xy[..., 0].reshape(1, 1, -1)
        y = pixel_xy[..., 1].reshape(1, 1, -1)
        moment_x = (weight * x).sum(dim=-1)
        moment_y = (weight * y).sum(dim=-1)
        mean_x = moment_x / mass.clamp(min=1e-6)
        mean_y = moment_y / mass.clamp(min=1e-6)
        var_x = (weight * (x - mean_x[:, :, None]).square()).sum(dim=-1) / mass.clamp(min=1e-6)
        max_cos = pixel_cos.amax(dim=-1)
        snr_cos = max_cos - pixel_cos.mean(dim=-1)
        abs_x = mean_x.abs()
        features = torch.stack(
            [
                mean_x,
                mean_y,
                mass,
                weight.amax(dim=-1),
                moment_x,
                moment_y,
                max_cos,
                snr_cos,
                var_x,
                abs_x,
            ],
            dim=-1,
        ).reshape(batch_n, 10 * self.heads)
        rms = (features.square().mean(dim=-1, keepdim=True) + 1e-5).sqrt()
        normed = (features / rms) * self.feat_scale
        result: torch.Tensor = self.out(normed)
        return result


def unit_registry(width: int = 16, text: int = 16, out_dim: int = 8) -> dict:
    """Parameter owners for the birth ledger of the frozen v1 identity."""
    return {
        "pixel-salience:substrate": ["pixel.weight", "pixel.bias", "out.weight", "out.bias"],
        "pixel-salience:query": [
            "byte.weight",
            "position",
            "text.weight",
            "text.bias",
        ],
        "pixel-salience:match": ["null", "log_temp"],
    }


def unit_registry_v2(width: int = 16, text: int = 16, out_dim: int = 8) -> dict:
    """Parameter owners for the birth ledger of the S0 v2 identity."""
    return {
        "pixel-salience-v2:substrate": [
            "pixel.weight",
            "pixel.bias",
            "pixel_prior.weight",
            "pixel_prior.bias",
            "feat_scale",
            "out.weight",
            "out.bias",
        ],
        "pixel-salience-v2:query": [
            "byte.weight",
            "position",
            "query_gate.weight",
            "query_gate.bias",
            "text_norm.weight",
            "text_norm.bias",
            "text.weight",
            "text.bias",
        ],
        "pixel-salience-v2:match": ["null", "null_bias", "log_temp"],
    }


def build_readout(identity: str, out_dim: int) -> tuple[nn.Module, dict]:
    """Return (module, birth-ledger owners) for one registered identity."""
    if identity == READOUT_V1:
        return PixelSalienceReadout(out_dim=out_dim), unit_registry()
    if identity == READOUT_V2:
        return PixelSalienceReadoutV2(out_dim=out_dim), unit_registry_v2()
    raise ValueError("Invalid readout identity")
