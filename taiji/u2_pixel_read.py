"""Pixel-salience readout unit: query-conditioned salience over the 16x16 raster.

Born neutral: the projection into the readout space is zero-initialised, so a
model that grows this unit reproduces its parent bit for bit until the unit is
credited by the task loss.  Nothing here imports model state; wiring lives in
``U2Subject`` behind the ``readout`` config switch.

The operator is the one measured by the 22.5.0.A probes: a multiplicative match
between a query self-encoded from the request bytes and each pixel's colour,
followed by a linear read of the weighted coordinate moments.
"""

from __future__ import annotations

import math

import torch
from torch import nn

SIDE = 16
MAX_BYTES = 64
UNIT_IDS = ("pixel-salience:substrate", "pixel-salience:query", "pixel-salience:match")


def pixel_grid(coordinates: torch.Tensor) -> torch.Tensor:
    """Expand 16 patch centres to the 16x16 pixel grid with the fixed sub-offsets."""
    centres = coordinates[:, :2]
    offsets = torch.tensor([-3 / 32, -1 / 32, 1 / 32, 3 / 32], dtype=coordinates.dtype)
    rows = []
    for index in range(SIDE):
        span_x = (centres[index][0] + offsets).reshape(-1, 1).expand(4, 4)
        span_y = (centres[index][1] + offsets).reshape(1, -1).expand(4, 4)
        rows.append(torch.stack([span_x, span_y], -1))
    return torch.stack(rows).reshape(4, 4, 4, 4, 2).permute(0, 2, 1, 3, 4).reshape(SIDE, SIDE, 2)


def raster(blocks: torch.Tensor) -> torch.Tensor:
    """Resplit 16 patch blocks (4x4 pixels x 3 channels) into a 16x16x3 raster."""
    count = blocks.shape[0]
    view = blocks.reshape(count, 4, 4, 4, 4, 3)
    return view.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(count, SIDE, SIDE, 3)


class PixelSalienceReadout(nn.Module):
    """Three-unit family: substrate (pixels), query (bytes), match (read)."""

    def __init__(self, width: int = 16, text: int = 16, out_dim: int = 8) -> None:
        super().__init__()
        self.byte = nn.Embedding(256, text)
        self.position = nn.Parameter(torch.zeros(1, MAX_BYTES, text))
        self.text = nn.Linear(2 * text, width)
        self.query = nn.Linear(width, width)
        self.pixel = nn.Linear(3, width)
        self.null = nn.Parameter(torch.zeros(width))
        self.out = nn.Linear(6, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, blocks: torch.Tensor, pixel_xy: torch.Tensor, request: torch.Tensor
    ) -> torch.Tensor:
        image = raster(blocks if blocks.dim() == 3 else blocks[None])
        flat = image.reshape(image.shape[0], SIDE * SIDE, 3)
        key = torch.tanh(self.pixel(flat))
        text = self.byte(request) + self.position[0, : request.numel()]
        question = torch.tanh(
            self.query(torch.tanh(self.text(torch.cat([text.mean(0), text.amax(0)], -1))))
        )
        score = torch.cat(
            [
                torch.einsum("d,nkd->nk", question, key) / math.sqrt(key.shape[-1]),
                (question @ self.null).reshape(1, 1).expand(image.shape[0], -1),
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
        return self.out(features)


def unit_registry(width: int = 16, text: int = 16, out_dim: int = 8) -> dict:
    """Parameter owners for the birth ledger, keyed by the three unit ids."""
    return {
        "pixel-salience:substrate": ["pixel.weight", "pixel.bias", "out.weight", "out.bias"],
        "pixel-salience:query": [
            "byte.weight",
            "position",
            "text.weight",
            "text.bias",
            "query.weight",
            "query.bias",
        ],
        "pixel-salience:match": ["null"],
    }
