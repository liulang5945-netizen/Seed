"""Column-structured observation pathway (``column-observation-v1``).

The native body encodes the rgb observation as sixteen 4x4 patches: one token per
patch, each mixing four pixel columns and four rows. A column is therefore not a
thing the body can represent, and P3.11-P3.16 measured that the pooled 8-dim
encoding cannot carry "where is the queried colour".

A pixel column is also 48 values (16 rows x 3 channels), the same width as a
patch, so the existing ``encoder.rgb`` (48 -> 8) can consume columns with no new
parameters. This module is that regrouping and nothing else:

  * no parameters, no buffers, nothing that enters ``U2Subject.state_dict()``;
  * every registered bundle keeps its ``parameter_sha256`` and loads unchanged;
  * the pathway carries its own identity string so a reader can tell which
    observation layout produced a token.

The feasibility probe that motivated it (7-way colour decodability per position,
ridge fit on train, tested on fit): patch tokens 0.385/0.387, column tokens
0.690/0.674, raw pixels 1.000. Regrouping nearly doubles it and still leaves the
token lossy, which is why the card that follows treats "tokens too lossy" as a
live outcome rather than a failure to explain away.
"""

from __future__ import annotations

import torch

from .u2_subject import bounded

IDENTITY = "column-observation-v1"
SIDE = 16
WIDTH = 48


def column_values(patches: torch.Tensor) -> torch.Tensor:
    """[16, 48] patch values -> [16, 48] column values.

    Patch index is ``block_row * 4 + block_column`` and each patch stores its
    4x4x3 pixels in ``(row, column, channel)`` order. Column ``k`` is block column
    ``k // 4`` and within-block column ``k % 4``, read across all four block rows.
    """
    if patches.shape != (16, WIDTH):
        raise ValueError(f"column_values_expects_16x48_got_{tuple(patches.shape)}")
    five = patches.reshape(4, 4, 4, 4, 3)
    return five.permute(1, 3, 0, 2, 4).reshape(16, WIDTH)


def column_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    """[16, 8] patch coordinates -> [16, 8] column coordinates.

    Slot 0 is x and slot 1 is y; a column sits at the pixel-column centre with
    y = 0. The remaining slots describe the frame and are copied unchanged.
    """
    if coordinates.shape != (16, 8):
        raise ValueError(f"column_coordinates_expects_16x8_got_{tuple(coordinates.shape)}")
    result = coordinates.clone()
    x = 2 * (torch.arange(SIDE, dtype=coordinates.dtype) + 0.5) / SIDE - 1
    result[:, 0] = x
    result[:, 1] = 0
    return result


def column_tokens(subject, observations, coordinates) -> torch.Tensor:
    """Run the body's own rgb encoder over columns and return [16, 8] tokens.

    Uses ``subject.encoder["rgb"]``, ``subject.metadata`` and ``subject.modality``
    exactly as ``U2Subject.encode`` does, so the only difference is the grouping.
    """
    patches = observations["rgb"]
    values = column_values(patches)
    frame = column_coordinates(coordinates["rgb"])
    return bounded(  # type: ignore[return-value]
        subject.encoder["rgb"](values) + subject.metadata(frame) + subject.modality[0]
    )


def column_index_of(column: int) -> float:
    """Public x coordinate of a pixel column, matching ``position_word``'s axis."""
    return 2 * (column + 0.5) / SIDE - 1


def class_of_column(column: int) -> str:
    """The label ``position_word`` would give to that column's centre."""
    x = column_index_of(column)
    if x < -1 / 3:
        return "左边"
    if x > 1 / 3:
        return "右边"
    return "中间"
