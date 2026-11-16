"""In-body column read (``column-read-v1``).

A.3 measured that the subject does not carry the scene read before it answers:
at prefix 0 the read class was 0.4375 against a shuffle null of 0.34375, the
painted-column colour probe was 0.0154, and the higher prefixes were teacher
forced leakage. Options A and B only made the pipeline answer; option C was
closed because no read existed inside the body.

This module is that read. The request embedding is the query, the body's own
column tokens are keys and values, and the attention over the sixteen columns
answers "which column holds the asked colour". Two facts the earlier line had to
learn the hard way are built in here rather than rediscovered:

* **position must leave the pooled summary.** A pooled value cannot say where, so
  the attended column coordinate travels as its own feature (P3.21/P3.22 lesson).
* **no free abstention slot.** "Unknown" is one of four classes a head has to
  earn from features (P3.22/P3.26 lesson); there is no parameter that can push
  the answer to "unknown" without matching something.
* **the class head must be on the byte path.** A first version injected only the
  features, which left the four-class head outside the gradient path of the byte
  loss: it could never be trained, and any read accuracy measured on it would
  have been an accident. The class distribution is therefore injected as well, so
  the byte loss earns the class. The residual projection is zero-initialised,
  which means the head's gradient is exactly zero on the first step and
  non-zero from the second on - that is a declared property, not a defect.

The read enters the subject at the joint readout, additively, through a
zero-initialised projection. At initialisation the byte path is therefore
bit-identical to the body without the read, so nothing regresses before training
starts, and the first step is a true starting point rather than a perturbation.

It enters on the **input** side of the readout, not the output side. That is the
second version of this design and the reason is measurement: a reader probes the
decision point, which is the readout's *input*, and A.3 measured that this input
does not carry a scene read (0.4375 against a shuffle null of 0.34375, painted
colour probe 0.0154). A read added to the output would leave that probe exactly
where it was, so the card's own gate could never be met. The delta lands on the
last eight of the twenty-four input values, which is the body's own fast slice.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch
from torch import nn

from .u2_observation_columns import column_tokens

IDENTITY = "column-read-v1"
QUERY = 8
TOKEN = 8
COLUMNS = 16
CLASSES = ("左边", "中间", "右边", "未知")
FEATURES = 11  # position, strength, entropy, pooled content (8)
INJECT = FEATURES + 4  # the class distribution joins the injected vector
READOUT_WIDTH = 24  # 3 * d: the joint readout input of the registered body


def column_positions() -> torch.Tensor:
    """Public x coordinate of every pixel column."""
    return 2 * (torch.arange(COLUMNS) + 0.5) / COLUMNS - 1


class ColumnRead(nn.Module):
    """Query-conditioned attention over the body's column tokens."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.width = width
        self.query = nn.Linear(QUERY, width)
        self.key = nn.Linear(TOKEN, width)
        self.value = nn.Linear(TOKEN, width)
        self.head = nn.Sequential(nn.Linear(FEATURES, width), nn.ReLU(), nn.Linear(width, 4))
        self.out = nn.Linear(INJECT, width)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("positions", column_positions(), persistent=False)

    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.shape != (COLUMNS, TOKEN):
            raise ValueError(f"column_read_expects_16x8_got_{tuple(tokens.shape)}")
        if query.shape != (QUERY,):
            raise ValueError(f"column_read_expects_query_8_got_{tuple(query.shape)}")
        projected_query = self.query(query)
        keys = self.key(tokens)
        values = self.value(tokens)
        scores = (keys * projected_query).sum(-1) / math.sqrt(self.width)
        attention = scores.softmax(-1)
        pooled = (attention[:, None] * values).sum(0)
        position = (attention * self.positions).sum()
        strength = attention.max()
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum()
        features = torch.cat([position[None], strength[None], entropy[None], pooled])
        logits = self.head(features)
        probabilities = logits.softmax(-1)
        return {
            "attention": attention,
            "pooled": pooled,
            "position": position,
            "strength": strength,
            "entropy": entropy,
            "features": features,
            "logits": logits,
            "probabilities": probabilities,
            "injection": torch.cat([features, probabilities]),
        }


def request_query(model: Any, example: dict) -> torch.Tensor:
    """The body's own request embedding, averaged over its units."""
    return cast(torch.Tensor, model.text_units(example["public"]["request"]).mean(0))


def read_for_example(model: Any, read: ColumnRead, example: dict) -> dict[str, torch.Tensor]:
    """Run the read on one example using the body's own encoder over columns."""
    tokens = column_tokens(
        model, example["public"]["observations"], example["public"]["coordinates"]
    )
    return cast(dict[str, torch.Tensor], read(tokens, request_query(model, example)))


def read_delta(read: ColumnRead, features: torch.Tensor) -> torch.Tensor:
    """The read's contribution to the joint readout input (eight dims)."""
    return cast(torch.Tensor, read.out(features))


def patch_source(source: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Add the read delta to the fast slice of a joint readout input."""
    if source.shape[-1] != READOUT_WIDTH:
        raise ValueError(f"unexpected_joint_readout_width_{source.shape[-1]}")
    patched = source.clone()
    patched[..., -8:] = patched[..., -8:] + delta.to(source.dtype)
    return patched


@contextmanager
def inject_read(model: Any, read: ColumnRead, provider) -> Iterator[None]:
    """Add the read to the joint readout input for the duration of the block.

    ``provider`` is called with no arguments and must return the injection vector
    of the example currently being answered - the eleven features plus the four
    class probabilities. It is called once per joint readout call, so a cached
    per-example read is the cheap way to use it.
    """

    def pre(module, inputs):  # noqa: ANN001 - torch hook signature
        source = inputs[0]
        features = provider()
        if features is None:
            return None
        return (patch_source(source, read_delta(read, features)),) + tuple(inputs[1:])

    handle = model.joint_readout.register_forward_pre_hook(pre)
    try:
        yield
    finally:
        handle.remove()


def parameter_census(read: ColumnRead) -> dict[str, Any]:
    """Named tensors and scalar count, for the contract to pin."""
    names = [name for name, _ in read.named_parameters()]
    scalars = sum(parameter.numel() for parameter in read.parameters())
    return {"names": names, "scalars": scalars, "tensors": len(names)}
