"""Closed-form value readout from producer features for the N177 E1 slice.

The readout is a trainable linear layer. E2b-2 calibrates it in closed form on
the eight public colour prototypes so the zero-step control matrix can be read
without an optimizer update. F may continue training the same parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .joint_state_consumption_course import E1_COLORS

VALUE_READOUT_FORMAT = "taiji-n177-e1-value-readout-v1"


@dataclass(frozen=True)
class ValueReadoutResultV1:
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    selected_index: int
    selected_value: str

    FORMAT = VALUE_READOUT_FORMAT


class FeatureValueReadoutV1(nn.Module):
    """Linear value head over a selected content feature (and optional query state)."""

    def __init__(self, *, feature_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim < 0:
            raise ValueError("readout dimensions must be positive")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(feature_dim + hidden_dim, len(E1_COLORS))
        self.values = E1_COLORS

    def forward(
        self,
        feature: torch.Tensor,
        query_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vector = feature.reshape(-1)
        if self.hidden_dim:
            if query_hidden is None:
                raise ValueError("readout requires the query hidden state")
            vector = torch.cat([query_hidden.reshape(-1), vector])
        elif query_hidden is not None:
            raise ValueError("readout was built without a query hidden input")
        if vector.numel() != self.linear.in_features:
            raise ValueError("readout input width differs from its declared dimensions")
        return self.linear(vector)

    def read(
        self,
        feature: torch.Tensor,
        query_hidden: torch.Tensor | None = None,
    ) -> ValueReadoutResultV1:
        logits = self.forward(feature, query_hidden)
        probabilities = torch.softmax(logits, dim=0)
        index = int(torch.argmax(logits))
        return ValueReadoutResultV1(
            logits=tuple(float(item) for item in logits.tolist()),
            probabilities=tuple(float(item) for item in probabilities.tolist()),
            selected_index=index,
            selected_value=self.values[index],
        )

    @classmethod
    def fit_closed_form(
        cls,
        features: Sequence[torch.Tensor],
        labels: Sequence[int],
        *,
        query_hiddens: Sequence[torch.Tensor] | None = None,
        ridge: float = 1e-3,
    ) -> FeatureValueReadoutV1:
        """Ridge-fit the value head without an optimizer step."""

        if len(features) != len(labels):
            raise ValueError("readout features and labels differ in length")
        if len(features) == 0:
            raise ValueError("readout fit requires at least one prototype")
        if ridge <= 0:
            raise ValueError("readout ridge penalty must be positive")
        rows = [torch.as_tensor(item, dtype=torch.float32).reshape(-1) for item in features]
        feature_dim = rows[0].numel()
        if any(row.numel() != feature_dim for row in rows):
            raise ValueError("readout prototype feature widths differ")
        hidden_dim = 0
        if query_hiddens is not None:
            if len(query_hiddens) != len(rows):
                raise ValueError("query hidden states differ in length from prototypes")
            hidden_rows = [
                torch.as_tensor(item, dtype=torch.float32).reshape(-1) for item in query_hiddens
            ]
            hidden_dim = hidden_rows[0].numel()
            if any(row.numel() != hidden_dim for row in hidden_rows):
                raise ValueError("query hidden state widths differ")
            rows = [
                torch.cat([hidden, feature])
                for hidden, feature in zip(hidden_rows, rows, strict=True)
            ]
        matrix = torch.stack(rows)
        targets = torch.zeros((len(labels), len(E1_COLORS)), dtype=torch.float32)
        for row, label in enumerate(labels):
            if not 0 <= label < len(E1_COLORS):
                raise ValueError("readout label is outside the frozen colour set")
            targets[row, label] = 1.0
        augmented = torch.cat(
            [matrix, torch.ones((matrix.shape[0], 1), dtype=torch.float32)], dim=1
        )
        penalty = ridge * torch.eye(augmented.shape[1], dtype=torch.float32)
        penalty[-1, -1] = 0.0
        weights = torch.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ targets)
        readout = cls(feature_dim=feature_dim, hidden_dim=hidden_dim)
        with torch.no_grad():
            readout.linear.weight.copy_(weights[:-1].T)
            readout.linear.bias.copy_(weights[-1])
        return readout
