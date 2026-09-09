"""Conditional route learner for M4.V2.R5 (resource-normalized modularity).

The learner maps a whitelist of route inputs — content bucket, predictive
context norm, candidate activity norm, surprise, and a resource scalar — to
the gate of an adaptive residual candidate.  It is the minimal
implementation of the pre-registered conditional-modularity hypothesis:
instead of one always-open residual population, the candidate receives
capacity conditioned on content/state, and the route itself is a small
locally-trained linear map.

Route inputs are whitelist-only (M4V2_R5_CONDITIONAL_MODULARITY
_PREREGISTRATION_20260909.md §2): evaluator task IDs, phase labels, file
names, or any evaluation-side signal are rejected by construction because
the learner only exposes numeric slots for the whitelist.
"""

from __future__ import annotations

import math
from typing import Any

import torch

CONDITIONAL_ROUTE_LEARNER_FORMAT = "taiji-conditional-route-learner-v1"
CONDITIONAL_ROUTE_LEARNER_VERSION = 1
CONTENT_BUCKETS = 4
INPUT_FEATURES = 4 + CONTENT_BUCKETS


class ConditionalRouteLearner:
    """A small, locally-trained linear gate on the candidate route.

    The gate is ``sigmoid(w . x + b)`` with an explicit feature vector built
    only from the whitelist.  Training uses a local, auditable rule: when the
    observed surprise is above its running mean, the inputs active at that
    moment are pushed to raise the gate; below the mean they are pushed to
    lower it.  No autograd, no global objective, no evaluation signal.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        ema_rate: float = 0.05,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("route learning_rate must be finite and positive")
        if not math.isfinite(ema_rate) or not 0.0 < ema_rate <= 1.0:
            raise ValueError("route ema_rate must be finite in (0, 1]")
        self.learning_rate = float(learning_rate)
        self.ema_rate = float(ema_rate)
        self.device = torch.device(device)
        generator = generator if generator is not None else torch.Generator(device="cpu")
        self.weight = torch.randn(INPUT_FEATURES, generator=generator) * 0.01
        self.bias = torch.zeros(1)
        self.error_baseline = 0.0
        self._lesioned = False
        self._last_input = torch.zeros(INPUT_FEATURES)
        self._last_gate = 0.0
        self._gate_history: list[float] = []

    @property
    def lesioned(self) -> bool:
        return self._lesioned

    @property
    def last_gate(self) -> float:
        return self._last_gate

    @property
    def gate_std(self) -> float:
        if len(self._gate_history) < 2:
            return 0.0
        values = torch.tensor(self._gate_history)
        return float(values.std(unbiased=False).item())

    def lesion(self) -> None:
        self._lesioned = True

    def unlesion(self) -> None:
        self._lesioned = False

    def reset_history(self) -> None:
        self._gate_history.clear()

    @staticmethod
    def content_bucket(content_digest: str) -> int:
        """Deterministic content feature derived from the record digest."""
        return int(content_digest[:8], 16) % CONTENT_BUCKETS

    def _features(
        self,
        *,
        content_bucket: int,
        surprise: float,
        context_norm: float,
        activity_norm: float,
        resource: float,
    ) -> torch.Tensor:
        if not 0 <= content_bucket < CONTENT_BUCKETS:
            raise ValueError("content bucket out of range")
        values = [surprise, context_norm, activity_norm, resource]
        for value in values[:3]:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("route inputs must be finite and non-negative")
        if not math.isfinite(resource):
            raise ValueError("route resource input must be finite")
        onehot = [0.0] * CONTENT_BUCKETS
        onehot[content_bucket] = 1.0
        return torch.tensor(values + onehot, dtype=torch.float32)

    def route(
        self,
        *,
        content_bucket: int,
        surprise: float,
        context_norm: float,
        activity_norm: float,
        resource: float,
    ) -> float:
        """Compute the gate for one step and remember the feature vector."""
        x = self._features(
            content_bucket=content_bucket,
            surprise=surprise,
            context_norm=context_norm,
            activity_norm=activity_norm,
            resource=resource,
        )
        self._last_input = x
        if self._lesioned:
            self._last_gate = 0.0
        else:
            logit = float(torch.dot(self.weight, x) + self.bias[0])
            self._last_gate = 1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0)))
        self._gate_history.append(self._last_gate)
        if len(self._gate_history) > 4096:
            self._gate_history.pop(0)
        return self._last_gate

    def learn(self, surprise: float) -> float:
        """Local rule: surprise above baseline raises the gate for the
        active input pattern; below baseline lowers it.  Returns the delta
        applied to the logit."""
        if self._lesioned:
            return 0.0
        if not math.isfinite(surprise) or surprise < 0.0:
            raise ValueError("route surprise input must be finite and non-negative")
        delta = float(surprise) - self.error_baseline
        self.error_baseline += self.ema_rate * (delta)
        logit_delta = self.learning_rate * delta
        with torch.no_grad():
            self.weight += logit_delta * self._last_input
            self.bias += logit_delta * 0.1
        return logit_delta

    def active_parameter_bytes(self) -> int:
        return int(
            self.weight.numel() * self.weight.element_size()
            + self.bias.numel() * self.bias.element_size()
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": CONDITIONAL_ROUTE_LEARNER_FORMAT,
            "version": CONDITIONAL_ROUTE_LEARNER_VERSION,
            "learning_rate": self.learning_rate,
            "ema_rate": self.ema_rate,
            "weight": self.weight.detach().cpu().clone(),
            "bias": self.bias.detach().cpu().clone(),
            "error_baseline": self.error_baseline,
            "lesioned": self._lesioned,
        }

    def load_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("format") != CONDITIONAL_ROUTE_LEARNER_FORMAT:
            raise ValueError("unsupported conditional route learner format")
        if int(payload.get("version", -1)) != CONDITIONAL_ROUTE_LEARNER_VERSION:
            raise ValueError("unsupported conditional route learner version")
        weight = payload["weight"].detach().to(self.device, dtype=torch.float32).clone()
        bias = payload["bias"].detach().to(self.device, dtype=torch.float32).clone()
        if weight.shape != self.weight.shape or bias.shape != self.bias.shape:
            raise ValueError("conditional route learner payload shape mismatch")
        self.weight = weight
        self.bias = bias
        self.error_baseline = float(payload["error_baseline"])
        self._lesioned = bool(payload.get("lesioned", False))

    def from_payload(self, payload: dict[str, Any]) -> None:
        self.load_payload(payload)
