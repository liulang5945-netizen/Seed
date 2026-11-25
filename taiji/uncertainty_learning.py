"""Learned uncertainty estimation facility (`taiji-uncertainty-learning-v1`, zero heavy training

Subject-side mechanism that learns to predict P(correct) from features and verifiable feedback,
instead of using fixed engineering priors. Arms:

* ``static``: constant 0.5 (no information);
* ``temperature``: a miscalibrated base score calibrated by the global temperature prototype
  from :mod:`taiji.uncertainty_calibration` (engineering control);
* ``learned``: logistic confidence model trained by gradient descent on a train split — the
  in-subject learned mechanism;
* ``oracle``: the true generative probability (control only).

All estimates are evaluated with ECE/Brier on a held-out split.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

SCHEMA = "taiji-uncertainty-learning-v1"
N_TRAIN = 3000
N_EVAL = 1000
COEF = (2.2, -1.4, 0.6)  # true logit = COEF @ [x0, x1, bias]


def make_data(seed: int, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(int(seed) ^ 0x5EED)
    x = torch.randn((n, 2), generator=gen)
    logits = COEF[0] * x[:, 0] + COEF[1] * x[:, 1] + COEF[2]
    true_p = torch.sigmoid(logits)
    u = torch.rand((n,), generator=gen)
    y = (u < true_p).to(torch.float32)
    return x, y, true_p


class BaseScore:
    """Miscalibrated engineering base score (over-confident by construction)."""

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = 2.0 * x[:, 0] - 1.2 * x[:, 1] + 0.3
        return torch.sigmoid(2.5 * logits)  # sharpened => over-confident


class TemperatureCalibrator:
    def __init__(self) -> None:
        self.temperature = 1.0
        self._base = BaseScore()

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        from taiji.uncertainty_calibration import fit_power_temperature

        fit = fit_power_temperature(self._base.predict(x).tolist(), y.tolist())
        self.temperature = fit["temperature"]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        p = self._base.predict(x).clamp(min=1e-6, max=1.0 - 1e-6)
        return p**self.temperature


@dataclass
class LearnedConfidence:
    w: torch.Tensor
    b: torch.Tensor

    @classmethod
    def train(
        cls, x: torch.Tensor, y: torch.Tensor, seed: int = 0, steps: int = 800, lr: float = 0.1
    ) -> LearnedConfidence:
        gen = torch.Generator().manual_seed(int(seed) ^ 0xABCD)
        w = torch.randn((x.shape[1],), generator=gen) * 0.1
        b = torch.zeros(())
        for _ in range(steps):
            p = torch.sigmoid(x @ w + b)
            grad_w = ((p - y).unsqueeze(1) * x).mean(dim=0)
            grad_b = (p - y).mean()
            w = w - lr * grad_w
            b = b - lr * grad_b
        return cls(w=w.detach(), b=b.detach())

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x @ self.w + self.b)


def brier(p: torch.Tensor, y: torch.Tensor) -> float:
    return round(float(((p - y) ** 2).mean()), 4)


def ece(p: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    total = 0.0
    n = len(p)
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (p >= lo) & (p < hi + (1e-9 if i == bins - 1 else 0))
        if mask.sum() == 0:
            continue
        total += abs(float(p[mask].mean()) - float(y[mask].mean())) * int(mask.sum()) / n
    return round(total, 4)


def evaluate(p: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    return {
        "ece": ece(p, y),
        "brier": brier(p, y),
        "mean_conf": round(float(p.mean()), 4),
        "acc": round(float(y.mean()), 4),
    }


def neg_log_likelihood(p: torch.Tensor, y: torch.Tensor) -> float:
    p = p.clamp(min=1e-6, max=1.0 - 1e-6)
    return round(-float((y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()), 4)


__all__ = [
    "SCHEMA",
    "N_TRAIN",
    "N_EVAL",
    "COEF",
    "BaseScore",
    "LearnedConfidence",
    "TemperatureCalibrator",
    "brier",
    "ece",
    "evaluate",
    "make_data",
    "neg_log_likelihood",
]
_ = F  # keep torch.nn.functional import for future variants
