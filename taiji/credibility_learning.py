"""Online source-credibility learning facility (`taiji-credibility-learning-v1`, zero training

A subject-side mechanism that learns per-source reliabilities from verifiable feedback, and
tracks a mid-stream reliability shift. Three arms:

* ``static``: the fixed priors from :mod:`taiji.evidence_ledger` (engineering constants);
* ``beta``: online Beta(alpha, beta) counts per source, predicting P(correct) with smoothing
  and updating from every observed outcome — a learned, in-subject mechanism;
* ``oracle``: the hidden true accuracies (control only, never used by the subject).

The simulator provides claims from five sources; one source ("forum") is unreliable early and
becomes reliable at a known index, so a static prior cannot track it while an adaptive
estimator can.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

SCHEMA = "taiji-credibility-learning-v1"
SOURCES = ("owner", "manual", "sensor", "forum", "rumor")
TRUE_ACC = {"owner": 0.95, "manual": 0.85, "sensor": 0.65, "forum": 0.35, "rumor": 0.18}
SHIFT_SOURCE = "forum"
SHIFT_INDEX = 1000
SHIFT_ACC = 0.82
N_ITEMS = 3000


class ClaimStream:
    """Deterministic stream of (source, correctness) items with a mid-stream reliability shift."""

    def __init__(self, seed: int, *, shift: bool = True) -> None:
        self.rng = random.Random(int(seed))
        self.shift = shift

    def accuracy(self, source: str, index: int) -> float:
        if self.shift and index >= SHIFT_INDEX and source == SHIFT_SOURCE:
            return SHIFT_ACC
        return TRUE_ACC[source]

    def generate(self, n: int = N_ITEMS) -> list[tuple[str, int]]:
        items = []
        for i in range(n):
            source = SOURCES[self.rng.randrange(len(SOURCES))]
            acc = self.accuracy(source, i)
            items.append((source, 1 if self.rng.random() < acc else 0))
        return items


class StaticPrior:
    """Control arm: fixed engineering priors, no learning."""

    def __init__(self) -> None:
        from taiji.evidence_ledger import CRED_PRIOR

        self.prior = dict(CRED_PRIOR)

    def predict(self, source: str, index: int) -> float:
        return self.prior[source]

    def update(self, source: str, correct: int) -> None:
        pass


class OracleAccuracy:
    def __init__(self, stream: ClaimStream) -> None:
        self.stream = stream

    def predict(self, source: str, index: int) -> float:
        return self.stream.accuracy(source, index)

    def update(self, source: str, correct: int) -> None:
        pass


@dataclass
class BetaCredibility:
    """Learned arm: Beta(1+correct, 1+wrong) per source; prediction is the posterior mean."""

    alpha: dict[str, float] = field(default_factory=dict)
    beta: dict[str, float] = field(default_factory=dict)
    forgetting: float = 0.0  # >0 enables exponential forgetting for non-stationary sources

    def predict(self, source: str, index: int) -> float:
        a = self.alpha.get(source, 1.0)
        b = self.beta.get(source, 1.0)
        return a / (a + b)

    def update(self, source: str, correct: int) -> None:
        a = self.alpha.get(source, 1.0)
        b = self.beta.get(source, 1.0)
        if self.forgetting > 0.0:
            a = 1.0 + self.forgetting * (a - 1.0)
            b = 1.0 + self.forgetting * (b - 1.0)
        self.alpha[source] = a + (1.0 if correct else 0.0)
        self.beta[source] = b + (0.0 if correct else 1.0)


def evaluate(estimator, items: list[tuple[str, int]]) -> dict:
    """Sequential prediction-then-update error; also Brier pre/post shift."""
    brier_sum = 0.0
    pre_sum = pre_n = 0
    post_sum = post_n = 0
    preds: list[float] = []
    for i, (source, correct) in enumerate(items):
        p = estimator.predict(source, i)
        preds.append(p)
        brier_sum += (p - correct) ** 2
        if i < SHIFT_INDEX:
            pre_sum += abs(p - correct)
            pre_n += 1
        else:
            post_sum += abs(p - correct)
            post_n += 1
        estimator.update(source, correct)
    n = len(items)
    return {
        "brier": round(brier_sum / max(1, n), 4),
        "pre_abs_error": round(pre_sum / max(1, pre_n), 4),
        "post_abs_error": round(post_sum / max(1, post_n), 4),
        "mean_pred": round(sum(preds) / max(1, len(preds)), 4),
    }
