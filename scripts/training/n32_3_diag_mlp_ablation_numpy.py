"""N32.3 diagnostic (numpy only): why the N32.3.02 MLP reference did not reproduce.

Recorded under the ledger entry that reviews N32.3.02.  Same data and features as
``n32_3_diag_mlp_q2_numpy.py`` (Q2 of contract v2, train worlds k < 6000, val 6000-6999,
4,000 rows, no augmentation).  Every arm sees ~1.27M sample presentations (53 epochs of
24,000 rows, batch 512, Adam), i.e. about what the N32.3.02 MLP saw (2,500 x 512).

Arms vary three factors: input standardisation (fit on train), depth/activation, lr.
Usage: python scripts/training/n32_3_diag_mlp_ablation_numpy.py
"""

from __future__ import annotations

import json

import numpy as np

from scripts.training.n32_3_diag_mlp_q2_numpy import SCENES, build

EPOCHS = 53
ARMS = (
    ("std", "relu", 2, 1e-3),
    ("raw", "relu", 2, 1e-3),
    ("std", "tanh", 1, 3e-3),
    ("raw", "tanh", 1, 3e-3),
    ("raw", "tanh", 1, 1e-3),
)


def run(xt, yt, xv, yv, scale: str, act: str, layers: int, lr: float, seed: int = 0) -> float:
    r = np.random.default_rng(seed)
    mu, sd = (xt.mean(0), xt.std(0) + 1e-6) if scale == "std" else (0.0, 1.0)
    a, bv = (xt - mu) / sd, (xv - mu) / sd
    dims = [a.shape[1], *([256] * layers), 10]
    gain = 2.0 if act == "relu" else 1.0
    p: list[np.ndarray] = []
    for i in range(len(dims) - 1):
        p.append(
            (r.standard_normal((dims[i], dims[i + 1])) * np.sqrt(gain / dims[i])).astype(np.float32)
        )
        p.append(np.zeros(dims[i + 1], np.float32))

    def f(z):
        return np.maximum(z, 0) if act == "relu" else np.tanh(z)

    def df(h):
        return (h > 0).astype(np.float32) if act == "relu" else 1 - h * h

    n = len(dims) - 1

    def fwd(x):
        hs = [x]
        for i in range(n - 1):
            hs.append(f(hs[-1] @ p[2 * i] + p[2 * i + 1]))
        return hs, hs[-1] @ p[2 * (n - 1)] + p[2 * (n - 1) + 1]

    m, v, st = [0 * q for q in p], [0 * q for q in p], 0
    for _ in range(EPOCHS):
        perm = r.permutation(len(a))
        for i in range(0, len(perm), 512):
            idx = perm[i : i + 512]
            hs, lo = fwd(a[idx])
            yb = yt[idx]
            g = np.exp(lo - lo.max(1, keepdims=True))
            g /= g.sum(1, keepdims=True)
            g[np.arange(len(yb)), yb] -= 1
            g /= len(yb)
            grads: list[np.ndarray] = [np.empty(0)] * len(p)
            for layer in range(n - 1, -1, -1):
                grads[2 * layer] = hs[layer].T @ g
                grads[2 * layer + 1] = g.sum(0)
                if layer > 0:
                    g = (g @ p[2 * layer].T) * df(hs[layer])
            st += 1
            for j in range(len(p)):
                m[j] = 0.9 * m[j] + 0.1 * grads[j]
                v[j] = 0.999 * v[j] + 0.001 * grads[j] ** 2
                p[j] -= lr * (m[j] / (1 - 0.9**st)) / (np.sqrt(v[j] / (1 - 0.999**st)) + 1e-8)
    return float((fwd(bv)[1].argmax(1) == yv).mean())


def main() -> None:
    scenes = json.loads(SCENES.read_text(encoding="utf-8"))
    xt, yt = build([s for s in scenes if s["k"] < 6000])
    xv, yv = build([s for s in scenes if 6000 <= s["k"] < 7000])
    for arm in ARMS:
        print(arm, round(run(xt, yt, xv, yv, *arm), 3))


if __name__ == "__main__":
    main()
