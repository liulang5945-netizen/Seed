"""N32.3 diagnostic (numpy only, no torch): is Q2 of contract v2 learnable from the encoding?

Recorded under D-20261125-518.  Re-implements ``taiji-23-composition-ticks-v1`` for Q2 (3 scene
ticks + 10 question ticks, flattened to 624 features) and trains a 2-layer ReLU MLP (256-256,
Adam 1e-3, batch 256) on train worlds k < 6000, evaluating on val worlds 6000-6999 (4,000 rows).
Arm "aug": leak-free name relabelling of *train* worlds only (3 random permutations of the six
names per world, applied to scene one-hot, scene character, question entity and answer); val
worlds are disjoint, so no held-out information enters training.  Confirm split never read.

Usage: python scripts/training/n32_3_diag_mlp_q2_numpy.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "tests/fixtures/taiji_23_composition_v2_scenes.json"
NAMES = ("红色圆片", "蓝色圆片", "绿色圆片", "黄色圆片", "紫色圆片", "青色圆片")
ANSWERS = ("左边", "中间", "右边", "未知", *NAMES)
INDEX = {a: i for i, a in enumerate(ANSWERS)}
DIRS = ("左边", "右边", "前边", "后边")
_EMB: dict[str, np.ndarray] = {}


def emb(ch: str) -> np.ndarray:
    if ch not in _EMB:
        raw = hashlib.sha256(("taiji-23-char:" + ch).encode("utf-8")).digest()
        v = np.array([b / 127.5 - 1.0 for b in raw[:32]], dtype=np.float32)
        _EMB[ch] = v / max(float(np.linalg.norm(v)), 1e-6)
    return _EMB[ch]


def order(seed: int, n: int) -> list[int]:
    keys = [hashlib.sha256(f"order:{seed}:{i}".encode()).hexdigest() for i in range(n)]
    return sorted(range(n), key=lambda i: keys[i])


def relation(src: str, d: str, bodies: list[dict]) -> str:
    xy = {b["name"]: (b["position"][0], b["position"][1]) for b in bodies}
    sx, sy = xy[src]
    cands = []
    for n, (x, y) in xy.items():
        if n == src:
            continue
        gap = {"左边": sx - x, "右边": x - sx, "前边": y - sy, "后边": sy - y}[d]
        if gap > 0:
            cands.append((gap, n))
    return min(cands)[1] if cands else "未知"


def encode(seed: int, bodies: list[dict], text: str) -> np.ndarray:
    x = np.zeros((len(bodies) + len(text), 48), np.float32)
    for t, i in enumerate(order(seed, len(bodies))):
        b = bodies[i]
        x[t, NAMES.index(b["name"])] = 1.0
        x[t, 6], x[t, 7] = b["position"][0], b["position"][1]
        x[t, 9] = 1.0
        x[t, 10:42] = emb(b["name"][0])
    for k, ch in enumerate(text):
        x[len(bodies) + k, 10:42] = emb(ch)
        x[len(bodies) + k, 42] = 1.0
    return x.reshape(-1)


def build(scenes: list[dict], perms: dict[int, tuple[str, ...]] | None = None):
    xs, ys = [], []
    for s in scenes:
        bodies = s["bodies"]
        if perms is not None:
            mp = dict(zip(NAMES, perms[s["k"]], strict=True))
            bodies = [{"name": mp[b["name"]], "position": b["position"]} for b in bodies]
        names = [b["name"] for b in bodies]
        x = names[s["seed"] % len(names)]
        for d in DIRS:
            xs.append(encode(s["seed"], bodies, f"{x}的{d}是谁？"))
            ys.append(INDEX[relation(x, d, bodies)])
    return np.stack(xs), np.array(ys)


def train(xt, yt, xv, yv, epochs: int, seed: int) -> list[tuple[int, float]]:
    r = np.random.default_rng(seed)
    mu, sd = xt.mean(0), xt.std(0) + 1e-6
    a, bv = (xt - mu) / sd, (xv - mu) / sd

    def init(i: int, o: int):
        w = (r.standard_normal((i, o)) * np.sqrt(2 / i)).astype(np.float32)
        return w, np.zeros(o, np.float32)

    p = [*init(a.shape[1], 256), *init(256, 256), *init(256, 10)]
    m, v, st, hist = [0 * q for q in p], [0 * q for q in p], 0, []
    for ep in range(epochs):
        perm = r.permutation(len(a))
        for i in range(0, len(perm), 256):
            idx = perm[i : i + 256]
            xb, yb = a[idx], yt[idx]
            h1 = np.maximum(xb @ p[0] + p[1], 0)
            h2 = np.maximum(h1 @ p[2] + p[3], 0)
            lo = h2 @ p[4] + p[5]
            g = np.exp(lo - lo.max(1, keepdims=True))
            g /= g.sum(1, keepdims=True)
            g[np.arange(len(yb)), yb] -= 1
            g /= len(yb)
            g2 = (g @ p[4].T) * (h2 > 0)
            g1 = (g2 @ p[2].T) * (h1 > 0)
            grads = [xb.T @ g1, g1.sum(0), h1.T @ g2, g2.sum(0), h2.T @ g, g.sum(0)]
            st += 1
            for j in range(6):
                if j % 2 == 0:
                    grads[j] = grads[j] + 1e-4 * p[j]
                m[j] = 0.9 * m[j] + 0.1 * grads[j]
                v[j] = 0.999 * v[j] + 0.001 * grads[j] ** 2
                p[j] -= 1e-3 * (m[j] / (1 - 0.9**st)) / (np.sqrt(v[j] / (1 - 0.999**st)) + 1e-8)
        if ep % 10 == 9:
            h = np.maximum(np.maximum(bv @ p[0] + p[1], 0) @ p[2] + p[3], 0) @ p[4] + p[5]
            hist.append((ep + 1, round(float((h.argmax(1) == yv).mean()), 3)))
    return hist


def main() -> None:
    scenes = json.loads(SCENES.read_text(encoding="utf-8"))
    trs = [s for s in scenes if s["k"] < 6000]
    vas = [s for s in scenes if 6000 <= s["k"] < 7000]
    xt, yt = build(trs)
    xv, yv = build(vas)
    print("val majority", round(float(np.bincount(yv).max() / len(yv)), 4))
    print("plain", train(xt, yt, xv, yv, 150, 0))
    rng = np.random.default_rng(1)
    parts = [(xt, yt)]
    for _ in range(3):
        perms = {s["k"]: tuple(np.array(NAMES)[rng.permutation(6)]) for s in trs}
        parts.append(build(trs, perms))
    xa = np.concatenate([q[0] for q in parts])
    ya = np.concatenate([q[1] for q in parts])
    print("aug x4", train(xa, ya, xv, yv, 40, 0))


if __name__ == "__main__":
    main()
