"""Self-read versus told: the pixel operator under a byte query.

The 0.93 fit came from a head whose query was a six-dim entity one-hot, so it
never read the request.  This probe holds the pixel operator fixed and only
changes the query source: pooled request bytes versus the one-hot prior, two
seeds, 1000 full-batch steps.  Cached features, no U2 training.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.probe_a_gate_ceiling_20261117 import _features  # noqa: E402
from scripts.training.probe_a_gate_pixel_read_20261117 import _decode  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_5_a_pixel_selfread_20261117.json"
SIDE = 16
STEPS = 1000
SEEDS = (20261110, 20261111)
LABELS = 4


class Head(nn.Module):
    def __init__(self, query: str, width: int = 16, text: int = 16) -> None:
        super().__init__()
        self.query_kind = query
        self.byte = nn.Embedding(256, text)
        self.position = nn.Parameter(torch.zeros(1, 64, text))
        self.text = nn.Linear(2 * text, width)
        self.onehot = nn.Linear(LABELS + 2, width)
        self.pixel = nn.Linear(3, width)
        self.null = nn.Parameter(torch.zeros(width))
        self.output = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, LABELS))

    def forward(self, image, onehot, request, mask):
        flat = image.reshape(image.shape[0], SIDE * SIDE, 3)
        key = torch.tanh(self.pixel(flat))
        if self.query_kind == "bytes":
            text = self.byte(request) + self.position[:, : request.shape[1]]
            text = text.masked_fill(~mask[..., None], 0.0)
            mean = text.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            peak = text.masked_fill(~mask[..., None], -1e4).amax(1)
            question = torch.tanh(self.text(torch.cat([mean, peak], -1)))
        else:
            question = torch.tanh(self.onehot(onehot))
        score = torch.einsum("nd,nkd->nk", question, key) / math.sqrt(key.shape[-1])
        score = torch.cat([score, (question @ self.null).reshape(-1, 1)], -1)
        weight = score.softmax(-1)[:, :-1]
        mass = weight.sum(-1)
        grid = torch.linspace(-1, 1, SIDE)
        x = grid.repeat(SIDE).reshape(1, -1)
        y = grid.repeat_interleave(SIDE).reshape(1, -1)
        mx = (weight * x).sum(-1)
        my = (weight * y).sum(-1)
        features = torch.stack(
            [mx / mass.clamp(min=1e-6), my / mass.clamp(min=1e-6), mass, weight.amax(-1), mx, my],
            -1,
        )
        return self.output(features)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.process_time()
    rows = _features()
    image, onehot = _decode(rows["train12"])
    label = rows["train12"]["label"]
    fit_image, fit_onehot = _decode(rows["fit3"])
    fit_label = rows["fit3"]["label"]
    request = rows["train12"]["request"]
    mask = rows["train12"]["mask"]
    fit_request = rows["fit3"]["request"]
    fit_mask = rows["fit3"]["mask"]
    arms = {}
    for kind in ("bytes", "entity"):
        seeds = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            head = Head(kind)
            opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
            for _ in range(STEPS):
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(image, onehot, request, mask), label)
                loss.backward()
                opt.step()
            with torch.no_grad():
                train_acc = float(
                    (head(image, onehot, request, mask).argmax(-1) == label).float().mean()
                )
                fit_acc = float(
                    (head(fit_image, fit_onehot, fit_request, fit_mask).argmax(-1) == fit_label)
                    .float()
                    .mean()
                )
            seeds.append({"seed": seed, "train": round(train_acc, 4), "fit": round(fit_acc, 4)})
        arms[kind] = {
            "seeds": seeds,
            "median_fit": sorted(s["fit"] for s in seeds)[len(seeds) // 2],
        }
    payload = {
        "schema": "taiji-u2-22-5-a-pixel-selfread-v1",
        "card": "22.5.0.A",
        "steps": STEPS,
        "arms": arms,
        "verdict": (
            "self_read_holds" if arms["bytes"]["median_fit"] >= 0.80 else "self_read_collapses"
        ),
        "cost": {"cpu": round(time.process_time() - started, 4)},
        "discipline": {"u2_optimizer_steps": 0, "check_reads": 0},
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "bytes="
        + json.dumps([s["fit"] for s in arms["bytes"]["seeds"]])
        + " entity="
        + json.dumps([s["fit"] for s in arms["entity"]["seeds"]])
        + " verdict="
        + payload["verdict"]
        + " cpu="
        + str(payload["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
