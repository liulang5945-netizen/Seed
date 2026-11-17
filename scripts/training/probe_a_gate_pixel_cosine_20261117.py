"""Cosine-normalised salience with a learnable temperature: does it remove the seed split?"""

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

OUT = ROOT / "reports/taiji_u2_22_5_a_pixel_cosine_20261117.json"
SIDE = 16
STEPS = 2000
SEEDS = (20261110, 20261111, 20261112, 20261113, 20261114)


class CosHead(nn.Module):
    def __init__(self, width: int = 16, text: int = 16) -> None:
        super().__init__()
        self.byte = nn.Embedding(256, text)
        self.position = nn.Parameter(torch.zeros(1, 64, text))
        self.text = nn.Linear(2 * text, width)
        self.pixel = nn.Linear(3, width)
        self.null = nn.Parameter(torch.zeros(width))
        self.log_temp = nn.Parameter(torch.zeros(()))
        self.output = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 4))

    def forward(self, image, request, mask):
        flat = image.reshape(image.shape[0], SIDE * SIDE, 3)
        key = F.normalize(self.pixel(flat), dim=-1)
        text = self.byte(request) + self.position[:, : request.shape[1]]
        text = text.masked_fill(~mask[..., None], 0.0)
        mean = text.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        peak = text.masked_fill(~mask[..., None], -1e4).amax(1)
        question = F.normalize(self.text(torch.cat([mean, peak], -1)), dim=-1)
        scale = math.sqrt(key.shape[-1]) * self.log_temp.exp()
        score = torch.einsum("nd,nkd->nk", question, key) * scale
        score = torch.cat([score, (question @ F.normalize(self.null, dim=-1)).reshape(-1, 1)], -1)
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
    image, _ = _decode(rows["train12"])
    label = rows["train12"]["label"]
    fit_image, _ = _decode(rows["fit3"])
    fit_label = rows["fit3"]["label"]
    request, mask = rows["train12"]["request"], rows["train12"]["mask"]
    fit_request, fit_mask = rows["fit3"]["request"], rows["fit3"]["mask"]
    results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        head = CosHead()
        opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(STEPS):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(head(image, request, mask), label).backward()
            opt.step()
        with torch.no_grad():
            train_acc = float((head(image, request, mask).argmax(-1) == label).float().mean())
            fit_acc = float(
                (head(fit_image, fit_request, fit_mask).argmax(-1) == fit_label).float().mean()
            )
        results.append(
            {
                "seed": seed,
                "train": round(train_acc, 4),
                "fit": round(fit_acc, 4),
                "temp": round(float(head.log_temp.exp()), 3),
            }
        )
    fits = sorted(row["fit"] for row in results)
    payload = {
        "schema": "taiji-u2-22-5-a-pixel-cosine-v1",
        "card": "22.5.0.A",
        "steps": STEPS,
        "seeds": results,
        "median_fit": fits[len(fits) // 2],
        "min_fit": fits[0],
        "max_fit": fits[-1],
        "verdict": "cosine_stabilises" if fits[len(fits) // 2] >= 0.80 else "cosine_does_not",
        "cost": {"cpu": round(time.process_time() - started, 4)},
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "fits="
        + json.dumps([row["fit"] for row in results])
        + " median="
        + str(payload["median_fit"])
        + " verdict="
        + payload["verdict"]
        + " cpu="
        + str(payload["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
