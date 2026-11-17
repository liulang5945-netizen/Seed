"""Five-seed confirmation of the byte-query pixel operator at 2000 steps."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.probe_a_gate_ceiling_20261117 import _features  # noqa: E402
from scripts.training.probe_a_gate_pixel_read_20261117 import _decode  # noqa: E402
from scripts.training.probe_a_gate_pixel_selfread_20261117 import Head  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_5_a_pixel_selfread_v2_20261117.json"
STEPS = 2000
SEEDS = (20261110, 20261111, 20261112, 20261113, 20261114)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.process_time()
    rows = _features()
    image, onehot = _decode(rows["train12"])
    label = rows["train12"]["label"]
    fit_image, fit_onehot = _decode(rows["fit3"])
    fit_label = rows["fit3"]["label"]
    request, mask = rows["train12"]["request"], rows["train12"]["mask"]
    fit_request, fit_mask = rows["fit3"]["request"], rows["fit3"]["mask"]
    results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        head = Head("bytes")
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
        results.append({"seed": seed, "train": round(train_acc, 4), "fit": round(fit_acc, 4)})
    fits = sorted(row["fit"] for row in results)
    payload = {
        "schema": "taiji-u2-22-5-a-pixel-selfread-v2",
        "card": "22.5.0.A",
        "query": "request bytes",
        "steps": STEPS,
        "seeds": results,
        "median_fit": fits[len(fits) // 2],
        "min_fit": fits[0],
        "max_fit": fits[-1],
        "verdict": "self_read_holds" if fits[len(fits) // 2] >= 0.80 else "self_read_collapses",
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
