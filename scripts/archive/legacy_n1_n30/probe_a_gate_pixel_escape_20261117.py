"""Escape curve for the three low seeds: evaluate at 2000, 4000 and 6000 steps."""

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

OUT = ROOT / "reports/taiji_u2_22_5_a_pixel_escape_20261117.json"
SEEDS = (20261111, 20261112, 20261113)
MARKS = (2000, 4000, 6000)


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
    per_seed = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        head = Head("bytes")
        opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
        marks = []
        for step in range(1, MARKS[-1] + 1):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(head(image, onehot, request, mask), label).backward()
            opt.step()
            if step in MARKS:
                with torch.no_grad():
                    fit_acc = float(
                        (head(fit_image, fit_onehot, fit_request, fit_mask).argmax(-1) == fit_label)
                        .float()
                        .mean()
                    )
                    train_acc = float(
                        (head(image, onehot, request, mask).argmax(-1) == label).float().mean()
                    )
                marks.append({"step": step, "fit": round(fit_acc, 4), "train": round(train_acc, 4)})
        per_seed.append({"seed": seed, "marks": marks})
    medians = {
        str(mark): sorted(row["marks"][index]["fit"] for row in per_seed)[1]
        for index, mark in enumerate(MARKS)
    }
    payload = {
        "schema": "taiji-u2-22-5-a-pixel-escape-v1",
        "card": "22.5.0.A",
        "marks": list(MARKS),
        "per_seed": per_seed,
        "median_fit": medians,
        "verdict": "escapes" if medians[str(MARKS[-1])] >= 0.80 else "stuck",
        "cost": {"cpu": round(time.process_time() - started, 4)},
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "medians="
        + json.dumps(medians)
        + " verdict="
        + payload["verdict"]
        + " cpu="
        + str(payload["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
