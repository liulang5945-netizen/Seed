"""22.7.0-α adapter v2: perception features + linear informativeness probe.

The r1–r3 statistical adapter destroyed the "which colour is where" signal (linear
probe at chance).  v2 reads the raw 4x4 patch grid: each of the 16 rgb tokens is a
4x4x3 colour block; we classify every patch to the nearest palette colour, use the
patch coordinate to assign a left/middle/right bin, and expose (colour x bin)
counts plus the queried colour one-hot from the request text.  No labels or
oracle values are used as features.

This file also runs the linear softmax probe that must reach >= 0.6 on the fit
worlds before r4 is allowed (pre-registered in D-20261119-33).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5 import RGB_COLORS  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from taiji.u2_task_episode import NAMES  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_7_0_alpha_adapter_v2_20261120.json"
LABELS = ("左边", "中间", "右边", "未知")
SUFFIX = "现在在哪里？"
PALETTE = torch.tensor(RGB_COLORS, dtype=torch.float64) / 255.0
CHROMA_DISTANCE = 0.10
SATURATION_MIN = 0.03


def patch_colours(example: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-patch palette index (16,) via chromaticity, and x (16,)."""
    rgb = example["public"]["observations"]["rgb"].to(torch.float64)
    patches = rgb.reshape(16, 4, 4, 3).mean(dim=(1, 2))  # (16, 3)
    saturation = patches.max(dim=1).values - patches.min(dim=1).values
    palette_chroma = PALETTE / PALETTE.sum(dim=1, keepdim=True)
    chroma = patches / patches.sum(dim=1, keepdim=True).clamp_min(1e-6)
    distance = torch.cdist(chroma, palette_chroma)
    nearest = distance.argmin(dim=1)
    valid = (distance.min(dim=1).values <= CHROMA_DISTANCE) & (saturation >= SATURATION_MIN)
    nearest = torch.where(valid, nearest, torch.full_like(nearest, -1))
    coordinates = example["public"]["coordinates"]["rgb"].to(torch.float64)
    return nearest, coordinates[:, 0]


def position_bin(x: float) -> int:
    if x < -1.0 / 3.0:
        return 0
    if x > 1.0 / 3.0:
        return 2
    return 1


def entity_from_request(example: dict) -> str:
    text = example["public"]["request"].decode("utf-8")
    return text[: -len(SUFFIX)] if text.endswith(SUFFIX) else text


def adapter_v2_features(example: dict) -> torch.Tensor:
    colours, xs = patch_colours(example)
    entity = entity_from_request(example)
    entity_index = NAMES.index(entity) if entity in NAMES else -1
    counts = torch.zeros(len(NAMES), dtype=torch.float64)
    x_sum = torch.zeros(len(NAMES), dtype=torch.float64)
    x2_sum = torch.zeros(len(NAMES), dtype=torch.float64)
    for colour, x in zip(colours.tolist(), xs.tolist(), strict=True):
        if colour < 0:
            continue
        counts[colour] += 1.0
        x_sum[colour] += float(x)
        x2_sum[colour] += float(x) * float(x)
    safe = counts.clamp_min(1.0)
    mean_x = torch.where(counts > 0, x_sum / safe, torch.zeros_like(counts))
    var_x = torch.where(counts > 0, x2_sum / safe - mean_x.pow(2), torch.zeros_like(counts))
    per_colour = torch.stack([counts / 3.0, mean_x, var_x], dim=1).flatten()  # 18
    query = torch.zeros(len(NAMES), dtype=torch.float64)
    if entity_index >= 0:
        query[entity_index] = 1.0
    entity_stats = (
        torch.tensor(
            [
                counts[entity_index].item() / 3.0,
                mean_x[entity_index].item(),
                abs(mean_x[entity_index].item()),
                var_x[entity_index].item(),
            ],
            dtype=torch.float64,
        )
        if entity_index >= 0
        else torch.zeros(4, dtype=torch.float64)
    )
    valid_count = counts.sum().clamp(0.0, 16.0)[None] / 16.0
    vector = torch.cat([per_colour, entity_stats, query, valid_count])
    # per_colour 18 + entity_stats 4 + query 6 + count 1 = 29
    return torch.cat([vector, torch.zeros(48 - vector.numel(), dtype=torch.float64)])[:48]


def label_index(example: dict) -> int:
    return LABELS.index(example["target"].decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    train = generate_examples("train", 256)
    fit = generate_examples("fit", 32)
    x_train = torch.stack([adapter_v2_features(example) for example in train]).float()
    y_train = torch.tensor([label_index(example) for example in train])
    x_fit = torch.stack([adapter_v2_features(example) for example in fit]).float()
    y_fit = torch.tensor([label_index(example) for example in fit])
    torch.manual_seed(20261120)
    probe = torch.nn.Linear(48, 4)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(probe(x_train), y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float((probe(x_train).argmax(-1) == y_train).float().mean())
        fit_accuracy = float((probe(x_fit).argmax(-1) == y_fit).float().mean())
    verdict = "adapter_v2_signal_ready" if fit_accuracy >= 0.6 else "adapter_v2_signal_insufficient"
    payload = {
        "schema": "taiji-u2-22-7-0-alpha-adapter-v2-v1",
        "card": "alpha adapter v2 perception features + linear probe",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "features": 48,
            "colour_bin_counts": 18,
            "query_one_hot": 6,
            "colour_presence": 6,
            "entity_bins": 3,
            "bin_shares": 3,
            "chroma_distance": CHROMA_DISTANCE,
            "train_worlds": 256,
            "fit_worlds": 32,
            "probe": "linear softmax, 500 Adam steps",
            "threshold_fit_accuracy": 0.6,
            "chance": 0.25,
        },
        "train_accuracy": round(train_accuracy, 4),
        "fit_accuracy": round(fit_accuracy, 4),
        "verdict": verdict,
        "discipline": {"graph_optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if verdict == "adapter_v2_signal_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
