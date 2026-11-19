"""22.7.0-α adapter v3: raw-media palette histogram + linear informativeness probe.

Adapter v2 (4x4 patch tokens) reached only 0.5 fit accuracy on the 4-way task: the
coarse patch grid plus chromaticity noise still loses the entity's position.  v3
goes to the SAME media the world exposes (``raw_media["rgb"]`` regenerated from
each example's audited seed) and counts near-palette pixels per horizontal third,
without using labels or the oracle.  A linear softmax probe must reach >= 0.6 on
the fit worlds before r4 is allowed (pre-registered in D-20261119-33).
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
from taiji.u2_task_world import PhysicalWorld  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_7_0_alpha_adapter_v3_20261120.json"
LABELS = ("左边", "中间", "右边", "未知")
SUFFIX = "现在在哪里？"
PALETTE = torch.tensor(RGB_COLORS, dtype=torch.float64) / 255.0
PALETTE_CHROMA = PALETTE / PALETTE.sum(dim=1, keepdim=True)
CHROMA_DISTANCE = 0.08
SATURATION_MIN = 0.10


def entity_from_request(example: dict) -> str:
    text = example["public"]["request"].decode("utf-8")
    return text[: -len(SUFFIX)] if text.endswith(SUFFIX) else text


def raw_media(example: dict) -> tuple[torch.Tensor, torch.Tensor]:
    seed = int(example["audit"]["seed"])
    world = PhysicalWorld.reset(seed, {"body_count": 3})
    result = world.elapse("public-scene")
    image = result["raw_media"]["rgb"].to(torch.float64) / 255.0
    height, width = image.shape[0], image.shape[1]
    pixels = image.reshape(-1, 3)
    columns = torch.arange(width, dtype=torch.float64).repeat(height)
    xs = columns * 2.0 / max(1, width - 1) - 1.0
    return pixels, xs


def adapter_v3_features(example: dict) -> torch.Tensor:
    pixels, xs = raw_media(example)  # (P, 3), (P,)
    saturation = pixels.max(dim=1).values - pixels.min(dim=1).values
    chroma = pixels / pixels.sum(dim=1, keepdim=True).clamp_min(1e-6)
    distance = torch.cdist(chroma, PALETTE_CHROMA)
    nearest = distance.argmin(dim=1)
    valid = (distance.min(dim=1).values <= CHROMA_DISTANCE) & (saturation >= SATURATION_MIN)
    count = float(valid.sum())
    counts = torch.zeros(len(NAMES), 3, dtype=torch.float64)
    for index in torch.nonzero(valid).flatten().tolist():
        colour = int(nearest[index].item())
        x = float(xs[index])
        third = 0 if x < -1.0 / 3.0 else (2 if x > 1.0 / 3.0 else 1)
        counts[colour, third] += 1.0
    totals = counts.sum(dim=1)
    scale = max(1.0, count)
    normal = counts.flatten() / scale  # 18
    presence = (totals > 0).to(torch.float64)  # 6
    shares = counts.sum(dim=0) / max(1.0, count)  # 3
    entity = entity_from_request(example)
    entity_index = NAMES.index(entity) if entity in NAMES else -1
    query = torch.zeros(len(NAMES), dtype=torch.float64)
    if entity_index >= 0:
        query[entity_index] = 1.0
    entity_stats = (
        torch.tensor(
            [
                totals[entity_index].item() / scale,
                counts[entity_index, 0].item() / scale,
                counts[entity_index, 2].item() / scale,
            ],
            dtype=torch.float64,
        )
        if entity_index >= 0
        else torch.zeros(3, dtype=torch.float64)
    )
    vector = torch.cat([normal, presence, shares, entity_stats, query, torch.tensor([count])])
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
    x_train = torch.stack([adapter_v3_features(example) for example in train]).float()
    y_train = torch.tensor([label_index(example) for example in train])
    x_fit = torch.stack([adapter_v3_features(example) for example in fit]).float()
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
    verdict = "adapter_v3_signal_ready" if fit_accuracy >= 0.6 else "adapter_v3_signal_insufficient"
    payload = {
        "schema": "taiji-u2-22-7-0-alpha-adapter-v3-v1",
        "card": "alpha adapter v3 raw-media palette histogram + linear probe",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "source": "raw_media rgb regenerated from the example's audited seed",
            "features": 48,
            "chroma_distance": CHROMA_DISTANCE,
            "saturation_min": SATURATION_MIN,
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
    return 0 if verdict == "adapter_v3_signal_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
