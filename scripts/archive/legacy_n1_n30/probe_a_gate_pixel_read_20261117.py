"""A-gate readout: does a pixel-level salience-to-centroid operator reach 0.80?

The paired query probe excluded the lexicon hypothesis (entity one-hot was no
better than request bytes: 0.448 vs 0.474 on the three-body fit), so the wall is
the query-to-token-to-position binding itself.  Patch tokens quantise x to four
columns (patch centres), while the reference oracle ``locate_public`` reads a
weighted centroid over the 16 pixel columns.  A head that only sees patch-level
tokens therefore has to demix each 48-value block to recover sub-patch x, which
is exactly what it fails to do.

This probe replaces the token set with the 256 pixels of the 16x16 raster, each
carrying its 3 colour channels and its exact x/y, and reads the position by a
multiplicative salience match followed by a coordinate average:

    s_i = <q(entity), k(pixel_i)> / sqrt(d),  w = softmax([s, s_null])
    features = [mu_x, mu_y, mass, max_w, sum w x, sum w y]

The coordinate readout is linear in the weights, so sub-patch resolution is
available by construction; the only learned parts are the entity-to-colour match
and the band thresholds.

A reconstruction check runs first: the cached 48-value blocks are resplit into
the 16x16 raster and the declared oracle ``locate_public`` is re-run on it.  If
that does not reproduce the labels, the reconstruction is wrong and no head
reading is admissible.

Declared exits (entity query, 768 training worlds, three head seeds):

1. ``not_convergent``: any arm misses 0.99 train accuracy inside the step cap.
2. ``pixel_read_feasible``: ``fit3`` median >= 0.80.  Then the missing operator
   is a sub-patch, salience-weighted coordinate read and the readout identity is
   rebuilt on it.
3. ``resolution_is_the_bound``: ``fit3`` median < 0.80.  Then the 16x16 raster
   (bodies about two pixels wide) is the bound and the A gate has to change its
   specification rather than its readout.

No U2 optimizer step, no check read and no final read occur here.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.probe_a_gate_ceiling_20261117 import (  # noqa: E402
    EVAL_EVERY,
    STEP_CAP,
    TARGET_TRAIN,
    _features,
)
from scripts.training.probe_a_query_bind_head_20261117 import (  # noqa: E402
    HEAD_SEEDS,
    _confusion,
)
from taiji.u2_task_episode import NAMES, PALETTE  # noqa: E402

REPORT = ROOT / "reports/taiji_u2_22_5_a_gate_pixel_read_20261117.json"
SIDE = 16
SUFFIX = "现在在哪里？"
UNKNOWN = 3


def _grid() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([2 * (col + 0.5) / SIDE - 1 for col in range(SIDE)], dtype=torch.float32)
    y = torch.tensor([1 - 2 * (row + 0.5) / SIDE for row in range(SIDE)], dtype=torch.float32)
    return x[None, :].expand(SIDE, SIDE), y[:, None].expand(SIDE, SIDE)


def _pixels(raw: torch.Tensor) -> torch.Tensor:
    """Resplit the 16 patch blocks (4x4 pixels x 3 channels) into a 16x16 raster."""
    count = raw.shape[0]
    block = raw[:, :, :48].reshape(count, 4, 4, 4, 4, 3)
    return block.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(count, SIDE, SIDE, 3)


def _entities(requests: list[bytes]) -> torch.Tensor:
    rows = []
    for raw in requests:
        text = raw.decode("utf-8")
        if not text.endswith(SUFFIX):
            raise ValueError("unexpected_request_form")
        rows.append(NAMES.index(text[: -len(SUFFIX)]))
    return torch.tensor(rows)


def _decode(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    packed, mask = data["request"].tolist(), data["mask"].tolist()
    requests = [
        bytes([value for value, keep in zip(row, flags, strict=True) if keep])
        for row, flags in zip(packed, mask, strict=True)
    ]
    return _pixels(data["raw"]), functional.one_hot(_entities(requests), len(NAMES)).float()


def _oracle(image: torch.Tensor, entity: torch.Tensor) -> torch.Tensor:
    """Re-run the declared oracle (``locate_public``) on the resplit raster."""
    grid_x, _ = _grid()
    flat = image.reshape(image.shape[0], SIDE * SIDE, 3).double()
    palette = torch.tensor(PALETTE, dtype=torch.float64)
    distance = torch.cdist(flat, palette)
    nearest = distance.argmin(-1)
    own = distance.gather(-1, entity[:, None, None].expand(-1, flat.shape[1], 1))[:, :, 0]
    weight = (1 - own / 0.35).clamp(min=0) * (nearest == entity[:, None])
    mass = weight.sum(-1)
    centroid = (weight * grid_x.reshape(-1)[None, :].double()).sum(-1) / mass.clamp(min=1e-9)
    label = torch.where(
        centroid < -1 / 3,
        torch.zeros_like(mass, dtype=torch.long),
        torch.where(
            centroid > 1 / 3,
            torch.full_like(mass, 2, dtype=torch.long),
            torch.ones_like(mass, dtype=torch.long),
        ),
    )
    return torch.where(mass < 0.25, torch.full_like(label, UNKNOWN), label)


class PixelSalienceHead(nn.Module):
    """Multiplicative query-to-pixel match, then a linear coordinate readout."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.query = nn.Linear(len(NAMES), width)
        self.pixel = nn.Linear(3, width)
        self.null = nn.Parameter(torch.zeros(width))
        self.output = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 4))

    def forward(self, image: torch.Tensor, onehot: torch.Tensor):
        grid_x, grid_y = _grid()
        flat = image.reshape(image.shape[0], SIDE * SIDE, 3)
        key = torch.tanh(self.pixel(flat))
        question = torch.tanh(self.query(onehot))
        score = torch.cat(
            [
                torch.einsum("nd,nkd->nk", question, key) / math.sqrt(key.shape[-1]),
                (question @ self.null).unsqueeze(-1),
            ],
            -1,
        )
        weight = score.softmax(-1)[:, :-1]
        mass = weight.sum(-1)
        moment_x = (weight * grid_x.reshape(-1)[None, :]).sum(-1)
        moment_y = (weight * grid_y.reshape(-1)[None, :]).sum(-1)
        features = torch.stack(
            [
                moment_x / mass.clamp(min=1e-6),
                moment_y / mass.clamp(min=1e-6),
                mass,
                weight.amax(-1),
                moment_x,
                moment_y,
            ],
            -1,
        )
        return self.output(features)


def _fit(image, onehot, label, seed: int) -> dict:
    started = time.process_time()
    torch.manual_seed(seed)
    head = PixelSalienceHead()
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    steps, accuracy = 0, 0.0
    while steps < STEP_CAP:
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(head(image, onehot), label)
        loss.backward()
        optimizer.step()
        steps += 1
        if steps % EVAL_EVERY == 0:
            with torch.no_grad():
                accuracy = (head(image, onehot).argmax(-1) == label).float().mean().item()
            if accuracy >= TARGET_TRAIN:
                break
    head.eval()
    with torch.no_grad():
        logits = head(image, onehot)
    return {
        "seed": seed,
        "steps": steps,
        "train_accuracy": (logits.argmax(-1) == label).float().mean().item(),
        "train_confusion": _confusion(logits.argmax(-1), label),
        "head": head,
        "cpu": round(time.process_time() - started, 4),
    }


def _score(head, image, onehot, label) -> tuple[float, list[list[int]]]:
    with torch.no_grad():
        logits = head(image, onehot)
    return (logits.argmax(-1) == label).float().mean().item(), _confusion(logits.argmax(-1), label)


def _median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2]


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    rows = _features()
    splits = {}
    for key in ("train12", "fit3", "fit12"):
        image, onehot = _decode(rows[key])
        splits[key] = {"image": image, "onehot": onehot, "label": rows[key]["label"]}
    oracle = {}
    for key in ("train12", "fit3", "fit12"):
        entity = splits[key]["onehot"].argmax(-1)
        predicted = _oracle(splits[key]["image"], entity)
        oracle[key] = {
            "agreement": (predicted == splits[key]["label"]).float().mean().item(),
            "confusion": _confusion(predicted, splits[key]["label"]),
        }
    if oracle["train12"]["agreement"] < 0.99:
        raise ValueError("raster_reconstruction_does_not_reproduce_the_oracle")
    arms = {}
    for name, split in (("entity_fit3", "fit3"), ("entity_fit12", "fit12")):
        results = []
        for seed in HEAD_SEEDS:
            fitted = _fit(
                splits["train12"]["image"],
                splits["train12"]["onehot"],
                splits["train12"]["label"],
                seed,
            )
            head = fitted.pop("head")
            accuracy, confusion = _score(
                head, splits[split]["image"], splits[split]["onehot"], splits[split]["label"]
            )
            fitted["fit_accuracy"] = accuracy
            fitted["fit_confusion"] = confusion
            results.append(fitted)
        arms[name] = {
            "fit_split": split,
            "results": results,
            "median_train_accuracy": _median([row["train_accuracy"] for row in results]),
            "median_fit_accuracy": _median([row["fit_accuracy"] for row in results]),
        }
    converged = all(
        row["train_accuracy"] >= 0.99 for arm in arms.values() for row in arm["results"]
    )
    if not converged:
        verdict = "not_convergent"
    elif arms["entity_fit3"]["median_fit_accuracy"] >= 0.80:
        verdict = "pixel_read_feasible"
    else:
        verdict = "resolution_is_the_bound"
    payload = {
        "schema": "taiji-u2-22-5-a-gate-pixel-read-v1",
        "card": "22.5.0.A",
        "verdict": verdict,
        "why": (
            "the paired query probe excluded the lexicon; patch tokens quantise x to four "
            "columns, so the next test is a pixel-level salience match with a linear "
            "coordinate readout"
        ),
        "oracle_reconstruction": oracle,
        "conditions": {
            "tokens": "256 pixels of the 16x16 raster, exact x/y",
            "match": "multiplicative query/pixel score plus a learned null slot",
            "readout": "[mu_x, mu_y, mass, max_w, sum w x, sum w y] -> 32 -> 4",
            "query": "6-dim entity one-hot (declared prior)",
            "train_split": "train12 (768 worlds, 1-2 bodies)",
            "step_cap": STEP_CAP,
            "target_train_accuracy": TARGET_TRAIN,
            "evaluate_every": EVAL_EVERY,
            "seeds": list(HEAD_SEEDS),
        },
        "arms": arms,
        "medians": {name: arm["median_fit_accuracy"] for name, arm in arms.items()},
        "declared_verdict": {
            "not_convergent": "any arm below 0.99 train accuracy at the step cap",
            "pixel_read_feasible": "entity_fit3 median >= 0.80",
            "resolution_is_the_bound": "entity_fit3 median < 0.80",
        },
        "discipline": {
            "u2_optimizer_steps": 0,
            "head_optimizer_steps": sum(
                row["steps"] for arm in arms.values() for row in arm["results"]
            ),
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.perf_counter() - started_wall, 4),
        },
        "source_pins": {
            "scripts/training/probe_a_gate_pixel_read_20261117.py": hashlib.sha256(
                Path(__file__).read_text(encoding="utf-8-sig").encode()
            ).hexdigest(),
            "scripts/training/probe_a_gate_ceiling_20261117.py": hashlib.sha256(
                (ROOT / "scripts/training/probe_a_gate_ceiling_20261117.py")
                .read_text(encoding="utf-8-sig")
                .encode()
            ).hexdigest(),
        },
    }
    REPORT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "pixel read verdict="
        + verdict
        + " oracle="
        + json.dumps({key: round(value["agreement"], 4) for key, value in oracle.items()})
        + " medians="
        + json.dumps({name: round(value, 4) for name, value in payload["medians"].items()})
        + " steps="
        + json.dumps({name: [row["steps"] for row in arm["results"]] for name, arm in arms.items()})
        + " cost="
        + str(payload["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
