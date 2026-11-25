"""Tick encoding of the frozen 23.1 composition fixture for research line N32.2 (R1).

Fixture ``taiji-23-composition-v1`` (D-20261119-167) stores worlds and Chinese questions,
not model inputs.  This module turns (world scene, question) into a (T, 48) tick sequence
whose format is identical for every question kind, so a task-agnostic model must infer
the kind from the question itself (D-20261125-506).

Tick layout (48 dims)
  scene ticks (one per body, order permuted by a hash of the world seed):
    0-5   name one-hot (``NAMES``)        6-8   position (first three coordinates)
    9     scene flag                      10-41 embedding of the colour character
  question ticks (one per character):
    10-41 character embedding             42    question flag
  43-47 unused; padding ticks are all zero (the GRU-RT backbone holds state on them).

Character embeddings are fixed unit vectors derived from sha256 of the character, so they
do not depend on vocabulary order.  Answers are 10 classes: ``POSITION_LABELS`` + ``NAMES``.

Data
  * training rows: fresh worlds with seeds ``TRAIN_SEED_BASE + k`` and train-style questions
    (kinds q1/q2/q3, template group 0, directions left/right) -- disjoint from every frozen
    partition (world seeds 230000-230895);
  * validation rows: the frozen ``train`` partition (512 worlds x q1/q2/q3), used only for
    evaluation in N32.2 (disclosed);
  * fit rows: frozen ``fit`` partition, report only (held-out directions/orders/wording, q4);
  * ``check`` / ``final`` are sealed: every builder here refuses them.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, cast

import torch

from taiji.native_composition import (
    NAMES,
    PARTITION_OFFSETS,
    PARTITION_SIZES,
    POSITION_LABELS,
    _questions,
    build_partition,
    world_scene,
)

ENCODING_VERSION = "taiji-23-composition-ticks-v1"
INPUT_DIM = 48
HORIZON = 24
EMBED_LO, EMBED_HI = 10, 42
EMBED_DIM = EMBED_HI - EMBED_LO
SCENE_FLAG = 9
QUESTION_FLAG = 42
ANSWERS: tuple[str, ...] = tuple(POSITION_LABELS) + tuple(NAMES)
ANSWER_INDEX = {a: i for i, a in enumerate(ANSWERS)}
KINDS = ("q1", "q2", "q3")
TRAIN_SEED_BASE = 23_100_000
FROZEN_SEED_RANGE = (230_000, 230_000 + sum(PARTITION_SIZES.values()))
SEALED = ("check", "final")


def char_embedding(ch: str) -> torch.Tensor:
    if len(ch) != 1:
        raise ValueError(f"single character expected, got {ch!r}")
    raw = hashlib.sha256(("taiji-23-char:" + ch).encode("utf-8")).digest()
    vec = torch.tensor([b / 127.5 - 1.0 for b in raw[:EMBED_DIM]], dtype=torch.float32)
    return cast(torch.Tensor, vec / vec.norm().clamp(min=1e-6))


def scene_order(seed: int, n: int) -> list[int]:
    keys = [hashlib.sha256(f"order:{seed}:{i}".encode()).hexdigest() for i in range(n)]
    return sorted(range(n), key=lambda i: keys[i])


def encode(scene: dict[str, Any], text: str) -> torch.Tensor:
    bodies = scene["bodies"]
    steps = len(bodies) + len(text)
    if steps > HORIZON:
        raise ValueError(f"sequence of {steps} ticks exceeds horizon {HORIZON}")
    ticks = torch.zeros(HORIZON, INPUT_DIM)
    for t, i in enumerate(scene_order(int(scene["seed"]), len(bodies))):
        body = bodies[i]
        name = body["name"]
        ticks[t, NAMES.index(name)] = 1.0
        pos = [float(v) for v in list(body["position"])[:3]]
        for j, v in enumerate(pos):
            if not math.isfinite(v):
                raise ValueError("non-finite position")
            ticks[t, 6 + j] = v
        ticks[t, SCENE_FLAG] = 1.0
        ticks[t, EMBED_LO:EMBED_HI] = char_embedding(name[0])
    for k, ch in enumerate(text):
        t = len(bodies) + k
        ticks[t, EMBED_LO:EMBED_HI] = char_embedding(ch)
        ticks[t, QUESTION_FLAG] = 1.0
    return ticks


def _check_partition(partition: str) -> None:
    if partition in SEALED:
        raise PermissionError(f"partition {partition!r} is sealed (check/final never read)")


def training_rows(count: int, start: int = 0) -> list[dict[str, Any]]:
    """``count`` worlds from ``TRAIN_SEED_BASE + start``; each yields q1, q2 and q3."""
    rows: list[dict[str, Any]] = []
    for k in range(start, start + count):
        seed = TRAIN_SEED_BASE + k
        scene = world_scene(seed)
        for q in _questions(seed, "train", scene):
            rows.append(
                {
                    "id": f"gen-{k}-{q['kind']}",
                    "seed": seed,
                    "kind": q["kind"],
                    "text": q["text"],
                    "answer": q["answer"],
                    "axes": q["axes"],
                    "scene": scene,
                    "world_digest": scene["world_digest"],
                }
            )
    return rows


def frozen_rows(partition: str) -> list[dict[str, Any]]:
    _check_partition(partition)
    if partition not in ("train", "fit"):
        raise ValueError(f"unknown partition {partition!r}")
    rows = build_partition(partition)
    scenes: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed not in scenes:
            scenes[seed] = world_scene(seed)
        row["scene"] = scenes[seed]
    return rows


def validation_rows() -> list[dict[str, Any]]:
    return frozen_rows("train")


def tensorize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ticks = torch.stack([encode(r["scene"], r["text"]) for r in rows])
    labels = torch.tensor([ANSWER_INDEX[r["answer"]] for r in rows], dtype=torch.long)
    kinds = [str(r["kind"]) for r in rows]
    return {"ticks": ticks, "labels": labels, "kinds": kinds}


def frozen_seed(seed: int) -> bool:
    return FROZEN_SEED_RANGE[0] <= seed < FROZEN_SEED_RANGE[1]


def describe() -> dict[str, Any]:
    return {
        "encoding_version": ENCODING_VERSION,
        "horizon": HORIZON,
        "answers": list(ANSWERS),
        "train_seed_base": TRAIN_SEED_BASE,
        "frozen_seed_range": list(FROZEN_SEED_RANGE),
        "partition_offsets": dict(PARTITION_OFFSETS),
    }
