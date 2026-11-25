"""Level-1 / data tests for the 23.1 tick encoding (N32.2.01, D-20261125-506).

Checks: frozen digests unchanged; sealed partitions refused; closed vocabulary with
distinct character embeddings; horizon <= 24; ticks decode back to the scene and the
question exactly; an independent oracle recomputes every answer from the decoded ticks
(accuracy 1.0); generated training worlds are disjoint from every frozen partition; the
encoding is deterministic.

Run: python -m pytest tests/test_composition_encoding_20261125.py -q
"""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402

from taiji import composition_encoding as ce  # noqa: E402
from taiji.native_composition import NAMES, partition_digest  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_u2_23_1_composition_contract_20261120.json"
DIRS = ("左边", "右边", "前边", "后边")
TRAIN_SAMPLE = 200


@lru_cache(maxsize=1)
def _val() -> list[dict[str, Any]]:
    return ce.validation_rows()


@lru_cache(maxsize=1)
def _fit() -> list[dict[str, Any]]:
    return ce.frozen_rows("fit")


@lru_cache(maxsize=1)
def _train() -> list[dict[str, Any]]:
    return ce.training_rows(TRAIN_SAMPLE)


def _all_rows() -> list[dict[str, Any]]:
    return [*_val(), *_fit(), *_train()]


def _vocab() -> list[str]:
    chars = {c for r in _all_rows() for c in r["text"]} | {n[0] for n in NAMES}
    return sorted(chars)


def _decode(ticks: torch.Tensor, vocab: list[str]) -> tuple[list[dict[str, Any]], str]:
    table = torch.stack([ce.char_embedding(c) for c in vocab])
    bodies, text = [], []
    for row in ticks:
        if float(row[ce.SCENE_FLAG]) == 1.0:
            name = NAMES[int(row[:6].argmax())]
            bodies.append({"name": name, "x": float(row[6]), "y": float(row[7])})
        elif float(row[ce.QUESTION_FLAG]) == 1.0:
            emb = row[ce.EMBED_LO : ce.EMBED_HI]
            text.append(vocab[int((table - emb).norm(dim=1).argmin())])
    return bodies, "".join(text)


def _nearest(bodies: list[dict[str, Any]], src: str | None, d: str) -> str | None:
    if src is None:
        return None
    me = next(b for b in bodies if b["name"] == src)
    best: tuple[float, str] | None = None
    for b in bodies:
        if b["name"] == src:
            continue
        dx, dy = b["x"] - me["x"], b["y"] - me["y"]
        gap = {"左边": -dx, "右边": dx, "前边": dy, "后边": -dy}[d]
        if gap > 0 and (best is None or (gap, b["name"]) < best):
            best = (gap, b["name"])
    return None if best is None else best[1]


def _oracle(bodies: list[dict[str, Any]], text: str) -> tuple[str, str]:
    entity = next(n for n in NAMES if n in text)
    dirs = [m.group(0) for m in re.finditer("|".join(DIRS), text)]
    if "根据" in text or "依照" in text or not dirs:
        kind = "q4" if ("根据" in text or "依照" in text) else "q1"
        x = next(b["x"] for b in bodies if b["name"] == entity)
        return kind, "左边" if x < -1 / 3 else "右边" if x > 1 / 3 else "中间"
    if len(dirs) == 1:
        ans = _nearest(bodies, entity, dirs[0])
        return "q2", ans or "未知"
    ans = _nearest(bodies, _nearest(bodies, entity, dirs[0]), dirs[1])
    return "q3", ans or "未知"


def test_frozen_digests_unchanged() -> None:
    digests = json.loads(MANIFEST.read_text(encoding="utf-8"))["criteria"]["digests"]
    assert partition_digest("train") == digests["train"]
    assert partition_digest("fit") == digests["fit"]


@pytest.mark.parametrize("partition", ce.SEALED)
def test_sealed_partitions_refused(partition: str) -> None:
    with pytest.raises(PermissionError):
        ce.frozen_rows(partition)


def test_closed_vocabulary_distinct_embeddings() -> None:
    vocab = _vocab()
    table = torch.stack([ce.char_embedding(c) for c in vocab])
    dist = torch.cdist(table, table) + torch.eye(len(vocab)) * 10.0
    assert float(dist.min()) > 0.2
    assert all(r["answer"] in ce.ANSWER_INDEX for r in _all_rows())


def test_horizon_and_shapes() -> None:
    batch = ce.tensorize(_all_rows())
    assert batch["ticks"].shape == (len(_all_rows()), ce.HORIZON, ce.INPUT_DIM)
    assert torch.isfinite(batch["ticks"]).all()


def test_decode_and_independent_oracle() -> None:
    vocab = _vocab()
    for row in _all_rows():
        ticks = ce.encode(row["scene"], row["text"])
        bodies, text = _decode(ticks, vocab)
        assert text == row["text"], row["id"]
        names = sorted(b["name"] for b in row["scene"]["bodies"])
        assert sorted(b["name"] for b in bodies) == names, row["id"]
        kind, answer = _oracle(bodies, text)
        assert kind == row["kind"], row["id"]
        assert answer == row["answer"], row["id"]


def test_training_worlds_disjoint_from_frozen() -> None:
    assert all(not ce.frozen_seed(r["seed"]) for r in _train())
    frozen = {r["world_digest"] for r in [*_val(), *_fit()]}
    assert not frozen & {r["world_digest"] for r in _train()}
    frozen_axes = {
        (
            r["kind"],
            r["axes"]["template_group"],
            r["axes"]["direction"],
            str(r["axes"]["pair_order"]),
        )
        for r in _val()
    }
    for r in _train():
        key = (
            r["kind"],
            r["axes"]["template_group"],
            r["axes"]["direction"],
            str(r["axes"]["pair_order"]),
        )
        assert key in frozen_axes, r["id"]
    assert {r["kind"] for r in _train()} == set(ce.KINDS)


def test_scene_order_is_permuted_not_sorted() -> None:
    orders = {tuple(ce.scene_order(r["seed"], 3)) for r in _train()}
    assert len(orders) > 1


def test_deterministic() -> None:
    rows = _train()[:30]
    a, b = ce.tensorize(rows), ce.tensorize(ce.training_rows(10))
    assert torch.equal(a["ticks"], b["ticks"])
    assert torch.equal(a["labels"], b["labels"])
