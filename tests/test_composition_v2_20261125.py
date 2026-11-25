"""Data tests for fixture ``taiji-23-composition-v2`` (N32.2.02, D-20261125-509).

Requires the stored scenes (written by scripts/training/n32_2_02_composition_v2_data.py).
Checks: manifest digests match the files; stored scenes regenerate bitwise; the confirm
split is sealed; 17 questions per world with 1/4/12 kinds and no opposite pairs; horizon
<= 24 and closed answer set; ticks decode back to question and scene, and an independent
oracle on the decoded ticks reproduces every answer; Q3 is not the v1 degenerate task;
world seeds are disjoint from 23.1 frozen worlds and from the v1 pool.

Run: python -m pytest tests/test_composition_v2_20261125.py -q
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402

from taiji import composition_encoding as ce  # noqa: E402
from taiji import native_composition_v2 as nc2  # noqa: E402
from taiji.native_composition import DIRECTIONS, NAMES  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_23_composition_v2_contract_20261125.json"
TRAIN_SAMPLE_WORLDS = 300


def _require_files() -> None:
    if not (nc2.SCENES_PATH.exists() and nc2.SEALED_PATH.exists() and MANIFEST.exists()):
        pytest.fail("run scripts/training/n32_2_02_composition_v2_data.py first")


@lru_cache(maxsize=1)
def _val() -> list[dict[str, Any]]:
    _require_files()
    return nc2.rows("val")


@lru_cache(maxsize=1)
def _train_sample() -> list[dict[str, Any]]:
    _require_files()
    return [r for r in nc2.rows("train") if int(r["k"]) < TRAIN_SAMPLE_WORLDS]


def _vocab(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({c for r in rows for c in r["text"]} | {n[0] for n in NAMES})


def _decode(ticks: torch.Tensor, table: torch.Tensor, vocab: list[str]) -> tuple[list, str]:
    bodies, text = [], []
    for row in ticks:
        if float(row[ce.SCENE_FLAG]) == 1.0:
            bodies.append(
                {"name": NAMES[int(row[:6].argmax())], "x": float(row[6]), "y": float(row[7])}
            )
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
    entity = next(n for n in NAMES if text.startswith(n))
    dirs = re.findall("|".join(DIRECTIONS), text)
    if not dirs:
        x = next(b["x"] for b in bodies if b["name"] == entity)
        return "q1", "左边" if x < -1 / 3 else "右边" if x > 1 / 3 else "中间"
    if len(dirs) == 1:
        return "q2", _nearest(bodies, entity, dirs[0]) or "未知"
    return "q3", _nearest(bodies, _nearest(bodies, entity, dirs[0]), dirs[1]) or "未知"


def test_manifest_digests_match_files() -> None:
    _require_files()
    digests = json.loads(MANIFEST.read_text(encoding="utf-8"))["criteria"]["digests"]
    open_sha = hashlib.sha256(nc2.SCENES_PATH.read_bytes()).hexdigest()
    assert open_sha == digests["open_file_sha256"]
    assert (
        hashlib.sha256(nc2.SEALED_PATH.read_bytes()).hexdigest() == digests["confirm_file_sha256"]
    )


def test_stored_scenes_regenerate_bitwise() -> None:
    _require_files()
    stored = {int(s["k"]): s for s in json.loads(nc2.SCENES_PATH.read_text(encoding="utf-8"))}
    for k in (0, 1, 2, 3, 4, 6000, 6999):
        fresh = nc2.generate_scenes(range(k, k + 1))[0]
        assert json.dumps(fresh, sort_keys=True) == json.dumps(stored[k], sort_keys=True), k


def test_confirm_split_sealed() -> None:
    _require_files()
    with pytest.raises(PermissionError):
        nc2.load_scenes("confirm")
    with pytest.raises(PermissionError):
        nc2.rows("confirm", confirmation_card="N32.2.03")
    stored = json.loads(nc2.SCENES_PATH.read_text(encoding="utf-8"))
    assert all(nc2.split_of(int(s["k"])) != "confirm" for s in stored)


def test_question_structure() -> None:
    assert len(nc2.PAIRS) == 12
    assert all(b != nc2.OPPOSITE[a] for a, b in nc2.PAIRS)
    by_world: dict[int, list[dict[str, Any]]] = {}
    for r in _val():
        by_world.setdefault(int(r["k"]), []).append(r)
    assert len(by_world) == 1000
    for qs in by_world.values():
        assert Counter(q["kind"] for q in qs) == {"q1": 1, "q2": 4, "q3": 12}
        assert len({q["text"] for q in qs}) == 17


def test_horizon_and_answer_set() -> None:
    rows = [*_val(), *_train_sample()]
    batch = ce.tensorize(rows)
    assert batch["ticks"].shape == (len(rows), ce.HORIZON, ce.INPUT_DIM)
    assert all(r["answer"] in ce.ANSWER_INDEX for r in rows)


def test_decode_and_independent_oracle() -> None:
    rows = [*_val(), *_train_sample()]
    vocab = _vocab(rows)
    table = torch.stack([ce.char_embedding(c) for c in vocab])
    for r in rows:
        bodies, text = _decode(ce.encode(r["scene"], r["text"]), table, vocab)
        assert text == r["text"], r["id"]
        kind, answer = _oracle(bodies, text)
        assert kind == r["kind"], r["id"]
        assert answer == r["answer"], r["id"]


def test_q3_not_degenerate() -> None:
    q3 = [r for r in _val() if r["kind"] == "q3"]
    v1_rule = sum(
        (r["entity"] if r["first_hop"] is not None else "未知") == r["answer"] for r in q3
    ) / len(q3)
    assert v1_rule < 0.60
    for pair in nc2.PAIRS:
        answers = {r["answer"] for r in q3 if tuple(r["axes"]["pair"]) == pair}
        assert len(answers) >= 3, pair


def test_world_seeds_disjoint() -> None:
    _require_files()
    seeds = [int(s["seed"]) for s in json.loads(nc2.SCENES_PATH.read_text(encoding="utf-8"))]
    seeds += [int(s["seed"]) for s in json.loads(nc2.SEALED_PATH.read_text(encoding="utf-8"))]
    assert len(set(seeds)) == nc2.N_WORLDS
    assert not any(ce.frozen_seed(s) for s in seeds)
    assert not any(ce.TRAIN_SEED_BASE <= s < ce.TRAIN_SEED_BASE + 12_000 for s in seeds)
