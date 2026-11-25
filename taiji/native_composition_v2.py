"""23 composition fixture v2 for research line N32.2 (R1): ``taiji-23-composition-v2``.

Why v2 (D-20261125-508): in v1 the train-style two-hop question only used the opposite
pairs (left, right) / (right, left); the nearest left neighbour's nearest right neighbour
is always X, so "answer X if X has a d1-neighbour, else unknown" solved v1 Q3 exactly.

v2 per world (one entity X chosen by the world seed), 17 questions:
  Q1  position            "{x}现在在哪里？"                       x1
  Q2  one-hop relation    "{x}的{d}是谁？"            d in 4 directions   x4
  Q3  two-hop relation    "{x}{d1}的那个物体的{d2}是谁？"  12 ordered pairs, opposite
                                                             pairs excluded   x12
Answers are computed with the v1 geometry (``_relation`` / ``_relative_position``).

Worlds: seeds ``WORLD_SEED_BASE + k`` (k < 8,000), disjoint from the 23.1 frozen worlds.
Splits by k: train [0, 6000), val [6000, 7000), confirm [7000, 8000).  Scenes are stored
once in ``tests/fixtures``; the confirm split lives in a separate sealed file that only a
confirmation card may open.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from taiji.native_composition import DIRECTIONS, NAMES, _relation, _relative_position, world_scene

FIXTURE_ID = "taiji-23-composition-v2"
ROOT = Path(__file__).resolve().parents[1]
SCENES_PATH = ROOT / "tests/fixtures/taiji_23_composition_v2_scenes.json"
SEALED_PATH = ROOT / "tests/fixtures/taiji_23_composition_v2_confirm_sealed.json"
WORLD_SEED_BASE = 23_200_000
N_WORLDS = 8_000
SPLITS = {"train": (0, 6_000), "val": (6_000, 7_000), "confirm": (7_000, 8_000)}
SEALED_SPLITS = ("confirm",)
OPPOSITE = {"左边": "右边", "右边": "左边", "前边": "后边", "后边": "前边"}
PAIRS: tuple[tuple[str, str], ...] = tuple(
    (a, b) for a in DIRECTIONS for b in DIRECTIONS if b != OPPOSITE[a]
)
TEMPLATES = {
    "q1": "{x}现在在哪里？",
    "q2": "{x}的{d}是谁？",
    "q3": "{x}{d1}的那个物体的{d2}是谁？",
}
KINDS = ("q1", "q2", "q3")


def world_seed(k: int) -> int:
    return WORLD_SEED_BASE + k


def split_of(k: int) -> str:
    for name, (lo, hi) in SPLITS.items():
        if lo <= k < hi:
            return name
    raise ValueError(f"world index {k} outside 0..{N_WORLDS - 1}")


def _compact(scene: dict[str, Any]) -> dict[str, Any]:
    bodies = [
        {"name": str(b["name"]), "position": [float(v) for v in list(b["position"])[:3]]}
        for b in scene["bodies"]
    ]
    return {"seed": int(scene["seed"]), "bodies": bodies}


def generate_scenes(indices: range) -> list[dict[str, Any]]:
    return [_compact(world_scene(world_seed(k))) | {"k": k} for k in indices]


def scenes_digest(scenes: list[dict[str, Any]]) -> str:
    payload = json.dumps(scenes, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_scenes() -> dict[str, str]:
    """Generate all 8,000 worlds once; returns the digests of the two stored files."""
    open_scenes = generate_scenes(range(*SPLITS["train"])) + generate_scenes(range(*SPLITS["val"]))
    sealed = generate_scenes(range(*SPLITS["confirm"]))
    for path, scenes in ((SCENES_PATH, open_scenes), (SEALED_PATH, sealed)):
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(scenes, sort_keys=True, ensure_ascii=False)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return {"open": scenes_digest(open_scenes), "confirm": scenes_digest(sealed)}


def load_scenes(split: str, *, confirmation_card: str | None = None) -> list[dict[str, Any]]:
    if split in SEALED_SPLITS:
        if not (confirmation_card and confirmation_card.endswith((".90", ".91", ".92"))):
            raise PermissionError(f"split {split!r} is sealed; only a confirmation card opens it")
        path = SEALED_PATH
    elif split in SPLITS:
        path = SCENES_PATH
    else:
        raise ValueError(f"unknown split {split!r}")
    scenes: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    lo, hi = SPLITS[split]
    return [s for s in scenes if lo <= int(s["k"]) < hi]


def entity_of(scene: dict[str, Any]) -> str:
    names = [str(b["name"]) for b in scene["bodies"]]
    return names[int(scene["seed"]) % len(names)]


def questions(scene: dict[str, Any]) -> list[dict[str, Any]]:
    bodies = scene["bodies"]
    x = entity_of(scene)
    out: list[dict[str, Any]] = [
        {
            "kind": "q1",
            "text": TEMPLATES["q1"].format(x=x),
            "answer": _relative_position(x, bodies),
            "axes": {"d": None, "pair": None},
        }
    ]
    for d in DIRECTIONS:
        target = _relation(x, d, bodies)
        out.append(
            {
                "kind": "q2",
                "text": TEMPLATES["q2"].format(x=x, d=d),
                "answer": target or "未知",
                "axes": {"d": d, "pair": None},
            }
        )
    for d1, d2 in PAIRS:
        first = _relation(x, d1, bodies)
        second = _relation(first, d2, bodies) if first is not None else None
        out.append(
            {
                "kind": "q3",
                "text": TEMPLATES["q3"].format(x=x, d1=d1, d2=d2),
                "answer": second or "未知",
                "axes": {"d": d1, "pair": [d1, d2]},
                "first_hop": first,
            }
        )
    for q in out:
        q["entity"] = x
    return out


def rows(split: str, *, confirmation_card: str | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scene in load_scenes(split, confirmation_card=confirmation_card):
        for i, q in enumerate(questions(scene)):
            result.append(
                q
                | {
                    "id": f"{split}-{scene['k']}-{i}",
                    "seed": scene["seed"],
                    "k": scene["k"],
                    "split": split,
                    "scene": scene,
                }
            )
    return result


def name_index(name: str) -> int:
    return NAMES.index(name)
