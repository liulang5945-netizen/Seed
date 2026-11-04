"""18.0 data premise: 220/220 name pools with byte-disjoint splits.

The 17 window closed as a negative result whose root cause is a data premise:
the frozen corpus contains only 3 object names x 5 place names, so 15
combinations can be recited and the copy operator never has to carry content.
This module builds the replacement premise required by option A:

* **unseen byte strings**: dev and confirm names are complete strings that never
  occur in any train record, goal or action.  The pools are disjoint by
  construction and the manifest counts occurrences instead of trusting it.
* **pools large enough that recitation is infeasible**: 220 object names
  (4 Han chars) and 220 place names (2 Han chars), generated from disjoint
  character pools; every name is unique.
* **combination held-out**: every dev/confirm (object, place) pair is absent from
  train, and the split is by name string first, so a memorised combination
  cannot leak across splits.

The byte interface is unchanged: goal 36 bytes, action 27 bytes, at most 512
source bytes.  Names are record bytes, not parameters, so the 3,563,859-parameter
container does not change.  Confirm examples are not generated; only the sealed
specification is written.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from taiji.living_core import EOS, RawSource

SCHEMA = "taiji-data-premise-18-0-v1"
DATA_SEED = 48051
ORDER_SEED = 48053
CONFIRM_SEED = 48052
NAME_SPLIT_SEED = 48054

OBJECT_NAME_CHARS = 4
PLACE_NAME_CHARS = 2
OBJECT_NAME_BYTES = 12
PLACE_NAME_BYTES = 6
GOAL_BYTES = 36
ACTION_BYTES = 27
MAX_SOURCE_BYTES = 512

SPLIT_SIZES: dict[str, int] = {"train": 160, "dev": 30, "confirm": 30}
SPLIT_WORLDS: dict[str, int] = {"train": 40, "dev": 12, "confirm": 12}
OBJECTS_PER_WORLD = 4
PLACES_PER_WORLD = 4
EXAMPLES_PER_WORLD = OBJECTS_PER_WORLD * PLACES_PER_WORLD
RECORD_LINES = OBJECTS_PER_WORLD + PLACES_PER_WORLD

# 20 x 11 = 220 object names (modifier 2 chars + noun 2 chars).
OBJECT_MODIFIERS: tuple[str, ...] = (
    "玄铁",
    "赤铜",
    "青玉",
    "墨色",
    "赭石",
    "夜蓝",
    "朱砂",
    "翠羽",
    "乌木",
    "银霜",
    "金漆",
    "玉髓",
    "铁英",
    "铜绿",
    "漆烟",
    "霜白",
    "烟青",
    "霞红",
    "松绿",
    "苔痕",
)
OBJECT_NOUNS: tuple[str, ...] = (
    "罗盘",
    "铃铛",
    "镇纸",
    "算筹",
    "铜镜",
    "竹简",
    "漆盒",
    "石砚",
    "香炉",
    "玉玦",
    "锡壶",
)

# 20 x 11 = 220 place names (prefix char + suffix char).
PLACE_PREFIXES: tuple[str, ...] = (
    "池",
    "廊",
    "阶",
    "瓮",
    "窗",
    "桥",
    "井",
    "舟",
    "亭",
    "篱",
    "圃",
    "崖",
    "溪",
    "沙",
    "竹",
    "梅",
    "兰",
    "松",
    "云",
    "雾",
)
PLACE_SUFFIXES: tuple[str, ...] = (
    "畔",
    "下",
    "前",
    "侧",
    "外",
    "尾",
    "沿",
    "角",
    "上",
    "东",
    "北",
)

# Descriptions are environment text, never names.  Train uses the five known
# object descriptions; dev/confirm add three new ones.
OBJECT_DESCRIPTIONS: tuple[str, ...] = (
    "细颈的瓶子",
    "带锁的匣子",
    "宽口的陶罐",
    "浅底的臼子",
    "束口的布袋",
)
NEW_OBJECT_DESCRIPTIONS: tuple[str, ...] = (
    "弧肩的坛子",
    "扁圆的盒子",
    "高足的盘子",
)
PLACE_DESCRIPTIONS: tuple[str, ...] = (
    "生火近旁",
    "屋檐底下",
    "书案上方",
    "屋角内侧",
    "楼梯入口",
)

# The frozen corpus.  No 18.0 name may occur inside any of these strings.
FROZEN_CORPUS: tuple[str, ...] = (
    "青釉瓷瓶",
    "铁灰木匣",
    "素白陶罐",
    "赭红石臼",
    "黛蓝布囊",
    "炉边",
    "檐下",
    "案头",
    "墙角",
    "梯口",
)


class DataPremiseError(RuntimeError):
    """The 18.0 material violates the declared premise; nothing is used."""


def object_names() -> tuple[str, ...]:
    names = tuple(modifier + noun for modifier in OBJECT_MODIFIERS for noun in OBJECT_NOUNS)
    if len(names) != 220 or len(set(names)) != 220:
        raise DataPremiseError("object name pool is not 220 unique names")
    return names


def place_names() -> tuple[str, ...]:
    names = tuple(prefix + suffix for prefix in PLACE_PREFIXES for suffix in PLACE_SUFFIXES)
    if len(names) != 220 or len(set(names)) != 220:
        raise DataPremiseError("place name pool is not 220 unique names")
    return names


def validate_pools() -> dict[str, Any]:
    objects = object_names()
    places = place_names()
    for name in objects:
        if len(name) != OBJECT_NAME_CHARS or len(name.encode("utf-8")) != OBJECT_NAME_BYTES:
            raise DataPremiseError(f"object name length {name!r}")
        if any(name in text for text in FROZEN_CORPUS):
            raise DataPremiseError(f"object name {name!r} occurs in the frozen corpus")
    for name in places:
        if len(name) != PLACE_NAME_CHARS or len(name.encode("utf-8")) != PLACE_NAME_BYTES:
            raise DataPremiseError(f"place name length {name!r}")
        if any(name in text for text in FROZEN_CORPUS):
            raise DataPremiseError(f"place name {name!r} occurs in the frozen corpus")
    for place in places:
        if any(place in obj for obj in objects):
            raise DataPremiseError(f"place name {place!r} is a substring of an object name")
    if set(objects) & set(places):
        raise DataPremiseError("object and place domains overlap")
    descriptions = OBJECT_DESCRIPTIONS + NEW_OBJECT_DESCRIPTIONS + PLACE_DESCRIPTIONS
    for description in descriptions:
        for name in objects + places:
            if name in description:
                raise DataPremiseError(f"name {name!r} occurs inside a description")
    for description in OBJECT_DESCRIPTIONS + NEW_OBJECT_DESCRIPTIONS:
        if len(description) != 5 or len(description.encode("utf-8")) != 15:
            raise DataPremiseError(f"object description length {description!r}")
    for description in PLACE_DESCRIPTIONS:
        if len(description) != 4 or len(description.encode("utf-8")) != 12:
            raise DataPremiseError(f"place description length {description!r}")
    for name in objects:
        if any(name[0] == description[0] for description in descriptions):
            raise DataPremiseError(
                f"object name {name[0]!r} shares a first char with a description"
            )
    for name in places:
        if any(name[0] == description[0] for description in descriptions):
            raise DataPremiseError(f"place name {name[0]!r} shares a first char with a description")
    return {
        "object_names": len(objects),
        "place_names": len(places),
        "object_name_bytes": OBJECT_NAME_BYTES,
        "place_name_bytes": PLACE_NAME_BYTES,
        "new_descriptions": list(NEW_OBJECT_DESCRIPTIONS),
        "frozen_corpus_refs": list(FROZEN_CORPUS),
    }


def split_pools() -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    objects = list(object_names())
    places = list(place_names())
    random.Random(f"{NAME_SPLIT_SEED}|objects").shuffle(objects)
    random.Random(f"{NAME_SPLIT_SEED}|places").shuffle(places)
    out: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    cursor_o = 0
    cursor_p = 0
    for split in ("train", "dev", "confirm"):
        size = SPLIT_SIZES[split]
        out[split] = (
            tuple(objects[cursor_o : cursor_o + size]),
            tuple(places[cursor_p : cursor_p + size]),
        )
        cursor_o += size
        cursor_p += size
    for left, right in (("train", "dev"), ("train", "confirm"), ("dev", "confirm")):
        if set(out[left][0]) & set(out[right][0]) or set(out[left][1]) & set(out[right][1]):
            raise DataPremiseError(f"name pools overlap between {left} and {right}")
    return out


def style_index(split: str, world: int, record_index: int) -> int:
    codes = {"train": 0, "dev": 1, "confirm": 2}
    return (world * 5 + record_index * 7 + codes[split] * 11) % 3


def style_line(description: str, name: str, style: int) -> str:
    if style == 0:
        return f"{description}是{name}。"
    if style == 1:
        return f"{name}又叫{description}。"
    if style == 2:
        return f"{description}对应{name}。"
    raise DataPremiseError(f"unknown style {style}")


def _world_descriptions(split: str, world: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if split == "train":
        object_descriptions = tuple(
            OBJECT_DESCRIPTIONS[(world + offset) % len(OBJECT_DESCRIPTIONS)]
            for offset in range(OBJECTS_PER_WORLD)
        )
    else:
        object_descriptions = (
            OBJECT_DESCRIPTIONS[world % len(OBJECT_DESCRIPTIONS)],
            OBJECT_DESCRIPTIONS[(world + 1) % len(OBJECT_DESCRIPTIONS)],
            NEW_OBJECT_DESCRIPTIONS[world % len(NEW_OBJECT_DESCRIPTIONS)],
            NEW_OBJECT_DESCRIPTIONS[(world + 1) % len(NEW_OBJECT_DESCRIPTIONS)],
        )
    place_descriptions = tuple(
        PLACE_DESCRIPTIONS[(world + offset) % len(PLACE_DESCRIPTIONS)]
        for offset in range(PLACES_PER_WORLD)
    )
    return object_descriptions, place_descriptions


def _record(
    split: str,
    world: int,
    position: int,
    kind: str,
    description: str,
    name: str,
    index: int,
) -> dict[str, Any]:
    style = style_index(split, world, index)
    token = hashlib.sha256(f"{SCHEMA}|{split}|{world}|{DATA_SEED}".encode()).hexdigest()[:12]
    raw = style_line(description, name, style)
    return {
        "source_id": f"cx18{token}-{position:02d}",
        "raw": raw,
        "style": style,
        "style_name": ("stated", "aka", "mapped")[style],
        "kind": kind,
        "description": description,
        "name": name,
    }


def build_world(split: str, world: int, pools: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]):
    if split not in ("train", "dev", "confirm"):
        raise DataPremiseError(f"unknown split {split!r}")
    if not 0 <= world < SPLIT_WORLDS[split]:
        raise DataPremiseError(f"world {world} outside split {split!r}")
    objects, places = pools[split]
    object_names = tuple(
        objects[(world * OBJECTS_PER_WORLD + offset) % len(objects)]
        for offset in range(OBJECTS_PER_WORLD)
    )
    place_names = tuple(
        places[(world * PLACES_PER_WORLD + offset) % len(places)]
        for offset in range(PLACES_PER_WORLD)
    )
    if len(set(object_names)) != OBJECTS_PER_WORLD or len(set(place_names)) != PLACES_PER_WORLD:
        raise DataPremiseError(f"{split}[{world}]: repeated name inside one world")
    object_descriptions, place_descriptions = _world_descriptions(split, world)

    lines: list[tuple[str, str, str, int]] = []
    for index, (description, name) in enumerate(
        zip(object_descriptions, object_names, strict=True)
    ):
        lines.append(("object", description, name, index))
    for offset, (description, name) in enumerate(zip(place_descriptions, place_names, strict=True)):
        lines.append(("place", description, name, offset + OBJECTS_PER_WORLD))
    order = list(range(len(lines)))
    random.Random(f"order|{split}|{world}|{ORDER_SEED}").shuffle(order)
    records = [
        _record(
            split,
            world,
            position,
            lines[line][0],
            lines[line][1],
            lines[line][2],
            lines[line][3],
        )
        for position, line in enumerate(order)
    ]
    record_keys = [record["description"] for record in records]
    object_rows = {
        record["description"]: index
        for index, record in enumerate(records)
        if record["kind"] == "object"
    }

    examples: list[dict[str, Any]] = []
    for object_description, object_name in zip(object_descriptions, object_names, strict=True):
        for place_description, place_name in zip(place_descriptions, place_names, strict=True):
            goal = f"让{object_description}到{place_description}。"
            action = f"把{object_name}放到{place_name}"
            if (
                len(goal.encode("utf-8")) != GOAL_BYTES
                or len(action.encode("utf-8")) != ACTION_BYTES
            ):
                raise DataPremiseError("goal/action byte length drift")
            examples.append(
                {
                    "index": len(examples),
                    "world": world,
                    "split": split,
                    "layer": "seen_combination" if split == "train" else "unseen_name",
                    "goal": goal,
                    "object_description": object_description,
                    "object_name": object_name,
                    "place_description": place_description,
                    "place_name": place_name,
                    "action": action,
                    "expected_hex": action.encode("utf-8").hex(),
                    "expected_bytes": ACTION_BYTES,
                    "selection_label": object_rows[object_description],
                    "record_keys": list(record_keys),
                    "records": records,
                    "name_seen": name_seen(object_name, pools),
                    "description_seen": description_seen(object_description, split),
                    "unseen_output_name": not name_seen(object_name, pools),
                    "new_description": not description_seen(object_description, split),
                    "unseen_combination": split != "train",
                    "both_new": (not name_seen(object_name, pools))
                    and (not description_seen(object_description, split)),
                }
            )
    total_record_bytes = sum(len(record["raw"].encode("utf-8")) for record in records)
    if total_record_bytes > MAX_SOURCE_BYTES:
        raise DataPremiseError(f"{split}[{world}]: source budget exceeded")
    if len(examples) != EXAMPLES_PER_WORLD:
        raise DataPremiseError(f"{split}[{world}]: example count drift")
    return {
        "split": split,
        "world": world,
        "object_names": list(object_names),
        "place_names": list(place_names),
        "object_descriptions": list(object_descriptions),
        "place_descriptions": list(place_descriptions),
        "records": records,
        "record_bytes": total_record_bytes,
        "examples": examples,
    }


def name_seen(name: str, pools: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]) -> bool:
    return name in pools["train"][0] or name in pools["train"][1]


def description_seen(description: str, split: str) -> bool:
    if split == "train":
        return True
    seen = set(OBJECT_DESCRIPTIONS) | set(PLACE_DESCRIPTIONS)
    return description in seen


def build_payload() -> dict[str, Any]:
    pools = split_pools()
    worlds: list[dict[str, Any]] = []
    for split in ("train", "dev"):
        for world in range(SPLIT_WORLDS[split]):
            worlds.append(build_world(split, world, pools))
    confirm = {
        "split": "confirm",
        "generated": False,
        "sealed": True,
        "worlds": SPLIT_WORLDS["confirm"],
        "examples": SPLIT_WORLDS["confirm"] * EXAMPLES_PER_WORLD,
        "object_names": list(pools["confirm"][0]),
        "place_names": list(pools["confirm"][1]),
        "object_descriptions": list(OBJECT_DESCRIPTIONS) + list(NEW_OBJECT_DESCRIPTIONS),
        "place_descriptions": list(PLACE_DESCRIPTIONS),
        "data_seed": CONFIRM_SEED,
        "note": "generated only after the candidate and evaluator are frozen",
    }
    return {
        "schema": SCHEMA,
        "data_seed": DATA_SEED,
        "order_seed": ORDER_SEED,
        "confirm_seed": CONFIRM_SEED,
        "name_split_seed": NAME_SPLIT_SEED,
        "split_sizes": dict(SPLIT_SIZES),
        "split_worlds": dict(SPLIT_WORLDS),
        "objects_per_world": OBJECTS_PER_WORLD,
        "places_per_world": PLACES_PER_WORLD,
        "examples_per_world": EXAMPLES_PER_WORLD,
        "record_lines": RECORD_LINES,
        "pool_validation": validate_pools(),
        "pools": {
            split: {"object_names": list(objs), "place_names": list(pls)}
            for split, (objs, pls) in pools.items()
        },
        "worlds": [
            {key: value for key, value in world.items() if key != "examples"} for world in worlds
        ],
        "examples": [example for world in worlds for example in world["examples"]],
        "confirm_spec": confirm,
    }


def fixtures(payload: dict[str, Any], split: str) -> tuple[dict, ...]:
    """Fixture dicts in the frozen training format, one per example."""
    by_world: dict[int, dict] = {}
    for world in payload["worlds"]:
        if world["split"] == split:
            by_world[world["world"]] = world
    out: list[dict] = []
    for example in payload["examples"]:
        if example["split"] != split:
            continue
        records = tuple(
            RawSource(source_id=record["source_id"], raw=record["raw"].encode("utf-8"))
            for record in example["records"]
        )
        out.append(
            {
                "index": int(example["index"]),
                "world": int(example["world"]),
                "goal": example["goal"],
                "records": records,
                "record_keys": list(example["record_keys"]),
                "selection_label": int(example["selection_label"]),
                "target_prefix": tuple(example["action"].encode("utf-8")),
                "target_bytes": tuple(example["action"].encode("utf-8")) + (EOS,),
                "flags": {
                    "name_seen": bool(example["name_seen"]),
                    "description_seen": bool(example["description_seen"]),
                    "unseen_output_name": bool(example["unseen_output_name"]),
                    "new_description": bool(example["new_description"]),
                    "unseen_combination": bool(example["unseen_combination"]),
                    "both_new": bool(example["both_new"]),
                },
                "object_description": example["object_description"],
                "place_description": example["place_description"],
                "object_name": example["object_name"],
                "place_name": example["place_name"],
                "raw_example": example,
            }
        )
    if len(out) != len(by_world) * EXAMPLES_PER_WORLD:
        raise DataPremiseError(f"{split}: fixture count drift")
    return tuple(out)


def split_proofs(payload: dict[str, Any]) -> dict[str, Any]:
    """Occurrence counts and combination intersections; zero means disjoint."""
    splits = ("train", "dev", "confirm")
    names: dict[str, dict[str, list[str]]] = {
        "train": {"object": [], "place": []},
        "dev": {"object": [], "place": []},
        "confirm": {"object": [], "place": []},
    }
    for split in splits:
        names[split]["object"] = list(payload["pools"][split]["object_names"])
        names[split]["place"] = list(payload["pools"][split]["place_names"])

    texts: dict[str, list[str]] = {"train": [], "dev": []}
    combos: dict[str, set[tuple[str, str]]] = {"train": set(), "dev": set()}
    for world in payload["worlds"]:
        split = world["split"]
        for record in world["records"]:
            texts[split].append(record["raw"])
        for example in [
            e for e in payload["examples"] if e["split"] == split and e["world"] == world["world"]
        ]:
            texts[split].append(example["goal"])
            texts[split].append(example["action"])
            combos[split].add((example["object_name"], example["place_name"]))
    occurrences: dict[str, dict[str, int]] = {}
    for target in splits:
        for source in ("train", "dev"):
            if target == source:
                continue
            for kind in ("object", "place"):
                key = f"{target}_{kind}_names_in_{source}"
                occurrences[key] = sum(
                    text.count(name) for name in names[target][kind] for text in texts[source]
                )
    intersection = sorted(set(combos["train"]) & set(combos["dev"]))
    return {
        "occurrences": occurrences,
        "zero_occurrence_keys": sorted(key for key, count in occurrences.items() if count == 0),
        "nonzero_occurrence_keys": sorted(key for key, count in occurrences.items() if count),
        "combination_intersection_train_dev": len(intersection),
        "combination_examples": {
            "train": len(combos["train"]),
            "dev": len(combos["dev"]),
        },
        "name_counts": {
            split: {
                "object": len(names[split]["object"]),
                "object_unique": len(set(names[split]["object"])),
                "place": len(names[split]["place"]),
                "place_unique": len(set(names[split]["place"])),
            }
            for split in splits
        },
    }


def stats(payload: dict[str, Any]) -> dict[str, Any]:
    train = [example for example in payload["examples"] if example["split"] == "train"]
    dev = [example for example in payload["examples"] if example["split"] == "dev"]
    return {
        "train_examples": len(train),
        "dev_examples": len(dev),
        "train_worlds": SPLIT_WORLDS["train"],
        "dev_worlds": SPLIT_WORLDS["dev"],
        "confirm_examples": payload["confirm_spec"]["examples"],
        "train_record_bytes": sum(
            world["record_bytes"] for world in payload["worlds"] if world["split"] == "train"
        ),
        "dev_record_bytes": sum(
            world["record_bytes"] for world in payload["worlds"] if world["split"] == "dev"
        ),
        "train_action_bytes": sum(example["expected_bytes"] for example in train),
        "dev_action_bytes": sum(example["expected_bytes"] for example in dev),
        "max_source_bytes": max(world["record_bytes"] for world in payload["worlds"]),
        "source_budget": MAX_SOURCE_BYTES,
    }


def digest(payload: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(payload["schema"].encode())
    for example in payload["examples"]:
        hasher.update(example["goal"].encode("utf-8"))
        hasher.update(example["expected_hex"].encode())
        hasher.update(example["object_name"].encode("utf-8"))
        hasher.update(example["place_name"].encode("utf-8"))
    return hasher.hexdigest()
