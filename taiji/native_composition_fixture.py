"""10.21.3 composition fixture: deterministic records-mapped goal data.

One world is a mapping world. The goal text names no entity; the only bridge from
a description to an entity name is the record set::

    goal    = "让方口的容器到靠墙高处。"      (36 bytes, descriptions only)
    records = ("方口的容器是棕漆木箱。",      (33 bytes)
               "靠墙高处是窗台。")            (24 bytes)
    action  = "把棕漆木箱放到窗台"            (27 bytes)

Each world permutes the split's description list onto the split's name list
without replacement, separately inside the object domain and inside the place
domain, so the same goal text has different true answers across worlds and the
two domains never cross. Training keeps the 12 of 3x5 combinations that are off
the (0,0)/(1,1)/(2,2) diagonal.

This module is data only. It reads nothing, writes nothing, trains nothing, and
makes no claim: it builds the declared sample sets and refuses the whole batch if
any record line, goal or action is not exactly the declared byte length.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from taiji.living_core import RawSource

SCHEMA = "taiji-composition-fixture-v1"

DATA_SEED = 47021
ORDER_SEED = 27021
CONFIRM_SEED = 47022
CONFIRM_ORDER_SEED = 27022

SPLIT_WORLDS: dict[str, int] = {"train": 16, "dev": 8, "confirm": 8}
SPLIT_SEEDS: dict[str, tuple[int, int]] = {
    "train": (DATA_SEED, ORDER_SEED),
    "dev": (DATA_SEED, ORDER_SEED),
    "confirm": (CONFIRM_SEED, CONFIRM_ORDER_SEED),
}

OBJECT_NAMES: tuple[str, ...] = ("棕漆木箱", "银灰铁盒", "深绿陶碗", "米白瓷碟", "墨黑石砚")
OBJECT_DESCRIPTIONS: tuple[str, ...] = (
    "方口的容器",
    "圆盖的器具",
    "窄沿的坛子",
    "有纹的磁盘",
    "沉手的石块",
)
PLACE_NAMES: tuple[str, ...] = ("窗台", "书架", "门后", "廊下", "柜顶")
PLACE_DESCRIPTIONS: tuple[str, ...] = ("靠墙高处", "向阳角上", "进门右侧", "屋角一侧", "高处横板")

TRAIN_OBJECT_COUNT = 3
HELD_OUT_OBJECT: dict[str, int] = {"dev": 3, "confirm": 4}
PLACE_COUNT = len(PLACE_NAMES)

GOAL_BYTES = 36
ACTION_BYTES = 27
OBJECT_NAME_BYTES = 12
PLACE_NAME_BYTES = 6
TRAIN_RECORD_LINES = 8
TRAIN_RECORD_BYTES = 219
DEV_RECORD_LINES = 9
DEV_RECORD_BYTES = 252
MAX_SOURCE_BYTES = 512

SEEN_PER_WORLD = 12
UNSEEN_COMBINATION_PER_WORLD = 3
UNSEEN_ENTITY_PER_WORLD = 5

# Declared before any timing: the 8 of the 12 training combinations the price
# stage consumes, chosen so all three training objects appear. Indices address
# TRAIN_COMBINATIONS, not the 3x5 grid.
PRICE_TRAIN_PICK: tuple[int, ...] = (0, 1, 2, 4, 5, 6, 8, 10)
# Declared before any timing: 4 seen combinations, 2 unseen combinations and 2
# unseen entity examples, so the 8 developmental forwards are not one layer only.
PRICE_DEV_SEEN_PICK: tuple[int, ...] = (0, 3, 6, 9)
PRICE_DEV_UNSEEN_COMBINATION_PICK: tuple[int, ...] = (0, 1)
PRICE_DEV_UNSEEN_ENTITY_PICK: tuple[int, ...] = (0, 1)

TRAIN_COMBINATIONS: tuple[tuple[int, int], ...] = tuple(
    (object_index, place_index)
    for object_index in range(TRAIN_OBJECT_COUNT)
    for place_index in range(PLACE_COUNT)
    if object_index != place_index
)
DIAGONAL_COMBINATIONS: tuple[tuple[int, int], ...] = tuple(
    (index, index) for index in range(TRAIN_OBJECT_COUNT)
)


class FixtureError(RuntimeError):
    """The generated batch violates the declared data contract; nothing is used."""


@dataclass(frozen=True)
class FixtureExample:
    """One goal, its world records and the full ranked candidate list."""

    index: int
    world_index: int
    layer: str
    goal: str
    object_description: str
    place_description: str
    object_text: str
    place_text: str
    action: str
    actions: tuple[tuple[str, str], ...]
    target: int
    records: tuple[RawSource, ...]

    @property
    def goal_bytes(self) -> bytes:
        return self.goal.encode()

    def payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "world_index": self.world_index,
            "layer": self.layer,
            "goal": self.goal,
            "object_text": self.object_text,
            "place_text": self.place_text,
            "action": self.action,
            "target": self.target,
            "action_count": len(self.actions),
            "actions": [list(pair) for pair in self.actions],
            "records": [
                {"source_id": record.source_id, "raw": record.raw.decode("utf-8")}
                for record in self.records
            ],
        }


@dataclass(frozen=True)
class FixtureWorld:
    """One records mapping world plus its declared sample list."""

    split: str
    world: int
    data_seed: int
    order_seed: int
    object_descriptions: tuple[str, ...]
    object_names: tuple[str, ...]
    place_descriptions: tuple[str, ...]
    place_names: tuple[str, ...]
    records: tuple[RawSource, ...]
    examples: tuple[FixtureExample, ...]

    def assignment(self, description: str) -> str:
        position = self.object_descriptions.index(description)
        return self.object_names[position]

    def place_assignment(self, description: str) -> str:
        position = self.place_descriptions.index(description)
        return self.place_names[position]

    def actions(self) -> tuple[tuple[str, str], ...]:
        ordered_objects = sorted(self.object_names, key=OBJECT_NAMES.index)
        ordered_places = sorted(self.place_names, key=PLACE_NAMES.index)
        return tuple(
            (object_text, place_text)
            for object_text in ordered_objects
            for place_text in ordered_places
        )

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "world": self.world,
            "data_seed": self.data_seed,
            "order_seed": self.order_seed,
            "object_mapping": dict(zip(self.object_descriptions, self.object_names, strict=True)),
            "place_mapping": dict(zip(self.place_descriptions, self.place_names, strict=True)),
            "record_lines": len(self.records),
            "record_bytes": sum(len(record.raw) for record in self.records),
            "examples": len(self.examples),
            "layers": _layer_counts(self.examples),
            "actions": len(self.actions()),
            "record_ids": [record.source_id for record in self.records],
        }


def _layer_counts(examples: tuple[FixtureExample, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.layer] = counts.get(example.layer, 0) + 1
    return counts


def split_objects(split: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Name and description lists for one split, as parallel tuples."""
    if split == "train":
        size = TRAIN_OBJECT_COUNT
    elif split in HELD_OUT_OBJECT:
        size = HELD_OUT_OBJECT[split] + 1
    else:
        raise FixtureError(f"unknown split {split!r}")
    return OBJECT_DESCRIPTIONS[:size], OBJECT_NAMES[:size]


def _permutation(
    descriptions: tuple[str, ...], names: tuple[str, ...], seed_key: str
) -> tuple[str, ...]:
    """Permute names onto descriptions without replacement."""
    if len(descriptions) != len(names):
        raise FixtureError("object/place domain sizes differ")
    order = list(range(len(names)))
    random.Random(seed_key).shuffle(order)
    return tuple(names[position] for position in order)


def _record_token(data_seed: int, split: str, world: int) -> str:
    """Opaque, versioned identity: the split is hashed, not encoded."""
    return hashlib.sha256(f"{SCHEMA}|{data_seed}|{split}|{world}".encode()).hexdigest()[:12]


def build_world(split: str, world: int) -> FixtureWorld:
    if split not in SPLIT_SEEDS:
        raise FixtureError(f"unknown split {split!r}")
    if not 0 <= world < SPLIT_WORLDS[split]:
        raise FixtureError(f"world {world} outside split {split!r}")
    data_seed, order_seed = SPLIT_SEEDS[split]
    object_descriptions, all_object_names = split_objects(split)
    object_list = OBJECT_NAMES[: len(object_descriptions)]
    place_descriptions = PLACE_DESCRIPTIONS
    place_list = PLACE_NAMES

    object_names = _permutation(
        object_descriptions, object_list, f"object|{split}|{world}|{data_seed}"
    )
    place_names = _permutation(place_descriptions, place_list, f"place|{split}|{world}|{data_seed}")
    if len(set(object_names)) != len(object_names):
        raise FixtureError("object mapping repeats a name")
    if len(set(place_names)) != len(place_names):
        raise FixtureError("place mapping repeats a name")
    if set(object_names) & set(place_names):
        raise FixtureError("object and place domains overlap")

    lines = [
        f"{description}是{name}。"
        for description, name in zip(object_descriptions, object_names, strict=True)
    ]
    lines += [
        f"{description}是{name}。"
        for description, name in zip(place_descriptions, place_names, strict=True)
    ]
    order = list(range(len(lines)))
    random.Random(f"order|{split}|{world}|{order_seed}").shuffle(order)
    token = _record_token(data_seed, split, world)
    records = tuple(
        RawSource(source_id=f"cx{token}-{position:02d}", raw=lines[line].encode())
        for position, line in enumerate(order)
    )

    actions = _actions_for(object_names, place_names)
    example_rows: list[FixtureExample] = []
    for layer, pairs in _example_combinations(split, object_descriptions):
        for object_index, place_index in pairs:
            example_rows.append(
                _make_example(
                    len(example_rows),
                    world,
                    layer,
                    object_descriptions[object_index],
                    place_descriptions[place_index],
                    object_names[object_index],
                    place_names[place_index],
                    actions,
                    records,
                )
            )
    fixture_world = FixtureWorld(
        split=split,
        world=world,
        data_seed=data_seed,
        order_seed=order_seed,
        object_descriptions=object_descriptions,
        object_names=object_names,
        place_descriptions=place_descriptions,
        place_names=place_names,
        records=records,
        examples=tuple(example_rows),
    )
    validate_world(fixture_world)
    return fixture_world


def _actions_for(
    object_names: tuple[str, ...], place_names: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    ordered_objects = sorted(object_names, key=OBJECT_NAMES.index)
    ordered_places = sorted(place_names, key=PLACE_NAMES.index)
    return tuple(
        (object_text, place_text)
        for object_text in ordered_objects
        for place_text in ordered_places
    )


def _example_combinations(
    split: str, object_descriptions: tuple[str, ...]
) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    if split == "train":
        return (("seen_combination", TRAIN_COMBINATIONS),)
    held_out = len(object_descriptions) - 1
    return (
        ("seen_combination", TRAIN_COMBINATIONS),
        ("unseen_combination", DIAGONAL_COMBINATIONS),
        (
            "unseen_entity",
            tuple((held_out, place_index) for place_index in range(PLACE_COUNT)),
        ),
    )


def _make_example(
    index: int,
    world: int,
    layer: str,
    object_description: str,
    place_description: str,
    object_text: str,
    place_text: str,
    actions: tuple[tuple[str, str], ...],
    records: tuple[RawSource, ...],
) -> FixtureExample:
    goal = f"让{object_description}到{place_description}。"
    action = f"把{object_text}放到{place_text}"
    try:
        target = actions.index((object_text, place_text))
    except ValueError as exc:  # pragma: no cover - guarded by construction
        raise FixtureError("true action missing from the candidate list") from exc
    return FixtureExample(
        index=index,
        world_index=world,
        layer=layer,
        goal=goal,
        object_description=object_description,
        place_description=place_description,
        object_text=object_text,
        place_text=place_text,
        action=action,
        actions=actions,
        target=target,
        records=records,
    )


def validate_world(world: FixtureWorld) -> None:
    """Refuse the batch on the first deviation; never truncate or drop a line."""
    prefix = f"{world.split}[{world.world}]"
    object_count = len(world.object_descriptions)
    expected_examples = (
        SEEN_PER_WORLD
        if world.split == "train"
        else SEEN_PER_WORLD + UNSEEN_COMBINATION_PER_WORLD + UNSEEN_ENTITY_PER_WORLD
    )
    if len(world.examples) != expected_examples:
        raise FixtureError(
            f"{prefix}: {len(world.examples)} examples, expected {expected_examples}"
        )
    handled = [record.raw for record in world.records]
    if len({record.source_id for record in world.records}) != len(world.records):
        raise FixtureError(f"{prefix}: duplicate record identity")
    if any(not record.source_id.startswith("cx") for record in world.records):
        raise FixtureError(f"{prefix}: record identity is not the opaque form")
    total_bytes = sum(len(line) for line in handled)
    lines_expected = TRAIN_RECORD_LINES if world.split == "train" else DEV_RECORD_LINES
    bytes_expected = TRAIN_RECORD_BYTES if world.split == "train" else DEV_RECORD_BYTES
    if len(handled) != lines_expected or total_bytes != bytes_expected:
        raise FixtureError(
            f"{prefix}: records {len(handled)}/{total_bytes}B, expected "
            f"{lines_expected}/{bytes_expected}B"
        )
    if total_bytes > MAX_SOURCE_BYTES:
        raise FixtureError(f"{prefix}: source budget exceeded")
    for line in handled:
        if not line.endswith("。".encode()) or line.count("是".encode()) != 1:
            raise FixtureError(f"{prefix}: record is not '描述是实体。'")
    for description, name in zip(world.object_descriptions, world.object_names, strict=True):
        _validate_pair(prefix, description, name, 5, 4)
    for description, name in zip(world.place_descriptions, world.place_names, strict=True):
        _validate_pair(prefix, description, name, 4, 2)
    for description in world.object_descriptions:
        if len(description) != 5 or len(description.encode()) != 15:
            raise FixtureError(f"{prefix}: object description length {description!r}")
    for description in world.place_descriptions:
        if len(description) != 4 or len(description.encode()) != 12:
            raise FixtureError(f"{prefix}: place description length {description!r}")
    actions = world.actions()
    if len(actions) != object_count * PLACE_COUNT:
        raise FixtureError(f"{prefix}: {len(actions)} candidate actions")
    for object_text, place_text in actions:
        if len(object_text.encode()) != OBJECT_NAME_BYTES:
            raise FixtureError(f"{prefix}: object name length {object_text!r}")
        if len(place_text.encode()) != PLACE_NAME_BYTES:
            raise FixtureError(f"{prefix}: place name length {place_text!r}")
        if len(f"把{object_text}放到{place_text}".encode()) != ACTION_BYTES:
            raise FixtureError(f"{prefix}: canonical action is not {ACTION_BYTES} bytes")
    for example in world.examples:
        if len(example.goal.encode()) != GOAL_BYTES:
            raise FixtureError(f"{prefix}: goal is not {GOAL_BYTES} bytes: {example.goal!r}")
        if example.actions != actions:
            raise FixtureError(f"{prefix}: example {example.index} action list drift")
        if actions[example.target] != (example.object_text, example.place_text):
            raise FixtureError(f"{prefix}: example {example.index} target mismatch")
        if example.action != f"把{example.object_text}放到{example.place_text}":
            raise FixtureError(f"{prefix}: example {example.index} action text")
        if example.records != world.records:
            raise FixtureError(f"{prefix}: example {example.index} records drift")
    _validate_goal_entity_free(prefix, world)


def _validate_pair(
    prefix: str, description: str, name: str, description_chars: int, name_chars: int
) -> None:
    if len(description) != description_chars or len(name) != name_chars:
        raise FixtureError(f"{prefix}: pair {description!r}/{name!r} length")
    if description[0] == name[0]:
        raise FixtureError(f"{prefix}: {description!r} shares its first character with {name!r}")
    if name in description:
        raise FixtureError(f"{prefix}: {name!r} appears inside {description!r}")


def _validate_goal_entity_free(prefix: str, world: FixtureWorld) -> None:
    all_names = set(OBJECT_NAMES) | set(PLACE_NAMES)
    for example in world.examples:
        for name in all_names:
            if name in example.goal:
                raise FixtureError(f"{prefix}: goal names the entity {name!r}")
        for name in world.object_names + world.place_names:
            if name in example.goal:
                raise FixtureError(f"{prefix}: goal contains a world entity name")


def build_split(split: str, worlds: int | None = None) -> tuple[FixtureWorld, ...]:
    """Build a split; any violation is raised before a caller can use one world."""
    count = SPLIT_WORLDS[split] if worlds is None else worlds
    if count < 1 or count > SPLIT_WORLDS[split]:
        raise FixtureError(f"split {split!r} has {SPLIT_WORLDS[split]} declared worlds")
    if split == "confirm":
        raise FixtureError("confirm material is sealed until the candidate is frozen")
    return tuple(build_world(split, world) for world in range(count))


def price_train_examples(world: FixtureWorld) -> tuple[FixtureExample, ...]:
    """The declared 8 training examples for the price stage."""
    seen = tuple(example for example in world.examples if example.layer == "seen_combination")
    if len(seen) != SEEN_PER_WORLD:
        raise FixtureError("training world does not carry 12 seen combinations")
    return tuple(seen[position] for position in PRICE_TRAIN_PICK)


def price_dev_examples(world: FixtureWorld) -> tuple[FixtureExample, ...]:
    """The declared 8 developmental examples: 4/2/2 across the three layers."""
    if world.split != "dev":
        raise FixtureError("price development reading needs a dev world")
    seen = tuple(example for example in world.examples if example.layer == "seen_combination")
    unseen = tuple(example for example in world.examples if example.layer == "unseen_combination")
    entity = tuple(example for example in world.examples if example.layer == "unseen_entity")
    picked = (
        tuple(seen[position] for position in PRICE_DEV_SEEN_PICK)
        + tuple(unseen[position] for position in PRICE_DEV_UNSEEN_COMBINATION_PICK)
        + tuple(entity[position] for position in PRICE_DEV_UNSEEN_ENTITY_PICK)
    )
    if len(picked) != 8:
        raise FixtureError("development price reading must hold exactly 8 examples")
    return picked


def fixture_payload(worlds: tuple[FixtureWorld, ...]) -> dict[str, Any]:
    """Audit payload for the manifest: identities, layer counts, byte totals."""
    return {
        "schema": SCHEMA,
        "data_seed": DATA_SEED,
        "order_seed": ORDER_SEED,
        "confirm_seed": CONFIRM_SEED,
        "confirm_order_seed": CONFIRM_ORDER_SEED,
        "declared_worlds": dict(SPLIT_WORLDS),
        "lengths": {
            "goal_bytes": GOAL_BYTES,
            "action_bytes": ACTION_BYTES,
            "train_records": [TRAIN_RECORD_LINES, TRAIN_RECORD_BYTES],
            "dev_records": [DEV_RECORD_LINES, DEV_RECORD_BYTES],
        },
        "vocabulary": {
            "object_names": list(OBJECT_NAMES),
            "object_descriptions": list(OBJECT_DESCRIPTIONS),
            "place_names": list(PLACE_NAMES),
            "place_descriptions": list(PLACE_DESCRIPTIONS),
            "train_objects": TRAIN_OBJECT_COUNT,
            "held_out_object": dict(HELD_OUT_OBJECT),
        },
        "combinations": {
            "seen_per_world": SEEN_PER_WORLD,
            "unseen_combination_per_world": UNSEEN_COMBINATION_PER_WORLD,
            "unseen_entity_per_world": UNSEEN_ENTITY_PER_WORLD,
            "train_kept": [list(pair) for pair in TRAIN_COMBINATIONS],
            "diagonal_excluded": [list(pair) for pair in DIAGONAL_COMBINATIONS],
        },
        "price_picks": {
            "train": list(PRICE_TRAIN_PICK),
            "dev_seen": list(PRICE_DEV_SEEN_PICK),
            "dev_unseen_combination": list(PRICE_DEV_UNSEEN_COMBINATION_PICK),
            "dev_unseen_entity": list(PRICE_DEV_UNSEEN_ENTITY_PICK),
        },
        "worlds": [world.summary() for world in worlds],
        "digest": fixture_digest(worlds),
    }


def fixture_digest(worlds: tuple[FixtureWorld, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(SCHEMA.encode())
    for world in worlds:
        digest.update(world.split.encode())
        digest.update(str(world.world).encode())
        digest.update(
            repr(tuple(zip(world.object_descriptions, world.object_names, strict=True))).encode()
        )
        digest.update(
            repr(tuple(zip(world.place_descriptions, world.place_names, strict=True))).encode()
        )
        for record in world.records:
            digest.update(record.source_id.encode())
            digest.update(record.raw)
        for example in world.examples:
            digest.update(example.goal.encode())
            digest.update(str(example.target).encode())
    return digest.hexdigest()
