"""15.1 rev2 data: unseen output names, three record styles and separate axes.

The generator materializes the 24 train / 12 dev worlds (288 / 240 targets)
and declares the sealed confirm specification.  It does not generate confirm
examples or records.  Names and descriptions are environment/scorer materials;
the model interface remains 256 bytes plus EOS.  Style IDs and source IDs never
enter the forward path.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

SCHEMA = "taiji-binding-copy-data-v2"

DATA_SEED = 48021
ORDER_SEED = 48023
CONFIRM_SEED = 48022
CONFIRM_ORDER_SEED = 48024

SPLIT_WORLDS: dict[str, int] = {"train": 24, "dev": 12, "confirm": 12}
SPLIT_SEEDS: dict[str, tuple[int, int]] = {
    "train": (DATA_SEED, ORDER_SEED),
    "dev": (DATA_SEED, ORDER_SEED),
    "confirm": (CONFIRM_SEED, CONFIRM_ORDER_SEED),
}

OBJECT_NAMES: tuple[str, ...] = (
    "青釉瓷瓶",
    "铁灰木匣",
    "素白陶罐",
    "赭红石臼",
    "黛蓝布囊",
)
OBJECT_DESCRIPTIONS: tuple[str, ...] = (
    "细颈的瓶子",
    "带锁的匣子",
    "宽口的陶罐",
    "浅底的臼子",
    "束口的布袋",
)
PLACE_NAMES: tuple[str, ...] = ("炉边", "檐下", "案头", "墙角", "梯口")
PLACE_DESCRIPTIONS: tuple[str, ...] = (
    "生火近旁",
    "屋檐底下",
    "书案上方",
    "屋角内侧",
    "楼梯入口",
)

TRAIN_OBJECT_INDICES: tuple[int, ...] = (0, 1, 2)
DEV_OBJECT_INDICES: tuple[int, ...] = (0, 1, 2, 3)
CONFIRM_OBJECT_INDICES: tuple[int, ...] = (0, 1, 2, 4)
TRAIN_OBJECT_COUNT = len(TRAIN_OBJECT_INDICES)
DEV_OBJECT_COUNT = len(DEV_OBJECT_INDICES)
CONFIRM_OBJECT_COUNT = len(CONFIRM_OBJECT_INDICES)
PLACE_COUNT = len(PLACE_NAMES)

GOAL_BYTES = 36
ACTION_BYTES = 27
OBJECT_NAME_BYTES = 12
OBJECT_DESCRIPTION_BYTES = 15
PLACE_NAME_BYTES = 6
PLACE_DESCRIPTION_BYTES = 12

TRAIN_RECORD_LINES = 8
DEV_RECORD_LINES = 9
CONFIRM_RECORD_LINES = 9
TRAIN_RECORD_MAX_BYTES = 243
DEV_RECORD_MAX_BYTES = 279
CONFIRM_RECORD_MAX_BYTES = 279
MAX_SOURCE_BYTES = 512

SEEN_PER_WORLD = 12
UNSEEN_COMBINATION_PER_WORLD = 3
NEW_DESCRIPTION_PER_WORLD = 5
DEV_EXAMPLES_PER_WORLD = 20

STYLES: tuple[str, ...] = ("stated", "aka", "mapped")
STYLE_CODES: dict[str, int] = {"train": 0, "dev": 1, "confirm": 2}

TRAIN_COMBINATIONS: tuple[tuple[int, int], ...] = tuple(
    (object_index, place_index)
    for object_index in range(TRAIN_OBJECT_COUNT)
    for place_index in range(PLACE_COUNT)
    if object_index != place_index
)
DIAGONAL_COMBINATIONS: tuple[tuple[int, int], ...] = tuple(
    (index, index) for index in range(TRAIN_OBJECT_COUNT)
)

AXIS_TARGET: dict[str, int] = {
    "seen_description_seen_name": 135,
    "new_description_seen_name": 45,
    "seen_description_new_name": 45,
    "new_description_new_name": 15,
    "new_description": 60,
    "unseen_output_name": 60,
    "both_new": 15,
    "unseen_combination": 36,
    "seen_combination": 144,
}
LAYER_TARGET: dict[str, int] = {
    "seen_combination": 144,
    "unseen_combination": 36,
    "new_description": 60,
}


class DataError(RuntimeError):
    """The v2 material violates the declared contract; nothing is used."""


@dataclass(frozen=True)
class BindingRecord:
    source_id: str
    raw: bytes
    style: int
    kind: str
    description: str
    name: str

    def payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "raw": self.raw.decode("utf-8"),
            "style": self.style,
            "style_name": STYLES[self.style],
            "kind": self.kind,
            "description": self.description,
            "name": self.name,
        }


@dataclass(frozen=True)
class BindingExample:
    index: int
    world: int
    split: str
    layer: str
    goal: str
    object_description: str
    object_name: str
    place_description: str
    place_name: str
    action: str
    expected: bytes
    target: int
    actions: tuple[str, ...]
    records: tuple[BindingRecord, ...]
    description_seen: bool
    name_seen: bool
    unseen_output_name: bool
    both_new: bool
    new_description: bool
    unseen_combination: bool

    @property
    def goal_bytes(self) -> bytes:
        return self.goal.encode()

    def payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "world": self.world,
            "split": self.split,
            "layer": self.layer,
            "goal": self.goal,
            "object_description": self.object_description,
            "object_name": self.object_name,
            "place_description": self.place_description,
            "place_name": self.place_name,
            "action": self.action,
            "expected_hex": self.expected.hex(),
            "expected_bytes": len(self.expected),
            "target": self.target,
            "actions": list(self.actions),
            "records": [record.payload() for record in self.records],
            "description_seen": self.description_seen,
            "name_seen": self.name_seen,
            "unseen_output_name": self.unseen_output_name,
            "both_new": self.both_new,
            "new_description": self.new_description,
            "unseen_combination": self.unseen_combination,
        }


@dataclass(frozen=True)
class BindingWorld:
    split: str
    world: int
    data_seed: int
    order_seed: int
    object_indices: tuple[int, ...]
    object_descriptions: tuple[str, ...]
    object_names: tuple[str, ...]
    place_descriptions: tuple[str, ...]
    place_names: tuple[str, ...]
    records: tuple[BindingRecord, ...]
    examples: tuple[BindingExample, ...]
    candidate_actions: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "world": self.world,
            "data_seed": self.data_seed,
            "order_seed": self.order_seed,
            "object_indices": list(self.object_indices),
            "object_mapping": dict(zip(self.object_descriptions, self.object_names, strict=True)),
            "place_mapping": dict(zip(self.place_descriptions, self.place_names, strict=True)),
            "record_lines": len(self.records),
            "record_bytes": sum(len(record.raw) for record in self.records),
            "record_styles": [record.style for record in self.records],
            "source_ids": [record.source_id for record in self.records],
            "examples": len(self.examples),
            "layers": _layer_counts(self.examples),
            "axes": axes_counts(self.examples),
            "candidate_actions": len(self.candidate_actions),
        }


def _layer_counts(examples: tuple[BindingExample, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.layer] = counts.get(example.layer, 0) + 1
    return counts


def axes_counts(examples: tuple[BindingExample, ...]) -> dict[str, int]:
    counts = {key: 0 for key in AXIS_TARGET}
    for example in examples:
        if example.description_seen and example.name_seen:
            counts["seen_description_seen_name"] += 1
        if not example.description_seen and example.name_seen:
            counts["new_description_seen_name"] += 1
        if example.description_seen and not example.name_seen:
            counts["seen_description_new_name"] += 1
        if not example.description_seen and not example.name_seen:
            counts["new_description_new_name"] += 1
        if example.new_description:
            counts["new_description"] += 1
        if example.unseen_output_name:
            counts["unseen_output_name"] += 1
        if example.both_new:
            counts["both_new"] += 1
        if example.unseen_combination:
            counts["unseen_combination"] += 1
        if example.layer == "seen_combination":
            counts["seen_combination"] += 1
    return counts


def style_index(split: str, world: int, record_index: int) -> int:
    """Public deterministic style assignment, declared before any timing."""
    return (world * 5 + record_index * 7 + STYLE_CODES[split] * 11) % 3


def style_line(description: str, name: str, style: int) -> str:
    if style == 0:
        return f"{description}是{name}。"
    if style == 1:
        return f"{name}又叫{description}。"
    if style == 2:
        return f"{description}对应{name}。"
    raise DataError(f"unknown record style {style}")


def _record_token(data_seed: int, split: str, world: int) -> str:
    return hashlib.sha256(f"{SCHEMA}|{data_seed}|{split}|{world}".encode()).hexdigest()[:12]


def _permutation(items: tuple[str, ...], seed_key: str) -> tuple[str, ...]:
    order = list(items)
    random.Random(seed_key).shuffle(order)
    return tuple(order)


def _object_indices_for(split: str) -> tuple[int, ...]:
    if split == "train":
        return TRAIN_OBJECT_INDICES
    if split == "dev":
        return DEV_OBJECT_INDICES
    if split == "confirm":
        return CONFIRM_OBJECT_INDICES
    raise DataError(f"unknown split {split!r}")


def _object_descriptions(split: str) -> tuple[str, ...]:
    return tuple(OBJECT_DESCRIPTIONS[index] for index in _object_indices_for(split))


def _object_names(split: str, world: int, data_seed: int) -> tuple[str, ...]:
    indices = _object_indices_for(split)
    names = tuple(OBJECT_NAMES[index] for index in indices)
    if split == "train":
        return _permutation(names, f"object|train|{world}|{data_seed}")
    # The last description is the new/held-out description; its assigned name
    # rotates through the four names so each name appears three times in 12
    # worlds.  The first three descriptions receive the remaining names.
    new_name = names[world % len(names)]
    remaining = [name for name in names if name != new_name]
    random.Random(f"object|{split}|{world}|{data_seed}").shuffle(remaining)
    return tuple(remaining) + (new_name,)


def _place_names(split: str, world: int, data_seed: int) -> tuple[str, ...]:
    return _permutation(PLACE_NAMES, f"place|{split}|{world}|{data_seed}")


def _candidate_actions(
    object_names: tuple[str, ...], place_names: tuple[str, ...]
) -> tuple[str, ...]:
    ordered_objects = sorted(object_names, key=OBJECT_NAMES.index)
    ordered_places = sorted(place_names, key=PLACE_NAMES.index)
    return tuple(
        f"把{object_name}放到{place_name}"
        for object_name in ordered_objects
        for place_name in ordered_places
    )


def _make_record(
    split: str,
    world: int,
    data_seed: int,
    position: int,
    kind: str,
    description: str,
    name: str,
    record_index: int,
) -> BindingRecord:
    style = style_index(split, world, record_index)
    token = _record_token(data_seed, split, world)
    return BindingRecord(
        source_id=f"cx{token}-{position:02d}",
        raw=style_line(description, name, style).encode(),
        style=style,
        kind=kind,
        description=description,
        name=name,
    )


def _make_example(
    index: int,
    world: int,
    split: str,
    layer: str,
    object_index: int,
    place_index: int,
    object_descriptions: tuple[str, ...],
    object_names: tuple[str, ...],
    place_descriptions: tuple[str, ...],
    place_names: tuple[str, ...],
    actions: tuple[str, ...],
    records: tuple[BindingRecord, ...],
) -> BindingExample:
    object_description = object_descriptions[object_index]
    object_name = object_names[object_index]
    place_description = place_descriptions[place_index]
    place_name = place_names[place_index]
    goal = f"让{object_description}到{place_description}。"
    action = f"把{object_name}放到{place_name}"
    try:
        target = actions.index(action)
    except ValueError as exc:  # pragma: no cover - guarded by construction
        raise DataError("true action missing from the candidate list") from exc
    description_seen = split == "train" or object_index < TRAIN_OBJECT_COUNT
    name_seen = object_name in OBJECT_NAMES[:TRAIN_OBJECT_COUNT]
    return BindingExample(
        index=index,
        world=world,
        split=split,
        layer=layer,
        goal=goal,
        object_description=object_description,
        object_name=object_name,
        place_description=place_description,
        place_name=place_name,
        action=action,
        expected=action.encode(),
        target=target,
        actions=actions,
        records=records,
        description_seen=description_seen,
        name_seen=name_seen,
        unseen_output_name=not name_seen,
        both_new=not description_seen and not name_seen,
        new_description=not description_seen,
        unseen_combination=layer == "unseen_combination",
    )


def build_world(split: str, world: int) -> BindingWorld:
    if split not in ("train", "dev"):
        raise DataError("confirm data is sealed; the public build is train/dev only")
    if not 0 <= world < SPLIT_WORLDS[split]:
        raise DataError(f"world {world} outside split {split!r}")
    data_seed, order_seed = SPLIT_SEEDS[split]
    object_indices = TRAIN_OBJECT_INDICES if split == "train" else DEV_OBJECT_INDICES
    object_descriptions = _object_descriptions(split)
    object_names = _object_names(split, world, data_seed)
    place_descriptions = PLACE_DESCRIPTIONS
    place_names = _place_names(split, world, data_seed)
    if len(set(object_names)) != len(object_names):
        raise DataError("object mapping repeats a name")
    if len(set(place_names)) != len(place_names):
        raise DataError("place mapping repeats a name")
    if set(object_names) & set(place_names):
        raise DataError("object and place domains overlap")

    lines: list[tuple[str, str, str, int]] = []
    for record_index, (description, name) in enumerate(
        zip(object_descriptions, object_names, strict=True)
    ):
        lines.append(("object", description, name, record_index))
    for offset, (description, name) in enumerate(zip(place_descriptions, place_names, strict=True)):
        lines.append(("place", description, name, offset + len(object_descriptions)))
    order = list(range(len(lines)))
    random.Random(f"order|{split}|{world}|{order_seed}").shuffle(order)
    records = tuple(
        _make_record(
            split,
            world,
            data_seed,
            position,
            lines[line][0],
            lines[line][1],
            lines[line][2],
            lines[line][3],
        )
        for position, line in enumerate(order)
    )

    actions = _candidate_actions(object_names, place_names)
    examples: list[BindingExample] = []
    if split == "train":
        for object_index, place_index in TRAIN_COMBINATIONS:
            examples.append(
                _make_example(
                    len(examples),
                    world,
                    split,
                    "seen_combination",
                    object_index,
                    place_index,
                    object_descriptions,
                    object_names,
                    place_descriptions,
                    place_names,
                    actions,
                    records,
                )
            )
    else:
        for object_index in range(len(object_descriptions)):
            for place_index in range(PLACE_COUNT):
                if object_index >= TRAIN_OBJECT_COUNT:
                    layer = "new_description"
                elif object_index == place_index:
                    layer = "unseen_combination"
                else:
                    layer = "seen_combination"
                examples.append(
                    _make_example(
                        len(examples),
                        world,
                        split,
                        layer,
                        object_index,
                        place_index,
                        object_descriptions,
                        object_names,
                        place_descriptions,
                        place_names,
                        actions,
                        records,
                    )
                )
    fixture = BindingWorld(
        split=split,
        world=world,
        data_seed=data_seed,
        order_seed=order_seed,
        object_indices=object_indices,
        object_descriptions=object_descriptions,
        object_names=object_names,
        place_descriptions=place_descriptions,
        place_names=place_names,
        records=records,
        examples=tuple(examples),
        candidate_actions=actions,
    )
    validate_world(fixture)
    return fixture


def _validate_pair(description: str, name: str, description_chars: int, name_chars: int) -> None:
    if len(description) != description_chars or len(name) != name_chars:
        raise DataError(f"pair {description!r}/{name!r} length")
    if description[0] == name[0]:
        raise DataError(f"{description!r} shares its first character with {name!r}")
    if name in description:
        raise DataError(f"{name!r} appears inside {description!r}")


def validate_world(world: BindingWorld) -> None:
    prefix = f"{world.split}[{world.world}]"
    if world.data_seed != SPLIT_SEEDS[world.split][0]:
        raise DataError(f"{prefix}: data seed drift")
    if world.order_seed != SPLIT_SEEDS[world.split][1]:
        raise DataError(f"{prefix}: order seed drift")
    expected_examples = SEEN_PER_WORLD if world.split == "train" else DEV_EXAMPLES_PER_WORLD
    if len(world.examples) != expected_examples:
        raise DataError(f"{prefix}: {len(world.examples)} examples")
    expected_layers = (
        {"seen_combination": SEEN_PER_WORLD}
        if world.split == "train"
        else {
            "seen_combination": SEEN_PER_WORLD,
            "unseen_combination": UNSEEN_COMBINATION_PER_WORLD,
            "new_description": NEW_DESCRIPTION_PER_WORLD,
        }
    )
    if _layer_counts(world.examples) != expected_layers:
        raise DataError(f"{prefix}: layer counts {_layer_counts(world.examples)}")

    lines_expected = TRAIN_RECORD_LINES if world.split == "train" else DEV_RECORD_LINES
    bytes_expected = TRAIN_RECORD_MAX_BYTES if world.split == "train" else DEV_RECORD_MAX_BYTES
    handled = [record.raw for record in world.records]
    total_bytes = sum(len(line) for line in handled)
    if len(handled) != lines_expected or total_bytes > bytes_expected:
        raise DataError(f"{prefix}: records {len(handled)}/{total_bytes}B")
    if total_bytes > MAX_SOURCE_BYTES:
        raise DataError(f"{prefix}: source budget exceeded")
    if len({record.source_id for record in world.records}) != len(world.records):
        raise DataError(f"{prefix}: duplicate source identity")
    if any(not record.source_id.startswith("cx") for record in world.records):
        raise DataError(f"{prefix}: source identity is not opaque")
    for record in world.records:
        if record.style not in (0, 1, 2):
            raise DataError(f"{prefix}: unknown style")
        if record.raw != style_line(record.description, record.name, record.style).encode():
            raise DataError(f"{prefix}: record text/style mismatch")
        if len(record.raw) not in (33, 36, 24, 27):
            raise DataError(f"{prefix}: record byte length {len(record.raw)}")

    if len(set(world.object_names)) != len(world.object_names):
        raise DataError(f"{prefix}: repeated object name")
    if len(set(world.place_names)) != len(world.place_names):
        raise DataError(f"{prefix}: repeated place name")
    if set(world.object_names) & set(world.place_names):
        raise DataError(f"{prefix}: object/place domains overlap")
    for description, name in zip(world.object_descriptions, world.object_names, strict=True):
        _validate_pair(description, name, 5, 4)
    for description, name in zip(world.place_descriptions, world.place_names, strict=True):
        _validate_pair(description, name, 4, 2)
    for description in world.object_descriptions:
        if len(description.encode()) != OBJECT_DESCRIPTION_BYTES:
            raise DataError(f"{prefix}: object description bytes")
    for description in world.place_descriptions:
        if len(description.encode()) != PLACE_DESCRIPTION_BYTES:
            raise DataError(f"{prefix}: place description bytes")
    for name in world.object_names:
        if len(name.encode()) != OBJECT_NAME_BYTES:
            raise DataError(f"{prefix}: object name bytes")
    for name in world.place_names:
        if len(name.encode()) != PLACE_NAME_BYTES:
            raise DataError(f"{prefix}: place name bytes")

    if len(world.candidate_actions) != len(world.object_names) * PLACE_COUNT:
        raise DataError(f"{prefix}: candidate action count")
    if len(set(world.candidate_actions)) != len(world.candidate_actions):
        raise DataError(f"{prefix}: repeated candidate action")
    for action in world.candidate_actions:
        if len(action.encode()) != ACTION_BYTES:
            raise DataError(f"{prefix}: candidate action bytes")

    for example in world.examples:
        if example.world != world.world or example.split != world.split:
            raise DataError(f"{prefix}: example identity drift")
        if len(example.goal.encode()) != GOAL_BYTES:
            raise DataError(f"{prefix}: goal bytes")
        if len(example.expected) != ACTION_BYTES or example.expected != example.action.encode():
            raise DataError(f"{prefix}: expected bytes")
        if example.actions != world.candidate_actions:
            raise DataError(f"{prefix}: action list drift")
        if example.actions[example.target] != example.action:
            raise DataError(f"{prefix}: target drift")
        if example.records != world.records:
            raise DataError(f"{prefix}: records drift")
        if example.description_seen != (
            world.split == "train"
            or example.object_description in world.object_descriptions[:TRAIN_OBJECT_COUNT]
        ):
            raise DataError(f"{prefix}: description_seen flag")
        all_names = set(OBJECT_NAMES) | set(PLACE_NAMES)
        for name in all_names:
            if name in example.goal:
                raise DataError(f"{prefix}: goal names an entity")
        for name in world.object_names + world.place_names:
            if name in example.goal:
                raise DataError(f"{prefix}: goal contains a world name")
    if world.split == "dev" and world.object_names != _object_names(
        "dev", world.world, world.data_seed
    ):
        # The new description's assigned name rotates through the four names
        # across 12 worlds; the first three descriptions get the remaining
        # names.  Recomputing the mapping checks the declared construction.
        raise DataError(f"{prefix}: held-out mapping drift")


def build_split(split: str, worlds: int | None = None) -> tuple[BindingWorld, ...]:
    if split == "confirm":
        raise DataError("confirm data is not generated; only the specification is sealed")
    count = SPLIT_WORLDS[split] if worlds is None else worlds
    if count < 1 or count > SPLIT_WORLDS[split]:
        raise DataError(f"split {split!r} has {SPLIT_WORLDS[split]} worlds")
    return tuple(build_world(split, world) for world in range(count))


def confirm_spec() -> dict[str, Any]:
    """The sealed confirm specification; no confirm example is generated."""
    return {
        "split": "confirm",
        "generated": False,
        "sealed": True,
        "data_seed": CONFIRM_SEED,
        "order_seed": CONFIRM_ORDER_SEED,
        "worlds": SPLIT_WORLDS["confirm"],
        "examples": SPLIT_WORLDS["confirm"] * DEV_EXAMPLES_PER_WORLD,
        "examples_per_world": DEV_EXAMPLES_PER_WORLD,
        "objects": CONFIRM_OBJECT_COUNT,
        "object_indices": list(CONFIRM_OBJECT_INDICES),
        "place_count": PLACE_COUNT,
        "record_lines": CONFIRM_RECORD_LINES,
        "record_bytes_max": CONFIRM_RECORD_MAX_BYTES,
        "layers": dict(LAYER_TARGET),
        "axes": dict(AXIS_TARGET),
        "styles": list(STYLES),
        "rotation": {
            "new_description_maps_to_each_of_4_names": 3,
            "new_name_on_a_seen_description": 9,
            "seen_name_on_new_description": 9,
            "both_new_worlds": 3,
        },
        "source_budget": MAX_SOURCE_BYTES,
        "note": (
            "generated only after the candidate and evaluator are frozen, by an "
            "independent data process; no confirm example or source id exists here"
        ),
    }


def rotation_counts(worlds: tuple[BindingWorld, ...]) -> dict[str, Any]:
    names = {name: 0 for name in OBJECT_NAMES[:4]}
    seen_name_on_new = 0
    new_name_on_seen = 0
    both = 0
    for world in worlds:
        if world.split != "dev":
            raise DataError("rotation is a dev property")
        new_name = world.object_names[-1]
        names[new_name] += 1
        if new_name in OBJECT_NAMES[:3]:
            seen_name_on_new += 1
        if OBJECT_NAMES[3] in world.object_names[:3]:
            new_name_on_seen += 1
        if new_name == OBJECT_NAMES[3]:
            both += 1
    return {
        "new_description_name_counts": names,
        "seen_name_on_new_description": seen_name_on_new,
        "new_name_on_seen_description": new_name_on_seen,
        "both_new_worlds": both,
    }


def fixture_payload(worlds: tuple[BindingWorld, ...], confirm: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": "v2",
        "data_seed": DATA_SEED,
        "order_seed": ORDER_SEED,
        "confirm_seed": CONFIRM_SEED,
        "confirm_order_seed": CONFIRM_ORDER_SEED,
        "split_worlds": dict(SPLIT_WORLDS),
        "split_seeds": {split: list(seed) for split, seed in SPLIT_SEEDS.items()},
        "vocabulary": {
            "object_names": list(OBJECT_NAMES),
            "object_descriptions": list(OBJECT_DESCRIPTIONS),
            "place_names": list(PLACE_NAMES),
            "place_descriptions": list(PLACE_DESCRIPTIONS),
            "train_object_indices": list(TRAIN_OBJECT_INDICES),
            "dev_object_indices": list(DEV_OBJECT_INDICES),
            "confirm_object_indices": list(CONFIRM_OBJECT_INDICES),
            "model_interface": "256 raw bytes plus EOS; names/descriptions are environment/scorer materials",
        },
        "styles": list(STYLES),
        "axes": dict(AXIS_TARGET),
        "layers": dict(LAYER_TARGET),
        "worlds": [world.summary() for world in worlds],
        "world_digests": [
            {"split": world.split, "world": world.world, "digest": split_digest(world)}
            for world in worlds
        ],
        "examples": [example.payload() for world in worlds for example in world.examples],
        "digest": fixture_digest(worlds),
        "confirm_spec": confirm,
    }


def fixture_digest(worlds: tuple[BindingWorld, ...]) -> str:
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
            digest.update(str(record.style).encode())
        for example in world.examples:
            digest.update(example.goal.encode())
            digest.update(example.expected)
            digest.update(example.layer.encode())
            digest.update(str(example.target).encode())
    return digest.hexdigest()


def split_digest(world: BindingWorld) -> str:
    digest = hashlib.sha256()
    digest.update(SCHEMA.encode())
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
        digest.update(str(record.style).encode())
    for example in world.examples:
        digest.update(example.goal.encode())
        digest.update(example.expected)
        digest.update(example.layer.encode())
        digest.update(str(example.target).encode())
    return digest.hexdigest()
