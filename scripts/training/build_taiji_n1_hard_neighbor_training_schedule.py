"""PLAN-N1-100 train-only balanced hard-neighbor schedule builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
from collections import Counter
from pathlib import Path

from scripts.training import run_taiji_n1_local_relation as core

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "plans/manifests/n1_native_evidence_curriculum_v3_20261015.json"
RENAMING = core.CURRICULUM
PAIRS = core.PAIR_MANIFEST
OUT = ROOT / "plans/manifests/n1_hard_neighbor_training_schedule_v1_20261017.json"
MANIFEST_SHA256 = "ac56fb0aba2ef602779f58749e6f8c35324c40c433fd16ce55b89689235b0604"
RENAMING_SHA256 = core.CURRICULUM_SHA
PAIRS_SHA256 = core.PAIR_MANIFEST_SHA
SEED = 10002
KINDS = ("lookup", "revision", "transition", "unknown")
OBSERVATION_FACTORS = (2, 4)
DISTANCES = (1, 2)
LETTERS = string.ascii_lowercase
_EPISODE_HEAD = re.compile(
    rb'^\{\s*"id"\s*:\s*"[^"\\]*(?:\\.[^"\\]*)*"\s*,\s*' rb'"split"\s*:\s*"([^"]+)"\s*,'
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object_end(data: bytes, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(data)):
        value = data[index]
        if in_string:
            if escaped:
                escaped = False
            elif value == ord("\\"):
                escaped = True
            elif value == ord('"'):
                in_string = False
            continue
        if value == ord('"'):
            in_string = True
        elif value == ord("{"):
            depth += 1
        elif value == ord("}"):
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError("unterminated episode object")


def _train_objects_from_bytes(raw: bytes) -> list[dict]:
    """Parse train objects only; non-train object payloads are skipped without JSON decoding."""
    marker = raw.find(b'"episodes"')
    if marker < 0:
        raise ValueError("frozen v3 manifest episodes array is missing")
    array_start = raw.find(b"[", marker)
    if array_start < 0:
        raise ValueError("frozen v3 manifest episodes array is malformed")

    selected: list[dict] = []
    cursor = array_start + 1
    while cursor < len(raw):
        while cursor < len(raw) and (raw[cursor] in b" \r\n\t,"):
            cursor += 1
        if cursor >= len(raw) or raw[cursor] == ord("]"):
            break
        if raw[cursor] != ord("{"):
            raise ValueError("frozen v3 episode must be an object")
        end = _object_end(raw, cursor)
        head = _EPISODE_HEAD.match(raw[cursor:end])
        if head is None:
            raise ValueError("frozen v3 episode identity/split prefix changed")
        split = head.group(1)
        if split not in (b"train", b"dev", b"holdout"):
            raise ValueError("frozen v3 split label is outside the signed source schema")
        if split == b"train":
            item = json.loads(raw[cursor:end].decode("utf-8"))
            if item.get("split") != "train":
                raise ValueError("train-only episode split changed while decoding")
            selected.append(item)
        # Dev/holdout suffixes are deliberately not decoded, inspected, or summarized.
        cursor = end
    return selected


def train_items_from_frozen_manifest(path: Path = MANIFEST) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != MANIFEST_SHA256:
        raise ValueError("frozen v3 manifest bytes changed")
    selected = _train_objects_from_bytes(raw)
    validate_train_items(selected)
    return selected, digest


def validate_train_items(items: list[dict]) -> None:
    if len(items) != 192 or any(item.get("split") != "train" for item in items):
        raise ValueError("the complete 192-item train split is required")
    if len({item.get("id") for item in items}) != 192:
        raise ValueError("train episode IDs must be unique")
    kinds = Counter(item.get("kind") for item in items)
    if kinds != Counter({kind: 48 for kind in KINDS}):
        raise ValueError("train items must contain 48 examples per task kind")
    for item in items:
        entities = item.get("entities")
        observations = item.get("observations")
        if (
            not isinstance(entities, list)
            or len(entities) != 3
            or len(set(entities)) != 3
            or any(len(name) != 10 or not name.isascii() or not name.islower() for name in entities)
            or not isinstance(observations, list)
            or len(observations) != 2
        ):
            raise ValueError("train-only source entity/observation shape changed")
        subject = item["question"].removeprefix("material of ").removesuffix("?")
        if item["kind"] in ("lookup", "revision", "unknown") and (
            subject not in entities or item["answer"] in item["question"]
        ):
            raise ValueError("train-only material question target is malformed")
        if item["kind"] == "lookup":
            exact = [
                line
                for line in observations
                if line.startswith(f"{subject} uses ") and line.endswith(".")
            ]
            if len(exact) != 1 or exact[0].removesuffix(".").split()[-1] != item["answer"]:
                raise ValueError("train-only lookup target row is not unique")
        elif item["kind"] == "revision":
            if (
                len(observations) != 2
                or not observations[0].startswith(f"{subject} uses ")
                or not observations[1].startswith(f"{subject} now uses ")
                or observations[1].removesuffix(".").split()[-1] != item["answer"]
            ):
                raise ValueError("train-only revision ordering or target changed")
        elif item["kind"] == "unknown":
            if item["answer"] != "unknown" or any(
                line.startswith(f"{subject} uses ") for line in observations
            ):
                raise ValueError("train-only unknown target must be absent")
        elif item["kind"] == "transition":
            if len(observations) != 2 or not item["question"].startswith("direction of "):
                raise ValueError("train-only transition two-fact input changed")


def _values(items: list[dict]) -> list[str]:
    return sorted(
        {
            line.removesuffix(".").split()[-1]
            for item in items
            for line in item["observations"]
            if " uses " in line
        }
    )


def _target(question: str) -> str:
    prefix = "material of "
    if not question.startswith(prefix) or not question.endswith("?"):
        raise ValueError("hard-view material question grammar changed")
    return question[len(prefix) : -1]


def _confusable(target: str, distance: int, rng: random.Random, used: set[str]) -> str:
    for _ in range(500):
        chars = list(target)
        for index in rng.sample(range(len(chars)), distance):
            chars[index] = rng.choice(LETTERS.replace(chars[index], ""))
        candidate = "".join(chars)
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError("could not generate a unique exact-distance distractor")


def _value_rows(
    count: int, *, pool: list[str], answer: str, existing: set[str], rng: random.Random
) -> list[str]:
    choices = [value for value in pool if value != answer and value not in existing]
    if len(choices) < count:
        raise ValueError("train-only value pool cannot provide distinct distractor materials")
    return rng.sample(choices, count)


def _hard_item(
    source: dict,
    names: list[str],
    *,
    observation_count: int,
    distance: int,
    slot: int,
    target_start: int | None,
    replicate: int,
    value_pool: list[str],
    rng: random.Random,
    used_names: set[str],
) -> tuple[dict, dict | None, list[int]]:
    item = core.renaming.rename_item(source, names)
    target = _target(item["question"])
    required = 1 if item["kind"] == "lookup" else 0 if item["kind"] == "unknown" else 2
    if item["kind"] == "revision":
        observation_count = 4
        required = 2
    if observation_count not in (2, 4) or observation_count < required:
        raise ValueError("unsupported hard-view observation count")

    distractor_count = observation_count - required
    confusable_names = [
        _confusable(target, distance, rng, used_names) for _ in range(distractor_count)
    ]
    present_values = {
        line.removesuffix(".").split()[-1] for line in item["observations"] if " uses " in line
    }
    extra_values = _value_rows(
        max(
            0,
            distractor_count
            - sum(
                " uses " in line
                for line in item["observations"]
                if not line.startswith(target + " ")
            ),
        ),
        pool=value_pool,
        answer=item["answer"],
        existing=present_values,
        rng=rng,
    )
    value_index = 0
    distractors = []
    existing_distractors = [
        line for line in item["observations"] if not line.startswith(target + " ")
    ]
    for index, name in enumerate(confusable_names):
        if index < len(existing_distractors):
            value = existing_distractors[index].removesuffix(".").split()[-1]
        else:
            value = extra_values[value_index]
            value_index += 1
        distractors.append(f"{name} uses {value}.")

    if item["kind"] == "lookup":
        target_row = next(
            line for line in item["observations"] if line.startswith(target + " uses ")
        )
        distractor_rows = distractors
        target_position = replicate % observation_count
        rows = list(distractor_rows)
        rows.insert(target_position, target_row)
        target_positions = [target_position]
    elif item["kind"] == "unknown":
        rows = list(distractors)
        rng.shuffle(rows)
        target_positions = []
    elif item["kind"] == "revision":
        target_rows = [line for line in item["observations"] if line.startswith(target + " ")]
        if (
            len(target_rows) != 2
            or not target_rows[0].startswith(target + " uses ")
            or not target_rows[1].startswith(target + " now uses ")
        ):
            raise ValueError("hard revision must preserve stale/current order")
        if target_start not in (0, 1, 2):
            raise ValueError("revision target-pair placement must be one of three valid starts")
        rows = list(distractors)
        rows[target_start:target_start] = target_rows
        target_positions = [target_start, target_start + 1]
    else:
        raise ValueError("transition rows must use the frozen two-fact rename path")

    if len(rows) != observation_count:
        raise ValueError("hard-view observation count construction failed")
    item.update(
        id=f"hard-{source['id']}-{slot:02d}",
        parent_id=source["id"],
        original_kind=source["kind"],
        observations=rows,
        entities=list(dict.fromkeys([*item["entities"], *confusable_names])),
    )
    second = core.renaming.second_question(item)
    if item["kind"] in ("lookup", "unknown"):
        if (
            second is None
            or second["parent_id"] != item["id"]
            or second["observations"] != item["observations"]
        ):
            raise ValueError("hard-view paired question must derive from the same observations")
        if not any(line.startswith(_target(second["question"]) + " uses ") for line in rows):
            raise ValueError("paired distractor question must target a present observation")
    elif second is not None:
        raise ValueError("revision must not acquire a second-question target")
    return item, second, target_positions


def _conditions(kind: str, rng: random.Random) -> list[dict]:
    if kind in ("lookup", "unknown"):
        entries = [
            dict(observation_count=observations, distance=distance)
            for observations in OBSERVATION_FACTORS
            for distance in DISTANCES
            for _ in range(4)
        ]
        cell_replicates = Counter()
        for entry in entries:
            cell = (entry["observation_count"], entry["distance"])
            entry["replicate"] = cell_replicates[cell]
            cell_replicates[cell] += 1
    elif kind == "revision":
        entries = []
        for distance in DISTANCES:
            for replicate in range(8):
                entries.append(
                    dict(
                        observation_count=4,
                        distance=distance,
                        replicate=replicate,
                        target_start=(replicate + distance - 1) % 3,
                    )
                )
    elif kind == "transition":
        entries = [
            dict(observation_count=2, distance=None, condition="frozen-rename") for _ in range(16)
        ]
    else:
        raise ValueError("unknown train task kind")
    rng.shuffle(entries)
    return entries


def load_training_sources(items: list[dict] | None = None) -> tuple[list[dict], dict, list[dict]]:
    if items is None:
        items, _ = train_items_from_frozen_manifest()
    validate_train_items(items)
    if sha256(RENAMING) != RENAMING_SHA256:
        raise ValueError("frozen train-only renaming stream changed")
    curriculum = json.loads(RENAMING.read_text(encoding="utf-8"))
    if curriculum.get("source_manifest_sha256") != MANIFEST_SHA256:
        raise ValueError("renaming stream source manifest identity changed")
    stream = curriculum.get("stream")
    if not isinstance(stream, list) or len(stream) != 3072:
        raise ValueError("exact 3072-entry frozen train renaming stream required")
    all_train_names = {name for item in items for name in item["entities"]}
    seen_stream_names: set[str] = set()
    for index, entry in enumerate(stream):
        expected_parent = items[index % 192]["id"]
        names = entry.get("names")
        if (
            entry.get("parent_id") != expected_parent
            or not isinstance(names, list)
            or len(names) != 3
            or len(set(names)) != 3
            or any(len(name) != 10 or not name.isascii() or not name.islower() for name in names)
            or set(names) & (all_train_names | seen_stream_names)
        ):
            raise ValueError("frozen stream must provide unique train-derived replacement names")
        seen_stream_names.update(names)
    probes = curriculum.get("probes")
    if not isinstance(probes, list) or len(probes) != 192:
        raise ValueError("exact frozen train-derived rename probes required")
    for item, probe in zip(items, probes, strict=True):
        names = probe.get("names")
        if (
            probe.get("parent_id") != item["id"]
            or not isinstance(names, list)
            or len(names) != 3
            or len(set(names)) != 3
            or any(len(name) != 10 or not name.isascii() or not name.islower() for name in names)
            or set(names) & (all_train_names | seen_stream_names)
        ):
            raise ValueError("frozen rename probes must stay disjoint from train and stream names")
        seen_stream_names.update(names)
    pairs = core.checked_pairs(items)
    return items, curriculum, pairs


def build_schedule(items: list[dict] | None = None, curriculum: dict | None = None) -> dict:
    if items is None:
        items, _ = train_items_from_frozen_manifest()
    validate_train_items(items)
    if curriculum is None:
        _, curriculum, _ = load_training_sources(items)
    stream = curriculum["stream"]
    rng = random.Random(SEED)
    used_names = {name for item in items for name in item["entities"]}
    used_names.update(name for entry in stream for name in entry["names"])
    used_names.update(name for probe in curriculum["probes"] for name in probe["names"])
    value_pool = _values(items)
    schedule = []
    parent_slots: Counter[str] = Counter()
    target_placements: dict[str, Counter[int]] = {}
    condition_counts: Counter[tuple[str, int, int | None]] = Counter()

    by_parent_conditions = {item["id"]: _conditions(item["kind"], rng) for item in items}
    for stream_index, stream_entry in enumerate(stream):
        parent_index = stream_index % 192
        slot = stream_index // 192
        source = items[parent_index]
        if stream_entry["parent_id"] != source["id"]:
            raise ValueError("hard schedule parent order differs from frozen stream")
        condition = by_parent_conditions[source["id"]][slot]
        if source["kind"] == "transition":
            item = core.renaming.rename_item(source, stream_entry["names"])
            item.update(
                id=f"hard-{source['id']}-{slot:02d}",
                parent_id=source["id"],
                original_kind=source["kind"],
            )
            second = None
            target_positions = []
        else:
            item, second, target_positions = _hard_item(
                source,
                stream_entry["names"],
                observation_count=condition["observation_count"],
                distance=condition["distance"],
                slot=slot,
                target_start=condition.get("target_start"),
                replicate=condition.get("replicate", slot),
                value_pool=value_pool,
                rng=rng,
                used_names=used_names,
            )
        if source["kind"] == "revision":
            starts = target_placements.setdefault(source["id"], Counter())
            starts[target_positions[0]] += 1
        row = dict(
            schedule_index=stream_index,
            parent_id=source["id"],
            parent_kind=source["kind"],
            slot=slot,
            condition=(
                condition.get("condition")
                or f"obs{condition['observation_count']}-d{condition['distance']}"
            ),
            observation_count=len(item["observations"]),
            hamming_distance=condition["distance"],
            target_positions=target_positions,
            item=item,
            second_question=second,
        )
        schedule.append(row)
        parent_slots[source["id"]] += 1
        condition_counts[(source["kind"], row["observation_count"], row["hamming_distance"])] += 1

    if len(schedule) != 3072 or set(parent_slots.values()) != {16}:
        raise ValueError("hard schedule must contain 16 slots for every train parent")
    for parent_id, positions in target_placements.items():
        if set(positions) != {0, 1, 2} or max(positions.values()) - min(positions.values()) > 1:
            raise ValueError(f"revision target-pair positions are not balanced for {parent_id}")
    expected = Counter(
        {
            ("lookup", 2, 1): 48 * 4,
            ("lookup", 2, 2): 48 * 4,
            ("lookup", 4, 1): 48 * 4,
            ("lookup", 4, 2): 48 * 4,
            ("unknown", 2, 1): 48 * 4,
            ("unknown", 2, 2): 48 * 4,
            ("unknown", 4, 1): 48 * 4,
            ("unknown", 4, 2): 48 * 4,
            ("revision", 4, 1): 48 * 8,
            ("revision", 4, 2): 48 * 8,
            ("transition", 2, None): 48 * 16,
        }
    )
    if condition_counts != expected:
        raise ValueError("hard schedule factorial cell denominators differ")

    return {
        "schema": "n1-hard-neighbor-training-schedule-v1",
        "seed": SEED,
        "source_manifest_sha256": MANIFEST_SHA256,
        "source_renaming_sha256": RENAMING_SHA256,
        "source_pair_manifest_sha256": PAIRS_SHA256,
        "builder_sha256": sha256(Path(__file__).resolve()),
        "items": schedule,
        "summary": {
            "schedule_items": len(schedule),
            "parents": len(parent_slots),
            "slots_per_parent": 16,
            "kind_counts": dict(sorted(Counter(row["parent_kind"] for row in schedule).items())),
            "condition_counts": {
                f"{kind}/obs{observations}/d{distance}": count
                for (kind, observations, distance), count in sorted(
                    condition_counts.items(),
                    key=lambda entry: (entry[0][0], entry[0][1], str(entry[0][2])),
                )
            },
            "revision_target_pair_start_counts": {
                str(position): sum(counts[position] for counts in target_placements.values())
                for position in range(3)
            },
            "parameter_updates": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
        },
    }


def write_schedule(path: Path = OUT) -> dict:
    if path.exists():
        raise FileExistsError(f"hard-neighbor schedule already exists: {path}")
    items, curriculum, _ = load_training_sources()
    schedule = build_schedule(items, curriculum)
    encoded = json.dumps(schedule, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        result = write_schedule()
        print(json.dumps({"path": str(OUT), "sha256": sha256(OUT), "summary": result["summary"]}))
    else:
        items, curriculum, _ = load_training_sources()
        result = build_schedule(items, curriculum)
        print(json.dumps({"accepted": True, "summary": result["summary"]}))


if __name__ == "__main__":
    main()
