"""Create the one-shot PLAN-N1-102 v9 sacrifice/confirmation face."""

from __future__ import annotations

import hashlib
import json
import random
import string
from collections import Counter
from pathlib import Path

from scripts.training import build_taiji_n1_hard_neighbor_training_schedule as train_data
from scripts.training import run_taiji_n1_hard_neighbor_training as training
from scripts.training import seal_ast  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/reference/PLAN-N1-102_v12_observation_range_probe_20261017.md"
V5 = ROOT / "plans/manifests/n1_fresh_confirmation_v5_20261016.json"
V6 = ROOT / "plans/manifests/n1_fresh_confirmation_v6_20261017.json"
V7 = ROOT / "plans/manifests/n1_hard_neighbor_v7_20261017.json"
V7_RANGE = ROOT / "reports/taiji_n1_hard_neighbor_v7_range_check_20261017.json"
V7_RANGE_ATTEMPT = V7_RANGE.with_name(V7_RANGE.stem + "_attempt.json")
V7_CONFIRM = ROOT / "reports/taiji_n1_hard_neighbor_v7_confirmation_20261017.json"
V7_CONFIRM_ATTEMPT = V7_CONFIRM.with_name(V7_CONFIRM.stem + "_attempt.json")
V8 = ROOT / "plans/manifests/n1_hard_neighbor_v8_20261017.json"
V8_RANGE = ROOT / "reports/taiji_n1_hard_neighbor_v8_range_check_20261017.json"
V8_RANGE_ATTEMPT = V8_RANGE.with_name(V8_RANGE.stem + "_attempt.json")
V8_CONFIRM = ROOT / "reports/taiji_n1_hard_neighbor_v8_confirmation_20261017.json"
V8_CONFIRM_ATTEMPT = V8_CONFIRM.with_name(V8_CONFIRM.stem + "_attempt.json")
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v9_20261017.json"
CHECKS = ROOT / "reports/taiji_n1_hard_neighbor_v9_face_checks_20261017.json"
ATTEMPT = CHECKS.with_name(CHECKS.stem + "_attempt.json")
SEED = 109
ITEMS_PER_KIND = 10
ITEMS = 40
OBSERVATIONS = 12
ENTITY_LENGTH = 10
ANSWER_LENGTHS = (6, 7)
DISTANCE = 1
KINDS = ("lookup", "revision", "transition", "unknown")
DIRECTIONS = ("north", "east", "south", "west")
SUBJECT_ROWS = {"lookup": 1, "revision": 2, "transition": 2, "unknown": 0}
ABSTENTION = "unknown"
LETTERS = string.ascii_lowercase
EXPECTED_SHA256 = {
    "v5": "22ba37976bc6eaaaf2425b84714a6efa6e95e7e5aa05f32eb3a7c0c10d9ffc2d",
    "v6": "7dddf00f3f2e24381728855b218e67bc8160fd6badeb1a7a933937d999766d49",
    "v7": "c5a7ef0bc18bf9376e5038348204e838bbc11d88ad6784605c858b12f6d992dc",
    "v7_range": "c0bf9e588b3785240c61a0a90907f4849de64b1027546a749d00a8e2768c30eb",
    "v7_range_attempt": "f45062b2abb8cb43ee026b379f64b7caca610c03923b927a933d7cc1e5f4529c",
    "training_report": "dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909",
    "v8": "85d417eb387e1c985c9b5665ade0536dd74b13b52126d7601ed5af225dbdc3fc",
    "v8_range": "0e983c31338c00536dfdbe1250c1ea57e5489536d7015391288b80517adbd406",
    "v8_range_attempt": "c63ca8997a6263e32635d1b8a944b0047521c1a1026c1601ba71e25d26e61e81",
    "plan": "3674c526a1924d4a96e83fb60e33cf10dce3f17fb6668d25e5cd3864c6e0720b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _JsonStream:
    """Small streaming JSON scanner that can skip an object tail without decoding it."""

    def __init__(self, path: Path):
        # 流式扫描器: 句柄生命周期跨越多个方法（mark/seek/close），无法收敛进
        # 单个 with 块——SIM115 在此为结构性误报
        self._stream = Path(path).open("r", encoding="utf-8")  # noqa: SIM115
        self._lookahead: str | None = None
        self._lookahead_cookie = None

    def mark(self):
        return self._lookahead_cookie if self._lookahead is not None else self._stream.tell()

    def seek(self, cookie) -> None:
        self._stream.seek(cookie)
        self._lookahead = None
        self._lookahead_cookie = None

    def close(self) -> None:
        self._stream.close()

    def peek(self) -> str:
        if self._lookahead is None:
            self._lookahead_cookie = self._stream.tell()
            self._lookahead = self._stream.read(1)
        return self._lookahead

    def get(self) -> str:
        if self._lookahead is None:
            return self._stream.read(1)
        value = self._lookahead
        self._lookahead = None
        self._lookahead_cookie = None
        return value

    def skip_ws(self, capture: list[str] | None = None) -> None:
        while self.peek() and self.peek() in " \t\r\n":
            char = self.get()
            if capture is not None:
                capture.append(char)

    def expect(self, expected: str, capture: list[str] | None = None) -> None:
        actual = self.get()
        if actual != expected:
            raise ValueError(f"expected {expected!r} while scanning v7 manifest, got {actual!r}")
        if capture is not None:
            capture.append(actual)

    def read_string(self) -> tuple[str, str]:
        raw = ['"']
        self.expect('"')
        escaped = False
        while True:
            char = self.get()
            if not char:
                raise ValueError("unterminated string in v7 manifest prefix")
            raw.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                encoded = "".join(raw)
                return json.loads(encoded), encoded

    def skip_value(self) -> None:
        """Skip one syntactically bounded JSON value without decoding its contents."""
        self.skip_ws()
        first = self.get()
        if not first:
            raise ValueError("unexpected end of v7 manifest")
        if first == '"':
            escaped = False
            while True:
                char = self.get()
                if not char:
                    raise ValueError("unterminated string while skipping v7 manifest")
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    return

        if first in "[{":
            curly = 1 if first == "{" else 0
            square = 1 if first == "[" else 0
            in_string = False
            escaped = False
            while curly or square:
                char = self.get()
                if not char:
                    raise ValueError("unterminated value while skipping v7 manifest")
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char == "{":
                    curly += 1
                elif char == "}":
                    curly -= 1
                elif char == "[":
                    square += 1
                elif char == "]":
                    square -= 1
                if curly < 0 or square < 0:
                    raise ValueError("unbalanced value while skipping v7 manifest")
            return

        while self.peek() and self.peek() not in ",]}":
            self.get()

    def read_object_at(self, cookie):
        """Decode one full object at a marked location, then restore the streaming cursor."""
        cursor = self.mark()
        self.seek(cookie)
        capture = []
        curly = 1
        square = 0
        in_string = False
        escaped = False
        self.expect("{", capture)
        while curly:
            char = self.get()
            if not char:
                raise ValueError("unterminated v7 episode object")
            capture.append(char)
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                curly += 1
            elif char == "}":
                curly -= 1
            elif char == "[":
                square += 1
            elif char == "]":
                square -= 1
                if square < 0:
                    raise ValueError("unbalanced array in v7 episode object")
        if square:
            raise ValueError("unbalanced array in v7 episode object")
        encoded = "".join(capture)
        self.seek(cursor)
        return json.loads(encoded)


def _read_episode_prefix(stream: _JsonStream) -> tuple[dict, dict | None]:
    start = stream.mark()
    stream.expect("{")
    fields = {}
    while True:
        stream.skip_ws()
        if stream.peek() == "}":
            stream.get()
            break
        key, raw_key = stream.read_string()
        stream.skip_ws()
        stream.expect(":")
        stream.skip_ws()
        if key in {"id", "split", "subset"}:
            if key in fields:
                raise ValueError(f"duplicate {key} in v7 episode")
            value, _ = stream.read_string()
            fields[key] = value
        else:
            stream.skip_value()
        stream.skip_ws()
        if stream.peek() == "}":
            stream.get()
            break
        stream.expect(",")

    if set(fields) != {"id", "split", "subset"}:
        raise ValueError("v7 episode must contain id, split and subset string fields")
    if fields["subset"] == "sacrifice":
        episode = stream.read_object_at(start)
        if any(episode.get(key) != value for key, value in fields.items()):
            raise ValueError("decoded sacrifice item differs from its inspected prefix")
        return fields, episode

    if fields["subset"] != "confirm":
        raise ValueError("v7 episode has an unexpected subset")
    return fields, None


def _stream_source_episode_prefixes(
    path: Path, *, expected_count: int | None = None
) -> list[tuple[dict, dict | None]]:
    """Decode sacrifice episodes only; scan confirm prefixes and skip their remaining fields."""
    stream = _JsonStream(path)
    episodes: list[tuple[dict, dict | None]] = []
    try:
        stream.skip_ws()
        stream.expect("{")
        found_episodes = False
        while True:
            stream.skip_ws()
            if stream.peek() == "}":
                stream.get()
                break
            key, _ = stream.read_string()
            stream.skip_ws()
            stream.expect(":")
            stream.skip_ws()
            if key == "episodes":
                if found_episodes:
                    raise ValueError("duplicate v7 episodes array")
                found_episodes = True
                stream.expect("[")
                stream.skip_ws()
                if stream.peek() != "]":
                    while True:
                        episodes.append(_read_episode_prefix(stream))
                        stream.skip_ws()
                        if stream.peek() == "]":
                            break
                        stream.expect(",")
                        stream.skip_ws()
                stream.expect("]")
            else:
                stream.skip_value()
            stream.skip_ws()
            if stream.peek() == "}":
                stream.get()
                break
            stream.expect(",")
        stream.skip_ws()
        if stream.peek():
            raise ValueError("trailing bytes after v7 manifest root object")
    finally:
        stream.close()
    if not found_episodes or (expected_count is not None and len(episodes) != expected_count):
        raise ValueError("v7 manifest episode count differs from the expected contract")
    return episodes


def exclusions_from_faces(v5: dict, v6: dict) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    words: set[str] = {ABSTENTION}
    for face in (v5, v6):
        for episode in face["episodes"]:
            names.update(episode["entities"])
            words.add(episode["answer"])
            for line in episode["observations"]:
                if " uses " in line:
                    words.add(line.removesuffix(".").split()[-1])
    return names, words


def exclusions_from_consumed_sacrifice(
    *,
    label: str,
    path: Path,
    range_report_path: Path,
    range_attempt_path: Path,
    confirm_report_path: Path,
    confirm_attempt_path: Path,
) -> tuple[set[str], set[str], dict]:
    if sha256(path) != EXPECTED_SHA256[label]:
        raise ValueError(f"{label} manifest bytes differ from the sealed sacrifice-only source")
    parsed = _stream_source_episode_prefixes(path, expected_count=40)
    prefixes = [prefix for prefix, _ in parsed]
    sacrifices = [episode for _, episode in parsed if episode is not None]
    confirm_prefixes = [prefix for prefix in prefixes if prefix["subset"] == "confirm"]
    if (
        len(sacrifices) != 20
        or len(confirm_prefixes) != 20
        or any(prefix["split"] != "holdout" for prefix in prefixes)
        or any(prefix["subset"] not in {"sacrifice", "confirm"} for prefix in prefixes)
        or len({prefix["id"] for prefix in prefixes}) != 40
    ):
        raise ValueError(f"{label} sacrifice/confirm prefix partition differs from the sealed face")

    range_report = read_json(range_report_path)
    attempt = read_json(range_attempt_path)
    if (
        sha256(range_report_path) != EXPECTED_SHA256[f"{label}_range"]
        or sha256(range_attempt_path) != EXPECTED_SHA256[f"{label}_range_attempt"]
        or range_report.get("complete") is not True
        or range_report.get("status") != "range_not_met"
        or range_report.get("face_sha256") != EXPECTED_SHA256[label]
        or range_report.get("queried_subsets") != ["sacrifice"]
        or range_report.get("confirm_subset_queried") is not False
        or range_report.get(f"{label}_sacrifice_model_queries") != 180
        or range_report.get(f"{label}_confirm_model_queries") != 0
        or range_report.get("parameter_updates") != 0
        or attempt.get("confirm_subset_queried") is not False
        or attempt.get(f"{label}_confirm_model_queries") != 0
        or attempt.get("parameter_updates") != 0
        or confirm_report_path.exists()
        or confirm_attempt_path.exists()
    ):
        raise ValueError(f"{label} evidence does not prove sacrifice-only consumption")

    sacrifice_ids = {episode["id"] for episode in sacrifices}
    queried_ids = {
        row["id"]
        for result in range_report.get("endpoint_results", [])
        for row in result.get("rows", [])
    }
    if queried_ids != sacrifice_ids or len(queried_ids) != 20:
        raise ValueError(f"{label} range receipt does not match the sacrifice item IDs")

    names: set[str] = set()
    words: set[str] = {ABSTENTION}
    for episode in sacrifices:
        names.update(episode["entities"])
        words.add(episode["answer"])
        for line in episode["observations"]:
            if " uses " in line:
                words.add(line.removesuffix(".").split()[-1])
    provenance = {
        "face_manifest_sha256": sha256(path),
        "range_report_sha256": sha256(range_report_path),
        "range_attempt_sha256": sha256(range_attempt_path),
        "sacrifice_item_count": len(sacrifices),
        "confirm_prefix_count_skipped": len(confirm_prefixes),
        "confirm_payload_fields_decoded": False,
        "confirm_model_queries": 0,
    }
    return names, words, provenance


def _novel(length: int, used: set[str], rng: random.Random) -> str:
    for _ in range(10000):
        value = "".join(rng.choices(LETTERS, k=length))
        if value not in used:
            used.add(value)
            return value
    raise ValueError("v9 generator exhausted unique names or answer values")


def _confusable(subject: str, used: set[str], rng: random.Random) -> str:
    for _ in range(500):
        chars = list(subject)
        position = rng.randrange(ENTITY_LENGTH)
        chars[position] = rng.choice(LETTERS.replace(chars[position], ""))
        candidate = "".join(chars)
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError("v9 generator could not create an exact d=1 entity")


def build_face(
    *, train_items: list[dict], excluded_names: set[str], excluded_words: set[str], seed: int = SEED
) -> dict:
    rng = random.Random(seed)
    used_names = set(excluded_names)
    used_words = set(excluded_words) | {ABSTENTION}
    seen = Counter()
    episodes = []
    for index in range(ITEMS):
        kind = KINDS[index % len(KINDS)]
        ordinal = seen[kind]
        seen[kind] += 1
        subject = _novel(ENTITY_LENGTH, used_names, rng)
        distractor_count = OBSERVATIONS - SUBJECT_ROWS[kind]
        distractors = [_confusable(subject, used_names, rng) for _ in range(distractor_count)]
        materials = [_novel(6, used_words, rng) for _ in range(distractor_count)]
        answer_length = ANSWER_LENGTHS[ordinal % len(ANSWER_LENGTHS)]
        target_rows = []
        if kind == "lookup":
            answer = _novel(answer_length, used_words, rng)
            question = f"material of {subject}?"
            target_rows = [f"{subject} uses {answer}."]
        elif kind == "revision":
            stale = _novel(6, used_words, rng)
            answer = _novel(answer_length, used_words, rng)
            question = f"material of {subject}?"
            target_rows = [f"{subject} uses {stale}.", f"{subject} now uses {answer}."]
        elif kind == "transition":
            answer = DIRECTIONS[ordinal % len(DIRECTIONS)]
            turn = rng.choice((1, 3))
            start = (DIRECTIONS.index(answer) - turn) % len(DIRECTIONS)
            question = f"direction of {subject}?"
            turn_word = "right" if turn == 1 else "left"
            target_rows = [f"{subject} faces {DIRECTIONS[start]}.", f"{subject} turns {turn_word}."]
        else:
            answer = ABSTENTION
            question = f"material of {subject}?"

        if kind == "transition":
            filler = [
                f"{name} faces {DIRECTIONS[(ordinal + offset) % len(DIRECTIONS)]}."
                for offset, name in enumerate(distractors)
            ]
        else:
            filler = [
                f"{name} uses {value}." for name, value in zip(distractors, materials, strict=True)
            ]
        rows: list[str | None] = [None] * OBSERVATIONS
        within_subset_ordinal = ordinal % (ITEMS_PER_KIND // 2)
        target_start = (
            within_subset_ordinal % (OBSERVATIONS - len(target_rows) + 1) if target_rows else 0
        )
        for offset, row in enumerate(target_rows):
            rows[target_start + offset] = row
        filler_iter = iter(filler)
        for position, row in enumerate(rows):
            if row is None:
                rows[position] = next(filler_iter)
        split = "sacrifice" if ordinal < ITEMS_PER_KIND // 2 else "confirm"
        episodes.append(
            {
                "id": f"v9-hard-{index:03d}",
                "split": "holdout",
                "subset": split,
                "kind": kind,
                "entities": [subject, *distractors],
                "observations": [str(row) for row in rows],
                "question": question,
                "answer": answer,
            }
        )

    subsets = {
        name: [episode["id"] for episode in episodes if episode["subset"] == name]
        for name in ("sacrifice", "confirm")
    }
    return {
        "schema": "n1-hard-neighbor-v9-confirmation-v1",
        "seed": seed,
        "items": ITEMS,
        "items_per_kind": ITEMS_PER_KIND,
        "observations_per_item": OBSERVATIONS,
        "entity_length": ENTITY_LENGTH,
        "name_neighbor_distance": DISTANCE,
        "subset_sizes": {name: len(ids) for name, ids in subsets.items()},
        "subsets": subsets,
        "exclusion_sources": {},
        "episodes": episodes,
    }


def audit_face(
    face: dict,
    *,
    train_items: list[dict],
    excluded_names: set[str],
    excluded_words: set[str],
) -> dict:
    episodes = face.get("episodes", [])
    ids = [episode.get("id") for episode in episodes]
    kinds = Counter(episode.get("kind") for episode in episodes)
    entities = {name for episode in episodes for name in episode.get("entities", [])}
    materials = {
        line.removesuffix(".").split()[-1]
        for episode in episodes
        for line in episode.get("observations", [])
        if " uses " in line
    }
    subset_map = face.get("subsets", {})
    subset_ids = {name: set(values) for name, values in subset_map.items()}
    per_subset_kind = {
        name: {
            kind: sum(
                episode["kind"] == kind for episode in episodes if episode["id"] in ids_for_subset
            )
            for kind in KINDS
        }
        for name, ids_for_subset in subset_ids.items()
    }
    distance_pairs = [
        sum(a != b for a, b in zip(episode["entities"][0], name, strict=True))
        for episode in episodes
        for name in episode["entities"][1:]
    ]
    known_words = set(excluded_words) | {
        line.removesuffix(".").split()[-1]
        for item in train_items
        for line in item["observations"]
        if " uses " in line
    }
    train_names = {name for item in train_items for name in item["entities"]}
    overlap_names = entities & (set(excluded_names) | train_names)
    overlap_words = materials & known_words
    semantic_errors = []
    target_starts = {subset: {kind: [] for kind in KINDS} for subset in ("sacrifice", "confirm")}
    duplicate_material_items = []
    for episode in episodes:
        subject = episode["entities"][0]
        rows = episode["observations"]
        target_rows = [row for row in rows if row.startswith(subject + " ")]
        if target_rows:
            target_starts[episode["subset"]][episode["kind"]].append(rows.index(target_rows[0]))
        item_materials = [line.removesuffix(".").split()[-1] for line in rows if " uses " in line]
        if len(item_materials) != len(set(item_materials)):
            duplicate_material_items.append(episode["id"])
        if episode["kind"] == "lookup":
            targets = [row for row in rows if row.startswith(subject + " uses ")]
            if len(targets) != 1 or targets[0].removesuffix(".").split()[-1] != episode["answer"]:
                semantic_errors.append(episode["id"])
        elif episode["kind"] == "revision":
            targets = [row for row in rows if row.startswith(subject + " ")]
            if (
                len(targets) != 2
                or not targets[0].startswith(subject + " uses ")
                or not targets[1].startswith(subject + " now uses ")
                or targets[1].removesuffix(".").split()[-1] != episode["answer"]
            ):
                semantic_errors.append(episode["id"])
        elif episode["kind"] == "unknown":
            if episode["answer"] != ABSTENTION or any(
                row.startswith(subject + " uses ") for row in rows
            ):
                semantic_errors.append(episode["id"])
        elif episode["kind"] == "transition":
            targets = [row for row in rows if row.startswith(subject + " ")]
            if (
                len(targets) != 2
                or not targets[0].startswith(subject + " faces ")
                or not targets[1].startswith(subject + " turns ")
            ):
                semantic_errors.append(episode["id"])

    target_position_balance = {}
    for subset in ("sacrifice", "confirm"):
        target_position_balance[subset] = {}
        for kind in ("lookup", "revision", "transition"):
            positions = target_starts[subset][kind]
            possible_positions = OBSERVATIONS - SUBJECT_ROWS[kind] + 1
            counts = Counter(positions)
            values = [counts[position] for position in range(possible_positions)]
            target_position_balance[subset][kind] = {
                "counts": {
                    str(position): counts[position] for position in range(possible_positions)
                },
                "balanced": len(positions) == ITEMS_PER_KIND // 2
                and max(values) - min(values) <= 1,
            }

    checks = {
        "frozen_recipe": (
            face.get("schema") == "n1-hard-neighbor-v9-confirmation-v1"
            and face.get("seed") == SEED
            and face.get("observations_per_item") == OBSERVATIONS
            and face.get("name_neighbor_distance") == DISTANCE
        ),
        "item_count": len(episodes) == ITEMS,
        "class_balance": kinds == Counter({kind: ITEMS_PER_KIND for kind in KINDS}),
        "unique_ids": len(set(ids)) == ITEMS,
        "subset_partition": (
            set(subset_ids) == {"sacrifice", "confirm"}
            and len(subset_ids["sacrifice"]) == 20
            and len(subset_ids["confirm"]) == 20
            and not (subset_ids["sacrifice"] & subset_ids["confirm"])
            and set(ids) == subset_ids["sacrifice"] | subset_ids["confirm"]
        ),
        "subset_class_balance": all(
            per_subset_kind[name] == {kind: 5 for kind in KINDS}
            for name in ("sacrifice", "confirm")
        ),
        "all_holdout": all(episode.get("split") == "holdout" for episode in episodes),
        "twelve_observations": all(
            len(episode.get("observations", [])) == OBSERVATIONS for episode in episodes
        ),
        "d1_name_confusability": bool(distance_pairs) and set(distance_pairs) == {DISTANCE},
        "fresh_entities": not overlap_names,
        "fresh_material_words": not overlap_words,
        "semantic_targets": not semantic_errors,
        "unique_materials_per_item": not duplicate_material_items,
        "balanced_target_positions": all(
            value["balanced"]
            for subset in target_position_balance.values()
            for value in subset.values()
        ),
        "unknown_label_reserved": ABSTENTION not in materials,
        "length_rotation": all(
            sum(len(episode["answer"]) == size and episode["kind"] == kind for episode in episodes)
            == 5
            for kind in ("lookup", "revision")
            for size in ANSWER_LENGTHS
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "per_kind": dict(sorted(kinds.items())),
        "per_subset_kind": per_subset_kind,
        "target_position_balance": target_position_balance,
        "entity_count": len(entities),
        "material_word_count": len(materials),
        "name_distance_pairs": len(distance_pairs),
        "name_distance_values": sorted(set(distance_pairs)),
        "entity_overlap_count": len(overlap_names),
        "material_word_overlap_count": len(overlap_words),
        "semantic_error_count": len(semantic_errors),
        "duplicate_material_item_count": len(duplicate_material_items),
        "parameter_updates": 0,
        "model_loads": 0,
        "model_queries": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "v7_confirm_payload_fields_decoded": False,
        "v7_confirm_model_queries": 0,
        "v8_confirm_payload_fields_decoded": False,
        "v8_confirm_model_queries": 0,
        "can_promote": False,
    }


def _training_precondition() -> dict:
    if not training.REPORT.is_file():
        raise FileNotFoundError("PLAN-N1-100 training report is required before v9 generation")
    if sha256(training.REPORT) != EXPECTED_SHA256["training_report"]:
        raise ValueError("PLAN-N1-100 training report bytes differ from the sealed evidence")
    report = read_json(training.REPORT)
    if not report.get("complete") or not report.get("training_sufficient"):
        raise ValueError("all nine training models and all frozen position gates must pass")
    identity = report.get("code_identity", {})
    if (
        identity.get("schema") != "n1-hard-neighbor-training-code-identity-v1"
        or not seal_ast.receipt_binding_matches(
            identity.get("controller_sha256"), Path(training.__file__).resolve()
        )
        or identity.get("schedule_sha256") != training.SCHEDULE_SHA256
    ):
        raise ValueError("training report is not bound to the frozen PLAN-N1-100 controller")
    if (
        len(report.get("models", [])) != 9
        or sum(
            model.get("arm") == "position" and model.get("training_sufficient") is True
            for model in report.get("models", [])
        )
        != 3
    ):
        raise ValueError(
            "all nine model receipts and all three position training gates are required"
        )
    if any(report.get(key) for key in ("dev_queries", "holdout_queries", "confirmation_queries")):
        raise ValueError("training report shows an unauthorized evaluation query")
    return report


def create_face() -> dict:
    if any(path.exists() for path in (FACE, CHECKS, ATTEMPT)):
        raise FileExistsError("PLAN-N1-102 v9 face is one-shot")
    if sha256(V5) != EXPECTED_SHA256["v5"] or sha256(V6) != EXPECTED_SHA256["v6"]:
        raise ValueError("v5/v6 exclusion face bytes differ from the frozen source faces")
    if sha256(PLAN) != EXPECTED_SHA256["plan"]:
        raise ValueError("PLAN-N1-102 bytes differ from the frozen approved protocol")
    training_report = _training_precondition()
    v7_names, v7_words, v7_provenance = exclusions_from_consumed_sacrifice(
        label="v7",
        path=V7,
        range_report_path=V7_RANGE,
        range_attempt_path=V7_RANGE_ATTEMPT,
        confirm_report_path=V7_CONFIRM,
        confirm_attempt_path=V7_CONFIRM_ATTEMPT,
    )
    v8_names, v8_words, v8_provenance = exclusions_from_consumed_sacrifice(
        label="v8",
        path=V8,
        range_report_path=V8_RANGE,
        range_attempt_path=V8_RANGE_ATTEMPT,
        confirm_report_path=V8_CONFIRM,
        confirm_attempt_path=V8_CONFIRM_ATTEMPT,
    )
    write_new(
        ATTEMPT,
        {
            "schema": "n1-hard-neighbor-v9-face-attempt-v1",
            "status": "started",
            "training_report_sha256": sha256(training.REPORT),
            "plan_sha256": sha256(PLAN),
            "training_code_identity": training_report.get("code_identity"),
            "prior_sacrifice_exclusion_sources": {"v7": v7_provenance, "v8": v8_provenance},
            "builder_sha256": sha256(Path(__file__).resolve()),
            "parameter_updates": 0,
            "model_loads": 0,
            "model_queries": 0,
            "v7_confirm_payload_fields_decoded": False,
            "v7_confirm_model_queries": 0,
            "v8_confirm_payload_fields_decoded": False,
            "v8_confirm_model_queries": 0,
        },
    )
    train_items, _ = train_data.train_items_from_frozen_manifest()
    v5 = read_json(V5)
    v6 = read_json(V6)
    excluded_names, excluded_words = exclusions_from_faces(v5, v6)
    excluded_names.update(name for item in train_items for name in item["entities"])
    excluded_words.update(
        line.removesuffix(".").split()[-1]
        for item in train_items
        for line in item["observations"]
        if " uses " in line
    )
    excluded_names.update(v7_names)
    excluded_words.update(v7_words)
    excluded_names.update(v8_names)
    excluded_words.update(v8_words)
    face = build_face(
        train_items=train_items,
        excluded_names=excluded_names,
        excluded_words=excluded_words,
    )
    face["exclusion_sources"] = {
        "v3_train_manifest_sha256": train_data.MANIFEST_SHA256,
        "v5_face_sha256": sha256(V5),
        "v6_face_sha256": sha256(V6),
        "v7_sacrifice": v7_provenance,
        "v8_sacrifice": v8_provenance,
        "training_report_sha256": sha256(training.REPORT),
        "plan_sha256": sha256(PLAN),
        "builder_sha256": sha256(Path(__file__).resolve()),
    }
    result = audit_face(
        face, train_items=train_items, excluded_names=excluded_names, excluded_words=excluded_words
    )
    if not result["accepted"]:
        write_new(CHECKS, {"accepted": False, "face_sha256": None, "report": result})
        raise ValueError("v9 face failed its frozen construction contract")
    write_new(FACE, face)
    result.update(
        {
            "schema": "n1-hard-neighbor-v9-face-checks-v1",
            "face": str(FACE),
            "face_sha256": sha256(FACE),
            "v5_sha256": sha256(V5),
            "v6_sha256": sha256(V6),
            "v7_sacrifice_exclusions": v7_provenance,
            "v8_sacrifice_exclusions": v8_provenance,
            "training_report_sha256": sha256(training.REPORT),
            "plan_sha256": sha256(PLAN),
            "builder_sha256": sha256(Path(__file__).resolve()),
        }
    )
    write_new(CHECKS, result)
    return result


if __name__ == "__main__":
    result = create_face()
    print(
        json.dumps(
            {
                "accepted": result["accepted"],
                "face": result["face"],
                "face_sha256": result["face_sha256"],
                "checks": result["checks"],
                "v7_confirm_payload_fields_decoded": False,
                "v7_confirm_model_queries": 0,
                "v8_confirm_payload_fields_decoded": False,
                "v8_confirm_model_queries": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
