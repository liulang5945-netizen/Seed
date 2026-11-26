"""Create the one-shot PLAN-N1-100 v7 sacrifice/confirmation face."""

from __future__ import annotations

import hashlib
import json
import random
import string
from collections import Counter
from pathlib import Path

from scripts.training import build_taiji_n1_hard_neighbor_training_schedule as train_data
from scripts.training import run_taiji_n1_hard_neighbor_training as training

ROOT = Path(__file__).resolve().parents[2]
V5 = ROOT / "plans/manifests/n1_fresh_confirmation_v5_20261016.json"
V6 = ROOT / "plans/manifests/n1_fresh_confirmation_v6_20261017.json"
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v7_20261017.json"
CHECKS = ROOT / "reports/taiji_n1_hard_neighbor_v7_face_checks_20261017.json"
ATTEMPT = CHECKS.with_name(CHECKS.stem + "_attempt.json")
SEED = 107
ITEMS_PER_KIND = 10
ITEMS = 40
OBSERVATIONS = 4
ENTITY_LENGTH = 10
ANSWER_LENGTHS = (6, 7)
DISTANCE = 1
KINDS = ("lookup", "revision", "transition", "unknown")
DIRECTIONS = ("north", "east", "south", "west")
SUBJECT_ROWS = {"lookup": 1, "revision": 2, "transition": 2, "unknown": 0}
ABSTENTION = "unknown"
LETTERS = string.ascii_lowercase


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def _novel(length: int, used: set[str], rng: random.Random) -> str:
    for _ in range(10000):
        value = "".join(rng.choices(LETTERS, k=length))
        if value not in used:
            used.add(value)
            return value
    raise ValueError("v7 generator exhausted unique names or answer values")


def _confusable(subject: str, used: set[str], rng: random.Random) -> str:
    for _ in range(500):
        chars = list(subject)
        position = rng.randrange(ENTITY_LENGTH)
        chars[position] = rng.choice(LETTERS.replace(chars[position], ""))
        candidate = "".join(chars)
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError("v7 generator could not create an exact d=1 entity")


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
        target_start = ordinal % (OBSERVATIONS - len(target_rows) + 1) if target_rows else 0
        for offset, row in enumerate(target_rows):
            rows[target_start + offset] = row
        filler_iter = iter(filler)
        for position, row in enumerate(rows):
            if row is None:
                rows[position] = next(filler_iter)
        split = "sacrifice" if ordinal < ITEMS_PER_KIND // 2 else "confirm"
        episodes.append(
            {
                "id": f"v7-hard-{index:03d}",
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
    face = {
        "schema": "n1-hard-neighbor-v7-confirmation-v1",
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
    return face


def audit_face(
    face: dict, *, train_items: list[dict], excluded_names: set[str], excluded_words: set[str]
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
    for episode in episodes:
        subject = episode["entities"][0]
        rows = episode["observations"]
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

    checks = {
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
        "four_observations": all(
            len(episode.get("observations", [])) == OBSERVATIONS for episode in episodes
        ),
        "d1_name_confusability": bool(distance_pairs) and set(distance_pairs) == {DISTANCE},
        "fresh_entities": not overlap_names,
        "fresh_material_words": not overlap_words,
        "semantic_targets": not semantic_errors,
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
        "entity_count": len(entities),
        "material_word_count": len(materials),
        "name_distance_pairs": len(distance_pairs),
        "name_distance_values": sorted(set(distance_pairs)),
        "entity_overlap_count": len(overlap_names),
        "material_word_overlap_count": len(overlap_words),
        "semantic_error_count": len(semantic_errors),
        "parameter_updates": 0,
        "model_loads": 0,
        "model_queries": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "can_promote": False,
    }


def _training_precondition() -> dict:
    if not training.REPORT.is_file():
        raise FileNotFoundError("PLAN-N1-100 training report is required before v7 generation")
    report = read_json(training.REPORT)
    if not report.get("complete") or not report.get("training_sufficient"):
        raise ValueError("all nine training models and all frozen position gates must pass")
    identity = report.get("code_identity", {})
    if (
        identity.get("schema") != "n1-hard-neighbor-training-code-identity-v1"
        or identity.get("controller_sha256") != sha256(Path(training.__file__).resolve())
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
        raise FileExistsError("PLAN-N1-100 v7 face is one-shot")
    training_report = _training_precondition()
    write_new(
        ATTEMPT,
        {
            "schema": "n1-hard-neighbor-v7-face-attempt-v1",
            "status": "started",
            "training_report_sha256": sha256(training.REPORT),
            "training_code_identity": training_report.get("code_identity"),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "parameter_updates": 0,
            "model_loads": 0,
            "model_queries": 0,
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
    face = build_face(
        train_items=train_items,
        excluded_names=excluded_names,
        excluded_words=excluded_words,
    )
    face["exclusion_sources"] = {
        "v3_train_manifest_sha256": train_data.MANIFEST_SHA256,
        "v5_face_sha256": sha256(V5),
        "v6_face_sha256": sha256(V6),
        "training_report_sha256": sha256(training.REPORT),
        "builder_sha256": sha256(Path(__file__).resolve()),
    }
    result = audit_face(
        face, train_items=train_items, excluded_names=excluded_names, excluded_words=excluded_words
    )
    if not result["accepted"]:
        write_new(CHECKS, {"accepted": False, "face_sha256": None, "report": result})
        raise ValueError("v7 face failed its frozen construction contract")
    write_new(FACE, face)
    result.update(
        {
            "schema": "n1-hard-neighbor-v7-face-checks-v1",
            "face": str(FACE),
            "face_sha256": sha256(FACE),
            "v5_sha256": sha256(V5),
            "v6_sha256": sha256(V6),
            "training_report_sha256": sha256(training.REPORT),
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
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
