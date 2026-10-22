"""构造并校验PLAN-N1-144的v23长上下文新确认面及来源排除并集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v22_confirmation_after_n142 as prior_face_builder
from scripts.training import eval_taiji_n1_v22_confirmation_after_n142 as n143_eval
from scripts.training import seal_ast

PLAN = (
    ROOT / "plans/reference/PLAN-N1-144_v23_long_context_confirmation_after_n143_draft_20261019.md"
)
PLAN_SHA256 = "b5527b5f8a56aa12604a7dbc40dad1fa74c3f4ea8e953a6b2539f1fa3c5fcf42"
SOURCE_PRECHECK = ROOT / "reports/taiji_n1_v23_source_union_precheck_20261019.json"
SOURCE_EXCLUSIONS = ROOT / "plans/manifests/n1_v23_after_n143_source_exclusions_20261019.json"
FACE = ROOT / "plans/manifests/n1_v23_confirmation_seed123_20261019.json"
FACE_ATTEMPT = ROOT / "plans/manifests/n1_v23_confirmation_seed123_20261019_attempt.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_v23_confirmation_face_checks_20261019.json"
PREPARED = ROOT / "reports/taiji_n1_v23_confirmation_face_readiness_20261019.json"

SEED = 123
ITEMS = 160
ITEMS_PER_KIND = 40
ITEMS_PER_SUBSET_KIND = 20
OBSERVATIONS = 32
ENTITY_LENGTH = 10
DISTANCE = 1
KINDS = ("lookup", "revision", "transition", "unknown")
DIRECTIONS = tuple(prior_face_builder.DIRECTIONS)
LETTERS = "abcdefghijklmnopqrstuvwxyz"
ABSTENTION = "unknown"
SUBSETS = ("sacrifice", "confirm")
LOOKUP_POSITIONS = (0, 8, 16, 24, 31)
PAIR_POSITIONS = (0, 8, 15, 23, 30)
EXPECTED_BASE_ENTITY_COUNT = 29_202
EXPECTED_BASE_WORD_COUNT = 8_208
EXPECTED_PRIOR_FACE_ENTITY_COUNT = 2_520
EXPECTED_PRIOR_FACE_WORD_COUNT = 1_920
EXPECTED_ENTITY_EXCLUSION_COUNT = 31_722
EXPECTED_WORD_EXCLUSION_COUNT = 10_128


class Refuse(RuntimeError):
    """一项冻结前置失败；不重抽或覆盖write-once输出。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_strings(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _merge_hashes(target: dict[str, str], incoming: dict[str, str]) -> None:
    for path, digest in incoming.items():
        resolved = str(Path(path).resolve())
        prior = target.get(resolved)
        if prior is not None and prior != digest:
            raise Refuse(f"同一来源出现冲突哈希：{resolved}")
        target[resolved] = digest


def _verify_hashes(source_hashes: dict[str, str]) -> bool:
    """S3 切换: 经 seal_ast 三通道校验（在位字节 / 在位 AST / git 历史取证）。"""
    return seal_ast.verify_hash_dict(source_hashes)[0]


def _filler_names(bound: dict) -> set[str]:
    names: set[str] = set()
    for variant in bound["context_variants"].values():
        for distance_data in variant["by_distance"].values():
            names.update(str(name) for name in distance_data["entities"])
    return names


def _collect_source_union() -> tuple[set[str], set[str], dict[str, str], dict]:
    old_manifest = read_json(prior_face_builder.SOURCE_EXCLUSIONS)
    prior_face = read_json(prior_face_builder.FACE)
    prior_checks = read_json(prior_face_builder.FACE_CHECKS)
    prior_prepared = read_json(prior_face_builder.PREPARED)
    n143_plan = prior_face_builder.PLAN
    n143_closeout = ROOT / "plans/reference/PLAN-N1-143_execution_closeout_20261019.md"
    n143_result = read_json(n143_eval.RESULT)
    n143_attempt = read_json(n143_eval.ATTEMPT)
    n143_readiness = read_json(n143_eval.READINESS)
    if (
        not n143_closeout.is_file()
        or n143_result.get("complete") is not True
        or n143_result.get("status") != "range_not_met"
        or n143_result.get("sacrifice_queries") != 960
        or n143_result.get("confirmation_queries") != 0
        or n143_result.get("total_v22_queries") != 960
        or n143_result.get("parameter_updates") != 0
        or n143_result.get("dev_queries") != 0
        or n143_result.get("v3_holdout_queries") != 0
        or n143_result.get("can_promote") is not False
        or n143_result.get("source_hashes_unchanged") is not True
        or n143_result.get("range_summary", {}).get("qualified_endpoints") != 6
        or n143_result.get("range_summary", {}).get("required_qualified_endpoints") != 8
        or len(n143_result.get("sacrifice_rows", [])) != 960
        or n143_result.get("confirmation_rows") != []
        or n143_attempt.get("status") != "started"
        or n143_attempt.get("parameter_updates") != 0
        or n143_attempt.get("dev_queries") != 0
        or n143_attempt.get("v3_holdout_queries") != 0
        or n143_readiness.get("status") != "ready_for_sacrifice"
        or n143_readiness.get("complete") is not True
        or n143_readiness.get("independent_full_bundle_restores") != 12
        or n143_readiness.get("source_hashes_unchanged") is not True
        or old_manifest.get("schema") != "n1-plan143-v22-source-exclusions-v1"
        or old_manifest.get("entity_count") != EXPECTED_BASE_ENTITY_COUNT
        or old_manifest.get("word_value_count") != EXPECTED_BASE_WORD_COUNT
        or len(old_manifest.get("excluded_entities", [])) != EXPECTED_BASE_ENTITY_COUNT
        or len(old_manifest.get("excluded_word_values", [])) != EXPECTED_BASE_WORD_COUNT
        or prior_face.get("schema") != "n1-v22-confirmation-face-v1"
        or prior_face.get("seed") != 122
        or len(prior_face.get("episodes", [])) != 160
        or prior_checks.get("accepted") is not True
        or not seal_ast.receipt_binding_matches(
            prior_checks.get("face_sha256"), prior_face_builder.FACE
        )
        or prior_prepared.get("complete") is not True
    ):
        raise Refuse("N1-143结案、v22 face或来源清单不符合N1-144前置")
    if (
        sha256(n143_eval.ATTEMPT) != n143_result.get("attempt_sha256")
        or sha256(n143_eval.PROGRESS) != n143_result.get("progress_sha256")
        or not prior_face_builder._verify_hashes(old_manifest.get("source_hashes", {}))
        or not prior_face_builder._verify_hashes(prior_prepared.get("source_hashes", {}))
        or not prior_face_builder._verify_hashes(n143_readiness.get("source_hashes", {}))
    ):
        raise Refuse("N1-143 attempt/progress/readiness/source hashes changed")

    excluded_entities = set(old_manifest["excluded_entities"])
    excluded_words = set(old_manifest["excluded_word_values"])
    face_entities = {name for item in prior_face["episodes"] for name in item["entities"]}
    face_materials = {
        row.removesuffix(".").split()[-1]
        for item in prior_face["episodes"]
        for row in item["observations"]
        if " uses " in row
    }
    face_materials |= {
        item["answer"]
        for item in prior_face["episodes"]
        if item.get("kind") in ("lookup", "revision")
    }
    if (
        len(face_entities) != EXPECTED_PRIOR_FACE_ENTITY_COUNT
        or len(face_materials) != EXPECTED_PRIOR_FACE_WORD_COUNT
        or face_entities & excluded_entities
        or face_materials & excluded_words
        or "unknown" not in excluded_words
        or not set(prior_face_builder.DIRECTIONS) <= excluded_words
    ):
        raise Refuse("v22 consumed face inventory differs from the frozen source audit")
    excluded_entities.update(face_entities)
    excluded_words.update(face_materials)
    if (
        len(excluded_entities) != EXPECTED_ENTITY_EXCLUSION_COUNT
        or len(excluded_words) != EXPECTED_WORD_EXCLUSION_COUNT
    ):
        raise Refuse("N1-143 face inclusion changed the frozen N1-144 exclusion union")

    source_hashes = dict(n143_result.get("source_hashes", {}))
    for path, digest in n143_result.get("source_hashes", {}).items():
        if not Path(path).is_file() or sha256(path) != digest:
            raise Refuse(f"N1-143 source hash mismatch: {path}")
    for path in (
        PLAN,
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("eval_taiji_n1_v23_confirmation_after_n143.py"),
        ROOT / "tests/taiji_native/test_n1_plan144_v23_face_builder_contract.py",
        ROOT / "tests/taiji_native/test_n1_plan144_v23_confirmation_runner_contract.py",
        n143_plan,
        n143_closeout,
        n143_eval.READINESS,
        n143_eval.READINESS_ATTEMPT,
        n143_eval.SOURCE_BINDING,
        n143_eval.ATTEMPT,
        n143_eval.PROGRESS,
        n143_eval.RESULT,
        prior_face_builder.SOURCE_PRECHECK,
        prior_face_builder.SOURCE_EXCLUSIONS,
        prior_face_builder.FACE,
        prior_face_builder.FACE_ATTEMPT,
        prior_face_builder.FACE_CHECKS,
        prior_face_builder.PREPARED,
    ):
        path = Path(path).resolve()
        if not path.is_file():
            raise Refuse(f"N1-143/v22 source missing: {path}")
        source_hashes[str(path)] = sha256(path)

    summary = {
        "n1_143_entity_exclusion_count": len(old_manifest["excluded_entities"]),
        "n1_143_word_exclusion_count": len(old_manifest["excluded_word_values"]),
        "consumed_v22_face_entity_count": len(face_entities),
        "consumed_v22_face_material_value_count": len(face_materials),
        "entity_overlap_with_old_union": len(
            face_entities & set(old_manifest["excluded_entities"])
        ),
        "word_overlap_with_old_union": len(
            face_materials & set(old_manifest["excluded_word_values"])
        ),
        "entity_exclusion_count": len(excluded_entities),
        "entity_exclusion_sha256": hashlib.sha256(
            "\n".join(sorted(excluded_entities)).encode("utf-8")
        ).hexdigest(),
        "word_exclusion_count": len(excluded_words),
        "word_exclusion_sha256": hashlib.sha256(
            "\n".join(sorted(excluded_words)).encode("utf-8")
        ).hexdigest(),
        "partial_v22_sacrifice_queries_excluded": n143_result["sacrifice_queries"],
        "partial_v22_confirmation_queries": n143_result["confirmation_queries"],
        "source_hash_count": len(source_hashes),
    }
    return excluded_entities, excluded_words, source_hashes, summary


def _novel(length: int, pool: set[str], rng: random.Random) -> str:
    for _ in range(100_000):
        value = "".join(rng.choices(LETTERS, k=length))
        if value not in pool:
            pool.add(value)
            return value
    raise Refuse("unique string pool exhausted")


def _confusable(subject: str, pool: set[str], rng: random.Random) -> str:
    for _ in range(100_000):
        chars = list(subject)
        position = rng.randrange(len(chars))
        chars[position] = rng.choice(LETTERS.replace(chars[position], ""))
        value = "".join(chars)
        if value not in pool:
            pool.add(value)
            return value
    raise Refuse("unique d=1 neighbor pool exhausted")


def build_face(excluded_entities: set[str], excluded_words: set[str], seed: int = SEED) -> dict:
    """纯内存、可复现地生成160题v23；不写文件、不查询模型。"""
    rng = random.Random(seed)
    used_entities = set(excluded_entities)
    used_words = set(excluded_words) | set(DIRECTIONS) | {ABSTENTION}
    episodes = []
    answer_lengths = Counter()
    starts = Counter()
    direction_counts = Counter()
    turn_counts = Counter()

    for subset in SUBSETS:
        for kind in KINDS:
            for local in range(ITEMS_PER_SUBSET_KIND):
                index = len(episodes)
                subject = _novel(ENTITY_LENGTH, used_entities, rng)
                answer_length = 6 if local % 2 == 0 else 7
                answer_lengths[(subset, kind, answer_length)] += 1
                start = (
                    LOOKUP_POSITIONS[local % len(LOOKUP_POSITIONS)]
                    if kind == "lookup"
                    else (
                        PAIR_POSITIONS[local % len(PAIR_POSITIONS)]
                        if kind in ("revision", "transition")
                        else None
                    )
                )
                if start is not None:
                    starts[(subset, kind, start)] += 1

                target_rows = (
                    1 if kind == "lookup" else 2 if kind in ("revision", "transition") else 0
                )
                distractors = [
                    _confusable(subject, used_entities, rng)
                    for _ in range(OBSERVATIONS - target_rows)
                ]
                observations: list[str | None] = [None] * OBSERVATIONS
                target_indices: list[int] = []
                answer_direction = None
                turn_template = None
                if kind == "lookup":
                    answer = _novel(answer_length, used_words, rng)
                    observations[start] = f"{subject} uses {answer}."
                    target_indices = [start]
                    question = f"material of {subject}?"
                elif kind == "revision":
                    stale_length = 6 if local % 2 == 0 else 7
                    stale = _novel(stale_length, used_words, rng)
                    answer = _novel(answer_length, used_words, rng)
                    observations[start] = f"{subject} uses {stale}."
                    observations[start + 1] = f"{subject} now uses {answer}."
                    target_indices = [start, start + 1]
                    question = f"material of {subject}?"
                elif kind == "transition":
                    answer_direction = DIRECTIONS[local % len(DIRECTIONS)]
                    turn_parity = (local // len(DIRECTIONS) + local % len(DIRECTIONS)) % 2
                    turn_template = "right" if turn_parity == 0 else "left"
                    delta = 1 if turn_template == "right" else -1
                    initial = DIRECTIONS[
                        (DIRECTIONS.index(answer_direction) - delta) % len(DIRECTIONS)
                    ]
                    answer = answer_direction
                    observations[start] = f"{subject} faces {initial}."
                    observations[start + 1] = f"{subject} turns {turn_template}."
                    target_indices = [start, start + 1]
                    direction_counts[(subset, answer_direction)] += 1
                    turn_counts[(subset, turn_template)] += 1
                    question = f"direction of {subject}?"
                else:
                    answer = ABSTENTION
                    question = f"material of {subject}?"

                filler_indices = [
                    position for position, row in enumerate(observations) if row is None
                ]
                require(
                    len(filler_indices) == len(distractors), f"{kind} filler slot count differs"
                )
                filler_rows = []
                for number, name in enumerate(distractors):
                    if kind == "transition":
                        direction = DIRECTIONS[(local + number + index) % len(DIRECTIONS)]
                        row = f"{name} faces {direction}."
                    else:
                        material_length = 6 if (local + number + index) % 2 == 0 else 7
                        material = _novel(material_length, used_words, rng)
                        row = f"{name} uses {material}."
                    filler_rows.append(row)
                for position, row in zip(filler_indices, filler_rows, strict=True):
                    observations[position] = row

                require(
                    all(row is not None for row in observations), f"{kind} context has empty rows"
                )
                entities = [subject, *distractors]
                require(len(entities) == len(set(entities)), f"{kind} episode repeats an entity")
                require(
                    all(len(name) == ENTITY_LENGTH for name in entities),
                    f"{kind} entity length differs",
                )
                require(
                    all(_hamming(subject, name) == DISTANCE for name in distractors),
                    f"{kind} d=1 differs",
                )
                episodes.append(
                    {
                        "id": f"v23-{subset}-{kind}-{local:03d}",
                        "split": "holdout",
                        "subset": subset,
                        "kind": kind,
                        "entities": entities,
                        "observations": [str(row) for row in observations],
                        "question": question,
                        "answer": answer,
                        "target_fact_indices": target_indices,
                        "answer_direction": answer_direction,
                        "turn_template": turn_template,
                        "target_start_index": start,
                        "answer_length": len(answer) if kind in ("lookup", "revision") else None,
                    }
                )

    return {
        "schema": "n1-v23-confirmation-face-v1",
        "seed": seed,
        "items": ITEMS,
        "items_per_kind": ITEMS_PER_KIND,
        "observations_per_item": OBSERVATIONS,
        "entity_length": ENTITY_LENGTH,
        "name_neighbor_distance": DISTANCE,
        "answer_lengths": [6, 7],
        "target_start_positions": {
            "lookup": list(LOOKUP_POSITIONS),
            "revision": list(PAIR_POSITIONS),
            "transition": list(PAIR_POSITIONS),
            "unknown": [],
        },
        "episodes": episodes,
        "shape_counts": {
            "items_by_subset_and_kind": {
                f"{subset}/{kind}": sum(
                    item["subset"] == subset and item["kind"] == kind for item in episodes
                )
                for subset in SUBSETS
                for kind in KINDS
            },
            "answer_length_counts": {
                f"{subset}/{kind}/{length}": count
                for (subset, kind, length), count in sorted(answer_lengths.items())
            },
            "target_start_counts": {
                f"{subset}/{kind}/{position}": count
                for (subset, kind, position), count in sorted(starts.items())
            },
            "direction_counts": {
                f"{subset}/{direction}": count
                for (subset, direction), count in sorted(direction_counts.items())
            },
            "turn_counts": {
                f"{subset}/{turn}": count for (subset, turn), count in sorted(turn_counts.items())
            },
            "new_entity_count": len(used_entities - excluded_entities),
            "new_word_count": len(used_words - excluded_words - set(DIRECTIONS) - {ABSTENTION}),
        },
    }


def _hamming(first: str, second: str) -> int:
    if len(first) != len(second):
        return max(len(first), len(second))
    return sum(a != b for a, b in zip(first, second, strict=True))


def _validate_item_target_semantics(item: dict, start: int | None) -> None:
    subject = item["entities"][0]
    observations = item["observations"]
    kind = item["kind"]
    valid = False
    if kind == "lookup":
        answer = item["answer"]
        valid = (
            item["question"] == f"material of {subject}?"
            and observations[start] == f"{subject} uses {answer}."
            and item.get("answer_length") == len(answer)
            and len(answer) in (6, 7)
        )
    elif kind == "revision":
        stale = observations[start].removesuffix(".").split()[-1]
        answer = item["answer"]
        valid = (
            item["question"] == f"material of {subject}?"
            and observations[start] == f"{subject} uses {stale}."
            and observations[start + 1] == f"{subject} now uses {answer}."
            and stale != answer
            and item.get("answer_length") == len(answer)
            and len(answer) in (6, 7)
        )
    elif kind == "transition":
        direction = item.get("answer_direction")
        turn = item.get("turn_template")
        if direction in DIRECTIONS and turn in ("left", "right"):
            delta = 1 if turn == "right" else -1
            initial = DIRECTIONS[(DIRECTIONS.index(direction) - delta) % len(DIRECTIONS)]
            valid = (
                item["question"] == f"direction of {subject}?"
                and item["answer"] == direction
                and observations[start] == f"{subject} faces {initial}."
                and observations[start + 1] == f"{subject} turns {turn}."
                and item.get("answer_length") is None
            )
    elif kind == "unknown":
        valid = (
            item["question"] == f"material of {subject}?"
            and item["answer"] == ABSTENTION
            and item.get("answer_length") is None
            and all(not row.startswith(f"{subject} ") for row in observations)
        )
    require(valid, f"v23 gold target semantics mismatch: {item.get('id')}")


def validate_face(face: dict, excluded_entities: set[str], excluded_words: set[str]) -> dict:
    episodes = face.get("episodes", [])
    require(face.get("schema") == "n1-v23-confirmation-face-v1", "v23 schema mismatch")
    require(face.get("seed") == SEED and len(episodes) == ITEMS, "v23 seed/item count mismatch")
    require(len({item.get("id") for item in episodes}) == ITEMS, "v23 IDs are not unique")
    require(all(item.get("split") == "holdout" for item in episodes), "v23 split must be holdout")
    require(
        all(len(item.get("observations", [])) == OBSERVATIONS for item in episodes),
        "v23 observation counts differ",
    )
    require(
        all(
            len(item.get("entities", []))
            == 1
            + OBSERVATIONS
            - (
                1
                if item["kind"] == "lookup"
                else 2 if item["kind"] in ("revision", "transition") else 0
            )
            for item in episodes
        ),
        "v23 entity shape differs",
    )
    require(
        all(len(set(item["entities"])) == len(item["entities"]) for item in episodes),
        "duplicate entity within an episode",
    )
    all_names = {name for item in episodes for name in item["entities"]}
    require(
        not (all_names & excluded_entities), "v23 entity overlaps consumed/source exclusion union"
    )
    require(
        len(all_names) == sum(len(item["entities"]) for item in episodes),
        "v23 entity repeats across episodes",
    )
    material_values = {
        row.removesuffix(".").split()[-1]
        for item in episodes
        for row in item["observations"]
        if " uses " in row
    } | {item["answer"] for item in episodes if item["kind"] in ("lookup", "revision")}
    require(
        not (material_values & excluded_words),
        "v23 material/answer overlaps consumed word exclusions",
    )
    require(
        all(len(value) in (6, 7) for value in material_values), "v23 materials must be 6/7 chars"
    )
    require(
        all(
            _hamming(item["entities"][0], name) == DISTANCE
            for item in episodes
            for name in item["entities"][1:]
        ),
        "v23 Hamming-distance contract failed",
    )
    require(
        face.get("target_start_positions")
        == {
            "lookup": list(LOOKUP_POSITIONS),
            "revision": list(PAIR_POSITIONS),
            "transition": list(PAIR_POSITIONS),
            "unknown": [],
        },
        "v23 target-start schedule differs",
    )
    for item in episodes:
        kind = item["kind"]
        local = int(item["id"].rsplit("-", 1)[-1])
        expected_start = (
            LOOKUP_POSITIONS[local % len(LOOKUP_POSITIONS)]
            if kind == "lookup"
            else (
                PAIR_POSITIONS[local % len(PAIR_POSITIONS)]
                if kind in ("revision", "transition")
                else None
            )
        )
        expected_indices = (
            [expected_start]
            if kind == "lookup"
            else [expected_start, expected_start + 1] if kind in ("revision", "transition") else []
        )
        require(
            item.get("target_start_index") == expected_start
            and item.get("target_fact_indices") == expected_indices,
            f"v23 target position/index mismatch: {item.get('id')}",
        )
        _validate_item_target_semantics(item, expected_start)
    counts = Counter((item["subset"], item["kind"]) for item in episodes)
    require(
        all(
            counts[(subset, kind)] == ITEMS_PER_SUBSET_KIND for subset in SUBSETS for kind in KINDS
        ),
        "v23 subset balance differs",
    )
    directions = Counter(
        (item["subset"], item["answer_direction"])
        for item in episodes
        if item["kind"] == "transition"
    )
    turns = Counter(
        (item["subset"], item["turn_template"]) for item in episodes if item["kind"] == "transition"
    )
    require(
        all(directions[(subset, direction)] == 5 for subset in SUBSETS for direction in DIRECTIONS),
        "v23 direction balance differs",
    )
    require(
        all(turns[(subset, turn)] == 10 for subset in SUBSETS for turn in ("left", "right")),
        "v23 direction balance differs",
    )
    return {
        "accepted": True,
        "items": len(episodes),
        "items_by_subset_and_kind": {f"{s}/{k}": counts[(s, k)] for s in SUBSETS for k in KINDS},
        "entity_count": len(all_names),
        "material_word_count": len(material_values),
        "face_seed": face["seed"],
        "schema": face["schema"],
    }


def prepare_face() -> dict:
    if not PLAN.is_file() or sha256(PLAN) != PLAN_SHA256:
        raise Refuse("PLAN-N1-144 frozen SHA-256 mismatch")
    output_paths = (SOURCE_PRECHECK, SOURCE_EXCLUSIONS, FACE, FACE_ATTEMPT, FACE_CHECKS, PREPARED)
    if any(path.exists() for path in output_paths):
        raise FileExistsError("PLAN-N1-144 face preparation is write-once")
    started = time.process_time()
    excluded_entities, excluded_words, source_hashes, union_summary = _collect_source_union()
    source_hashes[str(PLAN.resolve())] = sha256(PLAN)
    source_hashes[str(Path(__file__).resolve())] = sha256(Path(__file__))
    require(_verify_hashes(source_hashes), "source union changed during static preparation")
    precheck = {
        "schema": "n1-plan144-v23-source-union-precheck-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": True,
        "source_union": union_summary,
        "new_face_generated": False,
        "model_queries": 0,
        "parameter_updates": 0,
        "cpu_seconds": round(time.process_time() - started, 6),
        "can_promote": False,
    }
    write_new(SOURCE_PRECHECK, precheck)
    manifest = {
        "schema": "n1-plan144-v23-source-exclusions-v1",
        "status": "ready_for_one_time_face_generation",
        "plan_sha256": PLAN_SHA256,
        "seed": SEED,
        "entity_count": len(excluded_entities),
        "word_value_count": len(excluded_words),
        "entity_exclusion_sha256": digest_strings(excluded_entities),
        "word_exclusion_sha256": digest_strings(excluded_words),
        "excluded_entities": sorted(excluded_entities),
        "excluded_word_values": sorted(excluded_words),
        "provenance": union_summary,
        "source_hashes": source_hashes,
        "can_promote": False,
    }
    write_new(SOURCE_EXCLUSIONS, manifest)
    write_new(
        FACE_ATTEMPT,
        {
            "schema": "n1-plan144-v23-face-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "source_union_sha256": sha256(SOURCE_EXCLUSIONS),
            "seed": SEED,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    face = build_face(excluded_entities, excluded_words)
    face["plan_sha256"] = PLAN_SHA256
    face["source_exclusion_sha256"] = sha256(SOURCE_EXCLUSIONS)
    face["source_exclusion_entity_sha256"] = manifest["entity_exclusion_sha256"]
    face["source_exclusion_word_sha256"] = manifest["word_exclusion_sha256"]
    face["source_hashes"] = source_hashes
    write_new(FACE, face)
    checks = validate_face(face, excluded_entities, excluded_words)
    checks.update(
        {
            "plan_sha256": PLAN_SHA256,
            "face_sha256": sha256(FACE),
            "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
            "source_hashes_unchanged": _verify_hashes(source_hashes),
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        }
    )
    if checks["source_hashes_unchanged"] is not True:
        raise Refuse("source changed during v23 face construction")
    write_new(FACE_CHECKS, checks)
    prepared = {
        "schema": "n1-plan144-v23-face-readiness-v1",
        "status": "ready_for_checkpoint_readiness",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_precheck_sha256": sha256(SOURCE_PRECHECK),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": _verify_hashes(source_hashes),
        "model_queries": 0,
        "parameter_updates": 0,
        "cpu_seconds": round(time.process_time() - started, 6),
        "summary_zh": "v23长上下文新面构造及160题合同检查通过；没有加载模型、恢复checkpoint、训练或查询，可进入12端点checkpoint readiness。",
        "can_promote": False,
    }
    write_new(PREPARED, prepared)
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-144 v23 face builder and source audit")
    parser.add_argument("--prepare-face", action="store_true")
    args = parser.parse_args()
    if not args.prepare_face:
        parser.error("one-shot invocation requires --prepare-face")
    result = prepare_face()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "cpu_seconds",
                    "model_queries",
                    "parameter_updates",
                    "can_promote",
                    "summary_zh",
                )
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
