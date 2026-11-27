"""构造PLAN-N1-149的全新seed126嵌套上下文弃答诊断面。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v25_abstention_after_n147 as n148_builder
from scripts.training import eval_taiji_n1_v25_abstention_after_n147 as n148_eval
from scripts.training import seal_ast  # noqa: E402

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-149_matched_abstention_context_length_curve_diagnostic_draft_20261019.md"
)
PLAN_SHA256 = "d6dfb5a9e6e6f5d563a7d137ef838a197c7bbff98cbca52d85fe4a11de2e6497"
SOURCE_PRECHECK = (
    ROOT / "reports/taiji_n1_v26_abstention_length_source_union_precheck_20261019.json"
)
SOURCE_EXCLUSIONS = (
    ROOT / "plans/manifests/n1_v26_abstention_length_after_n148_source_exclusions_20261019.json"
)
FACE = ROOT / "plans/manifests/n1_v26_abstention_length_seed126_20261019.json"
FACE_ATTEMPT = ROOT / "plans/manifests/n1_v26_abstention_length_seed126_20261019_attempt.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_v26_abstention_length_face_checks_20261019.json"
PREPARED = ROOT / "reports/taiji_n1_v26_abstention_length_face_readiness_20261019.json"

N148_SOURCE_MANIFEST = n148_builder.SOURCE_EXCLUSIONS
N148_FACE = n148_builder.FACE
N148_FACE_CHECKS = n148_builder.FACE_CHECKS
N148_PREPARED = n148_builder.PREPARED
N148_READINESS = n148_eval.READINESS
N148_SOURCE_BINDING = n148_eval.SOURCE_BINDING
N148_ATTEMPT = n148_eval.ATTEMPT
N148_PROGRESS = n148_eval.PROGRESS
N148_RESULT = n148_eval.RESULT

SEED = 126
LOGICAL_PAIRS = 40
CONTEXT_LENGTHS = (8, 16, 32)
ITEMS_PER_LENGTH = LOGICAL_PAIRS * 2
ITEMS = LOGICAL_PAIRS * len(CONTEXT_LENGTHS) * 2
OBSERVATIONS_MAX = max(CONTEXT_LENGTHS)
ENTITY_LENGTH = 10
DISTANCE = 1
KINDS = ("lookup", "unknown")
CONDITIONS = ("known", "unknown")
LETTERS = "abcdefghijklmnopqrstuvwxyz"
ABSTENTION = "unknown"
EXPECTED_BASE_ENTITY_COUNT = 38_122
EXPECTED_BASE_WORD_COUNT = 15_288
EXPECTED_V25_ENTITY_COUNT = 1_320
EXPECTED_V25_WORD_COUNT = 1_320
EXPECTED_ENTITY_EXCLUSION_COUNT = 39_442
EXPECTED_WORD_EXCLUSION_COUNT = 16_608


class Refuse(RuntimeError):
    """来源或嵌套配对合同失败；不重抽。"""


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


def _material_values(face: dict) -> set[str]:
    values = {
        row.removesuffix(".").split()[-1]
        for item in face.get("episodes", [])
        for row in item.get("observations", [])
        if " uses " in row
    }
    values.update(
        item["answer"] for item in face.get("episodes", []) if item.get("kind") == "lookup"
    )
    return values


def _collect_source_union() -> tuple[set[str], set[str], dict[str, str], dict]:
    old_manifest = read_json(N148_SOURCE_MANIFEST)
    prior_face = read_json(N148_FACE)
    prior_checks = read_json(N148_FACE_CHECKS)
    prior_prepared = read_json(N148_PREPARED)
    n148_readiness = read_json(N148_READINESS)
    n148_binding = read_json(N148_SOURCE_BINDING)
    n148_result = read_json(N148_RESULT)
    require(
        old_manifest.get("schema") == "n1-plan148-v25-source-exclusions-v1"
        and old_manifest.get("entity_count") == EXPECTED_BASE_ENTITY_COUNT
        and old_manifest.get("word_value_count") == EXPECTED_BASE_WORD_COUNT
        and len(old_manifest.get("excluded_entities", [])) == EXPECTED_BASE_ENTITY_COUNT
        and len(old_manifest.get("excluded_word_values", [])) == EXPECTED_BASE_WORD_COUNT
        and prior_face.get("schema") == "n1-v25-matched-abstention-diagnostic-face-v1"
        and prior_face.get("seed") == 125
        and len(prior_face.get("episodes", [])) == 80
        and prior_checks.get("accepted") is True
        and seal_ast.receipt_binding_matches(prior_checks.get("face_sha256"), N148_FACE)
        and prior_prepared.get("complete") is True
        and n148_readiness.get("status") == "ready_for_diagnostic"
        and n148_readiness.get("complete") is True
        and n148_readiness.get("independent_full_bundle_restores") == 12
        and n148_readiness.get("source_hashes_unchanged") is True
        and n148_result.get("status") == "complete"
        and n148_result.get("complete") is True
        and n148_result.get("diagnostic_queries") == 960
        and len(n148_result.get("diagnostic_rows", [])) == 960
        and n148_result.get("source_hashes_unchanged") is True
        and seal_ast.receipt_binding_matches(n148_result.get("attempt_sha256"), N148_ATTEMPT)
        and seal_ast.receipt_binding_matches(n148_result.get("progress_sha256"), N148_PROGRESS)
        and n148_result.get("parameter_updates") == 0
        and n148_result.get("can_promote") is False
        and n148_binding.get("status") == "complete"
        and seal_ast.receipt_binding_matches(n148_binding.get("readiness_sha256"), N148_READINESS),
        "N1-148 face/readiness/result is incomplete or outside its frozen boundary",
    )
    if not n148_builder.n147_builder.n144_builder._verify_hashes(
        old_manifest.get("source_hashes", {})
    ):
        raise Refuse("N1-148 source exclusion chain changed")
    if not n148_builder.n147_builder.n144_builder._verify_hashes(
        prior_prepared.get("source_hashes", {})
    ):
        raise Refuse("N1-148 prepared face source hashes changed")
    if not n148_builder.n147_builder.n144_builder._verify_hashes(
        n148_result.get("source_hashes", {})
    ):
        raise Refuse("N1-148 result/checkpoint source hashes changed")

    base_entities = set(old_manifest["excluded_entities"])
    base_words = set(old_manifest["excluded_word_values"])
    v25_entities = {name for item in prior_face["episodes"] for name in item["entities"]}
    v25_materials = _material_values(prior_face)
    require(
        len(v25_entities) == EXPECTED_V25_ENTITY_COUNT
        and len(v25_materials) == EXPECTED_V25_WORD_COUNT
        and not (v25_entities & base_entities)
        and not (v25_materials & base_words),
        "consumed v25 face inventory overlaps or count differs",
    )
    entities = base_entities | v25_entities
    words = base_words | v25_materials
    require(
        len(entities) == EXPECTED_ENTITY_EXCLUSION_COUNT
        and len(words) == EXPECTED_WORD_EXCLUSION_COUNT,
        "N1-149 exclusion union count differs",
    )
    source_hashes = dict(n148_result["source_hashes"])
    paths = (
        PLAN,
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("eval_taiji_n1_v26_abstention_length_after_n148.py"),
        ROOT / "tests/taiji_native/test_n1_plan149_v26_abstention_length_contract.py",
        N148_SOURCE_MANIFEST,
        N148_FACE,
        N148_FACE_CHECKS,
        N148_PREPARED,
        N148_READINESS,
        n148_eval.READINESS_ATTEMPT,
        N148_SOURCE_BINDING,
        N148_ATTEMPT,
        N148_PROGRESS,
        N148_RESULT,
        ROOT / "plans/reference/PLAN-N1-148_execution_closeout_20261019.md",
        ROOT
        / "plans/reference/PLAN-N1-149_matched_abstention_context_length_curve_diagnostic_draft_20261019.md",
    )
    for path in paths:
        path = Path(path).resolve()
        require(path.is_file(), f"N1-148/N1-149 source missing: {path}")
        source_hashes[str(path)] = sha256(path)
    summary = {
        "n1_148_entity_exclusion_count": len(base_entities),
        "n1_148_word_exclusion_count": len(base_words),
        "consumed_v25_entity_count": len(v25_entities),
        "consumed_v25_material_count": len(v25_materials),
        "n1_148_scored_rows": len(n148_result["diagnostic_rows"]),
        "n1_148_queries": n148_result["diagnostic_queries"],
        "entity_overlap_with_previous_union": 0,
        "word_overlap_with_previous_union": 0,
        "entity_exclusion_count": len(entities),
        "entity_exclusion_sha256": digest_strings(entities),
        "word_exclusion_count": len(words),
        "word_exclusion_sha256": digest_strings(words),
        "source_hash_count": len(source_hashes),
    }
    return entities, words, source_hashes, summary


def build_face(excluded_entities: set[str], excluded_words: set[str], seed: int = SEED) -> dict:
    rng = random.Random(seed)
    used_entities = set(excluded_entities)
    used_words = set(excluded_words) | {ABSTENTION}
    episodes = []
    pair_positions = Counter()
    answer_lengths = Counter()
    for logical_pair in range(LOGICAL_PAIRS):
        pair_id = f"v26-pair-{logical_pair:03d}"
        target = n148_builder.n147_builder._novel(ENTITY_LENGTH, used_entities, rng)
        answer_length = 6 if logical_pair % 2 == 0 else 7
        answer = n148_builder.n147_builder._novel(answer_length, used_words, rng)
        answer_lengths[answer_length] += 1
        neighbors = [
            n148_builder.n147_builder._confusable(target, used_entities, rng) for _ in range(32)
        ]
        neighbor_values = [
            n148_builder.n147_builder._novel(
                6 if (logical_pair + index) % 2 == 0 else 7, used_words, rng
            )
            for index in range(32)
        ]
        common_rows = [
            f"{name} uses {value}."
            for name, value in zip(neighbors[:31], neighbor_values[:31], strict=True)
        ]
        unknown_replacement = f"{neighbors[31]} uses {neighbor_values[31]}."
        question = f"material of {target}?"
        for length in CONTEXT_LENGTHS:
            pair_positions[(length, 0)] += 1
            pair_group = f"{pair_id}-len{length}"
            known_rows = [f"{target} uses {answer}.", *common_rows[: length - 1]]
            unknown_rows = [unknown_replacement, *common_rows[: length - 1]]
            episodes.extend(
                [
                    {
                        "id": f"{pair_group}-known",
                        "logical_pair_id": pair_id,
                        "pair_id": pair_group,
                        "pair_condition": "known",
                        "split": "diagnostic",
                        "subset": "diagnostic",
                        "kind": "lookup",
                        "context_length": length,
                        "entities": [target, *neighbors[: length - 1]],
                        "observations": known_rows,
                        "question": question,
                        "answer": answer,
                        "target_fact_indices": [0],
                        "target_start_index": 0,
                        "paired_answer_length": answer_length,
                        "answer_length": answer_length,
                        "answer_direction": None,
                        "turn_template": None,
                    },
                    {
                        "id": f"{pair_group}-unknown",
                        "logical_pair_id": pair_id,
                        "pair_id": pair_group,
                        "pair_condition": "unknown",
                        "split": "diagnostic",
                        "subset": "diagnostic",
                        "kind": "unknown",
                        "context_length": length,
                        "entities": [target, neighbors[31], *neighbors[: length - 1]],
                        "observations": unknown_rows,
                        "question": question,
                        "answer": ABSTENTION,
                        "target_fact_indices": [],
                        "target_start_index": 0,
                        "paired_answer_length": answer_length,
                        "answer_length": None,
                        "answer_direction": None,
                        "turn_template": None,
                    },
                ]
            )
    return {
        "schema": "n1-v26-matched-abstention-length-diagnostic-face-v1",
        "seed": seed,
        "plan_sha256": PLAN_SHA256,
        "logical_pairs": LOGICAL_PAIRS,
        "context_lengths": list(CONTEXT_LENGTHS),
        "items": len(episodes),
        "entity_length": ENTITY_LENGTH,
        "name_neighbor_distance": DISTANCE,
        "target_start_index": 0,
        "answer_lengths": [6, 7],
        "episodes": episodes,
        "shape_counts": {
            "items_by_length_and_condition": {
                f"{length}/{condition}": sum(
                    item["context_length"] == length and item["pair_condition"] == condition
                    for item in episodes
                )
                for length in CONTEXT_LENGTHS
                for condition in CONDITIONS
            },
            "logical_pairs_by_length": {str(length): LOGICAL_PAIRS for length in CONTEXT_LENGTHS},
            "known_answer_lengths": {
                str(length): count for length, count in sorted(answer_lengths.items())
            },
            "new_entity_count": len(used_entities - excluded_entities),
            "new_word_count": len(used_words - excluded_words - {ABSTENTION}),
        },
    }


def validate_face(face: dict, excluded_entities: set[str], excluded_words: set[str]) -> dict:
    episodes = face.get("episodes", [])
    require(
        face.get("schema") == "n1-v26-matched-abstention-length-diagnostic-face-v1",
        "v26 face schema mismatch",
    )
    require(
        face.get("seed") == SEED and face.get("plan_sha256") == PLAN_SHA256,
        "v26 face seed/plan mismatch",
    )
    require(
        len(episodes) == ITEMS and len({item.get("id") for item in episodes}) == ITEMS,
        "v26 item IDs/count differ",
    )
    require(
        all(
            item.get("split") == "diagnostic" and item.get("subset") == "diagnostic"
            for item in episodes
        ),
        "v26 split changed",
    )
    logical: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for item in episodes:
        require(
            item.get("context_length") in CONTEXT_LENGTHS,
            f"unexpected context length: {item.get('id')}",
        )
        require(
            len(item.get("observations", [])) == item["context_length"],
            f"context row count differs: {item.get('id')}",
        )
        require(
            item.get("target_start_index") == 0,
            f"target index must remain fixed at 0: {item.get('id')}",
        )
        logical[item["logical_pair_id"]][int(item["context_length"])][item["pair_condition"]] = item
    require(len(logical) == LOGICAL_PAIRS, "v26 logical pair count differs")

    pair_positions = Counter()
    answer_lengths = Counter()
    global_entities: set[str] = set()
    global_materials: set[str] = set()
    for logical_id, by_length in logical.items():
        require(
            set(by_length) == set(CONTEXT_LENGTHS), f"logical pair missing a length: {logical_id}"
        )
        target_name = None
        common_question = None
        answer = None
        answer_length = None
        prior_prefix: dict[str, list[str]] = {}
        pair_entities: set[str] = set()
        pair_materials: set[str] = set()
        for length in CONTEXT_LENGTHS:
            conditions = by_length[length]
            require(
                set(conditions) == set(CONDITIONS),
                f"known/unknown condition missing: {logical_id}/{length}",
            )
            known = conditions["known"]
            unknown = conditions["unknown"]
            pair_positions[length] += 1
            current_target = known["entities"][0]
            target_name = current_target if target_name is None else target_name
            common_question = known["question"] if common_question is None else common_question
            answer = known["answer"] if answer is None else answer
            answer_length = len(known["answer"]) if answer_length is None else answer_length
            require(
                unknown["entities"][0] == target_name
                and known["question"] == unknown["question"] == common_question
                and known["answer"] == answer
                and len(known["answer"]) == answer_length
                and unknown["answer"] == ABSTENTION
                and known["target_fact_indices"] == [0]
                and unknown["target_fact_indices"] == []
                and known["observations"][0] == f"{target_name} uses {answer}."
                and not any(row.startswith(f"{target_name} ") for row in unknown["observations"]),
                f"known/unknown target semantics differ: {logical_id}/{length}",
            )
            require(
                all(
                    known["observations"][index] == unknown["observations"][index]
                    for index in range(1, length)
                ),
                f"paired same-length contexts differ outside target slot: {logical_id}/{length}",
            )
            for _condition, item in conditions.items():
                require(
                    item["logical_pair_id"] == logical_id
                    and item["pair_id"] == f"{logical_id}-len{length}",
                    f"pair ID mismatch: {item['id']}",
                )
                require(
                    len(set(item["entities"])) == len(item["entities"]),
                    f"duplicate entity in item: {item['id']}",
                )
                require(
                    all(
                        n148_builder.n147_builder._hamming(target_name, name) == DISTANCE
                        for name in item["entities"][1:]
                    ),
                    f"d=1 relation differs: {item['id']}",
                )
                pair_entities.update(item["entities"])
                pair_materials.update(
                    row.removesuffix(".").split()[-1]
                    for row in item["observations"]
                    if " uses " in row
                )
            if length == 8:
                prior_prefix = {
                    condition: list(item["observations"]) for condition, item in conditions.items()
                }
            else:
                for condition, item in conditions.items():
                    require(
                        item["observations"][: len(prior_prefix[condition])]
                        == prior_prefix[condition],
                        f"contexts are not nested at length {length}: {logical_id}/{condition}",
                    )
                    prior_prefix[condition] = list(item["observations"])
        require(answer_length in (6, 7), f"answer length differs: {logical_id}")
        answer_lengths[answer_length] += 1
        pair_materials.add(answer)
        require(
            not (pair_entities & excluded_entities),
            f"v26 entity overlaps prior faces: {logical_id}",
        )
        require(
            not (pair_materials & excluded_words),
            f"v26 material overlaps prior faces: {logical_id}",
        )
        require(
            not (pair_entities & global_entities),
            f"entity repeats across logical pairs: {logical_id}",
        )
        require(
            not (pair_materials & global_materials),
            f"material repeats across logical pairs: {logical_id}",
        )
        global_entities.update(pair_entities)
        global_materials.update(pair_materials)
    require(
        all(pair_positions[length] == LOGICAL_PAIRS for length in CONTEXT_LENGTHS),
        "logical-pair length balance differs",
    )
    require(all(answer_lengths[length] == 20 for length in (6, 7)), "answer length balance differs")
    require(
        len(global_entities) == 1_320 and len(global_materials) == 1_320,
        "v26 unique source counts differ",
    )
    return {
        "accepted": True,
        "logical_pairs": len(logical),
        "items": len(episodes),
        "items_by_length": {
            str(length): sum(item["context_length"] == length for item in episodes)
            for length in CONTEXT_LENGTHS
        },
        "logical_pairs_by_length": dict(sorted(pair_positions.items())),
        "known_answer_length_counts": dict(sorted(answer_lengths.items())),
        "unique_entities": len(global_entities),
        "unique_materials": len(global_materials),
    }


def prepare_face() -> dict:
    require(PLAN.is_file() and sha256(PLAN) == PLAN_SHA256, "PLAN-N1-149 frozen SHA-256 mismatch")
    outputs = (SOURCE_PRECHECK, SOURCE_EXCLUSIONS, FACE, FACE_ATTEMPT, FACE_CHECKS, PREPARED)
    if any(path.exists() for path in outputs):
        raise FileExistsError("PLAN-N1-149 face preparation is write-once")
    started = time.process_time()
    entities, words, source_hashes, union = _collect_source_union()
    source_hashes[str(PLAN.resolve())] = sha256(PLAN)
    source_hashes[str(Path(__file__).resolve())] = sha256(Path(__file__))
    require(
        n148_builder.n147_builder.n144_builder._verify_hashes(source_hashes),
        "N1-148/v25/N1-149 source changed before face generation",
    )
    face = build_face(entities, words)
    validate_face(face, entities, words)
    write_new(
        SOURCE_PRECHECK,
        {
            "schema": "n1-plan149-v26-source-union-precheck-v1",
            "status": "complete",
            "plan_sha256": PLAN_SHA256,
            "source_hashes": source_hashes,
            "source_hashes_unchanged": True,
            "source_union": union,
            "new_face_generated": False,
            "model_queries": 0,
            "parameter_updates": 0,
            "cpu_seconds": round(time.process_time() - started, 6),
            "can_promote": False,
        },
    )
    manifest = {
        "schema": "n1-plan149-v26-source-exclusions-v1",
        "status": "ready_for_one_time_face_generation",
        "plan_sha256": PLAN_SHA256,
        "seed": SEED,
        "entity_count": len(entities),
        "word_value_count": len(words),
        "entity_exclusion_sha256": digest_strings(entities),
        "word_exclusion_sha256": digest_strings(words),
        "excluded_entities": sorted(entities),
        "excluded_word_values": sorted(words),
        "provenance": union,
        "source_hashes": source_hashes,
        "can_promote": False,
    }
    write_new(SOURCE_EXCLUSIONS, manifest)
    write_new(
        FACE_ATTEMPT,
        {
            "schema": "n1-plan149-v26-face-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "source_union_sha256": sha256(SOURCE_EXCLUSIONS),
            "seed": SEED,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    face.update(
        {
            "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
            "source_exclusion_entity_sha256": manifest["entity_exclusion_sha256"],
            "source_exclusion_word_sha256": manifest["word_exclusion_sha256"],
            "source_hashes": source_hashes,
        }
    )
    write_new(FACE, face)
    checks = validate_face(face, entities, words)
    checks.update(
        {
            "plan_sha256": PLAN_SHA256,
            "face_sha256": sha256(FACE),
            "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
            "source_hashes_unchanged": n148_builder.n147_builder.n144_builder._verify_hashes(
                source_hashes
            ),
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        }
    )
    require(checks["source_hashes_unchanged"] is True, "source changed during v26 face generation")
    write_new(FACE_CHECKS, checks)
    prepared = {
        "schema": "n1-plan149-v26-face-readiness-v1",
        "status": "ready_for_checkpoint_readiness",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_precheck_sha256": sha256(SOURCE_PRECHECK),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": True,
        "model_queries": 0,
        "parameter_updates": 0,
        "cpu_seconds": round(time.process_time() - started, 6),
        "can_promote": False,
        "summary_zh": "v26 8/16/32行嵌套配对诊断面构造及合同通过；0模型/训练/查询，可进入readiness。",
    }
    write_new(PREPARED, prepared)
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-149 nested context-length diagnostic face builder"
    )
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
