"""构造PLAN-N1-147的全新匹配known/unknown长上下文诊断面。"""

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

from scripts.training import build_taiji_n1_v23_confirmation_after_n143 as n144_builder
from scripts.training import eval_taiji_n1_v23_confirmation_after_n143 as n144_eval

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-147_matched_known_unknown_abstention_diagnostic_draft_20261019.md"
)
PLAN_SHA256 = "4d6c9799741d3290f5f1e6cda4325268ce086027e4d14a72f82a7f3662e09be8"
SOURCE_PRECHECK = ROOT / "reports/taiji_n1_v24_abstention_source_union_precheck_20261019.json"
SOURCE_EXCLUSIONS = (
    ROOT / "plans/manifests/n1_v24_abstention_after_n146_source_exclusions_20261019.json"
)
FACE = ROOT / "plans/manifests/n1_v24_abstention_seed124_20261019.json"
FACE_ATTEMPT = ROOT / "plans/manifests/n1_v24_abstention_seed124_20261019_attempt.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_v24_abstention_face_checks_20261019.json"
PREPARED = ROOT / "reports/taiji_n1_v24_abstention_face_readiness_20261019.json"

N144_SOURCE_MANIFEST = n144_builder.SOURCE_EXCLUSIONS
N144_FACE = n144_builder.FACE
N144_FACE_CHECKS = n144_builder.FACE_CHECKS
N144_PREPARED = n144_builder.PREPARED
N144_RESULT = n144_eval.RESULT
N146_REPORT = ROOT / "reports/taiji_n1_v23_output_source_attribution_20261019.json"
N146_REPORT_SHA256 = "cfe3bf28269bf45abbb050535009ab9df70af340667c6389ff49704b9b978edb"

SEED = 124
PAIRS = 40
ITEMS = PAIRS * 2
OBSERVATIONS = 32
ENTITY_LENGTH = 10
DISTANCE = 1
KINDS = ("lookup", "unknown")
POSITIONS = (0, 8, 16, 24, 31)
CONDITIONS = ("known", "unknown")
LETTERS = "abcdefghijklmnopqrstuvwxyz"
ABSTENTION = "unknown"
EXPECTED_BASE_ENTITY_COUNT = 31_722
EXPECTED_BASE_WORD_COUNT = 10_128
EXPECTED_V23_ENTITY_COUNT = 5_080
EXPECTED_V23_WORD_COUNT = 3_840
EXPECTED_ENTITY_EXCLUSION_COUNT = 36_802
EXPECTED_WORD_EXCLUSION_COUNT = 13_968


class Refuse(RuntimeError):
    """冻结来源或匹配题面合同不满足，停止且不重抽。"""


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
    materials = {
        row.removesuffix(".").split()[-1]
        for item in face.get("episodes", [])
        for row in item.get("observations", [])
        if " uses " in row
    }
    materials.update(
        item["answer"] for item in face.get("episodes", []) if item.get("kind") == "lookup"
    )
    return materials


def _collect_source_union() -> tuple[set[str], set[str], dict[str, str], dict]:
    old_manifest = read_json(N144_SOURCE_MANIFEST)
    prior_face = read_json(N144_FACE)
    prior_checks = read_json(N144_FACE_CHECKS)
    prior_prepared = read_json(N144_PREPARED)
    n144_result = read_json(N144_RESULT)
    n146_report = read_json(N146_REPORT)
    require(
        old_manifest.get("schema") == "n1-plan144-v23-source-exclusions-v1"
        and old_manifest.get("entity_count") == EXPECTED_BASE_ENTITY_COUNT
        and old_manifest.get("word_value_count") == EXPECTED_BASE_WORD_COUNT
        and len(old_manifest.get("excluded_entities", [])) == EXPECTED_BASE_ENTITY_COUNT
        and len(old_manifest.get("excluded_word_values", [])) == EXPECTED_BASE_WORD_COUNT
        and prior_face.get("schema") == "n1-v23-confirmation-face-v1"
        and prior_face.get("seed") == 123
        and len(prior_face.get("episodes", [])) == 160
        and prior_checks.get("accepted") is True
        and prior_checks.get("face_sha256") == sha256(N144_FACE)
        and prior_prepared.get("complete") is True
        and n144_result.get("complete") is True
        and n144_result.get("status") == "confirmation_failed"
        and n144_result.get("total_v23_queries") == 1_920
        and n144_result.get("source_hashes_unchanged") is True
        and n144_result.get("can_promote") is False
        and n146_report.get("complete") is True
        and n146_report.get("n144_result_sha256") == sha256(N144_RESULT)
        and n146_report.get("face_sha256") == sha256(N144_FACE)
        and n146_report.get("can_promote") is False
        and sha256(N146_REPORT) == N146_REPORT_SHA256,
        "N1-144/v23/N1-146 source receipts differ from the frozen boundary",
    )
    if not n144_builder._verify_hashes(old_manifest.get("source_hashes", {})):
        raise Refuse("N1-144 source manifest hash mismatch")
    if not n144_builder._verify_hashes(prior_prepared.get("source_hashes", {})):
        raise Refuse("N1-144 prepared-face source hashes changed")
    if not n144_eval._verify_hashes(n144_result.get("source_hashes", {})):
        raise Refuse("N1-144 confirmation source/checkpoint hashes changed")

    base_entities = set(old_manifest["excluded_entities"])
    base_words = set(old_manifest["excluded_word_values"])
    v23_entities = {name for item in prior_face["episodes"] for name in item["entities"]}
    v23_materials = _material_values(prior_face)
    require(
        len(v23_entities) == EXPECTED_V23_ENTITY_COUNT
        and len(v23_materials) == EXPECTED_V23_WORD_COUNT
        and not (v23_entities & base_entities)
        and not (v23_materials & base_words),
        "consumed v23 face inventory overlaps or count differs",
    )
    entities = base_entities | v23_entities
    words = base_words | v23_materials
    require(
        len(entities) == EXPECTED_ENTITY_EXCLUSION_COUNT
        and len(words) == EXPECTED_WORD_EXCLUSION_COUNT,
        "PLAN-N1-147 source exclusion union count differs",
    )

    source_hashes = dict(n144_result["source_hashes"])
    paths = (
        PLAN,
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("eval_taiji_n1_v24_abstention_after_n146.py"),
        ROOT / "tests/taiji_native/test_n1_plan147_v24_abstention_contract.py",
        N144_SOURCE_MANIFEST,
        N144_FACE,
        N144_FACE_CHECKS,
        N144_PREPARED,
        n144_eval.READINESS,
        n144_eval.READINESS_ATTEMPT,
        n144_eval.SOURCE_BINDING,
        n144_eval.ATTEMPT,
        n144_eval.PROGRESS,
        n144_eval.RESULT,
        ROOT / "plans/reference/PLAN-N1-144_execution_closeout_20261019.md",
        ROOT / "plans/reference/PLAN-N1-145_v23_itemwise_error_attribution_draft_20261019.md",
        ROOT / "plans/reference/PLAN-N1-145_execution_closeout_20261019.md",
        ROOT / "reports/taiji_n1_v23_itemwise_error_attribution_20261019.json",
        ROOT / "plans/reference/PLAN-N1-146_v23_output_source_item_attribution_draft_20261019.md",
        ROOT / "plans/reference/PLAN-N1-146_execution_closeout_20261019.md",
        N146_REPORT,
    )
    for path in paths:
        path = Path(path).resolve()
        require(path.is_file(), f"N1-144/N1-146/N1-147 source missing: {path}")
        source_hashes[str(path)] = sha256(path)
    summary = {
        "n1_144_entity_exclusion_count": len(base_entities),
        "n1_144_word_exclusion_count": len(base_words),
        "consumed_v23_entity_count": len(v23_entities),
        "consumed_v23_material_count": len(v23_materials),
        "entity_overlap_with_previous_union": 0,
        "word_overlap_with_previous_union": 0,
        "entity_exclusion_count": len(entities),
        "entity_exclusion_sha256": digest_strings(entities),
        "word_exclusion_count": len(words),
        "word_exclusion_sha256": digest_strings(words),
        "source_hash_count": len(source_hashes),
    }
    return entities, words, source_hashes, summary


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
    rng = random.Random(seed)
    used_entities = set(excluded_entities)
    used_words = set(excluded_words) | {ABSTENTION}
    episodes = []
    position_counts = Counter()
    answer_length_counts = Counter()
    for pair in range(PAIRS):
        pair_id = f"v24-pair-{pair:03d}"
        target = _novel(ENTITY_LENGTH, used_entities, rng)
        answer_length = 6 if pair % 2 == 0 else 7
        answer = _novel(answer_length, used_words, rng)
        start = POSITIONS[pair % len(POSITIONS)]
        position_counts[start] += 1
        answer_length_counts[answer_length] += 1
        neighbors = [_confusable(target, used_entities, rng) for _ in range(OBSERVATIONS)]
        neighbor_values = [
            _novel(6 if (pair + index) % 2 == 0 else 7, used_words, rng)
            for index in range(OBSERVATIONS)
        ]
        common_rows = [
            f"{name} uses {value}."
            for name, value in zip(neighbors[:31], neighbor_values[:31], strict=True)
        ]
        replacement = f"{neighbors[31]} uses {neighbor_values[31]}."
        known_rows: list[str] = []
        unknown_rows: list[str] = []
        common_index = 0
        for position in range(OBSERVATIONS):
            if position == start:
                known_rows.append(f"{target} uses {answer}.")
                unknown_rows.append(replacement)
            else:
                row = common_rows[common_index]
                known_rows.append(row)
                unknown_rows.append(row)
                common_index += 1
        question = f"material of {target}?"
        episodes.extend(
            [
                {
                    "id": f"{pair_id}-known",
                    "pair_id": pair_id,
                    "pair_condition": "known",
                    "split": "diagnostic",
                    "subset": "diagnostic",
                    "kind": "lookup",
                    "entities": [target, *neighbors[:31]],
                    "observations": known_rows,
                    "question": question,
                    "answer": answer,
                    "target_fact_indices": [start],
                    "target_start_index": start,
                    "paired_answer_length": answer_length,
                    "answer_length": answer_length,
                    "answer_direction": None,
                    "turn_template": None,
                },
                {
                    "id": f"{pair_id}-unknown",
                    "pair_id": pair_id,
                    "pair_condition": "unknown",
                    "split": "diagnostic",
                    "subset": "diagnostic",
                    "kind": "unknown",
                    "entities": [target, *neighbors],
                    "observations": unknown_rows,
                    "question": question,
                    "answer": ABSTENTION,
                    "target_fact_indices": [],
                    "target_start_index": start,
                    "paired_answer_length": answer_length,
                    "answer_length": None,
                    "answer_direction": None,
                    "turn_template": None,
                },
            ]
        )
    return {
        "schema": "n1-v24-matched-abstention-diagnostic-face-v1",
        "seed": seed,
        "pairs": PAIRS,
        "items": ITEMS,
        "observations_per_item": OBSERVATIONS,
        "entity_length": ENTITY_LENGTH,
        "name_neighbor_distance": DISTANCE,
        "target_start_positions": list(POSITIONS),
        "answer_lengths": [6, 7],
        "episodes": episodes,
        "shape_counts": {
            "items_by_condition": {
                condition: sum(item["pair_condition"] == condition for item in episodes)
                for condition in CONDITIONS
            },
            "pairs_by_target_start": dict(sorted(position_counts.items())),
            "known_answer_length_counts": {
                str(length): count for length, count in sorted(answer_length_counts.items())
            },
            "new_entity_count": len(used_entities - excluded_entities),
            "new_word_count": len(used_words - excluded_words - {ABSTENTION}),
        },
    }


def _hamming(first: str, second: str) -> int:
    if len(first) != len(second):
        return max(len(first), len(second))
    return sum(left != right for left, right in zip(first, second, strict=True))


def validate_face(face: dict, excluded_entities: set[str], excluded_words: set[str]) -> dict:
    episodes = face.get("episodes", [])
    require(
        face.get("schema") == "n1-v24-matched-abstention-diagnostic-face-v1", "v24 schema mismatch"
    )
    require(face.get("seed") == SEED and len(episodes) == ITEMS, "v24 seed/item count mismatch")
    require(len({item.get("id") for item in episodes}) == ITEMS, "v24 item IDs are not unique")
    require(
        all(
            item.get("split") == "diagnostic" and item.get("subset") == "diagnostic"
            for item in episodes
        ),
        "v24 items must remain diagnostic split",
    )
    require(
        all(len(item.get("observations", [])) == OBSERVATIONS for item in episodes),
        "v24 context length differs",
    )
    by_pair: dict[str, dict[str, dict]] = {}
    for item in episodes:
        by_pair.setdefault(item["pair_id"], {})[item["pair_condition"]] = item
    require(len(by_pair) == PAIRS, "v24 pair count differs")
    pair_positions = Counter()
    pair_answer_lengths = Counter()
    global_pair_entities: set[str] = set()
    global_pair_materials: set[str] = set()
    for pair_id, conditions in by_pair.items():
        require(set(conditions) == set(CONDITIONS), f"paired conditions missing: {pair_id}")
        known = conditions["known"]
        unknown = conditions["unknown"]
        target = known["entities"][0]
        start = known["target_start_index"]
        answer_length = len(known["answer"])
        pair_positions[start] += 1
        pair_answer_lengths[answer_length] += 1
        require(
            unknown["entities"][0] == target
            and unknown["target_start_index"] == start
            and unknown["paired_answer_length"] == answer_length
            and known["paired_answer_length"] == answer_length
            and known["question"] == unknown["question"] == f"material of {target}?"
            and known["target_fact_indices"] == [start]
            and unknown["target_fact_indices"] == []
            and known["observations"][start] == f"{target} uses {known['answer']}."
            and unknown["answer"] == ABSTENTION,
            f"paired question/target metadata mismatch: {pair_id}",
        )
        require(
            answer_length in (6, 7) and unknown["answer_length"] is None,
            f"answer length mismatch: {pair_id}",
        )
        require(
            all(
                known["observations"][i] == unknown["observations"][i]
                for i in range(OBSERVATIONS)
                if i != start
            ),
            f"paired contexts differ outside target slot: {pair_id}",
        )
        replacement = unknown["observations"][start]
        require(
            replacement.startswith(f"{unknown['entities'][-1]} uses "),
            f"unknown replacement fact mismatch: {pair_id}",
        )
        require(
            not any(row.startswith(f"{target} ") for row in unknown["observations"]),
            f"unknown condition contains target fact: {pair_id}",
        )
        for item in (known, unknown):
            names = item["entities"]
            require(
                len(names) == len(set(names)), f"duplicate entity within paired item: {item['id']}"
            )
            require(
                all(len(name) == ENTITY_LENGTH for name in names),
                f"entity length differs: {item['id']}",
            )
            require(
                all(_hamming(target, name) == DISTANCE for name in names[1:]),
                f"d=1 name contract differs: {item['id']}",
            )
        pair_entities = set(known["entities"]) | set(unknown["entities"])
        pair_materials = {
            row.removesuffix(".").split()[-1]
            for item in (known, unknown)
            for row in item["observations"]
            if " uses " in row
        } | {known["answer"]}
        require(
            not (pair_entities & excluded_entities), f"entity overlaps consumed source: {pair_id}"
        )
        require(
            not (pair_materials & excluded_words), f"material overlaps consumed source: {pair_id}"
        )
        require(
            not (pair_entities & global_pair_entities),
            f"entity repeats across pair groups: {pair_id}",
        )
        require(
            not (pair_materials & global_pair_materials),
            f"material repeats across pair groups: {pair_id}",
        )
        global_pair_entities.update(pair_entities)
        global_pair_materials.update(pair_materials)
    require(
        all(pair_positions[position] == 8 for position in POSITIONS),
        "target-position pair balance differs",
    )
    require(
        all(pair_answer_lengths[length] == 20 for length in (6, 7)),
        "known answer-length pair balance differs",
    )
    require(len(global_pair_entities) == 1_320, "v24 source face entity count differs")
    require(len(global_pair_materials) == 1_320, "v24 source face material count differs")
    return {
        "accepted": True,
        "pairs": PAIRS,
        "items": len(episodes),
        "pairs_by_target_start": dict(sorted(pair_positions.items())),
        "known_answer_length_counts": dict(sorted(pair_answer_lengths.items())),
        "unique_entities": len(global_pair_entities),
        "unique_materials": len(global_pair_materials),
    }


def prepare_face() -> dict:
    require(PLAN.is_file() and sha256(PLAN) == PLAN_SHA256, "PLAN-N1-147 frozen SHA-256 mismatch")
    outputs = (SOURCE_PRECHECK, SOURCE_EXCLUSIONS, FACE, FACE_ATTEMPT, FACE_CHECKS, PREPARED)
    if any(path.exists() for path in outputs):
        raise FileExistsError("PLAN-N1-147 face preparation is write-once")
    started = time.process_time()
    entities, words, source_hashes, source_union = _collect_source_union()
    source_hashes[str(PLAN.resolve())] = sha256(PLAN)
    source_hashes[str(Path(__file__).resolve())] = sha256(Path(__file__))
    require(
        n144_builder._verify_hashes(source_hashes),
        "N1-144/v23/N1-146 source changed before face generation",
    )
    face = build_face(entities, words)
    validate_face(face, entities, words)
    precheck = {
        "schema": "n1-plan147-v24-source-union-precheck-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": True,
        "source_union": source_union,
        "new_face_generated": False,
        "model_queries": 0,
        "parameter_updates": 0,
        "cpu_seconds": round(time.process_time() - started, 6),
        "can_promote": False,
    }
    write_new(SOURCE_PRECHECK, precheck)
    manifest = {
        "schema": "n1-plan147-v24-source-exclusions-v1",
        "status": "ready_for_one_time_face_generation",
        "plan_sha256": PLAN_SHA256,
        "seed": SEED,
        "entity_count": len(entities),
        "word_value_count": len(words),
        "entity_exclusion_sha256": digest_strings(entities),
        "word_exclusion_sha256": digest_strings(words),
        "excluded_entities": sorted(entities),
        "excluded_word_values": sorted(words),
        "provenance": source_union,
        "source_hashes": source_hashes,
        "can_promote": False,
    }
    write_new(SOURCE_EXCLUSIONS, manifest)
    write_new(
        FACE_ATTEMPT,
        {
            "schema": "n1-plan147-v24-face-attempt-v1",
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
            "plan_sha256": PLAN_SHA256,
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
            "source_hashes_unchanged": n144_builder._verify_hashes(source_hashes),
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        }
    )
    require(checks["source_hashes_unchanged"] is True, "source changed during v24 face generation")
    write_new(FACE_CHECKS, checks)
    prepared = {
        "schema": "n1-plan147-v24-face-readiness-v1",
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
        "summary_zh": "v24 40组成对诊断面构造及差一行配对合同通过；0模型/训练/查询，可进入12端点checkpoint readiness。",
    }
    write_new(PREPARED, prepared)
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-147 matched known/unknown diagnostic face builder"
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
