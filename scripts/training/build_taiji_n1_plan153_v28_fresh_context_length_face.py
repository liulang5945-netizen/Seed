"""构造PLAN-N1-153 fresh内容上下文长度诊断面与来源排除审计。"""

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

from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_v27_length_robust_training_after_n149 as n150

PLAN = ROOT / "plans/reference/PLAN-N1-153_fresh_context_length_paired_diagnostic_20261020.md"
PLAN_SHA256 = "7f38c25832e5245f55e272d974ae00a831542d9b5f8cd9a5410901bc7d87e88b"
N150_RESULT = n150.RESULT
N150_RESULT_SHA256 = "e50e4b5aaf244ced4b5ef6232eb08a09206a5957f7d8f85abb1b3d841860ef0b"
N151A_RESULT = ROOT / "reports/taiji_n1_plan151a_train_score_ledger_recovery_result_20261020.json"
N151A_RESULT_SHA256 = "7fb5ffa74b44ffa918d42a508c5d527991445367b3ec22457bb0d21a718ab6d6"
N152_RESULT = ROOT / "reports/taiji_n1_plan152_length_itemwise_attribution_20261020.json"
N152_RESULT_SHA256 = "97277a783b6b3e6d62a56d87feeaae9caa2337d411d80bb53f537956981aa0bd"
N151A_PLAN = ROOT / "plans/reference/PLAN-N1-151A_source_binding_return_correction_20261020.md"
N151_FAILURE = ROOT / "reports/taiji_n1_plan151_preflight_binding_failure_20261020.json"
SOURCE_AUDIT = ROOT / "reports/taiji_n1_plan153_v28_fresh_context_source_audit_20261020.json"
FACE = ROOT / "plans/manifests/n1_plan153_v28_context_length_seed127_20261020.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_plan153_v28_fresh_context_face_checks_20261020.json"

SEED = 127
ITEMS_PER_KIND = 40
KINDS = ("lookup", "revision", "transition", "unknown")
LENGTHS = (8, 16, 32)
ENTITY_LENGTH = 10
MATERIAL_LENGTHS = (6, 7)
DISTANCE = 1
DIRECTIONS = ("north", "east", "south", "west")
TURNS = ("left", "right")
UNKNOWN = "unknown"
ENDPOINTS = tuple((seed, branch) for seed in n150_builder.SEEDS for branch in n150_builder.BRANCHES)
EXPECTED_QUERIES = len(ENDPOINTS) * ITEMS_PER_KIND * len(KINDS) * len(LENGTHS)


class Refuse(RuntimeError):
    """fresh face/source exclusion合同不成立。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    for raw_path, digest in incoming.items():
        path = str(Path(raw_path).resolve())
        old = target.get(path)
        if old is not None and old != digest:
            raise Refuse(f"conflicting PLAN-N1-153 source hash: {path}")
        target[path] = digest


def _hamming(first: str, second: str) -> int:
    if len(first) != len(second):
        raise Refuse("Hamming distance requires equal-length entity names")
    return sum(a != b for a, b in zip(first, second, strict=True))


def _add_row_values(row: str, entities: set[str], words: set[str]) -> None:
    parts = row.removesuffix(".").split()
    if not parts:
        raise Refuse("empty observation row in source/face")
    entities.add(parts[0])
    if len(parts) == 3 and parts[1] == "uses":
        words.add(parts[2])
    elif len(parts) == 4 and parts[1:3] == ["now", "uses"]:
        words.add(parts[3])
    elif len(parts) == 3 and parts[1] in ("faces", "turns"):
        words.add(parts[2])
    else:
        raise Refuse(f"unsupported source observation row: {row}")


def collect_n150_exclusions() -> tuple[set[str], set[str], dict, dict, dict]:
    n150_readiness, n150_preflight, _binding, bound, schedule = n150._verify_preflight()
    result = read_json(N150_RESULT)
    n151a = read_json(N151A_RESULT)
    n152 = read_json(N152_RESULT)
    n149_manifest = read_json(n150_builder.N149_MANIFEST)
    require(sha256(N150_RESULT) == N150_RESULT_SHA256, "N1-150 result SHA changed")
    require(sha256(N151A_RESULT) == N151A_RESULT_SHA256, "N1-151A result SHA changed")
    require(sha256(N152_RESULT) == N152_RESULT_SHA256, "N1-152 result SHA changed")
    require(
        result.get("complete") is True
        and result.get("training_updates") == 55_296
        and result.get("train_score_queries") == 7_488
        and result.get("source_hashes_unchanged") is True
        and result.get("can_promote") is False
        and n151a.get("complete") is True
        and n151a.get("repeated_train_only_queries") == 7_488
        and n151a.get("source_hashes_unchanged") is True
        and n152.get("complete") is True
        and n152.get("source_rows") == 1_728
        and n152.get("model_queries") == 0
        and n152.get("can_promote") is False,
        "N1-150/151A/152 evidence is incomplete or crosses its frozen boundary",
    )
    excluded_entities = set(n149_manifest["excluded_entities"])
    excluded_words = set(n149_manifest["excluded_word_values"]) | {UNKNOWN} | set(DIRECTIONS) | set(TURNS)
    source_entities = set(excluded_entities)
    source_words = set(excluded_words)
    for item in schedule["source_items"].values():
        names, words = n150_builder._extract_observation_values(item)
        source_entities.update(names)
        source_words.update(words)
    all_context_entities = set(source_entities)
    all_context_words = set(source_words)
    for views in schedule["views"].values():
        for row in views[32]:
            _add_row_values(row, all_context_entities, all_context_words)
    generated_entities = all_context_entities - source_entities
    generated_words = all_context_words - source_words
    expected_bank = schedule["manifest"]["context_bank"]
    require(
        len(generated_entities) == expected_bank["generated_distractor_entity_count"] == 94_848
        and len(generated_words) == expected_bank["generated_distractor_word_count"] == 94_848,
        "deterministic N1-150 generated training exclusions do not match the frozen source report",
    )
    exclusions = (all_context_entities, all_context_words)
    source_hashes = {}
    _merge_hashes(source_hashes, result.get("source_hashes", {}))
    _merge_hashes(source_hashes, n151a.get("source_hashes", {}))
    _merge_hashes(source_hashes, n152.get("source_hashes", {}))
    _merge_hashes(source_hashes, result.get("new_checkpoint_hashes", {}))
    paths = (
        PLAN,
        N150_RESULT,
        N151A_RESULT,
        N152_RESULT,
        N151A_PLAN,
        N151_FAILURE,
        n150_builder.N149_MANIFEST,
        n150_builder.PLAN,
        n150_builder.SCORING_ADDENDUM,
        n150_builder.RESOURCE_ADDENDUM,
        n150_builder.SOURCE_EXCLUSIONS,
        n150_builder.SCHEDULE,
        n150.PREFLIGHT,
        n150.PREFLIGHT_BINDING,
        n150.PROGRESS,
        Path(__file__).resolve(),
        ROOT / "scripts/training/run_taiji_n1_plan153_v28_fresh_context_length_confirmation.py",
        ROOT / "tests/taiji_native/test_n1_plan153_v28_fresh_context_length_confirmation_contract.py",
    )
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file():
            raise Refuse(f"frozen N1-150 source missing: {path}")
        _merge_hashes(source_hashes, {str(path): sha256(path)})
    if not n150_builder.verify_hashes(source_hashes):
        raise Refuse("N1-150 and N1-151A/152 source/checkpoint hashes changed")
    summary = {
        "excluded_entity_count": len(all_context_entities),
        "excluded_word_value_count": len(all_context_words),
        "n149_prior_entity_count": len(excluded_entities),
        "n149_prior_word_value_count": len(excluded_words),
        "n150_source_item_count": len(schedule["source_items"]),
        "n150_generated_entity_count": len(generated_entities),
        "n150_generated_word_count": len(generated_words),
        "entity_exclusion_sha256": digest_strings(all_context_entities),
        "word_exclusion_sha256": digest_strings(all_context_words),
    }
    return exclusions[0], exclusions[1], summary, source_hashes, {"bound": bound, "schedule": schedule}


def build_face(excluded_entities: set[str], excluded_words: set[str], seed: int = SEED) -> tuple[dict, dict]:
    helper = n150_builder.n149_builder.n148_builder.n147_builder
    used_entities = set(excluded_entities)
    used_words = set(excluded_words) | {UNKNOWN} | set(DIRECTIONS) | set(TURNS)
    rng = random.Random(seed)
    face_items = []
    answer_lengths = Counter()
    per_kind = Counter()
    for kind in KINDS:
        for index in range(ITEMS_PER_KIND):
            item_id = f"v28-{kind}-{index:03d}"
            target = helper._novel(ENTITY_LENGTH, used_entities, rng)
            material_length = MATERIAL_LENGTHS[index % 2]
            rows = []
            if kind == "lookup":
                answer = helper._novel(material_length, used_words, rng)
                answer_lengths[(kind, material_length)] += 1
                question = f"material of {target}?"
                neighbor_count = 31
                target_indices = [0]
            elif kind == "revision":
                old_length = MATERIAL_LENGTHS[(index + 1) % 2]
                old_value = helper._novel(old_length, used_words, rng)
                answer = helper._novel(material_length, used_words, rng)
                answer_lengths[(kind, material_length)] += 1
                question = f"material of {target}?"
                neighbor_count = 30
                target_indices = [0, 1]
                rows.extend((f"{target} uses {old_value}.", f"{target} now uses {answer}."))
            elif kind == "transition":
                initial = DIRECTIONS[index % len(DIRECTIONS)]
                turn = TURNS[(index // len(DIRECTIONS)) % len(TURNS)]
                offset = 1 if turn == "right" else -1
                answer = DIRECTIONS[(DIRECTIONS.index(initial) + offset) % len(DIRECTIONS)]
                question = f"direction of {target}?"
                neighbor_count = 30
                target_indices = [0, 1]
                rows.extend((f"{target} faces {initial}.", f"{target} turns {turn}."))
            else:
                answer = UNKNOWN
                question = f"material of {target}?"
                neighbor_count = 32
                target_indices = []

            neighbors = [helper._confusable(target, used_entities, rng) for _ in range(neighbor_count)]
            values = []
            for filler_index in range(neighbor_count):
                length = MATERIAL_LENGTHS[(filler_index + index) % 2]
                values.append(helper._novel(length, used_words, rng))

            if kind == "lookup":
                base_rows = [f"{target} uses {answer}.", f"{neighbors[0]} uses {values[0]}."]
                filler_rows = [f"{name} uses {value}." for name, value in zip(neighbors[1:], values[1:], strict=True)]
            elif kind in ("revision", "transition"):
                base_rows = rows
                filler_rows = [f"{name} uses {value}." for name, value in zip(neighbors, values, strict=True)]
            else:
                base_rows = [f"{neighbors[0]} uses {values[0]}.", f"{neighbors[1]} uses {values[1]}."]
                filler_rows = [f"{name} uses {value}." for name, value in zip(neighbors[2:], values[2:], strict=True)]

            if len(base_rows) != 2 or len(filler_rows) != 30:
                raise Refuse(f"fresh context base/filler length differs: {item_id}")
            full_rows = [*base_rows, *filler_rows]
            views = {
                str(length): full_rows[:length]
                for length in LENGTHS
            }
            entities = [target, *neighbors]
            item = {
                "id": item_id,
                "parent_id": item_id,
                "split": "diagnostic",
                "kind": kind,
                "entities": entities,
                "question": question,
                "answer": answer,
                "answer_length": len(answer.encode("utf-8")),
                "target_observation_indices": target_indices,
                "observations_by_length": views,
            }
            if kind != "unknown":
                answer_lengths[(kind, len(answer))] += 0
            face_items.append(item)
            per_kind[kind] += 1

    manifest = {
        "schema": "n1-plan153-v28-fresh-context-length-face-v1",
        "status": "prepared_once",
        "plan_sha256": PLAN_SHA256,
        "seed": seed,
        "items_per_kind": ITEMS_PER_KIND,
        "base_item_count": len(face_items),
        "lengths": list(LENGTHS),
        "queries_per_endpoint": len(face_items) * len(LENGTHS),
        "endpoint_count": len(ENDPOINTS),
        "expected_queries": EXPECTED_QUERIES,
        "items_by_kind": dict(sorted(per_kind.items())),
        "items": face_items,
        "can_promote": False,
    }
    checks = validate_face(manifest, excluded_entities, excluded_words)
    return manifest, {"checks": checks, "answer_length_counts": {f"{kind}/{length}": count for (kind, length), count in sorted(answer_lengths.items())}}


def validate_face(manifest: dict, excluded_entities: set[str], excluded_words: set[str]) -> dict:
    items = manifest.get("items", [])
    require(manifest.get("schema") == "n1-plan153-v28-fresh-context-length-face-v1", "N1-153 face schema mismatch")
    require(len(items) == ITEMS_PER_KIND * len(KINDS) == 160, "N1-153 requires 160 base items")
    require(manifest.get("expected_queries") == EXPECTED_QUERIES == 2_880, "N1-153 query denominator differs")
    kinds = Counter(item.get("kind") for item in items)
    require(kinds == Counter({kind: ITEMS_PER_KIND for kind in KINDS}), "N1-153 base kind balance differs")
    target_names = set()
    face_entities = set()
    face_entity_occurrences = []
    face_words = set()
    face_word_occurrences = []
    hamming_pairs = 0
    for item in items:
        if item.get("split") != "diagnostic" or item.get("id") in target_names:
            raise Refuse(f"invalid or duplicate diagnostic item: {item.get('id')}")
        target = n132._query_subject(item["question"])
        target_names.add(item["id"])
        rows8, rows16, rows32 = (item["observations_by_length"][str(length)] for length in LENGTHS)
        require(len(rows8) == 8 and len(rows16) == 16 and len(rows32) == 32, f"context length mismatch: {item['id']}")
        require(rows8 == rows16[:8] == rows32[:8], f"length contexts are not nested: {item['id']}")
        query = {**item, "observations": rows32}
        target_indices = n132._target_observation_indices(query)
        expected_indices = item["target_observation_indices"]
        require(target_indices == (expected_indices if expected_indices else None), f"target-row positions differ: {item['id']}")
        if target_indices is not None:
            require(rows32[: len(target_indices)] == [rows32[index] for index in target_indices], f"target facts are not first: {item['id']}")
        for name in item["entities"]:
            face_entity_occurrences.append(name)
        for row in rows32:
            name = row.removesuffix(".").split()[0]
            face_entities.add(name)
            if name != target:
                require(_hamming(target, name) == DISTANCE, f"fresh distractor is not d=1: {item['id']}/{name}")
                hamming_pairs += 1
            parts = row.removesuffix(".").split()
            if len(parts) == 3 and parts[1] == "uses":
                face_words.add(parts[2])
                face_word_occurrences.append(parts[2])
            elif len(parts) == 4 and parts[1:3] == ["now", "uses"]:
                face_words.add(parts[3])
                face_word_occurrences.append(parts[3])
        if item["kind"] in ("lookup", "revision"):
            require(item["answer"] not in excluded_words, f"answer overlaps an excluded word: {item['id']}")
    generated_targets = {n132._query_subject(item["question"]) for item in items}
    require(len(generated_targets) == len(items) and not (generated_targets & excluded_entities), "fresh target entities overlap source exclusions")
    require(not (face_entities & excluded_entities), "fresh face entities overlap source exclusions")
    require(not (face_words & excluded_words), "fresh face words overlap source exclusions")
    require(len(face_entity_occurrences) == len(set(face_entity_occurrences)), "face entity values are not globally unique")
    require(len(face_word_occurrences) == len(set(face_word_occurrences)), "face material values are not globally unique")
    return {
        "accepted": True,
        "base_item_count": len(items),
        "items_by_kind": dict(sorted(kinds.items())),
        "query_count_per_endpoint": len(items) * len(LENGTHS),
        "total_queries": len(items) * len(LENGTHS) * len(ENDPOINTS),
        "face_entity_count": len(face_entities),
        "face_word_count": len(face_words),
        "d1_distractor_rows_checked": hamming_pairs,
        "nested_contexts": True,
        "target_first": True,
        "source_overlap": False,
        "can_promote": False,
    }


def _collect_hashes(result: dict, source_hashes: dict[str, str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    _merge_hashes(hashes, result.get("source_hashes", {}))
    _merge_hashes(hashes, source_hashes)
    _merge_hashes(hashes, result.get("new_checkpoint_hashes", {}))
    return dict(sorted(hashes.items()))


def prepare_face() -> dict:
    outputs = (SOURCE_AUDIT, FACE, FACE_CHECKS)
    if any(path.exists() for path in outputs):
        raise FileExistsError("N1-153 source/face outputs are write-once")
    if sha256(PLAN) != PLAN_SHA256:
        raise Refuse("N1-153 plan SHA changed")
    started = time.process_time()
    excluded_entities, excluded_words, exclusion_summary, source_hashes, runtime = collect_n150_exclusions()
    face, face_summary = build_face(excluded_entities, excluded_words)
    require(n150_builder.verify_hashes(source_hashes), "N1-153 sources changed during face generation")
    cpu = time.process_time() - started
    require(cpu <= 3_600.0, "N1-153 face/source preparation exceeds CPU ceiling")
    face_bytes = json.dumps(face, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    checks = {
        "schema": "n1-plan153-v28-fresh-context-face-checks-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "face_sha256": hashlib.sha256(face_bytes).hexdigest(),
        **face_summary["checks"],
        "answer_length_counts": face_summary["answer_length_counts"],
        "source_exclusion_summary": exclusion_summary,
        "source_hash_count": len(source_hashes),
        "source_hashes_unchanged": True,
        "cpu_seconds": round(cpu, 6),
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "can_promote": False,
    }
    audit = {
        "schema": "n1-plan153-v28-fresh-context-source-audit-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        **exclusion_summary,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": True,
        "n150_generated_entity_count": 94_848,
        "n150_generated_word_count": 94_848,
        "new_face_entity_count": face_summary["checks"]["face_entity_count"],
        "new_face_word_count": face_summary["checks"]["face_word_count"],
        "new_entity_overlap": 0,
        "new_word_overlap": 0,
        "face_sha256": checks["face_sha256"],
        "can_promote": False,
    }
    write_new(SOURCE_AUDIT, audit)
    write_new(FACE, face)
    checks["source_audit_sha256"] = sha256(SOURCE_AUDIT)
    checks["face_manifest_sha256"] = sha256(FACE)
    write_new(FACE_CHECKS, checks)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-153 fresh context-length face preparation")
    parser.add_argument("--prepare-face", action="store_true")
    args = parser.parse_args()
    if not args.prepare_face:
        parser.error("one-shot invocation requires --prepare-face")
    result = prepare_face()
    keys = ("status", "accepted", "base_item_count", "total_queries", "source_exclusion_summary", "source_hash_count", "cpu_seconds", "model_loads", "checkpoint_loads", "model_queries", "parameter_updates", "can_promote")
    output = {key: result.get(key) for key in keys if key in result}
    output["summary_zh"] = "N1-153 fresh face/source preparation complete; no model load/query/update."
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
