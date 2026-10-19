"""构造PLAN-N1-148全新seed125 known/unknown配对诊断面。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v24_abstention_after_n146 as n147_builder
from scripts.training import eval_taiji_n1_v24_abstention_after_n146 as n147_eval

PLAN = ROOT / "plans/reference/PLAN-N1-148_matched_abstention_diagnostic_scorer_revision_20261019.md"
PLAN_SHA256 = "35214c44aec6eb33f1ea57bcada0eabd597d69fa29528228f0a158e21e1ba966"
SOURCE_PRECHECK = ROOT / "reports/taiji_n1_v25_abstention_source_union_precheck_20261019.json"
SOURCE_EXCLUSIONS = ROOT / "plans/manifests/n1_v25_abstention_after_n147_source_exclusions_20261019.json"
FACE = ROOT / "plans/manifests/n1_v25_abstention_seed125_20261019.json"
FACE_ATTEMPT = ROOT / "plans/manifests/n1_v25_abstention_seed125_20261019_attempt.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_v25_abstention_face_checks_20261019.json"
PREPARED = ROOT / "reports/taiji_n1_v25_abstention_face_readiness_20261019.json"

N147_SOURCE_MANIFEST = n147_builder.SOURCE_EXCLUSIONS
N147_FACE = n147_builder.FACE
N147_FACE_CHECKS = n147_builder.FACE_CHECKS
N147_PREPARED = n147_builder.PREPARED
N147_READINESS = n147_eval.READINESS
N147_READINESS_ATTEMPT = n147_eval.READINESS_ATTEMPT
N147_SOURCE_BINDING = n147_eval.SOURCE_BINDING
N147_ATTEMPT = n147_eval.ATTEMPT
N147_PROGRESS = n147_eval.PROGRESS
N147_RESULT = n147_eval.RESULT
N147_PLAN = n147_builder.PLAN
N147A_PLAN = ROOT / "plans/reference/PLAN-N1-147A_readiness_verifier_binding_correction_20261019.md"
N147A_FAILURE = ROOT / "reports/taiji_n1_v24_abstention_preflight_failure_20261019.json"
N147A_RUNNER = ROOT / "scripts/training/run_taiji_n1_v24_abstention_after_n146_repair.py"
N147A_TEST = ROOT / "tests/taiji_native/test_n1_plan147a_v24_readiness_runner_repair_contract.py"

SEED = 125
PAIRS = n147_builder.PAIRS
ITEMS = n147_builder.ITEMS
OBSERVATIONS = n147_builder.OBSERVATIONS
POSITIONS = n147_builder.POSITIONS
EXPECTED_BASE_ENTITY_COUNT = 36_802
EXPECTED_BASE_WORD_COUNT = 13_968
EXPECTED_V24_ENTITY_COUNT = 1_320
EXPECTED_V24_WORD_COUNT = 1_320
EXPECTED_ENTITY_EXCLUSION_COUNT = 38_122
EXPECTED_WORD_EXCLUSION_COUNT = 15_288


class Refuse(RuntimeError):
    """来源或匹配面合同失败；不重抽。"""


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


def _collect_source_union() -> tuple[set[str], set[str], dict[str, str], dict]:
    old_manifest = read_json(N147_SOURCE_MANIFEST)
    prior_face = read_json(N147_FACE)
    prior_checks = read_json(N147_FACE_CHECKS)
    prior_prepared = read_json(N147_PREPARED)
    readiness = read_json(N147_READINESS)
    binding = read_json(N147_SOURCE_BINDING)
    attempt = read_json(N147_ATTEMPT)
    result = read_json(N147_RESULT)
    failure_receipt = read_json(N147A_FAILURE)
    require(
        old_manifest.get("schema") == "n1-plan147-v24-source-exclusions-v1"
        and old_manifest.get("entity_count") == EXPECTED_BASE_ENTITY_COUNT
        and old_manifest.get("word_value_count") == EXPECTED_BASE_WORD_COUNT
        and len(old_manifest.get("excluded_entities", [])) == EXPECTED_BASE_ENTITY_COUNT
        and len(old_manifest.get("excluded_word_values", [])) == EXPECTED_BASE_WORD_COUNT
        and prior_face.get("schema") == "n1-v24-matched-abstention-diagnostic-face-v1"
        and prior_face.get("seed") == 124
        and len(prior_face.get("episodes", [])) == 80
        and prior_checks.get("accepted") is True
        and prior_checks.get("face_sha256") == sha256(N147_FACE)
        and prior_prepared.get("complete") is True
        and readiness.get("status") == "ready_for_diagnostic"
        and readiness.get("complete") is True
        and readiness.get("independent_full_bundle_restores") == 12
        and readiness.get("source_hashes_unchanged") is True
        and readiness.get("model_queries") == 0
        and binding.get("status") == "complete"
        and binding.get("readiness_sha256") == sha256(N147_READINESS)
        and attempt.get("status") == "started"
        and result.get("status") == "stopped"
        and result.get("complete") is False
        and result.get("diagnostic_queries") == 1
        and result.get("diagnostic_rows_completed") == 0
        and result.get("diagnostic_rows") == []
        and result.get("source_hashes_unchanged") is True
        and result.get("parameter_updates") == 0
        and result.get("optimizer_steps") == 0
        and result.get("dev_queries") == 0
        and result.get("v3_holdout_queries") == 0
        and result.get("can_promote") is False
        and result.get("errors") == [
            "N1-132/seed11/parent diagnostic: ValueError: confirmation holdout split and frozen arm required"
        ]
        and len(result.get("endpoint_receipts", [])) == 1
        and result["endpoint_receipts"][0].get("completed_rows") == 0
        and result["endpoint_receipts"][0].get("checkpoint_unchanged") is True
        and failure_receipt.get("status") == "stopped_before_checkpoint_restore"
        and failure_receipt.get("prepared_source_hashes_unchanged") is True,
        "N1-147 failed attempt is incomplete or inconsistent with its frozen source boundary",
    )
    require(
        sha256(N147_ATTEMPT) == result.get("attempt_sha256")
        and sha256(N147_PROGRESS) == result.get("progress_sha256")
        and sha256(N147_FACE) == result.get("face_sha256")
        and n147_builder.n144_builder._verify_hashes(old_manifest.get("source_hashes", {}))
        and n147_builder.n144_builder._verify_hashes(prior_prepared.get("source_hashes", {}))
        and n147_eval.frozen_eval._verify_hashes(result.get("source_hashes", {})),
        "N1-147 attempt/progress/source/checkpoint hash chain changed",
    )
    progress_events = [json.loads(line) for line in N147_PROGRESS.read_text(encoding="utf-8").splitlines() if line]
    require(
        all(event.get("event") != "query_row" for event in progress_events)
        and len(progress_events) == 1
        and progress_events[0].get("completed_rows") == 0,
        "N1-147 progress contains unexpected scored rows",
    )

    base_entities = set(old_manifest["excluded_entities"])
    base_words = set(old_manifest["excluded_word_values"])
    v24_entities = {name for item in prior_face["episodes"] for name in item["entities"]}
    v24_materials = n147_builder._material_values(prior_face)
    require(
        len(v24_entities) == EXPECTED_V24_ENTITY_COUNT
        and len(v24_materials) == EXPECTED_V24_WORD_COUNT
        and not (v24_entities & base_entities)
        and not (v24_materials & base_words),
        "consumed v24 face inventory overlaps or count differs",
    )
    entities = base_entities | v24_entities
    words = base_words | v24_materials
    require(
        len(entities) == EXPECTED_ENTITY_EXCLUSION_COUNT
        and len(words) == EXPECTED_WORD_EXCLUSION_COUNT,
        "N1-148 source exclusion union count differs",
    )

    source_hashes = dict(result["source_hashes"])
    paths = (
        PLAN,
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("eval_taiji_n1_v25_abstention_after_n147.py"),
        ROOT / "tests/taiji_native/test_n1_plan148_v25_abstention_scorer_contract.py",
        N147_PLAN,
        N147_SOURCE_MANIFEST,
        N147_FACE,
        N147_FACE_CHECKS,
        N147_PREPARED,
        N147_READINESS,
        N147_READINESS_ATTEMPT,
        N147_SOURCE_BINDING,
        N147_ATTEMPT,
        N147_PROGRESS,
        N147_RESULT,
        N147A_PLAN,
        N147A_FAILURE,
        N147A_RUNNER,
        N147A_TEST,
        ROOT / "plans/reference/PLAN-N1-147_execution_closeout_20261019.md",
    )
    for path in paths:
        path = Path(path).resolve()
        require(path.is_file(), f"N1-147/N1-148 source missing: {path}")
        source_hashes[str(path)] = sha256(path)
    summary = {
        "n1_147_entity_exclusion_count": len(base_entities),
        "n1_147_word_exclusion_count": len(base_words),
        "consumed_v24_entity_count": len(v24_entities),
        "consumed_v24_material_count": len(v24_materials),
        "v24_scored_rows": len(result["diagnostic_rows"]),
        "v24_query_counter": result["diagnostic_queries"],
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
    template_face = n147_builder.build_face(excluded_entities, excluded_words, seed=seed)
    template_face["schema"] = "n1-v25-matched-abstention-diagnostic-face-v1"
    for item in template_face["episodes"]:
        item["id"] = item["id"].replace("v24-pair-", "v25-pair-")
        item["pair_id"] = item["pair_id"].replace("v24-pair-", "v25-pair-")
    template_face["plan_sha256"] = PLAN_SHA256
    return template_face


def validate_face(face: dict, excluded_entities: set[str], excluded_words: set[str]) -> dict:
    require(face.get("schema") == "n1-v25-matched-abstention-diagnostic-face-v1", "v25 schema mismatch")
    require(face.get("seed") == SEED and face.get("plan_sha256") == PLAN_SHA256, "v25 seed/plan mismatch")
    require(len(face.get("episodes", [])) == ITEMS, "v25 item count mismatch")
    template = copy.deepcopy(face)
    template["schema"] = "n1-v24-matched-abstention-diagnostic-face-v1"
    template["seed"] = n147_builder.SEED
    template.pop("plan_sha256", None)
    for item in template["episodes"]:
        item["id"] = item["id"].replace("v25-pair-", "v24-pair-")
        item["pair_id"] = item["pair_id"].replace("v25-pair-", "v24-pair-")
    checks = n147_builder.validate_face(template, excluded_entities, excluded_words)
    require(checks.get("accepted") is True, "v25 face differs from matched-pair template contract")
    require(
        all(item["id"].startswith("v25-pair-") and item["pair_id"].startswith("v25-pair-") for item in face["episodes"]),
        "v25 identifiers retain an old seed family prefix",
    )
    return {
        "accepted": True,
        "pairs": PAIRS,
        "items": ITEMS,
        "pairs_by_target_start": checks["pairs_by_target_start"],
        "known_answer_length_counts": checks["known_answer_length_counts"],
        "unique_entities": checks["unique_entities"],
        "unique_materials": checks["unique_materials"],
    }


def prepare_face() -> dict:
    require(PLAN.is_file() and sha256(PLAN) == PLAN_SHA256, "PLAN-N1-148 frozen SHA-256 mismatch")
    outputs = (SOURCE_PRECHECK, SOURCE_EXCLUSIONS, FACE, FACE_ATTEMPT, FACE_CHECKS, PREPARED)
    if any(path.exists() for path in outputs):
        raise FileExistsError("PLAN-N1-148 face preparation is write-once")
    started = time.process_time()
    entities, words, source_hashes, source_union = _collect_source_union()
    source_hashes[str(PLAN.resolve())] = sha256(PLAN)
    source_hashes[str(Path(__file__).resolve())] = sha256(Path(__file__))
    require(n147_builder.n144_builder._verify_hashes(source_hashes), "N1-147/v24/N1-148 source changed before face generation")
    face = build_face(entities, words)
    validate_face(face, entities, words)
    write_new(
        SOURCE_PRECHECK,
        {
            "schema": "n1-plan148-v25-source-union-precheck-v1",
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
        },
    )
    manifest = {
        "schema": "n1-plan148-v25-source-exclusions-v1",
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
            "schema": "n1-plan148-v25-face-attempt-v1",
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
            "source_hashes_unchanged": n147_builder.n144_builder._verify_hashes(source_hashes),
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        }
    )
    require(checks["source_hashes_unchanged"] is True, "source changed during v25 face generation")
    write_new(FACE_CHECKS, checks)
    prepared = {
        "schema": "n1-plan148-v25-face-readiness-v1",
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
        "summary_zh": "v25全新seed125配对题面构造及合同通过；沿用known/unknown匹配结构，0模型/训练/查询。",
    }
    write_new(PREPARED, prepared)
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-148 fresh v25 matched abstention face builder")
    parser.add_argument("--prepare-face", action="store_true")
    args = parser.parse_args()
    if not args.prepare_face:
        parser.error("one-shot invocation requires --prepare-face")
    result = prepare_face()
    print(json.dumps({key: result.get(key) for key in ("status", "complete", "cpu_seconds", "model_queries", "parameter_updates", "can_promote", "summary_zh")}, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
