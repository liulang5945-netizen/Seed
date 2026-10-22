"""构造PLAN-N1-159 seed129 fresh复制面并审计完整来源排除。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (
    analyze_taiji_n1_plan158_plan157a2_fresh_face_itemwise_attribution as n158,
)
from scripts.training import build_taiji_n1_plan153_v28_fresh_context_length_face as n153_builder
from scripts.training import build_taiji_n1_plan157_seed128_candidate_vs_parent_face as n157_builder
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import run_taiji_n1_v27_length_robust_training_after_n149 as n150

PLAN = ROOT / "plans/reference/PLAN-N1-159_seed129_candidate_parent_fresh_replication_20261020.md"
PLAN_SHA256 = "a152a2ce63ae063a7863a11573ae964c01b1a1a3222552a15977bf9023834b5b"
PRIOR_CORRECTION_ADDENDUM = (
    ROOT / "plans/reference/PLAN-N1-159A_remaining_endpoint_continuation_20261020.md"
)
PRIOR_CORRECTION_SHA256 = "f1e32066c4976e9ed826475992be6540882ec2b4344027e73195030ce96144b1"
CORRECTION_ADDENDUM = (
    ROOT / "plans/reference/PLAN-N1-159A_per_endpoint_summary_correction_20261020.md"
)
CORRECTION_SHA256 = "e5e01e49355e860e450e5c1bba68ef86e5fb09a3edd7c5696b9e245465e5fab9"
N150_RESULT = n150.RESULT
N155A_RESULT = n157_builder.N155A_RESULT
N155A_RESULT_SHA256 = n157_builder.N155A_RESULT_SHA256
N155A_PROGRESS = n157_builder.N155A_PROGRESS
N155A_PROGRESS_SHA256 = n157_builder.N155A_PROGRESS_SHA256
N157A2_RESULT = n157_builder.RESULT
N157A2_RESULT_SHA256 = "c502ade59d20bcf05c7d84b898dcbc7b192187c42224c0e20ed342215a9cd1b5"
N157A2_PROGRESS = n157_builder.PROGRESS
N157A2_PROGRESS_SHA256 = "95992dc16b58ba32d5ea1de37837601b5eb1fef8d555e9bb24079d1193b2f441"
N158_REPORT = n158.REPORT
N158_REPORT_SHA256 = "6f0681bf2c819df9444ff7e89faaca6f536a3aac55181e68059f8b892d99591a"
FACE128 = n157_builder.FACE
PRIOR_SOURCE_AUDIT = ROOT / "reports/taiji_n1_plan159_candidate_parent_source_audit_20261020.json"
PRIOR_READINESS = ROOT / "reports/taiji_n1_plan159_candidate_parent_readiness_20261020.json"
PRIOR_READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan159_candidate_parent_readiness_attempt_20261020.json"
)
PRIOR_SOURCE_BINDING = (
    ROOT / "reports/taiji_n1_plan159_candidate_parent_source_binding_20261020.json"
)
PRIOR_PREFLIGHT = ROOT / "reports/taiji_n1_plan159_candidate_parent_preflight_20261020.json"
PRIOR_PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan159_candidate_parent_preflight_attempt_20261020.json"
)
PRIOR_PREFLIGHT_BINDING = (
    ROOT / "reports/taiji_n1_plan159_candidate_parent_preflight_binding_20261020.json"
)
PRIOR_FACE = ROOT / "plans/manifests/n1_plan159_v31_candidate_parent_seed129_20261020.json"
PRIOR_FACE_AUDIT = ROOT / "reports/taiji_n1_plan159_v31_candidate_parent_face_audit_20261020.json"
PRIOR_FACE_CHECKS = ROOT / "reports/taiji_n1_plan159_v31_candidate_parent_face_checks_20261020.json"
PRIOR_ATTEMPT = ROOT / "reports/taiji_n1_plan159_candidate_parent_attempt_20261020.json"
PRIOR_PROGRESS = ROOT / "reports/taiji_n1_plan159_candidate_parent_progress_20261020.jsonl"
PRIOR_RESULT = ROOT / "reports/taiji_n1_plan159_candidate_parent_result_20261020.json"
PRIOR_RESULT_SHA256 = "47ba3fae3f5921ea950f980edcfbb7b015cc477e7dc94265f1871f288a2a6e38"
PRIOR_ATTEMPT_SHA256 = "2aee1645c68db01f2b2455cfbd758183cfc40abf40b555d6812905f5efb3e9ce"
PRIOR_PROGRESS_SHA256 = "f9910360aa7f7a87f5328b6cdba5c896f53498321d72b54a0d0903c14f3cae00"
PRIOR_FACE_SHA256 = "b445e96f7231ba874efa5300f99fd26a92a90548e56d14ba9caf881d30b729d3"
PRIOR_LINKED_CPU_SECONDS = 9_956.421875
PRIOR_QUERIES = 480

SOURCE_AUDIT = ROOT / "reports/taiji_n1_plan159a_candidate_parent_source_audit_20261020.json"
READINESS = ROOT / "reports/taiji_n1_plan159a_candidate_parent_readiness_20261020.json"
READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan159a_candidate_parent_readiness_attempt_20261020.json"
)
SOURCE_BINDING = ROOT / "reports/taiji_n1_plan159a_candidate_parent_source_binding_20261020.json"
PREFLIGHT = ROOT / "reports/taiji_n1_plan159a_candidate_parent_preflight_20261020.json"
PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan159a_candidate_parent_preflight_attempt_20261020.json"
)
PREFLIGHT_BINDING = (
    ROOT / "reports/taiji_n1_plan159a_candidate_parent_preflight_binding_20261020.json"
)
FACE = PRIOR_FACE
FACE_AUDIT = PRIOR_FACE_AUDIT
FACE_CHECKS = PRIOR_FACE_CHECKS
ATTEMPT = ROOT / "reports/taiji_n1_plan159a_candidate_parent_attempt_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan159a_candidate_parent_progress_20261020.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan159a_candidate_parent_result_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan159_seed129_candidate_parent_diagnostic_contract.py"

SEEDS = (11, 12, 13)
BRANCHES = ("unknown8_rebalanced", "length_mixed_parent")
ENDPOINTS = tuple((seed, branch) for seed in SEEDS for branch in BRANCHES)
LENGTHS = (8, 16, 32)
KINDS = ("lookup", "revision", "transition", "unknown")
ITEMS_PER_KIND = 40
QUERIES_PER_ENDPOINT = ITEMS_PER_KIND * len(KINDS) * len(LENGTHS)
TOTAL_QUERIES = QUERIES_PER_ENDPOINT * len(ENDPOINTS)
CPU_LIMIT_SECONDS = 3_600.0
LINKED_CPU_LIMIT_SECONDS = 18_000.0
UNKNOWN = n153_builder.UNKNOWN
DIRECTIONS = n153_builder.DIRECTIONS


class Refuse(RuntimeError):
    """PLAN-N1-159来源、面或endpoint合同不成立。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def merge_hashes(target: dict[str, str], incoming: dict[str, str]) -> None:
    for raw_path, digest in incoming.items():
        path = str(Path(raw_path).resolve())
        old = target.get(path)
        if old is not None and old != digest:
            raise Refuse(f"conflicting PLAN-N1-159 source hash: {path}")
        target[path] = digest


def _face_values(face: dict) -> tuple[set[str], set[str]]:
    entities = set()
    words = set()
    for item in face.get("items", []):
        entities.update(str(name) for name in item.get("entities", []))
        answer = item.get("answer")
        if answer not in (None, UNKNOWN, *DIRECTIONS):
            words.add(str(answer))
        for row in item.get("observations_by_length", {}).get("32", []):
            n153_builder._add_row_values(row, entities, words)
    return entities, words


def source_context() -> tuple[list[dict], dict[str, str], dict]:
    if sha256(PLAN) != PLAN_SHA256:
        raise Refuse("PLAN-N1-159 plan SHA changed")
    n150_result = read_json(N150_RESULT)
    n155a_result = read_json(N155A_RESULT)
    n157a2_result = read_json(N157A2_RESULT)
    n158_report = read_json(N158_REPORT)
    prior_result = read_json(PRIOR_RESULT)
    prior_attempt = read_json(PRIOR_ATTEMPT)
    prior_readiness = read_json(PRIOR_READINESS)
    prior_preflight = read_json(PRIOR_PREFLIGHT)
    prior_face = read_json(PRIOR_FACE)
    prior_face_audit = read_json(PRIOR_FACE_AUDIT)
    prior_face_checks = read_json(PRIOR_FACE_CHECKS)
    require(sha256(N155A_RESULT) == N155A_RESULT_SHA256, "N1-155A result SHA changed")
    require(sha256(N155A_PROGRESS) == N155A_PROGRESS_SHA256, "N1-155A progress SHA changed")
    require(sha256(N157A2_RESULT) == N157A2_RESULT_SHA256, "N1-157A2 result SHA changed")
    require(sha256(N157A2_PROGRESS) == N157A2_PROGRESS_SHA256, "N1-157A2 progress SHA changed")
    require(sha256(N158_REPORT) == N158_REPORT_SHA256, "N1-158 attribution SHA changed")
    require(
        n150_result.get("complete") is True
        and n150_result.get("training_sufficient_all_endpoints") is True
        and n155a_result.get("complete") is True
        and n155a_result.get("source_hashes_unchanged") is True
        and n157a2_result.get("complete") is True
        and n157a2_result.get("fresh_queries") == TOTAL_QUERIES
        and n157a2_result.get("new_fresh_queries") == 2_400
        and n157a2_result.get("prior_consumed_queries") == 480
        and n157a2_result.get("face_sha256") == sha256(FACE128)
        and n157a2_result.get("source_hashes_unchanged") is True
        and n157a2_result.get("can_promote") is False
        and n158_report.get("complete") is True
        and n158_report.get("source_result_sha256") == N157A2_RESULT_SHA256
        and n158_report.get("face_sha256") == sha256(FACE128)
        and n158_report.get("can_promote") is False,
        "PLAN-N1-159 parent/candidate source execution evidence is incomplete",
    )
    prior_rows = prior_result.get("fresh_diagnostic_rows", [])
    prior_progress_rows = [json.loads(line) for line in PRIOR_PROGRESS.open(encoding="utf-8")]
    prior_strata = Counter(
        (int(row.get("context_length", -1)), str(row.get("kind"))) for row in prior_rows
    )
    require(
        sha256(PRIOR_RESULT) == PRIOR_RESULT_SHA256
        and sha256(PRIOR_ATTEMPT) == PRIOR_ATTEMPT_SHA256
        and sha256(PRIOR_PROGRESS) == PRIOR_PROGRESS_SHA256
        and sha256(PRIOR_FACE) == PRIOR_FACE_SHA256
        and prior_result.get("status") == "stopped"
        and prior_result.get("complete") is False
        and prior_result.get("fresh_queries") == PRIOR_QUERIES
        and prior_result.get("parameter_updates") == 0
        and prior_result.get("dev_queries") == 0
        and prior_result.get("holdout_queries") == 0
        and prior_result.get("confirmation_queries") == 0
        and len(prior_rows) == PRIOR_QUERIES
        and prior_progress_rows == prior_rows
        and {int(row.get("seed", -1)) for row in prior_rows} == {11}
        and {str(row.get("branch")) for row in prior_rows} == {"unknown8_rebalanced"}
        and prior_strata
        == Counter({(length, kind): ITEMS_PER_KIND for length in LENGTHS for kind in KINDS})
        and prior_result.get("errors")
        == ["Refuse: PLAN-N1-159 score stratum mismatch: 11/length_mixed_parent/8/lookup"]
        and prior_result.get("source_hashes_unchanged") is True
        and prior_result.get("progress_integrity", {}).get("complete") is True
        and prior_result.get("progress_integrity", {}).get("line_count") == PRIOR_QUERIES
        and prior_result.get("progress_integrity", {}).get("progress_sha256")
        == PRIOR_PROGRESS_SHA256
        and prior_result.get("cumulative_linked_cpu_seconds") == PRIOR_LINKED_CPU_SECONDS
        and prior_attempt.get("status") == "started"
        and prior_attempt.get("face_sha256") == PRIOR_FACE_SHA256
        and prior_attempt.get("planned_queries") == TOTAL_QUERIES
        and prior_readiness.get("complete") is True
        and prior_preflight.get("complete") is True
        and prior_preflight.get("independent_restore_count") == 6
        and prior_face.get("seed") == 129
        and prior_face.get("total_queries") == TOTAL_QUERIES
        and prior_face_audit.get("status") == "complete"
        and prior_face_checks.get("status") == "complete",
        "PLAN-N1-159A requires the sealed 480-row first endpoint and intact seed129 face/preflight",
    )
    endpoints, prior_sources, _prior_exclusions = n157_builder.source_context()
    if len(endpoints) != 6:
        raise Refuse("PLAN-N1-159 requires the same six frozen candidate/parent endpoints")
    sources = dict(prior_sources)
    merge_hashes(sources, n157a2_result.get("source_hashes", {}))
    merge_hashes(sources, n158_report.get("source_hashes", {}))
    for raw_path in (
        PLAN,
        PRIOR_CORRECTION_ADDENDUM,
        CORRECTION_ADDENDUM,
        N150_RESULT,
        N155A_RESULT,
        N155A_PROGRESS,
        N157A2_RESULT,
        N157A2_PROGRESS,
        N158_REPORT,
        PRIOR_RESULT,
        PRIOR_ATTEMPT,
        PRIOR_PROGRESS,
        PRIOR_SOURCE_AUDIT,
        PRIOR_READINESS,
        PRIOR_READINESS_ATTEMPT,
        PRIOR_SOURCE_BINDING,
        PRIOR_PREFLIGHT,
        PRIOR_PREFLIGHT_ATTEMPT,
        PRIOR_PREFLIGHT_BINDING,
        FACE128,
        PRIOR_FACE,
        PRIOR_FACE_AUDIT,
        PRIOR_FACE_CHECKS,
        n157_builder.FACE_AUDIT,
        n157_builder.FACE_CHECKS,
        Path(__file__).resolve(),
        ROOT / "scripts/training/run_taiji_n1_plan159_seed129_candidate_parent_diagnostic.py",
        TEST,
    ):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise Refuse(f"PLAN-N1-159 bound source missing: {path}")
        merge_hashes(sources, {str(path): sha256(path)})
    if not n150_builder.verify_hashes(sources):
        raise Refuse("PLAN-N1-159 prior source/checkpoint hashes changed")
    excluded_entities = set(_prior_exclusions["excluded_entities"])
    excluded_words = set(_prior_exclusions["excluded_words"])
    face128_entities, face128_words = _face_values(read_json(FACE128))
    excluded_entities.update(face128_entities)
    excluded_words.update(face128_words)
    exclusion_summary = {
        "prior_n157_face_entity_count": len(face128_entities),
        "prior_n157_face_word_value_count": len(face128_words),
        "excluded_entity_count": len(excluded_entities),
        "excluded_word_value_count": len(excluded_words),
        "entity_exclusion_sha256": n153_builder.digest_strings(excluded_entities),
        "word_exclusion_sha256": n153_builder.digest_strings(excluded_words),
        "prior_face128_sha256": sha256(FACE128),
    }
    return (
        endpoints,
        dict(sorted(sources.items())),
        {
            "excluded_entities": excluded_entities,
            "excluded_words": excluded_words,
            "exclusion_summary": exclusion_summary,
        },
    )


def source_readiness() -> dict:
    outputs = (
        SOURCE_AUDIT,
        READINESS,
        READINESS_ATTEMPT,
        SOURCE_BINDING,
        PREFLIGHT,
        PREFLIGHT_ATTEMPT,
        PREFLIGHT_BINDING,
        ATTEMPT,
        PROGRESS,
        RESULT,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("PLAN-N1-159 source/readiness/face outputs are write-once")
    started = time.process_time()
    endpoints, sources, exclusions = source_context()
    cpu = time.process_time() - started
    cumulative = PRIOR_LINKED_CPU_SECONDS + cpu
    complete = (
        len(endpoints) == 6
        and n150_builder.verify_hashes(sources)
        and cpu <= CPU_LIMIT_SECONDS
        and cumulative <= LINKED_CPU_LIMIT_SECONDS
    )
    audit = {
        "schema": "n1-plan159a-seed129-candidate-parent-source-audit-v1",
        "status": "complete" if complete else "stopped",
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": CORRECTION_SHA256,
        "prior_result_sha256": PRIOR_RESULT_SHA256,
        "prior_progress_sha256": PRIOR_PROGRESS_SHA256,
        "prior_face_sha256": PRIOR_FACE_SHA256,
        "prior_consumed_queries": PRIOR_QUERIES,
        "remaining_queries": TOTAL_QUERIES - PRIOR_QUERIES,
        "source_hashes": sources,
        "source_hashes_unchanged": n150_builder.verify_hashes(sources),
        "source_exclusion_summary": exclusions["exclusion_summary"],
        "face_seed": 129,
        "items_per_kind": ITEMS_PER_KIND,
        "nested_context_lengths": list(LENGTHS),
        "endpoint_count": len(endpoints),
        "queries_per_endpoint": QUERIES_PER_ENDPOINT,
        "total_queries": TOTAL_QUERIES,
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cpu_seconds": round(cpu, 6),
        "cumulative_linked_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "can_promote": False,
    }
    write_new(SOURCE_AUDIT, audit)
    prepared = dict(sources)
    prepared[str(SOURCE_AUDIT.resolve())] = sha256(SOURCE_AUDIT)
    readiness = {
        "schema": "n1-plan159a-seed129-candidate-parent-readiness-v1",
        "status": "ready_for_six_bundle_preflight" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": CORRECTION_SHA256,
        "prior_result_sha256": PRIOR_RESULT_SHA256,
        "prior_progress_sha256": PRIOR_PROGRESS_SHA256,
        "prior_face_sha256": PRIOR_FACE_SHA256,
        "prior_consumed_queries": PRIOR_QUERIES,
        "remaining_queries": TOTAL_QUERIES - PRIOR_QUERIES,
        "source_audit_sha256": sha256(SOURCE_AUDIT),
        "source_hashes": prepared,
        "source_hashes_unchanged": n150_builder.verify_hashes(prepared),
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
        "source_exclusion_summary": exclusions["exclusion_summary"],
        "face_seed": 129,
        "queries_per_endpoint": QUERIES_PER_ENDPOINT,
        "total_queries": TOTAL_QUERIES,
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cpu_seconds": round(cpu, 6),
        "cumulative_linked_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "can_promote": False,
    }
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan159a-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "correction_addendum_sha256": CORRECTION_SHA256,
            "prior_result_sha256": PRIOR_RESULT_SHA256,
            "cpu_seconds": round(cpu, 6),
            "model_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
        },
    )
    write_new(READINESS, readiness)
    write_new(
        SOURCE_BINDING,
        {
            "schema": "n1-plan159a-source-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "correction_addendum_sha256": CORRECTION_SHA256,
            "prior_result_sha256": PRIOR_RESULT_SHA256,
            "prior_consumed_queries": PRIOR_QUERIES,
            "remaining_queries": TOTAL_QUERIES - PRIOR_QUERIES,
            "readiness_sha256": sha256(READINESS),
            "source_hashes": prepared,
            "can_promote": False,
        },
    )
    return readiness


def prepare_face() -> dict:
    if any(path.exists() for path in (FACE, FACE_AUDIT, FACE_CHECKS)):
        raise FileExistsError("PLAN-N1-159 seed129 face outputs are write-once")
    if not READINESS.is_file() or read_json(READINESS).get("complete") is not True:
        raise Refuse("PLAN-N1-159 readiness is required before seed129 face generation")
    if not PREFLIGHT.is_file() or not PREFLIGHT_BINDING.is_file():
        raise Refuse("PLAN-N1-159 six-bundle restore preflight is required before face generation")
    preflight = read_json(PREFLIGHT)
    preflight_binding = read_json(PREFLIGHT_BINDING)
    if (
        preflight.get("complete") is not True
        or preflight.get("independent_restore_count") != 6
        or preflight_binding.get("status") != "complete"
        or preflight_binding.get("preflight_sha256") != sha256(PREFLIGHT)
        or preflight_binding.get("source_hashes") != preflight.get("source_hashes")
        or not n150_builder.verify_hashes(preflight.get("source_hashes", {}))
    ):
        raise Refuse("PLAN-N1-159 full-bundle preflight/source binding is stale")
    started = time.process_time()
    endpoints, sources, exclusions = source_context()
    ready_sources = dict(sources)
    ready_sources[str(SOURCE_AUDIT.resolve())] = sha256(SOURCE_AUDIT)
    if read_json(READINESS).get("source_hashes") != ready_sources:
        raise Refuse("PLAN-N1-159 readiness sources changed before face generation")
    face, generator_summary = n153_builder.build_face(
        exclusions["excluded_entities"], exclusions["excluded_words"], seed=129
    )
    for item in face["items"]:
        item["id"] = item["id"].replace("v28-", "v31-", 1)
        item["parent_id"] = item["id"]
    contract = n153_builder.validate_face(
        face,
        exclusions["excluded_entities"],
        exclusions["excluded_words"],
    )
    manifest = {
        **face,
        "plan159_schema": "n1-plan159-v31-candidate-parent-face-v1",
        "plan_sha256": PLAN_SHA256,
        "seed": 129,
        "source_exclusion_summary": exclusions["exclusion_summary"],
        "prior_seed128_face_sha256": sha256(FACE128),
        "endpoint_count": len(endpoints),
        "queries_per_endpoint": QUERIES_PER_ENDPOINT,
        "total_queries": TOTAL_QUERIES,
        "can_promote": False,
    }
    write_new(FACE, manifest)
    face_hashes = dict(ready_sources)
    face_hashes[str(READINESS.resolve())] = sha256(READINESS)
    face_hashes[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
    face_hashes[str(PREFLIGHT.resolve())] = sha256(PREFLIGHT)
    face_hashes[str(PREFLIGHT_BINDING.resolve())] = sha256(PREFLIGHT_BINDING)
    face_hashes[str(FACE.resolve())] = sha256(FACE)
    face_audit = {
        "schema": "n1-plan159-v31-seed129-face-audit-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "source_hashes": face_hashes,
        "source_hashes_unchanged": n150_builder.verify_hashes(face_hashes),
        "source_exclusion_summary": exclusions["exclusion_summary"],
        "face_contract": contract,
        "generator_summary": generator_summary,
        "prior_seed128_face_sha256": sha256(FACE128),
        "total_queries": TOTAL_QUERIES,
        "cpu_seconds": round(time.process_time() - started, 6),
        "can_promote": False,
    }
    write_new(FACE_AUDIT, face_audit)
    checks = {
        "schema": "n1-plan159-v31-seed129-face-checks-v1",
        "status": (
            "complete"
            if face_audit["source_hashes_unchanged"] and contract["accepted"]
            else "stopped"
        ),
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "face_audit_sha256": sha256(FACE_AUDIT),
        "face_contract": contract,
        "source_hashes_unchanged": face_audit["source_hashes_unchanged"],
        "prior_seed128_face_sha256": sha256(FACE128),
        "total_queries": TOTAL_QUERIES,
        "can_promote": False,
    }
    write_new(FACE_CHECKS, checks)
    if checks["status"] != "complete":
        raise Refuse("PLAN-N1-159 seed129 face/source contract failed; face is sealed")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-159 seed129 candidate-parent replication face"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--prepare-face", action="store_true")
    args = parser.parse_args()
    result = source_readiness() if args.validate_only else prepare_face()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "source_hashes_unchanged",
                    "face_seed",
                    "endpoint_count",
                    "total_queries",
                    "cpu_seconds",
                    "cumulative_linked_cpu_seconds",
                    "model_loads",
                    "checkpoint_loads",
                    "model_queries",
                    "parameter_updates",
                    "can_promote",
                )
                if key in result
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete", result.get("status") == "complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
