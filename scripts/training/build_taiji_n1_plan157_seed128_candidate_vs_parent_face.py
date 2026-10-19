"""构造PLAN-N1-157的seed128 fresh candidate-vs-parent诊断面并审计来源。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (
    analyze_taiji_n1_plan156_plan155a_train_score_itemwise_attribution as n156,
)
from scripts.training import build_taiji_n1_plan153_v28_fresh_context_length_face as n153_builder
from scripts.training import build_taiji_n1_plan155_short_unknown_context_length as n155_builder
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import (
    run_taiji_n1_plan155_short_unknown_context_length_training as n155_runner,
)
from scripts.training import run_taiji_n1_v27_length_robust_training_after_n149 as n150

PLAN = ROOT / "plans/reference/PLAN-N1-157_seed128_candidate_vs_parent_fresh_diagnostic_20261020.md"
PLAN_SHA256 = "ad097dc74c03d9df49a1101d4af3b7159c1ab4741e586b73557cf8e4f6f50e44"
N150_RESULT = n150.RESULT
N150_RESULT_SHA256 = n155_builder.N150_RESULT_SHA256
N155A_RESULT = n155_builder.RESULT
N155A_RESULT_SHA256 = "e577d4dda753d016a46e11ab2e0202b2845206af107c54f87aff7f9fdf1ebb11"
N155A_PROGRESS = n155_builder.PROGRESS
N155A_PROGRESS_SHA256 = "83e951930272d30ab3b70a0164dba0ef628d8876a8138708be08a97d3a7acf73"
N156_REPORT = n156.REPORT
N156_REPORT_SHA256 = "8952b96d41a2380a7325f691f211e4d63bfff0d4543e38665a39e30c9cd51a75"

SOURCE_AUDIT = ROOT / "reports/taiji_n1_plan157_candidate_parent_source_audit_20261020.json"
READINESS = ROOT / "reports/taiji_n1_plan157_candidate_parent_readiness_20261020.json"
READINESS_ATTEMPT = ROOT / "reports/taiji_n1_plan157_candidate_parent_readiness_attempt_20261020.json"
SOURCE_BINDING = ROOT / "reports/taiji_n1_plan157_candidate_parent_source_binding_20261020.json"
PREFLIGHT = ROOT / "reports/taiji_n1_plan157_candidate_parent_preflight_20261020.json"
PREFLIGHT_ATTEMPT = ROOT / "reports/taiji_n1_plan157_candidate_parent_preflight_attempt_20261020.json"
PREFLIGHT_BINDING = ROOT / "reports/taiji_n1_plan157_candidate_parent_preflight_binding_20261020.json"
FACE = ROOT / "plans/manifests/n1_plan157_v30_candidate_vs_parent_seed128_20261020.json"
FACE_AUDIT = ROOT / "reports/taiji_n1_plan157_v30_candidate_vs_parent_face_audit_20261020.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_plan157_v30_candidate_vs_parent_face_checks_20261020.json"
ATTEMPT = ROOT / "reports/taiji_n1_plan157_candidate_parent_attempt_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan157_candidate_parent_progress_20261020.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan157_candidate_parent_result_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan157_seed128_candidate_vs_parent_diagnostic_contract.py"

SEEDS = (11, 12, 13)
BRANCHES = ("unknown8_rebalanced", "length_mixed_parent")
ENDPOINTS = tuple((seed, branch) for seed in SEEDS for branch in BRANCHES)
PRIOR_LINKED_CPU_SECONDS = 9_328.078125
CPU_LIMIT_SECONDS = 3_600.0
LINKED_CPU_LIMIT_SECONDS = 18_000.0
ITEMS_PER_KIND = 40
LENGTHS = (8, 16, 32)
KINDS = ("lookup", "revision", "transition", "unknown")
QUERIES_PER_ENDPOINT = ITEMS_PER_KIND * len(KINDS) * len(LENGTHS)
TOTAL_QUERIES = QUERIES_PER_ENDPOINT * len(ENDPOINTS)
UNKNOWN = n153_builder.UNKNOWN
DIRECTIONS = n153_builder.DIRECTIONS
TURNS = n153_builder.TURNS


class Refuse(RuntimeError):
    """冻结来源、face或endpoint合同不成立。"""


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
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def digest_strings(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


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
            raise Refuse(f"conflicting PLAN-N1-157 source hash: {path}")
        target[path] = digest


def _result_rows(result: dict, branch: str) -> dict[int, dict]:
    rows = {
        int(row["seed"]): row
        for row in result.get("models", [])
        if row.get("branch") == branch and row.get("training_sufficient") is True
    }
    require(set(rows) == set(SEEDS), f"PLAN-N1-157 requires sufficient seed11/12/13 endpoints: {branch}")
    return rows


def source_context() -> tuple[list[dict], dict[str, str], dict]:
    if sha256(PLAN) != PLAN_SHA256:
        raise Refuse("PLAN-N1-157 plan SHA changed")
    n150_result = read_json(N150_RESULT)
    n155a_result = read_json(N155A_RESULT)
    n156_report = read_json(N156_REPORT)
    require(sha256(N150_RESULT) == N150_RESULT_SHA256, "N1-150 result SHA changed")
    require(sha256(N155A_RESULT) == N155A_RESULT_SHA256, "N1-155A result SHA changed")
    require(sha256(N155A_PROGRESS) == N155A_PROGRESS_SHA256, "N1-155A progress SHA changed")
    require(sha256(N156_REPORT) == N156_REPORT_SHA256, "N1-156 attribution SHA changed")
    require(
        n150_result.get("complete") is True
        and n150_result.get("training_sufficient_all_endpoints") is True
        and n150_result.get("can_promote") is False,
        "N1-150 length_mixed parent evidence is incomplete",
    )
    require(
        n155a_result.get("complete") is True
        and n155a_result.get("training_updates") == 55_296
        and n155a_result.get("train_score_queries") == 7_488
        and n155a_result.get("source_hashes_unchanged") is True
        and n155a_result.get("fresh_diagnostic_queries") == 0
        and n155a_result.get("can_promote") is False
        and n155a_result.get("progress_integrity", {}).get("progress_sha256")
        == N155A_PROGRESS_SHA256,
        "N1-155A training receipt is incomplete or its fresh face was already consumed",
    )
    require(
        n156_report.get("complete") is True
        and n156_report.get("n155a_result_sha256") == N155A_RESULT_SHA256
        and n156_report.get("progress_sha256") == N155A_PROGRESS_SHA256
        and n156_report.get("source_hashes_unchanged") is True
        and n156_report.get("can_promote") is False,
        "N1-156 attribution/source binding is incomplete",
    )
    parents = _result_rows(n150_result, "length_mixed")
    candidates = _result_rows(n155a_result, "unknown8_rebalanced")
    endpoints = []
    for seed in SEEDS:
        for branch, row in (
            ("unknown8_rebalanced", candidates[seed]),
            ("length_mixed_parent", parents[seed]),
        ):
            checkpoint = Path(row["final_checkpoint_path"]).resolve()
            digest = str(row["final_checkpoint_sha256"])
            if not checkpoint.is_file() or sha256(checkpoint) != digest:
                raise Refuse(f"PLAN-N1-157 endpoint checkpoint changed: seed{seed}/{branch}")
            endpoints.append(
                {
                    "seed": seed,
                    "branch": branch,
                    "source_branch": row["branch"],
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": digest,
                    "bundle_fingerprint": row["final_bundle_fingerprint"],
                    "total_updates": row["total_updates"],
                    "training_sufficient": row["training_sufficient"],
                }
            )

    _excluded_entities, _excluded_words, _summary, inherited_hashes, _runtime = (
        n153_builder.collect_n150_exclusions()
    )
    sources = {}
    merge_hashes(sources, n150_result.get("source_hashes", {}))
    merge_hashes(sources, n155a_result.get("source_hashes", {}))
    merge_hashes(sources, n156_report.get("source_hashes", {}))
    merge_hashes(sources, inherited_hashes)
    for raw_path in (
        PLAN,
        N150_RESULT,
        N155A_RESULT,
        N155A_PROGRESS,
        N156_REPORT,
        N155A_RESULT.parent / "taiji_n1_plan155a_unknown8_rebalance_attempt_20261020.json",
        n153_builder.FACE,
        n153_builder.FACE_CHECKS,
        n153_builder.SOURCE_AUDIT,
        Path(n155_builder.__file__).resolve(),
        Path(n155_runner.__file__).resolve(),
        Path(n150.__file__).resolve(),
        Path(n153_builder.__file__).resolve(),
        Path(n150_builder.__file__).resolve(),
        ROOT / "plans/reference/PLAN-N1-150_matched_long_context_training_continuation_draft_20261020.md",
        ROOT / "plans/reference/PLAN-N1-150A_train_only_scoring_denominator_freeze_20261020.md",
        ROOT / "plans/reference/PLAN-N1-150B_training_cpu_ceiling_revision_20261020.md",
        ROOT / "plans/reference/PLAN-N1-155_short_context_unknown_rebalance_and_fresh_diagnostic_20261020.md",
        ROOT / "plans/reference/PLAN-N1-155A_runtime_views_handoff_correction_20261020.md",
        ROOT / "plans/reference/PLAN-N1-156_plan155a_train_score_itemwise_attribution_20261020.md",
        Path(__file__).resolve(),
        ROOT / "scripts/training/run_taiji_n1_plan157_seed128_candidate_vs_parent_diagnostic.py",
        TEST,
    ):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise Refuse(f"PLAN-N1-157 bound source missing: {path}")
        merge_hashes(sources, {str(path): sha256(path)})
    for endpoint in endpoints:
        merge_hashes(
            sources,
            {endpoint["checkpoint_path"]: endpoint["checkpoint_sha256"]},
        )
    if not n150_builder.verify_hashes(sources):
        raise Refuse("PLAN-N1-157 source/checkpoint hashes changed")
    exclusion_entities, exclusion_words, exclusion_summary, _hashes, _runtime = (
        n153_builder.collect_n150_exclusions()
    )
    prior_face = n153_builder.read_json(n153_builder.FACE)
    for item in prior_face.get("items", []):
        exclusion_entities.update(str(name) for name in item.get("entities", []))
        answer = item.get("answer")
        if answer not in (None, UNKNOWN, *DIRECTIONS):
            exclusion_words.add(str(answer))
        for row in item.get("observations_by_length", {}).get("32", []):
            n153_builder._add_row_values(row, exclusion_entities, exclusion_words)
    exclusions = {
        "entity_count": len(exclusion_entities),
        "word_value_count": len(exclusion_words),
        "entity_sha256": n153_builder.digest_strings(exclusion_entities),
        "word_value_sha256": n153_builder.digest_strings(exclusion_words),
        "n150_exclusion_summary": exclusion_summary,
        "prior_n153_face_sha256": sha256(n153_builder.FACE),
    }
    return endpoints, dict(sorted(sources.items())), {
        "excluded_entities": exclusion_entities,
        "excluded_words": exclusion_words,
        "exclusion_summary": exclusions,
    }


def source_readiness() -> dict:
    outputs = (SOURCE_AUDIT, READINESS, READINESS_ATTEMPT, SOURCE_BINDING, PREFLIGHT,
               PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING, FACE, FACE_AUDIT, FACE_CHECKS,
               ATTEMPT, PROGRESS, RESULT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("PLAN-N1-157 readiness/face/diagnostic outputs are write-once")
    started = time.process_time()
    endpoints, sources, exclusions = source_context()
    cpu = time.process_time() - started
    cumulative = PRIOR_LINKED_CPU_SECONDS + cpu
    complete = (
        len(endpoints) == 6
        and len({(row["seed"], row["branch"]) for row in endpoints}) == 6
        and n150_builder.verify_hashes(sources)
        and cpu <= CPU_LIMIT_SECONDS
        and cumulative <= LINKED_CPU_LIMIT_SECONDS
    )
    audit = {
        "schema": "n1-plan157-candidate-parent-source-audit-v1",
        "status": "complete" if complete else "stopped",
        "plan_sha256": PLAN_SHA256,
        "n155a_result_sha256": N155A_RESULT_SHA256,
        "n155a_progress_sha256": N155A_PROGRESS_SHA256,
        "n156_report_sha256": N156_REPORT_SHA256,
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
        "source_hashes": sources,
        "source_hashes_unchanged": n150_builder.verify_hashes(sources),
        "source_exclusion_summary": exclusions["exclusion_summary"],
        "face_spec": {
            "seed": 128,
            "items_per_kind": ITEMS_PER_KIND,
            "kinds": list(KINDS),
            "nested_context_lengths": list(LENGTHS),
            "queries_per_endpoint": QUERIES_PER_ENDPOINT,
            "total_queries": TOTAL_QUERIES,
        },
        "cpu_seconds": round(cpu, 6),
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
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
        "schema": "n1-plan157-candidate-parent-readiness-v1",
        "status": "ready_for_six_bundle_preflight" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "source_audit_sha256": sha256(SOURCE_AUDIT),
        "source_hashes": prepared,
        "source_hashes_unchanged": n150_builder.verify_hashes(prepared),
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
        "source_exclusion_summary": exclusions["exclusion_summary"],
        "face_spec": audit["face_spec"],
        "cpu_seconds": round(cpu, 6),
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cumulative_linked_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "can_promote": False,
    }
    write_new(READINESS_ATTEMPT, {
        "schema": "n1-plan157-readiness-attempt-v1",
        "status": "started",
        "plan_sha256": PLAN_SHA256,
        "cpu_seconds": round(cpu, 6),
        "model_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
    })
    write_new(READINESS, readiness)
    write_new(SOURCE_BINDING, {
        "schema": "n1-plan157-source-binding-v1",
        "status": "complete" if complete else "stopped",
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_hashes": prepared,
        "can_promote": False,
    })
    return readiness


def prepare_face() -> dict:
    if any(path.exists() for path in (FACE, FACE_AUDIT, FACE_CHECKS)):
        raise FileExistsError("PLAN-N1-157 seed128 face outputs are write-once")
    if not READINESS.is_file() or read_json(READINESS).get("complete") is not True:
        raise Refuse("PLAN-N1-157 readiness is required before face generation")
    if not PREFLIGHT.is_file() or not PREFLIGHT_BINDING.is_file():
        raise Refuse("PLAN-N1-157 six-bundle restore preflight is required before face generation")
    preflight = read_json(PREFLIGHT)
    preflight_binding = read_json(PREFLIGHT_BINDING)
    if (
        preflight.get("complete") is not True
        or preflight.get("plan_sha256") != PLAN_SHA256
        or preflight.get("independent_restore_count") != 6
        or preflight_binding.get("status") != "complete"
        or preflight_binding.get("preflight_sha256") != sha256(PREFLIGHT)
        or preflight_binding.get("source_hashes") != preflight.get("source_hashes")
        or not n150_builder.verify_hashes(preflight.get("source_hashes", {}))
    ):
        raise Refuse("PLAN-N1-157 full-bundle restore preflight/source binding is stale")
    started = time.process_time()
    endpoints, sources, exclusions = source_context()
    readiness = read_json(READINESS)
    ready_sources = dict(sources)
    ready_sources[str(SOURCE_AUDIT.resolve())] = sha256(SOURCE_AUDIT)
    if readiness.get("source_hashes") != ready_sources or not n150_builder.verify_hashes(ready_sources):
        raise Refuse("PLAN-N1-157 readiness/source hashes changed before face generation")
    base_face, generator_summary = n153_builder.build_face(
        exclusions["excluded_entities"], exclusions["excluded_words"], seed=128
    )
    for item in base_face["items"]:
        item["id"] = item["id"].replace("v28-", "v30-", 1)
        item["parent_id"] = item["id"]
    checks = n153_builder.validate_face(
        base_face,
        exclusions["excluded_entities"],
        exclusions["excluded_words"],
    )
    face = {
        **base_face,
        "plan157_schema": "n1-plan157-v30-candidate-vs-parent-face-v1",
        "plan_sha256": PLAN_SHA256,
        "seed": 128,
        "endpoint_count": 6,
        "queries_per_endpoint": QUERIES_PER_ENDPOINT,
        "total_queries": TOTAL_QUERIES,
        "source_exclusion_summary": exclusions["exclusion_summary"],
        "candidate_endpoints": [row for row in endpoints if row["branch"] == "unknown8_rebalanced"],
        "parent_endpoints": [row for row in endpoints if row["branch"] == "length_mixed_parent"],
        "can_promote": False,
    }
    face_digest = canonical_sha(face)
    face_hashes = dict(ready_sources)
    for path in (READINESS, SOURCE_BINDING, PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING):
        face_hashes[str(Path(path).resolve())] = sha256(path)
    for row in endpoints:
        face_hashes[row["checkpoint_path"]] = row["checkpoint_sha256"]
    write_new(FACE, face)
    face_hashes[str(FACE.resolve())] = sha256(FACE)
    face_hashes[str(READINESS.resolve())] = sha256(READINESS)
    face_audit = {
        "schema": "n1-plan157-v30-candidate-parent-face-audit-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "face_canonical_sha256": face_digest,
        "source_hashes": face_hashes,
        "source_hashes_unchanged": n150_builder.verify_hashes(face_hashes),
        "source_exclusion_summary": exclusions["exclusion_summary"],
        "face_contract": checks,
        "generator_summary": generator_summary,
        "endpoint_count": len(endpoints),
        "total_queries": TOTAL_QUERIES,
        "cpu_seconds": round(time.process_time() - started, 6),
        "can_promote": False,
    }
    write_new(FACE_AUDIT, face_audit)
    face_checks = {
        "schema": "n1-plan157-v30-candidate-parent-face-checks-v1",
        "status": "complete" if face_audit["source_hashes_unchanged"] and checks["accepted"] else "stopped",
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "face_audit_sha256": sha256(FACE_AUDIT),
        "source_hashes_unchanged": face_audit["source_hashes_unchanged"],
        "face_contract": checks,
        "query_count_per_endpoint": QUERIES_PER_ENDPOINT,
        "total_queries": TOTAL_QUERIES,
        "can_promote": False,
    }
    write_new(FACE_CHECKS, face_checks)
    if face_checks["status"] != "complete":
        raise Refuse("PLAN-N1-157 generated face/source contract failed; face is sealed")
    return face_checks


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-157 candidate-vs-parent fresh-face preparation")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--prepare-face", action="store_true")
    args = parser.parse_args()
    report = source_readiness() if args.validate_only else prepare_face()
    print(json.dumps({key: report.get(key) for key in (
        "status", "complete", "source_hashes_unchanged", "endpoint_count", "total_queries",
        "cpu_seconds", "cumulative_linked_cpu_seconds", "model_loads", "checkpoint_loads",
        "model_queries", "parameter_updates", "can_promote") if key in report},
        ensure_ascii=True, sort_keys=True))
    return 0 if report.get("complete", report.get("status") == "complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
