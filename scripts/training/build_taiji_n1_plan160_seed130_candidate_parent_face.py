"""构造PLAN-N1-160 seed130 candidate-parent独立fresh复制面。"""

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
from scripts.training import build_taiji_n1_plan159_seed129_candidate_parent_face as n159_builder
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder

PLAN = ROOT / "plans/reference/PLAN-N1-160_seed130_candidate_parent_fresh_replication_20261020.md"
PLAN_SHA256 = "55ca0f18a741857ac903502213222feba92cbfc2a6b16215c221224fd8551d7a"
CORRECTION_ADDENDUM = ROOT / "plans/reference/PLAN-N1-160A_remaining_endpoint_continuation_20261020.md"
CORRECTION_SHA256 = "bf4e3d4953da077b45cf04ae073a8c415cdab40786b0c06ca545aa1eab96c3a4"
N159A_RESULT = ROOT / "reports/taiji_n1_plan159a_candidate_parent_result_20261020.json"
N159A_RESULT_SHA256 = "809bc1c79cc130129e07fcbed9080e6af9f1dea0322381c1b6b1e9e5204a65dd"
N159A_PROGRESS = ROOT / "reports/taiji_n1_plan159a_candidate_parent_progress_20261020.jsonl"
N159A_PROGRESS_SHA256 = "8c66b6edb1f2a13add52f1b6753ef4d4175118b1d2c81c54218d54f5d1a76b33"
N158_REPORT = n158.REPORT
N158_REPORT_SHA256 = "6f0681bf2c819df9444ff7e89faaca6f536a3aac55181e68059f8b892d99591a"
PRIOR_SOURCE_AUDIT = ROOT / "reports/taiji_n1_plan160_candidate_parent_source_audit_20261020.json"
PRIOR_READINESS = ROOT / "reports/taiji_n1_plan160_candidate_parent_readiness_20261020.json"
PRIOR_READINESS_ATTEMPT = ROOT / "reports/taiji_n1_plan160_candidate_parent_readiness_attempt_20261020.json"
PRIOR_SOURCE_BINDING = ROOT / "reports/taiji_n1_plan160_candidate_parent_source_binding_20261020.json"
PRIOR_PREFLIGHT = ROOT / "reports/taiji_n1_plan160_candidate_parent_preflight_20261020.json"
PRIOR_PREFLIGHT_ATTEMPT = ROOT / "reports/taiji_n1_plan160_candidate_parent_preflight_attempt_20261020.json"
PRIOR_PREFLIGHT_BINDING = ROOT / "reports/taiji_n1_plan160_candidate_parent_preflight_binding_20261020.json"
PRIOR_ATTEMPT = ROOT / "reports/taiji_n1_plan160_candidate_parent_attempt_20261020.json"
PRIOR_PROGRESS = ROOT / "reports/taiji_n1_plan160_candidate_parent_progress_20261020.jsonl"
PRIOR_RESULT = ROOT / "reports/taiji_n1_plan160_candidate_parent_result_20261020.json"
PRIOR_RESULT_SHA256 = "b6446a5542e9c4c149f00d36f4a9b0e4ed8b9ea6fe82cec68f76537d6b7d04d1"
PRIOR_ATTEMPT_SHA256 = "be6001a8322ad110715ebfe1527a18dcc59cf2edc58bf27e5a37a7b03c8baf8c"
PRIOR_PROGRESS_SHA256 = "984e50b6ee32b060ef84ba60b0292844238e542a671221530b1ee02d8b89dd78"
PRIOR_FACE_SHA256 = "5b15a47cd7ce0df046bc269fb1bf3a01f4f5242ce3c318c7f953e68547777ee3"
PRIOR_LINKED_CPU_SECONDS = 10_421.21875
PRIOR_QUERIES = 480
FACE128 = n159_builder.FACE128
FACE129 = n159_builder.FACE
FACE128_SHA256 = "cf34e7de874b316581c4c6520786e2bf6406d328d4989b869901eb667e9c9a2f"
FACE129_SHA256 = "b445e96f7231ba874efa5300f99fd26a92a90548e56d14ba9caf881d30b729d3"

SOURCE_AUDIT = ROOT / "reports/taiji_n1_plan160a_candidate_parent_source_audit_20261020.json"
READINESS = ROOT / "reports/taiji_n1_plan160a_candidate_parent_readiness_20261020.json"
READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan160a_candidate_parent_readiness_attempt_20261020.json"
)
SOURCE_BINDING = ROOT / "reports/taiji_n1_plan160a_candidate_parent_source_binding_20261020.json"
PREFLIGHT = ROOT / "reports/taiji_n1_plan160a_candidate_parent_preflight_20261020.json"
PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan160a_candidate_parent_preflight_attempt_20261020.json"
)
PREFLIGHT_BINDING = (
    ROOT / "reports/taiji_n1_plan160a_candidate_parent_preflight_binding_20261020.json"
)
FACE = ROOT / "plans/manifests/n1_plan160_v32_candidate_parent_seed130_20261020.json"
FACE_AUDIT = ROOT / "reports/taiji_n1_plan160_v32_candidate_parent_face_audit_20261020.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_plan160_v32_candidate_parent_face_checks_20261020.json"
ATTEMPT = ROOT / "reports/taiji_n1_plan160a_candidate_parent_attempt_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan160a_candidate_parent_progress_20261020.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan160a_candidate_parent_result_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan160_seed130_candidate_parent_diagnostic_contract.py"

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
    """PLAN-N1-160 source, cross-face exclusion or face-contract failure."""


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


def source_context() -> tuple[list[dict], dict[str, str], dict]:
    if sha256(PLAN) != PLAN_SHA256:
        raise Refuse("PLAN-N1-160 plan SHA changed")
    if sha256(CORRECTION_ADDENDUM) != CORRECTION_SHA256:
        raise Refuse("PLAN-N1-160A correction addendum SHA changed")
    n159a = read_json(N159A_RESULT)
    n158_result = read_json(N158_REPORT)
    prior_result = read_json(PRIOR_RESULT)
    prior_attempt = read_json(PRIOR_ATTEMPT)
    prior_readiness = read_json(PRIOR_READINESS)
    prior_preflight = read_json(PRIOR_PREFLIGHT)
    prior_face = read_json(FACE)
    prior_face_audit = read_json(FACE_AUDIT)
    prior_face_checks = read_json(FACE_CHECKS)
    require(sha256(N159A_RESULT) == N159A_RESULT_SHA256, "N1-159A result SHA changed")
    require(sha256(N159A_PROGRESS) == N159A_PROGRESS_SHA256, "N1-159A progress SHA changed")
    require(sha256(N158_REPORT) == N158_REPORT_SHA256, "N1-158 attribution SHA changed")
    require(
        sha256(FACE128) == FACE128_SHA256 and sha256(FACE129) == FACE129_SHA256,
        "prior fresh face SHA changed",
    )
    require(
        n159a.get("complete") is True
        and n159a.get("fresh_queries") == 2_880
        and n159a.get("source_hashes_unchanged") is True
        and n159a.get("can_promote") is False
        and n158_result.get("complete") is True
        and n158_result.get("source_result_sha256") == n159_builder.N157A2_RESULT_SHA256
        and n158_result.get("source_progress_sha256") == n159_builder.N157A2_PROGRESS_SHA256
        and n158_result.get("face_sha256") == sha256(FACE128)
        and n158_result.get("can_promote") is False,
        "N1-159A/N1-158 replication source evidence is incomplete",
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
        and sha256(FACE) == PRIOR_FACE_SHA256
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
        and prior_strata == Counter({(length, kind): ITEMS_PER_KIND for length in LENGTHS for kind in KINDS})
        and prior_result.get("errors")
        == ["Refuse: PLAN-N1-160 summary stratum differs: 11/length_mixed_parent/8/lookup"]
        and prior_result.get("source_hashes_unchanged") is True
        and prior_result.get("progress_integrity", {}).get("complete") is True
        and prior_result.get("progress_integrity", {}).get("line_count") == PRIOR_QUERIES
        and prior_result.get("progress_integrity", {}).get("progress_sha256") == PRIOR_PROGRESS_SHA256
        and prior_result.get("cumulative_linked_cpu_seconds") == PRIOR_LINKED_CPU_SECONDS
        and prior_attempt.get("status") == "started"
        and prior_attempt.get("face_sha256") == PRIOR_FACE_SHA256
        and prior_attempt.get("planned_queries") == TOTAL_QUERIES
        and prior_readiness.get("complete") is True
        and prior_preflight.get("complete") is True
        and prior_preflight.get("independent_restore_count") == 6
        and prior_face.get("seed") == 130
        and prior_face.get("total_queries") == TOTAL_QUERIES
        and prior_face_audit.get("status") == "complete"
        and prior_face_checks.get("status") == "complete",
        "PLAN-N1-160A requires the sealed first endpoint and intact seed130 face/preflight",
    )
    endpoints, prior_sources, prior_exclusions = n159_builder.source_context()
    if len(endpoints) != 6:
        raise Refuse("PLAN-N1-160 requires the frozen six candidate/parent endpoints")
    sources = dict(prior_sources)
    n150_result = read_json(n159_builder.N150_RESULT)
    merge = n159_builder.merge_hashes
    merge(sources, n150_result.get("source_hashes", {}))
    merge(sources, n159a.get("source_hashes", {}))
    merge(sources, n158_result.get("source_hashes", {}))
    for path in (
        CORRECTION_ADDENDUM,
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
        FACE,
        FACE_AUDIT,
        FACE_CHECKS,
    ):
        path = Path(path).resolve()
        if not path.is_file():
            raise Refuse(f"PLAN-N1-160A prior source missing: {path}")
        merge(sources, {str(path): sha256(path)})
    for path in (
        PLAN,
        N159A_RESULT,
        N159A_PROGRESS,
        N158_REPORT,
        FACE128,
        FACE129,
        n159_builder.FACE_AUDIT,
        n159_builder.FACE_CHECKS,
        Path(__file__).resolve(),
        ROOT / "scripts/training/run_taiji_n1_plan160_seed130_candidate_parent_diagnostic.py",
        TEST,
    ):
        path = Path(path).resolve()
        if not path.is_file():
            raise Refuse(f"PLAN-N1-160 bound source missing: {path}")
        merge(sources, {str(path): sha256(path)})
    for endpoint in endpoints:
        merge(sources, {endpoint["checkpoint_path"]: endpoint["checkpoint_sha256"]})
    if not n150_builder.verify_hashes(sources):
        raise Refuse("PLAN-N1-160 prior source/checkpoint hashes changed")

    excluded_entities = set(prior_exclusions["excluded_entities"])
    excluded_words = set(prior_exclusions["excluded_words"])
    face128_entities, face128_words = n159_builder._face_values(read_json(FACE128))
    face129_entities, face129_words = n159_builder._face_values(read_json(FACE129))
    excluded_entities.update(face128_entities)
    excluded_entities.update(face129_entities)
    excluded_words.update(face128_words)
    excluded_words.update(face129_words)
    summary = {
        "prior_seed128_face_sha256": sha256(FACE128),
        "prior_seed129_face_sha256": sha256(FACE129),
        "seed128_entity_count": len(face128_entities),
        "seed128_word_value_count": len(face128_words),
        "seed129_entity_count": len(face129_entities),
        "seed129_word_value_count": len(face129_words),
        "excluded_entity_count": len(excluded_entities),
        "excluded_word_value_count": len(excluded_words),
        "entity_exclusion_sha256": n153_builder.digest_strings(excluded_entities),
        "word_exclusion_sha256": n153_builder.digest_strings(excluded_words),
        "prior_n159_exclusion_summary": prior_exclusions["exclusion_summary"],
    }
    return (
        endpoints,
        dict(sorted(sources.items())),
        {
            "excluded_entities": excluded_entities,
            "excluded_words": excluded_words,
            "exclusion_summary": summary,
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
        raise FileExistsError("PLAN-N1-160 source/readiness/face outputs are write-once")
    started = time.process_time()
    endpoints, sources, exclusions = source_context()
    cpu = time.process_time() - started
    linked = PRIOR_LINKED_CPU_SECONDS + cpu
    complete = (
        len(endpoints) == 6
        and n150_builder.verify_hashes(sources)
        and cpu <= CPU_LIMIT_SECONDS
        and linked <= LINKED_CPU_LIMIT_SECONDS
    )
    audit = {
        "schema": "n1-plan160a-seed130-candidate-parent-source-audit-v1",
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
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
        "source_exclusion_summary": exclusions["exclusion_summary"],
        "face_spec": {
            "seed": 130,
            "items_per_kind": ITEMS_PER_KIND,
            "kinds": list(KINDS),
            "lengths": list(LENGTHS),
            "total_queries": TOTAL_QUERIES,
        },
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cpu_seconds": round(cpu, 6),
        "cumulative_linked_cpu_seconds": round(linked, 6),
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
        "schema": "n1-plan160a-seed130-candidate-parent-readiness-v1",
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
        "face_spec": audit["face_spec"],
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cpu_seconds": round(cpu, 6),
        "cumulative_linked_cpu_seconds": round(linked, 6),
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
            "schema": "n1-plan160a-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "correction_addendum_sha256": CORRECTION_SHA256,
            "prior_result_sha256": PRIOR_RESULT_SHA256,
            "prior_consumed_queries": PRIOR_QUERIES,
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
            "schema": "n1-plan160a-source-binding-v1",
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


def verify_sealed_face() -> dict:
    if not all(path.is_file() for path in (FACE, FACE_AUDIT, FACE_CHECKS)):
        raise FileNotFoundError("PLAN-N1-160A requires the existing seed130 face and audits")
    face = read_json(FACE)
    audit = read_json(FACE_AUDIT)
    checks = read_json(FACE_CHECKS)
    require(sha256(FACE) == PRIOR_FACE_SHA256, "PLAN-N1-160A seed130 face SHA changed")
    require(
        face.get("seed") == 130
        and face.get("total_queries") == TOTAL_QUERIES
        and audit.get("status") == "complete"
        and audit.get("face_sha256") == PRIOR_FACE_SHA256
        and checks.get("status") == "complete"
        and checks.get("face_sha256") == PRIOR_FACE_SHA256
        and checks.get("face_audit_sha256") == sha256(FACE_AUDIT),
        "PLAN-N1-160A sealed seed130 face audit is stale",
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-160A source readiness and sealed face verification"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--verify-sealed-face", action="store_true")
    args = parser.parse_args()
    result = source_readiness() if args.validate_only else verify_sealed_face()
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "status",
                    "complete",
                    "source_hashes_unchanged",
                    "endpoint_count",
                    "face_seed",
                    "total_queries",
                    "cpu_seconds",
                    "cumulative_linked_cpu_seconds",
                    "model_loads",
                    "checkpoint_loads",
                    "model_queries",
                    "parameter_updates",
                    "can_promote",
                )
                if k in result
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete", result.get("status") == "complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
