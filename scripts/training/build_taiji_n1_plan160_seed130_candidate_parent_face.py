"""构造PLAN-N1-160 seed130 candidate-parent独立fresh复制面。"""

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
    analyze_taiji_n1_plan158_plan157a2_fresh_face_itemwise_attribution as n158,
)
from scripts.training import build_taiji_n1_plan153_v28_fresh_context_length_face as n153_builder
from scripts.training import build_taiji_n1_plan159_seed129_candidate_parent_face as n159_builder
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder

PLAN = ROOT / "plans/reference/PLAN-N1-160_seed130_candidate_parent_fresh_replication_20261020.md"
PLAN_SHA256 = "55ca0f18a741857ac903502213222feba92cbfc2a6b16215c221224fd8551d7a"
N159A_RESULT = ROOT / "reports/taiji_n1_plan159a_candidate_parent_result_20261020.json"
N159A_RESULT_SHA256 = "809bc1c79cc130129e07fcbed9080e6af9f1dea0322381c1b6b1e9e5204a65dd"
N159A_PROGRESS = ROOT / "reports/taiji_n1_plan159a_candidate_parent_progress_20261020.jsonl"
N159A_PROGRESS_SHA256 = "8c66b6edb1f2a13add52f1b6753ef4d4175118b1d2c81c54218d54f5d1a76b33"
N158_REPORT = n158.REPORT
N158_REPORT_SHA256 = "6f0681bf2c819df9444ff7e89faaca6f536a3aac55181e68059f8b892d99591a"
FACE128 = n159_builder.FACE128
FACE129 = n159_builder.FACE
FACE128_SHA256 = "cf34e7de874b316581c4c6520786e2bf6406d328d4989b869901eb667e9c9a2f"
FACE129_SHA256 = "b445e96f7231ba874efa5300f99fd26a92a90548e56d14ba9caf881d30b729d3"

SOURCE_AUDIT = ROOT / "reports/taiji_n1_plan160_candidate_parent_source_audit_20261020.json"
READINESS = ROOT / "reports/taiji_n1_plan160_candidate_parent_readiness_20261020.json"
READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan160_candidate_parent_readiness_attempt_20261020.json"
)
SOURCE_BINDING = ROOT / "reports/taiji_n1_plan160_candidate_parent_source_binding_20261020.json"
PREFLIGHT = ROOT / "reports/taiji_n1_plan160_candidate_parent_preflight_20261020.json"
PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan160_candidate_parent_preflight_attempt_20261020.json"
)
PREFLIGHT_BINDING = (
    ROOT / "reports/taiji_n1_plan160_candidate_parent_preflight_binding_20261020.json"
)
FACE = ROOT / "plans/manifests/n1_plan160_v32_candidate_parent_seed130_20261020.json"
FACE_AUDIT = ROOT / "reports/taiji_n1_plan160_v32_candidate_parent_face_audit_20261020.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_plan160_v32_candidate_parent_face_checks_20261020.json"
ATTEMPT = ROOT / "reports/taiji_n1_plan160_candidate_parent_attempt_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan160_candidate_parent_progress_20261020.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan160_candidate_parent_result_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan160_seed130_candidate_parent_diagnostic_contract.py"

SEEDS = (11, 12, 13)
BRANCHES = ("unknown8_rebalanced", "length_mixed_parent")
ENDPOINTS = tuple((seed, branch) for seed in SEEDS for branch in BRANCHES)
LENGTHS = (8, 16, 32)
KINDS = ("lookup", "revision", "transition", "unknown")
ITEMS_PER_KIND = 40
QUERIES_PER_ENDPOINT = ITEMS_PER_KIND * len(KINDS) * len(LENGTHS)
TOTAL_QUERIES = QUERIES_PER_ENDPOINT * len(ENDPOINTS)
PRIOR_LINKED_CPU_SECONDS = 10_229.453125
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
    n159a = read_json(N159A_RESULT)
    n158_result = read_json(N158_REPORT)
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
        FACE,
        FACE_AUDIT,
        FACE_CHECKS,
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
        "schema": "n1-plan160-seed130-candidate-parent-source-audit-v1",
        "status": "complete" if complete else "stopped",
        "plan_sha256": PLAN_SHA256,
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
        "schema": "n1-plan160-seed130-candidate-parent-readiness-v1",
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
            "schema": "n1-plan160-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
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
            "schema": "n1-plan160-source-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_hashes": prepared,
            "can_promote": False,
        },
    )
    return readiness


def prepare_face() -> dict:
    if any(path.exists() for path in (FACE, FACE_AUDIT, FACE_CHECKS)):
        raise FileExistsError("PLAN-N1-160 seed130 face outputs are write-once")
    if not READINESS.is_file() or read_json(READINESS).get("complete") is not True:
        raise Refuse("PLAN-N1-160 readiness is required before face generation")
    if not PREFLIGHT.is_file() or not PREFLIGHT_BINDING.is_file():
        raise Refuse("PLAN-N1-160 six-bundle preflight is required before face generation")
    pf = read_json(PREFLIGHT)
    pfb = read_json(PREFLIGHT_BINDING)
    if (
        pf.get("complete") is not True
        or pf.get("independent_restore_count") != 6
        or pfb.get("preflight_sha256") != sha256(PREFLIGHT)
        or pfb.get("source_hashes") != pf.get("source_hashes")
    ):
        raise Refuse("PLAN-N1-160 preflight binding is stale")
    started = time.process_time()
    endpoints, sources, exclusions = source_context()
    ready_sources = dict(sources)
    ready_sources[str(SOURCE_AUDIT.resolve())] = sha256(SOURCE_AUDIT)
    if read_json(READINESS).get("source_hashes") != ready_sources:
        raise Refuse("PLAN-N1-160 readiness source map changed")
    face, generator = n153_builder.build_face(
        exclusions["excluded_entities"], exclusions["excluded_words"], seed=130
    )
    for item in face["items"]:
        item["id"] = item["id"].replace("v28-", "v32-", 1)
        item["parent_id"] = item["id"]
    contract = n153_builder.validate_face(
        face, exclusions["excluded_entities"], exclusions["excluded_words"]
    )
    manifest = {
        **face,
        "plan160_schema": "n1-plan160-v32-candidate-parent-face-v1",
        "plan_sha256": PLAN_SHA256,
        "seed": 130,
        "source_exclusion_summary": exclusions["exclusion_summary"],
        "prior_seed128_face_sha256": sha256(FACE128),
        "prior_seed129_face_sha256": sha256(FACE129),
        "total_queries": TOTAL_QUERIES,
        "can_promote": False,
    }
    write_new(FACE, manifest)
    face_sources = dict(ready_sources)
    for path in (READINESS, SOURCE_BINDING, PREFLIGHT, PREFLIGHT_BINDING):
        face_sources[str(Path(path).resolve())] = sha256(path)
    face_sources[str(FACE.resolve())] = sha256(FACE)
    audit = {
        "schema": "n1-plan160-v32-seed130-face-audit-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "source_hashes": face_sources,
        "source_hashes_unchanged": n150_builder.verify_hashes(face_sources),
        "source_exclusion_summary": exclusions["exclusion_summary"],
        "face_contract": contract,
        "generator_summary": generator,
        "cpu_seconds": round(time.process_time() - started, 6),
        "can_promote": False,
    }
    write_new(FACE_AUDIT, audit)
    checks = {
        "schema": "n1-plan160-v32-seed130-face-checks-v1",
        "status": "complete"
        if audit["source_hashes_unchanged"] and contract["accepted"]
        else "stopped",
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "face_audit_sha256": sha256(FACE_AUDIT),
        "face_contract": contract,
        "source_hashes_unchanged": audit["source_hashes_unchanged"],
        "total_queries": TOTAL_QUERIES,
        "can_promote": False,
    }
    write_new(FACE_CHECKS, checks)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-160 seed130 face source readiness and preparation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--prepare-face", action="store_true")
    args = parser.parse_args()
    result = source_readiness() if args.validate_only else prepare_face()
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
