"""N1-153A：在原N1-153 fresh face上完成可审计的逐题结果绑定。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_plan153_v28_fresh_context_length_face as builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_plan153_v28_fresh_context_length_confirmation as n153
from scripts.training import run_taiji_n1_v27_length_robust_training_after_n149 as n150

PLAN = ROOT / "plans/reference/PLAN-N1-153A_fresh_context_result_binding_recovery_20261020.md"
PLAN_SHA256 = "ab2c0040da74e63aea78226e7e542473fcd8c85412220e1d88d7cd3ca79cf185"
ORIGINAL_RESULT = n153.RESULT
ORIGINAL_RESULT_SHA256 = "cabd36441b8619875f6d17c70f0114f64c7986bb5abb9b1008071b07dd3be029"
N150_RESULT = n150.RESULT
PRIOR_QUERY_COUNT = 480
PRIOR_CPU_SECONDS = 113.78125
CPU_LIMIT_SECONDS = 3_600.0
ENDPOINTS = builder.ENDPOINTS
LENGTHS = builder.LENGTHS
KINDS = builder.KINDS
QUERIES_PER_ENDPOINT = 480
EXPECTED_NEW_QUERIES = QUERIES_PER_ENDPOINT * len(ENDPOINTS)
EXPECTED_CUMULATIVE_QUERIES = PRIOR_QUERY_COUNT + EXPECTED_NEW_QUERIES

READINESS = ROOT / "reports/taiji_n1_plan153a_fresh_context_readiness_20261020.json"
READINESS_ATTEMPT = ROOT / "reports/taiji_n1_plan153a_fresh_context_readiness_attempt_20261020.json"
SOURCE_BINDING = ROOT / "reports/taiji_n1_plan153a_fresh_context_source_binding_20261020.json"
PREFLIGHT = ROOT / "reports/taiji_n1_plan153a_fresh_context_checkpoint_preflight_20261020.json"
PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan153a_fresh_context_checkpoint_preflight_attempt_20261020.json"
)
PREFLIGHT_BINDING = (
    ROOT / "reports/taiji_n1_plan153a_fresh_context_checkpoint_preflight_binding_20261020.json"
)
ATTEMPT = ROOT / "reports/taiji_n1_plan153a_fresh_context_attempt_20261020.json"
RESULT = (
    ROOT / "reports/taiji_n1_plan153a_fresh_context_result_binding_recovery_result_20261020.json"
)
TEST = (
    ROOT / "tests/taiji_native/test_n1_plan153a_fresh_context_result_binding_recovery_contract.py"
)


class Refuse(RuntimeError):
    """N1-153A failed result, face, checkpoint or query合同不成立。"""


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


def _merge_hashes(target: dict[str, str], incoming: dict[str, str]) -> None:
    for raw_path, digest in incoming.items():
        path = str(Path(raw_path).resolve())
        old = target.get(path)
        if old is not None and old != digest:
            raise Refuse(f"conflicting N1-153A source hash: {path}")
        target[path] = digest


def verify_hashes(source_hashes: dict[str, str]) -> bool:
    return all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in source_hashes.items()
    )


def _validate_failed_n153(result: dict) -> dict:
    require(sha256(ORIGINAL_RESULT) == ORIGINAL_RESULT_SHA256, "N1-153 stopped result SHA changed")
    require(
        result.get("status") == "stopped"
        and result.get("complete") is False
        and result.get("query_count") == PRIOR_QUERY_COUNT
        and len(result.get("diagnostic_rows", [])) == 0
        and result.get("parameter_updates") == 0
        and result.get("source_hashes_unchanged") is True
        and result.get("can_promote") is False,
        "N1-153 prior failed attempt does not match the frozen one-endpoint stop receipt",
    )
    return result


def validate_exclusion_audit(
    audit: dict, exclusion_summary: dict, source_hashes: dict[str, str]
) -> None:
    for key, value in exclusion_summary.items():
        if audit.get(key) != value:
            raise Refuse(f"N1-153 source exclusion summary changed: {key}")
    if audit.get("source_hashes") != source_hashes:
        raise Refuse("N1-153 source audit hash map changed")


def _source_context() -> tuple[dict, dict, dict, dict, dict, dict, dict[str, str]]:
    n150_readiness, n150_preflight, _binding, bound, schedule = n150._verify_preflight()
    n150_result = read_json(n150.RESULT)
    previous = _validate_failed_n153(read_json(ORIGINAL_RESULT))
    n153_readiness = read_json(n153.READINESS)
    n153_readiness_binding = read_json(n153.SOURCE_BINDING)
    n153_preflight = read_json(n153.PREFLIGHT)
    n153_preflight_binding = read_json(n153.PREFLIGHT_BINDING)
    n151a_result = read_json(builder.N151A_RESULT)
    n152_result = read_json(builder.N152_RESULT)
    require(
        n153_readiness.get("complete") is True
        and n153_readiness.get("source_hashes_unchanged") is True
        and n153_readiness_binding.get("readiness_sha256") == sha256(n153.READINESS)
        and n153_readiness_binding.get("source_hashes") == n153_readiness.get("source_hashes")
        and n153_preflight.get("complete") is True
        and n153_preflight.get("independent_restore_count") == 6
        and n153_preflight.get("source_hashes_unchanged") is True
        and n153_preflight.get("model_queries") == 0
        and n153_preflight.get("parameter_updates") == 0
        and n153_preflight_binding.get("status") == "complete"
        and n153_preflight_binding.get("preflight_sha256") == sha256(n153.PREFLIGHT),
        "N1-153 readiness and 6/6 final restore preflight are not valid for continuation",
    )
    require(
        n153_preflight_binding.get("source_hashes") == n153_preflight.get("source_hashes"),
        "N1-153 prior preflight binding differs",
    )
    require(
        builder.n150_builder.verify_hashes(n153_preflight.get("source_hashes", {})),
        "N1-153 prior preflight source hashes changed",
    )
    require(
        abs(float(previous.get("cumulative_cpu_seconds", -1)) - PRIOR_CPU_SECONDS) < 1e-6,
        "N1-153 prior cumulative CPU does not match N1-153A plan",
    )
    require(
        n150_result.get("complete") is True
        and n150_result.get("training_sufficient_all_endpoints") is True,
        "N1-150 final models are not all train-sufficient",
    )
    require(
        n151a_result.get("complete") is True
        and n151a_result.get("repeated_train_only_queries") == 7_488,
        "N1-151A row recovery is incomplete",
    )
    require(
        n152_result.get("complete") is True and n152_result.get("source_rows") == 1_728,
        "N1-152 length analysis is incomplete",
    )
    face = read_json(builder.FACE)
    audit = read_json(builder.SOURCE_AUDIT)
    checks = read_json(builder.FACE_CHECKS)
    require(
        face.get("plan_sha256") == builder.PLAN_SHA256
        and checks.get("accepted") is True
        and checks.get("face_manifest_sha256") == sha256(builder.FACE)
        and audit.get("face_sha256") == checks.get("face_sha256"),
        "N1-153 one-shot face/source audit changed",
    )
    excluded_entities, excluded_words, exclusion_summary, inherited, _runtime = (
        builder.collect_n150_exclusions()
    )
    validate_exclusion_audit(audit, exclusion_summary, inherited)
    rebuilt_face, _summary = builder.build_face(excluded_entities, excluded_words)
    require(rebuilt_face == face, "N1-153 face cannot be reconstructed deterministically")
    sources = dict(inherited)
    paths = (
        PLAN,
        builder.PLAN,
        builder.N151A_PLAN,
        builder.N151_FAILURE,
        builder.N150_RESULT,
        builder.N151A_RESULT,
        builder.N152_RESULT,
        ORIGINAL_RESULT,
        n153.READINESS,
        n153.READINESS_ATTEMPT,
        n153.SOURCE_BINDING,
        n153.PREFLIGHT,
        n153.PREFLIGHT_ATTEMPT,
        n153.PREFLIGHT_BINDING,
        n153.ATTEMPT,
        builder.N150_CLOSEOUT
        if hasattr(builder, "N150_CLOSEOUT")
        else ROOT / "plans/reference/PLAN-N1-150_execution_closeout_20261020.md",
        ROOT / "plans/reference/PLAN-N1-152_execution_closeout_20261020.md",
        Path(n153.__file__).resolve(),
        Path(__file__).resolve(),
        TEST,
        builder.FACE,
        builder.FACE_CHECKS,
        builder.SOURCE_AUDIT,
        n150.PREFLIGHT,
        n150.PREFLIGHT_BINDING,
        n150.PROGRESS,
        ORIGINAL_RESULT,
    )
    for path in paths:
        path = Path(path).resolve()
        require(path.is_file(), f"N1-153A source missing: {path}")
        sources[str(path)] = sha256(path)
    for source_result in (n150_result, n151a_result, n152_result, previous):
        _merge_hashes(sources, source_result.get("new_checkpoint_hashes", {}))
        _merge_hashes(sources, source_result.get("source_hashes", {}))
    if not verify_hashes(sources):
        raise Refuse("N1-153A source/final checkpoint SHA changed")
    return (
        n150_readiness,
        n150_preflight,
        n150_result,
        previous,
        bound,
        schedule,
        dict(sorted(sources.items())),
    )


def validate_remaining_endpoints(rows: list[dict]) -> dict:
    expected = len(ENDPOINTS) * QUERIES_PER_ENDPOINT
    if len(rows) != expected:
        raise Refuse(f"N1-153A requires {expected} newly scored rows, got {len(rows)}")
    counts = Counter((int(row["seed"]), str(row["branch"])) for row in rows)
    if dict(counts) != {endpoint: QUERIES_PER_ENDPOINT for endpoint in ENDPOINTS}:
        raise Refuse(f"N1-153A per-endpoint query counts differ: {dict(counts)}")
    return {
        "endpoint_count": len(counts),
        "queries_per_endpoint": QUERIES_PER_ENDPOINT,
        "new_queries": len(rows),
        "complete": True,
    }


def _summary_for_endpoint(rows: list[dict], seed: int, branch: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if int(row["seed"]) == seed and str(row["branch"]) == branch:
            groups[(int(row["context_length"]), str(row["kind"]))].append(row)
    summary = []
    for length in LENGTHS:
        for kind in KINDS:
            selected = groups[(length, kind)]
            if len(selected) != 40:
                raise Refuse(
                    f"N1-153A endpoint summary row count differs: {seed}/{branch}/{length}/{kind}"
                )
            summary.append(
                {
                    "seed": seed,
                    "branch": branch,
                    "context_length": length,
                    "kind": kind,
                    "items": len(selected),
                    "free_exact": sum(row.get("free_exact") is True for row in selected),
                    "literal_unknown": sum(
                        row.get("output_hex") == builder.UNKNOWN.encode("utf-8").hex()
                        for row in selected
                    ),
                }
            )
    return summary


def _append_endpoint_rows(all_rows: list[dict], endpoint_rows: list[dict]) -> None:
    """Persist rows in memory before summarizing to prevent losing a finished endpoint."""
    all_rows.extend(endpoint_rows)


def source_readiness() -> dict:
    outputs = (
        READINESS,
        READINESS_ATTEMPT,
        SOURCE_BINDING,
        PREFLIGHT,
        PREFLIGHT_ATTEMPT,
        PREFLIGHT_BINDING,
        ATTEMPT,
        RESULT,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("N1-153A outputs are write-once")
    started = time.process_time()
    _n150_readiness, _n150_preflight, _n150_result, previous, _bound, _schedule, sources = (
        _source_context()
    )
    require(verify_hashes(sources), "N1-153A sources changed during no-query readiness")
    cpu = time.process_time() - started
    complete = cpu + PRIOR_CPU_SECONDS <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan153a-fresh-context-readiness-v1",
        "status": "ready_for_final_checkpoint_preflight" if complete else "stopped",
        "complete": complete,
        "plan_sha256": builder.PLAN_SHA256,
        "repair_plan_sha256": PLAN_SHA256,
        "original_n153_result_sha256": sha256(ORIGINAL_RESULT),
        "face_sha256": sha256(builder.FACE),
        "source_hashes": sources,
        "source_hashes_unchanged": True,
        "prior_query_count": previous["query_count"],
        "prior_cpu_seconds": previous["cpu_seconds"],
        "remaining_endpoint_count": len(ENDPOINTS),
        "queries_per_endpoint": QUERIES_PER_ENDPOINT,
        "planned_new_queries": EXPECTED_NEW_QUERIES,
        "cumulative_query_count_planned": EXPECTED_CUMULATIVE_QUERIES,
        "cpu_seconds": round(cpu, 6),
        "prior_cpu_seconds_including_failed_n153": PRIOR_CPU_SECONDS,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "can_promote": False,
        "summary_zh": "N1-153A同面恢复计划、source和六个final checkpoint哈希核验通过；0模型/checkpoint加载、0查询、0更新。",
    }
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan153a-readiness-attempt-v1",
            "status": "started",
            "repair_plan_sha256": PLAN_SHA256,
            "cpu_seconds": round(cpu, 6),
            "model_queries": 0,
            "parameter_updates": 0,
        },
    )
    write_new(READINESS, report)
    write_new(
        SOURCE_BINDING,
        {
            "schema": "n1-plan153a-readiness-binding-v1",
            "status": "complete" if complete else "stopped",
            "repair_plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_hashes": sources,
            "can_promote": False,
        },
    )
    return report


def _verify_readiness() -> tuple[dict, dict, dict[str, str]]:
    if not READINESS.is_file() or not SOURCE_BINDING.is_file():
        raise FileNotFoundError("N1-153A no-query readiness is required")
    report = read_json(READINESS)
    binding = read_json(SOURCE_BINDING)
    if (
        report.get("complete") is not True
        or report.get("repair_plan_sha256") != PLAN_SHA256
        or binding.get("status") != "complete"
        or binding.get("repair_plan_sha256") != PLAN_SHA256
        or binding.get("readiness_sha256") != sha256(READINESS)
        or binding.get("source_hashes") != report.get("source_hashes")
    ):
        raise Refuse("N1-153A readiness/source binding is stale")
    _ctx = _source_context()
    sources = _ctx[-1]
    if sources != report.get("source_hashes") or not verify_hashes(sources):
        raise Refuse("N1-153A source/checkpoint hash changed")
    return report, binding, sources


def checkpoint_preflight() -> dict:
    if any(path.exists() for path in (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING)):
        raise FileExistsError("N1-153A final restore preflight is write-once")
    started = time.process_time()
    readiness, _binding, sources = _verify_readiness()
    validation_cpu = time.process_time() - started
    remaining = (
        CPU_LIMIT_SECONDS - PRIOR_CPU_SECONDS - float(readiness["cpu_seconds"]) - validation_cpu
    )
    if remaining <= 0:
        raise TimeoutError("N1-153A CPU ceiling exhausted before final restore")
    budget = n132.CpuBudget(remaining)
    budget.guard(reserve=30)
    write_new(
        PREFLIGHT_ATTEMPT,
        {
            "schema": "n1-plan153a-checkpoint-preflight-attempt-v1",
            "status": "started",
            "repair_plan_sha256": PLAN_SHA256,
            "endpoint_count": 6,
            "model_queries": 0,
            "parameter_updates": 0,
        },
    )
    n150_result = read_json(N150_RESULT)
    models = {(int(row["seed"]), str(row["branch"])): row for row in n150_result["models"]}
    restores = []
    errors = []
    try:
        for seed, branch in ENDPOINTS:
            budget.guard(reserve=10)
            model = models[(seed, branch)]
            restore = n150._independent_restore(Path(model["final_checkpoint_path"]), budget)
            if restore.get("fingerprint") != model.get("final_bundle_fingerprint") or restore.get(
                "sha256"
            ) != model.get("final_checkpoint_sha256"):
                raise Refuse(f"N1-153A final checkpoint restore mismatch: {seed}/{branch}")
            restores.append(restore)
        if not verify_hashes(sources):
            raise Refuse("N1-153A sources changed during checkpoint restore")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = time.process_time() - started + budget.children
    prior_cpu = PRIOR_CPU_SECONDS + float(readiness["cpu_seconds"])
    cumulative = prior_cpu + cpu
    complete = not errors and len(restores) == 6 and cumulative <= CPU_LIMIT_SECONDS
    all_sources = dict(sources)
    all_sources[str(READINESS.resolve())] = sha256(READINESS)
    all_sources[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
    all_sources[str(PREFLIGHT_ATTEMPT.resolve())] = sha256(PREFLIGHT_ATTEMPT)
    report = {
        "schema": "n1-plan153a-checkpoint-preflight-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "repair_plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "preflight_attempt_sha256": sha256(PREFLIGHT_ATTEMPT),
        "source_hashes": all_sources,
        "source_hashes_unchanged": verify_hashes(all_sources),
        "independent_restores": restores,
        "independent_restore_count": len(restores),
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "prior_cpu_seconds_including_failed_n153": round(PRIOR_CPU_SECONDS, 6),
        "cpu_seconds": round(cpu, 6),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": "N1-153A复用face的六个final checkpoint独立恢复6/6；0模型查询、0更新。"
        if complete
        else "N1-153A final恢复前置停止，不查询、不重试。",
    }
    write_new(PREFLIGHT, report)
    write_new(
        PREFLIGHT_BINDING,
        {
            "schema": "n1-plan153a-checkpoint-preflight-binding-v1",
            "status": "complete" if complete else "stopped",
            "repair_plan_sha256": PLAN_SHA256,
            "preflight_sha256": sha256(PREFLIGHT),
            "source_hashes": all_sources,
            "can_promote": False,
        },
    )
    return report


def _verify_preflight() -> tuple[dict, dict, dict[str, str]]:
    if not all(
        path.is_file() for path in (READINESS, SOURCE_BINDING, PREFLIGHT, PREFLIGHT_BINDING)
    ):
        raise FileNotFoundError("N1-153A readiness and final restore preflight are required")
    readiness = read_json(READINESS)
    binding = read_json(SOURCE_BINDING)
    preflight = read_json(PREFLIGHT)
    preflight_binding = read_json(PREFLIGHT_BINDING)
    if (
        readiness.get("complete") is not True
        or readiness.get("repair_plan_sha256") != PLAN_SHA256
        or binding.get("status") != "complete"
        or binding.get("readiness_sha256") != sha256(READINESS)
        or preflight.get("complete") is not True
        or preflight.get("independent_restore_count") != 6
        or preflight.get("repair_plan_sha256") != PLAN_SHA256
        or preflight.get("source_hashes_unchanged") is not True
        or preflight.get("model_queries") != 0
        or preflight.get("parameter_updates") != 0
        or preflight_binding.get("status") != "complete"
        or preflight_binding.get("repair_plan_sha256") != PLAN_SHA256
        or preflight_binding.get("preflight_sha256") != sha256(PREFLIGHT)
    ):
        raise Refuse("N1-153A readiness/preflight is stale or incomplete")
    sources = preflight.get("source_hashes", {})
    if not verify_hashes(sources):
        raise Refuse("N1-153A source/checkpoint hashes changed before continuation")
    _source_context()
    return readiness, preflight, sources


def _summarize_endpoint(rows: list[dict], seed: int, branch: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if int(row["seed"]) == seed and str(row["branch"]) == branch:
            groups[(int(row["context_length"]), str(row["kind"]))].append(row)
    summary = []
    for length in LENGTHS:
        for kind in KINDS:
            items = groups[(length, kind)]
            if len(items) != 40:
                raise Refuse(f"N1-153A endpoint summary mismatch: {seed}/{branch}/{length}/{kind}")
            summary.append(
                {
                    "seed": seed,
                    "branch": branch,
                    "context_length": length,
                    "kind": kind,
                    "items": 40,
                    "free_exact": sum(x.get("free_exact") is True for x in items),
                    "literal_unknown": sum(
                        x.get("output_hex") == builder.UNKNOWN.encode("utf-8").hex() for x in items
                    ),
                }
            )
    return summary


def execute_approved() -> dict:
    started = time.process_time()
    readiness, preflight, source_hashes = _verify_preflight()
    validation_cpu = time.process_time() - started
    previous_result = read_json(ORIGINAL_RESULT)
    remaining = CPU_LIMIT_SECONDS - float(preflight["cumulative_cpu_seconds"]) - validation_cpu
    if remaining <= 0:
        raise TimeoutError("N1-153A CPU ceiling exhausted before query")
    if any(path.exists() for path in (ATTEMPT, RESULT)):
        raise FileExistsError("N1-153A continuation is one-shot")
    budget = n132.CpuBudget(remaining)
    input_hashes = dict(source_hashes)
    for path in (READINESS, SOURCE_BINDING, PREFLIGHT, PREFLIGHT_BINDING):
        input_hashes[str(path.resolve())] = sha256(path)
    if not verify_hashes(input_hashes):
        raise Refuse("N1-153A source chain differs before query")
    attempt = {
        "schema": "n1-plan153a-attempt-v1",
        "status": "started",
        "repair_plan_sha256": PLAN_SHA256,
        "original_n153_result_sha256": sha256(ORIGINAL_RESULT),
        "face_sha256": sha256(builder.FACE),
        "previous_queries": PRIOR_QUERY_COUNT,
        "planned_new_queries": EXPECTED_NEW_QUERIES,
        "cumulative_queries_planned": EXPECTED_CUMULATIVE_QUERIES,
        "parameter_updates": 0,
        "can_promote": False,
    }
    write_new(ATTEMPT, attempt)
    n150_result = read_json(N150_RESULT)
    models = {(int(row["seed"]), str(row["branch"])): row for row in n150_result["models"]}
    face = read_json(builder.FACE)
    new_rows = []
    endpoint_summaries = []
    errors = []
    query_count = 0
    try:
        for seed, branch in ENDPOINTS:
            budget.guard(reserve=30)
            model_record = models[(seed, branch)]
            checkpoint = Path(model_record["final_checkpoint_path"]).resolve()
            if sha256(checkpoint) != model_record["final_checkpoint_sha256"]:
                raise Refuse(f"N1-153A checkpoint changed: {seed}/{branch}")
            model, optimizer, payload, fingerprint = n150._restore_n150_checkpoint(checkpoint)
            if fingerprint != model_record["final_bundle_fingerprint"]:
                raise Refuse(f"N1-153A final bundle fingerprint mismatch: {seed}/{branch}")
            before = n150._fingerprint(model, optimizer)
            endpoint_rows = []
            try:
                for base in face["items"]:
                    for length in LENGTHS:
                        budget.guard(reserve=5)
                        item = n153._face_from_manifest_item(base, length)
                        session = f"n1-plan153a-{seed}-{branch}-{item['parent_id']}-l{length}"
                        if session in model._observations:
                            raise Refuse(f"N1-153A session collision: {session}")
                        query_count += 1
                        try:
                            row = n153._measure_diagnostic(model, item, session)
                        finally:
                            model._observations.pop(session, None)
                        durable = {"seed": seed, "branch": branch, "event": "diagnostic_row", **row}
                        endpoint_rows.append(durable)
                        new_rows.append(durable)
                if len(endpoint_rows) != QUERIES_PER_ENDPOINT:
                    raise Refuse(f"N1-153A query count differs: {seed}/{branch}")
                endpoint_summaries.extend(_summarize_endpoint(endpoint_rows, seed, branch))
                if n150._fingerprint(model, optimizer) != before:
                    raise Refuse(f"N1-153A scoring changed full bundle: {seed}/{branch}")
            finally:
                del model, optimizer, payload
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    input_hashes[str(ATTEMPT.resolve())] = sha256(ATTEMPT)
    source_unchanged = verify_hashes(input_hashes)
    if not source_unchanged:
        errors.append("N1-153A source/checkpoint SHA changed during query")
    stage_cpu = time.process_time() - started + budget.children
    cumulative = float(preflight["cumulative_cpu_seconds"]) + stage_cpu
    new_coverage = Counter((int(row["seed"]), str(row["branch"])) for row in new_rows)
    expected_counts = {endpoint: QUERIES_PER_ENDPOINT for endpoint in ENDPOINTS}
    complete = (
        not errors
        and query_count == EXPECTED_NEW_QUERIES
        and len(new_rows) == EXPECTED_NEW_QUERIES
        and dict(new_coverage) == expected_counts
        and len(endpoint_summaries) == 72
        and source_unchanged
        and cumulative <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan153a-fresh-context-result-binding-recovery-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "base_plan_sha256": builder.PLAN_SHA256,
        "repair_plan_sha256": PLAN_SHA256,
        "original_n153_result_sha256": sha256(ORIGINAL_RESULT),
        "n150_result_sha256": sha256(N150_RESULT),
        "readiness_sha256": sha256(READINESS),
        "preflight_sha256": sha256(PREFLIGHT),
        "attempt_sha256": sha256(ATTEMPT),
        "source_hashes": input_hashes,
        "source_hashes_unchanged": source_unchanged,
        "previous_failed_query_count": PRIOR_QUERY_COUNT,
        "previous_failed_rows_saved": len(previous_result.get("diagnostic_rows", [])),
        "new_query_count": query_count,
        "new_queries_per_endpoint": QUERIES_PER_ENDPOINT,
        "cumulative_query_count": PRIOR_QUERY_COUNT + query_count,
        "diagnostic_rows": new_rows,
        "summary_by_endpoint_length_kind": endpoint_summaries,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "prior_cpu_seconds_including_n153_failure": PRIOR_CPU_SECONDS,
        "cpu_seconds": round(stage_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": f"N1-153A结果绑定修复{('完整' if complete else '停止')}：{query_count}/2,880条修复诊断查询；计入N1-153失败尝试后累计{PRIOR_QUERY_COUNT + query_count}/3,360条，0训练/dev/holdout/confirmation，can_promote=false。",
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-153A corrected fresh context scorer")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--checkpoint-preflight", action="store_true")
    modes.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.validate_only:
        result = source_readiness()
    elif args.checkpoint_preflight:
        result = checkpoint_preflight()
    else:
        result = execute_approved()
    keys = (
        "status",
        "complete",
        "cpu_seconds",
        "cumulative_cpu_seconds",
        "endpoint_count",
        "new_query_count",
        "cumulative_query_count",
        "independent_restore_count",
        "parameter_updates",
        "dev_queries",
        "holdout_queries",
        "confirmation_queries",
        "can_promote",
        "errors",
        "summary_zh",
    )
    print(json.dumps({key: result.get(key) for key in keys}, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
