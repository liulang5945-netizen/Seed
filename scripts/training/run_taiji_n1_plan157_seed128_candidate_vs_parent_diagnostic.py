"""执行PLAN-N1-157 seed128 candidate-vs-parent fresh诊断，不训练、不晋级。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_plan157_seed128_candidate_vs_parent_face as builder
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_plan153_v28_fresh_context_length_confirmation as n153
from scripts.training import run_taiji_n1_plan155_short_unknown_context_length_training as n155
from scripts.training import run_taiji_n1_v27_length_robust_training_after_n149 as n150
from scripts.training import seal_ast  # noqa: E402

PLAN_SHA256 = builder.PLAN_SHA256
CPU_LIMIT_SECONDS = builder.CPU_LIMIT_SECONDS
LINKED_CPU_LIMIT_SECONDS = builder.LINKED_CPU_LIMIT_SECONDS
PRIOR_LINKED_CPU_SECONDS = builder.PRIOR_LINKED_CPU_SECONDS
ENDPOINTS = builder.ENDPOINTS
LENGTHS = builder.LENGTHS
KINDS = builder.KINDS
QUERIES_PER_ENDPOINT = builder.QUERIES_PER_ENDPOINT
TOTAL_QUERIES = builder.TOTAL_QUERIES
ITEMS_PER_KIND = builder.ITEMS_PER_KIND
UNKNOWN_HEX = builder.UNKNOWN.encode("utf-8").hex()

READINESS = builder.READINESS
SOURCE_AUDIT = builder.SOURCE_AUDIT
SOURCE_BINDING = builder.SOURCE_BINDING
PREFLIGHT = builder.PREFLIGHT
PREFLIGHT_ATTEMPT = builder.PREFLIGHT_ATTEMPT
PREFLIGHT_BINDING = builder.PREFLIGHT_BINDING
FACE = builder.FACE
FACE_AUDIT = builder.FACE_AUDIT
FACE_CHECKS = builder.FACE_CHECKS
ATTEMPT = builder.ATTEMPT
PROGRESS = builder.PROGRESS
RESULT = builder.RESULT
UNKNOWN_HEX = builder.UNKNOWN.encode("utf-8").hex()


class Refuse(RuntimeError):
    """PLAN-N1-157来源、bundle或diagnostic协议失败。"""


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


def _write_jsonl(stream, row: dict) -> None:
    json.dump(row, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")


def _fingerprint(model, optimizer) -> str:
    return n132.e11.scorer.paired.fingerprint(model.bundle(optimizer))


def _restore_endpoint(role: str, path: Path):
    if role == "unknown8_rebalanced":
        return n155._restore_checkpoint(path)
    if role == "length_mixed_parent":
        return n150._restore_n150_checkpoint(path)
    raise Refuse(f"unknown PLAN-N1-157 endpoint role: {role}")


def _restore_worker(role: str, path: Path) -> dict:
    started = time.process_time()
    model, optimizer, payload, fingerprint = _restore_endpoint(role, path)
    expected = payload.get("bundle_fingerprint", payload.get("final_bundle_fingerprint"))
    report = {
        "complete": True,
        "equal": fingerprint == expected,
        "fingerprint": fingerprint,
        "bundle_fingerprint": expected,
        "role": role,
        "seed": int(payload["seed"]),
        "branch": payload.get("branch"),
        "history_count": len(model._observations),
        "optimizer_parameters": len(optimizer.state),
        "cpu_seconds": round(time.process_time() - started, 6),
    }
    del model, optimizer, payload
    return report


def _independent_restore(endpoint: dict, budget: n132.CpuBudget) -> dict:
    budget.guard(reserve=5)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    timeout = min(90, max(1.0, budget.limit - budget.used() - 1.0))
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--restore-worker",
            endpoint["branch"],
            endpoint["checkpoint_path"],
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
        env=env,
    )
    report = json.loads(process.stdout)
    budget.children += float(report["cpu_seconds"])
    budget.guard()
    if report.get("complete") is not True or report.get("equal") is not True:
        raise Refuse(
            f"independent PLAN-N1-157 bundle restore mismatch: {endpoint['checkpoint_path']}"
        )
    return {"path": endpoint["checkpoint_path"], "sha256": endpoint["checkpoint_sha256"], **report}


def _verify_readiness() -> tuple[dict, dict[str, str]]:
    required = (READINESS, SOURCE_AUDIT, SOURCE_BINDING)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("PLAN-N1-157 source readiness is required")
    readiness = read_json(READINESS)
    audit = read_json(SOURCE_AUDIT)
    binding = read_json(SOURCE_BINDING)
    if (
        readiness.get("complete") is not True
        or readiness.get("status") != "ready_for_six_bundle_preflight"
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or readiness.get("prior_result_sha256") != builder.PRIOR_RESULT_SHA256
        or readiness.get("prior_consumed_queries") != builder.PRIOR_QUERIES
        or readiness.get("remaining_queries") != TOTAL_QUERIES - builder.PRIOR_QUERIES
        or readiness.get("prior_n157a_readiness_sha256") != sha256(builder.PRIOR_N157A_READINESS)
        or readiness.get("binding_failure_receipt_sha256") != builder.PRIOR_BINDING_FAILURE_SHA256
        or readiness.get("remaining_queries") != TOTAL_QUERIES - builder.PRIOR_QUERIES
        or audit.get("status") != "complete"
        or audit.get("plan_sha256") != PLAN_SHA256
        or audit.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or audit.get("prior_result_sha256") != builder.PRIOR_RESULT_SHA256
        or audit.get("prior_consumed_queries") != builder.PRIOR_QUERIES
        or audit.get("remaining_queries") != TOTAL_QUERIES - builder.PRIOR_QUERIES
        or audit.get("prior_n157a_readiness_sha256") != sha256(builder.PRIOR_N157A_READINESS)
        or audit.get("binding_failure_receipt_sha256") != builder.PRIOR_BINDING_FAILURE_SHA256
        or audit.get("binding_failure_cpu_reserve_seconds")
        != builder.BINDING_FAILURE_CPU_RESERVE_SECONDS
        or binding.get("status") != "complete"
        or binding.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or binding.get("prior_result_sha256") != builder.PRIOR_RESULT_SHA256
        or binding.get("prior_consumed_queries") != builder.PRIOR_QUERIES
        or binding.get("remaining_queries") != TOTAL_QUERIES - builder.PRIOR_QUERIES
        or binding.get("prior_n157a_readiness_sha256") != sha256(builder.PRIOR_N157A_READINESS)
        or binding.get("binding_failure_receipt_sha256") != builder.PRIOR_BINDING_FAILURE_SHA256
        or binding.get("readiness_sha256") != sha256(READINESS)
        or binding.get("source_hashes") != readiness.get("source_hashes")
    ):
        raise Refuse("PLAN-N1-157 readiness/source binding is stale")
    if not n150_builder.verify_hashes(readiness.get("source_hashes", {})):
        raise Refuse("PLAN-N1-157 bound source/checkpoint hashes changed")
    endpoints, sources, _exclusions = builder.source_context()
    if len(endpoints) != 6 or not set(sources.items()).issubset(
        set(readiness["source_hashes"].items())
    ):
        raise Refuse("PLAN-N1-157 fixed endpoint/source set changed")
    return readiness, dict(readiness["source_hashes"])


def checkpoint_preflight() -> dict:
    if any(path.exists() for path in (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING)):
        raise FileExistsError("PLAN-N1-157 checkpoint preflight is write-once")
    started = time.process_time()
    readiness, sources = _verify_readiness()
    validation_cpu = time.process_time() - started
    remaining = CPU_LIMIT_SECONDS - float(readiness["cpu_seconds"]) - validation_cpu
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-157 CPU budget exhausted during readiness validation")
    budget = n132.CpuBudget(remaining)
    budget.guard(reserve=45)
    write_new(
        PREFLIGHT_ATTEMPT,
        {
            "schema": "n1-plan157a2-candidate-parent-preflight-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "correction_addendum_sha256": builder.CORRECTION_SHA256,
            "prior_result_sha256": builder.PRIOR_RESULT_SHA256,
            "prior_consumed_queries": builder.PRIOR_QUERIES,
            "remaining_queries": TOTAL_QUERIES - builder.PRIOR_QUERIES,
            "binding_failure_receipt_sha256": builder.PRIOR_BINDING_FAILURE_SHA256,
            "readiness_sha256": sha256(READINESS),
            "endpoint_count": len(ENDPOINTS),
            "independent_restores_planned": len(ENDPOINTS),
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    endpoints, _sources, _exclusions = builder.source_context()
    source_hashes = dict(sources)
    source_hashes[str(PREFLIGHT_ATTEMPT.resolve())] = sha256(PREFLIGHT_ATTEMPT)
    restores = []
    errors = []
    try:
        for endpoint in endpoints:
            restores.append(_independent_restore(endpoint, budget))
        if len(restores) != 6:
            raise Refuse("PLAN-N1-157 requires six independent full-bundle restores")
        if not n150_builder.verify_hashes(source_hashes):
            raise Refuse("PLAN-N1-157 preflight source hashes changed")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = time.process_time() - started + budget.children
    stage_cumulative = float(readiness["cpu_seconds"]) + cpu
    linked_cumulative = PRIOR_LINKED_CPU_SECONDS + stage_cumulative
    complete = (
        not errors
        and len(restores) == 6
        and stage_cumulative <= CPU_LIMIT_SECONDS
        and linked_cumulative <= LINKED_CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan157a2-candidate-parent-preflight-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": builder.CORRECTION_SHA256,
        "prior_result_sha256": builder.PRIOR_RESULT_SHA256,
        "prior_consumed_queries": builder.PRIOR_QUERIES,
        "remaining_queries": TOTAL_QUERIES - builder.PRIOR_QUERIES,
        "prior_n157a_readiness_sha256": sha256(builder.PRIOR_N157A_READINESS),
        "binding_failure_receipt_sha256": builder.PRIOR_BINDING_FAILURE_SHA256,
        "readiness_sha256": sha256(READINESS),
        "preflight_attempt_sha256": sha256(PREFLIGHT_ATTEMPT),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": n150_builder.verify_hashes(source_hashes),
        "independent_restores": restores,
        "independent_restore_count": len(restores),
        "model_queries": 0,
        "parameter_updates": 0,
        "cpu_seconds": round(cpu, 6),
        "readiness_cpu_seconds": float(readiness["cpu_seconds"]),
        "cumulative_plan_cpu_seconds": round(stage_cumulative, 6),
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cumulative_linked_cpu_seconds": round(linked_cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
    }
    write_new(PREFLIGHT, report)
    write_new(
        PREFLIGHT_BINDING,
        {
            "schema": "n1-plan157a2-candidate-parent-preflight-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "correction_addendum_sha256": builder.CORRECTION_SHA256,
            "prior_result_sha256": builder.PRIOR_RESULT_SHA256,
            "prior_consumed_queries": builder.PRIOR_QUERIES,
            "remaining_queries": TOTAL_QUERIES - builder.PRIOR_QUERIES,
            "prior_n157a_readiness_sha256": sha256(builder.PRIOR_N157A_READINESS),
            "binding_failure_receipt_sha256": builder.PRIOR_BINDING_FAILURE_SHA256,
            "preflight_sha256": sha256(PREFLIGHT),
            "source_hashes": source_hashes,
            "can_promote": False,
        },
    )
    return report


def _verify_preflight_and_face() -> tuple[dict, dict, list[dict], dict[str, str]]:
    required = (
        READINESS,
        SOURCE_AUDIT,
        SOURCE_BINDING,
        PREFLIGHT,
        PREFLIGHT_BINDING,
        FACE,
        FACE_AUDIT,
        FACE_CHECKS,
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("PLAN-N1-157 readiness, preflight and prepared face are required")
    readiness = read_json(READINESS)
    preflight = read_json(PREFLIGHT)
    preflight_binding = read_json(PREFLIGHT_BINDING)
    face = read_json(FACE)
    face_audit = read_json(FACE_AUDIT)
    face_checks = read_json(FACE_CHECKS)
    if (
        readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or readiness.get("prior_result_sha256") != builder.PRIOR_RESULT_SHA256
        or readiness.get("prior_consumed_queries") != builder.PRIOR_QUERIES
        or readiness.get("prior_n157a_readiness_sha256") != sha256(builder.PRIOR_N157A_READINESS)
        or readiness.get("binding_failure_receipt_sha256") != builder.PRIOR_BINDING_FAILURE_SHA256
        or preflight.get("complete") is not True
        or preflight.get("plan_sha256") != PLAN_SHA256
        or preflight.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or preflight.get("prior_result_sha256") != builder.PRIOR_RESULT_SHA256
        or preflight.get("prior_consumed_queries") != builder.PRIOR_QUERIES
        or preflight.get("prior_n157a_readiness_sha256") != sha256(builder.PRIOR_N157A_READINESS)
        or preflight.get("binding_failure_receipt_sha256") != builder.PRIOR_BINDING_FAILURE_SHA256
        or preflight_binding.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or preflight_binding.get("prior_result_sha256") != builder.PRIOR_RESULT_SHA256
        or preflight_binding.get("prior_consumed_queries") != builder.PRIOR_QUERIES
        or preflight_binding.get("prior_n157a_readiness_sha256")
        != sha256(builder.PRIOR_N157A_READINESS)
        or preflight_binding.get("binding_failure_receipt_sha256")
        != builder.PRIOR_BINDING_FAILURE_SHA256
        or preflight.get("independent_restore_count") != 6
        or preflight.get("source_hashes_unchanged") is not True
        or preflight_binding.get("status") != "complete"
        or preflight_binding.get("preflight_sha256") != sha256(PREFLIGHT)
        or preflight_binding.get("source_hashes") != preflight.get("source_hashes")
        or face.get("plan157_schema") != "n1-plan157-v30-candidate-vs-parent-face-v1"
        or face.get("plan_sha256") != PLAN_SHA256
        or face.get("total_queries") != TOTAL_QUERIES
        or face_audit.get("status") != "complete"
        or not seal_ast.receipt_binding_matches(face_audit.get("face_sha256"), FACE)
        or face_audit.get("source_hashes_unchanged") is not True
        or face_checks.get("status") != "complete"
        or not seal_ast.receipt_binding_matches(face_checks.get("face_sha256"), FACE)
        or face_checks.get("face_audit_sha256") != sha256(FACE_AUDIT)
        or face_checks.get("total_queries") != TOTAL_QUERIES
    ):
        raise Refuse("PLAN-N1-157 preflight/face binding is stale or incomplete")
    source_hashes = dict(preflight["source_hashes"])
    for path in (
        READINESS,
        SOURCE_AUDIT,
        SOURCE_BINDING,
        PREFLIGHT,
        PREFLIGHT_BINDING,
        FACE,
        FACE_AUDIT,
        FACE_CHECKS,
    ):
        source_hashes[str(Path(path).resolve())] = sha256(path)
    if not n150_builder.verify_hashes(source_hashes):
        raise Refuse("PLAN-N1-157 source/preflight/face hashes changed")
    endpoints, _sources, _exclusions = builder.source_context()
    return readiness, preflight, endpoints, source_hashes


def _fresh_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (int(row["seed"]), str(row["branch"]), int(row["context_length"]), str(row["kind"]))
        ].append(row)
    summary = []
    endpoint_keys = sorted({(int(row["seed"]), str(row["branch"])) for row in rows})
    for seed, branch in endpoint_keys:
        for length in LENGTHS:
            for kind in KINDS:
                items = grouped[(seed, branch, length, kind)]
                if len(items) != ITEMS_PER_KIND:
                    raise Refuse(
                        f"PLAN-N1-157 fresh score stratum mismatch: {seed}/{branch}/{length}/{kind}"
                    )
                summary.append(
                    {
                        "seed": seed,
                        "branch": branch,
                        "context_length": length,
                        "kind": kind,
                        "items": len(items),
                        "free_exact": sum(item.get("free_exact") is True for item in items),
                        "literal_unknown": sum(
                            item.get("output_hex") == UNKNOWN_HEX for item in items
                        ),
                    }
                )
    return summary


def execute_approved() -> dict:
    if any(path.exists() for path in (ATTEMPT, PROGRESS, RESULT)):
        raise FileExistsError("PLAN-N1-157A2 continuation attempt is write-once")
    started = time.process_time()
    readiness, preflight, endpoints, base_sources = _verify_preflight_and_face()
    validation_cpu = time.process_time() - started
    prior_plan157a_cpu = float(readiness["cpu_seconds"]) + float(preflight["cpu_seconds"])
    remaining = CPU_LIMIT_SECONDS - prior_plan157a_cpu - validation_cpu
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-157A2 CPU ceiling exhausted before continuation")
    budget = n132.CpuBudget(remaining)
    if PRIOR_LINKED_CPU_SECONDS + prior_plan157a_cpu + validation_cpu >= LINKED_CPU_LIMIT_SECONDS:
        raise TimeoutError("PLAN-N1-157A2 linked CPU ceiling exhausted before continuation")
    prior_result = read_json(builder.PRIOR_RESULT)
    prior_rows = list(prior_result.get("fresh_diagnostic_rows", []))
    prior_endpoint_keys = {(int(row["seed"]), str(row["branch"])) for row in prior_rows}
    if len(prior_rows) != builder.PRIOR_QUERIES or prior_endpoint_keys != {
        (11, "unknown8_rebalanced")
    }:
        raise Refuse("PLAN-N1-157A2 partial face rows are not exactly the sealed first endpoint")
    prior_summaries = _fresh_summary(prior_rows)
    attempt = {
        "schema": "n1-plan157a2-candidate-parent-attempt-v1",
        "status": "started",
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": builder.CORRECTION_SHA256,
        "prior_result_sha256": builder.PRIOR_RESULT_SHA256,
        "prior_attempt_sha256": builder.PRIOR_ATTEMPT_SHA256,
        "prior_progress_sha256": builder.PRIOR_PROGRESS_SHA256,
        "readiness_sha256": sha256(READINESS),
        "preflight_sha256": sha256(PREFLIGHT),
        "face_sha256": sha256(FACE),
        "planned_endpoints": len(ENDPOINTS),
        "prior_completed_endpoint": {"seed": 11, "branch": "unknown8_rebalanced"},
        "prior_consumed_queries": builder.PRIOR_QUERIES,
        "planned_new_queries": TOTAL_QUERIES - builder.PRIOR_QUERIES,
        "planned_total_face_queries": TOTAL_QUERIES,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
    }
    write_new(ATTEMPT, attempt)
    query_count = 0
    model_load_count = 0
    rows = prior_rows.copy()
    summaries = prior_summaries.copy()
    errors = []
    progress = PROGRESS.open("x", encoding="utf-8", newline="\n")
    try:
        face = read_json(FACE)
        for endpoint in endpoints:
            endpoint_key = (int(endpoint["seed"]), str(endpoint["branch"]))
            if endpoint_key in prior_endpoint_keys:
                continue
            budget.guard(reserve=30)
            checkpoint = Path(endpoint["checkpoint_path"]).resolve()
            if sha256(checkpoint) != endpoint["checkpoint_sha256"]:
                raise Refuse(
                    f"PLAN-N1-157 endpoint checkpoint changed: {endpoint['seed']}/{endpoint['branch']}"
                )
            model, optimizer, payload, fingerprint = _restore_endpoint(
                endpoint["branch"], checkpoint
            )
            model_load_count += 1
            if fingerprint != endpoint["bundle_fingerprint"]:
                raise Refuse(
                    f"PLAN-N1-157 endpoint bundle fingerprint mismatch: {endpoint['seed']}/{endpoint['branch']}"
                )
            before = _fingerprint(model, optimizer)
            endpoint_rows = []
            for base_item in face["items"]:
                for length in LENGTHS:
                    budget.guard(reserve=5)
                    item = n153._face_from_manifest_item(base_item, length)
                    session = f"n1-plan157a2-fresh-{endpoint['seed']}-{endpoint['branch']}-{item['parent_id']}-l{length}"
                    if session in model._observations:
                        raise Refuse(f"PLAN-N1-157A2 session collision: {session}")
                    query_count += 1
                    try:
                        scored = n155._measure_fresh_diagnostic(model, item, session)
                    finally:
                        model._observations.pop(session, None)
                    record = {
                        "seed": int(endpoint["seed"]),
                        "branch": str(endpoint["branch"]),
                        "model_role": (
                            "candidate" if endpoint["branch"] == "unknown8_rebalanced" else "parent"
                        ),
                        "event": "fresh_diagnostic_row",
                        **scored,
                    }
                    endpoint_rows.append(record)
                    rows.append(record)
                    _write_jsonl(progress, record)
                    if query_count % 64 == 0:
                        progress.flush()
            if len(endpoint_rows) != QUERIES_PER_ENDPOINT:
                raise Refuse(
                    f"PLAN-N1-157A2 endpoint query denominator differs: {endpoint['seed']}/{endpoint['branch']}"
                )
            summaries.extend(_fresh_summary(endpoint_rows))
            if _fingerprint(model, optimizer) != before:
                raise Refuse(
                    f"PLAN-N1-157A2 diagnostic changed checkpoint bundle: {endpoint['seed']}/{endpoint['branch']}"
                )
            del model, optimizer, payload
        progress.flush()
        progress.close()
        progress = None
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if progress is not None:
            try:
                progress.flush()
                progress.close()
            except Exception as exc:
                errors.append(f"progress close: {type(exc).__name__}: {exc}")

    progress_integrity = {"complete": False, "line_count": 0, "malformed_rows": 0, "error": None}
    try:
        count = 0
        malformed = []
        for number, line in enumerate(PROGRESS.open(encoding="utf-8"), 1):
            count += 1
            try:
                json.loads(line)
            except Exception as exc:
                malformed.append({"line": number, "error": type(exc).__name__})
        if (
            malformed
            or count != query_count
            or len(rows) != builder.PRIOR_QUERIES + query_count
            or query_count > TOTAL_QUERIES - builder.PRIOR_QUERIES
        ):
            raise Refuse(
                "PLAN-N1-157A2 continuation progress ledger mismatch: "
                f"{count}/{len(rows)}/{query_count}"
            )
        progress_integrity = {
            "complete": True,
            "line_count": count,
            "malformed_rows": 0,
            "progress_sha256": sha256(PROGRESS),
        }
    except BaseException as exc:
        errors.append(f"progress integrity: {type(exc).__name__}: {exc}")
        progress_integrity["error"] = f"{type(exc).__name__}: {exc}"

    source_hashes = dict(base_sources)
    source_hashes[str(ATTEMPT.resolve())] = sha256(ATTEMPT)
    source_hashes[str(PROGRESS.resolve())] = sha256(PROGRESS)
    source_unchanged = n150_builder.verify_hashes(source_hashes)
    if not source_unchanged:
        errors.append("PLAN-N1-157 source/checkpoint hashes changed")
    execution_cpu = time.process_time() - started + budget.children
    plan157a_cumulative = prior_plan157a_cpu + execution_cpu
    linked_cumulative = PRIOR_LINKED_CPU_SECONDS + plan157a_cumulative
    total_face_queries = builder.PRIOR_QUERIES + query_count
    complete = (
        query_count == TOTAL_QUERIES - builder.PRIOR_QUERIES
        and total_face_queries == TOTAL_QUERIES
        and len(rows) == TOTAL_QUERIES
        and len(summaries) == len(ENDPOINTS) * len(LENGTHS) * len(KINDS)
        and progress_integrity.get("complete") is True
        and source_unchanged
        and plan157a_cumulative <= CPU_LIMIT_SECONDS
        and linked_cumulative <= LINKED_CPU_LIMIT_SECONDS
        and not errors
    )
    report = {
        "schema": "taiji-n1-plan157a2-candidate-parent-fresh-diagnostic-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": builder.CORRECTION_SHA256,
        "prior_result_sha256": builder.PRIOR_RESULT_SHA256,
        "prior_attempt_sha256": builder.PRIOR_ATTEMPT_SHA256,
        "prior_progress_sha256": builder.PRIOR_PROGRESS_SHA256,
        "readiness_sha256": sha256(READINESS),
        "preflight_sha256": sha256(PREFLIGHT),
        "face_sha256": sha256(FACE),
        "face_audit_sha256": sha256(FACE_AUDIT),
        "attempt_sha256": sha256(ATTEMPT),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": source_unchanged,
        "endpoints": endpoints,
        "prior_consumed_queries": builder.PRIOR_QUERIES,
        "new_fresh_queries": query_count,
        "fresh_queries": total_face_queries,
        "fresh_diagnostic_rows": rows,
        "fresh_summary_by_endpoint_length_kind": summaries,
        "progress_integrity": progress_integrity,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "model_loads": model_load_count,
        "checkpoint_loads": model_load_count,
        "independent_preflight_restores": preflight["independent_restore_count"],
        "cpu_seconds": round(execution_cpu, 6),
        "prior_plan157a_cpu_seconds": round(prior_plan157a_cpu, 6),
        "cumulative_plan157a_cpu_seconds": round(plan157a_cumulative, 6),
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cumulative_linked_cpu_seconds": round(linked_cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": (
            f"PLAN-N1-157A2 fresh candidate-vs-parent续接{'完整' if complete else '停止'}："
            f"新增{query_count}/{TOTAL_QUERIES-builder.PRIOR_QUERIES}条查询，连同已完成的{builder.PRIOR_QUERIES}条共{total_face_queries}/{TOTAL_QUERIES}条；"
            f"更正阶段累计CPU{plan157a_cumulative:.3f}/{CPU_LIMIT_SECONDS:.0f}秒，"
            f"关联累计CPU{linked_cumulative:.3f}/{LINKED_CPU_LIMIT_SECONDS:.0f}秒；"
            "0训练/参数更新/dev/holdout/confirmation，can_promote=false。"
        ),
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-157 seed128 candidate-vs-parent fresh diagnostic"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--checkpoint-preflight", action="store_true")
    modes.add_argument("--execute-approved", action="store_true")
    modes.add_argument("--restore-worker", nargs=2, metavar=("ROLE", "PATH"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker:
        result = _restore_worker(args.restore_worker[0], Path(args.restore_worker[1]))
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0 if result.get("complete") is True and result.get("equal") is True else 1
    result = checkpoint_preflight() if args.checkpoint_preflight else execute_approved()
    keys = (
        "status",
        "complete",
        "cpu_seconds",
        "new_fresh_queries",
        "fresh_queries",
        "cumulative_plan157a_cpu_seconds",
        "cumulative_linked_cpu_seconds",
        "independent_restore_count",
        "fresh_queries",
        "progress_integrity",
        "dev_queries",
        "holdout_queries",
        "confirmation_queries",
        "can_promote",
        "errors",
        "summary_zh",
    )
    print(
        json.dumps(
            {key: result.get(key) for key in keys if key in result},
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
