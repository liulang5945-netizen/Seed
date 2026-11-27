"""执行PLAN-N1-160A seed130 candidate-parent剩余端点续接诊断。"""

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

from scripts.training import build_taiji_n1_plan160_seed130_candidate_parent_face as builder
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_plan153_v28_fresh_context_length_confirmation as n153
from scripts.training import (
    run_taiji_n1_plan155_short_unknown_context_length_training as n155,
)
from scripts.training import run_taiji_n1_plan159_seed129_candidate_parent_diagnostic as n159
from scripts.training import seal_ast  # noqa: E402

PLAN_SHA256 = builder.PLAN_SHA256
CPU_LIMIT_SECONDS = builder.CPU_LIMIT_SECONDS
LINKED_CPU_LIMIT_SECONDS = builder.LINKED_CPU_LIMIT_SECONDS
PRIOR_LINKED_CPU_SECONDS = builder.PRIOR_LINKED_CPU_SECONDS
ENDPOINTS = builder.ENDPOINTS
LENGTHS = builder.LENGTHS
KINDS = builder.KINDS
ITEMS_PER_KIND = builder.ITEMS_PER_KIND
QUERIES_PER_ENDPOINT = builder.QUERIES_PER_ENDPOINT
TOTAL_QUERIES = builder.TOTAL_QUERIES

SOURCE_AUDIT = builder.SOURCE_AUDIT
READINESS = builder.READINESS
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
    """PLAN-N1-160A source, bundle, face or query contract failure."""


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


def write_jsonl(stream, value: dict) -> None:
    json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")


def fingerprint(model, optimizer) -> str:
    return n159._fingerprint(model, optimizer)


def restore_endpoint(branch: str, path: Path):
    return n159._restore_endpoint(branch, path)


def restore_worker(branch: str, path: Path) -> dict:
    started = time.process_time()
    model, optimizer, payload, restored_fingerprint = restore_endpoint(branch, path)
    expected = payload.get("bundle_fingerprint")
    report = {
        "complete": True,
        "equal": restored_fingerprint == expected,
        "fingerprint": restored_fingerprint,
        "bundle_fingerprint": expected,
        "branch": branch,
        "seed": int(payload["seed"]),
        "total_updates": payload["total_updates"],
        "history_count": len(model._observations),
        "optimizer_parameters": len(optimizer.state),
        "cpu_seconds": round(time.process_time() - started, 6),
    }
    del model, optimizer, payload
    return report


def independent_restore(endpoint: dict, budget: n132.CpuBudget) -> dict:
    budget.guard(reserve=5)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    timeout = min(90, max(1.0, budget.limit - budget.used() - 1.0))
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--restore-worker",
            str(endpoint["branch"]),
            str(endpoint["checkpoint_path"]),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
        env=env,
    )
    result = json.loads(process.stdout)
    budget.children += float(result["cpu_seconds"])
    budget.guard()
    if (
        result.get("complete") is not True
        or result.get("equal") is not True
        or result.get("fingerprint") != endpoint["bundle_fingerprint"]
    ):
        raise Refuse(f"PLAN-N1-160A independent restore mismatch: {endpoint['checkpoint_path']}")
    return {"path": endpoint["checkpoint_path"], "sha256": endpoint["checkpoint_sha256"], **result}


def verify_readiness() -> tuple[dict, dict[str, str]]:
    if not all(path.is_file() for path in (READINESS, SOURCE_AUDIT, SOURCE_BINDING)):
        raise FileNotFoundError("PLAN-N1-160A readiness is required")
    readiness = read_json(READINESS)
    audit = read_json(SOURCE_AUDIT)
    binding = read_json(SOURCE_BINDING)
    if (
        readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or readiness.get("prior_result_sha256") != builder.PRIOR_RESULT_SHA256
        or readiness.get("prior_consumed_queries") != builder.PRIOR_QUERIES
        or readiness.get("remaining_queries") != TOTAL_QUERIES - builder.PRIOR_QUERIES
        or audit.get("status") != "complete"
        or audit.get("plan_sha256") != PLAN_SHA256
        or audit.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or audit.get("prior_result_sha256") != builder.PRIOR_RESULT_SHA256
        or audit.get("prior_consumed_queries") != builder.PRIOR_QUERIES
        or audit.get("remaining_queries") != TOTAL_QUERIES - builder.PRIOR_QUERIES
        or binding.get("status") != "complete"
        or binding.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or binding.get("prior_result_sha256") != builder.PRIOR_RESULT_SHA256
        or binding.get("prior_consumed_queries") != builder.PRIOR_QUERIES
        or binding.get("remaining_queries") != TOTAL_QUERIES - builder.PRIOR_QUERIES
        or not seal_ast.receipt_binding_matches(binding.get("readiness_sha256"), READINESS)
        or binding.get("source_hashes") != readiness.get("source_hashes")
    ):
        raise Refuse("PLAN-N1-160A readiness/source binding is stale")
    if not n150_builder.verify_hashes(readiness.get("source_hashes", {})):
        raise Refuse("PLAN-N1-160A bound source/checkpoint hashes changed")
    endpoints, sources, _ = builder.source_context()
    if len(endpoints) != 6 or not set(sources.items()).issubset(
        set(readiness["source_hashes"].items())
    ):
        raise Refuse("PLAN-N1-160A fixed source/endpoint set changed")
    return readiness, dict(readiness["source_hashes"])


def checkpoint_preflight() -> dict:
    if any(path.exists() for path in (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING)):
        raise FileExistsError("PLAN-N1-160A checkpoint preflight is write-once")
    started = time.process_time()
    readiness, source_hashes = verify_readiness()
    validation_cpu = time.process_time() - started
    remaining = CPU_LIMIT_SECONDS - float(readiness["cpu_seconds"]) - validation_cpu
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-160A CPU ceiling exhausted before restore preflight")
    budget = n132.CpuBudget(remaining)
    budget.guard(reserve=45)
    write_new(
        PREFLIGHT_ATTEMPT,
        {
            "schema": "n1-plan160a-checkpoint-preflight-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "correction_addendum_sha256": builder.CORRECTION_SHA256,
            "prior_result_sha256": builder.PRIOR_RESULT_SHA256,
            "prior_consumed_queries": builder.PRIOR_QUERIES,
            "remaining_queries": TOTAL_QUERIES - builder.PRIOR_QUERIES,
            "readiness_sha256": sha256(READINESS),
            "restore_count_planned": 6,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    endpoints, _, _ = builder.source_context()
    bound_sources = dict(source_hashes)
    bound_sources[str(PREFLIGHT_ATTEMPT.resolve())] = sha256(PREFLIGHT_ATTEMPT)
    restores = []
    errors = []
    try:
        for endpoint in endpoints:
            restores.append(independent_restore(endpoint, budget))
        if len(restores) != 6 or not n150_builder.verify_hashes(bound_sources):
            raise Refuse("PLAN-N1-160A six-bundle restore/source gate failed")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = time.process_time() - started + budget.children
    plan_cpu = float(readiness["cpu_seconds"]) + cpu
    linked_cpu = PRIOR_LINKED_CPU_SECONDS + plan_cpu
    complete = (
        not errors
        and len(restores) == 6
        and plan_cpu <= CPU_LIMIT_SECONDS
        and linked_cpu <= LINKED_CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan160a-checkpoint-preflight-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": builder.CORRECTION_SHA256,
        "prior_result_sha256": builder.PRIOR_RESULT_SHA256,
        "prior_consumed_queries": builder.PRIOR_QUERIES,
        "remaining_queries": TOTAL_QUERIES - builder.PRIOR_QUERIES,
        "readiness_sha256": sha256(READINESS),
        "preflight_attempt_sha256": sha256(PREFLIGHT_ATTEMPT),
        "source_hashes": bound_sources,
        "source_hashes_unchanged": n150_builder.verify_hashes(bound_sources),
        "independent_restores": restores,
        "independent_restore_count": len(restores),
        "model_queries": 0,
        "parameter_updates": 0,
        "cpu_seconds": round(cpu, 6),
        "readiness_cpu_seconds": readiness["cpu_seconds"],
        "cumulative_plan160_cpu_seconds": round(plan_cpu, 6),
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cumulative_linked_cpu_seconds": round(linked_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
    }
    write_new(PREFLIGHT, report)
    write_new(
        PREFLIGHT_BINDING,
        {
            "schema": "n1-plan160a-checkpoint-preflight-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "correction_addendum_sha256": builder.CORRECTION_SHA256,
            "prior_result_sha256": builder.PRIOR_RESULT_SHA256,
            "prior_consumed_queries": builder.PRIOR_QUERIES,
            "remaining_queries": TOTAL_QUERIES - builder.PRIOR_QUERIES,
            "preflight_sha256": sha256(PREFLIGHT),
            "source_hashes": bound_sources,
            "can_promote": False,
        },
    )
    return report


def verify_ready_face() -> tuple[dict, dict, list[dict], dict[str, str]]:
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
        raise FileNotFoundError("PLAN-N1-160A readiness/preflight/face required")
    readiness = read_json(READINESS)
    preflight = read_json(PREFLIGHT)
    pf_binding = read_json(PREFLIGHT_BINDING)
    face = read_json(FACE)
    face_audit = read_json(FACE_AUDIT)
    face_checks = read_json(FACE_CHECKS)
    if (
        readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or readiness.get("prior_result_sha256") != builder.PRIOR_RESULT_SHA256
        or readiness.get("prior_consumed_queries") != builder.PRIOR_QUERIES
        or readiness.get("remaining_queries") != TOTAL_QUERIES - builder.PRIOR_QUERIES
        or preflight.get("complete") is not True
        or preflight.get("plan_sha256") != PLAN_SHA256
        or preflight.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or preflight.get("prior_result_sha256") != builder.PRIOR_RESULT_SHA256
        or preflight.get("prior_consumed_queries") != builder.PRIOR_QUERIES
        or preflight.get("remaining_queries") != TOTAL_QUERIES - builder.PRIOR_QUERIES
        or preflight.get("independent_restore_count") != 6
        or pf_binding.get("status") != "complete"
        or pf_binding.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or pf_binding.get("prior_result_sha256") != builder.PRIOR_RESULT_SHA256
        or pf_binding.get("prior_consumed_queries") != builder.PRIOR_QUERIES
        or not seal_ast.receipt_binding_matches(pf_binding.get("preflight_sha256"), PREFLIGHT)
        or face.get("plan160_schema") != "n1-plan160-v32-candidate-parent-face-v1"
        or face.get("seed") != 130
        or face.get("total_queries") != TOTAL_QUERIES
        or face_audit.get("status") != "complete"
        or not seal_ast.receipt_binding_matches(face_audit.get("face_sha256"), FACE)
        or face_checks.get("status") != "complete"
        or not seal_ast.receipt_binding_matches(face_checks.get("face_sha256"), FACE)
        or not seal_ast.receipt_binding_matches(face_checks.get("face_audit_sha256"), FACE_AUDIT)
    ):
        raise Refuse("PLAN-N1-160A preflight/face binding is stale")
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
        raise Refuse("PLAN-N1-160A source/face/checkpoint SHA changed")
    endpoints, _, _ = builder.source_context()
    return readiness, preflight, endpoints, source_hashes


def fresh_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[
            (int(row["seed"]), str(row["branch"]), int(row["context_length"]), str(row["kind"]))
        ].append(row)
    summary = []
    for seed, branch in sorted({(int(row["seed"]), str(row["branch"])) for row in rows}):
        for length in LENGTHS:
            for kind in KINDS:
                items = groups[(seed, branch, length, kind)]
                if len(items) != ITEMS_PER_KIND:
                    raise Refuse(
                        f"PLAN-N1-160A summary stratum differs: {seed}/{branch}/{length}/{kind}"
                    )
                summary.append(
                    {
                        "seed": seed,
                        "branch": branch,
                        "context_length": length,
                        "kind": kind,
                        "items": len(items),
                        "free_exact": sum(x.get("free_exact") is True for x in items),
                        "literal_unknown": sum(x.get("output_hex") == UNKNOWN_HEX for x in items),
                    }
                )
    return summary


def execute_approved() -> dict:
    if any(path.exists() for path in (ATTEMPT, PROGRESS, RESULT)):
        raise FileExistsError("PLAN-N1-160A continuation output is write-once")
    started = time.process_time()
    readiness, preflight, endpoints, base_sources = verify_ready_face()
    validation_cpu = time.process_time() - started
    prior_plan160a_cpu = float(readiness["cpu_seconds"]) + float(preflight["cpu_seconds"])
    remaining = CPU_LIMIT_SECONDS - prior_plan160a_cpu - validation_cpu
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-160A CPU ceiling exhausted before continuation")
    budget = n132.CpuBudget(remaining)
    prior_result = read_json(builder.PRIOR_RESULT)
    prior_rows = list(prior_result.get("fresh_diagnostic_rows", []))
    prior_endpoint_keys = {(int(row["seed"]), str(row["branch"])) for row in prior_rows}
    if len(prior_rows) != builder.PRIOR_QUERIES or prior_endpoint_keys != {
        (11, "unknown8_rebalanced")
    }:
        raise Refuse("PLAN-N1-160A prior face rows are not exactly the sealed seed11 candidate")
    prior_summaries = fresh_summary(prior_rows)
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan160a-candidate-parent-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "correction_addendum_sha256": builder.CORRECTION_SHA256,
            "prior_result_sha256": builder.PRIOR_RESULT_SHA256,
            "prior_attempt_sha256": builder.PRIOR_ATTEMPT_SHA256,
            "prior_progress_sha256": builder.PRIOR_PROGRESS_SHA256,
            "readiness_sha256": sha256(READINESS),
            "preflight_sha256": sha256(PREFLIGHT),
            "face_sha256": sha256(FACE),
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
        },
    )
    rows = prior_rows.copy()
    summaries = prior_summaries.copy()
    errors = []
    query_count = 0
    model_load_count = 0
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
                    f"PLAN-N1-160A checkpoint changed: {endpoint['seed']}/{endpoint['branch']}"
                )
            model, optimizer, payload, fp = n159._restore_endpoint(endpoint["branch"], checkpoint)
            model_load_count += 1
            if fp != endpoint["bundle_fingerprint"]:
                raise Refuse(
                    f"PLAN-N1-160A bundle fingerprint mismatch: {endpoint['seed']}/{endpoint['branch']}"
                )
            before = n159._fingerprint(model, optimizer)
            endpoint_rows = []
            for base in face["items"]:
                for length in LENGTHS:
                    budget.guard(reserve=5)
                    item = n153._face_from_manifest_item(base, length)
                    session = f"n1-plan160a-fresh-{endpoint['seed']}-{endpoint['branch']}-{item['parent_id']}-l{length}"
                    if session in model._observations:
                        raise Refuse(f"PLAN-N1-160A session collision: {session}")
                    query_count += 1
                    try:
                        score = n155._measure_fresh_diagnostic(model, item, session)
                    finally:
                        model._observations.pop(session, None)
                    row = {
                        "seed": endpoint["seed"],
                        "branch": endpoint["branch"],
                        "model_role": (
                            "candidate" if endpoint["branch"] == "unknown8_rebalanced" else "parent"
                        ),
                        "event": "fresh_diagnostic_row",
                        **score,
                    }
                    endpoint_rows.append(row)
                    rows.append(row)
                    write_jsonl(progress, row)
                    if query_count % 64 == 0:
                        progress.flush()
            if len(endpoint_rows) != QUERIES_PER_ENDPOINT:
                raise Refuse(
                    f"PLAN-N1-160A query count differs: {endpoint['seed']}/{endpoint['branch']}"
                )
            summaries.extend(fresh_summary(endpoint_rows))
            if n159._fingerprint(model, optimizer) != before:
                raise Refuse(
                    f"PLAN-N1-160A scoring changed bundle: {endpoint['seed']}/{endpoint['branch']}"
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
    integrity = {"complete": False, "line_count": 0, "malformed_rows": 0, "error": None}
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
                "PLAN-N1-160A continuation progress mismatch: " f"{count}/{len(rows)}/{query_count}"
            )
        integrity = {
            "complete": True,
            "line_count": count,
            "malformed_rows": 0,
            "progress_sha256": sha256(PROGRESS),
        }
    except BaseException as exc:
        errors.append(f"progress integrity: {type(exc).__name__}: {exc}")
        integrity["error"] = f"{type(exc).__name__}: {exc}"
    source_hashes = dict(base_sources)
    source_hashes[str(ATTEMPT.resolve())] = sha256(ATTEMPT)
    source_hashes[str(PROGRESS.resolve())] = sha256(PROGRESS)
    stable = n150_builder.verify_hashes(source_hashes)
    if not stable:
        errors.append("PLAN-N1-160A source hashes changed")
    execution_cpu = time.process_time() - started + budget.children
    stage_cpu = prior_plan160a_cpu + execution_cpu
    linked_cpu = PRIOR_LINKED_CPU_SECONDS + stage_cpu
    total_face_queries = builder.PRIOR_QUERIES + query_count
    complete = (
        query_count == TOTAL_QUERIES - builder.PRIOR_QUERIES
        and total_face_queries == TOTAL_QUERIES
        and len(rows) == TOTAL_QUERIES
        and len(summaries) == 72
        and integrity.get("complete") is True
        and stable
        and stage_cpu <= CPU_LIMIT_SECONDS
        and linked_cpu <= LINKED_CPU_LIMIT_SECONDS
        and not errors
    )
    report = {
        "schema": "taiji-n1-plan160a-seed130-candidate-parent-replication-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": builder.CORRECTION_SHA256,
        "prior_result_sha256": builder.PRIOR_RESULT_SHA256,
        "prior_attempt_sha256": builder.PRIOR_ATTEMPT_SHA256,
        "prior_progress_sha256": builder.PRIOR_PROGRESS_SHA256,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": stable,
        "readiness_sha256": sha256(READINESS),
        "preflight_sha256": sha256(PREFLIGHT),
        "face_sha256": sha256(FACE),
        "face_audit_sha256": sha256(FACE_AUDIT),
        "attempt_sha256": sha256(ATTEMPT),
        "prior_consumed_queries": builder.PRIOR_QUERIES,
        "new_fresh_queries": query_count,
        "fresh_queries": total_face_queries,
        "fresh_diagnostic_rows": rows,
        "fresh_summary_by_endpoint_length_kind": summaries,
        "progress_integrity": integrity,
        "model_loads": model_load_count,
        "checkpoint_loads": model_load_count,
        "independent_preflight_restores": 6,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "cpu_seconds": round(execution_cpu, 6),
        "prior_plan160a_cpu_seconds": round(prior_plan160a_cpu, 6),
        "cumulative_plan160a_cpu_seconds": round(stage_cpu, 6),
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cumulative_linked_cpu_seconds": round(linked_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": (
            f"PLAN-N1-160A续接{'完整' if complete else '停止'}：新增{query_count}/{TOTAL_QUERIES-builder.PRIOR_QUERIES}条，"
            f"合并既有{builder.PRIOR_QUERIES}条共{total_face_queries}/{TOTAL_QUERIES}条；"
            f"linked CPU{linked_cpu:.3f}/{LINKED_CPU_LIMIT_SECONDS:.0f}秒，"
            "0训练/dev/holdout/confirmation，can_promote=false。"
        ),
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-160A seed130 candidate-parent remaining-endpoint continuation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-preflight", action="store_true")
    group.add_argument("--execute-approved", action="store_true")
    group.add_argument("--restore-worker", nargs=2, metavar=("ROLE", "PATH"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker:
        r = n159._restore_worker(args.restore_worker[0], Path(args.restore_worker[1]))
        print(json.dumps(r, ensure_ascii=True, sort_keys=True))
        return 0 if r.get("complete") and r.get("equal") else 1
    r = checkpoint_preflight() if args.checkpoint_preflight else execute_approved()
    keys = (
        "status",
        "complete",
        "cpu_seconds",
        "cumulative_plan160_cpu_seconds",
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
    print(json.dumps({k: r.get(k) for k in keys if k in r}, ensure_ascii=True, sort_keys=True))
    return 0 if r.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
