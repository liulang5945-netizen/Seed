"""执行PLAN-N1-161 seed131未见问句句式迁移诊断。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_plan160_seed130_candidate_parent_face as n160
from scripts.training import build_taiji_n1_plan161_seed131_question_surface_face as builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_plan155_short_unknown_context_length_training as n155
from scripts.training import run_taiji_n1_plan159_seed129_candidate_parent_diagnostic as n159
from scripts.training import run_taiji_n1_plan160_seed130_candidate_parent_diagnostic as n160_runner

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
READINESS_ATTEMPT = builder.READINESS_ATTEMPT
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
UNKNOWN_HEX = builder.n153.UNKNOWN.encode("utf-8").hex()


class Refuse(RuntimeError):
    """PLAN-N1-161来源、bundle、面或计分合同错误。"""


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


def _fingerprint(model, optimizer) -> str:
    return n159._fingerprint(model, optimizer)


def _restore_endpoint(branch: str, path: Path):
    return n159._restore_endpoint(branch, path)


def restore_worker(branch: str, path: Path) -> dict:
    started = time.process_time()
    model, optimizer, payload, restored_fingerprint = _restore_endpoint(branch, path)
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
    # Reuse the already tested isolated-worker restore path from N1-160A.
    return n160_runner.independent_restore(endpoint, budget)


def verify_readiness() -> tuple[dict, dict[str, str]]:
    if not all(path.is_file() for path in (READINESS, SOURCE_AUDIT, SOURCE_BINDING)):
        raise FileNotFoundError("PLAN-N1-161 readiness/source binding required")
    readiness = read_json(READINESS)
    audit = read_json(SOURCE_AUDIT)
    binding = read_json(SOURCE_BINDING)
    if (
        readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or audit.get("status") != "complete"
        or audit.get("plan_sha256") != PLAN_SHA256
        or audit.get("source_hashes_unchanged") is not True
        or binding.get("status") != "complete"
        or binding.get("plan_sha256") != PLAN_SHA256
        or binding.get("readiness_sha256") != sha256(READINESS)
        or binding.get("source_hashes") != readiness.get("source_hashes")
        or readiness.get("source_audit_sha256") != sha256(SOURCE_AUDIT)
        or readiness.get("source_hashes_unchanged") is not True
    ):
        raise Refuse("PLAN-N1-161 readiness/source binding is stale")
    if not n160.n150_builder.verify_hashes(readiness.get("source_hashes", {})):
        raise Refuse("PLAN-N1-161 readiness source hashes changed")
    endpoints, sources, _ = builder.source_context()
    if len(endpoints) != 6 or not set(sources.items()).issubset(set(readiness["source_hashes"].items())):
        raise Refuse("PLAN-N1-161 frozen endpoint/source set changed")
    return readiness, dict(readiness["source_hashes"])


def checkpoint_preflight() -> dict:
    if any(path.exists() for path in (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING)):
        raise FileExistsError("PLAN-N1-161 checkpoint preflight is write-once")
    started = time.process_time()
    readiness, source_hashes = verify_readiness()
    validation_cpu = time.process_time() - started
    remaining = CPU_LIMIT_SECONDS - float(readiness["cpu_seconds"]) - validation_cpu
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-161 CPU ceiling exhausted before restore preflight")
    budget = n132.CpuBudget(remaining)
    budget.guard(reserve=45)
    write_new(
        PREFLIGHT_ATTEMPT,
        {
            "schema": "n1-plan161-checkpoint-preflight-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
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
        if len(restores) != 6 or not n160.n150_builder.verify_hashes(bound_sources):
            raise Refuse("PLAN-N1-161 six-bundle restore/source gate failed")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = time.process_time() - started + budget.children
    stage_cpu = float(readiness["cpu_seconds"]) + cpu
    linked_cpu = PRIOR_LINKED_CPU_SECONDS + stage_cpu
    complete = (
        not errors
        and len(restores) == 6
        and stage_cpu <= CPU_LIMIT_SECONDS
        and linked_cpu <= LINKED_CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan161-question-surface-preflight-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "preflight_attempt_sha256": sha256(PREFLIGHT_ATTEMPT),
        "source_hashes": bound_sources,
        "source_hashes_unchanged": n160.n150_builder.verify_hashes(bound_sources),
        "independent_restores": restores,
        "independent_restore_count": len(restores),
        "model_queries": 0,
        "parameter_updates": 0,
        "cpu_seconds": round(cpu, 6),
        "readiness_cpu_seconds": readiness["cpu_seconds"],
        "cumulative_plan161_cpu_seconds": round(stage_cpu, 6),
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
            "schema": "n1-plan161-question-surface-preflight-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
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
        raise FileNotFoundError("PLAN-N1-161 readiness/preflight/face required")
    readiness = read_json(READINESS)
    source_audit = read_json(SOURCE_AUDIT)
    source_binding = read_json(SOURCE_BINDING)
    preflight = read_json(PREFLIGHT)
    pf_binding = read_json(PREFLIGHT_BINDING)
    face = read_json(FACE)
    face_audit = read_json(FACE_AUDIT)
    face_checks = read_json(FACE_CHECKS)
    if (
        readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("source_hashes_unchanged") is not True
        or source_audit.get("status") != "complete"
        or source_audit.get("plan_sha256") != PLAN_SHA256
        or source_binding.get("status") != "complete"
        or source_binding.get("readiness_sha256") != sha256(READINESS)
        or source_binding.get("source_hashes") != readiness.get("source_hashes")
        or preflight.get("complete") is not True
        or preflight.get("plan_sha256") != PLAN_SHA256
        or preflight.get("readiness_sha256") != sha256(READINESS)
        or preflight.get("source_hashes_unchanged") is not True
        or preflight.get("independent_restore_count") != 6
        or pf_binding.get("status") != "complete"
        or pf_binding.get("plan_sha256") != PLAN_SHA256
        or pf_binding.get("readiness_sha256") != sha256(READINESS)
        or pf_binding.get("preflight_sha256") != sha256(PREFLIGHT)
        or pf_binding.get("source_hashes") != preflight.get("source_hashes")
        or face.get("plan161_schema") != "n1-plan161-v33-question-surface-face-v1"
        or face.get("seed") != builder.SEED
        or face.get("total_queries") != TOTAL_QUERIES
        or face_audit.get("status") != "complete"
        or face_audit.get("source_hashes_unchanged") is not True
        or face_audit.get("face_sha256") != sha256(FACE)
        or face_checks.get("status") != "complete"
        or face_checks.get("source_hashes_unchanged") is not True
        or face_checks.get("face_sha256") != sha256(FACE)
        or face_checks.get("face_audit_sha256") != sha256(FACE_AUDIT)
    ):
        raise Refuse("PLAN-N1-161 readiness/preflight/face binding is stale")
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
    if not n160.n150_builder.verify_hashes(source_hashes):
        raise Refuse("PLAN-N1-161 source/face/checkpoint SHA changed")
    endpoints, _, _ = builder.source_context()
    return readiness, preflight, endpoints, source_hashes


def fresh_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                int(row["seed"]),
                str(row["branch"]),
                int(row["context_length"]),
                str(row["kind"]),
                str(row["question_template_id"]),
            )
        ].append(row)
    pairs = sorted({(int(row["seed"]), str(row["branch"])) for row in rows})
    summary = []
    for seed, branch in pairs:
        for length in LENGTHS:
            for kind in KINDS:
                template_ids = (
                    "material_active",
                    "material_passive",
                ) if kind in builder.MATERIAL_KINDS else (
                    "direction_after_turning",
                    "direction_after_turn",
                )
                for template_id in template_ids:
                    items = groups[(seed, branch, length, kind, template_id)]
                    if len(items) != ITEMS_PER_KIND // 2:
                        raise Refuse(
                            "PLAN-N1-161 summary stratum differs: "
                            f"{seed}/{branch}/{length}/{kind}/{template_id}"
                        )
                    summary.append(
                        {
                            "seed": seed,
                            "branch": branch,
                            "context_length": length,
                            "kind": kind,
                            "question_template_id": template_id,
                            "items": len(items),
                            "free_exact": sum(row.get("free_exact") is True for row in items),
                            "teacher_exact": sum(row.get("teacher_exact") is True for row in items),
                            "literal_unknown": sum(row.get("output_hex") == UNKNOWN_HEX for row in items),
                        }
                    )
    return summary


def execute_approved() -> dict:
    if any(path.exists() for path in (ATTEMPT, PROGRESS, RESULT)):
        raise FileExistsError("PLAN-N1-161 diagnostic outputs are write-once")
    started = time.process_time()
    readiness, preflight, endpoints, base_sources = verify_ready_face()
    validation_cpu = time.process_time() - started
    prior_stage_cpu = (
        float(readiness["cpu_seconds"])
        + float(preflight["cpu_seconds"])
        + float(read_json(FACE_AUDIT)["cpu_seconds"])
    )
    remaining = CPU_LIMIT_SECONDS - prior_stage_cpu - validation_cpu
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-161 CPU ceiling exhausted before scoring")
    budget = n132.CpuBudget(remaining)
    face = read_json(FACE)
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan161-question-surface-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "preflight_sha256": sha256(PREFLIGHT),
            "face_sha256": sha256(FACE),
            "planned_queries": TOTAL_QUERIES,
            "model_queries": 0,
            "parameter_updates": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
            "confirmation_queries": 0,
            "can_promote": False,
        },
    )
    rows = []
    errors = []
    query_count = 0
    model_load_count = 0
    progress = PROGRESS.open("x", encoding="utf-8", newline="\n")
    try:
        for endpoint in endpoints:
            budget.guard(reserve=30)
            checkpoint = Path(endpoint["checkpoint_path"]).resolve()
            if sha256(checkpoint) != endpoint["checkpoint_sha256"]:
                raise Refuse(f"PLAN-N1-161 checkpoint SHA changed: {endpoint['seed']}/{endpoint['branch']}")
            model, optimizer, payload, fingerprint = _restore_endpoint(endpoint["branch"], checkpoint)
            model_load_count += 1
            if fingerprint != endpoint["bundle_fingerprint"]:
                raise Refuse(f"PLAN-N1-161 bundle fingerprint mismatch: {endpoint['seed']}/{endpoint['branch']}")
            before = _fingerprint(model, optimizer)
            endpoint_rows = []
            for base in face["items"]:
                for length in LENGTHS:
                    budget.guard(reserve=5)
                    observations = list(base["observations_by_length"][str(length)])
                    item = {
                        "id": f"{base['id']}-l{length}",
                        "parent_id": base["parent_id"],
                        "split": "diagnostic",
                        "kind": base["kind"],
                        "question_template_id": base["question_template_id"],
                        "question": base["question"],
                        "answer": base["answer"],
                        "entities": list(base["entities"]),
                        "observations": observations,
                        "context_length": length,
                        "answer_length": len(base["answer"].encode("utf-8")),
                    }
                    session = (
                        f"n1-plan161-{endpoint['seed']}-{endpoint['branch']}-"
                        f"{base['parent_id']}-l{length}"
                    )
                    if session in model._observations:
                        raise Refuse(f"PLAN-N1-161 session collision: {session}")
                    query_count += 1
                    try:
                        score = n155._measure_fresh_diagnostic(model, item, session)
                    finally:
                        model._observations.pop(session, None)
                    row = {
                        "seed": endpoint["seed"],
                        "branch": endpoint["branch"],
                        "model_role": "candidate"
                        if endpoint["branch"] == "unknown8_rebalanced"
                        else "parent",
                        "question_template_id": base["question_template_id"],
                        "question": base["question"],
                        "event": "question_surface_diagnostic_row",
                        **score,
                    }
                    endpoint_rows.append(row)
                    rows.append(row)
                    write_jsonl(progress, row)
                    if query_count % 64 == 0:
                        progress.flush()
            if len(endpoint_rows) != QUERIES_PER_ENDPOINT:
                raise Refuse(f"PLAN-N1-161 query count differs: {endpoint['seed']}/{endpoint['branch']}")
            if _fingerprint(model, optimizer) != before:
                raise Refuse(f"PLAN-N1-161 scoring changed bundle: {endpoint['seed']}/{endpoint['branch']}")
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
        if malformed or count != query_count or query_count > TOTAL_QUERIES:
            raise Refuse(f"PLAN-N1-161 progress mismatch: {count}/{query_count}/{TOTAL_QUERIES}")
        integrity = {
            "complete": True,
            "line_count": count,
            "malformed_rows": 0,
            "progress_sha256": sha256(PROGRESS),
        }
    except BaseException as exc:
        errors.append(f"progress integrity: {type(exc).__name__}: {exc}")
        integrity["error"] = f"{type(exc).__name__}: {exc}"
    summaries = []
    try:
        summaries = fresh_summary(rows)
    except BaseException as exc:
        errors.append(f"summary: {type(exc).__name__}: {exc}")
    source_hashes = dict(base_sources)
    source_hashes[str(ATTEMPT.resolve())] = sha256(ATTEMPT)
    source_hashes[str(PROGRESS.resolve())] = sha256(PROGRESS)
    stable = n160.n150_builder.verify_hashes(source_hashes)
    if not stable:
        errors.append("PLAN-N1-161 source hashes changed")
    execution_cpu = time.process_time() - started + budget.children
    stage_cpu = prior_stage_cpu + execution_cpu
    linked_cpu = PRIOR_LINKED_CPU_SECONDS + stage_cpu
    complete = (
        query_count == TOTAL_QUERIES
        and len(rows) == TOTAL_QUERIES
        and len(summaries) == 144
        and model_load_count == 6
        and integrity.get("complete") is True
        and stable
        and stage_cpu <= CPU_LIMIT_SECONDS
        and linked_cpu <= LINKED_CPU_LIMIT_SECONDS
        and not errors
    )
    report = {
        "schema": "taiji-n1-plan161-seed131-question-surface-diagnostic-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": stable,
        "readiness_sha256": sha256(READINESS),
        "preflight_sha256": sha256(PREFLIGHT),
        "face_sha256": sha256(FACE),
        "face_audit_sha256": sha256(FACE_AUDIT),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "attempt_sha256": sha256(ATTEMPT),
        "fresh_queries": query_count,
        "fresh_diagnostic_rows": rows,
        "fresh_summary_by_endpoint_length_kind_template": summaries,
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
        "prior_plan161_cpu_seconds": round(prior_stage_cpu, 6),
        "cumulative_plan161_cpu_seconds": round(stage_cpu, 6),
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cumulative_linked_cpu_seconds": round(linked_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": (
            f"PLAN-N1-161问句句式迁移诊断{'完整' if complete else '停止'}："
            f"{query_count}/{TOTAL_QUERIES}条，candidate/parent配对、模板分层完整性={len(summaries)}/144；"
            f"关联CPU{linked_cpu:.3f}/{LINKED_CPU_LIMIT_SECONDS:.0f}秒，"
            "0训练/dev/holdout/confirmation，can_promote=false。"
        ),
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-161 seed131 question-surface diagnostic")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-preflight", action="store_true")
    group.add_argument("--execute-approved", action="store_true")
    group.add_argument("--restore-worker", nargs=2, metavar=("ROLE", "PATH"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker:
        result = restore_worker(args.restore_worker[0], Path(args.restore_worker[1]))
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0 if result.get("complete") and result.get("equal") else 1
    result = checkpoint_preflight() if args.checkpoint_preflight else execute_approved()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "query_count",
                    "fresh_queries",
                    "independent_restore_count",
                    "model_loads",
                    "cpu_seconds",
                    "cumulative_linked_cpu_seconds",
                    "source_hashes_unchanged",
                    "can_promote",
                    "errors",
                    "summary_zh",
                )
                if key in result
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
