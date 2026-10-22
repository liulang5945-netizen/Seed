"""N1-153：N1-150 final bundles上的fresh内容长度配对诊断。"""

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
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_v27_length_robust_training_after_n149 as n150
from scripts.training import seal_ast  # noqa: E402

PLAN = builder.PLAN
PLAN_SHA256 = builder.PLAN_SHA256
CPU_LIMIT_SECONDS = 3_600.0
ENDPOINTS = builder.ENDPOINTS
LENGTHS = builder.LENGTHS
KINDS = builder.KINDS
EXPECTED_QUERIES = builder.EXPECTED_QUERIES
UNKNOWN_HEX = builder.UNKNOWN.encode("utf-8").hex()
SOURCE_AUDIT = builder.SOURCE_AUDIT
FACE = builder.FACE
FACE_CHECKS = builder.FACE_CHECKS
READINESS = ROOT / "reports/taiji_n1_plan153_v28_fresh_context_readiness_20261020.json"
READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan153_v28_fresh_context_readiness_attempt_20261020.json"
)
SOURCE_BINDING = ROOT / "reports/taiji_n1_plan153_v28_fresh_context_source_binding_20261020.json"
PREFLIGHT = ROOT / "reports/taiji_n1_plan153_v28_fresh_context_checkpoint_preflight_20261020.json"
PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan153_v28_fresh_context_checkpoint_preflight_attempt_20261020.json"
)
PREFLIGHT_BINDING = (
    ROOT / "reports/taiji_n1_plan153_v28_fresh_context_checkpoint_preflight_binding_20261020.json"
)
ATTEMPT = ROOT / "reports/taiji_n1_plan153_v28_fresh_context_attempt_20261020.json"
RESULT = (
    ROOT / "reports/taiji_n1_plan153_v28_fresh_context_length_confirmation_result_20261020.json"
)
N150_RESULT = n150.RESULT
N150_RESULT_SHA256 = builder.N150_RESULT_SHA256
N151A_RESULT = builder.N151A_RESULT
N151A_RESULT_SHA256 = builder.N151A_RESULT_SHA256
N152_RESULT = builder.N152_RESULT
N152_RESULT_SHA256 = builder.N152_RESULT_SHA256
TEST = ROOT / "tests/taiji_native/test_n1_plan153_v28_fresh_context_length_confirmation_contract.py"


class Refuse(RuntimeError):
    """N1-153来源、保存恢复、评分计数或资源合同不成立。"""


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
            raise Refuse(f"conflicting N1-153 source hash: {path}")
        target[path] = digest


def collect_source_hashes(
    n150_result: dict, n151a_result: dict, n152_result: dict, inherited: dict[str, str]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    _merge_hashes(hashes, inherited)
    _merge_hashes(hashes, n150_result.get("source_hashes", {}))
    _merge_hashes(hashes, n151a_result.get("source_hashes", {}))
    _merge_hashes(hashes, n152_result.get("source_hashes", {}))
    _merge_hashes(hashes, n150_result.get("new_checkpoint_hashes", {}))
    paths = (
        PLAN,
        builder.N151A_PLAN,
        builder.N151_FAILURE,
        n150_builder.PLAN,
        n150_builder.SCORING_ADDENDUM,
        n150_builder.RESOURCE_ADDENDUM,
        n150_builder.N149_MANIFEST,
        n150_builder.SOURCE_EXCLUSIONS,
        n150_builder.SCHEDULE,
        N150_RESULT,
        n150.ATTEMPT,
        n150.PROGRESS,
        N151A_RESULT,
        N152_RESULT,
        SOURCE_AUDIT,
        FACE,
        FACE_CHECKS,
        Path(__file__).resolve(),
        builder.__file__,
        TEST,
    )
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise Refuse(f"N1-153 source is missing: {path}")
        _merge_hashes(hashes, {str(path): sha256(path)})
    if not n150_builder.verify_hashes(hashes):
        raise Refuse("N1-153 source or checkpoint hash changed")
    return dict(sorted(hashes.items()))


def _source_context() -> tuple[dict, dict, dict, dict, dict, dict, dict, dict[str, str]]:
    n150_readiness, n150_preflight, _n150_binding, bound, schedule = n150._verify_preflight()
    n150_result = read_json(N150_RESULT)
    n151a_result = read_json(N151A_RESULT)
    n152_result = read_json(N152_RESULT)
    require(sha256(N150_RESULT) == N150_RESULT_SHA256, "N1-150 result SHA changed")
    require(sha256(N151A_RESULT) == N151A_RESULT_SHA256, "N1-151A result SHA changed")
    require(sha256(N152_RESULT) == N152_RESULT_SHA256, "N1-152 result SHA changed")
    require(
        n150_result.get("complete") is True
        and n150_result.get("training_sufficient_all_endpoints") is True
        and n150_result.get("can_promote") is False
        and n151a_result.get("complete") is True
        and n151a_result.get("repeated_train_only_queries") == 7_488
        and n152_result.get("complete") is True
        and n152_result.get("source_rows") == 1_728
        and n152_result.get("model_queries") == 0,
        "N1-150/151A/152 frozen source results are incomplete",
    )
    if not FACE.is_file() or not SOURCE_AUDIT.is_file() or not FACE_CHECKS.is_file():
        raise FileNotFoundError("N1-153 fresh face, source audit, and face checks are required")
    audit = read_json(SOURCE_AUDIT)
    face = read_json(FACE)
    checks = read_json(FACE_CHECKS)
    require(
        audit.get("status") == "complete"
        and checks.get("status") == "complete"
        and checks.get("accepted") is True
        and checks.get("plan_sha256") == PLAN_SHA256
        and seal_ast.receipt_binding_matches(checks.get("face_manifest_sha256"), FACE)
        and audit.get("face_sha256") == checks.get("face_sha256")
        and face.get("plan_sha256") == PLAN_SHA256,
        "N1-153 prepared face/source audit is stale or invalid",
    )
    inherited = dict(n150_result.get("source_hashes", {}))
    _merge_hashes(inherited, n151a_result.get("source_hashes", {}))
    _merge_hashes(inherited, n152_result.get("source_hashes", {}))
    for path, digest in n150_result.get("new_checkpoint_hashes", {}).items():
        _merge_hashes(inherited, {path: digest})
    sources = collect_source_hashes(n150_result, n151a_result, n152_result, inherited)
    return n150_readiness, n150_preflight, n150_result, bound, schedule, face, checks, sources


def _face_from_manifest_item(base: dict, length: int) -> dict:
    if length not in LENGTHS:
        raise ValueError(f"unsupported N1-153 context length: {length}")
    observations = list(base["observations_by_length"][str(length)])
    if len(observations) != length:
        raise ValueError(f"N1-153 manifest context length mismatch: {base['id']}/{length}")
    return {
        "id": f"{base['id']}-l{length}",
        "parent_id": base["id"],
        "split": "diagnostic",
        "kind": base["kind"],
        "entities": list(base["entities"]),
        "question": base["question"],
        "answer": base["answer"],
        "observations": observations,
        "context_length": length,
        "target_observation_indices": list(base["target_observation_indices"]),
    }


@torch.no_grad()
def _measure_diagnostic(model, item: dict, session: str) -> dict:
    if item.get("split") != "diagnostic":
        raise ValueError("N1-153 scorer only accepts diagnostic split")
    paired = n132.e11.scorer.paired
    data = paired.project(item)
    paired.observe(model, data, session)
    tape = paired.evidence(model, session, "position")
    target = item["answer"].encode("utf-8")
    labels = torch.tensor([*target, 256])
    logits, _ = model.teacher_logits(tape, session, data.question, target)
    predicted = logits.argmax(dim=-1)
    trace = model.generate_trace(tape, session, data.question, 64)
    return {
        "id": item["id"],
        "parent_id": item["parent_id"],
        "kind": item["kind"],
        "split": "diagnostic",
        "context_length": item["context_length"],
        "target_observation_indices": item["target_observation_indices"],
        "answer_length": len(target),
        "teacher_exact": bool(torch.equal(predicted, labels)),
        "teacher_tokens_correct": int((predicted == labels).sum()),
        "teacher_token_count": len(labels),
        "teacher_ce": float(torch.nn.functional.cross_entropy(logits, labels)),
        "teacher_symbols": predicted.tolist(),
        "target_symbols": labels.tolist(),
        "output_hex": trace.output.hex(),
        "raw_hex": trace.raw.hex(),
        "stop_reason": trace.stop_reason,
        "generated_steps": trace.generated_steps,
        "free_exact": trace.raw == target and trace.stop_reason == "boundary",
    }


def validate_face_queries(face: dict) -> dict:
    items = face.get("items", [])
    require(
        len(items) == 160 and face.get("base_item_count") == 160,
        "N1-153 requires 160 face base items",
    )
    rows = []
    per_kind = Counter()
    for base in items:
        per_kind[base.get("kind")] += 1
        for length in LENGTHS:
            item = _face_from_manifest_item(base, length)
            selected = n132._target_observation_indices(item)
            expected = base["target_observation_indices"] or None
            if selected != expected:
                raise Refuse(f"N1-153 runtime target rows differ: {base['id']}/{length}")
            rows.append(item)
    require(
        per_kind == Counter({kind: 40 for kind in KINDS}),
        "N1-153 runtime face kind balance differs",
    )
    require(len(rows) == 480, "N1-153 per-endpoint face query count must be 480")
    return {
        "items": len(items),
        "queries_per_endpoint": len(rows),
        "items_by_kind": dict(sorted(per_kind.items())),
    }


def _summary_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[
            (int(row["seed"]), str(row["branch"]), int(row["context_length"]), str(row["kind"]))
        ].append(row)
    summary = []
    for seed, branch in ENDPOINTS:
        for length in LENGTHS:
            for kind in KINDS:
                selected = groups[(seed, branch, length, kind)]
                if len(selected) != 40:
                    raise Refuse(
                        f"N1-153 result stratum count differs: {seed}/{branch}/{length}/{kind}"
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
                            row.get("output_hex") == UNKNOWN_HEX for row in selected
                        ),
                    }
                )
    return summary


def _validate_prepared_outputs() -> tuple[dict, dict, dict[str, str]]:
    n150_readiness, n150_preflight, n150_result, bound, schedule, face, checks, sources = (
        _source_context()
    )
    excluded_entities, excluded_words, exclusion_summary, rebuilt_sources, _runtime = (
        builder.collect_n150_exclusions()
    )
    audit = read_json(SOURCE_AUDIT)
    for key, value in exclusion_summary.items():
        if audit.get(key) != value:
            raise Refuse(f"N1-153 source-exclusion audit differs from rebuilt union: {key}")
    if audit.get("source_hashes") != rebuilt_sources:
        raise Refuse("N1-153 source audit hash map differs from rebuilt source union")
    runtime_face = builder.build_face(excluded_entities, excluded_words)[0]
    if runtime_face != face or not seal_ast.receipt_binding_matches(
        checks.get("face_manifest_sha256"), FACE
    ):
        raise Refuse("N1-153 deterministically rebuilt face differs from the frozen manifest")
    face_summary = validate_face_queries(face)
    return (
        {
            "bound": bound,
            "schedule": schedule,
            "face": face,
            "face_summary": face_summary,
            "n150_result": n150_result,
        },
        checks,
        sources,
    )


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
        raise FileExistsError("N1-153 runner outputs are write-once")
    started = time.process_time()
    runtime, checks, sources = _validate_prepared_outputs()
    if not n150_builder.verify_hashes(sources):
        raise Refuse("N1-153 source/checkpoint hashes changed during readiness")
    cpu = time.process_time() - started
    report = {
        "schema": "n1-plan153-v28-fresh-context-readiness-v1",
        "status": "ready_for_checkpoint_preflight" if cpu <= CPU_LIMIT_SECONDS else "stopped",
        "complete": cpu <= CPU_LIMIT_SECONDS,
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "source_audit_sha256": sha256(SOURCE_AUDIT),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "source_hashes": sources,
        "source_hashes_unchanged": True,
        "endpoint_count": len(ENDPOINTS),
        "base_item_count": runtime["face_summary"]["items"],
        "queries_per_endpoint": runtime["face_summary"]["queries_per_endpoint"],
        "planned_queries": EXPECTED_QUERIES,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "summary_zh": "N1-153 face、source exclusion、嵌套长度和六endpoint查询分母通过；0模型/checkpoint加载、0查询、0更新。",
    }
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan153-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "cpu_seconds": round(cpu, 6),
            "model_queries": 0,
            "parameter_updates": 0,
        },
    )
    write_new(READINESS, report)
    write_new(
        SOURCE_BINDING,
        {
            "schema": "n1-plan153-readiness-source-binding-v1",
            "status": "complete" if report["complete"] else "stopped",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_hashes": sources,
            "face_sha256": sha256(FACE),
            "can_promote": False,
        },
    )
    return report


def _verify_readiness() -> tuple[dict, dict, dict, dict[str, str]]:
    if not READINESS.is_file() or not SOURCE_BINDING.is_file():
        raise FileNotFoundError("N1-153 no-query readiness is required")
    report = read_json(READINESS)
    binding = read_json(SOURCE_BINDING)
    if (
        report.get("complete") is not True
        or report.get("status") != "ready_for_checkpoint_preflight"
        or report.get("plan_sha256") != PLAN_SHA256
        or binding.get("status") != "complete"
        or not seal_ast.receipt_binding_matches(binding.get("readiness_sha256"), READINESS)
        or binding.get("source_hashes") != report.get("source_hashes")
    ):
        raise Refuse("N1-153 readiness/source binding is stale")
    _runtime, _checks, sources = _validate_prepared_outputs()
    if sources != report.get("source_hashes") or not n150_builder.verify_hashes(sources):
        raise Refuse("N1-153 source hashes changed after readiness")
    return report, binding, read_json(FACE_CHECKS), sources


def checkpoint_preflight() -> dict:
    if any(path.exists() for path in (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING)):
        raise FileExistsError("N1-153 final checkpoint preflight is write-once")
    started = time.process_time()
    readiness, _binding, checks, sources = _verify_readiness()
    validation_cpu = time.process_time() - started
    remaining = CPU_LIMIT_SECONDS - float(readiness["cpu_seconds"]) - validation_cpu
    if remaining <= 0:
        raise TimeoutError("N1-153 CPU ceiling exhausted during readiness revalidation")
    budget = n132.CpuBudget(remaining)
    budget.guard(reserve=30)
    write_new(
        PREFLIGHT_ATTEMPT,
        {
            "schema": "n1-plan153-checkpoint-preflight-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "endpoint_count": len(ENDPOINTS),
            "model_queries": 0,
            "parameter_updates": 0,
        },
    )
    n150_result = read_json(N150_RESULT)
    expected = {(int(row["seed"]), str(row["branch"])): row for row in n150_result["models"]}
    restores = []
    errors = []
    try:
        for seed, branch in ENDPOINTS:
            budget.guard(reserve=10)
            model_row = expected[(seed, branch)]
            path = Path(model_row["final_checkpoint_path"]).resolve()
            restore = n150._independent_restore(path, budget)
            if restore.get("fingerprint") != model_row.get(
                "final_bundle_fingerprint"
            ) or restore.get("sha256") != model_row.get("final_checkpoint_sha256"):
                raise Refuse(f"N1-153 final checkpoint restore mismatch: {seed}/{branch}")
            restores.append(restore)
        if not n150_builder.verify_hashes(sources):
            raise Refuse("N1-153 source/checkpoint hashes changed during restore preflight")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = time.process_time() - started + budget.children
    cumulative = float(readiness["cpu_seconds"]) + cpu
    all_sources = dict(sources)
    all_sources[str(READINESS.resolve())] = sha256(READINESS)
    all_sources[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
    all_sources[str(PREFLIGHT_ATTEMPT.resolve())] = sha256(PREFLIGHT_ATTEMPT)
    complete = not errors and len(restores) == 6 and cumulative <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan153-final-checkpoint-preflight-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "preflight_attempt_sha256": sha256(PREFLIGHT_ATTEMPT),
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "source_hashes": all_sources,
        "source_hashes_unchanged": n150_builder.verify_hashes(all_sources),
        "independent_restores": restores,
        "independent_restore_count": len(restores),
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "cpu_seconds": round(cpu, 6),
        "prior_readiness_cpu_seconds": float(readiness["cpu_seconds"]),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": (
            "六个N1-150 final full bundle再次独立恢复6/6；0模型查询、0更新。"
            if complete
            else "checkpoint恢复前置停止，不查询、不重试。"
        ),
    }
    write_new(PREFLIGHT, report)
    write_new(
        PREFLIGHT_BINDING,
        {
            "schema": "n1-plan153-checkpoint-preflight-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "preflight_sha256": sha256(PREFLIGHT),
            "source_hashes": all_sources,
            "can_promote": False,
        },
    )
    return report


def _verify_preflight() -> tuple[dict, dict, dict, dict[str, str]]:
    if not all(
        path.is_file() for path in (READINESS, SOURCE_BINDING, PREFLIGHT, PREFLIGHT_BINDING)
    ):
        raise FileNotFoundError("N1-153 readiness and final restore preflight are required")
    readiness = read_json(READINESS)
    binding = read_json(SOURCE_BINDING)
    preflight = read_json(PREFLIGHT)
    preflight_binding = read_json(PREFLIGHT_BINDING)
    if (
        readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or binding.get("status") != "complete"
        or not seal_ast.receipt_binding_matches(binding.get("readiness_sha256"), READINESS)
        or preflight.get("complete") is not True
        or preflight.get("independent_restore_count") != 6
        or preflight.get("plan_sha256") != PLAN_SHA256
        or preflight.get("source_hashes_unchanged") is not True
        or preflight.get("model_queries") != 0
        or preflight.get("parameter_updates") != 0
        or preflight_binding.get("status") != "complete"
        or not seal_ast.receipt_binding_matches(
            preflight_binding.get("preflight_sha256"), PREFLIGHT
        )
        or preflight_binding.get("source_hashes") != preflight.get("source_hashes")
    ):
        raise Refuse("N1-153 readiness/final restore preflight is stale")
    if not n150_builder.verify_hashes(preflight.get("source_hashes", {})):
        raise Refuse("N1-153 source/checkpoint hashes changed before query")
    _runtime, _checks, sources = _validate_prepared_outputs()
    return readiness, preflight, read_json(FACE), sources


def execute_approved() -> dict:
    started = time.process_time()
    readiness, preflight, face, sources = _verify_preflight()
    validation_cpu = time.process_time() - started
    remaining = CPU_LIMIT_SECONDS - float(preflight["cumulative_cpu_seconds"]) - validation_cpu
    if remaining <= 0:
        raise TimeoutError("N1-153 CPU ceiling exhausted before query")
    if any(path.exists() for path in (ATTEMPT, RESULT)):
        raise FileExistsError("N1-153 query attempt is write-once")
    budget = n132.CpuBudget(remaining)
    input_hashes = dict(preflight["source_hashes"])
    for path in (READINESS, SOURCE_BINDING, PREFLIGHT, PREFLIGHT_BINDING):
        input_hashes[str(path.resolve())] = sha256(path)
    if not n150_builder.verify_hashes(input_hashes):
        raise Refuse("N1-153 source/checkpoint chain differs before query")
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan153-fresh-context-length-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "preflight_sha256": sha256(PREFLIGHT),
            "face_sha256": sha256(FACE),
            "planned_endpoints": len(ENDPOINTS),
            "queries_per_endpoint": 480,
            "planned_queries": EXPECTED_QUERIES,
            "parameter_updates": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
            "confirmation_queries": 0,
            "can_promote": False,
        },
    )
    n150_result = read_json(N150_RESULT)
    models = {(int(row["seed"]), str(row["branch"])): row for row in n150_result["models"]}
    rows = []
    endpoint_summaries = []
    query_count = 0
    errors = []
    try:
        for seed, branch in ENDPOINTS:
            budget.guard(reserve=30)
            model_record = models[(seed, branch)]
            checkpoint = Path(model_record["final_checkpoint_path"]).resolve()
            model, optimizer, payload, fingerprint = n150._restore_n150_checkpoint(checkpoint)
            if fingerprint != model_record["final_bundle_fingerprint"]:
                raise Refuse(f"N1-153 final bundle load mismatch: {seed}/{branch}")
            before = n150._fingerprint(model, optimizer)
            endpoint_rows = []
            try:
                for base in face["items"]:
                    for length in LENGTHS:
                        budget.guard(reserve=5)
                        item = _face_from_manifest_item(base, length)
                        session = f"n1-plan153-{seed}-{branch}-{item['parent_id']}-l{length}"
                        if session in model._observations:
                            raise Refuse(f"N1-153 diagnostic session collision: {session}")
                        query_count += 1
                        try:
                            row = _measure_diagnostic(model, item, session)
                        finally:
                            model._observations.pop(session, None)
                        endpoint_rows.append(
                            {"seed": seed, "branch": branch, "event": "diagnostic_row", **row}
                        )
                if len(endpoint_rows) != 480:
                    raise Refuse(f"N1-153 endpoint query count differs: {seed}/{branch}")
                if n150._fingerprint(model, optimizer) != before:
                    raise Refuse(f"N1-153 diagnostic scoring changed full bundle: {seed}/{branch}")
                endpoint_summaries.extend(_summary_rows(endpoint_rows))
                rows.extend(endpoint_rows)
            finally:
                del model, optimizer, payload
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    source_unchanged = n150_builder.verify_hashes(input_hashes)
    if not source_unchanged:
        errors.append("N1-153 source/checkpoint hash changed during query")
    stage_cpu = time.process_time() - started + budget.children
    cumulative = float(preflight["cumulative_cpu_seconds"]) + stage_cpu
    complete = (
        not errors
        and query_count == EXPECTED_QUERIES
        and len(rows) == EXPECTED_QUERIES
        and len(endpoint_summaries) == len(ENDPOINTS) * len(LENGTHS) * len(KINDS)
        and source_unchanged
        and cumulative <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan153-v28-fresh-context-length-result-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "source_audit_sha256": sha256(SOURCE_AUDIT),
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "readiness_sha256": sha256(READINESS),
        "preflight_sha256": sha256(PREFLIGHT),
        "preflight_binding_sha256": sha256(PREFLIGHT_BINDING),
        "attempt_sha256": sha256(ATTEMPT),
        "source_hashes": input_hashes,
        "source_hashes_unchanged": source_unchanged,
        "endpoint_count": len(ENDPOINTS),
        "queries_per_endpoint": 480,
        "query_count": query_count,
        "diagnostic_rows": rows,
        "summary_by_endpoint_length_kind": endpoint_summaries,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "cpu_seconds": round(stage_cpu, 6),
        "prior_readiness_preflight_cpu_seconds": float(preflight["cumulative_cpu_seconds"]),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": f"N1-153 fresh-context长度诊断{('完整' if complete else '停止')}：{query_count}/2,880条，0训练/更新/dev/holdout/confirmation，can_promote=false。",
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-153 fresh-context paired length diagnostic"
    )
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
        "base_item_count",
        "endpoint_count",
        "query_count",
        "independent_restore_count",
        "model_queries",
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
