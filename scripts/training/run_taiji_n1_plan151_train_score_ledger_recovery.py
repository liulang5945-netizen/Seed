"""重建N1-150六个endpoint的train-only逐题评分台账；不训练。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_v27_length_robust_training_after_n149 as n150

PLAN = ROOT / "plans/reference/PLAN-N1-151_train_score_ledger_recovery_draft_20261020.md"
PLAN_SHA256 = "fdab96decd468f94820c6ab33abbd5c18e78ca78c599e7521a27db74cef1f9a7"
REPAIR_PLAN = ROOT / "plans/reference/PLAN-N1-151A_source_binding_return_correction_20261020.md"
REPAIR_PLAN_SHA256 = "4892cc62f197689e0437a104bd2e592696ed67863a7746d4a7cb36a3ca0bc965"
FAILURE_RECEIPT = ROOT / "reports/taiji_n1_plan151_preflight_binding_failure_20261020.json"
N150_RESULT = n150.RESULT
N150_RESULT_SHA256 = "e50e4b5aaf244ced4b5ef6232eb08a09206a5957f7d8f85abb1b3d841860ef0b"
N150_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-150_execution_closeout_20261020.md"
N150_AUDIT = ROOT / "reports/taiji_n1_plan150_v27_length_progress_integrity_audit_20261020.json"

CPU_LIMIT_SECONDS = 3_600.0
ORIGINAL_GATE_QUERIES_PER_ENDPOINT = 960
LENGTH_QUERIES_PER_ENDPOINT = 288
TOTAL_QUERIES_PER_ENDPOINT = ORIGINAL_GATE_QUERIES_PER_ENDPOINT + LENGTH_QUERIES_PER_ENDPOINT
ENDPOINTS = tuple((seed, branch) for seed in n150_builder.SEEDS for branch in n150_builder.BRANCHES)
EXPECTED_TOTAL_QUERIES = TOTAL_QUERIES_PER_ENDPOINT * len(ENDPOINTS)

READINESS = ROOT / "reports/taiji_n1_plan151a_train_score_ledger_recovery_readiness_20261020.json"
READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan151a_train_score_ledger_recovery_readiness_attempt_20261020.json"
)
SOURCE_BINDING = (
    ROOT / "reports/taiji_n1_plan151a_train_score_ledger_recovery_source_binding_20261020.json"
)
PREFLIGHT = ROOT / "reports/taiji_n1_plan151a_train_score_ledger_recovery_preflight_20261020.json"
PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan151a_train_score_ledger_recovery_preflight_attempt_20261020.json"
)
PREFLIGHT_BINDING = (
    ROOT / "reports/taiji_n1_plan151a_train_score_ledger_recovery_preflight_binding_20261020.json"
)
ATTEMPT = ROOT / "reports/taiji_n1_plan151a_train_score_ledger_recovery_attempt_20261020.json"
RESULT = ROOT / "reports/taiji_n1_plan151a_train_score_ledger_recovery_result_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan151_train_score_ledger_recovery_contract.py"

GATE_PARITY_KEYS = (
    "items",
    "free_exact",
    "free_by_kind",
    "pair_exact",
    "rename_free_exact",
    "rename_free_by_kind",
    "rename_pair_exact",
    "teacher_exact",
    "factorial",
)


class Refuse(RuntimeError):
    """冻结来源、endpoint或评分合同不成立。"""


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


def verify_hashes(source_hashes: dict[str, str]) -> bool:
    return all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in source_hashes.items()
    )


def _merge_hashes(target: dict[str, str], incoming: dict[str, str]) -> None:
    for raw_path, digest in incoming.items():
        path = str(Path(raw_path).resolve())
        old = target.get(path)
        if old is not None and old != digest:
            raise Refuse(f"conflicting N1-151 source hash: {path}")
        target[path] = digest


def _endpoint_records(result: dict) -> dict[tuple[int, str], dict]:
    records = {(int(row["seed"]), str(row["branch"])): row for row in result.get("models", [])}
    require(set(records) == set(ENDPOINTS), "N1-150 must contain exactly the six frozen endpoints")
    for endpoint, row in records.items():
        require(
            row.get("training_sufficient") is True,
            f"N1-150 endpoint did not pass its original gate: {endpoint}",
        )
        require(
            row.get("original_train_gate_queries") == ORIGINAL_GATE_QUERIES_PER_ENDPOINT,
            f"N1-150 original gate denominator differs: {endpoint}",
        )
        require(
            row.get("length_score_queries") == LENGTH_QUERIES_PER_ENDPOINT,
            f"N1-150 length denominator differs: {endpoint}",
        )
        checkpoint = Path(row["final_checkpoint_path"]).resolve()
        require(
            checkpoint.is_file() and sha256(checkpoint) == row["final_checkpoint_sha256"],
            f"N1-150 final checkpoint changed: {endpoint}",
        )
    return records


def _load_n150_source_context() -> tuple[dict, dict, dict, dict, dict]:
    parts = n150._verify_preflight()
    result = read_json(N150_RESULT)
    readiness, preflight, _binding, result, bound, schedule = unpack_n150_preflight(parts, result)
    require(sha256(N150_RESULT) == N150_RESULT_SHA256, "N1-150 result hash changed")
    require(
        result.get("complete") is True
        and result.get("status") == "complete"
        and result.get("training_updates") == 55_296
        and result.get("train_score_queries") == 7_488
        and result.get("source_hashes_unchanged") is True
        and result.get("training_sufficient_all_endpoints") is True
        and result.get("can_promote") is False,
        "N1-150 completed train-only source result is incomplete or outside its frozen boundary",
    )
    require(sha256(n150.PROGRESS) == result.get("progress_sha256"), "N1-150 progress SHA changed")
    _endpoint_records(result)
    return readiness, preflight, result, bound, schedule


def unpack_n150_preflight(
    parts: tuple, result_report: dict
) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Keep the third N1-150 return value as its binding; load result separately."""
    if len(parts) != 5:
        raise Refuse("N1-150 frozen preflight return arity changed")
    readiness, preflight, binding, bound, schedule = parts
    return readiness, preflight, binding, result_report, bound, schedule


def collect_source_hashes(result: dict) -> dict[str, str]:
    sources: dict[str, str] = {}
    _merge_hashes(sources, result.get("source_hashes", {}))
    _merge_hashes(
        sources,
        {
            str(N150_RESULT.resolve()): sha256(N150_RESULT),
            str(n150.ATTEMPT.resolve()): sha256(n150.ATTEMPT),
            str(n150.PROGRESS.resolve()): sha256(n150.PROGRESS),
            str(n150.READINESS.resolve()): sha256(n150.READINESS),
            str(n150.SOURCE_BINDING.resolve()): sha256(n150.SOURCE_BINDING),
            str(n150.PREFLIGHT.resolve()): sha256(n150.PREFLIGHT),
            str(n150.PREFLIGHT_BINDING.resolve()): sha256(n150.PREFLIGHT_BINDING),
            str(N150_CLOSEOUT.resolve()): sha256(N150_CLOSEOUT),
            str(N150_AUDIT.resolve()): sha256(N150_AUDIT),
            str(PLAN.resolve()): sha256(PLAN),
            str(REPAIR_PLAN.resolve()): sha256(REPAIR_PLAN),
            str(FAILURE_RECEIPT.resolve()): sha256(FAILURE_RECEIPT),
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(TEST.resolve()): sha256(TEST),
        },
    )
    for path, digest in result.get("new_checkpoint_hashes", {}).items():
        _merge_hashes(sources, {path: digest})
    if not verify_hashes(sources):
        raise Refuse("N1-150/N1-151A source, result, progress, or checkpoint hash changed")
    return dict(sorted(sources.items()))


def _gate_signature(summary: dict) -> dict:
    return {key: summary.get(key) for key in GATE_PARITY_KEYS}


def validate_gate_parity(recovered: dict, frozen: dict) -> None:
    if _gate_signature(recovered) != _gate_signature(frozen):
        raise Refuse("recovered N1-132 original gate integer summaries differ from N1-150")


def validate_row_coverage(rows: list[dict]) -> dict:
    counts = Counter((int(row["seed"]), str(row["branch"]), str(row["event"])) for row in rows)
    expected = {}
    for seed, branch in ENDPOINTS:
        expected[(seed, branch, "original_train_gate_row")] = ORIGINAL_GATE_QUERIES_PER_ENDPOINT
        expected[(seed, branch, "length_score_row")] = LENGTH_QUERIES_PER_ENDPOINT
    if dict(counts) != expected or len(rows) != EXPECTED_TOTAL_QUERIES:
        raise Refuse(f"N1-151 itemwise row denominator mismatch: {dict(counts)}")
    return {
        "endpoint_count": len(ENDPOINTS),
        "rows_per_endpoint": TOTAL_QUERIES_PER_ENDPOINT,
        "original_gate_rows_per_endpoint": ORIGINAL_GATE_QUERIES_PER_ENDPOINT,
        "length_rows_per_endpoint": LENGTH_QUERIES_PER_ENDPOINT,
        "total_rows": len(rows),
        "complete": True,
    }


def _readiness_source_context() -> tuple[dict, dict, dict, dict, dict, dict[str, str]]:
    readiness, preflight, n150_result, bound, schedule = _load_n150_source_context()
    sources = collect_source_hashes(n150_result)
    inputs = bound["inputs"]
    require(len(inputs.get("items", [])) == 192, "N1-151 requires the original 192 train tasks")
    require(
        all(item.get("split") == "train" for item in inputs["items"]),
        "N1-151 source must remain train-only",
    )
    selected = n150._length_score_items(inputs)
    require(
        {length: len(items) for length, items in selected.items()} == {8: 96, 16: 96, 32: 96},
        "N1-150A length sample selection differs",
    )
    return readiness, preflight, n150_result, bound, schedule, sources


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
        raise FileExistsError("PLAN-N1-151 outputs are write-once")
    started = time.process_time()
    _n150_readiness, _n150_preflight, n150_result, bound, schedule, sources = (
        _readiness_source_context()
    )
    require(verify_hashes(sources), "N1-151 source changed during no-query readiness")
    cpu = time.process_time() - started
    complete = cpu <= CPU_LIMIT_SECONDS and len(sources) >= 12
    report = {
        "schema": "n1-plan151a-train-score-ledger-readiness-v1",
        "status": "ready_for_final_checkpoint_preflight" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "repair_addendum_sha256": REPAIR_PLAN_SHA256,
        "n150_result_sha256": sha256(N150_RESULT),
        "n150_result_status": n150_result["status"],
        "n150_final_endpoint_count": len(n150_result["models"]),
        "endpoint_count": len(ENDPOINTS),
        "n150_final_checkpoint_hashes": {
            str(Path(row["final_checkpoint_path"]).resolve()): row["final_checkpoint_sha256"]
            for row in n150_result["models"]
        },
        "schedule_identity_sha256": schedule["manifest"]["schedule_sha256"],
        "planned_endpoints": [f"{seed}/{branch}" for seed, branch in ENDPOINTS],
        "queries_per_endpoint": TOTAL_QUERIES_PER_ENDPOINT,
        "original_gate_queries_per_endpoint": ORIGINAL_GATE_QUERIES_PER_ENDPOINT,
        "length_queries_per_endpoint": LENGTH_QUERIES_PER_ENDPOINT,
        "planned_repeated_train_only_queries": EXPECTED_TOTAL_QUERIES,
        "source_hashes": sources,
        "source_hashes_unchanged": True,
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
        "summary_zh": "N1-150来源、六枚final checkpoint、train-only题目与7,488条重复评分分母核验通过；0模型/checkpoint加载、0查询、0更新。",
    }
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan151a-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "repair_addendum_sha256": REPAIR_PLAN_SHA256,
            "source_hash_count": len(sources),
            "cpu_seconds": round(cpu, 6),
            "model_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
        },
    )
    write_new(READINESS, report)
    write_new(
        SOURCE_BINDING,
        {
            "schema": "n1-plan151a-readiness-source-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "repair_addendum_sha256": REPAIR_PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_hashes": sources,
            "can_promote": False,
        },
    )
    return report


def _verify_readiness() -> tuple[dict, dict, dict, dict, dict, dict[str, str]]:
    if not READINESS.is_file() or not SOURCE_BINDING.is_file():
        raise FileNotFoundError("N1-151 no-query readiness is required")
    report = read_json(READINESS)
    binding = read_json(SOURCE_BINDING)
    if (
        report.get("status") != "ready_for_final_checkpoint_preflight"
        or report.get("complete") is not True
        or report.get("plan_sha256") != PLAN_SHA256
        or report.get("repair_addendum_sha256") != REPAIR_PLAN_SHA256
        or report.get("source_hashes_unchanged") is not True
        or binding.get("status") != "complete"
        or binding.get("plan_sha256") != PLAN_SHA256
        or binding.get("repair_addendum_sha256") != REPAIR_PLAN_SHA256
        or binding.get("readiness_sha256") != sha256(READINESS)
        or binding.get("source_hashes") != report.get("source_hashes")
    ):
        raise Refuse("N1-151 readiness binding is stale or incomplete")
    n150_readiness, n150_preflight, n150_result, bound, schedule = _load_n150_source_context()
    sources = collect_source_hashes(n150_result)
    if sources != report.get("source_hashes") or not verify_hashes(sources):
        raise Refuse("N1-151 bound source/checkpoint hashes changed")
    if schedule["manifest"]["schedule_sha256"] != report.get("schedule_identity_sha256"):
        raise Refuse("N1-151 reused N1-150 train schedule changed")
    return n150_readiness, n150_preflight, n150_result, bound, schedule, sources


def checkpoint_preflight() -> dict:
    outputs = (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING)
    if any(path.exists() for path in outputs):
        raise FileExistsError("N1-151 final checkpoint restore preflight is write-once")
    started = time.process_time()
    n150_readiness, n150_preflight, n150_result, bound, schedule, sources = _verify_readiness()
    validation_cpu = time.process_time() - started
    readiness_cpu = float(read_json(READINESS)["cpu_seconds"])
    remaining = CPU_LIMIT_SECONDS - readiness_cpu - validation_cpu
    if remaining <= 0:
        raise TimeoutError("N1-151 CPU ceiling exhausted during readiness revalidation")
    budget = n132.CpuBudget(remaining)
    budget.guard(reserve=30)
    write_new(
        PREFLIGHT_ATTEMPT,
        {
            "schema": "n1-plan151a-checkpoint-preflight-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "repair_addendum_sha256": REPAIR_PLAN_SHA256,
            "final_checkpoint_count": len(ENDPOINTS),
            "model_queries": 0,
            "parameter_updates": 0,
        },
    )
    expected = {(int(row["seed"]), str(row["branch"])): row for row in n150_result["models"]}
    restores = []
    errors = []
    try:
        for seed, branch in ENDPOINTS:
            budget.guard(reserve=10)
            model_row = expected[(seed, branch)]
            checkpoint = Path(model_row["final_checkpoint_path"]).resolve()
            restored = n150._independent_restore(checkpoint, budget)
            if (
                restored.get("sha256") != model_row.get("final_checkpoint_sha256")
                or restored.get("fingerprint") != model_row.get("final_bundle_fingerprint")
                or restored.get("total_updates")
                != n150_builder.n132.TOTAL_UPDATES + n150_builder.UPDATES_PER_MODEL
                or restored.get("history_count") != model_row.get("final_history_count")
            ):
                raise Refuse(f"N1-151 independent final restore differs: {seed}/{branch}")
            restores.append(restored)
        if not verify_hashes(sources):
            raise Refuse("N1-151 sources changed during final restore preflight")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = time.process_time() - started + budget.children
    cumulative_cpu = readiness_cpu + cpu
    complete = not errors and len(restores) == 6 and cumulative_cpu <= CPU_LIMIT_SECONDS
    all_sources = dict(sources)
    all_sources[str(READINESS.resolve())] = sha256(READINESS)
    all_sources[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
    all_sources[str(PREFLIGHT_ATTEMPT.resolve())] = sha256(PREFLIGHT_ATTEMPT)
    report = {
        "schema": "n1-plan151a-final-checkpoint-preflight-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "repair_addendum_sha256": REPAIR_PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "preflight_attempt_sha256": sha256(PREFLIGHT_ATTEMPT),
        "source_hashes": all_sources,
        "source_hashes_unchanged": verify_hashes(all_sources),
        "independent_restores": restores,
        "endpoint_count": len(ENDPOINTS),
        "independent_restore_count": len(restores),
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "cpu_seconds": round(cpu, 6),
        "prior_readiness_cpu_seconds": round(readiness_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
    }
    report["summary_zh"] = (
        "六个N1-150 final full bundle已再次独立恢复6/6；0模型查询、0参数更新，可进入逐题计分。"
        if complete
        else "final bundle恢复前置停止；不计分、不重试。"
    )
    write_new(PREFLIGHT, report)
    write_new(
        PREFLIGHT_BINDING,
        {
            "schema": "n1-plan151a-final-checkpoint-preflight-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "repair_addendum_sha256": REPAIR_PLAN_SHA256,
            "preflight_sha256": sha256(PREFLIGHT),
            "source_hashes": all_sources,
            "can_promote": False,
        },
    )
    return report


def _length_rows(
    model, optimizer, seed: int, branch: str, inputs: dict, views: dict, budget: n132.CpuBudget
) -> list[dict]:
    before = n150._fingerprint(model, optimizer)
    rows = []
    selected = n150._length_score_items(inputs)
    for length in n150_builder.LENGTHS:
        for base_item in selected[length]:
            budget.guard(reserve=5)
            key = n150_builder._context_key(base_item)
            item = n150._apply_context(base_item, key, length, views)
            session = f"n1-plan151-ledger-{seed}-{branch}-{length}-{item['id']}"
            if session in model._observations:
                raise Refuse(f"N1-151 score session already exists: {session}")
            try:
                row = n150._measure_train_length_view(model, item, session)
            finally:
                model._observations.pop(session, None)
            rows.append({"event": "length_score_row", "seed": seed, "branch": branch, **row})
    if len(rows) != LENGTH_QUERIES_PER_ENDPOINT or n150._fingerprint(model, optimizer) != before:
        raise Refuse(f"N1-151 length score count or bundle fingerprint differs: {seed}/{branch}")
    return rows


def _score_endpoint(
    model_row: dict, bound: dict, schedule: dict, budget: n132.CpuBudget
) -> tuple[dict, list[dict]]:
    seed, branch = int(model_row["seed"]), str(model_row["branch"])
    checkpoint = Path(model_row["final_checkpoint_path"]).resolve()
    if sha256(checkpoint) != model_row["final_checkpoint_sha256"]:
        raise Refuse(f"N1-150 final checkpoint changed before scoring: {seed}/{branch}")
    model, optimizer, payload, fingerprint = n150._restore_n150_checkpoint(checkpoint)
    if fingerprint != model_row.get("final_bundle_fingerprint"):
        raise Refuse(f"N1-151 model load differs from N1-150 final: {seed}/{branch}")
    before = n150._fingerprint(model, optimizer)
    inputs = bound["inputs"]
    gate_rows, gate_summary, sufficient, gate_queries = n132._evaluate_train_only(
        bound["g"], model, optimizer, seed, inputs, budget
    )
    require(
        gate_queries == ORIGINAL_GATE_QUERIES_PER_ENDPOINT
        and len(gate_rows) == ORIGINAL_GATE_QUERIES_PER_ENDPOINT,
        f"N1-151 original gate denominator differs: {seed}/{branch}",
    )
    validate_gate_parity(gate_summary, model_row["original_train_gate"])
    require(
        sufficient is model_row["training_sufficient"] is True,
        f"N1-151 original gate sufficiency differs: {seed}/{branch}",
    )
    rows = [
        {"event": "original_train_gate_row", "seed": seed, "branch": branch, **row}
        for row in gate_rows
    ]
    length_rows = _length_rows(model, optimizer, seed, branch, inputs, schedule["views"], budget)
    rows.extend(length_rows)
    length_summary = n150._length_score_summary(length_rows)
    require(
        length_summary == model_row["train_only_length_summary"],
        f"N1-151 length aggregates differ from N1-150: {seed}/{branch}",
    )
    require(
        n150._fingerprint(model, optimizer) == before,
        f"N1-151 scoring changed the complete bundle: {seed}/{branch}",
    )
    total_steps = {int(optimizer.state[param]["step"]) for param in model.parameters()}
    require(
        total_steps == {n132.TOTAL_UPDATES + n150_builder.UPDATES_PER_MODEL},
        f"N1-151 final optimizer step changed: {seed}/{branch}",
    )
    record = {
        "seed": seed,
        "branch": branch,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "bundle_fingerprint_before": before,
        "bundle_fingerprint_after": n150._fingerprint(model, optimizer),
        "original_train_gate_queries": gate_queries,
        "length_score_queries": len(length_rows),
        "total_score_queries": len(rows),
        "original_train_gate": gate_summary,
        "training_sufficient": sufficient,
        "train_only_length_summary": length_summary,
        "aggregate_parity_with_n150": True,
        "parameter_updates": 0,
        "optimizer_steps_unchanged": True,
    }
    del model, optimizer, payload
    return record, rows


def _verify_preflight() -> tuple[dict, dict, dict, dict, dict, dict, dict[str, str]]:
    if not all(
        path.is_file() for path in (READINESS, SOURCE_BINDING, PREFLIGHT, PREFLIGHT_BINDING)
    ):
        raise FileNotFoundError("N1-151 readiness and six-final-restore preflight are required")
    readiness = read_json(READINESS)
    source_binding = read_json(SOURCE_BINDING)
    preflight = read_json(PREFLIGHT)
    preflight_binding = read_json(PREFLIGHT_BINDING)
    if (
        readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("repair_addendum_sha256") != REPAIR_PLAN_SHA256
        or source_binding.get("status") != "complete"
        or source_binding.get("repair_addendum_sha256") != REPAIR_PLAN_SHA256
        or source_binding.get("readiness_sha256") != sha256(READINESS)
        or source_binding.get("source_hashes") != readiness.get("source_hashes")
        or preflight.get("complete") is not True
        or preflight.get("independent_restore_count") != 6
        or preflight.get("plan_sha256") != PLAN_SHA256
        or preflight.get("repair_addendum_sha256") != REPAIR_PLAN_SHA256
        or preflight.get("source_hashes_unchanged") is not True
        or preflight.get("model_queries") != 0
        or preflight.get("parameter_updates") != 0
        or preflight_binding.get("status") != "complete"
        or preflight_binding.get("repair_addendum_sha256") != REPAIR_PLAN_SHA256
        or preflight_binding.get("preflight_sha256") != sha256(PREFLIGHT)
        or preflight_binding.get("source_hashes") != preflight.get("source_hashes")
    ):
        raise Refuse("N1-151 readiness/final restore preflight is incomplete or stale")
    if not verify_hashes(preflight.get("source_hashes", {})):
        raise Refuse("N1-151 bound source/checkpoint hashes changed before scoring")
    n150_readiness, n150_preflight, n150_result, bound, schedule = _load_n150_source_context()
    sources = collect_source_hashes(n150_result)
    return (
        readiness,
        preflight,
        n150_readiness,
        n150_preflight,
        n150_result,
        {"bound": bound, "schedule": schedule},
        sources,
    )


def execute_approved() -> dict:
    started = time.process_time()
    readiness, preflight, _n150_readiness, _n150_preflight, n150_result, runtime, sources = (
        _verify_preflight()
    )
    validation_cpu = time.process_time() - started
    remaining = CPU_LIMIT_SECONDS - float(preflight["cumulative_cpu_seconds"]) - validation_cpu
    if remaining <= 0:
        raise TimeoutError("N1-151 CPU ceiling exhausted during source revalidation")
    outputs = (ATTEMPT, RESULT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("N1-151 score ledger recovery is one-shot; no retry")
    budget = n132.CpuBudget(remaining)
    input_hashes = dict(preflight["source_hashes"])
    for path in (READINESS, SOURCE_BINDING, PREFLIGHT, PREFLIGHT_BINDING):
        input_hashes[str(Path(path).resolve())] = sha256(path)
    if not verify_hashes(input_hashes):
        raise Refuse("N1-151 source/checkpoint hashes differ before scoring")
    attempt = {
        "schema": "n1-plan151a-train-score-ledger-attempt-v1",
        "status": "started",
        "plan_sha256": PLAN_SHA256,
        "repair_addendum_sha256": REPAIR_PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "preflight_sha256": sha256(PREFLIGHT),
        "n150_result_sha256": sha256(N150_RESULT),
        "planned_endpoints": len(ENDPOINTS),
        "queries_per_endpoint": TOTAL_QUERIES_PER_ENDPOINT,
        "planned_repeated_train_only_queries": EXPECTED_TOTAL_QUERIES,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
    }
    write_new(ATTEMPT, attempt)
    rows = []
    endpoints = []
    errors = []
    try:
        bound, schedule = runtime["bound"], runtime["schedule"]
        if len(bound["inputs"].get("items", [])) != 192:
            raise Refuse("N1-151 train task count changed")
        n150_models = {(int(row["seed"]), str(row["branch"])): row for row in n150_result["models"]}
        for endpoint in ENDPOINTS:
            budget.guard(reserve=30)
            record, endpoint_rows = _score_endpoint(n150_models[endpoint], bound, schedule, budget)
            endpoints.append(record)
            rows.extend(endpoint_rows)
        coverage = validate_row_coverage(rows)
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        coverage = {"complete": False, "total_rows": len(rows)}

    input_hashes[str(ATTEMPT.resolve())] = sha256(ATTEMPT)
    source_unchanged = verify_hashes(input_hashes)
    if not source_unchanged:
        errors.append("N1-151 source, progress, or checkpoint hash changed during scoring")
    stage_cpu = time.process_time() - started + budget.children
    cumulative = float(preflight["cumulative_cpu_seconds"]) + stage_cpu
    query_count = len(rows)
    complete = (
        not errors
        and query_count == EXPECTED_TOTAL_QUERIES
        and len(endpoints) == len(ENDPOINTS)
        and coverage.get("complete") is True
        and all(row.get("aggregate_parity_with_n150") is True for row in endpoints)
        and source_unchanged
        and cumulative <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan151a-train-score-ledger-recovery-result-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "repair_addendum_sha256": REPAIR_PLAN_SHA256,
        "n150_result_sha256": sha256(N150_RESULT),
        "readiness_sha256": sha256(READINESS),
        "preflight_sha256": sha256(PREFLIGHT),
        "preflight_binding_sha256": sha256(PREFLIGHT_BINDING),
        "attempt_sha256": sha256(ATTEMPT),
        "source_hashes": input_hashes,
        "source_hashes_unchanged": source_unchanged,
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
        "score_row_coverage": coverage,
        "itemwise_rows": rows,
        "repeated_train_only_queries": query_count,
        "original_gate_rows_per_endpoint": ORIGINAL_GATE_QUERIES_PER_ENDPOINT,
        "length_rows_per_endpoint": LENGTH_QUERIES_PER_ENDPOINT,
        "total_rows_per_endpoint": TOTAL_QUERIES_PER_ENDPOINT,
        "n150_original_train_only_queries": n150_result["train_score_queries"],
        "combined_n150_plus_n151_train_only_queries": n150_result["train_score_queries"]
        + query_count,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "cpu_seconds": round(stage_cpu, 6),
        "prior_readiness_and_preflight_cpu_seconds": float(preflight["cumulative_cpu_seconds"]),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": f"N1-151A逐题评分台账{('完整' if complete else '停止')}：{query_count}/7,488条train-only重复评分；与N1-150聚合一致={all(row.get('aggregate_parity_with_n150') is True for row in endpoints)}，0更新/dev/holdout/confirmation，can_promote=false。",
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-151A source-binding repair and score ledger recovery"
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
        "endpoint_count",
        "independent_restore_count",
        "planned_repeated_train_only_queries",
        "repeated_train_only_queries",
        "score_row_coverage",
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
