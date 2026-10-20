"""执行PLAN-N1-162B解析器修正后的问句表面配对续训。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_plan162_question_surface_schedule_and_face as builder
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_plan155_short_unknown_context_length_training as n155
from scripts.training import run_taiji_n1_v27_length_robust_training_after_n149 as n150

PLAN_SHA256 = builder.PLAN_SHA256
CPU_LIMIT_SECONDS = builder.CPU_LIMIT_SECONDS
LINKED_CPU_LIMIT_SECONDS = builder.LINKED_CPU_LIMIT_SECONDS
PRIOR_LINKED_CPU_SECONDS = builder.PRIOR_LINKED_CPU_SECONDS
PRIOR_FAILED_STAGE_CPU_SECONDS = 270.4375
REPAIR_ADDENDUM = ROOT / "plans/reference/PLAN-N1-162B_首步解析器修正与隔离续执行_20261020.md"
REPAIR_ADDENDUM_SHA256 = "70f84638da596795e2c3c06e86839f56cba07232b975332b19907076d8f76847"
REPAIR_TEST = ROOT / "tests/taiji_native/test_n1_plan162b_question_surface_matched_continuation_contract.py"
ORIGINAL_ATTEMPT = builder.ATTEMPT
ORIGINAL_PROGRESS = builder.PROGRESS
ORIGINAL_RESULT = builder.RESULT
ORIGINAL_ZERO_PREFLIGHT = builder.ZERO_PREFLIGHT
ORIGINAL_ZERO_PREFLIGHT_ATTEMPT = builder.ZERO_PREFLIGHT_ATTEMPT
ORIGINAL_PARTIAL_CHECKPOINT = ROOT / "data/checkpoints/n1_plan162_question_surface_20261020/s11_surface_control_partial.pt"
ORIGINAL_RESULT_SHA256 = "b7a2519a2706a7c892bfca782a4823dc5d2e0a1e901bd6eb600ca7523132042a"
ORIGINAL_ATTEMPT_SHA256 = "86b1ced3c420ef053ab54108d44abe623107375119c51684beb93f6815a9511b"
ORIGINAL_PROGRESS_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
ORIGINAL_PARTIAL_SHA256 = "c5881905fd981203d317e3c44856a96c63d6aa3131d874270c69f1ce0000a2f4"
PRIOR_LINKED_CPU_SECONDS = 11_058.34375
SEEDS = builder.SEEDS
ARMS = builder.ARMS
ENDPOINTS = builder.ENDPOINTS
LENGTHS = builder.LENGTHS
KINDS = builder.KINDS
UPDATES_PER_MODEL = builder.UPDATES_PER_MODEL
TOTAL_UPDATES = builder.TOTAL_UPDATES
TRAIN_SCORE_QUERIES = builder.TRAIN_SCORE_QUERIES
TRAIN_GATE_QUERIES_PER_ENDPOINT = builder.TRAIN_GATE_QUERIES_PER_ENDPOINT
LENGTH_QUERIES_PER_ENDPOINT = builder.LENGTH_QUERIES_PER_ENDPOINT
FACE_QUERIES = builder.FACE_QUERIES
FACE_QUERIES_PER_ENDPOINT = builder.FACE_QUERIES_PER_ENDPOINT

SOURCE_AUDIT = builder.SOURCE_AUDIT
SCHEDULE = builder.SCHEDULE
READINESS = builder.READINESS
READINESS_ATTEMPT = builder.READINESS_ATTEMPT
SOURCE_BINDING = builder.SOURCE_BINDING
CHECKPOINT_ROOT = ROOT / "data/checkpoints/n1_plan162b_question_surface_20261020"
ZERO_PREFLIGHT = ROOT / "reports/taiji_n1_plan162b_question_surface_zero_preflight_20261020.json"
ZERO_PREFLIGHT_ATTEMPT = ROOT / "reports/taiji_n1_plan162b_question_surface_zero_preflight_attempt_20261020.json"
SCORING_PREFLIGHT = ROOT / "reports/taiji_n1_plan162b_question_surface_scoring_preflight_20261020.json"
ATTEMPT = ROOT / "reports/taiji_n1_plan162b_question_surface_attempt_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan162b_question_surface_progress_20261020.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan162b_question_surface_result_20261020.json"
TRAINING_GATE_RECEIPT = ROOT / "reports/taiji_n1_plan162b_question_surface_training_gate_receipt_20261020.json"
_ORIGINAL_QUERY_SUBJECT = n132._query_subject

# The frozen N1-162 builder is reused unchanged; redirect its face gate inputs to
# this isolated corrected attempt after the six training gates are recorded.
builder.RESULT = TRAINING_GATE_RECEIPT
builder.PROGRESS = PROGRESS
FACE = builder.FACE
FACE_AUDIT = builder.FACE_AUDIT
FACE_CHECKS = builder.FACE_CHECKS


class Refuse(RuntimeError):
    """PLAN-N1-162来源、保存恢复、训练或评分合同错误。"""


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


def write_jsonl(stream, value: dict) -> None:
    json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")


def _fingerprint(model, optimizer) -> str:
    return n132.e11.scorer.paired.fingerprint(model.bundle(optimizer))


def _restore(path: Path, restore_format: str = "n155"):
    if restore_format == "n150":
        return n150._restore_n150_checkpoint(Path(path))
    return n155._restore_checkpoint(Path(path))


def _restore_worker(path: Path, restore_format: str) -> dict:
    started = time.process_time()
    model, optimizer, payload, fingerprint = _restore(path, restore_format)
    result = {
        "complete": True,
        "equal": fingerprint == payload["bundle_fingerprint"],
        "fingerprint": fingerprint,
        "bundle_fingerprint": payload["bundle_fingerprint"],
        "seed": payload.get("seed"),
        "branch": payload.get("branch"),
        "total_updates": payload.get("total_updates"),
        "history_count": len(model._observations),
        "optimizer_parameters": len(optimizer.state),
        "restore_format": restore_format,
        "cpu_seconds": round(time.process_time() - started, 6),
    }
    del model, optimizer, payload
    return result


def _independent_restore(path: Path, restore_format: str, budget: n132.CpuBudget) -> dict:
    budget.guard(reserve=5)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--restore-worker", str(path), "--restore-format", restore_format],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise Refuse(f"independent {restore_format} restore failed: {process.stderr[-1200:]}")
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise Refuse("independent restore returned no report")
    row = json.loads(lines[-1])
    budget.children += float(row.get("cpu_seconds", 0.0))
    row["path"] = str(Path(path).resolve())
    row["sha256"] = sha256(path)
    if row.get("complete") is not True or row.get("equal") is not True:
        raise Refuse(f"independent restore equality failed: {path}")
    return row


def _repair_sources(sources: dict[str, str]) -> dict[str, str]:
    if sha256(REPAIR_ADDENDUM) != REPAIR_ADDENDUM_SHA256:
        raise Refuse("PLAN-N1-162B correction addendum SHA changed")
    require(REPAIR_TEST.is_file(), "PLAN-N1-162B contract test is missing")
    if (
        sha256(ORIGINAL_RESULT) != ORIGINAL_RESULT_SHA256
        or sha256(ORIGINAL_ATTEMPT) != ORIGINAL_ATTEMPT_SHA256
        or sha256(ORIGINAL_PROGRESS) != ORIGINAL_PROGRESS_SHA256
        or sha256(ORIGINAL_PARTIAL_CHECKPOINT) != ORIGINAL_PARTIAL_SHA256
    ):
        raise Refuse("sealed N1-162 zero-update failure evidence changed")
    result = dict(sources)
    for path in (
        REPAIR_ADDENDUM,
        Path(__file__).resolve(),
        REPAIR_TEST,
        ORIGINAL_ATTEMPT,
        ORIGINAL_PROGRESS,
        ORIGINAL_RESULT,
        ORIGINAL_ZERO_PREFLIGHT,
        ORIGINAL_ZERO_PREFLIGHT_ATTEMPT,
        ORIGINAL_PARTIAL_CHECKPOINT,
    ):
        resolved = str(Path(path).resolve())
        digest = sha256(Path(path))
        if resolved in result and result[resolved] != digest:
            raise Refuse(f"PLAN-N1-162B source union collision: {resolved}")
        result[resolved] = digest
    return dict(sorted(result.items()))


def _verify_readiness() -> tuple[dict, dict, dict, dict[str, str]]:
    readiness, manifest, runtime, sources = builder._verify_readiness()
    sources = _repair_sources(sources)
    if not all(path.is_file() for path in (ZERO_PREFLIGHT, ZERO_PREFLIGHT_ATTEMPT)):
        raise FileNotFoundError("PLAN-N1-162B six-zero-bundle preflight is required")
    preflight = read_json(ZERO_PREFLIGHT)
    attempt = read_json(ZERO_PREFLIGHT_ATTEMPT)
    if (
        preflight.get("complete") is not True
        or preflight.get("plan_sha256") != PLAN_SHA256
        or preflight.get("budget_addendum_sha256") != builder.BUDGET_ADDENDUM_SHA256
        or preflight.get("repair_addendum_sha256") != REPAIR_ADDENDUM_SHA256
        or preflight.get("zero_checkpoint_count") != 6
        or preflight.get("independent_restore_count") != 6
        or preflight.get("source_hashes_unchanged") is not True
        or preflight.get("model_queries") != 0
        or preflight.get("parameter_updates") != 0
        or preflight.get("optimizer_steps") != 0
        or preflight.get("readiness_sha256") != sha256(READINESS)
        or attempt.get("status") != "started"
        or attempt.get("plan_sha256") != PLAN_SHA256
        or preflight.get("attempt_sha256") != sha256(ZERO_PREFLIGHT_ATTEMPT)
    ):
        raise Refuse("PLAN-N1-162B zero checkpoint preflight is stale or incomplete")
    if not n150_builder.verify_hashes(sources) or not n150_builder.verify_hashes(preflight.get("source_hashes", {})):
        raise Refuse("PLAN-N1-162 source or zero-bundle SHA changed")
    return readiness, preflight, runtime, dict(preflight["source_hashes"])


def checkpoint_preflight() -> dict:
    if any(path.exists() for path in (ZERO_PREFLIGHT, ZERO_PREFLIGHT_ATTEMPT, ATTEMPT, PROGRESS, RESULT)):
        raise FileExistsError("PLAN-N1-162 preflight/training outputs are write-once")
    started = time.process_time()
    readiness, _manifest, runtime, sources = builder._verify_readiness()
    sources = _repair_sources(sources)
    validation_cpu = time.process_time() - started
    remaining_stage = CPU_LIMIT_SECONDS - PRIOR_FAILED_STAGE_CPU_SECONDS - validation_cpu
    remaining_linked = LINKED_CPU_LIMIT_SECONDS - PRIOR_LINKED_CPU_SECONDS - PRIOR_FAILED_STAGE_CPU_SECONDS - validation_cpu
    remaining = min(remaining_stage, remaining_linked)
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-162B CPU budget exhausted before zero checkpoint preflight")
    budget = n132.CpuBudget(remaining)
    budget.guard(reserve=30)
    if CHECKPOINT_ROOT.exists():
        raise FileExistsError(f"PLAN-N1-162 write-once checkpoint root exists: {CHECKPOINT_ROOT}")
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=False)
    write_new(ZERO_PREFLIGHT_ATTEMPT, {
        "schema": "n1-plan162b-zero-checkpoint-preflight-attempt-v1",
        "status": "started",
        "plan_sha256": PLAN_SHA256,
        "budget_addendum_sha256": builder.BUDGET_ADDENDUM_SHA256,
        "repair_addendum_sha256": REPAIR_ADDENDUM_SHA256,
        "prior_failed_result_sha256": ORIGINAL_RESULT_SHA256,
        "prior_failed_attempt_sha256": ORIGINAL_ATTEMPT_SHA256,
        "readiness_sha256": sha256(READINESS),
        "planned_zero_bundles": 6,
        "model_queries": 0,
        "parameter_updates": 0,
        "can_promote": False,
    })
    candidates = runtime["n155a_candidates"]
    zero_rows = []
    restores = []
    pair_fingerprints = {}
    errors = []
    try:
        for seed in SEEDS:
            budget.guard(reserve=20)
            source = candidates[seed]
            source_path = Path(source["final_checkpoint_path"]).resolve()
            source_hash = source["final_checkpoint_sha256"]
            if not source_path.is_file() or sha256(source_path) != source_hash:
                raise Refuse(f"N1-155A seed{seed} candidate parent changed")
            model, optimizer, payload, fingerprint = n155._restore_checkpoint(source_path)
            if fingerprint != source.get("final_bundle_fingerprint") or payload.get("branch") != "unknown8_rebalanced":
                raise Refuse(f"N1-155A seed{seed} candidate bundle differs from frozen result")
            if any(int(optimizer.state[param]["step"]) != int(payload["total_updates"]) for param in model.parameters()):
                raise Refuse(f"N1-155A seed{seed} optimizer state does not match total updates")
            pair_fingerprints[seed] = fingerprint
            del model, optimizer, payload
            for arm in ARMS:
                zero_path = CHECKPOINT_ROOT / f"s{seed}_{arm}_zero.pt"
                if zero_path.exists():
                    raise FileExistsError(f"PLAN-N1-162 write-once zero bundle exists: {zero_path}")
                shutil.copyfile(source_path, zero_path)
                if sha256(zero_path) != source_hash:
                    raise Refuse(f"N1-162B zero bundle is not byte-identical to parent: {seed}/{arm}")
                zero_rows.append({
                    "seed": seed,
                    "arm": arm,
                    "path": str(zero_path.resolve()),
                    "sha256": source_hash,
                    "bundle_fingerprint": fingerprint,
                    "parent_checkpoint_path": str(source_path),
                    "parent_checkpoint_sha256": source_hash,
                    "parent_total_updates": source["total_updates"],
                    "initial_history_count": source["final_history_count"],
                })
        for seed in SEEDS:
            if len({row["bundle_fingerprint"] for row in zero_rows if row["seed"] == seed}) != 1:
                    raise Refuse(f"N1-162B zero arms differ at seed{seed}")
        for row in zero_rows:
            budget.guard(reserve=5)
            restored = _independent_restore(Path(row["path"]), "n155", budget)
            if restored["fingerprint"] != row["bundle_fingerprint"] or restored["seed"] != row["seed"] or restored["branch"] != "unknown8_rebalanced":
                raise Refuse(f"N1-162 zero bundle restore identity differs: {row['seed']}/{row['arm']}")
            restores.append({"seed": row["seed"], "arm": row["arm"], **restored})
        if len(zero_rows) != 6 or len(restores) != 6:
                raise Refuse("N1-162B requires six zero full bundles and six independent restores")
        if not n150_builder.verify_hashes(sources):
            raise Refuse("N1-162B source changed during zero preflight")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    source_hashes = dict(sources)
    source_hashes.update({row["path"]: row["sha256"] for row in zero_rows})
    cpu = time.process_time() - started + budget.children
    stage_cpu = PRIOR_FAILED_STAGE_CPU_SECONDS + cpu
    linked_cpu = PRIOR_LINKED_CPU_SECONDS + stage_cpu
    complete = not errors and len(zero_rows) == len(restores) == 6 and stage_cpu <= CPU_LIMIT_SECONDS and linked_cpu <= LINKED_CPU_LIMIT_SECONDS
    preflight = {
        "schema": "n1-plan162b-zero-checkpoint-preflight-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "budget_addendum_sha256": builder.BUDGET_ADDENDUM_SHA256,
        "repair_addendum_sha256": REPAIR_ADDENDUM_SHA256,
        "prior_failed_result_sha256": ORIGINAL_RESULT_SHA256,
        "prior_failed_attempt_sha256": ORIGINAL_ATTEMPT_SHA256,
        "readiness_sha256": sha256(READINESS),
        "attempt_sha256": sha256(ZERO_PREFLIGHT_ATTEMPT),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": n150_builder.verify_hashes(source_hashes),
        "zero_checkpoints": zero_rows,
        "independent_restores": restores,
        "zero_checkpoint_count": len(zero_rows),
        "independent_restore_count": len(restores),
        "arms_equal_at_zero_per_seed": True,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "cpu_seconds": round(cpu, 6),
        "readiness_cpu_seconds": float(readiness["cpu_seconds"]),
        "prior_failed_attempt_stage_cpu_seconds": PRIOR_FAILED_STAGE_CPU_SECONDS,
        "cumulative_stage_cpu_seconds": round(stage_cpu, 6),
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cumulative_linked_cpu_seconds": round(linked_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
    }
    preflight["summary_zh"] = "N1-162B六个zero full bundle逐seed按字节与指纹匹配N1-155A候选，并独立恢复6/6；零查询、零更新。" if complete else "N1-162B zero bundle前置停止；不训练、不查询、不重试。"
    write_new(ZERO_PREFLIGHT, preflight)
    return preflight


def _question_subject(question: str) -> str:
    for template in builder.TRAINING_TEMPLATES.values():
        prefix, suffix = template.split("{entity}")
        if question.startswith(prefix) and question.endswith(suffix):
            subject = question[len(prefix) : len(question) - len(suffix) if suffix else None]
            if re.fullmatch(r"[a-z]{10}", subject):
                return subject
            raise ValueError(f"invalid entity in scheduled question: {question!r}")
    return _ORIGINAL_QUERY_SUBJECT(question)


def _apply_question_template(item: dict, template_id: str) -> dict:
    if template_id == "original":
        return item
    template = builder.TRAINING_TEMPLATES.get(template_id)
    if template is None:
        raise Refuse(f"unknown N1-162 training question template: {template_id}")
    target = n132._query_subject(item["question"])
    item["question"] = template.format(entity=target)
    return item


def _training_step(model, optimizer, item: dict, second: dict | None, session: str, g: dict) -> dict:
    original_subject = n132._query_subject
    n132._query_subject = _question_subject
    try:
        return n132._training_step(model, optimizer, item, second, session, g)
    finally:
        n132._query_subject = original_subject


def _score_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["endpoint_family"]), int(row["seed"]), str(row["branch"]), str(row["kind"]), str(row["question_template_id"]), int(row["context_length"]))].append(row)
    summary = []
    for family, seed, branch in _scoring_endpoint_keys():
        for kind in KINDS:
            for template_id in _face_templates_for_kind(kind):
                for length in LENGTHS:
                    items = groups[(family, seed, branch, kind, template_id, length)]
                    require(len(items) == 20, f"N1-162 score stratum differs: {family}/{seed}/{branch}/{kind}/{template_id}/{length}")
                    summary.append({
                        "endpoint_family": family,
                        "seed": seed,
                        "branch": branch,
                        "kind": kind,
                        "question_template_id": template_id,
                        "context_length": length,
                        "items": len(items),
                        "free_exact": sum(row.get("free_exact") is True for row in items),
                        "teacher_exact": sum(row.get("teacher_exact") is True for row in items),
                        "literal_unknown": sum(row.get("raw_hex") == b"unknown".hex() for row in items),
                    })
    return summary


def _face_templates_for_kind(kind: str) -> tuple[str, str]:
    if kind in builder.MATERIAL_KINDS:
        return "material_associated", "material_belongs"
    return "direction_after_turning", "direction_final"


def _scoring_endpoint_keys() -> list[tuple[str, int, str]]:
    return [
        *(('n1_150_parent', seed, 'length_mixed') for seed in SEEDS),
        *(('n1_155a_candidate', seed, 'unknown8_rebalanced') for seed in SEEDS),
        *(('n1_162', seed, arm) for seed, arm in ENDPOINTS),
    ]


def _build_scoring_endpoints(runtime: dict, final_records: list[dict]) -> list[dict]:
    n150_result = runtime["n155_context"]["n150_result"]
    parent_rows = {
        int(row["seed"]): row
        for row in n150_result.get("models", [])
        if row.get("branch") == "length_mixed" and row.get("training_sufficient") is True
    }
    candidate_rows = runtime["n155a_candidates"]
    final_rows = {(int(row["seed"]), str(row["arm"])): row for row in final_records}
    require(set(parent_rows) == set(SEEDS), "N1-162 requires three sufficient N1-150 reference parents")
    require(set(candidate_rows) == set(SEEDS), "N1-162 requires three N1-155A candidate references")
    require(set(final_rows) == set(ENDPOINTS), "N1-162 requires six final arm endpoints")
    endpoints = []
    for seed in SEEDS:
        parent = parent_rows[seed]
        endpoints.append({"endpoint_family": "n1_150_parent", "seed": seed, "branch": "length_mixed", "path": parent["final_checkpoint_path"], "sha256": parent["final_checkpoint_sha256"], "fingerprint": parent["final_bundle_fingerprint"], "restore_format": "n150"})
    for seed in SEEDS:
        candidate = candidate_rows[seed]
        endpoints.append({"endpoint_family": "n1_155a_candidate", "seed": seed, "branch": "unknown8_rebalanced", "path": candidate["final_checkpoint_path"], "sha256": candidate["final_checkpoint_sha256"], "fingerprint": candidate["final_bundle_fingerprint"], "restore_format": "n155"})
    for seed, arm in ENDPOINTS:
        final = final_rows[(seed, arm)]
        endpoints.append({"endpoint_family": "n1_162", "seed": seed, "branch": arm, "path": final["final_checkpoint_path"], "sha256": final["final_checkpoint_sha256"], "fingerprint": final["final_bundle_fingerprint"], "restore_format": "n155"})
    return endpoints


def _independent_scoring_preflight(endpoints: list[dict], budget: n132.CpuBudget) -> dict:
    if SCORING_PREFLIGHT.exists():
        raise FileExistsError("N1-162 12-endpoint scoring preflight is write-once")
    restores = []
    errors = []
    try:
        if len(endpoints) != 12:
            raise Refuse("N1-162 scoring preflight must cover twelve endpoints")
        for endpoint in endpoints:
            budget.guard(reserve=5)
            path = Path(endpoint["path"]).resolve()
            if not path.is_file() or sha256(path) != endpoint["sha256"]:
                raise Refuse(f"N1-162 scoring checkpoint SHA changed: {path}")
            restored = _independent_restore(path, endpoint["restore_format"], budget)
            if restored["fingerprint"] != endpoint["fingerprint"] or restored["seed"] != endpoint["seed"]:
                raise Refuse(f"N1-162 scoring checkpoint restore differs: {endpoint['endpoint_family']}/{endpoint['seed']}/{endpoint['branch']}")
            restores.append({**endpoint, "restore": restored})
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    report = {
        "schema": "n1-plan162-twelve-endpoint-scoring-preflight-v1",
        "status": "complete" if not errors and len(restores) == 12 else "stopped",
        "complete": not errors and len(restores) == 12,
        "plan_sha256": PLAN_SHA256,
        "endpoint_count": len(endpoints),
        "independent_restore_count": len(restores),
        "restores": restores,
        "errors": errors,
        "model_queries": 0,
        "parameter_updates": 0,
        "cpu_seconds": round(budget.used(), 6),
        "can_promote": False,
    }
    write_new(SCORING_PREFLIGHT, report)
    return report


def _score_face(endpoints: list[dict], face: dict, budget: n132.CpuBudget, progress) -> tuple[list[dict], int, list[str]]:
    rows = []
    queries = 0
    errors = []
    for endpoint in endpoints:
        model = optimizer = payload = None
        endpoint_rows = []
        try:
            path = Path(endpoint["path"]).resolve()
            if sha256(path) != endpoint["sha256"]:
                raise Refuse(f"N1-162 checkpoint changed before scoring: {path}")
            model, optimizer, payload, fingerprint = _restore(path, endpoint["restore_format"])
            if fingerprint != endpoint["fingerprint"]:
                raise Refuse(f"N1-162 bundle fingerprint changed before scoring: {endpoint['endpoint_family']}/{endpoint['seed']}/{endpoint['branch']}")
            before = _fingerprint(model, optimizer)
            for base in face["items"]:
                for length in LENGTHS:
                    budget.guard(reserve=3)
                    item = {**base, "observations": list(base["observations_by_length"][str(length)]), "context_length": length}
                    session = f"n1-plan162-{endpoint['endpoint_family']}-{endpoint['seed']}-{endpoint['branch']}-{item['parent_id']}-l{length}"
                    if session in model._observations:
                        raise Refuse(f"N1-162 scoring session collision: {session}")
                    queries += 1
                    try:
                        scored = n155._measure_fresh_diagnostic(model, item, session)
                    finally:
                        model._observations.pop(session, None)
                    row = {
                        "endpoint_family": endpoint["endpoint_family"],
                        "seed": endpoint["seed"],
                        "branch": endpoint["branch"],
                        "question_template_id": base["question_template_id"],
                        **scored,
                    }
                    endpoint_rows.append(row)
                    write_jsonl(progress, {"event": "fresh_face_score_row", **row})
                    if queries % 64 == 0:
                        progress.flush()
            if len(endpoint_rows) != FACE_QUERIES_PER_ENDPOINT:
                raise Refuse(f"N1-162 endpoint query denominator differs: {endpoint['endpoint_family']}/{endpoint['seed']}/{endpoint['branch']}")
            rows.extend(endpoint_rows)
            if _fingerprint(model, optimizer) != before:
                raise Refuse(f"N1-162 diagnostic scoring changed bundle: {endpoint['endpoint_family']}/{endpoint['seed']}/{endpoint['branch']}")
        except BaseException as exc:
            if endpoint_rows and not any(row.get("endpoint_family") == endpoint["endpoint_family"] and row.get("seed") == endpoint["seed"] and row.get("branch") == endpoint["branch"] for row in rows):
                rows.extend(endpoint_rows)
            errors.append(f"{endpoint['endpoint_family']}/{endpoint['seed']}/{endpoint['branch']}: {type(exc).__name__}: {exc}")
            break
        finally:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            if payload is not None:
                del payload
    return rows, queries, errors


def _validate_progress(path: Path, updates: int, scores: int, fresh: int) -> dict:
    counts = Counter()
    malformed = []
    total = 0
    if not path.is_file():
        raise Refuse("N1-162 progress log is missing")
    for number, line in enumerate(path.open(encoding="utf-8"), 1):
        total += 1
        try:
            counts[str(json.loads(line).get("event"))] += 1
        except Exception as exc:
            malformed.append({"line": number, "error": type(exc).__name__})
    if malformed or counts["training_update"] != updates or counts["original_train_gate_row"] + counts["length_score_row"] != scores or counts["fresh_face_score_row"] != fresh:
        raise Refuse(f"N1-162 progress ledger counts differ: {dict(counts)} malformed={len(malformed)}")
    return {"complete": True, "line_count": total, "event_counts": dict(sorted(counts.items())), "malformed_rows": 0, "sha256": sha256(path)}


def execute_approved() -> dict:
    started = time.process_time()
    readiness, preflight, runtime, base_sources = _verify_readiness()
    validation_cpu = time.process_time() - started
    stage_so_far = PRIOR_FAILED_STAGE_CPU_SECONDS + float(preflight["cpu_seconds"]) + validation_cpu
    remaining = min(CPU_LIMIT_SECONDS - stage_so_far, LINKED_CPU_LIMIT_SECONDS - PRIOR_LINKED_CPU_SECONDS - stage_so_far)
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-162B stage/linked CPU budget exhausted before execution")
    if any(path.exists() for path in (ATTEMPT, PROGRESS, RESULT, SCORING_PREFLIGHT)):
        raise FileExistsError("PLAN-N1-162B training/scoring attempt is write-once")
    budget = n132.CpuBudget(remaining)
    source_hashes = dict(base_sources)
    for path in (SOURCE_AUDIT, SCHEDULE, READINESS, READINESS_ATTEMPT, SOURCE_BINDING, ZERO_PREFLIGHT, ZERO_PREFLIGHT_ATTEMPT):
        source_hashes[str(Path(path).resolve())] = sha256(path)
    zero_rows = preflight.get("zero_checkpoints", [])
    zero_map = {(int(row["seed"]), str(row["arm"])): row for row in zero_rows}
    if set(zero_map) != set(ENDPOINTS) or len(zero_rows) != 6:
        raise Refuse("PLAN-N1-162B six zero bundles are required before training")
    source_hashes.update({row["path"]: row["sha256"] for row in zero_rows})
    if not n150_builder.verify_hashes(source_hashes):
        raise Refuse("PLAN-N1-162B source/zero bundle changed before execution")
    attempt = {
        "schema": "n1-plan162b-question-surface-attempt-v1",
        "status": "started",
        "plan_sha256": PLAN_SHA256,
        "budget_addendum_sha256": builder.BUDGET_ADDENDUM_SHA256,
        "repair_addendum_sha256": REPAIR_ADDENDUM_SHA256,
        "prior_failed_result_sha256": ORIGINAL_RESULT_SHA256,
        "prior_failed_attempt_cpu_seconds": PRIOR_FAILED_STAGE_CPU_SECONDS,
        "readiness_sha256": sha256(READINESS),
        "zero_preflight_sha256": sha256(ZERO_PREFLIGHT),
        "planned_training_updates": TOTAL_UPDATES,
        "planned_train_score_queries": TRAIN_SCORE_QUERIES,
        "planned_fresh_face_queries": FACE_QUERIES,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
    }
    write_new(ATTEMPT, attempt)
    progress = PROGRESS.open("x", encoding="utf-8", newline="\n")
    update_count = 0
    train_score_queries = 0
    fresh_queries = 0
    final_records = []
    fresh_rows = []
    summaries = []
    errors = []
    active = None
    model = optimizer = payload = None
    partial_checkpoint = None
    face_status = "not_reached"
    scoring_preflight = None
    try:
        schedule_rows = read_json(SCHEDULE)["schedule_rows"]
        schedule = runtime["n155_schedule"]
        if len(schedule_rows) != UPDATES_PER_MODEL or len(schedule["slots"]) != UPDATES_PER_MODEL:
            raise Refuse("N1-162 runtime schedule does not contain 9,216 slots")
        for seed, arm in ENDPOINTS:
            budget.guard(reserve=60)
            zero = zero_map[(seed, arm)]
            model, optimizer, payload, fingerprint = n155._restore_checkpoint(Path(zero["path"]))
            initial_total = int(payload["total_updates"])
            initial_additional = int(payload["additional_updates"])
            initial_history = len(model._observations)
            if fingerprint != zero["bundle_fingerprint"] or initial_history != int(zero["initial_history_count"]):
                raise Refuse(f"N1-162 zero restore differs from preflight: {seed}/{arm}")
            parent_path = Path(payload["parent_checkpoint_path"]).resolve()
            parent_hash = str(payload["parent_checkpoint_sha256"])
            parent_total = int(payload["parent_total_updates"])
            if initial_total != parent_total + initial_additional:
                raise Refuse(f"N1-162 inherited optimizer lineage differs: {seed}/{arm}")
            active = {"seed": seed, "arm": arm, "completed_updates": 0, "payload": payload, "parent_path": parent_path, "parent_hash": parent_hash, "parent_total": parent_total}
            metrics = Counter()
            length_counts = Counter()
            template_counts = Counter()
            for row in schedule_rows:
                step = int(row["step"])
                if step >= UPDATES_PER_MODEL:
                    raise Refuse("N1-162 schedule contains out-of-range step")
                slot = schedule["slots"][step]
                if slot["item"].get("id") != row["item_id"] or slot.get("item_key") != row["item_key"] or slot.get("second_id") != row.get("second_id"):
                    raise Refuse(f"N1-162 schedule/source slot identity mismatch at {step}")
                length = int(row["context_length"])
                template_id = row["control_template_id"] if arm == "surface_control" else row["augmented_template_id"]
                item = n150._apply_context(slot["item"], slot["item_key"], length, schedule["views"])
                second = None
                if slot.get("second") is not None:
                    second = n150._with_observations(slot["second"], item["observations"], length)
                if template_id != "original":
                    target = n132._query_subject(item["question"])
                    item["question"] = builder.TRAINING_TEMPLATES[template_id].format(entity=target)
                    if second is not None:
                        second_target = n132._query_subject(second["question"])
                        second["question"] = builder.TRAINING_TEMPLATES[template_id].format(entity=second_target)
                if item.get("kind") == "unknown" and second is None:
                    raise Refuse(f"N1-162 unknown paired question is missing: {seed}/{arm}/{step}")
                budget.guard(reserve=12)
                result = _training_step(
                    model,
                    optimizer,
                    item,
                    second,
                    f"n1-plan162-{seed}-{arm}-{step}",
                    runtime["n155_bound"]["g"],
                )
                active["completed_updates"] += 1
                update_count += 1
                length_counts[length] += 1
                template_counts[(row["kind"], length, template_id)] += 1
                for key in ("total", "base_total", "qa", "observation", "binding"):
                    metrics[key] += float(result[key])
                write_jsonl(progress, {
                    "event": "training_update",
                    "seed": seed,
                    "arm": arm,
                    "step": step,
                    "parent_id": row["parent_id"],
                    "route": row["route"],
                    "kind": row["kind"],
                    "context_length": length,
                    "question_template_id": template_id,
                    "item_id": item["id"],
                    "question": item["question"],
                    "second_id": None if second is None else second["id"],
                    "second_question": None if second is None else second["question"],
                    **result,
                })
                if (step + 1) % 64 == 0:
                    progress.flush()
                if (step + 1) % 512 == 0:
                    print(f"plan162b seed={seed} arm={arm} step={step+1}/{UPDATES_PER_MODEL} cpu={budget.used():.1f}", flush=True)
            expected_lengths = Counter({8: 3_456, 16: 2_880, 32: 2_880})
            if active["completed_updates"] != UPDATES_PER_MODEL or length_counts != expected_lengths:
                raise Refuse(f"N1-162 update/context counts differ: {seed}/{arm}")
            total_steps = {int(optimizer.state[param]["step"]) for param in model.parameters()}
            if total_steps != {initial_total + UPDATES_PER_MODEL} or len(model._observations) != initial_history + UPDATES_PER_MODEL:
                raise Refuse(f"N1-162 final optimizer/history count differs: {seed}/{arm}")
            final_path = CHECKPOINT_ROOT / f"s{seed}_{arm}_final.pt"
            final_payload = n155._checkpoint_payload(
                model,
                optimizer,
                seed=seed,
                branch="unknown8_rebalanced",
                parent_path=parent_path,
                parent_hash=parent_hash,
                parent_total=parent_total,
                additional_updates=initial_additional + UPDATES_PER_MODEL,
            )
            if final_path.exists():
                raise FileExistsError(f"N1-162 write-once final checkpoint exists: {final_path}")
            with final_path.open("xb") as stream:
                torch.save(final_payload, stream)
                stream.flush()
            final_sha = sha256(final_path)
            final_restore = _independent_restore(final_path, "n155", budget)
            if final_restore["fingerprint"] != final_payload["bundle_fingerprint"]:
                raise Refuse(f"N1-162 final full bundle failed restore: {seed}/{arm}")
            before_score = _fingerprint(model, optimizer)
            gate_rows, gate_summary, sufficient, gate_queries = n132._evaluate_train_only(
                runtime["n155_bound"]["g"], model, optimizer, seed, runtime["n155_bound"]["inputs"], budget
            )
            require(gate_queries == TRAIN_GATE_QUERIES_PER_ENDPOINT and len(gate_rows) == TRAIN_GATE_QUERIES_PER_ENDPOINT, f"N1-162 original gate denominator differs: {seed}/{arm}")
            for score_row in gate_rows:
                write_jsonl(progress, {"event": "original_train_gate_row", "seed": seed, "arm": arm, **score_row})
            length_rows = n155._train_length_rows(model, optimizer, seed, "unknown8_rebalanced", runtime["n155_bound"]["inputs"], schedule["views"], budget)
            if len(length_rows) != LENGTH_QUERIES_PER_ENDPOINT:
                raise Refuse(f"N1-162 train length score denominator differs: {seed}/{arm}")
            for score_row in length_rows:
                write_jsonl(progress, {"event": "length_score_row", "seed": seed, "arm": arm, **score_row})
            train_score_queries += gate_queries + len(length_rows)
            if _fingerprint(model, optimizer) != before_score:
                raise Refuse(f"N1-162 train-only scoring mutated final bundle: {seed}/{arm}")
            final_records.append({
                "seed": seed,
                "arm": arm,
                "branch": "unknown8_rebalanced",
                "parent_checkpoint_path": str(parent_path),
                "parent_checkpoint_sha256": parent_hash,
                "source_n155a_checkpoint_path": zero["parent_checkpoint_path"],
                "source_n155a_checkpoint_sha256": zero["parent_checkpoint_sha256"],
                "source_n155a_total_updates": initial_total,
                "additional_updates_this_plan": UPDATES_PER_MODEL,
                "cumulative_additional_updates_from_n150": initial_additional + UPDATES_PER_MODEL,
                "total_updates": parent_total + initial_additional + UPDATES_PER_MODEL,
                "initial_history_count": initial_history,
                "final_history_count": len(model._observations),
                "optimizer_step_values": sorted(total_steps),
                "context_length_update_counts": dict(sorted(length_counts.items())),
                "question_template_counts": {f"{kind}/{length}/{template}": count for (kind, length, template), count in sorted(template_counts.items())},
                "loss_means": {key: float(metrics[key]) / UPDATES_PER_MODEL for key in ("total", "base_total", "qa", "observation", "binding")},
                "original_train_gate": gate_summary,
                "training_sufficient": sufficient,
                "original_train_gate_queries": gate_queries,
                "length_score_queries": len(length_rows),
                "final_checkpoint_path": str(final_path.resolve()),
                "final_checkpoint_sha256": final_sha,
                "final_bundle_fingerprint": final_payload["bundle_fingerprint"],
                "final_independent_restore": final_restore,
            })
            del model, optimizer, payload
            model = optimizer = payload = None
            active = None
        progress.flush()
        training_sufficient = len(final_records) == 6 and all(record["training_sufficient"] for record in final_records)
        if training_sufficient:
            if TRAINING_GATE_RECEIPT.exists():
                raise FileExistsError("N1-162B training gate receipt is write-once")
            write_new(TRAINING_GATE_RECEIPT, {
                "schema": "n1-plan162b-training-gate-receipt-v1",
                "status": "complete",
                "plan_sha256": PLAN_SHA256,
                "budget_addendum_sha256": builder.BUDGET_ADDENDUM_SHA256,
                "repair_addendum_sha256": REPAIR_ADDENDUM_SHA256,
                "training_complete": len(final_records) == 6 and update_count == TOTAL_UPDATES and train_score_queries == TRAIN_SCORE_QUERIES,
                "training_sufficient_all_endpoints": True,
                "training_updates": update_count,
                "train_score_queries": train_score_queries,
                "progress_sha256": sha256(PROGRESS),
                "models": final_records,
                "can_promote": False,
            })
            endpoints = _build_scoring_endpoints(runtime, final_records)
            scoring_preflight = _independent_scoring_preflight(endpoints, budget)
            if scoring_preflight.get("complete") is not True:
                raise Refuse("N1-162B twelve-endpoint restore preflight failed; face/query not reached")
            final_hashes = {row["final_checkpoint_path"]: row["final_checkpoint_sha256"] for row in final_records}
            progress.flush()
            face_checks = builder.prepare_fresh_face(final_checkpoint_hashes=final_hashes, training_progress_sha256=sha256(PROGRESS))
            face = read_json(FACE)
            if face_checks.get("source_hashes_unchanged") is not True or face_checks.get("face_contract", {}).get("total_queries") != FACE_QUERIES:
                raise Refuse("N1-162B seed132 face source/query contract differs")
            fresh_rows, fresh_queries, face_errors = _score_face(endpoints, face, budget, progress)
            errors.extend(face_errors)
            if not face_errors and fresh_queries == FACE_QUERIES and len(fresh_rows) == FACE_QUERIES:
                summaries = _score_summary(fresh_rows)
                face_status = "complete"
            else:
                face_status = "stopped"
        else:
            face_status = "skipped_original_train_gate_failed"
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            progress.flush()
            progress.close()
        except Exception as exc:
            errors.append(f"progress close: {type(exc).__name__}: {exc}")

    if active is not None and model is not None and optimizer is not None:
        try:
            payload = active["payload"]
            partial_path = CHECKPOINT_ROOT / f"s{active['seed']}_{active['arm']}_partial.pt"
            partial_payload = n155._checkpoint_payload(
                model,
                optimizer,
                seed=active["seed"],
                branch="unknown8_rebalanced",
                parent_path=active["parent_path"],
                parent_hash=active["parent_hash"],
                parent_total=int(payload["parent_total_updates"]),
                additional_updates=int(payload["additional_updates"]) + int(active["completed_updates"]),
            )
            if not partial_path.exists():
                with partial_path.open("xb") as stream:
                    torch.save(partial_payload, stream)
                    stream.flush()
            partial_checkpoint = {
                "path": str(partial_path.resolve()),
                "sha256": sha256(partial_path),
                "additional_updates_this_plan": active["completed_updates"],
                "total_updates": partial_payload["total_updates"],
                "bundle_fingerprint": partial_payload["bundle_fingerprint"],
                "automatic_resume_allowed": False,
            }
        except BaseException as exc:
            partial_checkpoint = {"save_error": f"{type(exc).__name__}: {exc}"}
            errors.append(f"partial checkpoint save: {type(exc).__name__}: {exc}")
    else:
        partial_checkpoint = None

    progress_integrity = {"complete": False, "error": None}
    try:
        progress_integrity = _validate_progress(PROGRESS, update_count, train_score_queries, fresh_queries)
    except BaseException as exc:
        errors.append(f"progress integrity: {type(exc).__name__}: {exc}")
        progress_integrity["error"] = f"{type(exc).__name__}: {exc}"
    source_hashes = dict(base_sources)
    for path in (SOURCE_AUDIT, SCHEDULE, READINESS, READINESS_ATTEMPT, SOURCE_BINDING, ZERO_PREFLIGHT, ZERO_PREFLIGHT_ATTEMPT, ATTEMPT, PROGRESS, SCORING_PREFLIGHT, TRAINING_GATE_RECEIPT, FACE, FACE_AUDIT, FACE_CHECKS):
        if path.is_file():
            source_hashes[str(path.resolve())] = sha256(path)
    for row in preflight.get("zero_checkpoints", []):
        source_hashes[row["path"]] = row["sha256"]
    new_checkpoints = {str(path.resolve()): sha256(path) for path in CHECKPOINT_ROOT.glob("*.pt") if path.is_file()}
    source_hashes.update(new_checkpoints)
    source_unchanged = n150_builder.verify_hashes(source_hashes)
    if not source_unchanged:
        errors.append("PLAN-N1-162B source/checkpoint hash changed")
    execution_cpu = time.process_time() - started + budget.children
    stage_cpu = PRIOR_FAILED_STAGE_CPU_SECONDS + float(preflight["cpu_seconds"]) + execution_cpu
    linked_cpu = PRIOR_LINKED_CPU_SECONDS + stage_cpu
    training_complete = update_count == TOTAL_UPDATES and train_score_queries == TRAIN_SCORE_QUERIES and len(final_records) == 6 and len([p for p in CHECKPOINT_ROOT.glob("*_final.pt") if p.is_file()]) == 6
    if not training_complete and not errors:
        errors.append("PLAN-N1-162B training endpoint/update denominator incomplete")
    face_expected = len(final_records) == 6 and all(row["training_sufficient"] for row in final_records)
    face_complete = face_status == "complete" and fresh_queries == FACE_QUERIES and len(fresh_rows) == FACE_QUERIES
    protocol_complete = (
        training_complete
        and progress_integrity.get("complete") is True
        and source_unchanged
        and stage_cpu <= CPU_LIMIT_SECONDS
        and linked_cpu <= LINKED_CPU_LIMIT_SECONDS
        and ((face_expected and face_complete) or (not face_expected and face_status == "skipped_original_train_gate_failed"))
        and not errors
    )
    report = {
        "schema": "n1-plan162b-question-surface-result-v1",
        "status": "complete" if protocol_complete else "stopped",
        "complete": protocol_complete,
        "plan_sha256": PLAN_SHA256,
        "budget_addendum_sha256": builder.BUDGET_ADDENDUM_SHA256,
        "repair_addendum_sha256": REPAIR_ADDENDUM_SHA256,
        "prior_failed_result_sha256": ORIGINAL_RESULT_SHA256,
        "prior_failed_attempt_stage_cpu_seconds": PRIOR_FAILED_STAGE_CPU_SECONDS,
        "readiness_sha256": sha256(READINESS),
        "zero_preflight_sha256": sha256(ZERO_PREFLIGHT),
        "training_gate_receipt_sha256": sha256(TRAINING_GATE_RECEIPT) if TRAINING_GATE_RECEIPT.is_file() else None,
        "attempt_sha256": sha256(ATTEMPT),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": source_unchanged,
        "models": final_records,
        "partial_checkpoint": partial_checkpoint,
        "training_updates": update_count,
        "parameter_updates": update_count,
        "optimizer_steps": update_count,
        "train_score_queries": train_score_queries,
        "training_complete": training_complete,
        "training_sufficient_all_endpoints": len(final_records) == 6 and all(row["training_sufficient"] for row in final_records),
        "scoring_preflight": scoring_preflight,
        "fresh_face_status": face_status,
        "fresh_face_sha256": sha256(FACE) if FACE.is_file() else None,
        "fresh_face_audit_sha256": sha256(FACE_AUDIT) if FACE_AUDIT.is_file() else None,
        "fresh_queries": fresh_queries,
        "fresh_diagnostic_rows": fresh_rows,
        "fresh_summary_by_endpoint_kind_template_length": summaries,
        "progress_integrity": progress_integrity,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "execution_cpu_seconds": round(execution_cpu, 6),
        "stage_cpu_seconds": round(stage_cpu, 6),
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cumulative_linked_cpu_seconds": round(linked_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": f"N1-162B{('完整' if protocol_complete else '停止')}：{update_count}/{TOTAL_UPDATES}次更新、{train_score_queries}/{TRAIN_SCORE_QUERIES}条train-only评分、{fresh_queries}/{FACE_QUERIES}条新面查询；stage累计CPU {stage_cpu:.1f}/{CPU_LIMIT_SECONDS:.0f}秒，linked {linked_cpu:.1f}/{LINKED_CPU_LIMIT_SECONDS:.0f}秒，0 dev/holdout/confirmation，can_promote=false。",
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-162 paired question-surface continuation")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--checkpoint-preflight", action="store_true")
    modes.add_argument("--execute-approved", action="store_true")
    modes.add_argument("--restore-worker", type=Path)
    parser.add_argument("--restore-format", choices=("n150", "n155"), default="n155")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker is not None:
        result = _restore_worker(args.restore_worker, args.restore_format)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0 if result.get("complete") is True and result.get("equal") is True else 1
    result = checkpoint_preflight() if args.checkpoint_preflight else execute_approved()
    keys = ("status", "complete", "cpu_seconds", "cumulative_stage_cpu_seconds", "cumulative_linked_cpu_seconds", "zero_checkpoint_count", "independent_restore_count", "training_updates", "train_score_queries", "training_sufficient_all_endpoints", "fresh_face_status", "fresh_queries", "dev_queries", "holdout_queries", "confirmation_queries", "can_promote", "errors", "summary_zh")
    print(json.dumps({key: result.get(key) for key in keys if key in result}, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
