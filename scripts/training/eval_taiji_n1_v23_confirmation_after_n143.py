"""PLAN-N1-144：v23长上下文新题面一次性范围门与条件确认。"""

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

from scripts.training import build_taiji_n1_v23_confirmation_after_n143 as face_builder
from scripts.training import eval_taiji_n1_v11_after_n120 as n120_eval
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import (
    run_taiji_n1_plan140_baseline_retention_balanced_matched_continuation as n140,
)
from scripts.training import (
    run_taiji_n1_plan141_interleaved_context_schedule_matched_continuation as n141,
)
from scripts.training import seal_ast

PLAN = face_builder.PLAN
PLAN_SHA256 = face_builder.PLAN_SHA256
CPU_LIMIT_SECONDS = 7_200.0
KINDS = tuple(face_builder.KINDS)

SOURCE_PRECHECK = face_builder.SOURCE_PRECHECK
SOURCE_EXCLUSIONS = face_builder.SOURCE_EXCLUSIONS
FACE = face_builder.FACE
FACE_ATTEMPT = face_builder.FACE_ATTEMPT
FACE_CHECKS = face_builder.FACE_CHECKS
PREPARED = face_builder.PREPARED

READINESS = ROOT / "reports/taiji_n1_v23_confirmation_readiness_20261019.json"
READINESS_ATTEMPT = ROOT / "reports/taiji_n1_v23_confirmation_readiness_attempt_20261019.json"
SOURCE_BINDING = ROOT / "reports/taiji_n1_v23_confirmation_source_binding_20261019.json"
ATTEMPT = ROOT / "reports/taiji_n1_v23_confirmation_attempt_20261019.json"
PROGRESS = ROOT / "reports/taiji_n1_v23_confirmation_progress_20261019.jsonl"
RESULT = ROOT / "reports/taiji_n1_v23_confirmation_result_20261019.json"


class CpuDeadline:
    def __init__(self, limit: float):
        self.started = time.process_time()
        self.limit = limit

    def used(self) -> float:
        return time.process_time() - self.started

    def guard(self, reserve: float = 0.0) -> None:
        if self.used() + reserve >= self.limit:
            raise TimeoutError("PLAN-N1-144累计CPU预算到达停止线")


class DurableRows(list):
    """Write every completed query row through before the next query starts."""

    def __init__(self, subset: str, spec: dict, progress_path: Path):
        super().__init__()
        self.subset = subset
        self.spec = spec
        self.progress_path = Path(progress_path)

    def append(self, row: dict) -> None:
        durable = dict(
            row,
            subset=self.subset,
            endpoint_id=self.spec["endpoint_id"],
            source=self.spec["source"],
            branch=self.spec["branch"],
            seed=self.spec["seed"],
        )
        super().append(durable)
        append_jsonl(self.progress_path, {"event": "query_row", "row": durable})


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


def append_jsonl(path: Path, value: dict) -> None:
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _merge_hashes(target: dict[str, str], incoming: dict[str, str]) -> None:
    for raw_path, digest in incoming.items():
        path = str(Path(raw_path).resolve())
        prior = target.get(path)
        if prior is not None and prior != digest:
            raise ValueError(f"冻结来源哈希冲突：{path}")
        target[path] = digest


def _verify_hashes(source_hashes: dict[str, str]) -> bool:
    """S3 切换: 经 seal_ast 三通道校验（在位字节 / 在位 AST / git 历史取证）。"""
    return seal_ast.verify_hash_dict(source_hashes)[0]


def _frozen_reports() -> tuple[dict, dict, dict]:
    n132_result = read_json(n132.RESULT)
    n140_result = read_json(n140.RESULT)
    n141_result = read_json(n141.RESULT)
    if (
        n132_result.get("complete") is not True
        or n140_result.get("complete") is not True
        or n141_result.get("complete") is not True
        or n140_result.get("train_score_queries") != 7_488
        or n141_result.get("train_score_queries") != 7_488
        or n141_result.get("training_updates") != 55_296
        or n141_result.get("training_sufficient_all_endpoints") is not True
        or any(
            report.get(key) != 0
            for report in (n140_result, n141_result)
            for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
        or n140_result.get("can_promote") is not False
        or n141_result.get("can_promote") is not False
    ):
        raise ValueError("N1-132/140/141冻结结果不完整或越过train-only边界")
    return n132_result, n140_result, n141_result


def _read_prepared_face() -> tuple[dict, dict, dict, dict, dict]:
    if not PLAN.is_file() or sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-144冻结SHA-256不匹配")
    for path in (SOURCE_PRECHECK, SOURCE_EXCLUSIONS, FACE, FACE_ATTEMPT, FACE_CHECKS, PREPARED):
        if not path.is_file():
            raise FileNotFoundError(f"PLAN-N1-144面准备文件缺失：{path}")
    precheck = read_json(SOURCE_PRECHECK)
    manifest = read_json(SOURCE_EXCLUSIONS)
    face = read_json(FACE)
    checks = read_json(FACE_CHECKS)
    prepared = read_json(PREPARED)
    if (
        precheck.get("status") != "complete"
        or precheck.get("plan_sha256") != PLAN_SHA256
        or precheck.get("source_union", {}).get("entity_exclusion_count") != 31_722
        or precheck.get("source_union", {}).get("word_exclusion_count") != 10_128
        or precheck.get("source_hashes_unchanged") is not True
        or precheck.get("model_queries") != 0
        or precheck.get("parameter_updates") != 0
        or manifest.get("schema") != "n1-plan144-v23-source-exclusions-v1"
        or manifest.get("status") != "ready_for_one_time_face_generation"
        or manifest.get("plan_sha256") != PLAN_SHA256
        or manifest.get("entity_count") != 31_722
        or manifest.get("word_value_count") != 10_128
        or len(manifest.get("excluded_entities", [])) != 31_722
        or len(set(manifest.get("excluded_entities", []))) != 31_722
        or len(manifest.get("excluded_word_values", [])) != 10_128
        or len(set(manifest.get("excluded_word_values", []))) != 10_128
        or face_builder.digest_strings(set(manifest.get("excluded_entities", [])))
        != manifest.get("entity_exclusion_sha256")
        or face_builder.digest_strings(set(manifest.get("excluded_word_values", [])))
        != manifest.get("word_exclusion_sha256")
        or read_json(FACE_ATTEMPT).get("plan_sha256") != PLAN_SHA256
        or read_json(FACE_ATTEMPT).get("model_queries") != 0
        or read_json(FACE_ATTEMPT).get("parameter_updates") != 0
        or not seal_ast.receipt_binding_matches(checks.get("face_sha256"), FACE)
        or checks.get("accepted") is not True
        or checks.get("plan_sha256") != PLAN_SHA256
        or not seal_ast.receipt_binding_matches(
            checks.get("source_exclusion_sha256"), SOURCE_EXCLUSIONS
        )
        or checks.get("model_queries") != 0
        or checks.get("parameter_updates") != 0
        or prepared.get("complete") is not True
        or prepared.get("status") != "ready_for_checkpoint_readiness"
        or prepared.get("plan_sha256") != PLAN_SHA256
        or not seal_ast.receipt_binding_matches(prepared.get("face_sha256"), FACE)
        or not seal_ast.receipt_binding_matches(prepared.get("face_checks_sha256"), FACE_CHECKS)
        or not seal_ast.receipt_binding_matches(
            prepared.get("source_exclusion_sha256"), SOURCE_EXCLUSIONS
        )
        or not seal_ast.receipt_binding_matches(
            prepared.get("source_precheck_sha256"), SOURCE_PRECHECK
        )
        or prepared.get("source_hashes_unchanged") is not True
        or prepared.get("model_queries") != 0
        or prepared.get("parameter_updates") != 0
    ):
        raise ValueError("v23面、来源排除或face readiness与冻结合同不一致")
    if not face_builder._verify_hashes(prepared.get("source_hashes", {})):
        raise ValueError("v23 face准备期来源哈希变化")
    if not seal_ast.receipt_binding_matches(
        checks.get("source_exclusion_sha256"), SOURCE_EXCLUSIONS
    ):
        raise ValueError("v23来源排除清单哈希变化")
    # Re-run the pure face checker from the saved manifest and face bytes.
    verified = face_builder.validate_face(
        face,
        set(manifest["excluded_entities"]),
        set(manifest["excluded_word_values"]),
    )
    if verified.get("accepted") is not True:
        raise ValueError("v23 saved face no longer passes its pure contract")
    return precheck, manifest, face, checks, prepared


def _endpoint_specs() -> list[dict]:
    n132_result, n140_result, n141_result = _frozen_reports()
    parent = {
        int(row["seed"]): row
        for row in n132_result.get("models", [])
        if row.get("arm") == "position"
    }
    if set(parent) != {11, 12, 13}:
        raise ValueError("N1-132三枚position父端点缺失")
    n140_models = {(int(row["seed"]), row["branch"]): row for row in n140_result["models"]}
    n141_models = {(int(row["seed"]), row["branch"]): row for row in n141_result["models"]}
    specs = []
    for seed in (11, 12, 13):
        row = parent[seed]
        specs.append(
            {
                "endpoint_id": f"N1-132/seed{seed}/parent",
                "source": "N1-132",
                "seed": seed,
                "branch": "parent",
                "arm": "position",
                "path": Path(row["final_checkpoint_path"]).resolve(),
                "sha256": row["final_checkpoint_sha256"],
                "plan_sha256": n132.PLAN_SHA256,
                "total_updates": row["total_updates"],
                "history_count": row["final_history_count"],
            }
        )
        n140_row = n140_models[(seed, "candidate")]
        specs.append(
            {
                "endpoint_id": f"N1-140/seed{seed}/block_candidate",
                "source": "N1-140",
                "seed": seed,
                "branch": "candidate",
                "arm": "position",
                "path": Path(n140_row["final_checkpoint_path"]).resolve(),
                "sha256": n140_row["final_checkpoint_sha256"],
                "plan_sha256": n140.PLAN_SHA256,
                "total_updates": n140_row["total_updates"],
                "history_count": n140_row["final_history_count"],
            }
        )
        for branch in ("candidate", "control"):
            n141_row = n141_models[(seed, branch)]
            specs.append(
                {
                    "endpoint_id": f"N1-141/seed{seed}/{branch}",
                    "source": "N1-141",
                    "seed": seed,
                    "branch": branch,
                    "arm": "position",
                    "path": Path(n141_row["final_checkpoint_path"]).resolve(),
                    "sha256": n141_row["final_checkpoint_sha256"],
                    "plan_sha256": n141.PLAN_SHA256,
                    "total_updates": n141_row["total_updates"],
                    "history_count": n141_row["final_history_count"],
                }
            )
    if len(specs) != 12 or len({spec["endpoint_id"] for spec in specs}) != 12:
        raise ValueError("PLAN-N1-144 requires exactly 12 distinct frozen endpoints")
    return specs


def _restore_endpoint(spec: dict):
    path = Path(spec["path"])
    if not path.is_file() or sha256(path) != spec["sha256"]:
        raise ValueError(f"checkpoint字节改变：{path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if spec["source"] == "N1-132":
        model, optimizer = n132._restore_bundle_payload(payload)
        if payload.get("arm") != "position" or payload.get("plan_sha256") != n132.PLAN_SHA256:
            raise ValueError(f"N1-132父checkpoint协议不符：{path}")
    elif spec["source"] == "N1-140":
        model, optimizer = n140._restore_payload(payload)
        if payload.get("branch") != "candidate" or payload.get("plan_sha256") != n140.PLAN_SHA256:
            raise ValueError(f"N1-140 comparison checkpoint协议不符：{path}")
    elif spec["source"] == "N1-141":
        model, optimizer = n141._restore_payload(payload)
        if (
            payload.get("branch") != spec["branch"]
            or payload.get("plan_sha256") != n141.PLAN_SHA256
        ):
            raise ValueError(f"N1-141 checkpoint协议不符：{path}")
    else:
        raise ValueError(f"unknown endpoint source: {spec['source']}")
    fingerprint = n132.e11.scorer.paired.fingerprint(model.bundle(optimizer))
    if (
        fingerprint != payload.get("bundle_fingerprint")
        or int(payload.get("seed", -1)) != spec["seed"]
        or int(payload.get("total_updates", -1)) != spec["total_updates"]
        or int(payload.get("history_count", -1)) != spec["history_count"]
        or len(optimizer.state) != 41
        or any(int(optimizer.state[p]["step"]) != spec["total_updates"] for p in model.parameters())
        or len(model._observations) != spec["history_count"]
    ):
        raise ValueError(f"endpoint完整bundle/history/Adam与冻结回执不符：{path}")
    return model, optimizer, payload, fingerprint


def _source_hashes(prepared: dict, specs: list[dict]) -> dict[str, str]:
    hashes = dict(prepared.get("source_hashes", {}))
    for path in (
        SOURCE_PRECHECK,
        SOURCE_EXCLUSIONS,
        FACE,
        FACE_ATTEMPT,
        FACE_CHECKS,
        PREPARED,
        READINESS,
        READINESS_ATTEMPT,
        SOURCE_BINDING,
        ATTEMPT,
        Path(__file__).resolve(),
        Path(face_builder.__file__).resolve(),
        ROOT / "tests/taiji_native/test_n1_plan144_v23_face_builder_contract.py",
        ROOT / "tests/taiji_native/test_n1_plan144_v23_confirmation_runner_contract.py",
    ):
        path = Path(path).resolve()
        if path.is_file():
            hashes[str(path)] = sha256(path)
    for spec in specs:
        path = Path(spec["path"]).resolve()
        hashes[str(path)] = spec["sha256"]
    return hashes


def validate_only() -> dict:
    if any(
        path.exists()
        for path in (READINESS, READINESS_ATTEMPT, SOURCE_BINDING, ATTEMPT, PROGRESS, RESULT)
    ):
        raise FileExistsError("PLAN-N1-144 checkpoint readiness/query phase is write-once")
    started = time.process_time()
    n120_eval.n120.configure()
    _precheck, _manifest, face, checks, prepared = _read_prepared_face()
    n132_result, n140_result, n141_result = _frozen_reports()
    specs = _endpoint_specs()
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan144-v23-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "planned_endpoints": 12,
            "planned_sacrifice_queries": 960,
            "planned_confirmation_queries": 960,
            "model_queries": 0,
            "parameter_updates": 0,
            "dev_queries": 0,
            "v3_holdout_queries": 0,
            "can_promote": False,
        },
    )
    errors = []
    restore_checks = []
    for spec in specs:
        before = time.process_time()
        model = optimizer = None
        try:
            model, optimizer, payload, fingerprint = _restore_endpoint(spec)
            restore_checks.append(
                {
                    "endpoint_id": spec["endpoint_id"],
                    "source": spec["source"],
                    "seed": spec["seed"],
                    "branch": spec["branch"],
                    "path": str(spec["path"]),
                    "checkpoint_sha256": spec["sha256"],
                    "bundle_fingerprint": fingerprint,
                    "history_count": len(model._observations),
                    "optimizer_parameters": len(optimizer.state),
                    "additional_updates": payload.get("additional_updates"),
                    "complete": True,
                    "equal": True,
                    "cpu_seconds": round(time.process_time() - before, 6),
                }
            )
        except BaseException as exc:
            errors.append(f"{spec['endpoint_id']}: {type(exc).__name__}: {exc}")
            break
        finally:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
    hashes = _source_hashes(prepared, specs)
    unchanged = _verify_hashes(hashes)
    if not unchanged:
        errors.append("checkpoint/readiness/face或来源哈希在12端点恢复期间变化")
    cpu = time.process_time() - started
    prior_cpu = float(prepared["cpu_seconds"])
    cumulative = prior_cpu + cpu
    complete = (
        not errors
        and len(restore_checks) == 12
        and all(row["complete"] and row["equal"] for row in restore_checks)
        and cumulative <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan144-v23-checkpoint-readiness-v1",
        "status": "ready_for_sacrifice" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "source_precheck_sha256": sha256(SOURCE_PRECHECK),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "prepared_readiness_sha256": sha256(PREPARED),
        "source_hashes": hashes,
        "source_hashes_unchanged": unchanged,
        "endpoints": restore_checks,
        "endpoint_count": len(restore_checks),
        "independent_full_bundle_restores": len(restore_checks),
        "all_full_bundles_equal": len(restore_checks) == 12
        and all(row["equal"] for row in restore_checks),
        "checkpoint_loads": len(restore_checks),
        "model_loads": len(restore_checks),
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "v3_holdout_queries": 0,
        "new_confirmation_queries": 0,
        "cpu_seconds": round(cpu, 6),
        "prior_face_preparation_cpu_seconds": round(prior_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
    }
    report["summary_zh"] = (
        "12个N1-132/140/141完整bundle独立恢复与哈希核对通过；0模型查询、0更新、0旧dev/holdout查询，可进入v23 sacrifice一次性范围门。"
        if complete
        else "v23范围查询前置未通过；停止，不训练、不查询、不重试。"
    )
    write_new(READINESS, report)
    binding = {
        "schema": "n1-plan144-v23-readiness-source-binding-v1",
        "status": "complete" if complete else "stopped",
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_hashes": hashes,
        "checkpoint_hashes": {str(Path(spec["path"]).resolve()): spec["sha256"] for spec in specs},
        "can_promote": False,
    }
    write_new(SOURCE_BINDING, binding)
    return report


def _verify_readiness() -> tuple[dict, dict, dict, dict, list[dict]]:
    if not READINESS.is_file() or not SOURCE_BINDING.is_file():
        raise FileNotFoundError("须先完成12端点v23 checkpoint readiness")
    readiness = read_json(READINESS)
    binding = read_json(SOURCE_BINDING)
    _precheck, manifest, face, checks, prepared = _read_prepared_face()
    specs = _endpoint_specs()
    if (
        readiness.get("complete") is not True
        or readiness.get("status") != "ready_for_sacrifice"
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("endpoint_count") != 12
        or readiness.get("independent_full_bundle_restores") != 12
        or readiness.get("all_full_bundles_equal") is not True
        or readiness.get("source_hashes_unchanged") is not True
        or readiness.get("model_queries") != 0
        or readiness.get("parameter_updates") != 0
        or readiness.get("dev_queries") != 0
        or readiness.get("v3_holdout_queries") != 0
        or readiness.get("new_confirmation_queries") != 0
        or binding.get("status") != "complete"
        or binding.get("plan_sha256") != PLAN_SHA256
        or not seal_ast.receipt_binding_matches(binding.get("readiness_sha256"), READINESS)
        or binding.get("source_hashes") != readiness.get("source_hashes")
    ):
        raise ValueError("v23 readiness/source binding不完整或已变化")
    hashes = dict(readiness["source_hashes"])
    hashes[str(READINESS.resolve())] = sha256(READINESS)
    hashes[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
    if not _verify_hashes(hashes):
        raise ValueError("v23查询前来源/face/checkpoint哈希变化")
    return readiness, binding, manifest, face, specs


def _score_endpoint(
    spec: dict,
    items: list[dict],
    subset: str,
    budget: CpuDeadline,
) -> tuple[list[dict], dict]:
    budget.guard(reserve=5)
    model = optimizer = None
    before = None
    loaded = False
    started = time.process_time()
    accounting = {"confirmation_queries": 0}
    rows = DurableRows(subset, spec, PROGRESS)
    error = None
    try:
        model, optimizer, payload, fingerprint = _restore_endpoint(spec)
        loaded = True
        before = fingerprint
        n120_eval.scorer.evaluate(
            model,
            optimizer,
            spec["seed"],
            "position",
            items,
            budget,
            output=rows,
            accounting=accounting,
        )
        after = n132.e11.scorer.paired.fingerprint(model.bundle(optimizer))
        if after != before:
            raise ValueError("frozen v23 scoring changed model/Adam/RNG/history")
        if accounting["confirmation_queries"] != len(items) or len(rows) != len(items):
            raise ValueError("v23 scorer did not emit exactly one row per query")
        if sha256(spec["path"]) != spec["sha256"]:
            raise ValueError("checkpoint bytes changed during scoring")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if model is not None and optimizer is not None and before is not None:
            try:
                after = n132.e11.scorer.paired.fingerprint(model.bundle(optimizer))
                if after != before:
                    error = "ValueError: frozen scoring changed model/Adam/RNG/history"
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
    receipt = {
        "endpoint_id": spec["endpoint_id"],
        "source": spec["source"],
        "branch": spec["branch"],
        "seed": spec["seed"],
        "subset": subset,
        "queries": accounting["confirmation_queries"],
        "completed_rows": len(rows),
        "model_loaded": loaded,
        "checkpoint_sha256": spec["sha256"],
        "checkpoint_unchanged": Path(spec["path"]).is_file()
        and sha256(spec["path"]) == spec["sha256"],
        "bundle_fingerprint": before,
        "bundle_unchanged_after_scoring": error is None,
        "cpu_seconds": round(time.process_time() - started, 6),
        "error": error,
    }
    if not receipt["checkpoint_unchanged"]:
        receipt["error"] = receipt["error"] or "ValueError: checkpoint changed during scoring"
    return rows, receipt


def _range_summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("subset") != "sacrifice":
            raise ValueError("range summary may only use sacrifice rows")
        grouped[row["endpoint_id"]].append(row)
    if len(grouped) != 12 or any(len(selected) != 80 for selected in grouped.values()):
        raise ValueError("range summary requires 80 sacrifice rows for each of 12 endpoints")
    by_endpoint = {}
    shared_ids = None
    qualified = 0
    for endpoint_id, selected in sorted(grouped.items()):
        ids = {row["id"] for row in selected}
        if len(ids) != 80 or (shared_ids is not None and ids != shared_ids):
            raise ValueError("all range endpoints must use the same 80 unique sacrifice IDs")
        shared_ids = ids
        kind_counts = Counter(row.get("kind") for row in selected)
        if any(kind_counts[kind] != 20 for kind in KINDS):
            raise ValueError("sacrifice must contain 20 items from each class")
        answerable = [row for row in selected if row.get("kind") != "unknown"]
        if len(answerable) != 60 or sum(row.get("kind") == "unknown" for row in selected) != 20:
            raise ValueError("sacrifice range denominator must be 60 answerable plus 20 unknown")
        errors = sum(row.get("free_exact") is not True for row in answerable)
        passes = errors >= 24
        qualified += int(passes)
        by_endpoint[endpoint_id] = {
            "items": len(selected),
            "answerable_items": len(answerable),
            "unknown_items": 20,
            "errors": errors,
            "error_floor": 24,
            "passes_error_floor": passes,
        }
    return {
        "by_endpoint": by_endpoint,
        "qualified_endpoints": qualified,
        "required_qualified_endpoints": 8,
        "range_check_passed": qualified >= 8,
    }


def confirmation_summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("subset") != "confirm":
            raise ValueError("confirmation summary may only use confirm rows")
        grouped[row["endpoint_id"]].append(row)
    if len(grouped) != 12 or any(len(selected) != 80 for selected in grouped.values()):
        raise ValueError("confirmation summary requires 80 rows for each of 12 endpoints")
    per_endpoint = {}
    family_totals = Counter()
    shared_ids = None
    for endpoint_id, selected in sorted(grouped.items()):
        ids = {row["id"] for row in selected}
        if len(ids) != 80:
            raise ValueError(f"duplicate confirmation IDs in {endpoint_id}")
        if shared_ids is not None and ids != shared_ids:
            raise ValueError(
                "all confirmation endpoints must use the same 80 unique confirmation IDs"
            )
        shared_ids = ids
        by_kind = {
            kind: sum(row.get("kind") == kind and row.get("free_exact") is True for row in selected)
            for kind in KINDS
        }
        kind_counts = Counter(row.get("kind") for row in selected)
        if any(kind_counts[kind] != 20 for kind in KINDS):
            raise ValueError(
                f"each endpoint requires 20 confirmation items per class: {endpoint_id}"
            )
        exact = sum(row.get("free_exact") is True for row in selected)
        source = selected[0]["source"]
        branch = selected[0]["branch"]
        seed = selected[0]["seed"]
        family_totals[(source, branch)] += exact
        per_endpoint[endpoint_id] = {
            "source": source,
            "branch": branch,
            "seed": seed,
            "items": len(selected),
            "exact": exact,
            "free_exact_by_kind": by_kind,
        }
    candidate = family_totals[("N1-141", "candidate")]
    n140_block = family_totals[("N1-140", "candidate")]
    control = family_totals[("N1-141", "control")]
    parent = family_totals[("N1-132", "parent")]
    candidate_endpoints = [per_endpoint[f"N1-141/seed{seed}/candidate"] for seed in (11, 12, 13)]
    gates = {
        "candidate_each_seed_at_least_40_of_80": all(
            row["exact"] >= 40 for row in candidate_endpoints
        ),
        "candidate_each_seed_each_kind_at_least_8_of_20": all(
            all(row["free_exact_by_kind"][kind] >= 8 for kind in KINDS)
            for row in candidate_endpoints
        ),
        "candidate_advantage_over_n141_control_at_least_48_of_240": candidate >= control + 48,
        "candidate_advantage_over_n132_parent_at_least_48_of_240": candidate >= parent + 48,
        "candidate_advantage_over_n140_block_at_least_24_of_240": candidate >= n140_block + 24,
    }
    return {
        "per_endpoint": per_endpoint,
        "aggregate_exact_by_family": {
            f"{source}/{branch}": total for (source, branch), total in sorted(family_totals.items())
        },
        "candidate_exact": candidate,
        "n141_control_exact": control,
        "n132_parent_exact": parent,
        "n140_block_candidate_exact": n140_block,
        "candidate_advantage_over_n141_control": candidate - control,
        "candidate_advantage_over_n132_parent": candidate - parent,
        "candidate_advantage_over_n140_block": candidate - n140_block,
        "gates": gates,
        "confirmation_pass": all(gates.values()),
    }


def _progress_summary(subset: str, receipt: dict) -> dict:
    return {
        "event": "endpoint_progress",
        "subset": subset,
        "endpoint_id": receipt["endpoint_id"],
        "queries": int(receipt.get("queries", 0)),
        "completed_rows": int(receipt.get("completed_rows", 0)),
        "checkpoint_sha256": receipt.get("checkpoint_sha256"),
        "bundle_fingerprint": receipt.get("bundle_fingerprint"),
        "error": receipt.get("error"),
    }


def _append_progress(subset: str, receipt: dict) -> None:
    append_jsonl(PROGRESS, _progress_summary(subset, receipt))


def execute_approved() -> dict:
    if any(path.exists() for path in (ATTEMPT, PROGRESS, RESULT)):
        raise FileExistsError("PLAN-N1-144 face is a one-shot query; no restart or retry")
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan144-v23-confirmation-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS) if READINESS.is_file() else None,
            "source_binding_sha256": sha256(SOURCE_BINDING) if SOURCE_BINDING.is_file() else None,
            "face_sha256": sha256(FACE) if FACE.is_file() else None,
            "planned_endpoints": 12,
            "planned_sacrifice_queries": 960,
            "planned_confirmation_queries": 960,
            "parameter_updates": 0,
            "dev_queries": 0,
            "v3_holdout_queries": 0,
            "can_promote": False,
        },
    )
    errors = []
    range_receipts = []
    range_rows = []
    confirm_receipts = []
    confirm_rows = []
    model_loads = 0
    sacrifice_queries = 0
    confirmation_queries = 0
    readiness = {}
    source_hashes = {}
    prior_cpu = 0.0
    range_gate = None
    confirmation = None
    execution_started = time.process_time()
    try:
        n120_eval.n120.configure()
        readiness, _binding, _manifest, face, specs = _verify_readiness()
        prior_cpu = float(readiness["cumulative_cpu_seconds"])
        remaining = CPU_LIMIT_SECONDS - prior_cpu
        spent = time.process_time() - execution_started
        if remaining - spent <= 0:
            raise TimeoutError("PLAN-N1-144 CPU allowance exhausted before first query")
        budget = CpuDeadline(remaining - spent)
        source_hashes = dict(readiness["source_hashes"])
        source_hashes[str(READINESS.resolve())] = sha256(READINESS)
        source_hashes[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
        if not _verify_hashes(source_hashes):
            raise ValueError("PLAN-N1-144 final query source/checkpoint hash mismatch")
        sacrifice_items = [item for item in face["episodes"] if item.get("subset") == "sacrifice"]
        confirm_items = [item for item in face["episodes"] if item.get("subset") == "confirm"]
        if (
            len(sacrifice_items) != 80
            or len(confirm_items) != 80
            or any(
                sum(item.get("kind") == kind for item in sacrifice_items) != 20 for kind in KINDS
            )
            or any(sum(item.get("kind") == kind for item in confirm_items) != 20 for kind in KINDS)
        ):
            raise ValueError("v23 sacrifice/confirm subset changed after face checks")
        PROGRESS.open("x", encoding="utf-8", newline="\n").close()
        for spec in specs:
            budget.guard(reserve=10)
            rows, receipt = _score_endpoint(spec, sacrifice_items, "sacrifice", budget)
            model_loads += int(receipt["model_loaded"])
            sacrifice_queries += int(receipt["queries"])
            range_rows.extend(rows)
            range_receipts.append(receipt)
            _append_progress("sacrifice", receipt)
            if receipt.get("error"):
                errors.append(f"{spec['endpoint_id']} sacrifice: {receipt['error']}")
                break
        if (
            not errors
            and sacrifice_queries == 960
            and len(range_rows) == 960
            and len(range_receipts) == 12
        ):
            range_gate = _range_summary(range_rows)
        if range_gate is not None and range_gate["range_check_passed"]:
            for spec in specs:
                budget.guard(reserve=10)
                rows, receipt = _score_endpoint(spec, confirm_items, "confirm", budget)
                model_loads += int(receipt["model_loaded"])
                confirmation_queries += int(receipt["queries"])
                confirm_rows.extend(rows)
                confirm_receipts.append(receipt)
                _append_progress("confirm", receipt)
                if receipt.get("error"):
                    errors.append(f"{spec['endpoint_id']} confirm: {receipt['error']}")
                    break
            if (
                not errors
                and confirmation_queries == 960
                and len(confirm_rows) == 960
                and len(confirm_receipts) == 12
            ):
                confirmation = confirmation_summary(confirm_rows)
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    unchanged = bool(source_hashes) and _verify_hashes(source_hashes)
    if not unchanged:
        errors.append("至少一个来源或checkpoint在range/confirm查询期间改变")
    execution_cpu = time.process_time() - execution_started
    cumulative_cpu = prior_cpu + execution_cpu
    range_complete = (
        sacrifice_queries == 960
        and len(range_rows) == 960
        and len(range_receipts) == 12
        and range_gate is not None
    )
    confirmation_complete = (
        confirmation_queries == 960 and len(confirm_receipts) == 12 and confirmation is not None
    )
    complete = (
        not errors
        and range_complete
        and (
            range_gate is not None
            and (not range_gate["range_check_passed"] or confirmation_complete)
        )
        and unchanged
        and cumulative_cpu <= CPU_LIMIT_SECONDS
    )
    if not range_gate or not range_gate.get("range_check_passed"):
        status = "range_not_met" if complete else "stopped"
    elif confirmation and confirmation.get("confirmation_pass"):
        status = "confirmation_passed" if complete else "stopped"
    else:
        status = "confirmation_failed" if complete else "stopped"
    report = {
        "schema": "n1-plan144-v23-fresh-confirmation-v1",
        "status": status,
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_binding_sha256": sha256(SOURCE_BINDING),
        "attempt_sha256": sha256(ATTEMPT),
        "progress_sha256": sha256(PROGRESS) if PROGRESS.is_file() else None,
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": unchanged,
        "range_summary": range_gate,
        "range_endpoint_receipts": range_receipts,
        "sacrifice_rows": range_rows,
        "confirmation_summary": confirmation,
        "confirmation_endpoint_receipts": confirm_receipts,
        "confirmation_rows": confirm_rows,
        "endpoint_count": 12,
        "model_loads": model_loads + int(readiness.get("model_loads", 0)),
        "checkpoint_loads": model_loads + int(readiness.get("checkpoint_loads", 0)),
        "sacrifice_queries": sacrifice_queries,
        "sacrifice_rows_completed": len(range_rows),
        "confirmation_queries": confirmation_queries,
        "confirmation_rows_completed": len(confirm_rows),
        "total_v23_queries": sacrifice_queries + confirmation_queries,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "v3_holdout_queries": 0,
        "cpu_seconds": round(execution_cpu, 6),
        "prior_cpu_seconds": round(prior_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "behavioral_pass": bool(confirmation and confirmation.get("confirmation_pass")),
        "can_promote": False,
        "errors": errors,
    }
    report["summary_zh"] = (
        f"v23一次性执行{status}：牺牲集{len(range_rows)}/960条；"
        f"确认集{confirmation_queries}/960条；12端点保持冻结、0更新、未访问dev/v3旧holdout；can_promote=false。"
    )
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-144 frozen v23 confirmation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    result = validate_only() if args.validate_only else execute_approved()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "cpu_seconds",
                    "cumulative_cpu_seconds",
                    "independent_full_bundle_restores",
                    "sacrifice_queries",
                    "confirmation_queries",
                    "total_v23_queries",
                    "behavioral_pass",
                    "dev_queries",
                    "v3_holdout_queries",
                    "can_promote",
                    "errors",
                    "summary_zh",
                )
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
