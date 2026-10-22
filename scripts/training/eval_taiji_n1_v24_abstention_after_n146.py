"""PLAN-N1-147：匹配known/unknown诊断的bundle readiness与一次性查询。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v24_abstention_after_n146 as face_builder
from scripts.training import eval_taiji_n1_v23_confirmation_after_n143 as frozen_eval

CONDITIONS = ("known", "unknown")
PLAN = face_builder.PLAN
PLAN_SHA256 = face_builder.PLAN_SHA256
CPU_LIMIT_SECONDS = 3_600.0
FACE = face_builder.FACE
SOURCE_PRECHECK = face_builder.SOURCE_PRECHECK
SOURCE_EXCLUSIONS = face_builder.SOURCE_EXCLUSIONS
FACE_ATTEMPT = face_builder.FACE_ATTEMPT
FACE_CHECKS = face_builder.FACE_CHECKS
PREPARED = face_builder.PREPARED

READINESS = ROOT / "reports/taiji_n1_v24_abstention_readiness_20261019.json"
READINESS_ATTEMPT = ROOT / "reports/taiji_n1_v24_abstention_readiness_attempt_20261019.json"
SOURCE_BINDING = ROOT / "reports/taiji_n1_v24_abstention_source_binding_20261019.json"
ATTEMPT = ROOT / "reports/taiji_n1_v24_abstention_attempt_20261019.json"
PROGRESS = ROOT / "reports/taiji_n1_v24_abstention_progress_20261019.jsonl"
RESULT = ROOT / "reports/taiji_n1_v24_abstention_diagnostic_20261019.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan147_v24_abstention_contract.py"


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


def _read_prepared_face() -> tuple[dict, dict, dict]:
    if not PLAN.is_file() or sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-147 frozen SHA-256 mismatch")
    paths = (SOURCE_PRECHECK, SOURCE_EXCLUSIONS, FACE, FACE_ATTEMPT, FACE_CHECKS, PREPARED)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("N1-147 face preparation is incomplete")
    precheck = read_json(SOURCE_PRECHECK)
    manifest = read_json(SOURCE_EXCLUSIONS)
    face = read_json(FACE)
    attempt = read_json(FACE_ATTEMPT)
    checks = read_json(FACE_CHECKS)
    prepared = read_json(PREPARED)
    if (
        precheck.get("status") != "complete"
        or precheck.get("plan_sha256") != PLAN_SHA256
        or precheck.get("source_union", {}).get("entity_exclusion_count") != 36_802
        or precheck.get("source_union", {}).get("word_exclusion_count") != 13_968
        or precheck.get("source_hashes_unchanged") is not True
        or precheck.get("model_queries") != 0
        or precheck.get("parameter_updates") != 0
        or manifest.get("schema") != "n1-plan147-v24-source-exclusions-v1"
        or manifest.get("status") != "ready_for_one_time_face_generation"
        or manifest.get("plan_sha256") != PLAN_SHA256
        or manifest.get("entity_count") != 36_802
        or manifest.get("word_value_count") != 13_968
        or len(set(manifest.get("excluded_entities", []))) != 36_802
        or len(set(manifest.get("excluded_word_values", []))) != 13_968
        or face_builder.digest_strings(set(manifest.get("excluded_entities", [])))
        != manifest.get("entity_exclusion_sha256")
        or face_builder.digest_strings(set(manifest.get("excluded_word_values", [])))
        != manifest.get("word_exclusion_sha256")
        or face.get("schema") != "n1-v24-matched-abstention-diagnostic-face-v1"
        or face.get("seed") != 124
        or face.get("plan_sha256") != PLAN_SHA256
        or len(face.get("episodes", [])) != 80
        or attempt.get("plan_sha256") != PLAN_SHA256
        or attempt.get("model_queries") != 0
        or attempt.get("parameter_updates") != 0
        or sha256(FACE) != checks.get("face_sha256")
        or checks.get("accepted") is not True
        or checks.get("plan_sha256") != PLAN_SHA256
        or checks.get("source_exclusion_sha256") != sha256(SOURCE_EXCLUSIONS)
        or checks.get("source_hashes_unchanged") is not True
        or checks.get("model_queries") != 0
        or checks.get("parameter_updates") != 0
        or prepared.get("complete") is not True
        or prepared.get("plan_sha256") != PLAN_SHA256
        or prepared.get("face_sha256") != sha256(FACE)
        or prepared.get("face_checks_sha256") != sha256(FACE_CHECKS)
        or prepared.get("source_exclusion_sha256") != sha256(SOURCE_EXCLUSIONS)
        or prepared.get("source_precheck_sha256") != sha256(SOURCE_PRECHECK)
        or prepared.get("source_hashes_unchanged") is not True
        or prepared.get("model_queries") != 0
        or prepared.get("parameter_updates") != 0
    ):
        raise ValueError("N1-147 face/source preparation differs from frozen contract")
    if not face_builder._verify_hashes(manifest.get("source_hashes", {})):
        raise ValueError("N1-147 source exclusion hashes changed")
    if not face_builder._verify_hashes(prepared.get("source_hashes", {})):
        raise ValueError("N1-147 prepared-face source hashes changed")
    verified = face_builder.validate_face(
        face,
        set(manifest["excluded_entities"]),
        set(manifest["excluded_word_values"]),
    )
    if verified.get("accepted") is not True:
        raise ValueError("saved v24 paired face no longer passes its contract")
    return manifest, face, prepared


def _endpoint_specs() -> list[dict]:
    specs = frozen_eval._endpoint_specs()
    if len(specs) != 12 or len({spec["endpoint_id"] for spec in specs}) != 12:
        raise ValueError("N1-147 requires the 12 frozen N1-132/140/141 endpoints")
    return specs


def _source_hashes(prepared: dict, specs: list[dict]) -> dict[str, str]:
    hashes = dict(prepared.get("source_hashes", {}))
    paths = (
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
        Path(frozen_eval.__file__).resolve(),
        TEST,
    )
    for path in paths:
        path = Path(path).resolve()
        if path.is_file():
            hashes[str(path)] = sha256(path)
    for spec in specs:
        hashes[str(Path(spec["path"]).resolve())] = spec["sha256"]
    return hashes


def validate_only() -> dict:
    outputs = (READINESS, READINESS_ATTEMPT, SOURCE_BINDING, ATTEMPT, PROGRESS, RESULT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("PLAN-N1-147 readiness/query phase is write-once")
    started = time.process_time()
    frozen_eval.n120_eval.n120.configure()
    _manifest, _face, prepared = _read_prepared_face()
    specs = _endpoint_specs()
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan147-v24-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "planned_endpoints": 12,
            "planned_diagnostic_queries": 960,
            "model_queries": 0,
            "parameter_updates": 0,
            "dev_queries": 0,
            "v3_holdout_queries": 0,
            "can_promote": False,
        },
    )
    restore_checks = []
    errors = []
    for spec in specs:
        before = time.process_time()
        model = optimizer = None
        try:
            model, optimizer, payload, fingerprint = frozen_eval._restore_endpoint(spec)
            restore_checks.append(
                {
                    "endpoint_id": spec["endpoint_id"],
                    "source": spec["source"],
                    "seed": spec["seed"],
                    "branch": spec["branch"],
                    "checkpoint_sha256": spec["sha256"],
                    "bundle_fingerprint": fingerprint,
                    "history_count": len(model._observations),
                    "optimizer_parameters": len(optimizer.state),
                    "equal": True,
                    "complete": True,
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
    unchanged = frozen_eval._verify_hashes(hashes)
    if not unchanged:
        errors.append("N1-147/v23来源或checkpoint哈希在bundle恢复期间变化")
    cpu = time.process_time() - started
    prior_cpu = float(prepared["cpu_seconds"])
    cumulative = prior_cpu + cpu
    complete = not errors and len(restore_checks) == 12 and cumulative <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan147-v24-checkpoint-readiness-v1",
        "status": "ready_for_diagnostic" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
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
        "cpu_seconds": round(cpu, 6),
        "prior_face_preparation_cpu_seconds": round(prior_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": (
            "12个冻结完整bundle独立恢复及哈希核对通过；0模型查询/更新，可进入v24配对诊断一次性查询。"
            if complete
            else "checkpoint readiness未通过；停止，不查询、不训练、不重试。"
        ),
    }
    write_new(READINESS, report)
    binding = {
        "schema": "n1-plan147-v24-readiness-source-binding-v1",
        "status": "complete" if complete else "stopped",
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_hashes": hashes,
        "checkpoint_hashes": {str(Path(spec["path"]).resolve()): spec["sha256"] for spec in specs},
        "can_promote": False,
    }
    write_new(SOURCE_BINDING, binding)
    return report


def _verify_readiness() -> tuple[dict, dict, dict, list[dict]]:
    if not READINESS.is_file() or not SOURCE_BINDING.is_file():
        raise FileNotFoundError("N1-147 requires completed 12-endpoint checkpoint readiness")
    readiness = read_json(READINESS)
    binding = read_json(SOURCE_BINDING)
    _manifest, face, prepared = _read_prepared_face()
    specs = _endpoint_specs()
    if (
        readiness.get("status") != "ready_for_diagnostic"
        or readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("face_sha256") != sha256(FACE)
        or readiness.get("endpoint_count") != 12
        or readiness.get("independent_full_bundle_restores") != 12
        or readiness.get("all_full_bundles_equal") is not True
        or readiness.get("source_hashes_unchanged") is not True
        or readiness.get("model_queries") != 0
        or readiness.get("parameter_updates") != 0
        or binding.get("status") != "complete"
        or binding.get("plan_sha256") != PLAN_SHA256
        or binding.get("readiness_sha256") != sha256(READINESS)
        or binding.get("source_hashes") != readiness.get("source_hashes")
    ):
        raise ValueError("N1-147 readiness/source binding mismatch")
    hashes = dict(readiness["source_hashes"])
    hashes[str(READINESS.resolve())] = sha256(READINESS)
    hashes[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
    if not frozen_eval._verify_hashes(hashes):
        raise ValueError("N1-147 query source/checkpoint hash mismatch")
    return readiness, binding, face, specs


def _paired_summary(rows: list[dict], face: dict) -> dict:
    face_by_id = {item["id"]: item for item in face["episodes"]}
    rows_by_endpoint: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        rows_by_endpoint[row["endpoint_id"]][row["id"]] = row
    by_endpoint = {}
    for endpoint_id, endpoint_rows in sorted(rows_by_endpoint.items()):
        pairs: dict[str, dict[str, dict]] = defaultdict(dict)
        for item_id, row in endpoint_rows.items():
            item = face_by_id[item_id]
            pairs[item["pair_id"]][item["pair_condition"]] = row
        counts = Counter()
        known_exact = known_abstain = unknown_abstain = unknown_false_answer = 0
        for pair_id, conditions in pairs.items():
            if set(conditions) != set(CONDITIONS):
                raise ValueError(f"query output lost a matched condition: {endpoint_id}/{pair_id}")
            known = conditions["known"]
            unknown = conditions["unknown"]
            known_exact += int(known.get("free_exact") is True)
            known_abstain += int(known.get("output_hex") == b"unknown".hex())
            unknown_is_abstain = unknown.get("output_hex") == b"unknown".hex()
            unknown_abstain += int(unknown_is_abstain)
            unknown_false_answer += int(not unknown_is_abstain)
            pattern = (
                "known_exact_unknown_abstain"
                if known.get("free_exact") is True and unknown_is_abstain
                else (
                    "known_abstains_unknown_abstain"
                    if known.get("output_hex") == b"unknown".hex() and unknown_is_abstain
                    else (
                        "known_exact_unknown_false_answer"
                        if known.get("free_exact") is True
                        else "other_pair_outcome"
                    )
                )
            )
            counts[pattern] += 1
        if len(pairs) != 40:
            raise ValueError(f"endpoint does not have 40 matched pairs: {endpoint_id}")
        by_endpoint[endpoint_id] = {
            "pairs": len(pairs),
            "known_exact": known_exact,
            "known_abstain_literal": known_abstain,
            "unknown_abstain_literal": unknown_abstain,
            "unknown_false_answer": unknown_false_answer,
            "paired_outcomes": dict(sorted(counts.items())),
        }
    return by_endpoint


def execute_approved() -> dict:
    if any(path.exists() for path in (ATTEMPT, PROGRESS, RESULT)):
        raise FileExistsError("PLAN-N1-147 diagnostic face is one-shot; no restart or retry")
    if not READINESS.is_file() or not SOURCE_BINDING.is_file():
        raise FileNotFoundError("run --validate-only and pass readiness before queries")
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan147-v24-diagnostic-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_binding_sha256": sha256(SOURCE_BINDING),
            "face_sha256": sha256(FACE),
            "planned_endpoints": 12,
            "planned_diagnostic_queries": 960,
            "parameter_updates": 0,
            "dev_queries": 0,
            "v3_holdout_queries": 0,
            "can_promote": False,
        },
    )
    errors = []
    receipts = []
    result_rows = []
    model_loads = 0
    diagnostic_queries = 0
    readiness = {}
    source_hashes = {}
    prior_cpu = 0.0
    started = time.process_time()
    original_progress = frozen_eval.PROGRESS
    try:
        frozen_eval.n120_eval.n120.configure()
        readiness, _binding, face, specs = _verify_readiness()
        prior_cpu = float(readiness["cumulative_cpu_seconds"])
        remaining = CPU_LIMIT_SECONDS - prior_cpu
        if remaining - (time.process_time() - started) <= 0:
            raise TimeoutError("N1-147 CPU allowance exhausted before first query")
        budget = frozen_eval.CpuDeadline(remaining - (time.process_time() - started))
        source_hashes = dict(readiness["source_hashes"])
        source_hashes[str(READINESS.resolve())] = sha256(READINESS)
        source_hashes[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
        source_hashes[str(ATTEMPT.resolve())] = sha256(ATTEMPT)
        if not frozen_eval._verify_hashes(source_hashes):
            raise ValueError("N1-147 final query source/checkpoint hash mismatch")
        items = list(face["episodes"])
        if (
            len(items) != 80
            or sum(item.get("pair_condition") == "known" for item in items) != 40
            or sum(item.get("pair_condition") == "unknown" for item in items) != 40
            or len({item.get("pair_id") for item in items}) != 40
        ):
            raise ValueError("v24 40-pair face shape changed after readiness")
        PROGRESS.open("x", encoding="utf-8", newline="\n").close()
        frozen_eval.PROGRESS = PROGRESS
        for spec in specs:
            budget.guard(reserve=10)
            rows, receipt = frozen_eval._score_endpoint(spec, items, "diagnostic", budget)
            model_loads += int(receipt["model_loaded"])
            diagnostic_queries += int(receipt["queries"])
            result_rows.extend(rows)
            receipts.append(receipt)
            frozen_eval._append_progress("diagnostic", receipt)
            if receipt.get("error"):
                errors.append(f"{spec['endpoint_id']} diagnostic: {receipt['error']}")
                break
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        frozen_eval.PROGRESS = original_progress

    unchanged = bool(source_hashes) and frozen_eval._verify_hashes(source_hashes)
    if not unchanged:
        errors.append("来源/checkpoint哈希在N1-147查询期间变化")
    execution_cpu = time.process_time() - started
    cumulative_cpu = prior_cpu + execution_cpu
    complete = (
        not errors
        and diagnostic_queries == 960
        and len(result_rows) == 960
        and len(receipts) == 12
        and unchanged
        and cumulative_cpu <= CPU_LIMIT_SECONDS
    )
    paired = _paired_summary(result_rows, face) if result_rows else {}
    report = {
        "schema": "n1-plan147-v24-paired-abstention-diagnostic-v1",
        "status": "complete" if complete else "stopped",
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
        "endpoint_count": 12,
        "diagnostic_queries": diagnostic_queries,
        "diagnostic_rows_completed": len(result_rows),
        "endpoint_receipts": receipts,
        "paired_summary": paired,
        "diagnostic_rows": result_rows,
        "model_loads": model_loads + int(readiness.get("model_loads", 0)),
        "checkpoint_loads": model_loads + int(readiness.get("checkpoint_loads", 0)),
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "v3_holdout_queries": 0,
        "cpu_seconds": round(execution_cpu, 6),
        "prior_cpu_seconds": round(prior_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "can_promote": False,
        "errors": errors,
        "summary_zh": f"v24配对弃答诊断{('完整' if complete else '停止')}：{diagnostic_queries}/960条；无训练/更新/dev/旧v3 holdout，不设置通过门，can_promote=false。",
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-147 matched known/unknown abstention diagnostic"
    )
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
                    "diagnostic_queries",
                    "model_queries",
                    "parameter_updates",
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
