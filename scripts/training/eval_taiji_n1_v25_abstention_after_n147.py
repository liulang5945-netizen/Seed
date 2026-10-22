"""PLAN-N1-148：仅接受diagnostic split的matched abstention scorer与runner。"""

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

from scripts.training import build_taiji_n1_v24_abstention_after_n146 as n147_builder
from scripts.training import build_taiji_n1_v25_abstention_after_n147 as face_builder
from scripts.training import eval_taiji_n1_v23_confirmation_after_n143 as frozen_eval
from scripts.training import eval_taiji_n1_v24_abstention_after_n146 as n147_eval
from scripts.training import seal_ast  # noqa: E402

PLAN = face_builder.PLAN
PLAN_SHA256 = face_builder.PLAN_SHA256
CPU_LIMIT_SECONDS = 3_600.0
SOURCE_PRECHECK = face_builder.SOURCE_PRECHECK
SOURCE_EXCLUSIONS = face_builder.SOURCE_EXCLUSIONS
FACE = face_builder.FACE
FACE_ATTEMPT = face_builder.FACE_ATTEMPT
FACE_CHECKS = face_builder.FACE_CHECKS
PREPARED = face_builder.PREPARED

READINESS = ROOT / "reports/taiji_n1_v25_abstention_readiness_20261019.json"
READINESS_ATTEMPT = ROOT / "reports/taiji_n1_v25_abstention_readiness_attempt_20261019.json"
SOURCE_BINDING = ROOT / "reports/taiji_n1_v25_abstention_source_binding_20261019.json"
ATTEMPT = ROOT / "reports/taiji_n1_v25_abstention_attempt_20261019.json"
PROGRESS = ROOT / "reports/taiji_n1_v25_abstention_progress_20261019.jsonl"
RESULT = ROOT / "reports/taiji_n1_v25_abstention_diagnostic_20261019.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan148_v25_abstention_scorer_contract.py"

CONDITIONS = ("known", "unknown")


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


class DurableRows(list):
    """Write each scored diagnostic row before the next model call."""

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


def validate_diagnostic_items(items: list[dict]) -> dict:
    if len(items) != 80 or len({item.get("id") for item in items}) != 80:
        raise ValueError("exactly 80 unique diagnostic items are required")
    if any(item.get("split") != "diagnostic" for item in items):
        raise ValueError("diagnostic split required; do not relabel as holdout")
    if any(item.get("arm", "position") != "position" for item in items):
        raise ValueError("frozen position arm required")
    if any(len(item.get("observations", [])) != 32 for item in items):
        raise ValueError("all v25 diagnostic items require 32 observations")
    pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in items:
        condition = item.get("pair_condition")
        if condition not in CONDITIONS or item.get("kind") != (
            "lookup" if condition == "known" else "unknown"
        ):
            raise ValueError(f"invalid matched diagnostic condition: {item.get('id')}")
        pairs[str(item.get("pair_id"))][condition] = item
    if len(pairs) != 40 or any(set(group) != set(CONDITIONS) for group in pairs.values()):
        raise ValueError("40 complete known/unknown pairs are required")
    for pair_id, group in pairs.items():
        known = group["known"]
        unknown = group["unknown"]
        start = known.get("target_start_index")
        target = known.get("entities", [None])[0]
        if (
            unknown.get("entities", [None])[0] != target
            or unknown.get("target_start_index") != start
            or known.get("question") != unknown.get("question")
            or known.get("target_fact_indices") != [start]
            or unknown.get("target_fact_indices") != []
            or known.get("observations", [None] * 32)[start]
            != f"{target} uses {known.get('answer')}."
            or any(
                known["observations"][index] != unknown["observations"][index]
                for index in range(32)
                if index != start
            )
        ):
            raise ValueError(f"matched pair contract mismatch: {pair_id}")
    return {
        "items": len(items),
        "pairs": len(pairs),
        "conditions": dict(Counter(item["pair_condition"] for item in items)),
    }


@torch.no_grad()
def measure_diagnostic(model, item: dict, arm: str, session: str) -> dict:
    """A diagnostic-only scorer; preserves the diagnostic split and never steps an optimizer."""
    if item.get("split") != "diagnostic" or arm != "position":
        raise ValueError("diagnostic split and frozen position arm are required")
    paired = frozen_eval.n132.e11.scorer.paired
    data = paired.project(item)
    paired.observe(model, data, session)
    tape = paired.evidence(model, session, arm)
    target = item["answer"].encode("utf-8")
    labels = torch.tensor([*target, 256])
    logits, _ = model.teacher_logits(tape, session, data.question, target)
    predicted = logits.argmax(dim=-1)
    trace = model.generate_trace(tape, session, data.question, 64)
    return {
        "id": item["id"],
        "pair_id": item["pair_id"],
        "pair_condition": item["pair_condition"],
        "kind": item["kind"],
        "split": "diagnostic",
        "target_start_index": item["target_start_index"],
        "paired_answer_length": item["paired_answer_length"],
        "answer_length": item["answer_length"],
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


def score_endpoint_diagnostic(
    model,
    optimizer,
    spec: dict,
    items: list[dict],
    *,
    fingerprint,
    budget_guard,
    output: DurableRows,
    accounting: dict,
) -> None:
    before = fingerprint(model.bundle(optimizer))
    _ = validate_diagnostic_items(items)
    try:
        for item in items:
            budget_guard()
            session = f"n1-v25-diagnostic-{spec['seed']}-{spec['branch']}-{item['id']}"
            if session in model._observations:
                raise ValueError("diagnostic session collides with frozen model history")
            accounting["model_queries"] += 1
            try:
                row = measure_diagnostic(model, item, "position", session)
            finally:
                model._observations.pop(session, None)
            if row.get("id") != item["id"] or row.get("pair_id") != item["pair_id"]:
                raise ValueError("diagnostic scorer returned a row for the wrong item")
            output.append(row)
    finally:
        if fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("diagnostic changed model, optimizer, RNG, or history bundle")


def _read_prepared_face() -> tuple[dict, dict, dict]:
    if not PLAN.is_file() or sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-148 frozen SHA-256 mismatch")
    paths = (SOURCE_PRECHECK, SOURCE_EXCLUSIONS, FACE, FACE_ATTEMPT, FACE_CHECKS, PREPARED)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("N1-148 face preparation is incomplete")
    precheck = read_json(SOURCE_PRECHECK)
    manifest = read_json(SOURCE_EXCLUSIONS)
    face = read_json(FACE)
    attempt = read_json(FACE_ATTEMPT)
    checks = read_json(FACE_CHECKS)
    prepared = read_json(PREPARED)
    if (
        precheck.get("status") != "complete"
        or precheck.get("plan_sha256") != PLAN_SHA256
        or precheck.get("source_union", {}).get("entity_exclusion_count") != 38_122
        or precheck.get("source_union", {}).get("word_exclusion_count") != 15_288
        or precheck.get("source_hashes_unchanged") is not True
        or precheck.get("model_queries") != 0
        or precheck.get("parameter_updates") != 0
        or manifest.get("schema") != "n1-plan148-v25-source-exclusions-v1"
        or manifest.get("status") != "ready_for_one_time_face_generation"
        or manifest.get("plan_sha256") != PLAN_SHA256
        or manifest.get("entity_count") != 38_122
        or manifest.get("word_value_count") != 15_288
        or len(set(manifest.get("excluded_entities", []))) != 38_122
        or len(set(manifest.get("excluded_word_values", []))) != 15_288
        or face_builder.digest_strings(set(manifest.get("excluded_entities", [])))
        != manifest.get("entity_exclusion_sha256")
        or face_builder.digest_strings(set(manifest.get("excluded_word_values", [])))
        != manifest.get("word_exclusion_sha256")
        or face.get("schema") != "n1-v25-matched-abstention-diagnostic-face-v1"
        or face.get("seed") != 125
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
        or not seal_ast.receipt_binding_matches(prepared.get("face_sha256"), FACE)
        or prepared.get("face_checks_sha256") != sha256(FACE_CHECKS)
        or prepared.get("source_exclusion_sha256") != sha256(SOURCE_EXCLUSIONS)
        or prepared.get("source_precheck_sha256") != sha256(SOURCE_PRECHECK)
        or prepared.get("source_hashes_unchanged") is not True
        or prepared.get("model_queries") != 0
        or prepared.get("parameter_updates") != 0
    ):
        raise ValueError("v25 face/source preparation differs from frozen contract")
    if not n147_builder.n144_builder._verify_hashes(manifest.get("source_hashes", {})):
        raise ValueError("N1-148 source exclusion hashes changed")
    if not n147_builder.n144_builder._verify_hashes(prepared.get("source_hashes", {})):
        raise ValueError("N1-148 prepared-face source hashes changed")
    verified = face_builder.validate_face(
        face,
        set(manifest["excluded_entities"]),
        set(manifest["excluded_word_values"]),
    )
    if verified.get("accepted") is not True:
        raise ValueError("saved v25 paired face no longer passes its contract")
    return manifest, face, prepared


def _endpoint_specs() -> list[dict]:
    return frozen_eval._endpoint_specs()


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
        Path(n147_builder.__file__).resolve(),
        Path(n147_eval.__file__).resolve(),
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


def _restore(spec: dict):
    return frozen_eval._restore_endpoint(spec)


def validate_only() -> dict:
    outputs = (READINESS, READINESS_ATTEMPT, SOURCE_BINDING, ATTEMPT, PROGRESS, RESULT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("PLAN-N1-148 readiness/query phase is write-once")
    started = time.process_time()
    frozen_eval.n120_eval.n120.configure()
    _manifest, _face, prepared = _read_prepared_face()
    specs = _endpoint_specs()
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan148-v25-readiness-attempt-v1",
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
    restores = []
    errors = []
    for spec in specs:
        before = time.process_time()
        model = optimizer = None
        try:
            model, optimizer, payload, fingerprint = _restore(spec)
            restores.append(
                {
                    "endpoint_id": spec["endpoint_id"],
                    "source": spec["source"],
                    "branch": spec["branch"],
                    "seed": spec["seed"],
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
    unchanged = n147_builder.n144_builder._verify_hashes(hashes)
    if not unchanged:
        errors.append("N1-148 source/checkpoint hash changed during readiness")
    cpu = time.process_time() - started
    prior_cpu = float(prepared["cpu_seconds"])
    cumulative = prior_cpu + cpu
    complete = not errors and len(restores) == 12 and cumulative <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan148-v25-checkpoint-readiness-v1",
        "status": "ready_for_diagnostic" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_hashes": hashes,
        "source_hashes_unchanged": unchanged,
        "endpoints": restores,
        "endpoint_count": len(restores),
        "independent_full_bundle_restores": len(restores),
        "all_full_bundles_equal": len(restores) == 12 and all(row["equal"] for row in restores),
        "checkpoint_loads": len(restores),
        "model_loads": len(restores),
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
    }
    write_new(READINESS, report)
    write_new(
        SOURCE_BINDING,
        {
            "schema": "n1-plan148-v25-readiness-source-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_hashes": hashes,
            "checkpoint_hashes": {
                str(Path(spec["path"]).resolve()): spec["sha256"] for spec in specs
            },
            "can_promote": False,
        },
    )
    report["summary_zh"] = (
        "12端点诊断bundle readiness完成；0查询/更新，可进入一次性v25诊断。"
        if complete
        else "bundle readiness停止；0查询，不重试。"
    )
    return report


def _verify_readiness() -> tuple[dict, dict, list[dict]]:
    if not READINESS.is_file() or not SOURCE_BINDING.is_file():
        raise FileNotFoundError("run --validate-only and pass readiness before N1-148 queries")
    readiness = read_json(READINESS)
    binding = read_json(SOURCE_BINDING)
    _manifest, face, prepared = _read_prepared_face()
    specs = _endpoint_specs()
    if (
        readiness.get("status") != "ready_for_diagnostic"
        or readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or not seal_ast.receipt_binding_matches(readiness.get("face_sha256"), FACE)
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
        raise ValueError("N1-148 readiness/source binding mismatch")
    hashes = dict(readiness["source_hashes"])
    hashes[str(READINESS.resolve())] = sha256(READINESS)
    hashes[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
    if not n147_builder.n144_builder._verify_hashes(hashes):
        raise ValueError("N1-148 query source/checkpoint hash mismatch")
    return readiness, face, specs


def _paired_summary(rows: list[dict], face: dict) -> dict:
    return n147_eval._paired_summary(rows, face)


def execute_approved() -> dict:
    if any(path.exists() for path in (ATTEMPT, PROGRESS, RESULT)):
        raise FileExistsError("PLAN-N1-148 v25 face is one-shot; no retry")
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan148-v25-diagnostic-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS) if READINESS.is_file() else None,
            "source_binding_sha256": sha256(SOURCE_BINDING) if SOURCE_BINDING.is_file() else None,
            "face_sha256": sha256(FACE) if FACE.is_file() else None,
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
    queries = 0
    model_loads = 0
    readiness = {}
    face = {}
    specs = []
    prior_cpu = 0.0
    sources = {}
    started = time.process_time()
    try:
        frozen_eval.n120_eval.n120.configure()
        readiness, face, specs = _verify_readiness()
        prior_cpu = float(readiness["cumulative_cpu_seconds"])
        budget = frozen_eval.CpuDeadline(
            CPU_LIMIT_SECONDS - prior_cpu - (time.process_time() - started)
        )
        items = list(face["episodes"])
        validation = validate_diagnostic_items(items)
        if validation != {"items": 80, "pairs": 40, "conditions": {"known": 40, "unknown": 40}}:
            raise ValueError("v25 diagnostic face counts changed after readiness")
        sources = dict(readiness["source_hashes"])
        sources[str(READINESS.resolve())] = sha256(READINESS)
        sources[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
        sources[str(ATTEMPT.resolve())] = sha256(ATTEMPT)
        if not n147_builder.n144_builder._verify_hashes(sources):
            raise ValueError("N1-148 source/checkpoint hash changed before first diagnostic call")
        PROGRESS.open("x", encoding="utf-8", newline="\n").close()
        for spec in specs:
            budget.guard(reserve=10)
            model = optimizer = None
            before = None
            accounting = {"model_queries": 0}
            output = DurableRows("diagnostic", spec, PROGRESS)
            receipt = {
                "endpoint_id": spec["endpoint_id"],
                "source": spec["source"],
                "branch": spec["branch"],
                "seed": spec["seed"],
                "queries": 0,
                "completed_rows": 0,
                "model_loaded": False,
                "checkpoint_sha256": spec["sha256"],
                "error": None,
            }
            endpoint_started = time.process_time()
            try:
                model, optimizer, payload, before = _restore(spec)
                receipt["model_loaded"] = True
                score_endpoint_diagnostic(
                    model,
                    optimizer,
                    spec,
                    items,
                    fingerprint=frozen_eval.n132.e11.scorer.paired.fingerprint,
                    budget_guard=lambda: budget.guard(reserve=5),
                    output=output,
                    accounting=accounting,
                )
            except BaseException as exc:
                receipt["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if model is not None and optimizer is not None and before is not None:
                    try:
                        if (
                            frozen_eval.n132.e11.scorer.paired.fingerprint(model.bundle(optimizer))
                            != before
                        ):
                            receipt["error"] = (
                                "ValueError: frozen diagnostic changed bundle fingerprint"
                            )
                    except BaseException as exc:
                        receipt["error"] = f"{type(exc).__name__}: {exc}"
                if model is not None:
                    del model
                if optimizer is not None:
                    del optimizer
            receipt["queries"] = accounting["model_queries"]
            receipt["completed_rows"] = len(output)
            receipt["bundle_fingerprint"] = before
            receipt["checkpoint_unchanged"] = (
                Path(spec["path"]).is_file() and sha256(spec["path"]) == spec["sha256"]
            )
            receipt["cpu_seconds"] = round(time.process_time() - endpoint_started, 6)
            if not receipt["checkpoint_unchanged"]:
                receipt["error"] = (
                    receipt["error"] or "ValueError: checkpoint bytes changed during diagnostic"
                )
            result_rows.extend(output)
            receipts.append(receipt)
            queries += int(receipt["queries"])
            model_loads += int(receipt["model_loaded"])
            append_jsonl(PROGRESS, {"event": "endpoint_progress", **receipt})
            if receipt["error"]:
                errors.append(f"{spec['endpoint_id']}: {receipt['error']}")
                break
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    unchanged = bool(sources) and n147_builder.n144_builder._verify_hashes(sources)
    if not unchanged:
        errors.append("N1-148 diagnostic source/checkpoint hash changed")
    execution_cpu = time.process_time() - started
    cumulative = prior_cpu + execution_cpu
    complete = (
        not errors
        and queries == 960
        and len(result_rows) == 960
        and len(receipts) == 12
        and unchanged
        and cumulative <= CPU_LIMIT_SECONDS
    )
    paired = _paired_summary(result_rows, face) if result_rows else {}
    report = {
        "schema": "n1-plan148-v25-matched-abstention-diagnostic-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_binding_sha256": sha256(SOURCE_BINDING),
        "attempt_sha256": sha256(ATTEMPT),
        "progress_sha256": sha256(PROGRESS) if PROGRESS.is_file() else None,
        "face_sha256": sha256(FACE),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_hashes": sources,
        "source_hashes_unchanged": unchanged,
        "endpoint_count": 12,
        "diagnostic_queries": queries,
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
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "can_promote": False,
        "errors": errors,
        "summary_zh": f"v25 diagnostic split弃答诊断{('完整' if complete else '停止')}：{queries}/960条；0训练/更新/dev/旧v3 holdout，can_promote=false。",
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-148 diagnostic-split matched abstention scorer"
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
                    "parameter_updates",
                    "can_promote",
                    "errors",
                )
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
