"""PLAN-N1-155匹配unknown短上下文续训与fresh诊断执行器。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_plan155_short_unknown_context_length as builder
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_plan153_v28_fresh_context_length_confirmation as n153
from scripts.training import run_taiji_n1_v27_length_robust_training_after_n149 as n150

PLAN = builder.PLAN
PLAN_SHA256 = builder.PLAN_SHA256
CPU_LIMIT_SECONDS = builder.CPU_LIMIT_SECONDS
SEEDS = builder.SEEDS
BRANCHES = builder.BRANCHES
ENDPOINTS = builder.ENDPOINTS
LENGTHS = builder.LENGTHS
KINDS = builder.KINDS
UPDATES_PER_MODEL = builder.UPDATES_PER_MODEL
TOTAL_UPDATES = builder.TOTAL_UPDATES
TRAIN_SCORE_QUERIES = builder.TRAIN_SCORE_QUERIES
FRESH_QUERIES = builder.FRESH_QUERIES
FRESH_QUERIES_PER_ENDPOINT = builder.FRESH_QUERIES_PER_ENDPOINT
FRESH_ITEMS_PER_KIND = builder.FRESH_ITEMS_PER_KIND
TRAIN_GATE_QUERIES_PER_ENDPOINT = 960
LENGTH_QUERIES_PER_ENDPOINT = 288
PARENT_TOTAL_UPDATES = n132.TOTAL_UPDATES + n132.ADDITIONAL_UPDATES

SOURCE_AUDIT = builder.SOURCE_AUDIT
SCHEDULE = builder.SCHEDULE
READINESS = builder.READINESS
READINESS_ATTEMPT = builder.READINESS_ATTEMPT
SOURCE_BINDING = builder.SOURCE_BINDING
CHECKPOINT_ROOT = builder.CHECKPOINT_ROOT
PREFLIGHT = builder.PREFLIGHT
PREFLIGHT_ATTEMPT = builder.PREFLIGHT_ATTEMPT
PREFLIGHT_BINDING = builder.PREFLIGHT_BINDING
ATTEMPT = builder.ATTEMPT
PROGRESS = builder.PROGRESS
RESULT = builder.RESULT
N150_RESULT = n150.RESULT
FACE = builder.FACE
FACE_AUDIT = builder.FACE_AUDIT
FACE_CHECKS = builder.FACE_CHECKS
UNKNOWN_HEX = builder.UNKNOWN.encode("utf-8").hex()


class Refuse(RuntimeError):
    """Frozen N1-155 checkpoint, training, gate or diagnostic contract failed."""


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


def _write_jsonl(stream, value: dict) -> None:
    json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")


def _fingerprint(model, optimizer) -> str:
    return n132.e11.scorer.paired.fingerprint(model.bundle(optimizer))


def _checkpoint_payload(
    model,
    optimizer,
    *,
    seed: int,
    branch: str,
    parent_path: Path,
    parent_hash: str,
    parent_total: int,
    additional_updates: int,
) -> dict:
    bundle = copy.deepcopy(model.bundle(optimizer))
    return {
        "schema": "n1-plan155a-v28-unknown8-rebalance-checkpoint-v1",
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": builder.CORRECTION_SHA256,
        "parent_plan_sha256": n150.PLAN_SHA256,
        "seed": seed,
        "arm": "position",
        "branch": branch,
        "parent_checkpoint_path": str(Path(parent_path).resolve()),
        "parent_checkpoint_sha256": parent_hash,
        "parent_total_updates": parent_total,
        "additional_updates": additional_updates,
        "total_updates": parent_total + additional_updates,
        "history_count": len(model._observations),
        "candidate": bundle,
        "bundle_fingerprint": _fingerprint(model, optimizer),
    }


def _save_checkpoint(model, optimizer, path: Path, **metadata) -> dict:
    if Path(path).exists():
        raise FileExistsError(f"N1-155 write-once checkpoint exists: {path}")
    payload = _checkpoint_payload(model, optimizer, **metadata)
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
    return payload


def _restore_payload(payload: dict):
    if (
        payload.get("schema") != "n1-plan155a-v28-unknown8-rebalance-checkpoint-v1"
        or payload.get("plan_sha256") != PLAN_SHA256
        or payload.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or payload.get("parent_plan_sha256") != n150.PLAN_SHA256
        or payload.get("arm") != "position"
        or payload.get("branch") not in BRANCHES
    ):
        raise ValueError("N1-155 checkpoint protocol mismatch")
    parent_path = Path(payload["parent_checkpoint_path"])
    if not parent_path.is_file() or sha256(parent_path) != payload["parent_checkpoint_sha256"]:
        raise ValueError(f"N1-155 parent checkpoint changed: {parent_path}")
    model = n132.LocalRelationEvidenceCandidate.restore_bundle(copy.deepcopy(payload["candidate"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=n132.FINAL_LR)
    optimizer.load_state_dict(copy.deepcopy(payload["candidate"]["optimizer"]))
    fingerprint = _fingerprint(model, optimizer)
    expected = int(payload["total_updates"])
    if (
        fingerprint != payload.get("bundle_fingerprint")
        or len(optimizer.state) != 41
        or any(int(optimizer.state[param]["step"]) != expected for param in model.parameters())
        or len(model._observations) != int(payload["history_count"])
    ):
        raise ValueError("N1-155 full model/Adam/RNG/history restore mismatch")
    return model, optimizer, payload, fingerprint


def _restore_checkpoint(path: Path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return _restore_payload(payload)


def _restore_worker(path: Path) -> dict:
    started = time.process_time()
    model, optimizer, payload, fingerprint = _restore_checkpoint(path)
    result = {
        "complete": True,
        "equal": fingerprint == payload["bundle_fingerprint"],
        "fingerprint": fingerprint,
        "bundle_fingerprint": payload["bundle_fingerprint"],
        "seed": payload["seed"],
        "branch": payload["branch"],
        "additional_updates": payload["additional_updates"],
        "total_updates": payload["total_updates"],
        "history_count": len(model._observations),
        "optimizer_parameters": len(optimizer.state),
        "cpu_seconds": round(time.process_time() - started, 6),
    }
    del model, optimizer
    return result


def _independent_restore(path: Path, budget: n132.CpuBudget) -> dict:
    budget.guard(reserve=5)
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--restore-worker", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=min(90, max(1.0, budget.limit - budget.used() - 1.0)),
        env=environment,
    )
    restored = json.loads(process.stdout)
    budget.children += float(restored["cpu_seconds"])
    budget.guard()
    if restored.get("complete") is not True or restored.get("equal") is not True:
        raise Refuse(f"independent N1-155 restore mismatch: {path}")
    return {"path": str(Path(path).resolve()), "sha256": sha256(path), **restored}


def _write_metrics(progress, event: str, seed: int, branch: str, step: int, row: dict) -> None:
    _write_jsonl(progress, {"event": event, "seed": seed, "branch": branch, "step": step, **row})


def _train_length_rows(
    model, optimizer, seed: int, branch: str, inputs: dict, views: dict, budget: n132.CpuBudget
) -> list[dict]:
    before = _fingerprint(model, optimizer)
    selected = n150._length_score_items(inputs)
    rows = []
    for length in LENGTHS:
        for base_item in selected[length]:
            budget.guard(reserve=5)
            key = n150_builder._context_key(base_item)
            item = n150._apply_context(base_item, key, length, views)
            session = f"n1-plan155-length-score-{seed}-{branch}-{length}-{item['id']}"
            if session in model._observations:
                raise Refuse(f"N1-155 length score session collision: {session}")
            try:
                row = n150._measure_train_length_view(model, item, session)
            finally:
                model._observations.pop(session, None)
            rows.append({"event": "length_score_row", "seed": seed, "branch": branch, **row})
    if len(rows) != LENGTH_QUERIES_PER_ENDPOINT or _fingerprint(model, optimizer) != before:
        raise Refuse(f"N1-155 length score count/fingerprint differs: {seed}/{branch}")
    return rows


@torch.no_grad()
def _measure_fresh_diagnostic(model, item: dict, session: str) -> dict:
    if item.get("split") != "diagnostic":
        raise ValueError("N1-155 fresh scorer requires diagnostic split")
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
        "answer_length": len(target),
        "teacher_exact": bool(torch.equal(predicted, labels)),
        "teacher_tokens_correct": int((predicted == labels).sum()),
        "teacher_token_count": len(labels),
        "teacher_ce": float(torch.nn.functional.cross_entropy(logits, labels)),
        "output_hex": trace.output.hex(),
        "raw_hex": trace.raw.hex(),
        "stop_reason": trace.stop_reason,
        "generated_steps": trace.generated_steps,
        "free_exact": trace.raw == target and trace.stop_reason == "boundary",
    }


def _fresh_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[
            (int(row["seed"]), str(row["branch"]), int(row["context_length"]), str(row["kind"]))
        ].append(row)
    summary = []
    for seed, branch in builder.ENDPOINTS:
        for length in LENGTHS:
            for kind in KINDS:
                items = groups[(seed, branch, length, kind)]
                if len(items) != FRESH_ITEMS_PER_KIND:
                    raise Refuse(f"fresh score stratum differs: {seed}/{branch}/{length}/{kind}")
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


def _verify_readiness() -> tuple[dict, dict, dict, dict, dict[str, str]]:
    if not all(path.is_file() for path in (READINESS, SOURCE_AUDIT, SCHEDULE, SOURCE_BINDING)):
        raise FileNotFoundError("N1-155 schedule readiness is required")
    readiness = read_json(READINESS)
    audit = read_json(SOURCE_AUDIT)
    manifest = read_json(SCHEDULE)
    binding = read_json(SOURCE_BINDING)
    if (
        readiness.get("complete") is not True
        or readiness.get("status") != "ready_for_zero_checkpoint_preflight"
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or readiness.get("prior_failed_attempt_cpu_seconds") != builder.PRIOR_FAILED_CPU_SECONDS
        or readiness.get("cumulative_cpu_seconds")
        != round(builder.PRIOR_FAILED_CPU_SECONDS + float(readiness.get("cpu_seconds", -1)), 6)
        or binding.get("status") != "complete"
        or binding.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or binding.get("readiness_sha256") != sha256(READINESS)
        or binding.get("source_hashes") != readiness.get("source_hashes")
        or readiness.get("schedule_sha256") != sha256(SCHEDULE)
        or audit.get("status") != "complete"
        or audit.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or audit.get("prior_failed_attempt_result_sha256") != builder.PRIOR_FAILED_RESULT_SHA256
        or audit.get("prior_failed_attempt_cpu_seconds") != builder.PRIOR_FAILED_CPU_SECONDS
    ):
        raise Refuse("N1-155 readiness/source binding is stale")
    if not n150_builder.verify_hashes(readiness.get("source_hashes", {})):
        raise Refuse("N1-155 readiness source hashes changed")
    context, bound, base_schedule, sources = builder._source_context()
    schedule = builder.build_schedule(base_schedule)
    expected = dict(schedule["manifest"], source_audit_sha256=sha256(SOURCE_AUDIT))
    if expected != manifest or expected["schedule_sha256"] != readiness.get(
        "schedule_identity_sha256"
    ):
        raise Refuse("N1-155 fixed task/length schedule changed")
    if not set(sources.items()).issubset(set(readiness["source_hashes"].items())):
        raise Refuse("N1-155 base source set differs from readiness")
    return (
        readiness,
        audit,
        context,
        {"bound": bound, "schedule": schedule},
        dict(readiness["source_hashes"]),
    )


def _parent_rows(n150_result: dict) -> dict[int, dict]:
    rows = {
        int(row["seed"]): row
        for row in n150_result.get("models", [])
        if row.get("branch") == builder.PARENT_BRANCH and row.get("training_sufficient") is True
    }
    if set(rows) != set(SEEDS):
        raise Refuse("all three N1-150 length-mixed parent checkpoints are required")
    return rows


def checkpoint_preflight() -> dict:
    if any(path.exists() for path in (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING)):
        raise FileExistsError("N1-155 zero bundle preflight is write-once")
    started = time.process_time()
    readiness, _audit, _context, runtime, sources = _verify_readiness()
    validation_cpu = time.process_time() - started
    prior_cpu = float(readiness["prior_failed_attempt_cpu_seconds"])
    remaining = CPU_LIMIT_SECONDS - prior_cpu - float(readiness["cpu_seconds"]) - validation_cpu
    if remaining <= 0:
        raise TimeoutError("N1-155 CPU budget exhausted during readiness revalidation")
    budget = n132.CpuBudget(remaining)
    budget.guard(reserve=45)
    if CHECKPOINT_ROOT.exists():
        raise FileExistsError(f"N1-155 checkpoint root already exists: {CHECKPOINT_ROOT}")
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=False)
    write_new(
        PREFLIGHT_ATTEMPT,
        {
            "schema": "n1-plan155a-zero-preflight-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "correction_addendum_sha256": builder.CORRECTION_SHA256,
            "prior_failed_attempt_cpu_seconds": prior_cpu,
            "readiness_sha256": sha256(READINESS),
            "zero_checkpoint_count_planned": 6,
            "parameter_updates": 0,
            "model_queries": 0,
            "can_promote": False,
        },
    )
    n150_result = read_json(N150_RESULT)
    parents = _parent_rows(n150_result)
    zero_rows = []
    restores = []
    errors = []
    source_hashes = dict(sources)
    source_hashes[str(READINESS.resolve())] = sha256(READINESS)
    source_hashes[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
    source_hashes[str(PREFLIGHT_ATTEMPT.resolve())] = sha256(PREFLIGHT_ATTEMPT)
    try:
        for seed in SEEDS:
            budget.guard(reserve=20)
            parent = parents[seed]
            parent_path = Path(parent["final_checkpoint_path"]).resolve()
            parent_hash = str(parent["final_checkpoint_sha256"])
            model, optimizer, payload, fingerprint = n150._restore_n150_checkpoint(parent_path)
            parent_total = int(parent["total_updates"])
            if fingerprint != parent.get("final_bundle_fingerprint"):
                raise Refuse(f"N1-150 parent fingerprint differs: seed{seed}")
            pair_fingerprints = set()
            for branch in BRANCHES:
                zero_path = CHECKPOINT_ROOT / f"s{seed}_{branch}_zero.pt"
                zero_payload = _save_checkpoint(
                    model,
                    optimizer,
                    zero_path,
                    seed=seed,
                    branch=branch,
                    parent_path=parent_path,
                    parent_hash=parent_hash,
                    parent_total=parent_total,
                    additional_updates=0,
                )
                if zero_payload["bundle_fingerprint"] != fingerprint:
                    raise Refuse(f"N1-155 zero save changed parent bundle: seed{seed}/{branch}")
                pair_fingerprints.add(zero_payload["bundle_fingerprint"])
                row = {
                    "seed": seed,
                    "branch": branch,
                    "path": str(zero_path.resolve()),
                    "sha256": sha256(zero_path),
                    "bundle_fingerprint": zero_payload["bundle_fingerprint"],
                    "parent_checkpoint_path": str(parent_path),
                    "parent_checkpoint_sha256": parent_hash,
                    "parent_total_updates": parent_total,
                }
                zero_rows.append(row)
                source_hashes[row["path"]] = row["sha256"]
            if len(pair_fingerprints) != 1:
                raise Refuse(f"N1-155 candidate/control zero bundles differ: seed{seed}")
            del model, optimizer, payload
        for row in zero_rows:
            budget.guard(reserve=5)
            restores.append(_independent_restore(Path(row["path"]), budget))
        if len(zero_rows) != 6 or len(restores) != 6:
            raise Refuse("N1-155 requires six zero checkpoints and six restores")
        if not n150_builder.verify_hashes(source_hashes):
            raise Refuse("N1-155 source or zero checkpoint SHA changed")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = time.process_time() - started + budget.children
    cumulative = prior_cpu + float(readiness["cpu_seconds"]) + cpu
    complete = (
        not errors
        and len(zero_rows) == 6
        and len(restores) == 6
        and cumulative <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan155a-zero-checkpoint-preflight-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": builder.CORRECTION_SHA256,
        "prior_failed_attempt_cpu_seconds": prior_cpu,
        "readiness_sha256": sha256(READINESS),
        "preflight_attempt_sha256": sha256(PREFLIGHT_ATTEMPT),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": n150_builder.verify_hashes(source_hashes),
        "zero_checkpoints": zero_rows,
        "independent_restores": restores,
        "zero_checkpoint_count": len(zero_rows),
        "independent_restore_count": len(restores),
        "candidate_control_zero_equal_per_seed": True,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "cpu_seconds": round(cpu, 6),
        "prior_readiness_cpu_seconds": float(readiness["cpu_seconds"]),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
    }
    report["summary_zh"] = (
        "六个zero full bundle保存并独立恢复6/6，candidate/control按seed fingerprint一致；0训练/查询。"
        if complete
        else "N1-155 zero bundle前置停止；不训练、不查询、不重试。"
    )
    write_new(PREFLIGHT, report)
    write_new(
        PREFLIGHT_BINDING,
        {
            "schema": "n1-plan155a-zero-preflight-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "correction_addendum_sha256": builder.CORRECTION_SHA256,
            "preflight_sha256": sha256(PREFLIGHT),
            "source_hashes": source_hashes,
            "zero_checkpoint_hashes": {row["path"]: row["sha256"] for row in zero_rows},
            "can_promote": False,
        },
    )
    return report


def _verify_preflight() -> tuple[dict, dict, dict, dict, dict, dict[str, str]]:
    if not all(
        path.is_file()
        for path in (
            READINESS,
            SOURCE_AUDIT,
            SCHEDULE,
            SOURCE_BINDING,
            PREFLIGHT,
            PREFLIGHT_BINDING,
        )
    ):
        raise FileNotFoundError("N1-155 readiness and six-zero-restore preflight are required")
    readiness = read_json(READINESS)
    source_binding = read_json(SOURCE_BINDING)
    schedule_manifest = read_json(SCHEDULE)
    preflight = read_json(PREFLIGHT)
    preflight_binding = read_json(PREFLIGHT_BINDING)
    if (
        readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or readiness.get("prior_failed_attempt_cpu_seconds") != builder.PRIOR_FAILED_CPU_SECONDS
        or readiness.get("cumulative_cpu_seconds")
        != round(builder.PRIOR_FAILED_CPU_SECONDS + float(readiness.get("cpu_seconds", -1)), 6)
        or source_binding.get("status") != "complete"
        or source_binding.get("plan_sha256") != PLAN_SHA256
        or source_binding.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or source_binding.get("readiness_sha256") != sha256(READINESS)
        or source_binding.get("source_hashes") != readiness.get("source_hashes")
        or preflight.get("complete") is not True
        or preflight.get("plan_sha256") != PLAN_SHA256
        or preflight.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or preflight.get("prior_failed_attempt_cpu_seconds") != builder.PRIOR_FAILED_CPU_SECONDS
        or preflight.get("cumulative_cpu_seconds")
        != round(
            builder.PRIOR_FAILED_CPU_SECONDS
            + float(readiness.get("cpu_seconds", -1))
            + float(preflight.get("cpu_seconds", -1)),
            6,
        )
        or preflight.get("zero_checkpoint_count") != 6
        or preflight.get("independent_restore_count") != 6
        or preflight.get("source_hashes_unchanged") is not True
        or preflight.get("model_queries") != 0
        or preflight.get("parameter_updates") != 0
        or preflight.get("optimizer_steps") != 0
        or preflight_binding.get("status") != "complete"
        or preflight_binding.get("plan_sha256") != PLAN_SHA256
        or preflight_binding.get("correction_addendum_sha256") != builder.CORRECTION_SHA256
        or preflight_binding.get("preflight_sha256") != sha256(PREFLIGHT)
        or preflight_binding.get("source_hashes") != preflight.get("source_hashes")
    ):
        raise Refuse("N1-155 readiness/zero restore preflight is stale or incomplete")
    if not n150_builder.verify_hashes(readiness.get("source_hashes", {})):
        raise Refuse("N1-155 readiness source/checkpoint hashes changed")
    if not n150_builder.verify_hashes(preflight.get("source_hashes", {})):
        raise Refuse("N1-155 zero bundle/source hashes changed")
    context, bound, base_schedule, sources = builder._source_context()
    schedule = builder.build_schedule(base_schedule)
    expected = dict(schedule["manifest"], source_audit_sha256=sha256(SOURCE_AUDIT))
    if expected != schedule_manifest or expected["schedule_sha256"] != readiness.get(
        "schedule_identity_sha256"
    ):
        raise Refuse("N1-155 fixed unknown/length schedule changed")
    return readiness, preflight, context, bound, schedule, dict(preflight["source_hashes"])


def _validate_progress_log(
    path: Path, *, expected_updates: int, expected_train_scores: int
) -> dict:
    counts = Counter()
    total = 0
    malformed = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            total += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                malformed.append({"line": number, "error": type(exc).__name__})
                continue
            counts[str(row.get("event"))] += 1
    if malformed:
        raise Refuse(f"N1-155 progress log has malformed rows: {malformed[:5]}")
    if (
        counts["training_update"] != expected_updates
        or counts["original_train_gate_row"] + counts["length_score_row"] != expected_train_scores
    ):
        raise Refuse(f"N1-155 progress event counts differ: {dict(counts)}")
    return {
        "line_count": total,
        "event_counts": dict(sorted(counts.items())),
        "malformed_rows": 0,
        "training_update_rows": counts["training_update"],
        "train_score_rows": counts["original_train_gate_row"] + counts["length_score_row"],
        "complete": True,
    }


def _score_fresh_face(
    models: list[dict],
    face: dict,
    budget: n132.CpuBudget,
) -> tuple[list[dict], list[dict], int, list[str]]:
    rows = []
    summaries = []
    queries = 0
    errors = []
    for record in models:
        seed, branch = int(record["seed"]), str(record["branch"])
        endpoint_rows = []
        model = optimizer = payload = None
        try:
            checkpoint = Path(record["final_checkpoint_path"]).resolve()
            if sha256(checkpoint) != record["final_checkpoint_sha256"]:
                raise Refuse(
                    f"N1-155 final checkpoint SHA changed before fresh scoring: {seed}/{branch}"
                )
            model, optimizer, payload, fingerprint = _restore_checkpoint(checkpoint)
            if fingerprint != record["final_bundle_fingerprint"]:
                raise Refuse(
                    f"N1-155 final full bundle differs before fresh scoring: {seed}/{branch}"
                )
            before = _fingerprint(model, optimizer)
            for base in face["items"]:
                for length in LENGTHS:
                    budget.guard(reserve=5)
                    item = n153._face_from_manifest_item(base, length)
                    session = f"n1-plan155-fresh-{seed}-{branch}-{item['parent_id']}-l{length}"
                    if session in model._observations:
                        raise Refuse(f"N1-155 fresh diagnostic session collision: {session}")
                    queries += 1
                    try:
                        row = _measure_fresh_diagnostic(model, item, session)
                    finally:
                        model._observations.pop(session, None)
                    endpoint_rows.append(
                        {"seed": seed, "branch": branch, "event": "fresh_diagnostic_row", **row}
                    )
            if len(endpoint_rows) != FRESH_QUERIES_PER_ENDPOINT:
                raise Refuse(f"N1-155 fresh query count differs: {seed}/{branch}")
            # Retain the full endpoint before any later summary validation can fail.
            rows.extend(endpoint_rows)
            summaries.extend(_fresh_summary(endpoint_rows))
            if _fingerprint(model, optimizer) != before:
                raise Refuse(
                    f"N1-155 fresh diagnostic scoring changed final bundle: {seed}/{branch}"
                )
        except BaseException as exc:
            if endpoint_rows and not any(
                item.get("seed") == seed and item.get("branch") == branch for item in rows
            ):
                rows.extend(endpoint_rows)
            errors.append(f"{seed}/{branch}: {type(exc).__name__}: {exc}")
            break
        finally:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
            if payload is not None:
                del payload
    return rows, summaries, queries, errors


def execute_approved() -> dict:
    started = time.process_time()
    readiness, preflight, _source_context, bound, schedule, base_sources = _verify_preflight()
    validation_cpu = time.process_time() - started
    remaining = CPU_LIMIT_SECONDS - float(preflight["cumulative_cpu_seconds"]) - validation_cpu
    if remaining <= 0:
        raise TimeoutError("N1-155 CPU ceiling exhausted during preflight revalidation")
    outputs = (ATTEMPT, PROGRESS, RESULT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("N1-155 training/diagnostic attempt is one-shot")
    budget = n132.CpuBudget(remaining)
    input_hashes = dict(preflight["source_hashes"])
    for path in (READINESS, SOURCE_AUDIT, SCHEDULE, SOURCE_BINDING, PREFLIGHT, PREFLIGHT_BINDING):
        input_hashes[str(Path(path).resolve())] = sha256(path)
    zero_rows = preflight.get("zero_checkpoints", [])
    if len(zero_rows) != 6:
        raise Refuse("N1-155 six zero bundles are required before training")
    input_hashes.update({row["path"]: row["sha256"] for row in zero_rows})
    if not n150_builder.verify_hashes(input_hashes):
        raise Refuse("N1-155 source/zero bundle hash mismatch before training")
    attempt = {
        "schema": "n1-plan155a-v28-unknown8-rebalance-attempt-v1",
        "status": "started",
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": builder.CORRECTION_SHA256,
        "prior_failed_attempt_result_sha256": builder.PRIOR_FAILED_RESULT_SHA256,
        "readiness_sha256": sha256(READINESS),
        "preflight_sha256": sha256(PREFLIGHT),
        "schedule_sha256": sha256(SCHEDULE),
        "planned_endpoints": len(ENDPOINTS),
        "updates_per_model": UPDATES_PER_MODEL,
        "planned_training_updates": TOTAL_UPDATES,
        "planned_train_score_queries": TRAIN_SCORE_QUERIES,
        "planned_fresh_queries": FRESH_QUERIES,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
    }
    write_new(ATTEMPT, attempt)
    update_count = 0
    train_score_queries = 0
    endpoint_records = []
    errors = []
    active = None
    model = optimizer = None
    partial_checkpoint = None
    training_sufficient_all = False
    face_status = "not_reached"
    face_rows = []
    face_summary = []
    fresh_queries = 0
    progress = PROGRESS.open("x", encoding="utf-8", newline="\n")
    try:
        context, schedule, slots = bound, schedule, schedule["slots"]
        schedule_rows = read_json(SCHEDULE)["schedule_rows"]
        zero_map = {(int(row["seed"]), str(row["branch"])): row for row in zero_rows}
        if set(zero_map) != {(seed, branch) for seed, branch in ENDPOINTS}:
            raise Refuse("N1-155 zero bundle set does not cover six endpoints")
        for seed in SEEDS:
            for branch in BRANCHES:
                budget.guard(reserve=60)
                zero = zero_map[(seed, branch)]
                model, optimizer, payload, fingerprint = _restore_checkpoint(Path(zero["path"]))
                parent_total = int(payload["parent_total_updates"])
                initial_history = len(model._observations)
                if fingerprint != zero["bundle_fingerprint"] or any(
                    int(optimizer.state[param]["step"]) != parent_total
                    for param in model.parameters()
                ):
                    raise Refuse(f"N1-155 zero bundle parent state mismatch: {seed}/{branch}")
                parent_path = Path(payload["parent_checkpoint_path"]).resolve()
                parent_hash = payload["parent_checkpoint_sha256"]
                active = {
                    "seed": seed,
                    "branch": branch,
                    "completed_updates": 0,
                    "parent_path": str(parent_path),
                    "parent_hash": parent_hash,
                    "parent_total": parent_total,
                }
                metrics_sums = Counter()
                length_counts = Counter()
                kind_length_counts = Counter()
                for row in schedule_rows:
                    if int(row["step"]) >= UPDATES_PER_MODEL:
                        raise Refuse("N1-155 schedule contains an out-of-range step")
                    length = int(
                        row["candidate_length"]
                        if branch == "unknown8_rebalanced"
                        else row["control_length"]
                    )
                    slot = slots[int(row["step"])]
                    item = n150._apply_context(
                        slot["item"], slot["item_key"], length, schedule["views"]
                    )
                    second = None
                    if slot.get("second") is not None:
                        second = n150._with_observations(
                            slot["second"], item["observations"], length
                        )
                    budget.guard(reserve=15)
                    result = n132._training_step(
                        model,
                        optimizer,
                        item,
                        second,
                        f"n1-plan155-{seed}-{branch}-{row['step']}",
                        context["g"],
                    )
                    active["completed_updates"] += 1
                    update_count += 1
                    length_counts[length] += 1
                    kind_length_counts[(row["kind"], length)] += 1
                    for name in ("total", "base_total", "qa", "observation", "binding"):
                        metrics_sums[name] += float(result[name])
                    _write_jsonl(
                        progress,
                        {
                            "event": "training_update",
                            "seed": seed,
                            "branch": branch,
                            "step": row["step"],
                            "parent_id": row["parent_id"],
                            "route": row["route"],
                            "kind": row["kind"],
                            "context_length": length,
                            "item_id": item["id"],
                            "second_id": None if second is None else second["id"],
                            **result,
                        },
                    )
                    if (int(row["step"]) + 1) % 64 == 0:
                        progress.flush()
                    if (int(row["step"]) + 1) % 512 == 0:
                        progress.flush()
                        print(
                            f"plan155 seed={seed} branch={branch} step={int(row['step'])+1}/{UPDATES_PER_MODEL} "
                            f"qa={result['qa']:.4f} obs={result['observation']:.4f} "
                            f"bind={result['binding']:.4f} cpu={budget.used():.1f}",
                            flush=True,
                        )
                expected_lengths = (
                    Counter({8: 3_456, 16: 2_880, 32: 2_880})
                    if branch == "unknown8_rebalanced"
                    else Counter({8: 3_072, 16: 3_072, 32: 3_072})
                )
                if (
                    active["completed_updates"] != UPDATES_PER_MODEL
                    or length_counts != expected_lengths
                ):
                    raise Refuse(f"N1-155 update/length counts differ: {seed}/{branch}")
                total_steps = {int(optimizer.state[param]["step"]) for param in model.parameters()}
                if (
                    total_steps != {parent_total + UPDATES_PER_MODEL}
                    or len(model._observations) != initial_history + UPDATES_PER_MODEL
                ):
                    raise Refuse(f"N1-155 final optimizer/history count differs: {seed}/{branch}")
                final_path = CHECKPOINT_ROOT / f"s{seed}_{branch}_final.pt"
                final_payload = _save_checkpoint(
                    model,
                    optimizer,
                    final_path,
                    seed=seed,
                    branch=branch,
                    parent_path=parent_path,
                    parent_hash=parent_hash,
                    parent_total=parent_total,
                    additional_updates=UPDATES_PER_MODEL,
                )
                final_restore = _independent_restore(final_path, budget)
                frozen_fingerprint = _fingerprint(model, optimizer)
                gate_rows, gate_summary, sufficient, gate_queries = n132._evaluate_train_only(
                    context["g"], model, optimizer, seed, context["inputs"], budget
                )
                require(
                    gate_queries == TRAIN_GATE_QUERIES_PER_ENDPOINT
                    and len(gate_rows) == TRAIN_GATE_QUERIES_PER_ENDPOINT,
                    f"N1-155 original gate denominator differs: {seed}/{branch}",
                )
                for score_row in gate_rows:
                    _write_jsonl(
                        progress,
                        {
                            "event": "original_train_gate_row",
                            "seed": seed,
                            "branch": branch,
                            **score_row,
                        },
                    )
                length_rows = _train_length_rows(
                    model, optimizer, seed, branch, context["inputs"], schedule["views"], budget
                )
                for score_row in length_rows:
                    _write_jsonl(progress, score_row)
                train_score_queries += gate_queries + len(length_rows)
                if _fingerprint(model, optimizer) != frozen_fingerprint:
                    raise Refuse(f"N1-155 train-only scoring changed final bundle: {seed}/{branch}")
                train_summary = n150._length_score_summary(length_rows)
                loss_means = {
                    key: float(metrics_sums[key]) / UPDATES_PER_MODEL
                    for key in ("total", "base_total", "qa", "observation", "binding")
                }
                final_record = {
                    "seed": seed,
                    "branch": branch,
                    "parent_checkpoint_path": str(parent_path),
                    "parent_checkpoint_sha256": parent_hash,
                    "zero_checkpoint_path": zero["path"],
                    "zero_checkpoint_sha256": zero["sha256"],
                    "zero_bundle_fingerprint": zero["bundle_fingerprint"],
                    "additional_updates": UPDATES_PER_MODEL,
                    "total_updates": parent_total + UPDATES_PER_MODEL,
                    "initial_history_count": initial_history,
                    "final_history_count": len(model._observations),
                    "optimizer_step_values": sorted(total_steps),
                    "context_length_update_counts": dict(sorted(length_counts.items())),
                    "task_kind_length_update_counts": {
                        f"{kind}/{length}": count
                        for (kind, length), count in sorted(kind_length_counts.items())
                    },
                    "loss_means": loss_means,
                    "original_train_gate": gate_summary,
                    "training_sufficient": sufficient,
                    "original_train_gate_queries": gate_queries,
                    "length_score_queries": len(length_rows),
                    "train_only_length_summary": train_summary,
                    "final_checkpoint_path": str(final_path.resolve()),
                    "final_checkpoint_sha256": sha256(final_path),
                    "final_bundle_fingerprint": final_payload["bundle_fingerprint"],
                    "final_independent_restore": final_restore,
                }
                endpoint_records.append(final_record)
                del model, optimizer, payload
                model = optimizer = None
                active = None
        progress.flush()
        progress.close()
        progress = None
        training_sufficient_all = len(endpoint_records) == 6 and all(
            row["training_sufficient"] for row in endpoint_records
        )
        face_status = "skipped_original_train_gate_failed"
        face_rows = []
        face_summary = []
        fresh_queries = 0
        if len(endpoint_records) == 6 and training_sufficient_all:
            final_hashes = {
                row["final_checkpoint_path"]: row["final_checkpoint_sha256"]
                for row in endpoint_records
            }
            progress_sha = sha256(PROGRESS)
            face_checks = builder.prepare_fresh_face(
                final_checkpoint_hashes=final_hashes, training_progress_sha256=progress_sha
            )
            face_manifest = read_json(FACE)
            if (
                face_checks.get("source_hashes_unchanged") is False
                or face_checks.get("total_queries") != FRESH_QUERIES
            ):
                raise Refuse("N1-155 fresh face source/query contract differs")
            face_rows, face_summary, fresh_queries, face_errors = _score_fresh_face(
                endpoint_records, face_manifest, budget
            )
            errors.extend(face_errors)
            face_status = (
                "complete" if not face_errors and fresh_queries == FRESH_QUERIES else "stopped"
            )

    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if progress is not None:
            try:
                progress.flush()
                progress.close()
            except Exception as exc:
                errors.append(f"progress close: {type(exc).__name__}: {exc}")

    if active is not None and model is not None and optimizer is not None:
        try:
            partial = CHECKPOINT_ROOT / f"s{active['seed']}_{active['branch']}_partial.pt"
            partial_payload = _save_checkpoint(
                model,
                optimizer,
                partial,
                seed=active["seed"],
                branch=active["branch"],
                parent_path=Path(active["parent_path"]),
                parent_hash=active["parent_hash"],
                parent_total=active["parent_total"],
                additional_updates=active["completed_updates"],
            )
            partial_checkpoint = {
                "path": str(partial.resolve()),
                "sha256": sha256(partial),
                "additional_updates": active["completed_updates"],
                "bundle_fingerprint": partial_payload["bundle_fingerprint"],
                "automatic_resume_allowed": False,
            }
        except BaseException as exc:
            partial_checkpoint = {"save_error": f"{type(exc).__name__}: {exc}"}
            errors.append(f"partial checkpoint save: {type(exc).__name__}: {exc}")
    else:
        partial_checkpoint = None

    progress_integrity = {"complete": False, "line_count": None, "event_counts": {}, "error": None}
    try:
        expected_train_rows = update_count + train_score_queries
        counts = Counter()
        total_lines = 0
        malformed = []
        if PROGRESS.is_file():
            for number, line in enumerate(PROGRESS.open(encoding="utf-8"), 1):
                total_lines += 1
                try:
                    event = json.loads(line).get("event")
                except Exception as exc:
                    malformed.append({"line": number, "error": type(exc).__name__})
                    continue
                counts[str(event)] += 1
        if (
            malformed
            or counts["training_update"] != update_count
            or counts["original_train_gate_row"] + counts["length_score_row"] != train_score_queries
            or total_lines != expected_train_rows
        ):
            raise Refuse(
                f"N1-155 single-handle progress row count differs: {dict(counts)} malformed={len(malformed)}"
            )
        progress_integrity = {
            "complete": True,
            "line_count": total_lines,
            "event_counts": dict(sorted(counts.items())),
            "malformed_rows": len(malformed),
            "progress_sha256": sha256(PROGRESS),
        }
    except BaseException as exc:
        errors.append(f"progress integrity: {type(exc).__name__}: {exc}")
        progress_integrity["error"] = f"{type(exc).__name__}: {exc}"

    input_hashes = dict(preflight["source_hashes"])
    for path in (
        READINESS,
        SOURCE_AUDIT,
        SCHEDULE,
        SOURCE_BINDING,
        PREFLIGHT,
        PREFLIGHT_BINDING,
        ATTEMPT,
    ):
        if path.is_file():
            input_hashes[str(path.resolve())] = sha256(path)
    if PROGRESS.is_file():
        input_hashes[str(PROGRESS.resolve())] = sha256(PROGRESS)
    new_checkpoints = {
        str(path.resolve()): sha256(path) for path in CHECKPOINT_ROOT.glob("*.pt") if path.is_file()
    }
    input_hashes.update(new_checkpoints)
    if not FACE.is_file() and FACE_AUDIT.exists():
        errors.append("N1-155 face audit exists without face manifest")
    for path in (FACE, FACE_AUDIT, FACE_CHECKS):
        if path.is_file():
            input_hashes[str(path.resolve())] = sha256(path)
    source_unchanged = n150_builder.verify_hashes(input_hashes)
    if not source_unchanged:
        errors.append("N1-155 source/checkpoint/face hash changed")

    stage_cpu = time.process_time() - started + budget.children
    cumulative = float(preflight["cumulative_cpu_seconds"]) + stage_cpu
    training_complete = (
        update_count == TOTAL_UPDATES
        and train_score_queries == TRAIN_SCORE_QUERIES
        and len(endpoint_records) == 6
        and len([path for path in CHECKPOINT_ROOT.glob("*_final.pt") if path.is_file()]) == 6
    )
    face_expected = len(endpoint_records) == 6 and training_sufficient_all
    face_complete = (
        face_status == "complete"
        and fresh_queries == FRESH_QUERIES
        and len(face_rows) == FRESH_QUERIES
    )
    protocol_complete = (
        training_complete
        and progress_integrity.get("complete") is True
        and source_unchanged
        and cumulative <= CPU_LIMIT_SECONDS
        and (
            (face_expected and face_complete)
            or (not face_expected and face_status == "skipped_original_train_gate_failed")
        )
        and not errors
    )
    report = {
        "schema": "n1-plan155a-v28-unknown8-rebalance-result-v1",
        "status": "complete" if protocol_complete else "stopped",
        "complete": protocol_complete,
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": builder.CORRECTION_SHA256,
        "prior_failed_attempt_result_sha256": builder.PRIOR_FAILED_RESULT_SHA256,
        "prior_failed_attempt_cpu_seconds": builder.PRIOR_FAILED_CPU_SECONDS,
        "readiness_sha256": sha256(READINESS),
        "source_binding_sha256": sha256(SOURCE_BINDING),
        "preflight_sha256": sha256(PREFLIGHT),
        "preflight_binding_sha256": sha256(PREFLIGHT_BINDING),
        "attempt_sha256": sha256(ATTEMPT),
        "source_hashes": input_hashes,
        "source_hashes_unchanged": source_unchanged,
        "models": endpoint_records,
        "partial_checkpoint": partial_checkpoint,
        "training_updates": update_count,
        "parameter_updates": update_count,
        "optimizer_steps": update_count,
        "train_score_queries": train_score_queries,
        "training_complete": training_complete,
        "training_sufficient_all_endpoints": training_sufficient_all,
        "fresh_face_status": face_status,
        "fresh_face_sha256": sha256(FACE) if FACE.is_file() else None,
        "fresh_face_audit_sha256": sha256(FACE_AUDIT) if FACE_AUDIT.is_file() else None,
        "fresh_diagnostic_queries": fresh_queries,
        "fresh_diagnostic_rows": face_rows,
        "fresh_diagnostic_summary_by_endpoint_length_kind": face_summary,
        "progress_integrity": progress_integrity,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "cpu_seconds": round(stage_cpu, 6),
        "prior_readiness_preflight_cpu_seconds": float(preflight["cumulative_cpu_seconds"]),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": f"N1-155训练{('完整' if training_complete else '停止')}：{update_count}/{TOTAL_UPDATES}次更新、{train_score_queries}/{TRAIN_SCORE_QUERIES}条train-only评分；fresh诊断状态={face_status}、{fresh_queries}/{FRESH_QUERIES}条，累积CPU{cumulative:.3f}/{CPU_LIMIT_SECONDS:.0f}秒，0 dev/holdout/confirmation，can_promote=false。",
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-155 unknown short-context rebalanced training and fresh diagnostic"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--checkpoint-preflight", action="store_true")
    modes.add_argument("--execute-approved", action="store_true")
    modes.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker is not None:
        result = _restore_worker(args.restore_worker)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0 if result.get("complete") is True and result.get("equal") is True else 1
    result = checkpoint_preflight() if args.checkpoint_preflight else execute_approved()
    keys = (
        "status",
        "complete",
        "cpu_seconds",
        "cumulative_cpu_seconds",
        "zero_checkpoint_count",
        "independent_restore_count",
        "training_updates",
        "train_score_queries",
        "training_sufficient_all_endpoints",
        "fresh_face_status",
        "fresh_diagnostic_queries",
        "progress_integrity",
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
