"""PLAN-N1-150 matched long-context continuation with an eight-row control."""

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

from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import seal_ast  # noqa: E402

PLAN = builder.PLAN
PLAN_SHA256 = builder.PLAN_SHA256
SCORING_ADDENDUM_SHA256 = builder.SCORING_ADDENDUM_SHA256
RESOURCE_ADDENDUM_SHA256 = builder.RESOURCE_ADDENDUM_SHA256
CPU_LIMIT_SECONDS = 18_000.0
SEEDS = builder.SEEDS
BRANCHES = builder.BRANCHES
UPDATES_PER_MODEL = builder.UPDATES_PER_MODEL
EXPECTED_TOTAL_UPDATES = builder.EXPECTED_TOTAL_UPDATES
ORIGINAL_GATE_QUERIES_PER_ENDPOINT = 960
LENGTH_SCORE_QUERIES_PER_ENDPOINT = 288
EXPECTED_TOTAL_TRAIN_SCORE_QUERIES = 7_488
PRECHECK = builder.PRECHECK
SOURCE_EXCLUSIONS = builder.SOURCE_EXCLUSIONS
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
TEST = ROOT / "tests/taiji_native/test_n1_plan150_v27_length_robust_training_contract.py"


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


def _fingerprint(model, optimizer) -> str:
    return n132.e11.scorer.paired.fingerprint(model.bundle(optimizer))


def _n150_payload(
    model,
    optimizer,
    *,
    seed: int,
    branch: str,
    parent_path: Path,
    parent_hash: str,
    additional_updates: int,
) -> dict:
    candidate = copy.deepcopy(model.bundle(optimizer))
    return {
        "schema": "n1-plan150-v27-length-robust-checkpoint-v1",
        "plan_sha256": PLAN_SHA256,
        "scoring_addendum_sha256": SCORING_ADDENDUM_SHA256,
        "resource_addendum_sha256": RESOURCE_ADDENDUM_SHA256,
        "seed": seed,
        "arm": "position",
        "branch": branch,
        "parent_checkpoint_path": str(Path(parent_path).resolve()),
        "parent_checkpoint_sha256": parent_hash,
        "parent_total_updates": n132.TOTAL_UPDATES,
        "additional_updates": additional_updates,
        "total_updates": n132.TOTAL_UPDATES + additional_updates,
        "history_count": len(model._observations),
        "candidate": candidate,
        "bundle_fingerprint": _fingerprint(model, optimizer),
    }


def _save_n150_checkpoint(model, optimizer, path: Path, **metadata) -> dict:
    if Path(path).exists():
        raise FileExistsError(f"N1-150 write-once checkpoint already exists: {path}")
    payload = _n150_payload(model, optimizer, **metadata)
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
    return payload


def _restore_n150_payload(payload: dict):
    if (
        payload.get("schema") != "n1-plan150-v27-length-robust-checkpoint-v1"
        or payload.get("plan_sha256") != PLAN_SHA256
        or payload.get("scoring_addendum_sha256") != SCORING_ADDENDUM_SHA256
        or payload.get("resource_addendum_sha256") != RESOURCE_ADDENDUM_SHA256
        or payload.get("arm") != "position"
        or payload.get("branch") not in builder.BRANCHES
    ):
        raise ValueError("N1-150 checkpoint protocol mismatch")
    parent_path = Path(payload["parent_checkpoint_path"])
    if not parent_path.is_file() or sha256(parent_path) != payload["parent_checkpoint_sha256"]:
        raise ValueError(f"N1-150 parent checkpoint changed: {parent_path}")
    model = n132.LocalRelationEvidenceCandidate.restore_bundle(copy.deepcopy(payload["candidate"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=n132.FINAL_LR)
    optimizer.load_state_dict(copy.deepcopy(payload["candidate"]["optimizer"]))
    fingerprint = _fingerprint(model, optimizer)
    expected_steps = int(payload["total_updates"])
    if (
        fingerprint != payload.get("bundle_fingerprint")
        or len(optimizer.state) != 41
        or any(
            int(optimizer.state[param]["step"]) != expected_steps for param in model.parameters()
        )
        or len(model._observations) != int(payload["history_count"])
    ):
        raise ValueError("N1-150 complete model/Adam/RNG/history restore mismatch")
    return model, optimizer, payload, fingerprint


def _restore_n150_checkpoint(path: Path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return _restore_n150_payload(payload)


def _restore_worker(path: Path) -> dict:
    started = time.process_time()
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model, optimizer, payload, fingerprint = _restore_n150_payload(payload)
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
        raise ValueError(f"independent N1-150 restore mismatch: {path}")
    return {"path": str(Path(path).resolve()), "sha256": sha256(path), **restored}


def _runtime() -> tuple[dict, dict]:
    exclusions = read_json(SOURCE_EXCLUSIONS)
    bound = n132._build_context(load_inputs=True)
    schedule = builder.build_schedule(
        bound["g"],
        bound["inputs"],
        set(exclusions["excluded_entities"]),
        set(exclusions["excluded_word_values"]),
    )
    return bound, schedule


def _verify_readiness() -> tuple[dict, dict, dict, dict]:
    readiness = read_json(READINESS)
    binding = read_json(SOURCE_BINDING)
    precheck = read_json(PRECHECK)
    schedule_manifest = read_json(SCHEDULE)
    if (
        readiness.get("complete") is not True
        or readiness.get("status") != "ready_for_zero_checkpoint_preflight"
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("scoring_addendum_sha256") != SCORING_ADDENDUM_SHA256
        or readiness.get("resource_addendum_sha256") != RESOURCE_ADDENDUM_SHA256
        or readiness.get("source_hashes_unchanged") is not True
        or binding.get("status") != "complete"
        or binding.get("plan_sha256") != PLAN_SHA256
        or binding.get("readiness_sha256") != sha256(READINESS)
        or binding.get("source_hashes") != readiness.get("source_hashes")
        or binding.get("schedule_sha256") != sha256(SCHEDULE)
        or not seal_ast.receipt_binding_matches(
            binding.get("source_exclusion_sha256"), SOURCE_EXCLUSIONS
        )
        or binding.get("resource_addendum_sha256") != RESOURCE_ADDENDUM_SHA256
        or precheck.get("status") != "complete"
        or precheck.get("resource_addendum_sha256") != RESOURCE_ADDENDUM_SHA256
        or readiness.get("precheck_sha256") != sha256(PRECHECK)
        or not seal_ast.receipt_binding_matches(
            readiness.get("source_exclusion_sha256"), SOURCE_EXCLUSIONS
        )
        or readiness.get("schedule_sha256") != sha256(SCHEDULE)
        or schedule_manifest.get("resource_addendum_sha256") != RESOURCE_ADDENDUM_SHA256
    ):
        raise ValueError("N1-150 readiness/source binding mismatch")
    if not builder.verify_hashes(readiness["source_hashes"]):
        raise ValueError("N1-150 source/checkpoint hashes changed")
    bound, schedule = _runtime()
    recomputed = dict(
        schedule["manifest"],
        source_exclusion_sha256=sha256(SOURCE_EXCLUSIONS),
        resource_addendum_sha256=RESOURCE_ADDENDUM_SHA256,
    )
    if recomputed != schedule_manifest or recomputed["schedule_sha256"] != readiness.get(
        "schedule_identity_sha256"
    ):
        raise ValueError("N1-150 schedule/context bank differs from frozen readiness")
    return (
        readiness,
        binding,
        precheck,
        {"bound": bound, "schedule": schedule, "schedule_manifest": schedule_manifest},
    )


def checkpoint_preflight() -> dict:
    if any(path.exists() for path in (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING)):
        raise FileExistsError("N1-150 zero checkpoint preflight is write-once")
    started = time.process_time()
    readiness, _binding, _precheck, runtime = _verify_readiness()
    validation_cpu = time.process_time() - started
    remaining_cpu = CPU_LIMIT_SECONDS - float(readiness["cumulative_cpu_seconds"]) - validation_cpu
    if remaining_cpu <= 0:
        raise TimeoutError("PLAN-N1-150 CPU ceiling exhausted during readiness revalidation")
    budget = n132.CpuBudget(remaining_cpu)
    budget.guard(reserve=30)
    if CHECKPOINT_ROOT.exists():
        raise FileExistsError(f"N1-150 checkpoint directory already exists: {CHECKPOINT_ROOT}")
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=False)
    write_new(
        PREFLIGHT_ATTEMPT,
        {
            "schema": "n1-plan150-v27-zero-checkpoint-preflight-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "scoring_addendum_sha256": SCORING_ADDENDUM_SHA256,
            "resource_addendum_sha256": RESOURCE_ADDENDUM_SHA256,
            "readiness_sha256": sha256(READINESS),
            "schedule_sha256": sha256(SCHEDULE),
            "planned_zero_checkpoints": 6,
            "parameter_updates": 0,
            "model_queries": 0,
            "can_promote": False,
        },
    )
    saved = []
    restores = []
    errors = []
    source_hashes = dict(readiness["source_hashes"])
    source_hashes[str(READINESS.resolve())] = sha256(READINESS)
    source_hashes[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
    source_hashes[str(PREFLIGHT_ATTEMPT.resolve())] = sha256(PREFLIGHT_ATTEMPT)
    try:
        parents, parent_hashes = builder.parent_checkpoints()
        for seed in SEEDS:
            budget.guard(reserve=30)
            parent = parents[seed]
            parent_path = Path(parent["final_checkpoint_path"]).resolve()
            parent_hash = parent["final_checkpoint_sha256"]
            parent_payload = torch.load(parent_path, map_location="cpu", weights_only=True)
            model, optimizer = n132._restore_bundle_payload(parent_payload)
            before = _fingerprint(model, optimizer)
            for branch in BRANCHES:
                zero_path = CHECKPOINT_ROOT / f"s{seed}_{branch}_zero.pt"
                payload = _save_n150_checkpoint(
                    model,
                    optimizer,
                    zero_path,
                    seed=seed,
                    branch=branch,
                    parent_path=parent_path,
                    parent_hash=parent_hash,
                    additional_updates=0,
                )
                if payload["bundle_fingerprint"] != before:
                    raise ValueError(
                        f"zero-copy payload changed parent bundle: seed{seed}/{branch}"
                    )
                saved.append(
                    {
                        "seed": seed,
                        "branch": branch,
                        "path": str(zero_path.resolve()),
                        "sha256": sha256(zero_path),
                        "bundle_fingerprint": before,
                    }
                )
            if _fingerprint(model, optimizer) != before:
                raise ValueError(f"zero-copy save changed source model: seed{seed}")
            del model, optimizer
        for record in saved:
            restores.append(_independent_restore(Path(record["path"]), budget))
        for seed in SEEDS:
            pair = [row for row in saved if row["seed"] == seed]
            if len(pair) != 2 or len({row["bundle_fingerprint"] for row in pair}) != 1:
                raise ValueError(f"candidate/control zero bundle mismatch: seed{seed}")
        source_hashes.update({row["path"]: row["sha256"] for row in saved})
        source_hashes[str(READINESS.resolve())] = sha256(READINESS)
        source_hashes[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
        source_hashes[str(PREFLIGHT_ATTEMPT.resolve())] = sha256(PREFLIGHT_ATTEMPT)
        if not builder.verify_hashes(source_hashes):
            raise ValueError("N1-150 source or zero checkpoint SHA changed during preflight")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = time.process_time() - started + budget.children
    cumulative = float(readiness["cumulative_cpu_seconds"]) + cpu
    complete = (
        not errors and len(saved) == 6 and len(restores) == 6 and cumulative <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan150-v27-zero-checkpoint-preflight-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "scoring_addendum_sha256": SCORING_ADDENDUM_SHA256,
        "resource_addendum_sha256": RESOURCE_ADDENDUM_SHA256,
        "readiness_sha256": sha256(READINESS),
        "schedule_sha256": sha256(SCHEDULE),
        "preflight_attempt_sha256": sha256(PREFLIGHT_ATTEMPT),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": builder.verify_hashes(source_hashes),
        "zero_checkpoints": saved,
        "independent_restores": restores,
        "zero_checkpoint_count": len(saved),
        "independent_restore_count": len(restores),
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "model_queries": 0,
        "cpu_seconds": round(cpu, 6),
        "prior_cpu_seconds": round(float(readiness["cumulative_cpu_seconds"]), 6),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
    }
    report["summary_zh"] = (
        "六个zero bundle已保存并独立恢复6/6，三seed候选/控制fingerprint一致；0更新/查询。"
        if complete
        else "zero checkpoint preflight停止；不训练、不重试。"
    )
    write_new(PREFLIGHT, report)
    write_new(
        PREFLIGHT_BINDING,
        {
            "schema": "n1-plan150-v27-zero-checkpoint-preflight-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "scoring_addendum_sha256": SCORING_ADDENDUM_SHA256,
            "resource_addendum_sha256": RESOURCE_ADDENDUM_SHA256,
            "preflight_sha256": sha256(PREFLIGHT),
            "source_hashes": source_hashes,
            "zero_checkpoint_hashes": {row["path"]: row["sha256"] for row in saved},
            "can_promote": False,
        },
    )
    return report


def _entities_for_question(item: dict, observations: list[str]) -> list[str]:
    subject = n132._query_subject(item["question"])
    names = [subject]
    seen = {subject}
    for row in observations:
        parts = row.split()
        if parts and parts[0] not in seen:
            names.append(parts[0])
            seen.add(parts[0])
    return names


def _apply_context(
    item: dict, context_key: str, length: int, views: dict[str, dict[int, list[str]]]
) -> dict:
    if context_key not in views or length not in views[context_key]:
        raise ValueError(f"train item lacks frozen length view: {item.get('id')}/{length}")
    return _with_observations(item, list(views[context_key][length]), length)


def _with_observations(item: dict, observations: list[str], length: int) -> dict:
    result = copy.deepcopy(item)
    observations = list(observations)
    result["observations"] = observations
    result["entities"] = _entities_for_question(result, observations)
    result["context_length"] = length
    if result.get("split") != "train":
        raise ValueError("length-expanded training item must remain train split")
    return result


@torch.no_grad()
def _measure_train_length_view(model, item: dict, session: str) -> dict:
    if item.get("split") != "train":
        raise ValueError("length descriptive scores must remain train-only")
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
        "parent_id": item.get("parent_id"),
        "kind": item["kind"],
        "split": "train",
        "context_length": item["context_length"],
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


def _length_score_items(inputs: dict) -> dict[int, list[dict]]:
    by_kind = {
        kind: sorted(
            (item for item in inputs["items"] if item.get("kind") == kind),
            key=lambda row: row["id"],
        )
        for kind in builder.TRAIN_KINDS
    }
    if any(len(rows) != 48 for rows in by_kind.values()):
        raise ValueError("original train source must retain 48 items per task kind")
    selected = [item for kind in builder.TRAIN_KINDS for item in by_kind[kind][:24]]
    if len(selected) != 96:
        raise ValueError("length-view descriptor subset must contain 96 base tasks")
    return {length: selected for length in builder.LENGTHS}


def _score_length_descriptors(
    model,
    optimizer,
    seed: int,
    branch: str,
    inputs: dict,
    views: dict,
    budget: n132.CpuBudget,
    progress: Path,
):
    before = _fingerprint(model, optimizer)
    rows = []
    queries = 0
    for length in builder.LENGTHS:
        for base_item in _length_score_items(inputs)[length]:
            budget.guard(reserve=5)
            key = builder._context_key(base_item)
            item = _apply_context(base_item, key, length, views)
            session = f"n1-plan150-length-score-{seed}-{branch}-{length}-{item['id']}"
            if session in model._observations:
                raise ValueError(f"length-score session collides with model history: {session}")
            try:
                row = _measure_train_length_view(model, item, session)
            finally:
                model._observations.pop(session, None)
            durable = {"event": "length_score_row", "seed": seed, "branch": branch, **row}
            append_jsonl(progress, durable)
            rows.append(durable)
            queries += 1
    if queries != 288 or _fingerprint(model, optimizer) != before:
        raise ValueError("length-score count or full-bundle fingerprint changed")
    return rows, queries


def _length_score_summary(rows: list[dict]) -> dict:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["context_length"]), str(row["kind"]))].append(row)
    summary = {}
    for length in builder.LENGTHS:
        summary[str(length)] = {}
        for kind in builder.TRAIN_KINDS:
            selected = grouped[(length, kind)]
            if len(selected) != 24:
                raise ValueError(f"length score requires 24 train items per class: {length}/{kind}")
            exact = sum(row.get("free_exact") is True for row in selected)
            abstain = sum(row.get("output_hex") == b"unknown".hex() for row in selected)
            summary[str(length)][kind] = {
                "items": len(selected),
                "exact": exact,
                "literal_unknown": abstain,
            }
    return summary


def run_training() -> dict:
    start = time.process_time()
    readiness, preflight, _preflight_binding, bound, schedule = _verify_preflight()
    validation_cpu = time.process_time() - start
    remaining_cpu = CPU_LIMIT_SECONDS - float(preflight["cumulative_cpu_seconds"]) - validation_cpu
    if remaining_cpu <= 0:
        raise TimeoutError("PLAN-N1-150 CPU ceiling exhausted during preflight revalidation")
    outputs = (ATTEMPT, PROGRESS, RESULT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("N1-150 training result is one-shot; no resume/retry")
    budget = n132.CpuBudget(remaining_cpu)
    source_hashes = dict(preflight["source_hashes"])
    for path in (
        READINESS,
        SOURCE_BINDING,
        PRECHECK,
        SOURCE_EXCLUSIONS,
        SCHEDULE,
        PREFLIGHT,
        PREFLIGHT_BINDING,
    ):
        source_hashes[str(Path(path).resolve())] = sha256(path)
    zero_records = preflight.get("zero_checkpoints", [])
    for row in zero_records:
        source_hashes[str(Path(row["path"]).resolve())] = row["sha256"]
    if len(zero_records) != 6 or not builder.verify_hashes(source_hashes):
        raise ValueError("N1-150 zero checkpoint/source chain is incomplete before training")
    attempt = {
        "schema": "n1-plan150-v27-length-training-attempt-v1",
        "status": "started",
        "plan_sha256": PLAN_SHA256,
        "scoring_addendum_sha256": SCORING_ADDENDUM_SHA256,
        "resource_addendum_sha256": builder.RESOURCE_ADDENDUM_SHA256,
        "readiness_sha256": sha256(READINESS),
        "preflight_sha256": sha256(PREFLIGHT),
        "schedule_sha256": sha256(SCHEDULE),
        "planned_models": 6,
        "updates_per_model": UPDATES_PER_MODEL,
        "planned_updates": EXPECTED_TOTAL_UPDATES,
        "planned_train_score_queries": EXPECTED_TOTAL_TRAIN_SCORE_QUERIES,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
    }
    write_new(ATTEMPT, attempt)
    update_count = 0
    train_score_queries = 0
    records = []
    errors = []
    partial_checkpoint = None
    active = None
    model = optimizer = None
    progress = PROGRESS.open("x", encoding="utf-8", newline="\n")
    try:
        g, inputs = bound["g"], bound["inputs"]
        schedule_manifest = read_json(SCHEDULE)
        if schedule_manifest.get("schedule_sha256") != schedule["manifest"].get("schedule_sha256"):
            raise ValueError("N1-150 schedule changed after preflight")
        parent_rows, parent_hashes = builder.parent_checkpoints()
        zero_by_key = {(int(row["seed"]), row["branch"]): row for row in zero_records}
        if set(zero_by_key) != {
            (seed, branch) for seed in builder.SEEDS for branch in builder.BRANCHES
        }:
            raise ValueError("N1-150 zero checkpoint set does not cover all six endpoints")

        for seed in builder.SEEDS:
            for branch in builder.BRANCHES:
                budget.guard(reserve=60)
                zero = zero_by_key[(seed, branch)]
                model, optimizer, zero_payload, before = _restore_n150_checkpoint(
                    Path(zero["path"])
                )
                active = {
                    "seed": seed,
                    "branch": branch,
                    "completed_updates": 0,
                    "parent_checkpoint_path": zero_payload["parent_checkpoint_path"],
                    "parent_checkpoint_sha256": zero_payload["parent_checkpoint_sha256"],
                }
                initial_history = len(model._observations)
                expected_parent_steps = n132.TOTAL_UPDATES
                if any(
                    int(optimizer.state[param]["step"]) != expected_parent_steps
                    for param in model.parameters()
                ):
                    raise ValueError(
                        f"zero bundle Adam state is not at N1-132 endpoint: seed{seed}/{branch}"
                    )
                step_losses = []
                context_length_counts = Counter()
                for slot in schedule["slots"]:
                    budget.guard(reserve=20)
                    context_length = (
                        slot["candidate_length"]
                        if branch == "length_mixed"
                        else builder.SHORT_CONTROL_LENGTH
                    )
                    item = _apply_context(
                        slot["item"], slot["item_key"], context_length, schedule["views"]
                    )
                    second = None
                    if slot["second"] is not None:
                        second = _with_observations(
                            slot["second"], item["observations"], context_length
                        )
                    metrics = n132._training_step(
                        model,
                        optimizer,
                        item,
                        second,
                        f"n1-plan150-{seed}-{branch}-{slot['step']}",
                        g,
                    )
                    step_losses.append(metrics)
                    context_length_counts[context_length] += 1
                    active["completed_updates"] += 1
                    update_count += 1
                    progress.write(
                        json.dumps(
                            {
                                "event": "training_update",
                                "seed": seed,
                                "branch": branch,
                                "step": slot["step"],
                                "parent_id": slot["parent_id"],
                                "route": slot["route"],
                                "kind": slot["kind"],
                                "context_length": context_length,
                                "item_id": item["id"],
                                "second_id": None if second is None else second["id"],
                                **metrics,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    if (slot["step"] + 1) % 64 == 0:
                        progress.flush()
                    if (slot["step"] + 1) % 512 == 0:
                        progress.flush()
                        print(
                            f"plan150 seed={seed} branch={branch} step={slot['step'] + 1}/{UPDATES_PER_MODEL} "
                            f"qa={metrics['qa']:.4f} obs={metrics['observation']:.4f} "
                            f"bind={metrics['binding']:.4f} cpu={budget.used():.1f}",
                            flush=True,
                        )
                total_steps = {int(optimizer.state[param]["step"]) for param in model.parameters()}
                if (
                    active["completed_updates"] != UPDATES_PER_MODEL
                    or total_steps != {n132.TOTAL_UPDATES + UPDATES_PER_MODEL}
                    or len(model._observations) != initial_history + UPDATES_PER_MODEL
                    or context_length_counts
                    != (
                        Counter({8: UPDATES_PER_MODEL})
                        if branch == "eight_row_control"
                        else Counter({length: UPDATES_PER_MODEL // 3 for length in builder.LENGTHS})
                    )
                ):
                    raise ValueError(
                        f"N1-150 fixed updates/history/length distribution differ: seed{seed}/{branch}"
                    )

                parent_path = Path(zero_payload["parent_checkpoint_path"]).resolve()
                final_path = CHECKPOINT_ROOT / f"s{seed}_{branch}_final.pt"
                final_payload = _save_n150_checkpoint(
                    model,
                    optimizer,
                    final_path,
                    seed=seed,
                    branch=branch,
                    parent_path=parent_path,
                    parent_hash=zero_payload["parent_checkpoint_sha256"],
                    additional_updates=UPDATES_PER_MODEL,
                )
                final_restore = _independent_restore(final_path, budget)
                gate_fingerprint = _fingerprint(model, optimizer)
                gate_rows, gate_summary, sufficient, gate_queries = n132._evaluate_train_only(
                    g, model, optimizer, seed, inputs, budget
                )
                if (
                    gate_queries != ORIGINAL_GATE_QUERIES_PER_ENDPOINT
                    or len(gate_rows) != ORIGINAL_GATE_QUERIES_PER_ENDPOINT
                ):
                    raise ValueError(
                        f"original N1-132 gate query denominator changed: seed{seed}/{branch}"
                    )
                for row in gate_rows:
                    durable = {
                        "event": "original_train_gate_row",
                        "seed": seed,
                        "branch": branch,
                        **row,
                    }
                    append_jsonl(PROGRESS, durable)
                train_score_queries += gate_queries
                length_rows, length_queries = _score_length_descriptors(
                    model,
                    optimizer,
                    seed,
                    branch,
                    inputs,
                    schedule["views"],
                    budget,
                    PROGRESS,
                )
                if (
                    length_queries != LENGTH_SCORE_QUERIES_PER_ENDPOINT
                    or len(length_rows) != LENGTH_SCORE_QUERIES_PER_ENDPOINT
                ):
                    raise ValueError(
                        f"N1-150 length score denominator changed: seed{seed}/{branch}"
                    )
                train_score_queries += length_queries
                if _fingerprint(model, optimizer) != gate_fingerprint:
                    raise ValueError(
                        f"train-only scoring changed frozen final bundle: seed{seed}/{branch}"
                    )
                loss_means = {
                    name: sum(float(row[name]) for row in step_losses) / len(step_losses)
                    for name in ("total", "base_total", "qa", "observation", "binding")
                }
                records.append(
                    {
                        "seed": seed,
                        "branch": branch,
                        "parent_checkpoint_path": str(parent_path),
                        "parent_checkpoint_sha256": zero_payload["parent_checkpoint_sha256"],
                        "zero_checkpoint_path": zero["path"],
                        "zero_checkpoint_sha256": zero["sha256"],
                        "zero_bundle_fingerprint": zero_payload["bundle_fingerprint"],
                        "additional_updates": UPDATES_PER_MODEL,
                        "total_updates": n132.TOTAL_UPDATES + UPDATES_PER_MODEL,
                        "initial_history_count": initial_history,
                        "final_history_count": len(model._observations),
                        "optimizer_step_values": sorted(total_steps),
                        "context_length_update_counts": dict(sorted(context_length_counts.items())),
                        "loss_means": loss_means,
                        "original_train_gate": gate_summary,
                        "training_sufficient": sufficient,
                        "original_train_gate_queries": gate_queries,
                        "length_score_queries": length_queries,
                        "train_only_length_summary": _length_score_summary(length_rows),
                        "final_checkpoint_path": str(final_path.resolve()),
                        "final_checkpoint_sha256": sha256(final_path),
                        "final_bundle_fingerprint": final_payload["bundle_fingerprint"],
                        "final_independent_restore": final_restore,
                    }
                )
                del model, optimizer
                model = optimizer = None
                active = None
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        progress.flush()
        progress.close()

    partial_checkpoint = None
    if active is not None and model is not None and optimizer is not None:
        try:
            partial_path = CHECKPOINT_ROOT / f"s{active['seed']}_{active['branch']}_partial.pt"
            payload = _save_n150_checkpoint(
                model,
                optimizer,
                partial_path,
                seed=active["seed"],
                branch=active["branch"],
                parent_path=Path(active["parent_checkpoint_path"]),
                parent_hash=active["parent_checkpoint_sha256"],
                additional_updates=active["completed_updates"],
            )
            partial_checkpoint = {
                "path": str(partial_path.resolve()),
                "sha256": sha256(partial_path),
                "additional_updates": active["completed_updates"],
                "bundle_fingerprint": payload["bundle_fingerprint"],
                "automatic_resume_allowed": False,
            }
        except BaseException as exc:
            errors.append(f"partial checkpoint save: {type(exc).__name__}: {exc}")

    input_hashes = dict(preflight["source_hashes"])
    input_hashes[str(PREFLIGHT.resolve())] = sha256(PREFLIGHT)
    input_hashes[str(PREFLIGHT_BINDING.resolve())] = sha256(PREFLIGHT_BINDING)
    input_hashes[str(ATTEMPT.resolve())] = sha256(ATTEMPT)
    source_unchanged = builder.verify_hashes(input_hashes)
    if not source_unchanged:
        errors.append("N1-150 training source/zero checkpoint hash changed")
    new_checkpoints = {
        str(path.resolve()): sha256(path) for path in CHECKPOINT_ROOT.glob("*.pt") if path.is_file()
    }
    stage_cpu = time.process_time() - start + budget.children
    cumulative = float(preflight["cumulative_cpu_seconds"]) + stage_cpu
    complete = (
        not errors
        and update_count == EXPECTED_TOTAL_UPDATES
        and train_score_queries == EXPECTED_TOTAL_TRAIN_SCORE_QUERIES
        and len(records) == 6
        and sum(path.name.endswith("_final.pt") for path in CHECKPOINT_ROOT.glob("*.pt")) == 6
        and source_unchanged
        and cumulative <= CPU_LIMIT_SECONDS
    )
    sufficient_all = len(records) == 6 and all(row["training_sufficient"] for row in records)
    report = {
        "schema": "n1-plan150-v27-length-robust-training-result-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "scoring_addendum_sha256": SCORING_ADDENDUM_SHA256,
        "resource_addendum_sha256": RESOURCE_ADDENDUM_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_binding_sha256": sha256(SOURCE_BINDING),
        "preflight_sha256": sha256(PREFLIGHT),
        "preflight_binding_sha256": sha256(PREFLIGHT_BINDING),
        "attempt_sha256": sha256(ATTEMPT),
        "progress_sha256": sha256(PROGRESS),
        "source_hashes": input_hashes,
        "source_hashes_unchanged": source_unchanged,
        "new_checkpoint_hashes": new_checkpoints,
        "models": records,
        "partial_checkpoint": partial_checkpoint,
        "training_updates": update_count,
        "parameter_updates": update_count,
        "optimizer_steps": update_count,
        "train_score_queries": train_score_queries,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "training_sufficient_all_endpoints": sufficient_all,
        "cpu_seconds": round(stage_cpu, 6),
        "prior_cpu_seconds": float(preflight["cumulative_cpu_seconds"]),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "can_promote": False,
        "errors": errors,
        "summary_zh": f"N1-150长度混合训练{('完整' if complete else '停止')}：{update_count}/55,296次更新、{train_score_queries}/{EXPECTED_TOTAL_TRAIN_SCORE_QUERIES}条train-only评分；训练门全过={sufficient_all}，0 dev/holdout/confirmation，can_promote=false。",
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-150 matched length-robust training")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--checkpoint-preflight", action="store_true")
    modes.add_argument("--execute-approved", action="store_true")
    modes.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker is not None:
        result = _restore_worker(args.restore_worker)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("complete") is True and result.get("equal") is True else 1
    if args.validate_only:
        result = builder.source_readiness()
    elif args.checkpoint_preflight:
        result = checkpoint_preflight()
    else:
        result = run_training()
    keys = (
        "status",
        "complete",
        "cpu_seconds",
        "cumulative_cpu_seconds",
        "parent_count",
        "independent_restores",
        "training_updates",
        "train_score_queries",
        "training_sufficient_all_endpoints",
        "dev_queries",
        "holdout_queries",
        "confirmation_queries",
        "can_promote",
        "errors",
        "summary_zh",
    )
    print(json.dumps({key: result.get(key) for key in keys}, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("complete") is True else 1


def _verify_preflight() -> tuple[dict, dict, dict, dict, dict]:
    if (
        not READINESS.is_file()
        or not SOURCE_BINDING.is_file()
        or not PREFLIGHT.is_file()
        or not PREFLIGHT_BINDING.is_file()
    ):
        raise FileNotFoundError(
            "N1-150 readiness and six-copy zero checkpoint preflight are required"
        )
    readiness = read_json(READINESS)
    readiness_binding = read_json(SOURCE_BINDING)
    preflight = read_json(PREFLIGHT)
    preflight_binding = read_json(PREFLIGHT_BINDING)
    if (
        readiness.get("status") != "ready_for_zero_checkpoint_preflight"
        or readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("scoring_addendum_sha256") != SCORING_ADDENDUM_SHA256
        or readiness.get("resource_addendum_sha256") != RESOURCE_ADDENDUM_SHA256
        or readiness_binding.get("status") != "complete"
        or readiness_binding.get("readiness_sha256") != sha256(READINESS)
        or readiness_binding.get("source_hashes") != readiness.get("source_hashes")
        or readiness_binding.get("schedule_sha256") != sha256(SCHEDULE)
        or not seal_ast.receipt_binding_matches(
            readiness_binding.get("source_exclusion_sha256"), SOURCE_EXCLUSIONS
        )
        or preflight.get("status") != "complete"
        or preflight.get("complete") is not True
        or preflight.get("plan_sha256") != PLAN_SHA256
        or preflight.get("scoring_addendum_sha256") != SCORING_ADDENDUM_SHA256
        or preflight.get("resource_addendum_sha256") != RESOURCE_ADDENDUM_SHA256
        or preflight.get("readiness_sha256") != sha256(READINESS)
        or preflight.get("zero_checkpoint_count") != 6
        or preflight.get("independent_restore_count") != 6
        or preflight.get("source_hashes_unchanged") is not True
        or preflight.get("parameter_updates") != 0
        or preflight.get("optimizer_steps") != 0
        or preflight.get("model_queries") != 0
        or preflight_binding.get("status") != "complete"
        or preflight_binding.get("preflight_sha256") != sha256(PREFLIGHT)
        or preflight_binding.get("plan_sha256") != PLAN_SHA256
        or preflight_binding.get("source_hashes") != preflight.get("source_hashes")
        or preflight_binding.get("resource_addendum_sha256") != RESOURCE_ADDENDUM_SHA256
    ):
        raise ValueError("N1-150 readiness/checkpoint preflight is incomplete or stale")
    if not builder.verify_hashes(readiness.get("source_hashes", {})):
        raise ValueError("N1-150 readiness source hashes changed")
    if not builder.verify_hashes(preflight.get("source_hashes", {})):
        raise ValueError("N1-150 zero bundle/source hashes changed")
    bound, schedule = _runtime()
    scheduled_manifest = read_json(SCHEDULE)
    recorded_exclusion_sha256 = scheduled_manifest.get("source_exclusion_sha256")
    expected = dict(
        schedule["manifest"],
        source_exclusion_sha256=recorded_exclusion_sha256,
        resource_addendum_sha256=RESOURCE_ADDENDUM_SHA256,
    )
    if (
        expected != scheduled_manifest
        or not isinstance(recorded_exclusion_sha256, str)
        or not seal_ast.recorded_value_attested(SOURCE_EXCLUSIONS, recorded_exclusion_sha256)
        or expected["schedule_sha256"] != readiness.get("schedule_identity_sha256")
    ):
        raise ValueError("N1-150 deterministic length schedule changed after readiness")
    return readiness, preflight, preflight_binding, bound, schedule


if __name__ == "__main__":
    raise SystemExit(main())
