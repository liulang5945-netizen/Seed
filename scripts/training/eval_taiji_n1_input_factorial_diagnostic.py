"""One-shot, stateless PLAN-N1-105 diagnostic evaluation; never rejudges gates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import (  # noqa: E402
    build_taiji_n1_input_factorial_diagnostic as face_builder,
)
from scripts.training import (  # noqa: E402
    eval_taiji_n1_hard_neighbor_v9 as v9_evaluator,
)
from scripts.training import (  # noqa: E402
    eval_taiji_n1_transition_factorial_confirmation as base_evaluator,
)
from scripts.training import run_taiji_n1_hard_neighbor_training as training  # noqa: E402
from scripts.training import seal_ast  # noqa: E402

PLAN = face_builder.PLAN
FACE = face_builder.FACE
CHECKS = face_builder.CHECKS
READINESS = ROOT / "reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_readiness_20261017.json"
REPAIR_RECEIPT = (
    ROOT / "reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_preflight_repair_20261017.json"
)
ATTEMPT = ROOT / "reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_attempt_20261017.json"
REPORT = ROOT / "reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_20261017.json"

SEEDS = training.SEEDS
ARMS = training.ARMS
KINDS = face_builder.KINDS
ANSWER_LENGTHS = face_builder.ANSWER_LENGTHS
OBSERVATIONS = face_builder.OBSERVATIONS
EXPECTED_ITEMS = sum(face_builder.EXPECTED_ITEMS.values())
EXPECTED_QUERIES = EXPECTED_ITEMS * len(SEEDS) * len(ARMS)
CPU_LIMIT_SECONDS = 1200.0
PLAN_SHA256 = face_builder.PLAN_SHA256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def _absent_confirmation_artifacts() -> bool:
    return all(
        not path.exists()
        for path in (
            v9_evaluator.face_builder.V7_CONFIRM,
            v9_evaluator.face_builder.V7_CONFIRM_ATTEMPT,
            v9_evaluator.face_builder.V8_CONFIRM,
            v9_evaluator.face_builder.V8_CONFIRM_ATTEMPT,
        )
    )


def _verify_preflight_repair(
    construction_guards: dict[str, str],
    current_guards: dict[str, str],
    *,
    face_sha256: str,
    checks_sha256: str,
) -> dict | None:
    changed = sorted(
        path
        for path in set(construction_guards) | set(current_guards)
        if construction_guards.get(path) != current_guards.get(path)
    )
    evaluator_path = str(Path(__file__).resolve())
    test_path = str(face_builder.CONTRACT_TEST.resolve())
    expected_changes = sorted((evaluator_path, test_path))
    if not changed and not REPAIR_RECEIPT.exists():
        return None
    if not REPAIR_RECEIPT.is_file():
        raise ValueError(
            "construction-time source hashes changed without a preflight repair receipt"
        )
    receipt = read_json(REPAIR_RECEIPT)
    if (
        receipt.get("schema") != "n1-hard-neighbor-v10-factorial-preflight-repair-v1"
        or receipt.get("status") != "implemented_preflight_assertion_repair"
        or receipt.get("plan_sha256") != PLAN_SHA256
        or receipt.get("face_sha256") != face_sha256
        or receipt.get("checks_sha256") != checks_sha256
        or receipt.get("construction_source_guards") != construction_guards
        or receipt.get("repaired_source_guards") != current_guards
        or receipt.get("changed_paths") != expected_changes
        or changed != expected_changes
        or receipt.get("failed_preflight_checkpoint_loads") != 1
        or receipt.get("failed_preflight_readiness_written") is not False
        or receipt.get("failed_preflight_attempt_written") is not False
        or receipt.get("diagnostic_model_queries") != 0
        or receipt.get("parameter_updates") != 0
        or receipt.get("optimizer_steps") != 0
        or receipt.get("failed_preflight_bundle_restore_passed") is not True
        or receipt.get("failure_was_empty_history_assertion") is not True
        or type(receipt.get("failed_preflight_cpu_upper_bound_seconds")) not in (int, float)
        or not 0 <= receipt.get("failed_preflight_cpu_upper_bound_seconds", -1) <= 10
    ):
        raise ValueError(
            "preflight repair receipt does not authorize the exact evaluator/test-only change"
        )
    return receipt


def validate_face_and_sources() -> tuple[dict, dict, dict, dict[str, str], dict | None]:
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-105 bytes differ from its frozen SHA")
    if not FACE.is_file() or not CHECKS.is_file():
        raise FileNotFoundError("the frozen diagnostic face and construction checks are required")
    face = read_json(FACE)
    checks = read_json(CHECKS)
    if (
        checks.get("status") != "complete"
        or checks.get("complete") is not True
        or checks.get("accepted") is not True
        or checks.get("model_queries") != 0
        or checks.get("parameter_updates") != 0
        or checks.get("optimizer_steps") != 0
        or checks.get("plan_sha256") != PLAN_SHA256
        or checks.get("face_file_sha256") != sha256(FACE)
        or not seal_ast.receipt_binding_matches(checks.get("face_sha256"), FACE)
        or face.get("schema") != face_builder.SCHEMA
        or face.get("split") != "diagnostic"
        or len(face.get("episodes", [])) != EXPECTED_ITEMS
        or face.get("items") != EXPECTED_ITEMS
        or face.get("construction", {}).get("plan_sha256") != PLAN_SHA256
        or face.get("construction", {}).get("builder_sha256") != checks.get("builder_sha256")
        or face.get("construction", {}).get("source_guards") != checks.get("source_guards")
    ):
        raise ValueError("diagnostic face/check receipt differs from the frozen contract")
    exclusions, excluded_words, provenance, current_guards = face_builder.collect_exclusions()
    if (
        provenance != face.get("exclusion_sources")
        or provenance != checks.get("exclusion_sources")
        or provenance.get("v7_v8_confirmation_payload_fields_decoded") is not False
        or provenance.get("v7_v8_confirmation_artifacts_absent") is not True
        or not _absent_confirmation_artifacts()
    ):
        raise ValueError("exclusion sources changed or a prohibited confirmation artifact appeared")
    repair = _verify_preflight_repair(
        checks["source_guards"],
        current_guards,
        face_sha256=sha256(FACE),
        checks_sha256=sha256(CHECKS),
    )
    audit = face_builder.audit_face(face, excluded_names=exclusions, excluded_words=excluded_words)
    if not audit.get("accepted"):
        raise ValueError(
            f"diagnostic face no longer satisfies construction contract: {audit['errors']}"
        )
    return face, checks, audit, current_guards, repair


def _evaluation_source_guards(base_guards: dict[str, str]) -> dict[str, str]:
    guards = dict(base_guards)
    paths = [
        PLAN,
        FACE,
        CHECKS,
        Path(__file__).resolve(),
        Path(face_builder.__file__).resolve(),
        Path(v9_evaluator.__file__).resolve(),
        Path(base_evaluator.__file__).resolve(),
        Path(base_evaluator.train.__file__).resolve(),
        Path(base_evaluator.runner.__file__).resolve(),
        Path(base_evaluator.train.engineering.__file__).resolve(),
        Path(base_evaluator.face.__file__).resolve(),
        Path(base_evaluator.paired.__file__).resolve(),
        Path(ROOT / "taiji/native_evidence_candidate.py"),
        Path(ROOT / "taiji/fabric.py"),
        Path(ROOT / "taiji/contracts.py"),
        Path(training.REPORT),
        *training.source_paths(),
    ]
    if REPAIR_RECEIPT.is_file():
        paths.append(REPAIR_RECEIPT)
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"evaluation source is missing: {path}")
        guards[str(Path(path).resolve())] = sha256(Path(path))
    return guards


def _sources_unchanged(guards: dict[str, str]) -> bool:
    return all(
        Path(path).is_file() and sha256(Path(path)) == digest for path, digest in guards.items()
    )


def _runtime() -> tuple[dict, dict]:
    torch.set_num_threads(1)
    loaded, g, inputs = training.make_runner(load_inputs=False)
    if inputs is not None:
        raise ValueError("diagnostic runner unexpectedly loaded training inputs")
    return loaded, g


def _restore_endpoints(
    g: dict, paths: dict[tuple[int, str], Path], items: list[dict]
) -> list[dict]:
    restores = []
    expected_endpoints = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(paths) != expected_endpoints:
        raise ValueError("exactly the nine frozen PLAN-N1-100 endpoints are required")
    for seed in SEEDS:
        for arm in ARMS:
            path = paths[seed, arm]
            model, optimizer = v9_evaluator._new_model(g, path, seed, arm)
            if not isinstance(model._observations, dict):
                raise ValueError(f"endpoint {seed}/{arm} restored an invalid observation history")
            diagnostic_sessions = {f"n1-v10-diagnostic-{seed}-{arm}-{item['id']}" for item in items}
            if diagnostic_sessions & set(model._observations):
                raise ValueError(f"endpoint {seed}/{arm} already contains a diagnostic session")
            bundle_fingerprint = g["paired"].fingerprint(model.bundle(optimizer))
            history_keys = sorted(str(key) for key in model._observations)
            restores.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "checkpoint_sha256": sha256(path),
                    "full_bundle_fingerprint": bundle_fingerprint,
                    "full_bundle_equal": True,
                    "initial_history_entries": len(model._observations),
                    "initial_history_keys_sha256": hashlib.sha256(
                        "\n".join(history_keys).encode("utf-8")
                    ).hexdigest(),
                    "diagnostic_session_collisions": 0,
                }
            )
            del model, optimizer
            gc.collect()
    return restores


def run_readiness() -> dict:
    if READINESS.exists() or ATTEMPT.exists() or REPORT.exists():
        raise FileExistsError("PLAN-N1-105 readiness and evaluation artifacts are write-once")
    started = time.process_time()
    face, checks, face_audit, source_guards, repair = validate_face_and_sources()
    source_identity = _evaluation_source_guards(source_guards)
    training_report, checkpoint_paths, checkpoint_guards = v9_evaluator.training_endpoints()
    if repair is not None and repair.get("checkpoint_guards_after_failure") != checkpoint_guards:
        raise ValueError("checkpoint bytes changed after the failed query-free preflight")
    loaded, g = _runtime()
    runtime_identity = g["identity"]()
    restores = _restore_endpoints(g, checkpoint_paths, face["episodes"])
    code_identity_unchanged = g["identity"]() == runtime_identity
    sources_unchanged = _sources_unchanged(source_identity)
    checkpoints_unchanged = _sources_unchanged(checkpoint_guards)
    cpu_seconds = time.process_time() - started
    build_cpu_seconds = checks.get("cpu_seconds")
    repair_cpu_upper_bound = (
        float(repair.get("failed_preflight_cpu_upper_bound_seconds", 0.0))
        if repair is not None
        else 0.0
    )
    complete = (
        len(restores) == 9
        and code_identity_unchanged
        and sources_unchanged
        and checkpoints_unchanged
        and type(build_cpu_seconds) in (int, float)
        and build_cpu_seconds + repair_cpu_upper_bound + cpu_seconds <= CPU_LIMIT_SECONDS
    )
    result = {
        "schema": "n1-hard-neighbor-v10-factorial-diagnostic-readiness-v1",
        "status": "complete" if complete else "stopped_preflight",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "checks_sha256": sha256(CHECKS),
        "preflight_repair_receipt_sha256": sha256(REPAIR_RECEIPT) if repair else None,
        "training_report_sha256": sha256(training.REPORT),
        "training_report_complete": training_report.get("complete") is True,
        "training_sufficient": training_report.get("training_sufficient") is True,
        "checkpoint_guards": checkpoint_guards,
        "source_guards": source_identity,
        "runtime_identity": runtime_identity,
        "restore_checks": restores,
        "prior_failed_preflight_checkpoint_loads": 1 if repair else 0,
        "prior_failed_preflight_model_queries": 0,
        "construction_audit": face_audit,
        "items": len(face["episodes"]),
        "seed_order": list(SEEDS),
        "arm_order": list(ARMS),
        "planned_model_queries": EXPECTED_QUERIES,
        "model_loads": len(restores),
        "checkpoint_loads": len(restores),
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "sources_unchanged": sources_unchanged,
        "checkpoints_unchanged": checkpoints_unchanged,
        "code_identity_unchanged": code_identity_unchanged,
        "build_cpu_seconds": round(float(build_cpu_seconds), 6),
        "prior_failed_preflight_cpu_upper_bound_seconds": round(repair_cpu_upper_bound, 6),
        "preflight_cpu_seconds": round(cpu_seconds, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "total_cpu_before_query_seconds": round(
            float(build_cpu_seconds) + repair_cpu_upper_bound + cpu_seconds, 6
        ),
        "can_promote": False,
    }
    write_new(READINESS, result)
    return result


@torch.no_grad()
def measure_diagnostic(model, item: dict, arm: str, session: str) -> dict:
    if item.get("split") != "diagnostic" or arm not in ARMS:
        raise ValueError("diagnostic split and frozen arm are required")
    data = base_evaluator.paired.project(item)
    base_evaluator.paired.observe(model, data, session)
    tape = base_evaluator.paired.evidence(model, session, arm)
    target = item["answer"].encode("utf-8")
    labels = torch.tensor([*target, 256])
    logits, _ = model.teacher_logits(tape, session, data.question, target)
    predicted = logits.argmax(dim=-1)
    trace = model.generate_trace(tape, session, data.question, 64)
    return {
        "id": item["id"],
        "kind": item["kind"],
        "template_index": item["template_index"],
        "split": "diagnostic",
        "target_row_position_1based": item["target_row_position_1based"],
        "material_answer_length_chars": item["material_answer_length_chars"],
        "teacher_exact": bool(torch.equal(predicted, labels)),
        "teacher_token_correct": int((predicted == labels).sum()),
        "teacher_token_count": len(labels),
        "teacher_cross_entropy": float(torch.nn.functional.cross_entropy(logits, labels)),
        "output_hex": trace.output.hex(),
        "raw_hex": trace.raw.hex(),
        "stop_reason": trace.stop_reason,
        "generated_steps": trace.generated_steps,
        "free_exact": trace.raw == target and trace.stop_reason == "boundary",
    }


def score_endpoint(
    model,
    optimizer,
    seed: int,
    arm: str,
    items: list[dict],
    *,
    fingerprint,
    budget_guard,
    accounting: dict,
) -> list[dict]:
    before = fingerprint(model.bundle(optimizer))
    rows = []
    try:
        for item in items:
            budget_guard()
            session = f"n1-v10-diagnostic-{seed}-{arm}-{item['id']}"
            if session in model._observations:
                raise ValueError("diagnostic session collides with frozen model history")
            accounting["model_queries"] += 1
            try:
                scored = measure_diagnostic(model, item, arm, session)
            finally:
                model._observations.pop(session, None)
            if scored.get("id") != item["id"] or scored.get("kind") != item["kind"]:
                raise ValueError("diagnostic scorer returned a row for the wrong input")
            rows.append(dict(scored, seed=seed, arm=arm))
    finally:
        if fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("diagnostic changed model, optimizer, RNG, or history bundle")
    return rows


def _summarize(rows: list[dict], items: list[dict]) -> dict:
    item_by_id = {item["id"]: item for item in items}
    expected_ids = set(item_by_id)
    by_endpoint: dict[tuple[int, str], list[dict]] = defaultdict(list)
    seen = set()
    for row in rows:
        seed, arm, item_id = row.get("seed"), row.get("arm"), row.get("id")
        key = (seed, arm, item_id)
        if seed not in SEEDS or arm not in ARMS or item_id not in item_by_id or key in seen:
            raise ValueError("diagnostic rows contain an invalid or duplicate endpoint/item key")
        if type(row.get("free_exact")) is not bool:
            raise ValueError("diagnostic free_exact must be boolean")
        item = item_by_id[item_id]
        if row.get("kind") != item["kind"]:
            raise ValueError("diagnostic result kind differs from manifest")
        seen.add(key)
        by_endpoint[(seed, arm)].append(row)
    if len(rows) != EXPECTED_QUERIES or any(
        {row["id"] for row in by_endpoint[(seed, arm)]} != expected_ids
        for seed in SEEDS
        for arm in ARMS
    ):
        raise ValueError(
            "each frozen endpoint must have exactly one result for every diagnostic item"
        )

    per_endpoint = []
    for seed in SEEDS:
        for arm in ARMS:
            selected = by_endpoint[(seed, arm)]
            per_kind = {
                kind: {
                    "denominator": sum(row["kind"] == kind for row in selected),
                    "free_exact_correct": sum(
                        row["free_exact"] for row in selected if row["kind"] == kind
                    ),
                }
                for kind in KINDS
            }
            per_endpoint.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "denominator": len(selected),
                    "free_exact_correct": sum(row["free_exact"] for row in selected),
                    "per_kind": per_kind,
                }
            )

    cell_rows: dict[tuple[str, int | None, int | None], list[dict]] = defaultdict(list)
    matched_rows: dict[tuple[int, str, str, int, int | None], dict[int, dict]] = defaultdict(dict)
    position_rows: dict[tuple[int, str, str, int, int | None], list[dict]] = defaultdict(list)
    for row in rows:
        item = item_by_id[row["id"]]
        kind = item["kind"]
        position = item["target_row_position_1based"]
        length = item["material_answer_length_chars"]
        cell_rows[(kind, position, length)].append(row)
        if kind in {"lookup", "revision"}:
            matched_key = (row["seed"], row["arm"], kind, item["template_index"], position)
            matched_rows[matched_key][length] = row
        if kind in {"lookup", "revision", "transition"}:
            profile_key = (row["seed"], row["arm"], kind, item["template_index"], length)
            position_rows[profile_key].append(
                {"position": position, "free_exact": row["free_exact"], "denominator": 1}
            )

    feature_cells = []
    for (kind, position, length), selected in sorted(
        cell_rows.items(),
        key=lambda entry: (
            KINDS.index(entry[0][0]),
            entry[0][1] is None,
            entry[0][1] or 0,
            entry[0][2] is None,
            entry[0][2] or 0,
        ),
    ):
        per_seed_arm = {}
        for seed in SEEDS:
            per_seed_arm[str(seed)] = {}
            for arm in ARMS:
                endpoint_rows = [
                    row for row in selected if row["seed"] == seed and row["arm"] == arm
                ]
                per_seed_arm[str(seed)][arm] = {
                    "denominator": len(endpoint_rows),
                    "free_exact_correct": sum(row["free_exact"] for row in endpoint_rows),
                }
        feature_cells.append(
            {
                "kind": kind,
                "target_row_position_1based": position,
                "material_answer_length_chars": length,
                "denominator": len({row["id"] for row in selected}),
                "unique_items": len({row["id"] for row in selected}),
                "per_seed_arm": per_seed_arm,
            }
        )

    paired_length_contrasts = []
    for key, lengths in sorted(matched_rows.items()):
        if set(lengths) != set(ANSWER_LENGTHS):
            raise ValueError(f"matched template is missing an answer-length partner: {key}")
        six, seven = lengths[6]["free_exact"], lengths[7]["free_exact"]
        paired_length_contrasts.append(
            {
                "seed": key[0],
                "arm": key[1],
                "kind": key[2],
                "template_index": key[3],
                "target_row_position_1based": key[4],
                "length6_free_exact": six,
                "length7_free_exact": seven,
                "contrast": (
                    "both_correct"
                    if six and seven
                    else "length6_only" if six else "length7_only" if seven else "both_wrong"
                ),
                "denominator": 1,
            }
        )

    paired_position_profiles = []
    for key, values in sorted(position_rows.items()):
        kind, answer_length = key[2], key[4]
        expected_positions = list(
            range(1, OBSERVATIONS + 1) if kind == "lookup" else range(1, OBSERVATIONS)
        )
        values = sorted(values, key=lambda value: value["position"])
        if [value["position"] for value in values] != expected_positions:
            raise ValueError(f"matched template is missing a target-position partner: {key}")
        if kind in {"lookup", "revision"} and answer_length not in ANSWER_LENGTHS:
            raise ValueError(f"matched template has an invalid answer-length profile: {key}")
        if kind == "transition" and answer_length is not None:
            raise ValueError(f"transition profile has an inapplicable answer length: {key}")
        paired_position_profiles.append(
            {
                "seed": key[0],
                "arm": key[1],
                "kind": kind,
                "template_index": key[3],
                "material_answer_length_chars": answer_length,
                "denominator": len(expected_positions),
                "free_exact_by_position": values,
            }
        )

    if (
        len(feature_cells) != 58
        or len(paired_length_contrasts) != 9 * 4 * (12 + 11)
        or len(paired_position_profiles) != 9 * 4 * (2 * 2 + 1)
    ):
        raise ValueError(
            "diagnostic feature strata or matched-pair denominators differ from the frozen design"
        )
    return {
        "endpoint_count": len(per_endpoint),
        "items_per_endpoint": EXPECTED_ITEMS,
        "rows_analyzed": len(rows),
        "per_endpoint": per_endpoint,
        "feature_cells": feature_cells,
        "paired_length_contrasts": paired_length_contrasts,
        "paired_position_profiles": paired_position_profiles,
        "interpretation": "descriptive diagnostic contrasts only; no threshold or capability gate computed",
    }


def run_execute() -> dict:
    if ATTEMPT.exists() or REPORT.exists():
        raise FileExistsError("PLAN-N1-105 diagnostic evaluation is one-shot and cannot be retried")
    started = time.process_time()
    if not READINESS.is_file():
        raise FileNotFoundError(
            "successful full-bundle readiness is required before diagnostic queries"
        )
    readiness = read_json(READINESS)
    if (
        readiness.get("status") != "complete"
        or readiness.get("complete") is not True
        or readiness.get("model_queries") != 0
        or readiness.get("restore_checks") is None
        or len(readiness["restore_checks"]) != 9
        or readiness.get("plan_sha256") != PLAN_SHA256
    ):
        raise ValueError("diagnostic readiness receipt is incomplete or not query-free")

    face, checks, audit, source_guards, repair = validate_face_and_sources()
    _, checkpoint_paths, checkpoint_guards = v9_evaluator.training_endpoints()
    repair_sha256 = sha256(REPAIR_RECEIPT) if repair is not None else None
    if (
        not seal_ast.receipt_binding_matches(readiness.get("face_sha256"), FACE)
        or readiness.get("checks_sha256") != sha256(CHECKS)
        or readiness.get("preflight_repair_receipt_sha256") != repair_sha256
        or readiness.get("checkpoint_guards") != checkpoint_guards
        or readiness.get("source_guards") != _evaluation_source_guards(source_guards)
    ):
        raise ValueError(
            "readiness bindings differ from current face, code, or checkpoint endpoints"
        )

    build_cpu_seconds = checks["cpu_seconds"]
    readiness_cpu_seconds = readiness["preflight_cpu_seconds"]
    repair_cpu_upper_bound = (
        float(repair.get("failed_preflight_cpu_upper_bound_seconds", 0.0))
        if repair is not None
        else 0.0
    )
    if (
        build_cpu_seconds
        + repair_cpu_upper_bound
        + readiness_cpu_seconds
        + (time.process_time() - started)
        >= CPU_LIMIT_SECONDS
    ):
        raise TimeoutError("construction and recovery preflight consumed the full CPU ceiling")

    _, g = _runtime()
    runtime_identity = g["identity"]()
    rows: list[dict] = []
    accounting = {"model_queries": 0}
    loaded_models = 0
    source_identity = _evaluation_source_guards(source_guards)
    attempt = {
        "schema": "n1-hard-neighbor-v10-factorial-diagnostic-attempt-v1",
        "status": "started",
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "checks_sha256": sha256(CHECKS),
        "readiness_sha256": sha256(READINESS),
        "preflight_repair_receipt_sha256": repair_sha256,
        "source_guards": source_identity,
        "checkpoint_guards": checkpoint_guards,
        "seed_order": list(SEEDS),
        "arm_order": list(ARMS),
        "item_count": EXPECTED_ITEMS,
        "planned_model_queries": EXPECTED_QUERIES,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "face_consumed": True,
        "can_promote": False,
    }
    remaining_before_attempt = (
        CPU_LIMIT_SECONDS
        - build_cpu_seconds
        - repair_cpu_upper_bound
        - readiness_cpu_seconds
        - (time.process_time() - started)
    )
    if remaining_before_attempt <= 2.0:
        raise TimeoutError("insufficient CPU reserve before the one-shot diagnostic attempt")
    write_new(ATTEMPT, attempt)
    error = None
    status = "complete"
    sources_unchanged = False
    code_identity_unchanged = False

    def guard_query() -> None:
        remaining = (
            CPU_LIMIT_SECONDS
            - build_cpu_seconds
            - repair_cpu_upper_bound
            - readiness_cpu_seconds
            - (time.process_time() - started)
        )
        if remaining <= 0.5:
            raise TimeoutError("CPU ceiling reached before the next diagnostic item")

    try:
        for seed in SEEDS:
            for arm in ARMS:
                remaining = (
                    CPU_LIMIT_SECONDS
                    - build_cpu_seconds
                    - repair_cpu_upper_bound
                    - readiness_cpu_seconds
                    - (time.process_time() - started)
                )
                if remaining <= 2.0:
                    raise TimeoutError("CPU ceiling reached before the next endpoint")
                model, optimizer = v9_evaluator._new_model(
                    g, checkpoint_paths[seed, arm], seed, arm
                )
                loaded_models += 1
                endpoint_rows = score_endpoint(
                    model,
                    optimizer,
                    seed,
                    arm,
                    face["episodes"],
                    fingerprint=g["paired"].fingerprint,
                    budget_guard=guard_query,
                    accounting=accounting,
                )
                rows.extend(endpoint_rows)
                del model, optimizer
                gc.collect()
    except KeyboardInterrupt:
        status = "interrupted"
        error = "face consumed; automatic retry prohibited"
    except BaseException as exc:
        status = "stopped"
        error = f"{type(exc).__name__}: {exc}"

    cpu_seconds = time.process_time() - started
    try:
        sources_unchanged = _sources_unchanged(source_identity)
        checkpoints_unchanged = _sources_unchanged(checkpoint_guards)
        code_identity_unchanged = g["identity"]() == runtime_identity
    except BaseException as exc:
        sources_unchanged = False
        checkpoints_unchanged = False
        code_identity_unchanged = False
        error = error or f"postrun_guard_error:{type(exc).__name__}: {exc}"
        status = "stopped"
    rows_valid = False
    summary = None
    try:
        summary = _summarize(rows, face["episodes"])
        rows_valid = True
    except BaseException as exc:
        error = error or f"summary_error:{type(exc).__name__}: {exc}"
        status = "stopped"
    total_cpu_seconds = (
        build_cpu_seconds + repair_cpu_upper_bound + readiness_cpu_seconds + cpu_seconds
    )
    complete = (
        status == "complete"
        and rows_valid
        and accounting["model_queries"] == EXPECTED_QUERIES
        and len(rows) == EXPECTED_QUERIES
        and loaded_models == 9
        and sources_unchanged
        and checkpoints_unchanged
        and code_identity_unchanged
        and total_cpu_seconds <= CPU_LIMIT_SECONDS
    )
    result = {
        "schema": "n1-hard-neighbor-v10-factorial-diagnostic-result-v1",
        "status": "complete" if complete else status,
        "complete": complete,
        "exploratory_diagnostic_only": True,
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "checks_sha256": sha256(CHECKS),
        "readiness_sha256": sha256(READINESS),
        "attempt_sha256": sha256(ATTEMPT),
        "preflight_repair_receipt_sha256": repair_sha256,
        "source_guards": source_identity,
        "checkpoint_guards": checkpoint_guards,
        "seed_order": list(SEEDS),
        "arm_order": list(ARMS),
        "items": EXPECTED_ITEMS,
        "model_queries": accounting["model_queries"],
        "planned_model_queries": EXPECTED_QUERIES,
        "model_loads_during_evaluation": loaded_models,
        "checkpoint_loads_during_evaluation": loaded_models,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "source_updates": 0,
        "existing_confirmation_face_queries": 0,
        "dev_queries": 0,
        "v7_v8_confirmation_payload_fields_decoded": False,
        "sources_unchanged": sources_unchanged,
        "checkpoints_unchanged": checkpoints_unchanged,
        "code_identity_unchanged": code_identity_unchanged,
        "can_promote": False,
        "gate_recomputed": False,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "build_cpu_seconds": round(build_cpu_seconds, 6),
        "prior_failed_preflight_cpu_upper_bound_seconds": round(repair_cpu_upper_bound, 6),
        "readiness_cpu_seconds": round(readiness_cpu_seconds, 6),
        "evaluation_cpu_seconds": round(cpu_seconds, 6),
        "total_cpu_seconds": round(total_cpu_seconds, 6),
        "summary": summary,
        "rows": rows,
        "error": error,
        "interpretation_limit": (
            "matched synthetic diagnostic outcomes only; no causal claim beyond the templates, "
            "no frozen gate or capability verdict, and no promotion"
        ),
    }
    write_new(REPORT, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--readiness", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.readiness:
        result = run_readiness()
        output = {
            "status": result["status"],
            "complete": result["complete"],
            "report": str(READINESS),
            "restore_checks": len(result["restore_checks"]),
            "model_queries": result["model_queries"],
            "total_cpu_before_query_seconds": result["total_cpu_before_query_seconds"],
        }
    else:
        result = run_execute()
        output = {
            "status": result["status"],
            "complete": result["complete"],
            "report": str(REPORT),
            "report_sha256": sha256(REPORT),
            "model_queries": result["model_queries"],
            "total_cpu_seconds": result["total_cpu_seconds"],
            "can_promote": False,
        }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
