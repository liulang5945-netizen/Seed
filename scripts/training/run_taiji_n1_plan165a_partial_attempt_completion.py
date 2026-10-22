"""Finish frozen N1-165 endpoints after an externally stopped run."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_plan155_short_unknown_context_length_training as n155
from scripts.training import run_taiji_n1_plan162b_question_surface_matched_continuation as n162
from scripts.training import run_taiji_n1_plan165_transition_template_mid_dose as n165

PLAN = ROOT / "plans/reference/PLAN-N1-165A_partial_attempt_completion_20261020.md"
PLAN_SHA256 = "04f38f7711ac907869334e8ad0c785733cc615271a07cf3df32810bbb7f6ee27"
OLD_ATTEMPT = n165.ATTEMPT
OLD_PROGRESS = n165.PROGRESS
OLD_ZERO_PREFLIGHT = n165.ZERO_PREFLIGHT
OLD_RESULT = n165.RESULT
NEW_ATTEMPT = ROOT / "reports/taiji_n1_plan165a_partial_attempt_completion_attempt_20261020.json"
NEW_PROGRESS = ROOT / "reports/taiji_n1_plan165a_partial_attempt_completion_progress_20261020.jsonl"
NEW_RESULT = ROOT / "reports/taiji_n1_plan165a_partial_attempt_completion_result_20261020.json"
SEEDS = (11, 12, 13)
ARM = n165.schedule_builder.CANDIDATE_ARM
UPDATES_PER_ENDPOINT = n165.UPDATES_PER_MODEL
GATE_QUERIES = n165.GATE_QUERIES_PER_ENDPOINT
LENGTH_QUERIES = n165.LENGTH_QUERIES_PER_ENDPOINT
SCORES_PER_ENDPOINT = GATE_QUERIES + LENGTH_QUERIES
DISCARDED_SEED12_UPDATES = 7_564
PARTIAL_SEED12_UPDATE_OPERATIONS = 1
FIRST_ATTEMPT_PROGRESS_UPDATE_ROWS = 25_996
FIRST_ATTEMPT_UPDATE_OPERATIONS = (
    FIRST_ATTEMPT_PROGRESS_UPDATE_ROWS + PARTIAL_SEED12_UPDATE_OPERATIONS
)
FIRST_ATTEMPT_CPU_UPPER_SECONDS = 4_000.0
CPU_LIMIT_SECONDS = 7_000.0
LINKED_CPU_LIMIT_SECONDS = 50_000.0
PRIOR_LINKED_CPU_SECONDS = n165.PRIOR_LINKED_CPU_SECONDS + FIRST_ATTEMPT_CPU_UPPER_SECONDS
EXPECTED_CONTEXT_COUNTS = Counter({8: 6_912, 16: 5_760, 32: 5_760})


class Refuse(RuntimeError):
    """The preserved N1-165 attempt cannot be completed losslessly."""


def write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"N1-165A write-once output already exists: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise Refuse(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _collect_old_progress() -> (
    tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, str]]
):
    digest = n162.sha256(OLD_PROGRESS)
    if digest != "11c59e4348baec67f7c77ff79f8a3574789cb68bc945f8ed266e482652071abf":
        raise Refuse("interrupted N1-165 progress SHA differs from the frozen recovery plan")
    raw_lines = OLD_PROGRESS.read_bytes().splitlines(keepends=True)
    if len(raw_lines) != 27_245 or raw_lines[-1].endswith(b"\n"):
        raise Refuse("interrupted progress does not have the frozen incomplete terminal record")
    try:
        json.loads(raw_lines[-1])
    except json.JSONDecodeError:
        pass
    else:
        raise Refuse("interrupted progress terminal record is unexpectedly complete")
    partial_tail = raw_lines[-1]
    if (
        len(partial_tail) != 2_410
        or hashlib.sha256(partial_tail).hexdigest()
        != "0d94ef091aeb5ddd3b037c3a5be14576283868c13d90eb5b220c846ab527a110"
        or not partial_tail.endswith(b'"route"')
        or b'"event":"training_update"' not in partial_tail
    ):
        raise Refuse(
            "interrupted progress terminal fragment differs from the frozen partial update"
        )
    rows = []
    for line_number, line in enumerate(raw_lines[:-1], 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise Refuse(f"invalid complete JSONL row at {OLD_PROGRESS}:{line_number}") from exc
    counts = Counter(row.get("event") for row in rows)
    training = [row for row in rows if row.get("event") == "training_update"]
    gates = [row for row in rows if row.get("event") == "original_train_gate_row"]
    lengths = [row for row in rows if row.get("event") == "length_score_row"]
    by_seed: dict[str, list[dict[str, Any]]] = {"11": [], "12": [], "13": []}
    for row in training:
        by_seed[str(row["seed"])].append(row)
    if (
        counts["training_update"] != FIRST_ATTEMPT_PROGRESS_UPDATE_ROWS
        or len(by_seed["11"]) != UPDATES_PER_ENDPOINT
        or len(by_seed["12"]) != DISCARDED_SEED12_UPDATES
        or by_seed["13"]
        or len(gates) != GATE_QUERIES
        or len(lengths) != LENGTH_QUERIES
    ):
        raise Refuse(f"interrupted progress counts differ: {dict(counts)}")
    if [row.get("step") for row in by_seed["11"]] != list(range(UPDATES_PER_ENDPOINT)):
        raise Refuse("seed11 accepted training steps are incomplete or reordered")
    if [row.get("step") for row in by_seed["12"]] != list(range(DISCARDED_SEED12_UPDATES)):
        raise Refuse("seed12 discarded prefix is incomplete or reordered")
    if by_seed["12"][-1].get("step") != DISCARDED_SEED12_UPDATES - 1:
        raise Refuse("last fully persisted seed12 update differs from the recovery plan")
    if any(row.get("seed") != 11 or row.get("arm") != ARM for row in gates + lengths):
        raise Refuse("only seed11 score rows are expected in the interrupted attempt")
    score_rows = {"gates": gates, "lengths": lengths}
    return (
        rows,
        score_rows,
        {
            "progress_sha256": digest,
            "event_counts": dict(counts),
            "valid_jsonl_rows": len(rows),
            "partial_terminal_record_bytes": len(partial_tail),
            "partial_terminal_record_sha256": hashlib.sha256(partial_tail).hexdigest(),
        },
    )


def _validate_interrupted_state(
    manifest: dict, runtime: dict, metadata: dict[str, Any]
) -> dict[str, Any]:
    if NEW_ATTEMPT.exists() or NEW_PROGRESS.exists() or NEW_RESULT.exists() or OLD_RESULT.exists():
        raise FileExistsError("N1-165A and official result paths are write-once")
    if n162.sha256(PLAN) != PLAN_SHA256:
        raise Refuse("N1-165A recovery plan SHA changed")
    old_attempt = n162.read_json(OLD_ATTEMPT)
    if (
        old_attempt.get("status") != "started"
        or old_attempt.get("plan_sha256") != n165.PLAN_SHA256
        or old_attempt.get("planned_updates") != 55_296
        or old_attempt.get("planned_train_score_queries") != 3_744
    ):
        raise Refuse("original N1-165 attempt is not the expected interrupted attempt")
    if (
        n162.sha256(OLD_ATTEMPT)
        != "06742a1e8cd72d224cf67e3c6023064c9253592ebf1149f89dc8d69bd969b5d1"
    ):
        raise Refuse("original N1-165 attempt SHA differs from the frozen recovery plan")
    old_preflight = n162.read_json(OLD_ZERO_PREFLIGHT)
    if (
        old_preflight.get("status") != "complete"
        or old_preflight.get("complete") is not True
        or old_preflight.get("zero_checkpoint_count") != 3
        or old_preflight.get("independent_restore_count") != 3
        or old_preflight.get("parameter_updates") != 0
        or n162.sha256(OLD_ZERO_PREFLIGHT)
        != "e628c03cd5e4ccf4f54e598635d593e7e715324438866a501768a936ea4ad003"
    ):
        raise Refuse("original N1-165 zero checkpoint preflight is not complete")

    rows, score_rows, old_progress_receipt = _collect_old_progress()
    checkpoint_root = n165.CHECKPOINT_ROOT
    required = {
        "s11_zero": checkpoint_root / f"s11_{ARM}_zero.pt",
        "s11_final": checkpoint_root / f"s11_{ARM}_final.pt",
        "s12_zero": checkpoint_root / f"s12_{ARM}_zero.pt",
        "s13_zero": checkpoint_root / f"s13_{ARM}_zero.pt",
    }
    forbidden = [
        checkpoint_root / f"s12_{ARM}_final.pt",
        checkpoint_root / f"s13_{ARM}_final.pt",
        checkpoint_root / f"s12_{ARM}_165a_partial.pt",
        checkpoint_root / f"s13_{ARM}_165a_partial.pt",
    ]
    if any(path.exists() for path in forbidden):
        raise Refuse("a seed12/13 recovery final or partial checkpoint already exists")
    if any(not path.is_file() for path in required.values()):
        raise Refuse("required N1-165 zero/final checkpoint is missing")
    expected_zeros = {int(row["seed"]): row for row in old_preflight["zero_checkpoints"]}
    zero_hashes = {}
    for seed in (11, 12, 13):
        path = required[f"s{seed}_zero"]
        row = expected_zeros.get(seed)
        if (
            row is None
            or row.get("arm") != ARM
            or n162.sha256(path) != row.get("zero_checkpoint_sha256")
        ):
            raise Refuse(f"N1-165 zero checkpoint differs from preflight: seed{seed}")
        zero_hashes[str(seed)] = row["zero_checkpoint_sha256"]
    seed11_sha = n162.sha256(required["s11_final"])
    if not seed11_sha:
        raise Refuse("seed11 final checkpoint could not be hashed")

    # Rebuild the frozen scorer/training view from the runtime returned by N1-165 readiness.
    base_schedule = n162.builder.build_schedule(runtime["n155_schedule"])
    if n162.builder.canonical_sha(base_schedule["rows"]) != manifest.get("base_schedule_sha256"):
        raise Refuse("N1-165 base schedule differs from the frozen manifest during recovery")
    if len(base_schedule["rows"]) != n165.SLOTS_PER_PASS:
        raise Refuse("N1-165 base schedule row count differs during recovery")
    return {
        "old_attempt": old_attempt,
        "old_preflight": old_preflight,
        "old_progress_rows": rows,
        "old_score_rows": score_rows,
        "old_progress_receipt": old_progress_receipt,
        "zero_checkpoint_hashes": zero_hashes,
        "seed11_final_path": required["s11_final"],
        "seed11_final_sha256": seed11_sha,
        "base_schedule": base_schedule,
        "parents": metadata["parents"],
        "n163_dose_anchors": metadata["n163_dose_anchors"],
    }


def _score_endpoint(
    seed: int,
    model,
    optimizer,
    runtime: dict[str, Any],
    base_schedule: dict[str, Any],
    budget: n165.CpuBudget,
):
    before = n162._fingerprint(model, optimizer)
    gate_rows, gate_summary, sufficient, gate_queries = n132._evaluate_train_only(
        runtime["n155_bound"]["g"],
        model,
        optimizer,
        seed,
        runtime["n155_bound"]["inputs"],
        budget,
    )
    if gate_queries != GATE_QUERIES or len(gate_rows) != GATE_QUERIES:
        raise Refuse(f"N1-165A original train gate denominator differs for seed{seed}")
    length_rows = n155._train_length_rows(
        model,
        optimizer,
        seed,
        "unknown8_rebalanced",
        runtime["n155_bound"]["inputs"],
        base_schedule["views"],
        budget,
    )
    if len(length_rows) != LENGTH_QUERIES:
        raise Refuse(f"N1-165A train length denominator differs for seed{seed}")
    if n162._fingerprint(model, optimizer) != before:
        raise Refuse(f"N1-165A scoring mutated the model/optimizer for seed{seed}")
    return gate_rows, gate_summary, sufficient, length_rows


def _seed_training_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("event") == "training_update" and int(row.get("seed", -1)) == seed
    ]
    if len(selected) != UPDATES_PER_ENDPOINT:
        raise Refuse(f"N1-165A preserved seed{seed} training row count differs")
    return selected


def _summarize_losses(rows: list[dict[str, Any]], seed: int) -> Counter:
    selected = _seed_training_rows(rows, seed)
    return Counter(
        {
            key: sum(float(row[key]) for row in selected)
            for key in ("total", "base_total", "qa", "observation", "binding")
        }
    )


def _context_counts(rows: list[dict[str, Any]], seed: int) -> Counter:
    return Counter(int(row["context_length"]) for row in _seed_training_rows(rows, seed))


def _template_counts(rows: list[dict[str, Any]], seed: int) -> Counter:
    return Counter(
        (str(row["kind"]), int(row["context_length"]), str(row["question_template_id"]))
        for row in _seed_training_rows(rows, seed)
    )


def _record(
    seed: int,
    parent: dict,
    zero_sha: str,
    zero_fingerprint: str,
    final_path: Path,
    final_sha: str,
    final_payload: dict,
    final_restore: dict,
    model,
    optimizer,
    metrics: Counter,
    length_counts: Counter,
    template_counts: Counter,
    gate_summary: dict,
    sufficient: bool,
    gate_queries: int,
    length_queries: int,
) -> dict[str, Any]:
    final_steps = n165._optimizer_step_values(optimizer)
    return {
        "seed": seed,
        "arm": ARM,
        "parent_experimental_arm": n165.schedule_builder.PARENT_ARM,
        "parent_checkpoint_path": parent["final_checkpoint_path"],
        "parent_checkpoint_sha256": parent["final_checkpoint_sha256"],
        "parent_bundle_fingerprint": parent["final_bundle_fingerprint"],
        "parent_total_updates": n165.EXPECTED_PARENT_STEPS,
        "parent_history_count": n165.EXPECTED_PARENT_HISTORY,
        "additional_updates_this_plan": UPDATES_PER_ENDPOINT,
        "cumulative_additional_updates_from_n150": int(
            parent["cumulative_additional_updates_from_n150"]
        )
        + UPDATES_PER_ENDPOINT,
        "total_updates": n165.EXPECTED_FINAL_STEPS,
        "initial_history_count": n165.EXPECTED_PARENT_HISTORY,
        "final_history_count": len(model._observations),
        "optimizer_step_values": sorted(final_steps),
        "context_length_update_counts": dict(sorted(length_counts.items())),
        "question_template_counts": {
            f"{kind}/{length}/{template}": count
            for (kind, length, template), count in sorted(template_counts.items())
        },
        "loss_means": {
            key: float(metrics[key]) / UPDATES_PER_ENDPOINT
            for key in ("total", "base_total", "qa", "observation", "binding")
        },
        "original_train_gate": gate_summary,
        "training_sufficient": sufficient,
        "original_train_gate_queries": gate_queries,
        "length_score_queries": length_queries,
        "final_checkpoint_path": str(final_path.resolve()),
        "final_checkpoint_sha256": final_sha,
        "final_bundle_fingerprint": final_payload["bundle_fingerprint"],
        "final_independent_restore": final_restore,
        "zero_checkpoint_sha256": zero_sha,
        "zero_fingerprint": zero_fingerprint,
    }


def execute() -> dict[str, Any]:
    started = time.process_time()
    if any(path.exists() for path in (NEW_ATTEMPT, NEW_PROGRESS, NEW_RESULT, OLD_RESULT)):
        raise FileExistsError("N1-165A recovery outputs and official result are write-once")
    if n162.sha256(PLAN) != PLAN_SHA256:
        raise Refuse("N1-165A recovery plan SHA mismatch")
    _readiness, manifest, runtime, metadata, source_hashes = n165._load_readiness()
    setup_cpu = time.process_time() - started
    budget = n165.CpuBudget(CPU_LIMIT_SECONDS - setup_cpu)
    if budget.limit <= 0:
        raise TimeoutError("N1-165A recovery CPU budget exhausted during source validation")

    interrupted = _validate_interrupted_state(manifest, runtime, metadata)
    source_hashes = dict(source_hashes)
    source_hashes.update(
        {
            str(PLAN.resolve()): PLAN_SHA256,
            str(OLD_ATTEMPT.resolve()): n162.sha256(OLD_ATTEMPT),
            str(OLD_PROGRESS.resolve()): interrupted["old_progress_receipt"]["progress_sha256"],
            str(OLD_ZERO_PREFLIGHT.resolve()): n162.sha256(OLD_ZERO_PREFLIGHT),
            str(Path(__file__).resolve()): n162.sha256(Path(__file__).resolve()),
            str(interrupted["seed11_final_path"].resolve()): interrupted["seed11_final_sha256"],
        }
    )
    if not n162.n150_builder.verify_hashes(source_hashes):
        raise Refuse("N1-165A recovery input hashes changed")

    attempt = {
        "schema": "taiji-n1-plan165a-recovery-attempt-v1",
        "status": "started",
        "plan_sha256": PLAN_SHA256,
        "n165_plan_sha256": n165.PLAN_SHA256,
        "n165_readiness_sha256": n162.sha256(n165.READINESS),
        "n165_manifest_sha256": n162.sha256(n165.MANIFEST),
        "n165_interrupted_progress_sha256": interrupted["old_progress_receipt"]["progress_sha256"],
        "interrupted_progress_update_rows": FIRST_ATTEMPT_PROGRESS_UPDATE_ROWS,
        "interrupted_update_operations_upper": FIRST_ATTEMPT_UPDATE_OPERATIONS,
        "discarded_seed12_updates": DISCARDED_SEED12_UPDATES,
        "partial_terminal_update_operations": PARTIAL_SEED12_UPDATE_OPERATIONS,
        "completion_updates": 2 * UPDATES_PER_ENDPOINT,
        "accepted_endpoint_updates": 3 * UPDATES_PER_ENDPOINT,
        "repeated_seed11_train_score_queries": SCORES_PER_ENDPOINT,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "prior_linked_cpu_upper_seconds": PRIOR_LINKED_CPU_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "can_promote": False,
    }
    write_new(NEW_ATTEMPT, attempt)

    update_count = 0
    score_count = 0
    final_records = []
    active = None
    model = optimizer = payload = None
    errors = []
    try:
        base_schedule = interrupted["base_schedule"]
        if n162.builder.canonical_sha(base_schedule["rows"]) != manifest["base_schedule_sha256"]:
            raise Refuse("N1-165A reconstructed base schedule differs from frozen manifest")

        # Seed11 was fully trained before interruption. Restore it, then repeat the same train-only scores
        # solely to rebuild the final result and compare every row with the preserved progress receipt.
        seed11 = 11
        seed11_path = interrupted["seed11_final_path"]
        seed11_restore = n165._independent_restore(seed11_path, arm=ARM, seed=seed11, budget=budget)
        model, optimizer, seed11_payload, seed11_fingerprint = n155._restore_checkpoint(seed11_path)
        if seed11_fingerprint != seed11_payload["bundle_fingerprint"]:
            raise Refuse("N1-165A seed11 final restore fingerprint differs")
        seed11_gate, seed11_gate_summary, seed11_sufficient, seed11_length = _score_endpoint(
            seed11, model, optimizer, runtime, interrupted["base_schedule"], budget
        )
        if n162._fingerprint(model, optimizer) != seed11_payload["bundle_fingerprint"]:
            raise Refuse("N1-165A seed11 scoring changed its final model/optimizer bundle")
        old_gate = interrupted["old_score_rows"]["gates"]
        old_length = interrupted["old_score_rows"]["lengths"]
        current_gate = [
            {**row, "event": "original_train_gate_row", "seed": seed11, "arm": ARM}
            for row in seed11_gate
        ]
        current_length = [
            {**row, "event": "length_score_row", "seed": seed11, "arm": ARM}
            for row in seed11_length
        ]
        if current_gate != old_gate or current_length != old_length:
            raise Refuse("N1-165A seed11 repeated train scores differ from the preserved rows")
        with NEW_PROGRESS.open("x", encoding="utf-8", newline="\n") as progress:
            n165.write_jsonl(
                progress,
                {
                    "event": "seed11_score_reconciliation",
                    "seed": seed11,
                    "arm": ARM,
                    "gate_rows": len(current_gate),
                    "length_rows": len(current_length),
                    "rows_match_preserved_progress": True,
                    "preserved_gate_rows_sha256": _digest(old_gate),
                    "preserved_length_rows_sha256": _digest(old_length),
                    "duplicate_train_queries": SCORES_PER_ENDPOINT,
                },
            )
            progress.flush()
            zero11 = next(
                row for row in interrupted["old_preflight"]["zero_checkpoints"] if row["seed"] == 11
            )
            seed11_record = _record(
                seed11,
                metadata["parents"][seed11],
                zero11["zero_checkpoint_sha256"],
                zero11["zero_fingerprint"],
                seed11_path,
                interrupted["seed11_final_sha256"],
                seed11_payload,
                seed11_restore,
                model,
                optimizer,
                _summarize_losses(interrupted["old_progress_rows"], seed11),
                _context_counts(interrupted["old_progress_rows"], seed11),
                _template_counts(interrupted["old_progress_rows"], seed11),
                seed11_gate_summary,
                seed11_sufficient,
                len(seed11_gate),
                len(seed11_length),
            )
            score_count += SCORES_PER_ENDPOINT
            del model, optimizer, seed11_payload
            model = optimizer = payload = None
            final_records.append(seed11_record)

            for seed in (12, 13):
                budget.guard(reserve=60)
                parent = metadata["parents"][seed]
                parent_path = Path(parent["final_checkpoint_path"]).resolve()
                parent_hash = parent["final_checkpoint_sha256"]
                zero_path = n165.CHECKPOINT_ROOT / f"s{seed}_{ARM}_zero.pt"
                zero_receipt = n165._independent_restore(
                    zero_path, arm=ARM, seed=seed, budget=budget
                )
                zero_hash = n162.sha256(zero_path)
                if zero_hash != interrupted["zero_checkpoint_hashes"][str(seed)]:
                    raise Refuse(f"N1-165A zero bundle changed before replay: seed{seed}")
                model, optimizer, payload, fingerprint = n155._restore_checkpoint(zero_path)
                if (
                    fingerprint != payload["bundle_fingerprint"]
                    or payload.get("continuation_plan_sha256") != n165.PLAN_SHA256
                    or payload.get("continuation_arm") != ARM
                    or payload.get("parent_checkpoint_sha256") != parent_hash
                    or len(model._observations) != n165.EXPECTED_PARENT_HISTORY
                    or n165._optimizer_step_values(optimizer) != {n165.EXPECTED_PARENT_STEPS}
                ):
                    raise Refuse(f"N1-165A zero bundle state differs for seed{seed}")
                active = {
                    "seed": seed,
                    "arm": ARM,
                    "completed_updates": 0,
                    "parent_path": parent_path,
                    "parent_hash": parent_hash,
                }
                metrics = Counter()
                length_counts = Counter()
                template_counts = Counter()
                old_prefix = (
                    {
                        int(row["step"]): row
                        for row in interrupted["old_progress_rows"]
                        if row.get("event") == "training_update"
                        and int(row.get("seed", -1)) == seed
                    }
                    if seed == 12
                    else {}
                )
                for step in range(UPDATES_PER_ENDPOINT):
                    budget.guard(reserve=20)
                    pass_index, slot_index = divmod(step, n165.SLOTS_PER_PASS)
                    source_row = manifest["rows"][slot_index]
                    slot = base_schedule["slots"][slot_index]
                    length = int(source_row["context_length"])
                    template_id = source_row["n165_candidate_template_id"]
                    item, second = n165._prepare_item(
                        slot, source_row, length, template_id, base_schedule["views"]
                    )
                    metrics_row = n162._training_step(
                        model,
                        optimizer,
                        item,
                        second,
                        f"n1-plan165-{seed}-{ARM}-pass{pass_index}-step{slot_index}",
                        runtime["n155_bound"]["g"],
                    )
                    active["completed_updates"] = step + 1
                    update_count += 1
                    progress_row = {
                        "event": "training_update",
                        "seed": seed,
                        "arm": ARM,
                        "step": step,
                        "pass": pass_index,
                        "slot_index": slot_index,
                        "parent_id": source_row["parent_id"],
                        "route": source_row["route"],
                        "kind": source_row["kind"],
                        "context_length": length,
                        "question_template_id": template_id,
                        "item_id": item["id"],
                        "question": item["question"],
                        "second_id": None if second is None else second["id"],
                        "second_question": None if second is None else second["question"],
                        **metrics_row,
                    }
                    if (
                        seed == 12
                        and step < DISCARDED_SEED12_UPDATES
                        and progress_row != old_prefix[step]
                    ):
                        raise Refuse(
                            f"N1-165A deterministic replay prefix differs at seed12 step {step}"
                        )
                    n165.write_jsonl(progress, progress_row)
                    if (step + 1) % 64 == 0:
                        progress.flush()
                    length_counts[length] += 1
                    template_counts[(str(source_row["kind"]), length, template_id)] += 1
                    for key in ("total", "base_total", "qa", "observation", "binding"):
                        metrics[key] += float(metrics_row[key])
                    if (step + 1) % 1_024 == 0:
                        print(
                            f"plan165a seed={seed} arm={ARM} step={step+1}/{UPDATES_PER_ENDPOINT} cpu={budget.used():.1f}",
                            flush=True,
                        )

                if (
                    active["completed_updates"] != UPDATES_PER_ENDPOINT
                    or length_counts != EXPECTED_CONTEXT_COUNTS
                ):
                    raise Refuse(f"N1-165A endpoint update/context counts differ for seed{seed}")
                if (
                    n165._optimizer_step_values(optimizer) != {n165.EXPECTED_FINAL_STEPS}
                    or len(model._observations) != n165.EXPECTED_FINAL_HISTORY
                ):
                    raise Refuse(f"N1-165A optimizer/history differs for seed{seed}")
                final_path = n165.CHECKPOINT_ROOT / f"s{seed}_{ARM}_final.pt"
                final_payload = n165._save_checkpoint(
                    model,
                    optimizer,
                    final_path,
                    seed=seed,
                    arm=ARM,
                    parent_path=parent_path,
                    parent_hash=parent_hash,
                    additional_updates=UPDATES_PER_ENDPOINT,
                )
                final_sha = n162.sha256(final_path)
                active = None
                final_restore = n165._independent_restore(
                    final_path, arm=ARM, seed=seed, budget=budget
                )
                if final_restore["fingerprint"] != final_payload["bundle_fingerprint"]:
                    raise Refuse(f"N1-165A final full bundle restore mismatch for seed{seed}")
                gate_rows, gate_summary, sufficient, queries = n132._evaluate_train_only(
                    runtime["n155_bound"]["g"],
                    model,
                    optimizer,
                    seed,
                    runtime["n155_bound"]["inputs"],
                    budget,
                )
                if queries != GATE_QUERIES or len(gate_rows) != GATE_QUERIES:
                    raise Refuse(f"N1-165A original train gate denominator differs for seed{seed}")
                for row in gate_rows:
                    n165.write_jsonl(
                        progress,
                        {**row, "event": "original_train_gate_row", "seed": seed, "arm": ARM},
                    )
                length_rows = n155._train_length_rows(
                    model,
                    optimizer,
                    seed,
                    "unknown8_rebalanced",
                    runtime["n155_bound"]["inputs"],
                    base_schedule["views"],
                    budget,
                )
                if len(length_rows) != LENGTH_QUERIES:
                    raise Refuse(f"N1-165A length score denominator differs for seed{seed}")
                for row in length_rows:
                    n165.write_jsonl(
                        progress, {**row, "event": "length_score_row", "seed": seed, "arm": ARM}
                    )
                if n162._fingerprint(model, optimizer) != final_payload["bundle_fingerprint"]:
                    raise Refuse(f"N1-165A train scoring altered final model state for seed{seed}")
                score_count += GATE_QUERIES + LENGTH_QUERIES
                zero_receipt = next(
                    row
                    for row in interrupted["old_preflight"]["zero_checkpoints"]
                    if row["seed"] == seed
                )
                record = _record(
                    seed,
                    parent,
                    zero_hash,
                    zero_receipt["zero_fingerprint"],
                    final_path,
                    final_sha,
                    final_payload,
                    final_restore,
                    model,
                    optimizer,
                    metrics,
                    length_counts,
                    template_counts,
                    gate_summary,
                    sufficient,
                    queries,
                    len(length_rows),
                )
                final_records.append(record)
                del payload
                model = optimizer = payload = None

            progress.flush()
        if (
            update_count != 2 * UPDATES_PER_ENDPOINT
            or score_count != 3 * SCORES_PER_ENDPOINT
            or len(final_records) != 3
        ):
            raise Refuse("N1-165A recovered update/score/endpoint totals differ")
        for parent in metadata["parents"].values():
            if (
                n162.sha256(Path(parent["final_checkpoint_path"]))
                != parent["final_checkpoint_sha256"]
            ):
                raise Refuse("N1-165A mutated a frozen N1-162B parent checkpoint")
        source_hashes[str(NEW_PROGRESS.resolve())] = n162.sha256(NEW_PROGRESS)
        for record in final_records:
            source_hashes[record["final_checkpoint_path"]] = record["final_checkpoint_sha256"]
        if not n162.n150_builder.verify_hashes(source_hashes):
            raise Refuse("N1-165A recovery source/checkpoint hashes changed")

        completion_cpu = budget.used() + setup_cpu
        linked_upper = PRIOR_LINKED_CPU_SECONDS + completion_cpu
        if completion_cpu > CPU_LIMIT_SECONDS or linked_upper > LINKED_CPU_LIMIT_SECONDS:
            raise TimeoutError("N1-165A completion CPU budget exceeded")
        result = {
            "schema": "taiji-n1-plan165-transition-template-mid-dose-result-v2",
            "status": "complete",
            "complete": True,
            "plan_sha256": n165.PLAN_SHA256,
            "completion_plan_sha256": PLAN_SHA256,
            "readiness_sha256": n162.sha256(n165.READINESS),
            "manifest_sha256": n162.sha256(n165.MANIFEST),
            "n162b_result_sha256": n165.schedule_builder.PRIOR_RESULT_SHA256,
            "f1_evidence_sha256": n165.schedule_builder.F1_RESULT_SHA256,
            "g2_evidence_sha256": n165.schedule_builder.G2_RESULT_SHA256,
            "n163_anchor_result_sha256": n165.schedule_builder.N163_RESULT_SHA256,
            "n164_attribution_sha256": n165.schedule_builder.N164_REPORT_SHA256,
            "n163_dose_anchors": metadata["n163_dose_anchors"],
            "seeds": list(SEEDS),
            "arms": [ARM],
            "updates_per_model": UPDATES_PER_ENDPOINT,
            "total_updates": 3 * UPDATES_PER_ENDPOINT,
            "train_score_queries": 3 * SCORES_PER_ENDPOINT,
            "gate_queries_per_endpoint": GATE_QUERIES,
            "length_queries_per_endpoint": LENGTH_QUERIES,
            "repeated_seed11_train_score_queries": SCORES_PER_ENDPOINT,
            "completion_train_score_calls": score_count,
            "initial_attempt_progress_update_rows": FIRST_ATTEMPT_PROGRESS_UPDATE_ROWS,
            "initial_attempt_update_operations_upper": FIRST_ATTEMPT_UPDATE_OPERATIONS,
            "initial_attempt_train_score_calls": SCORES_PER_ENDPOINT,
            "total_train_score_calls_across_attempts": SCORES_PER_ENDPOINT + score_count,
            "discarded_seed12_partial_updates": DISCARDED_SEED12_UPDATES
            + PARTIAL_SEED12_UPDATE_OPERATIONS,
            "partial_terminal_update_operations": PARTIAL_SEED12_UPDATE_OPERATIONS,
            "partial_terminal_record_bytes": interrupted["old_progress_receipt"][
                "partial_terminal_record_bytes"
            ],
            "partial_terminal_record_sha256": interrupted["old_progress_receipt"][
                "partial_terminal_record_sha256"
            ],
            "completion_updates": update_count,
            "physical_parameter_update_operations": FIRST_ATTEMPT_UPDATE_OPERATIONS + update_count,
            "initial_attempt_status": "interrupted_before_result",
            "initial_seed11_score_rows_reproduced_exactly": True,
            "seed12_replay_prefix_rows_matched": DISCARDED_SEED12_UPDATES,
            "seed12_replay_prefix_total": DISCARDED_SEED12_UPDATES,
            "candidate_sufficient_all_seeds": all(
                row["training_sufficient"] for row in final_records
            ),
            "mid_dose_sufficient_all_seeds": all(
                row["training_sufficient"] for row in final_records
            ),
            "training_sufficient_all_endpoints": all(
                row["training_sufficient"] for row in final_records
            ),
            "zero_checkpoint_count": 3,
            "zero_independent_restores": 3,
            "recovery_zero_rechecks": 2,
            "final_checkpoint_count": 3,
            "final_independent_restores": sum(
                int(row["final_independent_restore"]["complete"]) for row in final_records
            ),
            "checkpoint_writes_completion": 2,
            "checkpoint_writes_total_across_attempts": 6,
            "checkpoint_writes": 6,
            "fresh_face_queries": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
            "confirmation_queries": 0,
            "parameter_updates": 3 * UPDATES_PER_ENDPOINT,
            "completion_cpu_seconds": round(completion_cpu, 6),
            "completion_cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "interrupted_attempt_cpu_upper_seconds": FIRST_ATTEMPT_CPU_UPPER_SECONDS,
            "prior_linked_cpu_upper_seconds": PRIOR_LINKED_CPU_SECONDS,
            "cumulative_linked_cpu_upper_seconds": round(linked_upper, 6),
            "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
            "source_hashes": dict(sorted(source_hashes.items())),
            "source_hashes_unchanged": True,
            "models": final_records,
            "errors": [],
            "can_promote": False,
        }
        recovery = {
            "schema": "taiji-n1-plan165a-completion-result-v1",
            "status": "complete",
            "complete": True,
            "completion_plan_sha256": PLAN_SHA256,
            "n165_result_sha256": _digest(result),
            "initial_attempt_sha256": n162.sha256(OLD_ATTEMPT),
            "initial_progress_sha256": interrupted["old_progress_receipt"]["progress_sha256"],
            "initial_partial_terminal_record_sha256": interrupted["old_progress_receipt"][
                "partial_terminal_record_sha256"
            ],
            "seed11_score_rows_reconciled": SCORES_PER_ENDPOINT,
            "seed12_discarded_prefix_replayed_and_matched": DISCARDED_SEED12_UPDATES,
            "seed12_unpersisted_partial_update_operations_discarded": PARTIAL_SEED12_UPDATE_OPERATIONS,
            "seed12_and_seed13_recovered_updates": update_count,
            "cumulative_linked_cpu_upper_seconds": result["cumulative_linked_cpu_upper_seconds"],
            "can_promote": False,
            "summary_zh": "N1-165在用户调整优先级前被外部中断；本completion保留seed11，重放seed12无checkpoint前缀并完成seed12/13。旧attempt/进度保留，seed11重评分与seed12已完整写入的前缀逐行匹配；末尾一条不完整JSON记录不用于匹配。",
        }
        write_new(OLD_RESULT, result)
        write_new(NEW_RESULT, recovery)
        return result
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        if active is not None and model is not None and optimizer is not None:
            try:
                partial_path = n165.CHECKPOINT_ROOT / f"s{active['seed']}_{ARM}_165a_partial.pt"
                partial_payload = n165._save_checkpoint(
                    model,
                    optimizer,
                    partial_path,
                    seed=int(active["seed"]),
                    arm=ARM,
                    parent_path=Path(active["parent_path"]),
                    parent_hash=str(active["parent_hash"]),
                    additional_updates=int(active["completed_updates"]),
                )
                partial_checkpoint = {
                    "path": str(partial_path.resolve()),
                    "sha256": n162.sha256(partial_path),
                    "additional_updates": active["completed_updates"],
                    "fingerprint": partial_payload["bundle_fingerprint"],
                }
            except Exception as save_exc:
                partial_checkpoint = {"save_error": f"{type(save_exc).__name__}: {save_exc}"}
        else:
            partial_checkpoint = None
        failure = {
            "schema": "taiji-n1-plan165a-completion-result-v1",
            "status": "partial" if update_count else "failed_before_training",
            "complete": False,
            "completion_plan_sha256": PLAN_SHA256,
            "initial_attempt_sha256": n162.sha256(OLD_ATTEMPT) if OLD_ATTEMPT.is_file() else None,
            "initial_progress_sha256": (
                n162.sha256(OLD_PROGRESS) if OLD_PROGRESS.is_file() else None
            ),
            "recovery_updates": update_count,
            "recovery_score_queries": score_count,
            "completed_models": final_records,
            "active_endpoint": active,
            "partial_checkpoint": partial_checkpoint,
            "cpu_seconds": round(budget.used(), 6),
            "cpu_limit_seconds": budget.limit,
            "errors": errors,
            "can_promote": False,
        }
        if not NEW_RESULT.exists():
            write_new(NEW_RESULT, failure)
        raise
    finally:
        if "progress" in locals():
            try:
                progress.flush()
                progress.close()
            except Exception:
                pass


def main() -> int:
    torch.set_num_threads(1)
    try:
        report = execute()
    except Exception as exc:
        print(f"N1-165A refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "total_updates",
                    "train_score_queries",
                    "completion_cpu_seconds",
                    "candidate_sufficient_all_seeds",
                    "can_promote",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
