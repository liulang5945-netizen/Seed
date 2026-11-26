"""Paired continuation from N1-162B augmented checkpoints for PLAN-N1-165."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_plan165_transition_template_schedule as schedule_builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_plan155_short_unknown_context_length_training as n155
from scripts.training import run_taiji_n1_plan162b_question_surface_matched_continuation as n162

PLAN = schedule_builder.PLAN
PLAN_SHA256 = schedule_builder.PLAN_SHA256
MANIFEST = schedule_builder.MANIFEST
READINESS = schedule_builder.READINESS
PRIOR_RESULT = schedule_builder.PRIOR_RESULT
ZERO_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan165_transition_template_mid_dose_zero_attempt_20261020.json"
)
ZERO_PREFLIGHT = (
    ROOT / "reports/taiji_n1_plan165_transition_template_mid_dose_zero_preflight_20261020.json"
)
ATTEMPT = ROOT / "reports/taiji_n1_plan165_transition_template_mid_dose_attempt_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan165_transition_template_mid_dose_progress_20261020.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan165_transition_template_mid_dose_result_20261020.json"
CHECKPOINT_ROOT = ROOT / "data/checkpoints/n1_plan165_transition_template_mid_dose_20261020"

SEEDS = schedule_builder.PARENT_SEEDS
ARMS = (schedule_builder.CANDIDATE_ARM,)
SLOTS_PER_PASS = 9_216
PASSES = 2
UPDATES_PER_MODEL = SLOTS_PER_PASS * PASSES
TOTAL_UPDATES = len(SEEDS) * len(ARMS) * UPDATES_PER_MODEL
GATE_QUERIES_PER_ENDPOINT = 960
LENGTH_QUERIES_PER_ENDPOINT = 288
TRAIN_SCORE_QUERIES = (
    len(SEEDS) * len(ARMS) * (GATE_QUERIES_PER_ENDPOINT + LENGTH_QUERIES_PER_ENDPOINT)
)
STAGE_CPU_LIMIT_SECONDS = 10_500.0
LINKED_CPU_LIMIT_SECONDS = 50_000.0
PRIOR_LINKED_CPU_SECONDS = 38_253.796875
EXPECTED_PARENT_STEPS = 61_440
EXPECTED_PARENT_HISTORY = 79_872
EXPECTED_FINAL_STEPS = EXPECTED_PARENT_STEPS + UPDATES_PER_MODEL
EXPECTED_FINAL_HISTORY = EXPECTED_PARENT_HISTORY + UPDATES_PER_MODEL
EXPECTED_CONTEXT_COUNTS = Counter({8: 6_912, 16: 5_760, 32: 5_760})


class Refuse(RuntimeError):
    """N1-165 来源、bundle、training或评分合同不满足。"""


class CpuBudget:
    def __init__(self, limit: float) -> None:
        self.limit = float(limit)
        self.started = time.process_time()
        self.child_seconds = 0.0

    def used(self) -> float:
        return time.process_time() - self.started + self.child_seconds

    def guard(self, reserve: float = 0.0) -> None:
        used = self.used()
        if used + reserve > self.limit:
            raise TimeoutError(
                f"N1-165 CPU budget exceeded: {used:.3f}+{reserve:.3f}/{self.limit:.3f}"
            )


def write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"N1-165 write-once output exists: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def write_jsonl(stream, value: dict) -> None:
    json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")


def _optimizer_step_values(optimizer) -> set[int]:
    return {int(state["step"]) for state in optimizer.state.values() if "step" in state}


def _worker_restore(path: Path) -> dict:
    started = time.process_time()
    model, optimizer, payload, fingerprint = n155._restore_checkpoint(path)
    steps = sorted(_optimizer_step_values(optimizer))
    result = {
        "complete": True,
        "equal": fingerprint == payload["bundle_fingerprint"],
        "fingerprint": fingerprint,
        "bundle_fingerprint": payload["bundle_fingerprint"],
        "continuation_plan_sha256": payload.get("continuation_plan_sha256"),
        "continuation_arm": payload.get("continuation_arm"),
        "parent_experimental_arm": payload.get("parent_experimental_arm"),
        "seed": payload.get("seed"),
        "branch": payload.get("branch"),
        "total_updates": payload.get("total_updates"),
        "additional_updates": payload.get("additional_updates"),
        "history_count": len(model._observations),
        "optimizer_parameters": len(optimizer.state),
        "optimizer_step_values": steps,
        "cpu_seconds": round(time.process_time() - started, 6),
    }
    del model, optimizer, payload
    return result


def _independent_restore(path: Path, *, arm: str, seed: int, budget: CpuBudget) -> dict:
    budget.guard(reserve=5)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--restore-worker", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=min(120.0, max(1.0, budget.limit - budget.used() - 1.0)),
        env=env,
    )
    restored = json.loads(proc.stdout)
    budget.child_seconds += float(restored["cpu_seconds"])
    expected_steps = (
        EXPECTED_PARENT_STEPS if path.name.endswith("_zero.pt") else EXPECTED_FINAL_STEPS
    )
    expected_history = (
        EXPECTED_PARENT_HISTORY if path.name.endswith("_zero.pt") else EXPECTED_FINAL_HISTORY
    )
    if (
        restored.get("complete") is not True
        or restored.get("equal") is not True
        or restored.get("continuation_plan_sha256") != PLAN_SHA256
        or restored.get("continuation_arm") != arm
        or restored.get("parent_experimental_arm") != schedule_builder.PARENT_ARM
        or restored.get("seed") != seed
        or restored.get("total_updates") != expected_steps
        or restored.get("history_count") != expected_history
        or restored.get("optimizer_parameters") != 41
        or restored.get("optimizer_step_values") != [expected_steps]
    ):
        raise Refuse(f"N1-165 independent restore mismatch: {path}")
    budget.guard()
    return {"path": str(path.resolve()), "sha256": n162.sha256(path), **restored}


def _load_readiness() -> tuple[dict, dict, dict, dict, dict[str, str]]:
    if n162.sha256(PLAN) != PLAN_SHA256:
        raise Refuse("PLAN-N1-165 plan SHA changed")
    readiness = n162.read_json(READINESS)
    manifest = n162.read_json(MANIFEST)
    if (
        readiness.get("status") != "ready"
        or readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("readiness_rebinding_addendum_sha256")
        != schedule_builder.REBINDING_PLAN_SHA256
        or readiness.get("manifest_sha256") != n162.sha256(MANIFEST)
        or readiness.get("source_hashes_unchanged") is not True
        or manifest.get("plan_sha256") != PLAN_SHA256
        or manifest.get("readiness_rebinding_addendum_sha256")
        != schedule_builder.REBINDING_PLAN_SHA256
        or manifest.get("rows_sha256") != n162.builder.canonical_sha(manifest.get("rows", []))
        or len(manifest.get("rows", [])) != SLOTS_PER_PASS
        or manifest.get("total_updates") != TOTAL_UPDATES
    ):
        raise Refuse("N1-165 schedule readiness or manifest is stale/incomplete")
    sources = dict(readiness.get("source_hashes", {}))
    if not sources or not n162.n150_builder.verify_hashes(sources):
        raise Refuse("N1-165 readiness-bound source hashes changed")
    runtime_readiness, zero, runtime, runtime_sources = n162._verify_readiness()
    sources.update(runtime_sources)
    if not n162.n150_builder.verify_hashes(sources):
        raise Refuse("N1-165 training runtime sources changed")
    prior = n162.read_json(PRIOR_RESULT)
    n163_anchor = n162.read_json(schedule_builder.N163_RESULT)
    if (
        n162.sha256(schedule_builder.N163_RESULT) != schedule_builder.N163_RESULT_SHA256
        or n163_anchor.get("complete") is not True
        or n163_anchor.get("source_hashes_unchanged") is not True
        or n163_anchor.get("can_promote") is not False
    ):
        raise Refuse("N1-163 dose-anchor result is stale or incomplete")
    parent_map = {}
    for seed in SEEDS:
        matches = [
            row
            for row in prior["models"]
            if row.get("seed") == seed and row.get("arm") == schedule_builder.PARENT_ARM
        ]
        if len(matches) != 1:
            raise Refuse(f"N1-165 parent endpoint ambiguous: seed{seed}")
        parent = matches[0]
        parent_map[seed] = parent
        if parent["final_checkpoint_sha256"] != readiness["parent_endpoints"][str(seed)]["sha256"]:
            raise Refuse(f"N1-165 source parent differs from readiness: seed{seed}")
        anchors = readiness.get("n163_dose_anchors", {}).get(str(seed), {})
        if set(anchors) != {schedule_builder.N163_CONTROL_ARM, schedule_builder.N163_CANDIDATE_ARM}:
            raise Refuse(f"N1-165 N1-163 dose anchors are incomplete for seed{seed}")
        for anchor_arm, anchor in anchors.items():
            if (
                anchor.get("path") not in sources
                or sources[anchor["path"]] != anchor.get("sha256")
                or n162.sha256(Path(anchor["path"])) != anchor.get("sha256")
            ):
                raise Refuse(f"N1-165 dose anchor source binding differs: seed{seed}/{anchor_arm}")
    return (
        readiness,
        manifest,
        runtime,
        {
            "prior": prior,
            "parents": parent_map,
            "zero": zero,
            "runtime_readiness": runtime_readiness,
            "n163_anchor": n163_anchor,
            "n163_dose_anchors": readiness["n163_dose_anchors"],
        },
        sources,
    )


def _save_checkpoint(
    model,
    optimizer,
    path: Path,
    *,
    seed: int,
    arm: str,
    parent_path: Path,
    parent_hash: str,
    additional_updates: int,
) -> dict:
    if path.exists():
        raise FileExistsError(f"N1-165 checkpoint path is write-once: {path}")
    payload = n155._checkpoint_payload(
        model,
        optimizer,
        seed=seed,
        branch="unknown8_rebalanced",
        parent_path=parent_path,
        parent_hash=parent_hash,
        parent_total=EXPECTED_PARENT_STEPS,
        additional_updates=additional_updates,
    )
    payload["continuation_plan_sha256"] = PLAN_SHA256
    payload["continuation_arm"] = arm
    payload["parent_experimental_arm"] = schedule_builder.PARENT_ARM
    payload["parent_result_sha256"] = schedule_builder.PRIOR_RESULT_SHA256
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
    return payload


def checkpoint_preflight(manifest: dict, metadata: dict, budget: CpuBudget) -> dict:
    if ZERO_ATTEMPT.exists() or ZERO_PREFLIGHT.exists():
        raise FileExistsError("N1-165 zero preflight outputs are write-once")
    if CHECKPOINT_ROOT.exists():
        raise FileExistsError(f"N1-165 checkpoint root already exists: {CHECKPOINT_ROOT}")
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=False)
    write_new(
        ZERO_ATTEMPT,
        {
            "schema": "taiji-n1-plan165-zero-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "parent_result_sha256": schedule_builder.PRIOR_RESULT_SHA256,
            "planned_zero_bundles": len(SEEDS) * len(ARMS),
            "model_queries": 0,
            "parameter_updates": 0,
            "optimizer_steps": 0,
            "checkpoint_root": str(CHECKPOINT_ROOT.resolve()),
            "can_promote": False,
        },
    )
    pairs = []
    pair_fingerprints = {}
    try:
        for seed in SEEDS:
            budget.guard(reserve=20)
            parent = metadata["parents"][seed]
            parent_path = Path(parent["final_checkpoint_path"]).resolve()
            parent_hash = str(parent["final_checkpoint_sha256"])
            if n162.sha256(parent_path) != parent_hash:
                raise Refuse(f"N1-165 parent checkpoint changed before zero save: seed{seed}")
            model, optimizer, payload, fingerprint = n155._restore_checkpoint(parent_path)
            if (
                fingerprint != parent["final_bundle_fingerprint"]
                or len(model._observations) != EXPECTED_PARENT_HISTORY
                or _optimizer_step_values(optimizer) != {EXPECTED_PARENT_STEPS}
                or payload.get("total_updates") != EXPECTED_PARENT_STEPS
            ):
                raise Refuse(f"N1-165 parent full bundle mismatch: seed{seed}")
            pair_fingerprints[seed] = fingerprint
            for arm in ARMS:
                path = CHECKPOINT_ROOT / f"s{seed}_{arm}_zero.pt"
                zero_payload = _save_checkpoint(
                    model,
                    optimizer,
                    path,
                    seed=seed,
                    arm=arm,
                    parent_path=parent_path,
                    parent_hash=parent_hash,
                    additional_updates=0,
                )
                receipt = _independent_restore(path, arm=arm, seed=seed, budget=budget)
                if (
                    receipt["fingerprint"] != fingerprint
                    or zero_payload["bundle_fingerprint"] != fingerprint
                ):
                    raise Refuse(f"N1-165 pair zero bundles differ: seed{seed}/{arm}")
                pairs.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "parent_checkpoint_path": str(parent_path),
                        "parent_checkpoint_sha256": parent_hash,
                        "parent_fingerprint": fingerprint,
                        "zero_checkpoint_path": str(path.resolve()),
                        "zero_checkpoint_sha256": n162.sha256(path),
                        "zero_fingerprint": zero_payload["bundle_fingerprint"],
                        "zero_history_count": len(model._observations),
                        "zero_optimizer_step": EXPECTED_PARENT_STEPS,
                        "independent_restore": receipt,
                    }
                )
            if n162.sha256(parent_path) != parent_hash:
                raise Refuse(f"N1-165 parent changed while saving zero pair: seed{seed}")
            del model, optimizer, payload

        expected_zero_count = len(SEEDS) * len(ARMS)
        if len(pairs) != expected_zero_count or any(
            pair["zero_fingerprint"] != pair_fingerprints[pair["seed"]] for pair in pairs
        ):
            raise Refuse("N1-165 zero bundles did not preserve their complete parent state")
        report = {
            "schema": "taiji-n1-plan165-zero-preflight-v1",
            "status": "complete",
            "complete": True,
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": n162.sha256(READINESS),
            "manifest_sha256": n162.sha256(MANIFEST),
            "zero_checkpoint_count": len(pairs),
            "independent_restore_count": sum(
                int(pair["independent_restore"]["complete"]) for pair in pairs
            ),
            "zero_fingerprints_match_parent": True,
            "parent_endpoint_count": len(pair_fingerprints),
            "zero_checkpoints": pairs,
            "parameter_updates": 0,
            "model_queries": 0,
            "optimizer_steps": 0,
            "cpu_seconds": round(budget.used(), 6),
            "checkpoint_writes": len(pairs),
            "can_promote": False,
        }
    except Exception as exc:
        report = {
            "schema": "taiji-n1-plan165-zero-preflight-v1",
            "status": "failed",
            "complete": False,
            "plan_sha256": PLAN_SHA256,
            "zero_checkpoint_count": len(pairs),
            "independent_restore_count": sum(
                int(pair["independent_restore"]["complete"]) for pair in pairs
            ),
            "zero_checkpoints": pairs,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "cpu_seconds": round(budget.used(), 6),
            "checkpoint_writes": sum(1 for p in CHECKPOINT_ROOT.glob("*_zero.pt") if p.is_file()),
            "can_promote": False,
        }
        write_new(ZERO_PREFLIGHT, report)
        raise
    write_new(ZERO_PREFLIGHT, report)
    budget.guard()
    return report


def _prepare_item(
    slot: dict, row: dict, length: int, template_id: str, views: dict
) -> tuple[dict, dict | None]:
    item = n162.n150._apply_context(slot["item"], slot["item_key"], length, views)
    second = None
    if slot.get("second") is not None:
        second = n162.n150._with_observations(slot["second"], item["observations"], length)
    if template_id != "original":
        target = n132._query_subject(item["question"])
        item["question"] = n162.builder.TRAINING_TEMPLATES[template_id].format(entity=target)
        if second is not None:
            second_target = n132._query_subject(second["question"])
            second["question"] = n162.builder.TRAINING_TEMPLATES[template_id].format(
                entity=second_target
            )
    if item.get("kind") == "unknown" and second is None:
        raise Refuse("N1-165 unknown paired query is missing its second question")
    return item, second


def execute() -> dict:
    output_paths = (ZERO_ATTEMPT, ZERO_PREFLIGHT, ATTEMPT, PROGRESS, RESULT)
    if any(path.exists() for path in output_paths):
        raise FileExistsError("N1-165 reports are write-once")
    validation_started = time.process_time()
    readiness, manifest, runtime, metadata, source_hashes = _load_readiness()
    readiness_cpu = float(readiness["cpu_seconds"])
    validation_cpu = time.process_time() - validation_started
    linked_before = PRIOR_LINKED_CPU_SECONDS + readiness_cpu + validation_cpu
    remaining_stage = STAGE_CPU_LIMIT_SECONDS - readiness_cpu - validation_cpu
    remaining_linked = LINKED_CPU_LIMIT_SECONDS - linked_before
    if min(remaining_stage, remaining_linked) <= 0:
        raise TimeoutError("N1-165 linked CPU budget is exhausted before checkpoint preflight")
    budget = CpuBudget(min(remaining_stage, remaining_linked))

    write_new(
        ATTEMPT,
        {
            "schema": "taiji-n1-plan165-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": n162.sha256(READINESS),
            "manifest_sha256": n162.sha256(MANIFEST),
            "zero_preflight_required": True,
            "planned_updates": TOTAL_UPDATES,
            "planned_train_score_queries": TRAIN_SCORE_QUERIES,
            "fresh_face_queries": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
            "confirmation_queries": 0,
            "cpu_limit_seconds": round(min(remaining_stage, remaining_linked), 6),
            "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
            "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
            "readiness_cpu_seconds": readiness_cpu,
            "live_source_validation_cpu_seconds": round(validation_cpu, 6),
            "checkpoint_root": str(CHECKPOINT_ROOT.resolve()),
            "can_promote": False,
        },
    )

    zero_preflight = checkpoint_preflight(manifest, metadata, budget)
    if (
        zero_preflight.get("complete") is not True
        or zero_preflight.get("zero_checkpoint_count") != len(SEEDS) * len(ARMS)
        or zero_preflight.get("independent_restore_count") != len(SEEDS) * len(ARMS)
        or zero_preflight.get("zero_fingerprints_match_parent") is not True
        or zero_preflight.get("parameter_updates") != 0
    ):
        raise Refuse("N1-165 checkpoint save/restore preflight failed; no training allowed")

    progress = PROGRESS.open("x", encoding="utf-8")
    final_records = []
    errors = []
    update_count = 0
    score_count = 0
    active = None
    model = optimizer = payload = None
    partial_checkpoint = None
    try:
        base_schedule = n162.builder.build_schedule(runtime["n155_schedule"])
        if n162.builder.canonical_sha(base_schedule["rows"]) != manifest.get(
            "base_schedule_sha256"
        ):
            raise Refuse("N1-165 reconstructed base schedule differs from frozen manifest")
        derived_rows = manifest["rows"]
        for seed in SEEDS:
            parent = metadata["parents"][seed]
            parent_path = Path(parent["final_checkpoint_path"]).resolve()
            parent_hash = str(parent["final_checkpoint_sha256"])
            for arm in ARMS:
                budget.guard(reserve=60)
                zero_path = CHECKPOINT_ROOT / f"s{seed}_{arm}_zero.pt"
                zero_sha = n162.sha256(zero_path)
                if n162.sha256(parent_path) != parent_hash:
                    raise Refuse(f"N1-165 parent checkpoint changed before training: seed{seed}")
                model, optimizer, payload, fingerprint = n155._restore_checkpoint(zero_path)
                if (
                    fingerprint != payload["bundle_fingerprint"]
                    or payload.get("continuation_arm") != arm
                    or payload.get("continuation_plan_sha256") != PLAN_SHA256
                    or payload.get("parent_checkpoint_sha256") != parent_hash
                    or payload.get("total_updates") != EXPECTED_PARENT_STEPS
                    or len(model._observations) != EXPECTED_PARENT_HISTORY
                    or _optimizer_step_values(optimizer) != {EXPECTED_PARENT_STEPS}
                ):
                    raise Refuse(f"N1-165 restored zero bundle mismatch: seed{seed}/{arm}")
                active = {
                    "seed": seed,
                    "arm": arm,
                    "completed_updates": 0,
                    "parent_path": parent_path,
                    "parent_hash": parent_hash,
                }
                metrics = Counter()
                length_counts = Counter()
                template_counts = Counter()

                for step in range(UPDATES_PER_MODEL):
                    budget.guard(reserve=20)
                    pass_index, slot_index = divmod(step, SLOTS_PER_PASS)
                    source_row = derived_rows[slot_index]
                    slot = base_schedule["slots"][slot_index]
                    length = int(source_row["context_length"])
                    template_id = source_row["n165_candidate_template_id"]
                    item, second = _prepare_item(
                        slot, source_row, length, template_id, base_schedule["views"]
                    )
                    result = n162._training_step(
                        model,
                        optimizer,
                        item,
                        second,
                        f"n1-plan165-{seed}-{arm}-pass{pass_index}-step{slot_index}",
                        runtime["n155_bound"]["g"],
                    )
                    active["completed_updates"] += 1
                    update_count += 1
                    length_counts[length] += 1
                    template_counts[(str(source_row["kind"]), length, template_id)] += 1
                    for key in ("total", "base_total", "qa", "observation", "binding"):
                        metrics[key] += float(result[key])
                    write_jsonl(
                        progress,
                        {
                            "event": "training_update",
                            "seed": seed,
                            "arm": arm,
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
                            **result,
                        },
                    )
                    if (step + 1) % 64 == 0:
                        progress.flush()
                    if (step + 1) % 1_024 == 0:
                        print(
                            f"plan165 seed={seed} arm={arm} step={step+1}/{UPDATES_PER_MODEL} cpu={budget.used():.1f}",
                            flush=True,
                        )

                if (
                    active["completed_updates"] != UPDATES_PER_MODEL
                    or length_counts != EXPECTED_CONTEXT_COUNTS
                ):
                    raise Refuse(f"N1-165 update/context counts differ: {seed}/{arm}")
                final_step_values = _optimizer_step_values(optimizer)
                if (
                    final_step_values != {EXPECTED_FINAL_STEPS}
                    or len(model._observations) != EXPECTED_FINAL_HISTORY
                ):
                    raise Refuse(f"N1-165 optimizer/history continuation differs: {seed}/{arm}")

                final_path = CHECKPOINT_ROOT / f"s{seed}_{arm}_final.pt"
                final_payload = _save_checkpoint(
                    model,
                    optimizer,
                    final_path,
                    seed=seed,
                    arm=arm,
                    parent_path=parent_path,
                    parent_hash=parent_hash,
                    additional_updates=UPDATES_PER_MODEL,
                )
                final_sha = n162.sha256(final_path)
                final_restore = _independent_restore(final_path, arm=arm, seed=seed, budget=budget)
                if final_restore["fingerprint"] != final_payload["bundle_fingerprint"]:
                    raise Refuse(f"N1-165 final full bundle restore mismatch: {seed}/{arm}")

                before_score = n162._fingerprint(model, optimizer)
                gate_rows, gate_summary, sufficient, gate_queries = n132._evaluate_train_only(
                    runtime["n155_bound"]["g"],
                    model,
                    optimizer,
                    seed,
                    runtime["n155_bound"]["inputs"],
                    budget,
                )
                if (
                    gate_queries != GATE_QUERIES_PER_ENDPOINT
                    or len(gate_rows) != GATE_QUERIES_PER_ENDPOINT
                ):
                    raise Refuse(f"N1-165 original train gate denominator differs: {seed}/{arm}")
                for score_row in gate_rows:
                    write_jsonl(
                        progress,
                        {**score_row, "event": "original_train_gate_row", "seed": seed, "arm": arm},
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
                if len(length_rows) != LENGTH_QUERIES_PER_ENDPOINT:
                    raise Refuse(f"N1-165 train length score denominator differs: {seed}/{arm}")
                for score_row in length_rows:
                    write_jsonl(
                        progress,
                        {**score_row, "event": "length_score_row", "seed": seed, "arm": arm},
                    )
                score_count += gate_queries + len(length_rows)
                if n162._fingerprint(model, optimizer) != before_score:
                    raise Refuse(f"N1-165 train-only scoring mutated final bundle: {seed}/{arm}")

                final_records.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "parent_experimental_arm": schedule_builder.PARENT_ARM,
                        "parent_checkpoint_path": str(parent_path),
                        "parent_checkpoint_sha256": parent_hash,
                        "parent_bundle_fingerprint": parent["final_bundle_fingerprint"],
                        "parent_total_updates": EXPECTED_PARENT_STEPS,
                        "parent_history_count": EXPECTED_PARENT_HISTORY,
                        "additional_updates_this_plan": UPDATES_PER_MODEL,
                        "cumulative_additional_updates_from_n150": int(
                            parent["cumulative_additional_updates_from_n150"]
                        )
                        + UPDATES_PER_MODEL,
                        "total_updates": EXPECTED_FINAL_STEPS,
                        "initial_history_count": EXPECTED_PARENT_HISTORY,
                        "final_history_count": len(model._observations),
                        "optimizer_step_values": sorted(final_step_values),
                        "context_length_update_counts": dict(sorted(length_counts.items())),
                        "question_template_counts": {
                            f"{kind}/{length}/{template}": count
                            for (kind, length, template), count in sorted(template_counts.items())
                        },
                        "loss_means": {
                            key: float(metrics[key]) / UPDATES_PER_MODEL
                            for key in ("total", "base_total", "qa", "observation", "binding")
                        },
                        "original_train_gate": gate_summary,
                        "training_sufficient": sufficient,
                        "original_train_gate_queries": gate_queries,
                        "length_score_queries": len(length_rows),
                        "final_checkpoint_path": str(final_path.resolve()),
                        "final_checkpoint_sha256": final_sha,
                        "final_bundle_fingerprint": final_payload["bundle_fingerprint"],
                        "final_independent_restore": final_restore,
                        "zero_checkpoint_sha256": zero_sha,
                        "zero_fingerprint": fingerprint,
                    }
                )
                del payload
                model = None
                optimizer = None
                active = None

        progress.flush()
        if (
            update_count != TOTAL_UPDATES
            or score_count != TRAIN_SCORE_QUERIES
            or len(final_records) != len(SEEDS) * len(ARMS)
        ):
            raise Refuse("N1-165 total update/score/endpoint counts differ")
        if not n162.n150_builder.verify_hashes(source_hashes):
            raise Refuse("N1-165 source hashes changed during training")
        for parent in metadata["parents"].values():
            if (
                n162.sha256(Path(parent["final_checkpoint_path"]))
                != parent["final_checkpoint_sha256"]
            ):
                raise Refuse("N1-165 mutated a frozen N1-162B parent checkpoint")

        candidate_records = final_records
        report = {
            "schema": "taiji-n1-plan165-transition-template-mid-dose-result-v1",
            "status": "complete",
            "complete": True,
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": n162.sha256(READINESS),
            "manifest_sha256": n162.sha256(MANIFEST),
            "n162b_result_sha256": schedule_builder.PRIOR_RESULT_SHA256,
            "f1_evidence_sha256": schedule_builder.F1_RESULT_SHA256,
            "g2_evidence_sha256": schedule_builder.G2_RESULT_SHA256,
            "n163_anchor_result_sha256": schedule_builder.N163_RESULT_SHA256,
            "n164_attribution_sha256": schedule_builder.N164_REPORT_SHA256,
            "n163_dose_anchors": metadata["n163_dose_anchors"],
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "updates_per_model": UPDATES_PER_MODEL,
            "total_updates": update_count,
            "train_score_queries": score_count,
            "gate_queries_per_endpoint": GATE_QUERIES_PER_ENDPOINT,
            "length_queries_per_endpoint": LENGTH_QUERIES_PER_ENDPOINT,
            "candidate_sufficient_all_seeds": len(candidate_records) == 3
            and all(row["training_sufficient"] for row in candidate_records),
            "mid_dose_sufficient_all_seeds": len(candidate_records) == len(SEEDS)
            and all(row["training_sufficient"] for row in candidate_records),
            "training_sufficient_all_endpoints": all(
                row["training_sufficient"] for row in final_records
            ),
            "fresh_face_queries": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
            "confirmation_queries": 0,
            "parameter_updates": update_count,
            "checkpoint_writes": len(SEEDS) * len(ARMS) * 2,
            "zero_checkpoint_count": len(SEEDS) * len(ARMS),
            "final_checkpoint_count": len(final_records),
            "zero_independent_restores": zero_preflight["independent_restore_count"],
            "final_independent_restores": sum(
                int(row["final_independent_restore"]["complete"]) for row in final_records
            ),
            "stage_cpu_seconds": round(budget.used() + readiness_cpu + validation_cpu, 6),
            "execution_cpu_seconds": round(budget.used(), 6),
            "cpu_limit_seconds": STAGE_CPU_LIMIT_SECONDS,
            "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
            "cumulative_linked_cpu_seconds": round(
                PRIOR_LINKED_CPU_SECONDS + readiness_cpu + validation_cpu + budget.used(), 6
            ),
            "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
            "source_hashes": dict(sorted(source_hashes.items())),
            "source_hashes_unchanged": True,
            "models": final_records,
            "errors": errors,
            "can_promote": False,
        }
        write_new(RESULT, report)
        return report
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        if active is not None and model is not None and optimizer is not None:
            try:
                partial_path = CHECKPOINT_ROOT / f"s{active['seed']}_{active['arm']}_partial.pt"
                partial_payload = _save_checkpoint(
                    model,
                    optimizer,
                    partial_path,
                    seed=int(active["seed"]),
                    arm=str(active["arm"]),
                    parent_path=Path(active["parent_path"]),
                    parent_hash=str(active["parent_hash"]),
                    additional_updates=int(active["completed_updates"]),
                )
                partial_checkpoint = {
                    "path": str(partial_path.resolve()),
                    "sha256": n162.sha256(partial_path),
                    "completed_updates": active["completed_updates"],
                    "bundle_fingerprint": partial_payload["bundle_fingerprint"],
                }
            except Exception as save_exc:
                errors.append(f"partial checkpoint save: {type(save_exc).__name__}: {save_exc}")
        failure = {
            "schema": "taiji-n1-plan165-transition-template-mid-dose-result-v1",
            "status": "partial" if update_count else "failed_before_training",
            "complete": False,
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": n162.sha256(READINESS) if READINESS.is_file() else None,
            "manifest_sha256": n162.sha256(MANIFEST) if MANIFEST.is_file() else None,
            "training_updates": update_count,
            "train_score_queries": score_count,
            "completed_models": final_records,
            "active_model": active,
            "partial_checkpoint": partial_checkpoint,
            "cpu_seconds": round(budget.used(), 6),
            "cpu_limit_seconds": budget.limit,
            "errors": errors,
            "checkpoint_writes": (
                sum(1 for p in CHECKPOINT_ROOT.glob("*.pt") if p.is_file())
                if CHECKPOINT_ROOT.is_dir()
                else 0
            ),
            "can_promote": False,
        }
        if not RESULT.exists():
            write_new(RESULT, failure)
        raise
    finally:
        try:
            progress.flush()
            progress.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker is not None:
        try:
            print(
                json.dumps(_worker_restore(args.restore_worker), ensure_ascii=False, sort_keys=True)
            )
            return 0
        except Exception as exc:
            print(f"N1-165 restore refused: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    try:
        report = execute()
    except Exception as exc:
        print(f"N1-165 refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "total_updates",
                    "train_score_queries",
                    "stage_cpu_seconds",
                    "candidate_sufficient_all_seeds",
                    "mid_dose_sufficient_all_seeds",
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
