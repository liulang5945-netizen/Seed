"""构造PLAN-N1-162配对问句schedule及seed132未见句式诊断面。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_plan153_v28_fresh_context_length_face as n153
from scripts.training import build_taiji_n1_plan161_seed131_question_surface_face as n161
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import (
    run_taiji_n1_plan155_short_unknown_context_length_training as n155_runner,
)
from scripts.training import seal_ast  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-162_question_surface_matched_continuation_20261020.md"
PLAN_SHA256 = "50b7381ceb4cd8424a4314737e1e3f9d7288721e889f702fb4d0784778b3a901"
BUDGET_ADDENDUM = ROOT / "plans/reference/PLAN-N1-162A_执行预算可行性修订_20261020.md"
BUDGET_ADDENDUM_SHA256 = "f3dfee49db86f0d92e9c3bb1c016065c696cb51d4387901c9d0e95f8a0ec6972"
N155A_RESULT = n155_runner.RESULT
N155A_RESULT_SHA256 = "e577d4dda753d016a46e11ab2e0202b2845206af107c54f87aff7f9fdf1ebb11"
N161_RESULT = n161.RESULT
N161_RESULT_SHA256 = "2808a969299143c692643bf4bd5ccabdd67a63021f5ad9ba0cab65bcb51a385a"

SOURCE_AUDIT = ROOT / "reports/taiji_n1_plan162_question_surface_source_audit_20261020.json"
SCHEDULE = ROOT / "plans/manifests/n1_plan162_v34_question_surface_schedule_20261020.json"
READINESS = ROOT / "reports/taiji_n1_plan162_question_surface_readiness_20261020.json"
READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan162_question_surface_readiness_attempt_20261020.json"
)
SOURCE_BINDING = ROOT / "reports/taiji_n1_plan162_question_surface_source_binding_20261020.json"
FACE = ROOT / "plans/manifests/n1_plan162_v34_question_surface_seed132_20261020.json"
FACE_AUDIT = ROOT / "reports/taiji_n1_plan162_question_surface_face_audit_20261020.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_plan162_question_surface_face_checks_20261020.json"
CHECKPOINT_ROOT = ROOT / "data/checkpoints/n1_plan162_question_surface_20261020"
ATTEMPT = ROOT / "reports/taiji_n1_plan162_question_surface_attempt_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan162_question_surface_progress_20261020.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan162_question_surface_result_20261020.json"
ZERO_PREFLIGHT = ROOT / "reports/taiji_n1_plan162_question_surface_zero_preflight_20261020.json"
ZERO_PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan162_question_surface_zero_preflight_attempt_20261020.json"
)
SCORING_PREFLIGHT = (
    ROOT / "reports/taiji_n1_plan162_question_surface_scoring_preflight_20261020.json"
)
TEST = ROOT / "tests/taiji_native/test_n1_plan162_question_surface_matched_continuation_contract.py"
RUNNER = ROOT / "scripts/training/run_taiji_n1_plan162_question_surface_matched_continuation.py"

SEEDS = (11, 12, 13)
ARMS = ("surface_control", "surface_augmented")
ENDPOINTS = tuple((seed, arm) for seed in SEEDS for arm in ARMS)
KINDS = ("lookup", "revision", "transition", "unknown")
LENGTHS = (8, 16, 32)
UPDATES_PER_MODEL = 9_216
UPDATES_PER_PARENT = 48
TOTAL_UPDATES = len(ENDPOINTS) * UPDATES_PER_MODEL
TRAIN_GATE_QUERIES_PER_ENDPOINT = 960
LENGTH_QUERIES_PER_ENDPOINT = 288
TRAIN_SCORE_QUERIES_PER_ENDPOINT = TRAIN_GATE_QUERIES_PER_ENDPOINT + LENGTH_QUERIES_PER_ENDPOINT
TRAIN_SCORE_QUERIES = len(ENDPOINTS) * TRAIN_SCORE_QUERIES_PER_ENDPOINT
ITEMS_PER_KIND = 40
FACE_QUERIES_PER_ENDPOINT = ITEMS_PER_KIND * len(KINDS) * len(LENGTHS)
SCORING_ENDPOINTS = 12
FACE_QUERIES = SCORING_ENDPOINTS * FACE_QUERIES_PER_ENDPOINT
PRIOR_LINKED_CPU_SECONDS = 11_058.34375
CPU_LIMIT_SECONDS = 12_000.0
LINKED_CPU_LIMIT_SECONDS = 24_000.0
FACE_SEED = 132

TRAINING_TEMPLATES = {
    "material_active": "what material does {entity} use?",
    "material_passive": "which material is used by {entity}?",
    "direction_after_turning": "which direction does {entity} face after turning?",
    "direction_after_turn": "what direction is {entity} facing after its turn?",
}
HOLDOUT_TEMPLATES = {
    "material_associated": "what material is associated with {entity}?",
    "material_belongs": "which material belongs to {entity}?",
    "direction_after_turning": "after turning, what direction does {entity} face?",
    "direction_final": "what is the final direction for {entity}?",
}
MATERIAL_KINDS = frozenset(("lookup", "revision", "unknown"))


class Refuse(RuntimeError):
    """PLAN-N1-162来源、schedule或face合同错误。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def digest_strings(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _merge_hashes(target: dict[str, str], incoming: dict[str, str]) -> None:
    for raw_path, digest in incoming.items():
        path = str(Path(raw_path).resolve())
        previous = target.get(path)
        if (
            previous is not None
            and previous != digest
            and not seal_ast.recorded_value_attested(path, previous)
        ):
            raise Refuse(f"conflicting PLAN-N1-162 source hash: {path}")
        target[path] = digest


def _template_pair(kind: str) -> tuple[str, str]:
    if kind in MATERIAL_KINDS:
        return "material_active", "material_passive"
    if kind == "transition":
        return "direction_after_turning", "direction_after_turn"
    raise Refuse(f"unsupported training kind: {kind}")


def _expected_length_counts(kind: str) -> dict[int, int]:
    return {8: 24, 16: 12, 32: 12} if kind == "unknown" else {8: 16, 16: 16, 32: 16}


def build_schedule(base_schedule: dict) -> dict:
    """Bind per-slot paired surface assignments without changing source order."""
    slots = base_schedule.get("slots", [])
    base_rows = base_schedule.get("rows", [])
    require(
        len(slots) == len(base_rows) == UPDATES_PER_MODEL,
        "N1-162 requires the frozen 9,216 parent slots",
    )
    rows = []
    parent_kinds: dict[str, str] = {}
    per_parent = Counter()
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for index, base in enumerate(base_rows):
        slot = slots[index]
        step = int(base["step"])
        require(step == index == int(slot["step"]), "N1-162 source slot order/step changed")
        parent = str(base["parent_id"])
        kind = str(base["kind"])
        length = int(base["candidate_length"])
        require(
            slot["item"].get("split") == "train" and slot["item"].get("kind") == kind,
            "N1-162 source item changed",
        )
        require(
            slot["item"].get("id") == base["item_id"]
            and slot.get("second_id") == base.get("second_id"),
            "N1-162 source item identity changed",
        )
        if parent in parent_kinds:
            require(
                parent_kinds[parent] == kind, f"N1-162 parent has multiple task kinds: {parent}"
            )
        parent_kinds[parent] = kind
        per_parent[parent] += 1
        groups[(parent, kind, length)].append(base)

    treatment_templates: dict[int, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    assignments: dict[int, str] = {}
    for (parent, kind, length), group in groups.items():
        group.sort(key=lambda row: int(row["step"]))
        expected = _expected_length_counts(kind)[length]
        require(
            len(group) == expected, f"N1-162 per-parent context length differs: {parent}/{length}"
        )
        first, second = _template_pair(kind)
        half = len(group) // 2
        quarter = len(group) // 4
        require(
            half + quarter * 2 == len(group), "N1-162 template slot count must divide into 2:1:1"
        )
        labels = ["original"] * half + [first] * quarter + [second] * quarter
        for base, template in zip(group, labels, strict=True):
            assignments[int(base["step"])] = template
            treatment_templates[length][kind][template] += 1

    for index, base in enumerate(base_rows):
        kind = str(base["kind"])
        length = int(base["candidate_length"])
        treatment_template = assignments[index]
        rows.append(
            {
                "step": index,
                "parent_id": str(base["parent_id"]),
                "route": str(base["route"]),
                "route_ordinal": int(base["route_ordinal"]),
                "kind": kind,
                "context_length": length,
                "item_id": str(base["item_id"]),
                "item_key": str(base["item_key"]),
                "second_id": base.get("second_id"),
                "second_key": base.get("second_key"),
                "control_template_id": "original",
                "augmented_template_id": treatment_template,
            }
        )
    require(
        len(parent_kinds) == 192 and set(per_parent.values()) == {48},
        "N1-162 parent update balance differs",
    )
    route_counts = Counter((row["parent_id"], row["route"]) for row in rows)
    require(set(route_counts.values()) == {24}, "N1-162 raw/hard source slot balance differs")
    kind_counts = Counter(row["kind"] for row in rows)
    require(
        kind_counts == Counter({kind: 2_304 for kind in KINDS}),
        "N1-162 task-kind update balance differs",
    )
    per_parent_augmented = {
        "lookup": {
            "8": {"original": 8, "material_active": 4, "material_passive": 4},
            "16": {"original": 8, "material_active": 4, "material_passive": 4},
            "32": {"original": 8, "material_active": 4, "material_passive": 4},
        },
        "revision": {
            "8": {"original": 8, "material_active": 4, "material_passive": 4},
            "16": {"original": 8, "material_active": 4, "material_passive": 4},
            "32": {"original": 8, "material_active": 4, "material_passive": 4},
        },
        "transition": {
            "8": {"original": 8, "direction_after_turning": 4, "direction_after_turn": 4},
            "16": {"original": 8, "direction_after_turning": 4, "direction_after_turn": 4},
            "32": {"original": 8, "direction_after_turning": 4, "direction_after_turn": 4},
        },
        "unknown": {
            "8": {"original": 12, "material_active": 6, "material_passive": 6},
            "16": {"original": 6, "material_active": 3, "material_passive": 3},
            "32": {"original": 6, "material_active": 3, "material_passive": 3},
        },
    }
    expected_augmented = {
        kind: {
            str(length): {template: count * 48 for template, count in values.items()}
            for length, values in by_length.items()
        }
        for kind, by_length in per_parent_augmented.items()
    }
    actual_augmented = {
        kind: {
            str(length): dict(sorted(treatment_templates[length][kind].items()))
            for length in LENGTHS
        }
        for kind in KINDS
    }
    require(
        actual_augmented == expected_augmented,
        "N1-162 per-kind/length question-surface ratios differ",
    )
    manifest = {
        "schema": "n1-plan162-v34-question-surface-schedule-v1",
        "plan_sha256": PLAN_SHA256,
        "budget_addendum_sha256": BUDGET_ADDENDUM_SHA256,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "parent_count": len(parent_kinds),
        "updates_per_model": UPDATES_PER_MODEL,
        "updates_per_parent": UPDATES_PER_PARENT,
        "total_updates": TOTAL_UPDATES,
        "train_score_queries": TRAIN_SCORE_QUERIES,
        "face_queries": FACE_QUERIES,
        "context_length_counts_per_model": {"8": 3_456, "16": 2_880, "32": 2_880},
        "augmented_template_counts_per_kind_length": expected_augmented,
        "augmented_template_counts_per_parent": per_parent_augmented,
        "schedule_rows": rows,
        "schedule_sha256": canonical_sha(rows),
        "source_schedule_sha256": base_schedule["manifest"]["schedule_sha256"],
        "can_promote": False,
    }
    return {"manifest": manifest, "slots": slots, "rows": rows, "views": base_schedule["views"]}


def validate_schedule(manifest: dict) -> dict:
    rows = manifest.get("schedule_rows", [])
    require(
        manifest.get("schedule_sha256") == canonical_sha(rows), "N1-162 schedule digest mismatch"
    )
    require(len(rows) == UPDATES_PER_MODEL, "N1-162 schedule row count differs")
    require(
        [int(row["step"]) for row in rows] == list(range(UPDATES_PER_MODEL)),
        "N1-162 schedule order differs",
    )
    parent_counts = Counter(row["parent_id"] for row in rows)
    route_counts = Counter((row["parent_id"], row["route"]) for row in rows)
    require(
        len(parent_counts) == 192 and set(parent_counts.values()) == {48},
        "N1-162 parent balance differs",
    )
    require(set(route_counts.values()) == {24}, "N1-162 route balance differs")
    require(
        manifest.get("total_updates") == TOTAL_UPDATES
        and manifest.get("train_score_queries") == TRAIN_SCORE_QUERIES,
        "N1-162 frozen denominator differs",
    )
    return {
        "accepted": True,
        "rows": len(rows),
        "parents": len(parent_counts),
        "routes": len(route_counts),
    }


def _source_context() -> tuple[dict, dict, dict[str, str], dict]:
    require(sha256(PLAN) == PLAN_SHA256, "PLAN-N1-162 plan SHA changed")
    require(
        sha256(BUDGET_ADDENDUM) == BUDGET_ADDENDUM_SHA256,
        "PLAN-N1-162A budget addendum SHA changed",
    )
    n155_readiness, n155_preflight, n155_context, n155_bound, n155_schedule, n155_sources = (
        n155_runner._verify_preflight()
    )
    n155a = read_json(N155A_RESULT)
    require(
        sha256(N155A_RESULT) == N155A_RESULT_SHA256
        and n155a.get("complete") is True
        and n155a.get("training_updates") == 55_296
        and n155a.get("train_score_queries") == 7_488
        and n155a.get("training_complete") is True
        and n155a.get("source_hashes_unchanged") is True
        and n155a.get("can_promote") is False,
        "N1-155A candidate source result is incomplete or changed",
    )
    candidate_rows = {
        int(row["seed"]): row
        for row in n155a.get("models", [])
        if row.get("branch") == "unknown8_rebalanced" and row.get("training_sufficient") is True
    }
    require(
        set(candidate_rows) == set(SEEDS),
        "N1-162 requires all three sufficient N1-155A candidate bundles",
    )
    require(
        n150_builder.verify_hashes(n155a.get("source_hashes", {})),
        "N1-155A source/checkpoint hashes changed",
    )
    n161_result = read_json(N161_RESULT)
    require(
        sha256(N161_RESULT) == N161_RESULT_SHA256
        and n161_result.get("complete") is True
        and n161_result.get("fresh_queries") == 2_880
        and n161_result.get("source_hashes_unchanged") is True
        and n161_result.get("can_promote") is False,
        "N1-161 completed question-surface source evidence is invalid",
    )
    n161_endpoints, n161_sources, exclusions = n161.source_context()
    require(len(n161_endpoints) == 6, "N1-162 requires all frozen N1-161 source endpoints")
    sources: dict[str, str] = {}
    _merge_hashes(sources, n155_sources)
    _merge_hashes(sources, n155a.get("source_hashes", {}))
    _merge_hashes(sources, n161_sources)
    _merge_hashes(sources, n161_result.get("source_hashes", {}))
    for path in (
        PLAN,
        BUDGET_ADDENDUM,
        N155A_RESULT,
        N161_RESULT,
        Path(__file__).resolve(),
        RUNNER,
        TEST,
    ):
        require(Path(path).is_file(), f"N1-162 source file missing: {path}")
        _merge_hashes(sources, {str(Path(path).resolve()): sha256(Path(path))})
    for seed, row in candidate_rows.items():
        checkpoint = Path(row["final_checkpoint_path"]).resolve()
        require(
            checkpoint.is_file() and sha256(checkpoint) == row["final_checkpoint_sha256"],
            f"N1-155A seed{seed} candidate checkpoint changed",
        )
        _merge_hashes(sources, {str(checkpoint): row["final_checkpoint_sha256"]})
    require(n150_builder.verify_hashes(sources), "N1-162 bound source/checkpoint SHA mismatch")
    runtime = {
        "n155_readiness": n155_readiness,
        "n155_preflight": n155_preflight,
        "n155_context": n155_context,
        "n155_bound": n155_bound,
        "n155_schedule": n155_schedule,
        "n155a_result": n155a,
        "n155a_candidates": candidate_rows,
        "n161_result": n161_result,
        "n161_exclusions": exclusions,
    }
    return runtime, dict(sorted(sources.items())), exclusions, candidate_rows


def source_readiness() -> dict:
    outputs = (
        SOURCE_AUDIT,
        SCHEDULE,
        READINESS,
        READINESS_ATTEMPT,
        SOURCE_BINDING,
        ZERO_PREFLIGHT,
        ZERO_PREFLIGHT_ATTEMPT,
        SCORING_PREFLIGHT,
        ATTEMPT,
        PROGRESS,
        RESULT,
        FACE,
        FACE_AUDIT,
        FACE_CHECKS,
    )
    if any(path.exists() for path in outputs) or CHECKPOINT_ROOT.exists():
        raise FileExistsError("PLAN-N1-162 one-shot outputs/checkpoint root already exist")
    started = time.process_time()
    runtime, sources, exclusions, candidates = _source_context()
    schedule = build_schedule(runtime["n155_schedule"])
    validation = validate_schedule(schedule["manifest"])
    readiness_attempt = {
        "schema": "n1-plan162-question-surface-readiness-attempt-v1",
        "status": "started",
        "plan_sha256": PLAN_SHA256,
        "budget_addendum_sha256": BUDGET_ADDENDUM_SHA256,
        "source_hash_count": len(sources),
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "can_promote": False,
    }
    write_new(READINESS_ATTEMPT, readiness_attempt)
    source_hashes = dict(sources)
    source_hashes[str(READINESS_ATTEMPT.resolve())] = sha256(READINESS_ATTEMPT)
    if not n150_builder.verify_hashes(sources):
        raise Refuse("N1-162 source hashes changed during readiness")
    write_new(
        SOURCE_AUDIT,
        {
            "schema": "n1-plan162-question-surface-source-audit-v1",
            "status": "complete",
            "plan_sha256": PLAN_SHA256,
            "budget_addendum_sha256": BUDGET_ADDENDUM_SHA256,
            "n155a_candidate_endpoints": [
                {
                    "seed": seed,
                    "checkpoint_path": row["final_checkpoint_path"],
                    "checkpoint_sha256": row["final_checkpoint_sha256"],
                    "training_sufficient": row["training_sufficient"],
                }
                for seed, row in sorted(candidates.items())
            ],
            "n161_face_sha256": sha256(n161.FACE),
            "source_hash_count": len(sources),
            "source_hashes": sources,
            "source_hashes_unchanged": True,
            "excluded_entity_count": len(exclusions["excluded_entities"]),
            "excluded_word_value_count": len(exclusions["excluded_words"]),
            "excluded_entity_sha256": digest_strings(exclusions["excluded_entities"]),
            "excluded_word_value_sha256": digest_strings(exclusions["excluded_words"]),
            "can_promote": False,
        },
    )
    manifest = dict(schedule["manifest"], source_audit_sha256=sha256(SOURCE_AUDIT))
    write_new(SCHEDULE, manifest)
    source_hashes[str(SOURCE_AUDIT.resolve())] = sha256(SOURCE_AUDIT)
    source_hashes[str(SCHEDULE.resolve())] = sha256(SCHEDULE)
    source_hashes[str(READINESS_ATTEMPT.resolve())] = sha256(READINESS_ATTEMPT)
    cpu = time.process_time() - started
    readiness = {
        "schema": "n1-plan162-question-surface-readiness-v1",
        "status": "ready_for_zero_checkpoint_preflight",
        "complete": validation["accepted"] and n150_builder.verify_hashes(sources),
        "plan_sha256": PLAN_SHA256,
        "budget_addendum_sha256": BUDGET_ADDENDUM_SHA256,
        "schedule_sha256": sha256(SCHEDULE),
        "schedule_identity_sha256": manifest["schedule_sha256"],
        "source_audit_sha256": sha256(SOURCE_AUDIT),
        "source_hashes": sources,
        "source_hashes_unchanged": n150_builder.verify_hashes(sources),
        "candidate_endpoint_count": len(candidates),
        "schedule_validation": validation,
        "training_updates_planned": TOTAL_UPDATES,
        "train_score_queries_planned": TRAIN_SCORE_QUERIES,
        "fresh_face_queries_planned": FACE_QUERIES,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_prior_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "can_promote": False,
    }
    write_new(READINESS, readiness)
    write_new(
        SOURCE_BINDING,
        {
            "schema": "n1-plan162-question-surface-source-binding-v1",
            "status": "complete" if readiness["complete"] else "stopped",
            "plan_sha256": PLAN_SHA256,
            "budget_addendum_sha256": BUDGET_ADDENDUM_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_audit_sha256": sha256(SOURCE_AUDIT),
            "schedule_sha256": sha256(SCHEDULE),
            "source_hashes": sources,
            "can_promote": False,
        },
    )
    return readiness


def _verify_readiness() -> tuple[dict, dict, dict, dict[str, str]]:
    required = (SOURCE_AUDIT, SCHEDULE, READINESS, READINESS_ATTEMPT, SOURCE_BINDING)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("PLAN-N1-162 source readiness is required")
    audit, manifest, readiness, attempt, binding = (read_json(path) for path in required)
    if (
        sha256(PLAN) != PLAN_SHA256
        or sha256(BUDGET_ADDENDUM) != BUDGET_ADDENDUM_SHA256
        or readiness.get("complete") is not True
        or readiness.get("status") != "ready_for_zero_checkpoint_preflight"
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("budget_addendum_sha256") != BUDGET_ADDENDUM_SHA256
        or not seal_ast.receipt_binding_matches(readiness.get("schedule_sha256"), SCHEDULE)
        or not seal_ast.receipt_binding_matches(readiness.get("source_audit_sha256"), SOURCE_AUDIT)
        or binding.get("status") != "complete"
        or not seal_ast.receipt_binding_matches(binding.get("readiness_sha256"), READINESS)
        or binding.get("source_hashes") != readiness.get("source_hashes")
        or audit.get("status") != "complete"
        or audit.get("source_hashes") != readiness.get("source_hashes")
        or attempt.get("plan_sha256") != PLAN_SHA256
        or not seal_ast.receipt_binding_matches(manifest.get("source_audit_sha256"), SOURCE_AUDIT)
    ):
        raise Refuse("PLAN-N1-162 readiness binding is stale")
    runtime, sources, _exclusions, _candidates = _source_context()
    schedule = build_schedule(runtime["n155_schedule"])
    expected = dict(schedule["manifest"], source_audit_sha256=sha256(SOURCE_AUDIT))
    if expected != manifest or expected["schedule_sha256"] != readiness.get(
        "schedule_identity_sha256"
    ):
        raise Refuse("PLAN-N1-162 fixed schedule changed")
    if sources != readiness.get("source_hashes") or not n150_builder.verify_hashes(sources):
        raise Refuse("PLAN-N1-162 source union/hash changed")
    return readiness, manifest, runtime, dict(sources)


def _face_template_id(kind: str, index: int) -> str:
    if kind in MATERIAL_KINDS:
        return "material_associated" if index % 2 == 0 else "material_belongs"
    if kind == "transition":
        return "direction_after_turning" if index % 2 == 0 else "direction_final"
    raise Refuse(f"unsupported face kind: {kind}")


def _add_seed131_values(item: dict, entities: set[str], words: set[str]) -> None:
    entities.update(str(value) for value in item.get("entities", []))
    answer = item.get("answer")
    if answer not in (None, "unknown", "north", "east", "south", "west", "left", "right"):
        words.add(str(answer))
    for row in item.get("observations_by_length", {}).get("32", []):
        n153._add_row_values(row, entities, words)


def _validate_face(manifest: dict, excluded_entities: set[str], excluded_words: set[str]) -> dict:
    items = manifest.get("items", [])
    require(
        manifest.get("schema") == "n1-plan162-v34-question-surface-face-v1",
        "N1-162 face schema differs",
    )
    require(
        manifest.get("seed") == FACE_SEED and len(items) == ITEMS_PER_KIND * len(KINDS),
        "N1-162 face seed/size differs",
    )
    kind_counts = Counter(item.get("kind") for item in items)
    require(
        kind_counts == Counter({kind: ITEMS_PER_KIND for kind in KINDS}),
        "N1-162 face kind balance differs",
    )
    template_counts = Counter()
    all_entities = []
    all_words = []
    item_ids = set()
    for item in items:
        kind = str(item["kind"])
        item_id = str(item["id"])
        target = str(item["entities"][0])
        index = int(item_id.rsplit("-", 1)[1])
        template_id = _face_template_id(kind, index)
        require(
            item_id not in item_ids and item.get("split") == "diagnostic",
            f"invalid/duplicate N1-162 face item: {item_id}",
        )
        item_ids.add(item_id)
        require(
            item.get("question_template_id") == template_id,
            f"N1-162 question template assignment differs: {item_id}",
        )
        require(
            item.get("question") == HOLDOUT_TEMPLATES[template_id].format(entity=target),
            f"N1-162 question bytes differ: {item_id}",
        )
        template_counts[(kind, template_id)] += 1
        views = item.get("observations_by_length", {})
        rows8, rows16, rows32 = (views.get(str(length), []) for length in LENGTHS)
        require(
            len(rows8) == 8 and len(rows16) == 16 and len(rows32) == 32,
            f"N1-162 context length differs: {item_id}",
        )
        require(rows8 == rows16[:8] == rows32[:8], f"N1-162 contexts are not nested: {item_id}")
        target_indices = item.get("target_observation_indices", [])
        expected_indices = {"lookup": [0], "revision": [0, 1], "transition": [0, 1], "unknown": []}[
            kind
        ]
        require(
            target_indices == expected_indices, f"N1-162 target evidence indices differ: {item_id}"
        )
        if target_indices:
            require(
                rows32[: len(target_indices)] == [rows32[index] for index in target_indices],
                f"N1-162 target facts are not fixed at the prefix: {item_id}",
            )
        all_entities.extend(str(value) for value in item.get("entities", []))
        for row in rows32:
            parts = row.removesuffix(".").split()
            require(len(parts) >= 3, f"N1-162 observation row is malformed: {item_id}")
            if len(parts) == 3 and parts[1] == "uses":
                all_words.append(parts[2])
            elif len(parts) == 4 and parts[1:3] == ["now", "uses"]:
                all_words.append(parts[3])
            elif not (len(parts) == 3 and parts[1] in ("faces", "turns")):
                raise Refuse(f"N1-162 observation row is unsupported: {row}")
    require(len(all_entities) == len(set(all_entities)), "N1-162 face entity values are not unique")
    require(
        len(all_words) == len(set(all_words)), "N1-162 face material/answer values are not unique"
    )
    require(not (set(all_entities) & excluded_entities), "N1-162 face overlaps excluded entities")
    require(
        not (set(all_words) & excluded_words),
        "N1-162 face overlaps excluded answer/observation values",
    )
    expected_templates = {
        (kind, template): 20
        for kind in KINDS
        for template in (
            ("material_associated", "material_belongs")
            if kind in MATERIAL_KINDS
            else ("direction_after_turning", "direction_final")
        )
    }
    require(template_counts == expected_templates, "N1-162 face template balance differs")
    return {
        "accepted": True,
        "items": len(items),
        "items_by_kind": dict(sorted(kind_counts.items())),
        "template_counts": {
            f"{kind}/{template}": count
            for (kind, template), count in sorted(template_counts.items())
        },
        "queries_per_endpoint": len(items) * len(LENGTHS),
        "total_queries": len(items) * len(LENGTHS) * SCORING_ENDPOINTS,
        "nested_contexts": True,
        "target_evidence_prefix": True,
        "source_overlap": False,
        "can_promote": False,
    }


def prepare_fresh_face(
    *, final_checkpoint_hashes: dict[str, str], training_progress_sha256: str
) -> dict:
    if any(path.exists() for path in (FACE, FACE_AUDIT, FACE_CHECKS)):
        raise FileExistsError("N1-162 seed132 face outputs are write-once")
    require(
        RESULT.is_file() and PROGRESS.is_file() and sha256(PROGRESS) == training_progress_sha256,
        "N1-162 completed progress is required before face generation",
    )
    result = read_json(RESULT)
    require(
        result.get("training_complete") is True
        and result.get("training_sufficient_all_endpoints") is True,
        "N1-162 all six training gates must pass before face generation",
    )
    require(
        len(final_checkpoint_hashes) == 6, "N1-162 face requires all six final checkpoint hashes"
    )
    for raw_path, digest in final_checkpoint_hashes.items():
        require(
            Path(raw_path).is_file() and sha256(Path(raw_path)) == digest,
            f"N1-162 final checkpoint changed: {raw_path}",
        )
    started = time.process_time()
    _runtime, sources, exclusions, _candidates = _source_context()
    readiness = read_json(READINESS)
    require(
        sources == readiness.get("source_hashes"),
        "N1-162 source union changed before face generation",
    )
    excluded_entities = set(exclusions["excluded_entities"])
    excluded_words = set(exclusions["excluded_words"])
    prior_face = read_json(n161.FACE)
    for item in prior_face.get("items", []):
        _add_seed131_values(item, excluded_entities, excluded_words)
    exclusion_summary = {
        "prior_question_surface_face_sha256": sha256(n161.FACE),
        "excluded_entity_count": len(excluded_entities),
        "excluded_word_value_count": len(excluded_words),
        "excluded_entity_sha256": digest_strings(excluded_entities),
        "excluded_word_value_sha256": digest_strings(excluded_words),
    }
    face, generator = n153.build_face(excluded_entities, excluded_words, seed=FACE_SEED)
    per_template = Counter()
    for item in face["items"]:
        original_id = str(item["id"])
        index = int(original_id.rsplit("-", 1)[1])
        kind = str(item["kind"])
        target = str(item["entities"][0])
        template_id = _face_template_id(kind, index)
        item["id"] = original_id.replace("v28-", "v34-", 1)
        item["parent_id"] = item["id"]
        item["question_template_id"] = template_id
        item["question"] = HOLDOUT_TEMPLATES[template_id].format(entity=target)
        per_template[(kind, template_id)] += 1
    manifest = {
        **face,
        "schema": "n1-plan162-v34-question-surface-face-v1",
        "plan_sha256": PLAN_SHA256,
        "budget_addendum_sha256": BUDGET_ADDENDUM_SHA256,
        "seed": FACE_SEED,
        "question_templates": HOLDOUT_TEMPLATES,
        "question_template_counts": {
            f"{kind}/{template}": count for (kind, template), count in sorted(per_template.items())
        },
        "source_exclusion_summary": exclusion_summary,
        "prior_seed131_face_sha256": sha256(n161.FACE),
        "endpoint_count": SCORING_ENDPOINTS,
        "total_queries": FACE_QUERIES,
        "can_promote": False,
    }
    checks = _validate_face(manifest, excluded_entities, excluded_words)
    question_patterns = {
        key: re.compile(
            "^"
            + re.escape(template.format(entity="__ENTITY__")).replace("__ENTITY__", r"[a-z]{10}")
            + "$"
        )
        for key, template in HOLDOUT_TEMPLATES.items()
    }
    training_questions = [
        question for path in n161.TRAIN_MANIFESTS for question in n161._questions(read_json(path))
    ]
    collisions = {
        key: sum(pattern.fullmatch(question) is not None for question in training_questions)
        for key, pattern in question_patterns.items()
    }
    require(
        not any(collisions.values()),
        f"N1-162 holdout templates overlap v3 training questions: {collisions}",
    )
    treatment_templates = set(TRAINING_TEMPLATES.values())
    require(
        not (treatment_templates & set(HOLDOUT_TEMPLATES.values())),
        "N1-162 training and holdout template forms overlap",
    )
    write_new(FACE, manifest)
    source_hashes = dict(sources)
    source_hashes[str(FACE.resolve())] = sha256(FACE)
    source_hashes[str(RESULT.resolve())] = sha256(RESULT)
    source_hashes[str(PROGRESS.resolve())] = training_progress_sha256
    source_hashes.update(
        {str(Path(path).resolve()): digest for path, digest in final_checkpoint_hashes.items()}
    )
    audit = {
        "schema": "n1-plan162-v34-question-surface-face-audit-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "budget_addendum_sha256": BUDGET_ADDENDUM_SHA256,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": n150_builder.verify_hashes(source_hashes),
        "face_sha256": sha256(FACE),
        "face_contract": checks,
        "training_template_collisions": collisions,
        "source_exclusion_summary": exclusion_summary,
        "generator_summary": generator,
        "cpu_seconds": round(time.process_time() - started, 6),
        "can_promote": False,
    }
    write_new(FACE_AUDIT, audit)
    face_checks = {
        "schema": "n1-plan162-v34-question-surface-face-checks-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "face_audit_sha256": sha256(FACE_AUDIT),
        "face_contract": checks,
        "source_hashes_unchanged": audit["source_hashes_unchanged"],
        "model_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "can_promote": False,
    }
    write_new(FACE_CHECKS, face_checks)
    return face_checks


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="PLAN-N1-162 question-surface schedule and seed132 face"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--prepare-fresh-face", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = source_readiness()
    else:
        raise SystemExit(
            "--prepare-fresh-face is called by the approved training runner after all gates pass"
        )
    keys = (
        "status",
        "complete",
        "cpu_seconds",
        "source_hash_count",
        "training_updates_planned",
        "train_score_queries_planned",
        "fresh_face_queries_planned",
        "can_promote",
    )
    print(json.dumps({key: result.get(key) for key in keys}, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
