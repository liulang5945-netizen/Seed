"""构造PLAN-N1-155未知类短上下文再平衡训练schedule。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import analyze_taiji_n1_plan152_length_itemwise_attribution as n152
from scripts.training import build_taiji_n1_plan153_v28_fresh_context_length_face as n153_builder
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_plan151_train_score_ledger_recovery as n151
from scripts.training import (
    run_taiji_n1_plan153_v28_fresh_context_length_confirmation as n153_runner,
)
from scripts.training import run_taiji_n1_v27_length_robust_training_after_n149 as n150
from scripts.training import seal_ast  # noqa: E402

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-155_short_context_unknown_rebalance_and_fresh_diagnostic_20261020.md"
)
PLAN_SHA256 = "6664a7c1e51778f67917a310832dbfa1cbd66033efc3db50fc03c0dc13a9cd82"
CORRECTION_ADDENDUM = (
    ROOT / "plans/reference/PLAN-N1-155A_runtime_views_handoff_correction_20261020.md"
)
CORRECTION_SHA256 = "b475de271c8e276c1212fb4938c4880abd0d332518e5ce85ab9320f5f88c7b30"
PRIOR_FAILED_RESULT = ROOT / "reports/taiji_n1_plan155_unknown8_rebalance_result_20261020.json"
PRIOR_FAILED_ATTEMPT = ROOT / "reports/taiji_n1_plan155_unknown8_rebalance_attempt_20261020.json"
PRIOR_FAILED_PROGRESS = ROOT / "reports/taiji_n1_plan155_unknown8_rebalance_progress_20261020.jsonl"
PRIOR_SOURCE_AUDIT = ROOT / "reports/taiji_n1_plan155_unknown8_rebalance_source_audit_20261020.json"
PRIOR_SCHEDULE = ROOT / "plans/manifests/n1_plan155_unknown8_rebalance_schedule_20261020.json"
PRIOR_READINESS = ROOT / "reports/taiji_n1_plan155_unknown8_rebalance_readiness_20261020.json"
PRIOR_READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan155_unknown8_rebalance_readiness_attempt_20261020.json"
)
PRIOR_SOURCE_BINDING = (
    ROOT / "reports/taiji_n1_plan155_unknown8_rebalance_source_binding_20261020.json"
)
PRIOR_CHECKPOINT_ROOT = ROOT / "data/checkpoints/n1_plan155_v28_unknown8_rebalance_20261020"
PRIOR_PREFLIGHT = ROOT / "reports/taiji_n1_plan155_unknown8_rebalance_preflight_20261020.json"
PRIOR_PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan155_unknown8_rebalance_preflight_attempt_20261020.json"
)
PRIOR_PREFLIGHT_BINDING = (
    ROOT / "reports/taiji_n1_plan155_unknown8_rebalance_preflight_binding_20261020.json"
)
PRIOR_FAILED_RESULT_SHA256 = "86a1b99508a1986818600c70ea4e9202c26ad8f596324bd551d5a9caa038ce07"
PRIOR_FAILED_ATTEMPT_SHA256 = "ace1aa1c2ad75445fb04533a0f87ed86524ca6c66114f9f522704d8e4317e4da"
PRIOR_FAILED_CPU_SECONDS = 130.921875
PRIOR_FAILED_STAGE_CPU_SECONDS = 22.765625
PRIOR_EMPTY_PROGRESS_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
PRIOR_PARTIAL_CHECKPOINT_SHA256 = "21b032076d1ad0e93800062f222085f210c232204c0469e19abde27215a50550"
N150_RESULT = n150.RESULT
N150_RESULT_SHA256 = "e50e4b5aaf244ced4b5ef6232eb08a09206a5957f7d8f85abb1b3d841860ef0b"
N151A_RESULT = n151.RESULT
N151A_RESULT_SHA256 = "7fb5ffa74b44ffa918d42a508c5d527991445367b3ec22457bb0d21a718ab6d6"
N152_RESULT = n152.RESULT
N152_RESULT_SHA256 = "97277a783b6b3e6d62a56d87feeaae9caa2337d411d80bb53f537956981aa0bd"
N153A_RESULT = (
    ROOT / "reports/taiji_n1_plan153a_fresh_context_result_binding_recovery_result_20261020.json"
)
N153A_RESULT_SHA256 = "af8853b4f4677b469164e3691e429476d02887adc5a47f9981ed5b976926d06a"
N154_RESULT = ROOT / "reports/taiji_n1_plan154_fresh_context_itemwise_attribution_20261020.json"
N154_RESULT_SHA256 = "da296992376b27fc785067d03b505968dbbee6987a3edc13d6c90773772f7e0b"
N153_FACE = n153_builder.FACE
N153_FACE_CHECKS = n153_builder.FACE_CHECKS
N153_SOURCE_AUDIT = n153_builder.SOURCE_AUDIT
N153_FAILURE_RESULT = n153_runner.RESULT
N153_FAILURE_ATTEMPT = n153_runner.ATTEMPT

SOURCE_AUDIT = ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_source_audit_20261020.json"
SCHEDULE = ROOT / "plans/manifests/n1_plan155a_unknown8_rebalance_schedule_20261020.json"
READINESS = ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_readiness_20261020.json"
READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_readiness_attempt_20261020.json"
)
SOURCE_BINDING = ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_source_binding_20261020.json"
CHECKPOINT_ROOT = ROOT / "data/checkpoints/n1_plan155a_v28_unknown8_rebalance_20261020"
PREFLIGHT = ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_preflight_20261020.json"
PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_preflight_attempt_20261020.json"
)
PREFLIGHT_BINDING = (
    ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_preflight_binding_20261020.json"
)
ATTEMPT = ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_attempt_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_progress_20261020.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_result_20261020.json"
FACE = ROOT / "plans/manifests/n1_plan155a_v29_short_unknown_length_seed128_20261020.json"
FACE_AUDIT = ROOT / "reports/taiji_n1_plan155a_v29_short_unknown_face_audit_20261020.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_plan155a_v29_short_unknown_face_checks_20261020.json"

SEEDS = n150_builder.SEEDS
BRANCHES = ("unknown8_rebalanced", "uniform_length_replay")
PARENT_BRANCH = "length_mixed"
ENDPOINTS = tuple((seed, branch) for seed in SEEDS for branch in BRANCHES)
LENGTHS = n150_builder.LENGTHS
KINDS = n150_builder.TRAIN_KINDS
UPDATES_PER_MODEL = n150_builder.UPDATES_PER_MODEL
UPDATES_PER_PARENT = n150_builder.UPDATES_PER_PARENT
TOTAL_UPDATES = len(SEEDS) * len(BRANCHES) * UPDATES_PER_MODEL
TRAIN_SCORE_QUERIES_PER_ENDPOINT = 1_248
TRAIN_SCORE_QUERIES = TRAIN_SCORE_QUERIES_PER_ENDPOINT * len(SEEDS) * len(BRANCHES)
FRESH_ITEMS_PER_KIND = 40
FRESH_QUERIES_PER_ENDPOINT = FRESH_ITEMS_PER_KIND * len(KINDS) * len(LENGTHS)
FRESH_QUERIES = FRESH_QUERIES_PER_ENDPOINT * len(SEEDS) * len(BRANCHES)
CPU_LIMIT_SECONDS = 18_000.0
UNKNOWN = "unknown"
DIRECTIONS = ("north", "east", "south", "west")
TURNS = ("left", "right")
ENTITY_LENGTH = 10
UNKNOWN_LENGTH_PATTERN = (8, 16, 8, 32) * 6
UNIFORM_LENGTH_PATTERN = (8, 16, 32) * 8


class Refuse(RuntimeError):
    """N1-155来源、schedule或face合同不成立。"""


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
        old = target.get(path)
        if old is not None and old != digest:
            raise Refuse(f"conflicting N1-155 source hash: {path}")
        target[path] = digest


def _source_context() -> tuple[dict, dict, dict, dict, dict[str, str]]:
    n150_readiness, n150_preflight, _binding, bound, n150_schedule = n150._verify_preflight()
    n150_result = read_json(N150_RESULT)
    n151a = read_json(N151A_RESULT)
    n152_result = read_json(N152_RESULT)
    n153a_result = read_json(N153A_RESULT)
    n154_result = read_json(N154_RESULT)
    n153_failed_result = read_json(N153_FAILURE_RESULT)
    prior_failed_result = read_json(PRIOR_FAILED_RESULT)
    prior_failed_attempt = read_json(PRIOR_FAILED_ATTEMPT)
    prior_readiness = read_json(PRIOR_READINESS)
    prior_source_audit = read_json(PRIOR_SOURCE_AUDIT)
    prior_schedule = read_json(PRIOR_SCHEDULE)
    prior_source_binding = read_json(PRIOR_SOURCE_BINDING)
    prior_preflight = read_json(PRIOR_PREFLIGHT)
    prior_preflight_binding = read_json(PRIOR_PREFLIGHT_BINDING)
    require(sha256(N150_RESULT) == N150_RESULT_SHA256, "N1-150 result SHA changed")
    require(sha256(N151A_RESULT) == N151A_RESULT_SHA256, "N1-151A result SHA changed")
    require(sha256(N152_RESULT) == N152_RESULT_SHA256, "N1-152 result SHA changed")
    require(sha256(N153A_RESULT) == N153A_RESULT_SHA256, "N1-153A result SHA changed")
    require(sha256(N154_RESULT) == N154_RESULT_SHA256, "N1-154 result SHA changed")
    require(
        n153_failed_result.get("status") == "stopped"
        and n153_failed_result.get("query_count") == 480
        and len(n153_failed_result.get("diagnostic_rows", [])) == 0
        and n153_failed_result.get("source_hashes_unchanged") is True
        and n153_failed_result.get("can_promote") is False,
        "N1-153 failed attempt receipt changed",
    )
    require(
        n150_result.get("complete") is True
        and n150_result.get("training_updates") == 55_296
        and n150_result.get("train_score_queries") == 7_488
        and n150_result.get("training_sufficient_all_endpoints") is True
        and n150_result.get("can_promote") is False
        and seal_ast.receipt_binding_matches(n150_result.get("progress_sha256"), n150.PROGRESS)
        and n151a.get("complete") is True
        and n151a.get("source_hashes_unchanged") is True
        and n152_result.get("complete") is True
        and n152_result.get("coverage", {}).get("rows") == 1_728
        and n152_result.get("can_promote") is False,
        "N1-150/151A/152 frozen source evidence is incomplete",
    )
    require(
        n153a_result.get("complete") is True
        and n153a_result.get("new_query_count") == 2_880
        and n153a_result.get("source_hashes_unchanged") is True,
        "N1-153A fresh result is incomplete",
    )
    require(
        n154_result.get("complete") is True
        and n154_result.get("coverage", {}).get("rows") == 2_880
        and n154_result.get("model_queries") == 0,
        "N1-154 itemwise attribution is incomplete",
    )
    require(
        sha256(CORRECTION_ADDENDUM) == CORRECTION_SHA256
        and sha256(PRIOR_FAILED_RESULT) == PRIOR_FAILED_RESULT_SHA256
        and sha256(PRIOR_FAILED_ATTEMPT) == PRIOR_FAILED_ATTEMPT_SHA256,
        "N1-155A correction or prior failure receipt SHA changed",
    )
    previous_partial = prior_failed_result.get("partial_checkpoint", {})
    previous_zero_rows = prior_preflight.get("zero_checkpoints", [])
    previous_paths = (
        PRIOR_FAILED_RESULT,
        PRIOR_FAILED_ATTEMPT,
        PRIOR_FAILED_PROGRESS,
        PRIOR_SOURCE_AUDIT,
        PRIOR_SCHEDULE,
        PRIOR_READINESS,
        PRIOR_READINESS_ATTEMPT,
        PRIOR_SOURCE_BINDING,
        PRIOR_PREFLIGHT,
        PRIOR_PREFLIGHT_ATTEMPT,
        PRIOR_PREFLIGHT_BINDING,
    )
    require(
        prior_failed_result.get("status") == "stopped"
        and prior_failed_result.get("complete") is False
        and prior_failed_result.get("training_updates") == 0
        and prior_failed_result.get("parameter_updates") == 0
        and prior_failed_result.get("train_score_queries") == 0
        and prior_failed_result.get("dev_queries") == 0
        and prior_failed_result.get("holdout_queries") == 0
        and prior_failed_result.get("confirmation_queries") == 0
        and prior_failed_result.get("fresh_diagnostic_queries") == 0
        and prior_failed_result.get("errors") == ["KeyError: 'views'"]
        and prior_failed_result.get("source_hashes_unchanged") is True
        and prior_failed_result.get("plan_sha256") == PLAN_SHA256
        and seal_ast.receipt_binding_matches(
            prior_failed_result.get("attempt_sha256"), PRIOR_FAILED_ATTEMPT
        )
        and seal_ast.receipt_binding_matches(
            prior_failed_result.get("preflight_sha256"), PRIOR_PREFLIGHT
        )
        and prior_failed_result.get("cpu_seconds") == PRIOR_FAILED_STAGE_CPU_SECONDS
        and prior_failed_result.get("cumulative_cpu_seconds") == PRIOR_FAILED_CPU_SECONDS
        and prior_failed_attempt.get("status") == "started"
        and prior_failed_attempt.get("plan_sha256") == PLAN_SHA256
        and prior_failed_attempt.get("parameter_updates") == 0
        and prior_failed_attempt.get("dev_queries") == 0
        and prior_failed_attempt.get("holdout_queries") == 0
        and prior_failed_attempt.get("confirmation_queries") == 0
        and sha256(PRIOR_FAILED_PROGRESS) == PRIOR_EMPTY_PROGRESS_SHA256
        and prior_readiness.get("complete") is True
        and prior_readiness.get("plan_sha256") == PLAN_SHA256
        and prior_readiness.get("source_hashes_unchanged") is True
        and prior_source_audit.get("status") == "complete"
        and prior_source_audit.get("source_hashes_unchanged") is True
        and prior_schedule.get("schedule_sha256") == prior_readiness.get("schedule_identity_sha256")
        and seal_ast.receipt_binding_matches(
            prior_schedule.get("source_audit_sha256"), PRIOR_SOURCE_AUDIT
        )
        and prior_source_binding.get("status") == "complete"
        and seal_ast.receipt_binding_matches(
            prior_source_binding.get("readiness_sha256"), PRIOR_READINESS
        )
        and prior_preflight.get("complete") is True
        and prior_preflight.get("plan_sha256") == PLAN_SHA256
        and prior_preflight.get("zero_checkpoint_count") == 6
        and prior_preflight.get("independent_restore_count") == 6
        and prior_preflight.get("source_hashes_unchanged") is True
        and prior_preflight.get("model_queries") == 0
        and prior_preflight.get("parameter_updates") == 0
        and seal_ast.receipt_binding_matches(
            prior_preflight.get("readiness_sha256"), PRIOR_READINESS
        )
        and prior_preflight_binding.get("status") == "complete"
        and prior_preflight_binding.get("plan_sha256") == PLAN_SHA256
        and seal_ast.receipt_binding_matches(
            prior_preflight_binding.get("preflight_sha256"), PRIOR_PREFLIGHT
        )
        and prior_preflight_binding.get("source_hashes") == prior_preflight.get("source_hashes")
        and len(previous_zero_rows) == 6
        and prior_failed_result.get("prior_readiness_preflight_cpu_seconds") == 108.15625
        and isinstance(previous_partial, dict)
        and previous_partial.get("additional_updates") == 0
        and Path(previous_partial.get("path", "")).resolve().parent
        == PRIOR_CHECKPOINT_ROOT.resolve()
        and previous_partial.get("sha256") == PRIOR_PARTIAL_CHECKPOINT_SHA256
        and sha256(Path(previous_partial["path"])) == PRIOR_PARTIAL_CHECKPOINT_SHA256,
        "N1-155A requires the sealed zero-update KeyError receipt and intact prior preflight",
    )
    previous_execution_sources = {}
    for raw_path in previous_paths:
        if not Path(raw_path).is_file():
            raise Refuse(f"N1-155A previous execution receipt missing: {raw_path}")
        _merge_hashes(
            previous_execution_sources,
            {str(Path(raw_path).resolve()): sha256(Path(raw_path))},
        )
    for row in previous_zero_rows:
        checkpoint = Path(row["path"]).resolve()
        if not checkpoint.is_file() or sha256(checkpoint) != row["sha256"]:
            raise Refuse(f"N1-155A prior zero checkpoint changed: {checkpoint}")
        _merge_hashes(previous_execution_sources, {str(checkpoint): row["sha256"]})
    previous_partial_path = Path(previous_partial["path"]).resolve()
    _merge_hashes(
        previous_execution_sources,
        {str(previous_partial_path): PRIOR_PARTIAL_CHECKPOINT_SHA256},
    )
    parents = {
        int(row["seed"]): row
        for row in n150_result.get("models", [])
        if row.get("branch") == PARENT_BRANCH and row.get("training_sufficient") is True
    }
    require(
        set(parents) == set(SEEDS), "N1-155 requires three sufficient N1-150 length-mixed parents"
    )
    sources = {}
    _merge_hashes(sources, previous_execution_sources)
    _merge_hashes(sources, n150_result.get("source_hashes", {}))
    _merge_hashes(sources, n151a.get("source_hashes", {}))
    _merge_hashes(sources, n152_result.get("source_hashes", {}))
    _merge_hashes(sources, n153a_result.get("source_hashes", {}))
    _merge_hashes(sources, n154_result.get("source_hashes", {}))
    _merge_hashes(sources, n150_result.get("new_checkpoint_hashes", {}))
    paths = (
        PLAN,
        CORRECTION_ADDENDUM,
        Path(__file__).resolve(),
        ROOT / "scripts/training/run_taiji_n1_plan155_short_unknown_context_length_training.py",
        ROOT / "tests/taiji_native/test_n1_plan155_short_unknown_context_length_contract.py",
        N150_RESULT,
        N151A_RESULT,
        N152_RESULT,
        N153A_RESULT,
        N154_RESULT,
        N153_FAILURE_RESULT,
        N153_FAILURE_ATTEMPT,
        n153_runner.READINESS,
        n153_runner.SOURCE_BINDING,
        n153_runner.PREFLIGHT,
        n153_runner.PREFLIGHT_BINDING,
        N153_FAILURE_RESULT,
        N153_FAILURE_ATTEMPT,
        n153_runner.READINESS,
        n153_runner.SOURCE_BINDING,
        n153_runner.PREFLIGHT,
        n153_runner.PREFLIGHT_BINDING,
        N153_FACE,
        N153_FACE_CHECKS,
        N153_SOURCE_AUDIT,
        n150.ATTEMPT,
        n150.PROGRESS,
        n150.READINESS,
        n150.SOURCE_BINDING,
        n150.PREFLIGHT,
        n150.PREFLIGHT_BINDING,
        n150_builder.PLAN,
        n150_builder.SCORING_ADDENDUM,
        n150_builder.RESOURCE_ADDENDUM,
        n150_builder.SCHEDULE,
    )
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise Refuse(f"frozen N1-150/153 source missing: {path}")
        _merge_hashes(sources, {str(path): sha256(path)})
    for seed, row in parents.items():
        checkpoint = Path(row["final_checkpoint_path"]).resolve()
        if not checkpoint.is_file() or sha256(checkpoint) != row["final_checkpoint_sha256"]:
            raise Refuse(f"N1-150 length-mixed seed{seed} parent changed")
        _merge_hashes(sources, {str(checkpoint): row["final_checkpoint_sha256"]})
    if not n150_builder.verify_hashes(sources):
        raise Refuse("N1-155 bound training source/checkpoint hashes changed")
    return (
        {
            "n150_readiness": n150_readiness,
            "n150_preflight": n150_preflight,
            "n150_result": n150_result,
            "parents": parents,
        },
        bound,
        n150_schedule,
        dict(sorted(sources.items())),
    )


def build_schedule(base_schedule: dict) -> dict:
    slots = base_schedule.get("slots", [])
    if len(slots) != UPDATES_PER_MODEL:
        raise Refuse("N1-155 base schedule must contain 9,216 source-preserving slots")
    occurrence = Counter()
    route_counts = Counter()
    parent_counts = Counter()
    parent_kind = {}
    kind_counts = Counter()
    length_counts = Counter()
    route_length_counts = Counter()
    rows = []
    for slot in slots:
        item = slot["item"]
        parent, route = n150_builder._parent_and_route(item)
        route_key = (parent, route)
        ordinal = occurrence[route_key]
        occurrence[route_key] += 1
        parent_counts[parent] += 1
        route_counts[route_key] += 1
        kind = str(item["kind"])
        if parent in parent_kind and parent_kind[parent] != kind:
            raise Refuse(f"N1-155 parent has multiple task kinds: {parent}")
        parent_kind[parent] = kind
        kind_counts[kind] += 1
        if kind == UNKNOWN:
            candidate_length = UNKNOWN_LENGTH_PATTERN[ordinal]
        else:
            candidate_length = UNIFORM_LENGTH_PATTERN[ordinal]
        control_length = UNIFORM_LENGTH_PATTERN[ordinal]
        length_counts[(kind, candidate_length)] += 1
        route_length_counts[(parent, route, kind, candidate_length)] += 1
        rows.append(
            {
                "step": int(slot["step"]),
                "parent_id": parent,
                "route": route,
                "route_ordinal": ordinal,
                "kind": kind,
                "item_key": slot["item_key"],
                "item_id": item["id"],
                "second_key": slot.get("second_key"),
                "second_id": slot.get("second_id"),
                "candidate_length": candidate_length,
                "control_length": control_length,
            }
        )
    require(
        len(parent_counts) == 192 and set(parent_counts.values()) == {48},
        "N1-155 parent update balance differs",
    )
    require(set(route_counts.values()) == {24}, "N1-155 raw/hard route update balance differs")
    require(
        kind_counts == Counter({kind: 2_304 for kind in KINDS}),
        "N1-155 task kind update totals differ",
    )
    expected_candidate = {
        (kind, length): count
        for kind in KINDS
        for length, count in (
            ((8, 768), (16, 768), (32, 768))
            if kind != UNKNOWN
            else ((8, 1_152), (16, 576), (32, 576))
        )
    }
    require(
        {key: length_counts[key] for key in expected_candidate} == expected_candidate,
        "N1-155 candidate kind/length marginal differs",
    )
    expected_route = {}
    for parent, route in route_counts:
        kind = parent_kind[parent]
        for length, count in (
            ((8, 8), (16, 8), (32, 8)) if kind != UNKNOWN else ((8, 12), (16, 6), (32, 6))
        ):
            expected_route[(parent, route, kind, length)] = count
    require(
        {key: route_length_counts[key] for key in expected_route} == expected_route,
        "N1-155 per-parent raw/hard candidate exposure differs",
    )
    require(
        all(row["control_length"] == UNIFORM_LENGTH_PATTERN[row["route_ordinal"]] for row in rows),
        "N1-155 control no longer replays uniform lengths",
    )
    manifest = {
        "schema": "n1-plan155a-v28-unknown8-rebalance-schedule-v1",
        "plan_sha256": PLAN_SHA256,
        "seeds": list(SEEDS),
        "branches": list(BRANCHES),
        "updates_per_model": UPDATES_PER_MODEL,
        "updates_per_parent": 48,
        "parent_count": len(parent_counts),
        "updates_by_kind": dict(sorted(kind_counts.items())),
        "candidate_updates_by_kind_length": {
            kind: {str(length): length_counts[(kind, length)] for length in LENGTHS}
            for kind in KINDS
        },
        "control_updates_by_length": {str(length): UPDATES_PER_MODEL // 3 for length in LENGTHS},
        "unknown_candidate_updates_by_length_per_parent": {"8": 24, "16": 12, "32": 12},
        "known_candidate_updates_by_length_per_parent": {"8": 16, "16": 16, "32": 16},
        "answerable_binding_queries_per_model": base_schedule["manifest"][
            "answerable_binding_queries_per_model"
        ],
        "total_updates": TOTAL_UPDATES,
        "train_score_queries": TRAIN_SCORE_QUERIES,
        "fresh_face_queries_planned": FRESH_QUERIES,
        "schedule_rows": rows,
        "schedule_sha256": canonical_sha(rows),
        "context_source_sha256": base_schedule["manifest"]["schedule_sha256"],
        "can_promote": False,
    }
    runtime_schedule = {
        "manifest": manifest,
        "slots": slots,
        "rows": rows,
        "length_counts": length_counts,
    }
    if "views" in base_schedule:
        runtime_schedule["views"] = base_schedule["views"]
    return runtime_schedule


def validate_schedule(manifest: dict) -> dict:
    rows = manifest.get("schedule_rows", [])
    if manifest.get("schedule_sha256") != canonical_sha(rows):
        raise Refuse("N1-155 schedule digest mismatch")
    if len(rows) != UPDATES_PER_MODEL:
        raise Refuse("N1-155 schedule must contain 9,216 rows")
    candidate = Counter((row["kind"], row["candidate_length"]) for row in rows)
    control = Counter(row["control_length"] for row in rows)
    expected_candidate = {
        (kind, length): count
        for kind in KINDS
        for length, count in (
            ((8, 768), (16, 768), (32, 768))
            if kind != UNKNOWN
            else ((8, 1_152), (16, 576), (32, 576))
        )
    }
    if dict(candidate) != expected_candidate or dict(control) != {8: 3_072, 16: 3_072, 32: 3_072}:
        raise Refuse("N1-155 candidate/control length marginals differ")
    return {
        "accepted": True,
        "rows": len(rows),
        "candidate_by_kind_length": {
            f"{kind}/{length}": count for (kind, length), count in candidate.items()
        },
        "control_by_length": dict(control),
    }


def source_readiness() -> dict:
    outputs = (
        SOURCE_AUDIT,
        SCHEDULE,
        READINESS,
        READINESS_ATTEMPT,
        SOURCE_BINDING,
        PREFLIGHT,
        PREFLIGHT_ATTEMPT,
        PREFLIGHT_BINDING,
        ATTEMPT,
        PROGRESS,
        RESULT,
        FACE,
        FACE_AUDIT,
        FACE_CHECKS,
    )
    if any(path.exists() for path in outputs) or CHECKPOINT_ROOT.exists():
        raise FileExistsError("PLAN-N1-155 one-shot source/schedule outputs already exist")
    if sha256(PLAN) != PLAN_SHA256:
        raise Refuse("PLAN-N1-155 plan SHA changed")
    started = time.process_time()
    context, bound, base_schedule, sources = _source_context()
    schedule = build_schedule(base_schedule)
    validate_schedule(schedule["manifest"])
    require(n150_builder.verify_hashes(sources), "N1-155 sources changed during schedule readiness")
    cpu = time.process_time() - started
    audit = {
        "schema": "n1-plan155a-v28-unknown8-source-audit-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": CORRECTION_SHA256,
        "prior_failed_attempt_result_sha256": PRIOR_FAILED_RESULT_SHA256,
        "prior_failed_attempt_cpu_seconds": PRIOR_FAILED_CPU_SECONDS,
        "source_hashes": sources,
        "source_hashes_unchanged": True,
        "n150_length_mixed_parent_hashes": {
            str(Path(row["final_checkpoint_path"]).resolve()): row["final_checkpoint_sha256"]
            for row in context["parents"].values()
        },
        "n150_train_context_schedule_sha256": base_schedule["manifest"]["schedule_sha256"],
        "unknown8_candidate_exposure": {"8": 1_152, "16": 576, "32": 576},
        "known_candidate_exposure_per_kind": {"8": 768, "16": 768, "32": 768},
        "control_exposure": {"8": 3_072, "16": 3_072, "32": 3_072},
        "updates_per_model": UPDATES_PER_MODEL,
        "total_training_updates": TOTAL_UPDATES,
        "planned_train_score_queries": TRAIN_SCORE_QUERIES,
        "planned_fresh_queries": FRESH_QUERIES,
        "cpu_seconds": round(cpu, 6),
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "can_promote": False,
    }
    write_new(SOURCE_AUDIT, audit)
    schedule_manifest = dict(schedule["manifest"], source_audit_sha256=sha256(SOURCE_AUDIT))
    write_new(SCHEDULE, schedule_manifest)
    prepared_sources = dict(sources)
    prepared_sources[str(SOURCE_AUDIT.resolve())] = sha256(SOURCE_AUDIT)
    prepared_sources[str(SCHEDULE.resolve())] = sha256(SCHEDULE)
    cumulative = PRIOR_FAILED_CPU_SECONDS + cpu
    complete = n150_builder.verify_hashes(prepared_sources) and cumulative <= CPU_LIMIT_SECONDS
    readiness = {
        "schema": "n1-plan155a-v28-unknown8-readiness-v1",
        "status": "ready_for_zero_checkpoint_preflight" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "correction_addendum_sha256": CORRECTION_SHA256,
        "prior_failed_attempt_cpu_seconds": PRIOR_FAILED_CPU_SECONDS,
        "cumulative_cpu_seconds": round(cumulative, 6),
        "source_audit_sha256": sha256(SOURCE_AUDIT),
        "schedule_sha256": sha256(SCHEDULE),
        "schedule_identity_sha256": schedule_manifest["schedule_sha256"],
        "source_hashes": prepared_sources,
        "source_hashes_unchanged": n150_builder.verify_hashes(prepared_sources),
        "parents": audit["n150_length_mixed_parent_hashes"],
        "schedule_summary": {
            "parents": 192,
            "updates_per_model": UPDATES_PER_MODEL,
            "total_training_updates": TOTAL_UPDATES,
            "candidate_unknown_by_length": {"8": 1_152, "16": 576, "32": 576},
            "candidate_known_by_kind_length": schedule_manifest["candidate_updates_by_kind_length"],
            "control_by_length": schedule_manifest["control_updates_by_length"],
            "train_score_queries": TRAIN_SCORE_QUERIES,
            "fresh_queries": FRESH_QUERIES,
        },
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
    }
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan155a-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "correction_addendum_sha256": CORRECTION_SHA256,
            "prior_failed_attempt_cpu_seconds": PRIOR_FAILED_CPU_SECONDS,
            "cpu_seconds": round(cpu, 6),
            "model_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
        },
    )
    write_new(READINESS, readiness)
    write_new(
        SOURCE_BINDING,
        {
            "schema": "n1-plan155a-readiness-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "correction_addendum_sha256": CORRECTION_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_hashes": prepared_sources,
            "can_promote": False,
        },
    )
    return readiness


def prepare_fresh_face(
    *, final_checkpoint_hashes: dict[str, str], training_progress_sha256: str
) -> dict:
    if any(path.exists() for path in (FACE, FACE_AUDIT, FACE_CHECKS)):
        raise FileExistsError("N1-155 seed128 face outputs are write-once")
    if not READINESS.is_file() or not read_json(READINESS).get("complete"):
        raise Refuse("N1-155 schedule readiness is required before face generation")
    if (
        not ATTEMPT.is_file()
        or not PROGRESS.is_file()
        or sha256(PROGRESS) != training_progress_sha256
    ):
        raise Refuse("N1-155 completed training progress is required before face generation")
    if len(final_checkpoint_hashes) != 6:
        raise Refuse("N1-155 face requires all six final checkpoint hashes")
    for path, digest in final_checkpoint_hashes.items():
        if not Path(path).is_file() or sha256(Path(path)) != digest:
            raise Refuse(f"N1-155 final checkpoint changed before face generation: {path}")
    started = time.process_time()
    _context, _bound, _base_schedule, sources = _source_context()
    readiness = read_json(READINESS)
    readiness_sources = dict(sources)
    readiness_sources[str(SOURCE_AUDIT.resolve())] = sha256(SOURCE_AUDIT)
    readiness_sources[str(SCHEDULE.resolve())] = sha256(SCHEDULE)
    if readiness.get("source_hashes") != readiness_sources:
        raise Refuse("N1-155 readiness source set changed before fresh face")

    excluded_entities, excluded_words, n150_exclusion_summary, inherited, _runtime = (
        n153_builder.collect_n150_exclusions()
    )
    if any(sources.get(path) != digest for path, digest in inherited.items()):
        raise Refuse("N1-155 inherited N1-150 source union differs from readiness")
    old_face = n153_builder.read_json(N153_FACE)
    old_face_entities = set()
    old_face_words = set()
    for item in old_face.get("items", []):
        old_face_entities.update(str(name) for name in item.get("entities", []))
        answer = item.get("answer")
        if answer not in (None, UNKNOWN, *DIRECTIONS):
            old_face_words.add(str(answer))
        for row in item.get("observations_by_length", {}).get("32", []):
            parts = row.removesuffix(".").split()
            if parts:
                old_face_entities.add(parts[0])
            if len(parts) == 3 and parts[1] == "uses":
                old_face_words.add(parts[2])
    excluded_entities.update(old_face_entities)
    excluded_words.update(old_face_words)
    exclusion_summary = {
        "n150_base_exclusions": n150_exclusion_summary,
        "prior_n153_face_entity_count": len(old_face_entities),
        "prior_n153_face_word_count": len(old_face_words),
        "excluded_entity_count": len(excluded_entities),
        "excluded_word_value_count": len(excluded_words),
        "entity_exclusion_sha256": digest_strings(excluded_entities),
        "word_exclusion_sha256": digest_strings(excluded_words),
    }

    base_face, generator_summary = n153_builder.build_face(
        excluded_entities, excluded_words, seed=128
    )
    for item in base_face["items"]:
        item["id"] = item["id"].replace("v28-", "v29-", 1)
        item["parent_id"] = item["id"]
    face_entities = set()
    face_entity_rows = []
    face_materials = set()
    face_material_rows = []
    target_names = set()
    for item in base_face["items"]:
        target = n132._query_subject(item["question"])
        target_names.add(target)
        face_entities.update(item.get("entities", []))
        face_entity_rows.extend(item.get("entities", []))
        item_materials = set()
        if item.get("answer") not in (None, UNKNOWN, *DIRECTIONS):
            face_materials.add(item["answer"])
            item_materials.add(item["answer"])
        for row in item["observations_by_length"]["32"]:
            parts = row.removesuffix(".").split()
            if parts:
                face_entities.add(parts[0])
            if len(parts) == 3 and parts[1] == "uses":
                face_materials.add(parts[2])
                item_materials.add(parts[2])
        face_material_rows.extend(item_materials)
    if face_entities & excluded_entities or face_materials & excluded_words:
        raise Refuse("N1-155 fresh face overlaps prior/train/N1-150 generated exclusions")
    if len(face_entity_rows) != len(set(face_entity_rows)) or len(face_material_rows) != len(
        set(face_material_rows)
    ):
        raise Refuse("N1-155 fresh face target/distractor names or material values are not unique")
    if len(target_names) != 160:
        raise Refuse("N1-155 fresh face must contain 160 unique target names")

    face_manifest = {
        **base_face,
        "schema": "n1-plan155a-v29-short-unknown-length-face-v1",
        "correction_addendum_sha256": CORRECTION_SHA256,
        "plan_sha256": PLAN_SHA256,
        "seed": 128,
        "source_exclusion_summary": exclusion_summary,
        "prior_n153_face_sha256": sha256(N153_FACE),
        "queries_per_endpoint": FRESH_QUERIES_PER_ENDPOINT,
        "endpoint_count": len(ENDPOINTS),
        "total_queries": FRESH_QUERIES,
        "can_promote": False,
    }
    write_new(FACE, face_manifest)
    face_hash = sha256(FACE)
    score_sources = dict(readiness_sources)
    score_sources[str(FACE.resolve())] = face_hash
    for raw_path, digest in final_checkpoint_hashes.items():
        score_sources[str(Path(raw_path).resolve())] = digest
    score_sources[str(ATTEMPT.resolve())] = sha256(ATTEMPT)
    score_sources[str(PROGRESS.resolve())] = training_progress_sha256
    audit = {
        "schema": "n1-plan155a-v29-face-source-audit-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "source_hashes": score_sources,
        "source_hashes_unchanged": n150_builder.verify_hashes(score_sources),
        "source_exclusion_summary": exclusion_summary,
        "new_face_target_count": len(target_names),
        "new_face_entity_count": len(face_entity_rows),
        "new_face_material_count": len(face_material_rows),
        "target_overlap": 0,
        "material_overlap": 0,
        "face_sha256": face_hash,
        "generator_summary": generator_summary,
        "cpu_seconds": round(time.process_time() - started, 6),
        "can_promote": False,
    }
    write_new(FACE_AUDIT, audit)
    checks = {
        "schema": "n1-plan155a-v29-face-checks-v1",
        "correction_addendum_sha256": CORRECTION_SHA256,
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "face_sha256": face_hash,
        "face_manifest_sha256": sha256(FACE),
        "face_audit_sha256": sha256(FACE_AUDIT),
        "face_item_count": len(face_manifest["items"]),
        "items_by_kind": face_manifest["items_by_kind"],
        "queries_per_endpoint": FRESH_QUERIES_PER_ENDPOINT,
        "total_queries": FRESH_QUERIES,
        "source_hash_count": len(score_sources),
        "source_hashes_unchanged": audit["source_hashes_unchanged"],
        "new_face_entity_count": len(face_entity_rows),
        "new_face_material_count": len(face_material_rows),
        "target_entity_overlap": 0,
        "material_overlap": 0,
        "nested_contexts": True,
        "can_promote": False,
    }
    write_new(FACE_CHECKS, checks)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-155 unknown-short-context rebalancing schedule"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--prepare-fresh-face", action="store_true")
    args = parser.parse_args()
    result = source_readiness() if args.validate_only else prepare_fresh_face()
    keys = (
        "status",
        "complete",
        "accepted",
        "cpu_seconds",
        "endpoint_count",
        "total_training_updates",
        "fresh_queries",
        "source_hash_count",
        "target_overlap",
        "material_overlap",
        "model_loads",
        "model_queries",
        "parameter_updates",
        "can_promote",
    )
    print(
        json.dumps(
            {key: result.get(key) for key in keys if key in result},
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete", result.get("status") == "complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
