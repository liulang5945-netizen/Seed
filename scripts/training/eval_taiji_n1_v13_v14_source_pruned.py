"""One-shot PLAN-N1-126 oracle source-pruning diagnostic on consumed confirms."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v13_after_n122 as b13  # noqa: E402
from scripts.training import build_taiji_n1_v14_after_n123 as b14  # noqa: E402
from scripts.training import eval_taiji_n1_v11_after_n120 as e11  # noqa: E402
from scripts.training import eval_taiji_n1_v13_after_n122 as e13  # noqa: E402
from scripts.training import eval_taiji_n1_v14_after_n123 as e14  # noqa: E402
from scripts.training import (
    run_taiji_n1_expanded_normalized_rwl_length_balanced as n120,
)  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-126_source_pruned_saved_error_diagnostic_draft_20261019.md"
PLAN_SHA256 = "795aaa57900a05e63edf6097578523121cf63ab82971cdeada2874aebc2b0f96"
INITIAL_READINESS_PLAN_SHA256 = "24740d08e0d61fa0ccf77530f309d47be42b84af07349bd4d4217e9009b3b614"
READINESS = ROOT / "reports/taiji_n1_v13_v14_source_pruned_readiness_20261019.json"
READINESS_REVALIDATION = ROOT / "reports/taiji_n1_v13_v14_source_pruned_readiness_revalidation_20261019.json"
ATTEMPT = ROOT / "reports/taiji_n1_v13_v14_source_pruned_attempt_20261019.json"
RESULT = ROOT / "reports/taiji_n1_v13_v14_source_pruned_result_20261019.json"
CPU_LIMIT_SECONDS = 600
SEEDS = e11.SEEDS
ARMS = ("position", "cut")
KINDS = ("lookup", "revision", "transition")
FACE_SOURCES = {
    "v13": {
        "face": b13.FACE,
        "checks": b13.FACE_CHECKS,
        "readiness": b13.READINESS,
        "confirmation": b13.CONFIRM_REPORT,
        "range": b13.RANGE_REPORT,
        "builder": b13,
        "evaluator": e13,
        "seed": 113,
        "schema": "n1-hard-neighbor-v13-confirmation-v1",
    },
    "v14": {
        "face": b14.FACE,
        "checks": b14.FACE_CHECKS,
        "readiness": b14.READINESS,
        "confirmation": b14.CONFIRM_REPORT,
        "range": b14.RANGE_REPORT,
        "builder": b14,
        "evaluator": e14,
        "seed": 114,
        "schema": "n1-hard-neighbor-v14-confirmation-v1",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def relevant_rows(item: dict) -> list[str]:
    subject = item["entities"][0]
    kind = item["kind"]
    if kind == "lookup":
        rows = [row for row in item["observations"] if row.startswith(subject + " uses ")]
        expected = 1
    elif kind == "revision":
        rows = [
            row
            for row in item["observations"]
            if row.startswith(subject + " uses ") or row.startswith(subject + " now uses ")
        ]
        expected = 2
    elif kind == "transition":
        rows = [
            row
            for row in item["observations"]
            if row.startswith(subject + " faces ") or row.startswith(subject + " turns ")
        ]
        expected = 2
    else:
        raise ValueError(f"unsupported intervention kind: {kind}")
    if len(rows) != expected:
        raise ValueError(
            f"expected {expected} gold-selected rows for {item['id']}, got {len(rows)}"
        )
    original_order = [row for row in item["observations"] if row in set(rows)]
    if rows != original_order:
        raise ValueError(f"source rows were reordered for {item['id']}")
    return rows


def _source_set(*, allow_readiness: bool = False) -> tuple[dict, dict]:
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-126 changed after approval")
    output_paths = (ATTEMPT, RESULT) if allow_readiness else (READINESS, ATTEMPT, RESULT)
    if any(path.exists() for path in output_paths):
        raise FileExistsError("PLAN-N1-126 outputs are write-once")

    readiness13 = b13._verify_readiness()
    readiness14 = b14._verify_readiness()
    face_data = {}
    original_data = {}
    source_hashes = {str(PLAN.resolve()): PLAN_SHA256}
    for label, spec in FACE_SOURCES.items():
        face, checks = spec["evaluator"]._face_and_checks()
        range_report = read_json(spec["range"])
        confirmation = read_json(spec["confirmation"])
        if (
            face.get("schema") != spec["schema"]
            or face.get("seed") != spec["seed"]
            or checks.get("accepted") is not True
            or range_report.get("complete") is not True
            or range_report.get("range_gate", {}).get("range_check_passed") is not True
            or range_report.get("model_queries") != 180
            or confirmation.get("complete") is not True
            or confirmation.get("confirmation_queries") != 360
            or confirmation.get("total_model_queries") != 540
            or confirmation.get("source_hashes_unchanged") is not True
            or confirmation.get("can_promote") is not False
        ):
            raise ValueError(f"frozen {label} face or reports are incomplete")
        items = [
            item
            for item in face["episodes"]
            if item["id"] in set(face["subsets"]["confirm"]) and item["kind"] in KINDS
        ]
        if len(items) != 15 or any(
            sum(item["kind"] == kind for item in items) != 5 for kind in KINDS
        ):
            raise ValueError(f"{label} intervention denominator must be 5 per decodable kind")
        for item in items:
            relevant_rows(item)
        original = {}
        for row in confirmation["candidate_rows"]:
            key = (row["seed"], row["arm"], row["id"])
            if (
                row["seed"] in SEEDS
                and row["arm"] in ARMS
                and row["id"] in {item["id"] for item in items}
            ):
                if key in original:
                    raise ValueError(f"duplicate saved native row: {key}")
                original[key] = row
        if len(original) != 90:
            raise ValueError(f"{label} saved native rows must cover 15 × 6 cases")
        face_data[label] = (face, checks, items)
        original_data[label] = original
        for path in (
            spec["face"],
            spec["checks"],
            spec["readiness"],
            spec["range"],
            spec["confirmation"],
        ):
            source_hashes[str(Path(path).resolve())] = sha256(Path(path))
    if readiness13["checkpoint_hashes"] != readiness14["checkpoint_hashes"]:
        raise ValueError("v13/v14 checkpoint hash maps differ")
    if len(readiness14.get("checkpoint_hashes", {})) != 36:
        raise ValueError("all 36 frozen endpoint hashes are required")
    for path, expected in readiness14["checkpoint_hashes"].items():
        if not Path(path).is_file() or sha256(Path(path)) != expected:
            raise ValueError(f"frozen checkpoint changed: {path}")

    baseline, candidate, _, _, restore, _ = e14._bound_inputs()
    _, candidate_paths = e14._checkpoint_paths(baseline, candidate)
    if (
        len(candidate_paths) != 9
        or restore.get("independent_full_bundle_restores") != 18
        or restore.get("all_full_bundles_equal") is not True
    ):
        raise ValueError("N1-120 candidate endpoints and restore evidence are required")
    checkpoint_hashes = {
        str(path.resolve()): digest
        for (seed, arm), (path, digest) in candidate_paths.items()
        if arm in ARMS
    }
    if len(checkpoint_hashes) != 6 or any(
        sha256(Path(path)) != digest for path, digest in checkpoint_hashes.items()
    ):
        raise ValueError("six position/cut checkpoint hashes are required")

    scorer = e11._verify_scorer_readiness()
    if scorer.get("model_queries") != 0 or scorer.get("parameter_updates") != 0:
        raise ValueError("scorer readiness must remain query/update free")
    n120.configure()
    _, g, _ = n120._BASE.make_runner(load_inputs=False)
    if not callable(g.get("load")) or not callable(e11.scorer.evaluate):
        raise TypeError("candidate checkpoint loader or frozen scorer is unavailable")

    bound_sources = {
        **readiness13["source_hashes"],
        **readiness14["source_hashes"],
        **readiness14["checkpoint_hashes"],
        **scorer["source_hashes"],
    }
    bound_sources.update(source_hashes)
    bound_sources[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    return {
        "readiness13": readiness13,
        "readiness14": readiness14,
        "face_data": face_data,
        "original_data": original_data,
        "source_hashes": bound_sources,
        "checkpoint_hashes": checkpoint_hashes,
        "candidate": candidate,
        "candidate_paths": candidate_paths,
        "candidate_globals": g,
        "scorer_readiness": scorer,
    }, {
        "baseline": len(baseline.get("models", [])),
        "candidate": len(candidate.get("models", [])),
        "source_hash_count": len(bound_sources),
        "checkpoint_hash_count": len(checkpoint_hashes),
        "restore_count": restore["independent_full_bundle_restores"],
    }


def source_readiness() -> dict:
    started = time.process_time()
    bound, counts = _source_set()
    for path, expected in {**bound["source_hashes"], **bound["checkpoint_hashes"]}.items():
        file = Path(path)
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"source changed during PLAN-N1-126 readiness: {path}")
    cpu = time.process_time() - started
    if cpu > CPU_LIMIT_SECONDS:
        status = "stopped_cpu_limit"
    else:
        status = "ready_for_approved_execution"
    report = {
        "schema": "n1-plan126-source-pruned-readiness-v1",
        "status": status,
        "complete": status == "ready_for_approved_execution",
        "plan_sha256": PLAN_SHA256,
        "source_hashes": bound["source_hashes"],
        "checkpoint_hashes": bound["checkpoint_hashes"],
        "checkpoint_hash_count": counts["checkpoint_hash_count"],
        "baseline_endpoints_verified": counts["baseline"],
        "candidate_endpoints_verified": counts["candidate"],
        "independent_full_bundle_restores_verified": counts["restore_count"],
        "faces": {
            label: {
                "face_sha256": sha256(FACE_SOURCES[label]["face"]),
                "confirmation_sha256": sha256(FACE_SOURCES[label]["confirmation"]),
                "intervention_items": 15,
                "items_by_kind": {kind: 5 for kind in KINDS},
                "saved_native_rows_bound": len(bound["original_data"][label]),
            }
            for label in FACE_SOURCES
        },
        "planned_model_queries": 180,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "can_promote": False,
    }
    write_new(READINESS, report)
    return report


def source_revalidation() -> dict:
    if (
        not READINESS.is_file()
        or READINESS_REVALIDATION.exists()
        or ATTEMPT.exists()
        or RESULT.exists()
    ):
        raise FileExistsError(
            "source revalidation requires the initial readiness and no query attempt"
        )
    started = time.process_time()
    parent = read_json(READINESS)
    if (
        parent.get("complete") is not True
        or parent.get("status") != "ready_for_approved_execution"
        or parent.get("plan_sha256") != INITIAL_READINESS_PLAN_SHA256
        or parent.get("model_queries") != 0
        or parent.get("parameter_updates") != 0
    ):
        raise ValueError("the initial source readiness is not the expected zero-query receipt")
    bound, counts = _source_set(allow_readiness=True)
    plan_key = str(PLAN.resolve())
    runner_key = str(Path(__file__).resolve())
    permitted_revisions = {plan_key, runner_key}
    stale = [
        path
        for path, expected in parent.get("source_hashes", {}).items()
        if path not in permitted_revisions and bound["source_hashes"].get(path) != expected
    ]
    if stale or parent.get("checkpoint_hashes") != bound["checkpoint_hashes"]:
        raise ValueError(f"non-document/non-runner readiness inputs changed: {stale[:3]}")
    source_hashes = dict(bound["source_hashes"])
    source_hashes[str(READINESS.resolve())] = sha256(READINESS)
    for path, expected in {**source_hashes, **bound["checkpoint_hashes"]}.items():
        file = Path(path)
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"source changed during readiness revalidation: {path}")
    revalidation_cpu = time.process_time() - started
    cumulative_cpu = float(parent["cpu_seconds"]) + revalidation_cpu
    complete = cumulative_cpu <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan126-source-pruned-readiness-revalidation-v1",
        "status": "ready_for_approved_execution" if complete else "stopped_cpu_limit",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "parent_readiness_sha256": sha256(READINESS),
        "parent_readiness_plan_sha256": parent["plan_sha256"],
        "source_hashes": source_hashes,
        "checkpoint_hashes": bound["checkpoint_hashes"],
        "checkpoint_hash_count": counts["checkpoint_hash_count"],
        "baseline_endpoints_verified": counts["baseline"],
        "candidate_endpoints_verified": counts["candidate"],
        "independent_full_bundle_restores_verified": counts["restore_count"],
        "parent_readiness_cpu_seconds": float(parent["cpu_seconds"]),
        "cpu_seconds": round(revalidation_cpu, 6),
        "cumulative_preflight_cpu_seconds": round(cumulative_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "planned_model_queries": 180,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "can_promote": False,
    }
    write_new(READINESS_REVALIDATION, report)
    return report


def _verify_readiness() -> dict:
    report_path = READINESS_REVALIDATION if READINESS_REVALIDATION.is_file() else READINESS
    if not report_path.is_file():
        raise FileNotFoundError("PLAN-N1-126 zero-query readiness is required")
    report = read_json(report_path)
    cumulative_preflight = float(
        report.get("cumulative_preflight_cpu_seconds", report.get("cpu_seconds", 0))
    )
    if (
        report.get("complete") is not True
        or report.get("status") != "ready_for_approved_execution"
        or report.get("plan_sha256") != PLAN_SHA256
        or report.get("model_loads") != 0
        or report.get("model_queries") != 0
        or report.get("parameter_updates") != 0
        or report.get("checkpoint_hash_count") != 6
        or report.get("planned_model_queries") != 180
        or cumulative_preflight > CPU_LIMIT_SECONDS
    ):
        raise ValueError("PLAN-N1-126 readiness is stale or incomplete")
    for path, expected in report["source_hashes"].items():
        file = Path(path)
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"source changed after PLAN-N1-126 readiness: {path}")
    for path, expected in report["checkpoint_hashes"].items():
        file = Path(path)
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"checkpoint changed after PLAN-N1-126 readiness: {path}")
    report["resolved_readiness_path"] = str(report_path.resolve())
    report["resolved_readiness_sha256"] = sha256(report_path)
    report["cumulative_preflight_cpu_seconds"] = cumulative_preflight
    return report


def run(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-126 model queries require owner approval")
    if ATTEMPT.exists() or RESULT.exists():
        raise FileExistsError("PLAN-N1-126 query stage is one-shot")
    stage_started = time.process_time()
    readiness = _verify_readiness()
    bound, _ = _source_set(allow_readiness=True)
    remaining = (
        CPU_LIMIT_SECONDS
        - float(readiness["cumulative_preflight_cpu_seconds"])
        - (time.process_time() - stage_started)
    )
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-126 cumulative CPU budget is exhausted before queries")
    budget = e11.CpuDeadline(remaining)
    n120.configure()
    _, g, _ = n120._BASE.make_runner(load_inputs=False)
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan126-source-pruned-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": readiness["resolved_readiness_sha256"],
            "readiness_path": readiness["resolved_readiness_path"],
            "face_hashes": {label: sha256(FACE_SOURCES[label]["face"]) for label in FACE_SOURCES},
            "checkpoint_hashes": readiness["checkpoint_hashes"],
            "planned_queries": 180,
            "query_mode": "free_generation_after_oracle_source_row_pruning",
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    rows = []
    loads = 0
    queries = 0
    errors = []
    try:
        for label in FACE_SOURCES:
            _, _, items = bound["face_data"][label]
            original = bound["original_data"][label]
            for seed in SEEDS:
                for arm in ARMS:
                    budget.guard(reserve=5)
                    path, digest = bound["candidate_paths"][seed, arm]
                    pruned_items = [{**item, "observations": relevant_rows(item)} for item in items]
                    batch, count, loaded, error = e14._score_endpoint(
                        g,
                        path,
                        digest,
                        bound["candidate"]["protocol"],
                        seed,
                        arm,
                        pruned_items,
                        budget,
                    )
                    loads += int(loaded)
                    queries += count
                    for row in batch:
                        native = original[seed, arm, row["id"]]
                        rows.append(
                            {
                                **row,
                                "face": label,
                                "input_condition": "oracle_target_rows_only",
                                "native_full_context_free_exact": native["free_exact"],
                                "native_full_context_output_hex": native["output_hex"],
                                "native_to_pruned_gain": row["free_exact"]
                                and not native["free_exact"],
                                "native_to_pruned_loss": native["free_exact"]
                                and not row["free_exact"],
                            }
                        )
                    if error:
                        errors.append(f"{label}/{seed}/{arm}: {error}")
                        break
                if errors:
                    break
            if errors:
                break
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    source_hashes = {
        **readiness["source_hashes"],
        **readiness["checkpoint_hashes"],
        readiness["resolved_readiness_path"]: readiness["resolved_readiness_sha256"],
        str(ATTEMPT.resolve()): sha256(ATTEMPT),
    }
    unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == expected
        for path, expected in source_hashes.items()
    )
    stage_cpu = time.process_time() - stage_started
    cumulative_cpu = float(readiness["cumulative_preflight_cpu_seconds"]) + stage_cpu
    summaries = {}
    for label in FACE_SOURCES:
        summaries[label] = {}
        for arm in ARMS:
            selected = [row for row in rows if row["face"] == label and row["arm"] == arm]
            summaries[label][arm] = {
                "rows": len(selected),
                "native_exact": sum(row["native_full_context_free_exact"] for row in selected),
                "source_pruned_exact": sum(row["free_exact"] for row in selected),
                "gains": sum(row["native_to_pruned_gain"] for row in selected),
                "losses": sum(row["native_to_pruned_loss"] for row in selected),
                "exact_by_kind": {
                    kind: sum(row["kind"] == kind and row["free_exact"] for row in selected)
                    for kind in KINDS
                },
            }
    complete = (
        not errors
        and len(rows) == 180
        and queries == 180
        and loads == 12
        and unchanged
        and cumulative_cpu <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan126-source-pruned-result-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": readiness["resolved_readiness_sha256"],
        "readiness_path": readiness["resolved_readiness_path"],
        "attempt_sha256": sha256(ATTEMPT),
        "source_hashes_unchanged": unchanged,
        "source_hashes": source_hashes,
        "endpoint_checkpoint_hashes": readiness["checkpoint_hashes"],
        "rows": rows,
        "summary": summaries,
        "model_loads": loads,
        "model_queries": queries,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "unknown_queries": 0,
        "cpu_seconds": round(stage_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "oracle_intervention_is_capability": False,
        "can_promote": False,
        "errors": errors,
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--readiness", action="store_true")
    mode.add_argument("--revalidate-sources", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    if args.readiness:
        result = source_readiness()
    elif args.revalidate_sources:
        result = source_revalidation()
    else:
        result = run(owner_approved=args.owner_approved)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "schema",
                    "status",
                    "complete",
                    "cpu_seconds",
                    "model_queries",
                    "can_promote",
                    "errors",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
