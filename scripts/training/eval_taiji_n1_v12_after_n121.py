"""One-shot PLAN-N1-122 range and conditional confirmation evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v12_after_n121 as face_builder  # noqa: E402
from scripts.training import eval_taiji_n1_v11_after_n120 as v11_eval  # noqa: E402
from scripts.training import (
    run_taiji_n1_expanded_normalized_rwl_length_balanced as n120,
)  # noqa: E402
from scripts.training import run_taiji_n1_hard_neighbor_training as baseline_training  # noqa: E402
from scripts.training import seal_ast  # noqa: E402

FACE = face_builder.FACE
FACE_ATTEMPT = face_builder.FACE_ATTEMPT
FACE_CHECKS = face_builder.FACE_CHECKS
SOURCE_EXCLUSIONS = face_builder.SOURCE_EXCLUSIONS
READINESS = face_builder.READINESS
RANGE_REPORT = face_builder.RANGE_REPORT
RANGE_ATTEMPT = face_builder.RANGE_ATTEMPT
CONFIRM_REPORT = face_builder.CONFIRM_REPORT
CONFIRM_ATTEMPT = face_builder.CONFIRM_ATTEMPT

MAX_CUMULATIVE_CPU_SECONDS = face_builder.CPU_LIMIT_SECONDS
RANGE_CPU_LIMIT_SECONDS = 600
CONFIRM_CPU_LIMIT_SECONDS = 600
SEEDS = v11_eval.SEEDS
ARMS = v11_eval.ARMS
KINDS = v11_eval.KINDS


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _face_and_checks() -> tuple[dict, dict]:
    if not FACE.is_file() or not FACE_CHECKS.is_file():
        raise FileNotFoundError("approved N1-122 face and construction checks are required")
    face = read_json(FACE)
    checks = read_json(FACE_CHECKS)
    if (
        checks.get("accepted") is not True
        or not seal_ast.receipt_binding_matches(checks.get("face_sha256"), FACE)
        or face.get("schema") != "n1-hard-neighbor-v12-confirmation-v1"
        or face.get("seed") != 112
        or face.get("items") != 40
        or face.get("observations_per_item") != 4
        or face.get("name_neighbor_distance") != 1
        or len(face.get("episodes", [])) != 40
        or set(face.get("subsets", {})) != {"sacrifice", "confirm"}
        or len(face.get("subsets", {}).get("sacrifice", [])) != 20
        or len(face.get("subsets", {}).get("confirm", [])) != 20
        or checks.get("model_queries") != 0
        or checks.get("parameter_updates") != 0
        or checks.get("plan_sha256") != face_builder.PLAN_SHA256
        or checks.get("source_readiness_sha256") != sha256(READINESS)
        or checks.get("source_exclusion_manifest_sha256") != sha256(SOURCE_EXCLUSIONS)
        or checks.get("checks", {}).get("answer_length_split") is not True
    ):
        raise ValueError("N1-122 face or checks differ from the frozen contract")
    face_sources = face.get("exclusion_sources", {}).get("source_hashes", {})
    if not face_sources:
        raise ValueError("N1-122 face source-hash chain is required")
    for raw_path, expected in face_sources.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"N1-122 face source changed: {path}")
    return face, checks


def _bound_inputs() -> tuple[dict, dict, dict, dict, dict, dict]:
    readiness = face_builder._verify_readiness()
    baseline, candidate, revalidation, prequery, restore, _ = v11_eval._read_bound_inputs()
    v11_scorer_readiness = v11_eval._verify_scorer_readiness()
    face_builder._verify_readiness(readiness)
    return baseline, candidate, revalidation, prequery, restore, v11_scorer_readiness


def _checkpoint_paths(baseline: dict, candidate: dict) -> tuple[dict, dict]:
    return (
        v11_eval._final_paths(baseline, cohort="baseline"),
        v11_eval._final_paths(candidate, cohort="candidate"),
    )


def _score_endpoint(
    g: dict,
    path: Path,
    digest: str,
    protocol: dict,
    seed: int,
    arm: str,
    items: list[dict],
    budget: v11_eval.CpuDeadline,
) -> tuple[list[dict], int, bool, str | None]:
    rows: list[dict] = []
    accounting = {"confirmation_queries": 0}
    model = optimizer = paired = None
    before = None
    error = None
    loaded = False
    try:
        model, optimizer, before, paired = v11_eval._load_model(
            g, path, digest, seed, arm, protocol
        )
        loaded = True
        v11_eval.scorer.evaluate(
            model,
            optimizer,
            seed,
            arm,
            items,
            budget,
            output=rows,
            accounting=accounting,
        )
    except BaseException as exc:  # preserve partial rows and the attempted query count
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if loaded:
            try:
                if paired.fingerprint(model.bundle(optimizer)) != before:
                    error = (
                        "ValueError: frozen scoring changed parameters, optimizer, RNG or history"
                    )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            del model, optimizer
    return rows, accounting["confirmation_queries"], loaded, error


def _all_sources_unchanged(
    source_hashes: dict[str, str], checkpoint_hashes: dict[str, str]
) -> bool:
    return all(
        Path(path).is_file() and sha256(Path(path)) == expected
        for path, expected in {**source_hashes, **checkpoint_hashes}.items()
    )


def _prior_cpu(readiness: dict, checks: dict) -> float:
    return float(readiness.get("cpu_seconds", 0)) + float(checks.get("build_cpu_seconds", 0))


def run_range_check(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-122 range queries require owner approval")
    if RANGE_REPORT.exists() or RANGE_ATTEMPT.exists():
        raise FileExistsError("PLAN-N1-122 range stage is one-shot")
    readiness = _verify_readiness_and_sources()
    face, checks = _face_and_checks()
    baseline, candidate, _, _, _, _ = _bound_inputs()
    _, candidate_paths = _checkpoint_paths(baseline, candidate)
    items = [item for item in face["episodes"] if item["id"] in set(face["subsets"]["sacrifice"])]
    if len(items) != 20:
        raise ValueError("exactly twenty sacrifice items are required")
    prior_cpu = _prior_cpu(readiness, checks)
    remaining = MAX_CUMULATIVE_CPU_SECONDS - prior_cpu
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-122 cumulative CPU budget is exhausted")
    budget = v11_eval.CpuDeadline(min(RANGE_CPU_LIMIT_SECONDS, remaining))
    n120.configure()
    _, g, _ = n120._BASE.make_runner(load_inputs=False)
    checkpoint_hashes = {
        f"{seed}_{arm}": value[1] for (seed, arm), value in candidate_paths.items()
    }
    write_new(
        RANGE_ATTEMPT,
        {
            "schema": "n1-plan122-v12-range-attempt-v1",
            "status": "started",
            "face_sha256": sha256(FACE),
            "readiness_sha256": sha256(READINESS),
            "candidate_checkpoint_hashes": checkpoint_hashes,
            "queried_subset": "sacrifice",
            "confirmation_queried": False,
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    endpoint_rows = []
    total_queries = 0
    model_loads = 0
    error = None
    try:
        for seed in SEEDS:
            for arm in ARMS:
                budget.guard(reserve=5)
                path, digest = candidate_paths[seed, arm]
                rows, queries, loaded, endpoint_error = _score_endpoint(
                    g, path, digest, candidate["protocol"], seed, arm, items, budget
                )
                total_queries += queries
                model_loads += int(loaded)
                endpoint_rows.append({"seed": seed, "arm": arm, "rows": rows})
                if endpoint_error:
                    error = endpoint_error
                    break
            if error:
                break
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"

    summary = None
    if error is None:
        try:
            summary = v11_eval._range_summary(endpoint_rows)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    cumulative_cpu = prior_cpu + budget.used()
    source_hashes = {
        **readiness["source_hashes"],
        **readiness["checkpoint_hashes"],
        **face["exclusion_sources"]["source_hashes"],
        str(READINESS.resolve()): sha256(READINESS),
        str(SOURCE_EXCLUSIONS.resolve()): sha256(SOURCE_EXCLUSIONS),
        str(FACE.resolve()): sha256(FACE),
        str(FACE_ATTEMPT.resolve()): sha256(FACE_ATTEMPT),
        str(FACE_CHECKS.resolve()): sha256(FACE_CHECKS),
        str(RANGE_ATTEMPT.resolve()): sha256(RANGE_ATTEMPT),
    }
    unchanged = _all_sources_unchanged(
        {
            path: digest
            for path, digest in source_hashes.items()
            if path not in readiness["checkpoint_hashes"]
        },
        readiness["checkpoint_hashes"],
    )
    complete = (
        error is None
        and len(endpoint_rows) == 9
        and total_queries == 180
        and sum(len(endpoint["rows"]) for endpoint in endpoint_rows) == 180
        and cumulative_cpu <= MAX_CUMULATIVE_CPU_SECONDS
        and unchanged
    )
    range_passed = bool(summary and summary["range_check_passed"])
    report = {
        "schema": "n1-plan122-v12-sacrifice-range-v1",
        "status": (
            "complete" if complete and range_passed else "range_not_met" if complete else "stopped"
        ),
        "complete": complete,
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "readiness_sha256": sha256(READINESS),
        "source_hashes_unchanged": unchanged,
        "queried_subset": "sacrifice",
        "range_gate": summary,
        "endpoint_results": endpoint_rows,
        "candidate_checkpoint_hashes": checkpoint_hashes,
        "model_loads": model_loads,
        "checkpoint_loads": model_loads,
        "model_queries": total_queries,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "confirmation_queries": 0,
        "cumulative_cpu_seconds": cumulative_cpu,
        "cumulative_cpu_limit_seconds": MAX_CUMULATIVE_CPU_SECONDS,
        "cpu_seconds": round(budget.used(), 6),
        "error": error,
        "can_promote": False,
    }
    write_new(RANGE_REPORT, report)
    return report


def _verify_readiness_and_sources() -> dict:
    readiness = face_builder._verify_readiness()
    v11_eval._verify_scorer_readiness()
    return readiness


def _require_passing_range(range_report: dict) -> None:
    if (
        range_report.get("complete") is not True
        or range_report.get("status") != "complete"
        or range_report.get("range_gate", {}).get("range_check_passed") is not True
        or range_report.get("model_queries") != 180
        or range_report.get("confirmation_queries") != 0
        or range_report.get("queried_subset") != "sacrifice"
        or not seal_ast.receipt_binding_matches(range_report.get("face_sha256"), FACE)
        or range_report.get("readiness_sha256") != sha256(READINESS)
        or range_report.get("source_hashes_unchanged") is not True
    ):
        raise ValueError("the one-shot 180-query range gate must pass before confirm")


def run_confirmation(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-122 confirmation queries require owner approval")
    if CONFIRM_REPORT.exists() or CONFIRM_ATTEMPT.exists():
        raise FileExistsError("PLAN-N1-122 confirmation stage is one-shot")
    if not RANGE_REPORT.is_file():
        raise FileNotFoundError("a completed range report is required before confirm")
    range_report = read_json(RANGE_REPORT)
    readiness = _verify_readiness_and_sources()
    face, checks = _face_and_checks()
    _require_passing_range(range_report)
    baseline, candidate, _, _, _, _ = _bound_inputs()
    baseline_paths, candidate_paths = _checkpoint_paths(baseline, candidate)
    confirm_ids = set(face["subsets"]["confirm"])
    items = [item for item in face["episodes"] if item["id"] in confirm_ids]
    if len(items) != 20 or confirm_ids & set(face["subsets"]["sacrifice"]):
        raise ValueError("exact disjoint twenty-item confirm subset required")
    prior_cpu = _prior_cpu(readiness, checks) + float(range_report["cpu_seconds"])
    remaining = MAX_CUMULATIVE_CPU_SECONDS - prior_cpu
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-122 cumulative CPU budget is exhausted")
    budget = v11_eval.CpuDeadline(min(CONFIRM_CPU_LIMIT_SECONDS, remaining))
    write_new(
        CONFIRM_ATTEMPT,
        {
            "schema": "n1-plan122-v12-confirm-attempt-v1",
            "status": "started",
            "face_sha256": sha256(FACE),
            "range_report_sha256": sha256(RANGE_REPORT),
            "readiness_sha256": sha256(READINESS),
            "baseline_report_sha256": v11_eval.BASELINE_REPORT_SHA256,
            "candidate_report_sha256": v11_eval.CANDIDATE_REPORT_SHA256,
            "queried_subset": "confirm",
            "parameter_updates": 0,
            "optimizer_steps": 0,
            "can_promote": False,
        },
    )
    cohort_rows = {"baseline": [], "candidate": []}
    total_queries = 0
    model_loads = 0
    errors = []

    def score_cohort(label: str, paths: dict, protocol: dict, make_runner) -> None:
        nonlocal total_queries, model_loads
        _, g, _ = make_runner()
        for seed in SEEDS:
            for arm in ARMS:
                budget.guard(reserve=5)
                path, digest = paths[seed, arm]
                rows, queries, loaded, error = _score_endpoint(
                    g, path, digest, protocol, seed, arm, items, budget
                )
                total_queries += queries
                model_loads += int(loaded)
                cohort_rows[label].extend(rows)
                if error:
                    errors.append(f"{label}/{seed}/{arm}: {error}")
                    return

    try:
        # The baseline loader has its own optimizer policy; score it before switching to N1-120.
        score_cohort(
            "baseline",
            baseline_paths,
            baseline["protocol"],
            lambda: baseline_training.make_runner(load_inputs=False),
        )
        if not errors:
            n120.configure()
            score_cohort(
                "candidate",
                candidate_paths,
                candidate["protocol"],
                lambda: n120._BASE.make_runner(load_inputs=False),
            )
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    baseline_gates = None
    candidate_gates = None
    if not errors:
        try:
            baseline_gates = v11_eval._cohort_gates(cohort_rows["baseline"], "baseline")
            candidate_gates = v11_eval._cohort_gates(cohort_rows["candidate"], "candidate")
        except BaseException as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    cumulative_cpu = prior_cpu + budget.used()
    source_hashes = {
        **readiness["source_hashes"],
        **readiness["checkpoint_hashes"],
        **face["exclusion_sources"]["source_hashes"],
        str(READINESS.resolve()): sha256(READINESS),
        str(SOURCE_EXCLUSIONS.resolve()): sha256(SOURCE_EXCLUSIONS),
        str(FACE.resolve()): sha256(FACE),
        str(FACE_ATTEMPT.resolve()): sha256(FACE_ATTEMPT),
        str(FACE_CHECKS.resolve()): sha256(FACE_CHECKS),
        str(RANGE_ATTEMPT.resolve()): sha256(RANGE_ATTEMPT),
        str(CONFIRM_ATTEMPT.resolve()): sha256(CONFIRM_ATTEMPT),
        str(RANGE_REPORT.resolve()): sha256(RANGE_REPORT),
    }
    unchanged = _all_sources_unchanged(
        {
            path: digest
            for path, digest in source_hashes.items()
            if path not in readiness["checkpoint_hashes"]
        },
        readiness["checkpoint_hashes"],
    )
    complete = (
        not errors
        and len(cohort_rows["baseline"]) == 180
        and len(cohort_rows["candidate"]) == 180
        and total_queries == 360
        and cumulative_cpu <= MAX_CUMULATIVE_CPU_SECONDS
        and unchanged
    )
    report = {
        "schema": "n1-plan122-v12-confirmation-result-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "range_report_sha256": sha256(RANGE_REPORT),
        "readiness_sha256": sha256(READINESS),
        "baseline_report_sha256": v11_eval.BASELINE_REPORT_SHA256,
        "candidate_report_sha256": v11_eval.CANDIDATE_REPORT_SHA256,
        "baseline_gates": baseline_gates,
        "candidate_gates": candidate_gates,
        "baseline_rows": cohort_rows["baseline"],
        "candidate_rows": cohort_rows["candidate"],
        "confirmation_queries": total_queries,
        "range_queries": range_report["model_queries"],
        "total_model_queries": range_report["model_queries"] + total_queries,
        "model_loads": model_loads,
        "checkpoint_loads": model_loads,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "source_hashes_unchanged": unchanged,
        "cumulative_cpu_seconds": cumulative_cpu,
        "cumulative_cpu_limit_seconds": MAX_CUMULATIVE_CPU_SECONDS,
        "cpu_seconds": round(budget.used(), 6),
        "can_promote": False,
        "errors": errors,
    }
    write_new(CONFIRM_REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--range-check", action="store_true")
    modes.add_argument("--confirm", action="store_true")
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    if args.range_check:
        result = run_range_check(owner_approved=args.owner_approved)
    else:
        result = run_confirmation(owner_approved=args.owner_approved)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "range_gate",
                    "model_queries",
                    "confirmation_queries",
                    "total_model_queries",
                    "baseline_gates",
                    "candidate_gates",
                    "cumulative_cpu_seconds",
                    "can_promote",
                    "error",
                    "errors",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
