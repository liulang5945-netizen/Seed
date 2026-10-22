"""PLAN-N1-101 v8 sacrifice range check and one-shot paired confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import (  # noqa: E402
    build_taiji_n1_hard_neighbor_v8 as face_builder,
)
from scripts.training import (
    eval_taiji_n1_transition_factorial_confirmation as base85,
)
from scripts.training import (
    eval_taiji_n1_transition_factorial_confirmation_run89 as old_bind,
)
from scripts.training import (
    eval_taiji_n1_transition_factorial_confirmation_v6 as v6_gates,
)
from scripts.training import (
    run_taiji_n1_hard_neighbor_training as training,
)
from scripts.training import seal_ast  # noqa: E402

FACE = face_builder.FACE
FACE_CHECKS = face_builder.CHECKS
RANGE_REPORT = ROOT / "reports/taiji_n1_hard_neighbor_v8_range_check_20261017.json"
RANGE_ATTEMPT = RANGE_REPORT.with_name(RANGE_REPORT.stem + "_attempt.json")
CONFIRM_REPORT = ROOT / "reports/taiji_n1_hard_neighbor_v8_confirmation_20261017.json"
CONFIRM_ATTEMPT = CONFIRM_REPORT.with_name(CONFIRM_REPORT.stem + "_attempt.json")
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")
KINDS = ("lookup", "revision", "transition", "unknown")
RANGE_CPU_LIMIT_SECONDS = 180
CONFIRM_CPU_LIMIT_SECONDS = 600
MIN_USABLE_ENDPOINTS = 6
TRIPLE_ERROR_FLOOR = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
        stream.flush()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class CpuDeadline:
    def __init__(self, seconds: int):
        self.start = time.process_time()
        self.limit = seconds

    def used(self) -> float:
        return time.process_time() - self.start

    def guard(self, reserve: int = 0) -> None:
        if self.used() + reserve >= self.limit:
            raise TimeoutError("PLAN-N1-101 independent CPU limit reached")


def load_face() -> tuple[dict, dict]:
    if not FACE.is_file() or not FACE_CHECKS.is_file():
        raise FileNotFoundError("accepted v8 face and construction report are required")
    face = read_json(FACE)
    checks = read_json(FACE_CHECKS)
    if (
        checks.get("accepted") is not True
        or not seal_ast.receipt_binding_matches(checks.get("face_sha256"), FACE)
        or checks.get("training_report_sha256") != sha256(training.REPORT)
        or face.get("schema") != "n1-hard-neighbor-v8-confirmation-v1"
        or face.get("seed") != 108
        or face.get("observations_per_item") != 8
        or face.get("name_neighbor_distance") != 1
        or checks.get("plan_sha256") != face_builder.sha256(face_builder.PLAN)
        or face.get("exclusion_sources", {}).get("plan_sha256")
        != face_builder.sha256(face_builder.PLAN)
        or len(face.get("episodes", [])) != 40
        or len(face.get("subsets", {}).get("sacrifice", [])) != 20
        or len(face.get("subsets", {}).get("confirm", [])) != 20
        or checks.get("v7_confirm_payload_fields_decoded") is not False
        or checks.get("v7_confirm_model_queries") != 0
    ):
        raise ValueError("v8 face/check report binding or denominator differs")
    if any(episode.get("split") != "holdout" for episode in face["episodes"]):
        raise ValueError("v8 must remain an explicitly held-out face")
    if set(face["subsets"]["sacrifice"]) & set(face["subsets"]["confirm"]):
        raise ValueError("v8 sacrifice and confirm subsets overlap")
    if set(face["subsets"]["sacrifice"]) | set(face["subsets"]["confirm"]) != {
        episode["id"] for episode in face["episodes"]
    }:
        raise ValueError("v8 subsets must partition all 40 face items")
    return face, checks


def training_endpoints() -> tuple[dict, dict, dict]:
    report = read_json(training.REPORT)
    if (
        not report.get("complete")
        or not report.get("training_sufficient")
        or len(report.get("models", [])) != 9
        or report.get("can_promote") is not False
        or any(
            report.get(key) for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
    ):
        raise ValueError("complete all-gates-passing PLAN-N1-100 training report required")
    paths = {}
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    for model in report["models"]:
        key = (model.get("seed"), model.get("arm"))
        if key not in expected or key in paths:
            raise ValueError("duplicate or unexpected PLAN-N1-100 endpoint")
        finals = [
            check
            for check in model.get("restore_checks", [])
            if Path(check.get("path", "")).name.endswith("_final.pt")
        ]
        if len(finals) != 1 or finals[0].get("equal") is not True:
            raise ValueError(f"one independently restored final endpoint required for {key}")
        path = Path(finals[0]["path"])
        if not path.is_file() or sha256(path) != finals[0].get("sha256"):
            raise ValueError(f"sealed PLAN-N1-100 checkpoint changed for {key}")
        paths[key] = path
    if set(paths) != expected:
        raise ValueError("all nine final training endpoints are required")
    return report, paths, {str(path.resolve()): sha256(path) for path in paths.values()}


def _new_model(g: dict, path: Path, seed: int, arm: str):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("protocol") != g["PROTOCOL"]
        or payload.get("seed") != seed
        or payload.get("arm") != arm
        or payload.get("completed_updates") != training.UPDATES
        or payload.get("phase") != "final"
    ):
        raise ValueError(f"PLAN-N1-100 endpoint marker differs for {seed}/{arm}")
    model, optimizer = g["load"](payload)
    if g["paired"].fingerprint(model.bundle(optimizer)) != payload["expected_signature"]["bundle"]:
        raise ValueError(f"PLAN-N1-100 endpoint full bundle differs for {seed}/{arm}")
    return model, optimizer


def _old_endpoints() -> tuple[dict, dict]:
    receipts = old_bind.family_receipts()
    paths, guards = old_bind.endpoints(receipts)
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(paths) != expected:
        raise ValueError("the frozen run88c/run89a/run89b nine-endpoint family is incomplete")
    return paths, guards


def _score(
    model, optimizer, seed: int, arm: str, items: list[dict], budget: CpuDeadline
) -> list[dict]:
    rows = base85.evaluate(model, optimizer, seed, arm, items, budget)
    if len(rows) != len(items) or {row["id"] for row in rows} != {item["id"] for item in items}:
        raise ValueError("each endpoint must return exactly one score per selected v8 item")
    return rows


def range_lines(results: list[dict]) -> dict:
    if len(results) != 9:
        raise ValueError("range check needs exactly nine new endpoints")
    if any(len(result.get("rows", [])) != 20 for result in results):
        raise ValueError("range check needs exactly twenty sacrifice rows per endpoint")
    endpoints = {
        (result["seed"], result["arm"]): sum(
            row["kind"] != "unknown" and not row["free_exact"] for row in result["rows"]
        )
        for result in results
    }
    usable = sum(errors >= TRIPLE_ERROR_FLOOR for errors in endpoints.values())
    per_class = {
        kind: {
            "mean_errors": round(
                sum(
                    sum(row["kind"] == kind and not row["free_exact"] for row in result["rows"])
                    for result in results
                )
                / len(results),
                4,
            ),
            "reported_only": True,
        }
        for kind in KINDS
    }
    return {
        "range_check_passed": usable >= MIN_USABLE_ENDPOINTS,
        "usable_endpoints": usable,
        "endpoint_count": len(results),
        "decodable_items_per_endpoint": 15,
        "triple_error_floor": TRIPLE_ERROR_FLOOR,
        "minimum_usable_endpoints": MIN_USABLE_ENDPOINTS,
        "per_class": per_class,
        "addressable_errors_by_endpoint": {
            f"{seed}_{arm}": errors for (seed, arm), errors in sorted(endpoints.items())
        },
    }


def range_readiness() -> dict:
    face, checks = load_face()
    report, paths, checkpoint_guards = training_endpoints()
    sacrifice_ids = set(face["subsets"]["sacrifice"])
    items = [episode for episode in face["episodes"] if episode["id"] in sacrifice_ids]
    if len(items) != 20 or set(face["subsets"]["confirm"]) & {item["id"] for item in items}:
        raise ValueError("only the twenty-item sacrifice subset may be passed to range check")
    return {
        "face": face,
        "checks": checks,
        "training_report": report,
        "paths": paths,
        "checkpoint_guards": checkpoint_guards,
        "items": items,
    }


def run_range_check() -> dict:
    if RANGE_REPORT.exists() or RANGE_ATTEMPT.exists():
        raise FileExistsError("v8 sacrifice range check is one-shot")
    ready = range_readiness()
    loaded, g, _ = training.make_runner(load_inputs=False)
    budget = CpuDeadline(RANGE_CPU_LIMIT_SECONDS)
    source_guards = {
        "training_report_sha256": sha256(training.REPORT),
        "plan_sha256": sha256(face_builder.PLAN),
        "training_code_identity": ready["training_report"]["code_identity"],
        "schedule_sha256": training.SCHEDULE_SHA256,
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(face_builder.__file__).resolve()),
        "new_checkpoints": ready["checkpoint_guards"],
    }
    raw_guards = {
        str(path.resolve()): sha256(path)
        for path in (
            training.REPORT,
            training.SCHEDULE,
            FACE,
            FACE_CHECKS,
            Path(__file__).resolve(),
            Path(training.__file__).resolve(),
            Path(face_builder.__file__).resolve(),
            face_builder.PLAN,
        )
    }
    initial_identity = g["identity"]()
    write_new(
        RANGE_ATTEMPT,
        {
            "schema": "n1-hard-neighbor-v8-range-attempt-v1",
            "status": "started",
            "source_guards": source_guards,
            "raw_source_guards": raw_guards,
            "queried_subsets": ["sacrifice"],
            "confirm_subset_queried": False,
            "parameter_updates": 0,
            "v3_dev_queries": 0,
            "v3_holdout_queries": 0,
            "v8_confirm_model_queries": 0,
        },
    )
    results = []
    error = None
    try:
        for seed in SEEDS:
            for arm in ARMS:
                budget.guard(reserve=8)
                model, optimizer = _new_model(g, ready["paths"][seed, arm], seed, arm)
                rows = _score(model, optimizer, seed, arm, ready["items"], budget)
                results.append({"seed": seed, "arm": arm, "rows": rows})
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    gate = None
    if error is None:
        gate = range_lines(results)
    raw_sources_unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest for path, digest in raw_guards.items()
    )
    checkpoints_unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in ready["checkpoint_guards"].items()
    )
    code_unchanged = g["identity"]() == initial_identity
    complete = (
        error is None
        and len(results) == 9
        and budget.used() <= RANGE_CPU_LIMIT_SECONDS
        and raw_sources_unchanged
        and checkpoints_unchanged
        and code_unchanged
    )
    payload = {
        "schema": "n1-hard-neighbor-v8-range-result-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "range_check_passed": bool(gate and gate["range_check_passed"]),
        "range_gate": gate,
        "face_sha256": sha256(FACE),
        "face": str(FACE),
        "queried_subsets": ["sacrifice"],
        "confirm_subset_queried": False,
        "v8_sacrifice_model_queries": sum(len(entry["rows"]) for entry in results),
        "endpoint_results": results,
        "source_guards": source_guards,
        "raw_source_guards": raw_guards,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "v3_dev_queries": 0,
        "v3_holdout_queries": 0,
        "v8_confirm_model_queries": 0,
        "sources_unchanged": raw_sources_unchanged and checkpoints_unchanged,
        "code_unchanged": code_unchanged,
        "cpu_limit_seconds": RANGE_CPU_LIMIT_SECONDS,
        "cpu_seconds": round(budget.used(), 6),
        "can_promote": False,
        "error": error,
    }
    if complete and not payload["range_check_passed"]:
        payload["status"] = "range_not_met"
    write_new(RANGE_REPORT, payload)
    return payload


def _verify_range_pass() -> tuple[dict, dict]:
    if not RANGE_REPORT.is_file():
        raise FileNotFoundError("completed v8 sacrifice range report is required")
    report = read_json(RANGE_REPORT)
    if (
        report.get("complete") is not True
        or report.get("status") != "complete"
        or report.get("range_check_passed") is not True
        or report.get("confirm_subset_queried") is not False
        or report.get("queried_subsets") != ["sacrifice"]
        or report.get("v8_sacrifice_model_queries") != 180
        or report.get("cpu_seconds", RANGE_CPU_LIMIT_SECONDS + 1) > RANGE_CPU_LIMIT_SECONDS
        or not seal_ast.receipt_binding_matches(report.get("face_sha256"), FACE)
    ):
        raise ValueError("v8 range gate has not passed with an intact sacrifice-only receipt")
    face, checks = load_face()
    if report.get("source_guards", {}).get("face_checks_sha256") != sha256(FACE_CHECKS):
        raise ValueError("v8 face construction report changed after range check")
    return report, {"face": face, "checks": checks}


def run_confirmation() -> dict:
    if CONFIRM_REPORT.exists() or CONFIRM_ATTEMPT.exists():
        raise FileExistsError("v8 confirmation is one-shot")
    range_report, face_bundle = _verify_range_pass()
    face = face_bundle["face"]
    confirm_ids = set(face["subsets"]["confirm"])
    items = [episode for episode in face["episodes"] if episode["id"] in confirm_ids]
    if len(items) != 20 or {episode["id"] for episode in items} & set(face["subsets"]["sacrifice"]):
        raise ValueError("exact disjoint twenty-item v8 confirmation subset required")
    train_report, new_paths, new_checkpoint_guards = training_endpoints()
    old_paths, old_guards = _old_endpoints()
    _, g, _ = training.make_runner(load_inputs=False)
    budget = CpuDeadline(CONFIRM_CPU_LIMIT_SECONDS)
    code_identity = {
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "face_builder_sha256": sha256(Path(face_builder.__file__).resolve()),
        "training_controller_sha256": sha256(Path(training.__file__).resolve()),
        "training_code_identity": train_report["code_identity"],
        "old_family_binding_sha256": sha256(Path(old_bind.__file__).resolve()),
        "v6_gate_evaluator_sha256": sha256(Path(v6_gates.__file__).resolve()),
    }
    source_guards = {
        "training_report_sha256": sha256(training.REPORT),
        "plan_sha256": sha256(face_builder.PLAN),
        "range_report_sha256": sha256(RANGE_REPORT),
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "new_checkpoints": new_checkpoint_guards,
        "old_endpoint_guards": old_guards,
        "old_family_reports": {
            arm: receipts[arm]["report_sha256"]
            for arm, receipts in old_bind.family_receipts().items()
        },
    }
    raw_guards = {
        str(path.resolve()): sha256(path)
        for path in (
            training.REPORT,
            training.SCHEDULE,
            RANGE_REPORT,
            FACE,
            FACE_CHECKS,
            Path(__file__).resolve(),
            Path(training.__file__).resolve(),
            Path(face_builder.__file__).resolve(),
            face_builder.PLAN,
            Path(old_bind.__file__).resolve(),
            Path(v6_gates.__file__).resolve(),
        )
    }
    write_new(
        CONFIRM_ATTEMPT,
        {
            "schema": "n1-hard-neighbor-v8-confirm-attempt-v1",
            "status": "started",
            "code_identity": code_identity,
            "source_guards": source_guards,
            "raw_source_guards": raw_guards,
            "range_report_sha256": sha256(RANGE_REPORT),
            "queried_subsets": ["confirm"],
            "v8_confirm_model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    rows = []
    restores = []
    error = None
    try:
        for seed in SEEDS:
            for arm in ARMS:
                budget.guard(reserve=8)
                new_model, new_optimizer = _new_model(g, new_paths[seed, arm], seed, arm)
                current = g["paired"].fingerprint(new_model.bundle(new_optimizer))
                new_rows = _score(new_model, new_optimizer, seed, arm, items, budget)
                if g["paired"].fingerprint(new_model.bundle(new_optimizer)) != current:
                    raise ValueError(f"new v8 endpoint changed while scoring {seed}/{arm}")
                rows.extend(dict(row, family="hard_neighbor_v8") for row in new_rows)
                restores.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "family": "hard_neighbor_v8",
                        "full_bundle_equal": True,
                    }
                )

                old_path = old_paths[seed, arm]
                old_payload = torch.load(old_path, map_location="cpu", weights_only=True)
                if (
                    old_payload.get("seed") != seed
                    or old_payload.get("arm") != arm
                    or old_payload.get("completed_updates") != training.UPDATES
                ):
                    raise ValueError(f"old descriptive endpoint marker differs for {seed}/{arm}")
                old_model, old_optimizer = old_bind.load_with_family_protocol(old_payload)
                before = base85.paired.fingerprint(old_model.bundle(old_optimizer))
                old_rows_for_endpoint = _score(old_model, old_optimizer, seed, arm, items, budget)
                if base85.paired.fingerprint(old_model.bundle(old_optimizer)) != before:
                    raise ValueError(f"old descriptive endpoint changed while scoring {seed}/{arm}")
                rows.extend(
                    dict(row, family="prior_family_descriptive") for row in old_rows_for_endpoint
                )
                restores.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "family": "prior_family_descriptive",
                        "full_bundle_equal": True,
                    }
                )

        expected = {
            (seed, arm, item["id"], family)
            for seed in SEEDS
            for arm in ARMS
            for item in items
            for family in ("hard_neighbor_v8", "prior_family_descriptive")
        }
        actual = {(row["seed"], row["arm"], row["id"], row["family"]) for row in rows}
        if len(rows) != 360 or actual != expected or len(actual) != 360:
            raise ValueError("v8 confirmation must contain exactly 360 unique rows")
        new_rows = [row for row in rows if row["family"] == "hard_neighbor_v8"]
        if any(
            sum(
                row["kind"] == kind for row in new_rows if row["seed"] == seed and row["arm"] == arm
            )
            != 5
            for seed in SEEDS
            for arm in ARMS
            for kind in KINDS
        ):
            raise ValueError("new-family confirmation cells must contain five items per kind")
        adjudication = v6_gates.evaluate_gates(new_rows)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        adjudication = None

    unchanged_new = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in new_checkpoint_guards.items()
    )
    unchanged_old = all(
        Path(path).is_file() and old_bind.lr88.guard_sha256(Path(path)) == digest
        for path, digest in old_guards.items()
    )
    raw_sources_unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest for path, digest in raw_guards.items()
    )
    complete = (
        error is None
        and len(rows) == 360
        and len(restores) == 18
        and budget.used() <= CONFIRM_CPU_LIMIT_SECONDS
        and unchanged_new
        and unchanged_old
        and raw_sources_unchanged
    )
    payload = {
        "schema": "n1-hard-neighbor-v8-confirmation-result-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "passed": bool(complete and adjudication and adjudication["all_four_gates"]),
        "training_report_sha256": sha256(training.REPORT),
        "range_report_sha256": sha256(RANGE_REPORT),
        "face_sha256": sha256(FACE),
        "queried_subsets": ["confirm"],
        "confirm_items": len(items),
        "model_queries": len(rows),
        "v8_confirm_model_queries": len(rows),
        "v8_new_family_model_queries": sum(row["family"] == "hard_neighbor_v8" for row in rows),
        "prior_family_descriptive_model_queries": sum(
            row["family"] == "prior_family_descriptive" for row in rows
        ),
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "restore_checks": restores,
        "rows": rows,
        "new_family_adjudication": adjudication,
        "old_family_role": "same-face descriptive controls only",
        "source_guards": source_guards,
        "raw_source_guards": raw_guards,
        "sources_unchanged": unchanged_new and unchanged_old and raw_sources_unchanged,
        "code_identity": code_identity,
        "cpu_limit_seconds": CONFIRM_CPU_LIMIT_SECONDS,
        "cpu_seconds": round(budget.used(), 6),
        "can_promote": False,
        "TRAIN_SHA": None,
        "error": error,
    }
    write_new(CONFIRM_REPORT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--range-readiness", action="store_true")
    modes.add_argument("--range-check", action="store_true")
    modes.add_argument("--confirm-readiness", action="store_true")
    modes.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.range_readiness:
        ready = range_readiness()
        print(
            json.dumps(
                {
                    "sacrifice_items": len(ready["items"]),
                    "confirm_items": 20,
                    "endpoints": len(ready["paths"]),
                    "confirm_subset_queried": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.range_check:
        result = run_range_check()
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "complete",
                        "status",
                        "range_check_passed",
                        "range_gate",
                        "cpu_seconds",
                        "error",
                    )
                },
                sort_keys=True,
            )
        )
        return 0 if result["complete"] else 1
    if args.confirm_readiness:
        report, _ = _verify_range_pass()
        print(
            json.dumps(
                {
                    "range_report_sha256": sha256(RANGE_REPORT),
                    "range_check_passed": report["range_check_passed"],
                    "confirm_items": 20,
                    "model_queries": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    result = run_confirmation()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in ("complete", "passed", "new_family_adjudication", "cpu_seconds", "error")
            },
            sort_keys=True,
        )
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
