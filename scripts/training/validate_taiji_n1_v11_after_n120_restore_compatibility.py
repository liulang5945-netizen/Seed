"""Restore-only compatibility check for both PLAN-N1-121 model cohorts."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import (
    run_taiji_n1_expanded_normalized_rwl_length_balanced as n120,
)  # noqa: E402
from scripts.training import run_taiji_n1_hard_neighbor_training as baseline_training  # noqa: E402

BASELINE_REPORT = ROOT / "reports/taiji_n1_hard_neighbor_training_20261017.json"
BASELINE_REPORT_SHA256 = "dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909"
CANDIDATE_REPORT = ROOT / "reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019.json"
CANDIDATE_REPORT_SHA256 = "873c237623fa7b97de0f4d8d95edaf4a5d8f9bf944c8af301468bd1d0f19ab2f"
CANDIDATE_REVALIDATION = (
    ROOT
    / "reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019_receipt_revalidation.json"
)
CANDIDATE_REVALIDATION_SHA256 = "22055b74784e7c35891098595928af238561405a52cbfb4b94eb87491ba7e03d"
N121_PLAN = (
    ROOT / "plans/reference/PLAN-N1-121_v11_confirmation_after_n120_training_draft_20261019.md"
)
N121_PLAN_SHA256 = "5c2ac778e8e205e1574a2bd93d77dfc1c597490aced4db7a0b0f6d2efbbfefb8"
PREQUERY_READINESS = ROOT / "reports/taiji_n1_v11_after_n120_prequery_readiness_20261019.json"
PREQUERY_READINESS_SHA256 = "5a9883ec5dd924f13279c59c23ecdd7eaf89f54062ba237d60a85e03e33da82b"
OUTPUT = ROOT / "reports/taiji_n1_v11_after_n120_restore_readiness_20261019.json"
CPU_LIMIT_SECONDS = 180
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(path: Path, expected_sha256: str) -> dict:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError(f"frozen PLAN-N1-121 source changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def restore_cohort(report: dict, runner_globals: dict, *, label: str) -> list[dict]:
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    models = report.get("models", [])
    if len(models) != 9 or {(model.get("seed"), model.get("arm")) for model in models} != expected:
        raise ValueError(f"complete nine-endpoint {label} cohort required")
    results = []
    for model_record in sorted(models, key=lambda value: (value["seed"], value["arm"])):
        seed, arm = model_record["seed"], model_record["arm"]
        checks = [
            check
            for check in model_record.get("restore_checks", [])
            if Path(check.get("path", "")).name == f"s{seed}_{arm}_final.pt"
        ]
        if len(checks) != 1 or checks[0].get("equal") is not True:
            raise ValueError(f"sealed final restore receipt required for {label}/{seed}/{arm}")
        path = Path(checks[0]["path"])
        expected_file_sha = checks[0].get("sha256")
        if not path.is_file() or sha256(path) != expected_file_sha:
            raise ValueError(f"checkpoint changed before restore: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("protocol") != report["protocol"]
            or payload.get("seed") != seed
            or payload.get("arm") != arm
            or payload.get("phase") != "final"
            or payload.get("completed_updates") != 24576
        ):
            raise ValueError(f"checkpoint identity mismatch: {label}/{seed}/{arm}")
        model, optimizer = runner_globals["load"](payload)
        paired = runner_globals["paired"]
        fingerprint = paired.fingerprint(model.bundle(optimizer))
        expected_bundle = payload.get("expected_signature", {}).get("bundle")
        if not expected_bundle or fingerprint != expected_bundle:
            raise ValueError(
                f"full model/optimizer/RNG/history restore mismatch: {label}/{seed}/{arm}"
            )
        results.append(
            {
                "seed": seed,
                "arm": arm,
                "path": str(path),
                "checkpoint_sha256": expected_file_sha,
                "full_bundle_equal": True,
            }
        )
        del model, optimizer, payload
    return results


def audit() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"write-once restore-readiness report exists: {OUTPUT}")
    started = time.process_time()
    baseline = read_source(BASELINE_REPORT, BASELINE_REPORT_SHA256)
    candidate = read_source(CANDIDATE_REPORT, CANDIDATE_REPORT_SHA256)
    revalidation = read_source(CANDIDATE_REVALIDATION, CANDIDATE_REVALIDATION_SHA256)
    prequery = read_source(PREQUERY_READINESS, PREQUERY_READINESS_SHA256)
    if sha256(N121_PLAN) != N121_PLAN_SHA256:
        raise ValueError("frozen PLAN-N1-121 changed")
    if (
        baseline.get("complete") is not True
        or baseline.get("training_sufficient") is not True
        or candidate.get("status") != "stopped"
        or revalidation.get("complete") is not True
        or revalidation.get("training_sufficient") is not True
        or revalidation.get("independent_restore_checks") != 18
        or prequery.get("complete") is not True
        or prequery.get("checkpoint_files_hashed") != 36
        or prequery.get("confirm_face_generated") is not False
        or prequery.get("confirm_face_consumed") is not False
        or any(
            report.get(key) != 0
            for report in (baseline, candidate)
            for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
    ):
        raise ValueError("matched baseline and revalidated candidate cohorts required")

    _, baseline_globals, _ = baseline_training.make_runner(load_inputs=False)
    baseline_restores = restore_cohort(baseline, baseline_globals, label="PLAN-N1-100")
    n120.configure()
    _, candidate_globals, _ = n120._BASE.make_runner(load_inputs=False)
    candidate_restores = restore_cohort(candidate, candidate_globals, label="PLAN-N1-120")
    cpu_seconds = round(time.process_time() - started, 6)
    total_restores = len(baseline_restores) + len(candidate_restores)
    complete = total_restores == 18 and cpu_seconds <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan121-v11-after-n120-restore-readiness-v1",
        "status": "ready_for_owner_approval" if complete else "stopped",
        "complete": complete,
        "baseline_report_sha256": BASELINE_REPORT_SHA256,
        "candidate_report_sha256": CANDIDATE_REPORT_SHA256,
        "candidate_receipt_revalidation_sha256": CANDIDATE_REVALIDATION_SHA256,
        "n121_plan_sha256": N121_PLAN_SHA256,
        "prequery_readiness_sha256": PREQUERY_READINESS_SHA256,
        "baseline_restores": baseline_restores,
        "candidate_restores": candidate_restores,
        "independent_full_bundle_restores": total_restores,
        "all_full_bundles_equal": all(
            item["full_bundle_equal"] for item in baseline_restores + candidate_restores
        ),
        "checkpoint_loads": total_restores,
        "model_loads": total_restores,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "confirm_face_generated": False,
        "confirm_face_consumed": False,
        "can_promote": False,
        "cpu_seconds": cpu_seconds,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    return report


if __name__ == "__main__":
    result = audit()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "complete",
                    "status",
                    "independent_full_bundle_restores",
                    "checkpoint_loads",
                    "model_queries",
                    "parameter_updates",
                    "cpu_seconds",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
