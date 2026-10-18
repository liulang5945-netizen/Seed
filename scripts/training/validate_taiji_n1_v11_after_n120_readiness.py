"""Zero-query readiness check for the proposed PLAN-N1-121 confirmation."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASELINE_REPORT = ROOT / "reports/taiji_n1_hard_neighbor_training_20261017.json"
BASELINE_REPORT_SHA256 = "dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909"
CANDIDATE_REPORT = ROOT / "reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019.json"
CANDIDATE_REPORT_SHA256 = "873c237623fa7b97de0f4d8d95edaf4a5d8f9bf944c8af301468bd1d0f19ab2f"
CANDIDATE_REVALIDATION = (
    ROOT
    / "reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019_receipt_revalidation.json"
)
CANDIDATE_REVALIDATION_SHA256 = "22055b74784e7c35891098595928af238561405a52cbfb4b94eb87491ba7e03d"
N110_PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-110_length_balanced_training_and_v11_confirmation_draft_20261017.md"
)
N110_PLAN_SHA256 = "16d0cecb325af6463c107973fc0e741478140134a7642cd64c86e045c6069783"
N121_PLAN = (
    ROOT / "plans/reference/PLAN-N1-121_v11_confirmation_after_n120_training_draft_20261019.md"
)
N121_PLAN_SHA256 = "5c2ac778e8e205e1574a2bd93d77dfc1c597490aced4db7a0b0f6d2efbbfefb8"
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v11_confirmation_seed111_20261019.json"
READINESS = ROOT / "reports/taiji_n1_v11_after_n120_prequery_readiness_20261019.json"
CPU_LIMIT_SECONDS = 60
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound(path: Path, expected: str) -> dict:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"frozen PLAN-N1-121 source hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_receipts(report: dict, cohort: str) -> dict[str, str]:
    models = report.get("models", [])
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if len(models) != 9 or {(m.get("seed"), m.get("arm")) for m in models} != expected:
        raise ValueError(f"complete nine-endpoint {cohort} report required")
    hashes = {}
    for model in models:
        if model.get("complete") is not True or model.get("completed_updates") != 24576:
            raise ValueError(f"complete 24,576-update {cohort} endpoint required")
        restores = model.get("restore_checks", [])
        if len(restores) != 2 or any(check.get("equal") is not True for check in restores):
            raise ValueError(
                f"zero/final restore pair must pass for {cohort}/{model['seed']}/{model['arm']}"
            )
        names = {Path(check.get("path", "")).name for check in restores}
        expected_names = {
            f"s{model['seed']}_{model['arm']}_zero.pt",
            f"s{model['seed']}_{model['arm']}_final.pt",
        }
        if names != expected_names:
            raise ValueError(f"exact zero/final restore paths required for {cohort}")
        for check in restores:
            path = Path(check["path"])
            expected_hash = check.get("sha256")
            if not path.is_file() or sha256(path) != expected_hash:
                raise ValueError(f"checkpoint hash mismatch in {cohort}: {path}")
            hashes[str(path.resolve())] = expected_hash
    if len(hashes) != 18:
        raise ValueError(f"18 unique full-bundle checkpoint hashes required for {cohort}")
    return hashes


def audit() -> dict:
    if READINESS.exists():
        raise FileExistsError(f"write-once readiness report exists: {READINESS}")
    started = time.process_time()
    baseline = load_bound(BASELINE_REPORT, BASELINE_REPORT_SHA256)
    candidate = load_bound(CANDIDATE_REPORT, CANDIDATE_REPORT_SHA256)
    revalidation = load_bound(CANDIDATE_REVALIDATION, CANDIDATE_REVALIDATION_SHA256)
    n110_hash = sha256(N110_PLAN)
    n121_hash = sha256(N121_PLAN)
    if (
        n110_hash != N110_PLAN_SHA256
        or n121_hash != N121_PLAN_SHA256
        or baseline.get("complete") is not True
        or baseline.get("training_sufficient") is not True
        or baseline.get("can_promote") is not False
        or candidate.get("status") != "stopped"
        or candidate.get("completed_updates") != 221184
        or candidate.get("sources_unchanged") is not True
        or candidate.get("code_unchanged") is not True
        or revalidation.get("complete") is not True
        or revalidation.get("training_sufficient") is not True
        or revalidation.get("model_receipts_validated") != 9
        or revalidation.get("independent_restore_checks") != 18
        or revalidation.get("can_promote") is not False
        or any(
            report.get(key) != 0
            for report in (baseline, candidate)
            for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
    ):
        raise ValueError("matched, gate-passing baseline/candidate cohorts are required")

    if FACE.exists():
        raise FileExistsError(f"v11 seed-111 face already exists: {FACE}")
    manifest_hits = [
        path.name
        for path in (ROOT / "plans/manifests").iterdir()
        if "v11" in path.name.lower() or "seed111" in path.name.lower()
    ]
    report_hits = [
        path.name
        for path in (ROOT / "reports").iterdir()
        if "v11_after_n120" in path.name.lower() or "seed111" in path.name.lower()
    ]
    if manifest_hits or report_hits:
        raise FileExistsError(
            f"v11 face/attempt artifacts already exist: {manifest_hits + report_hits}"
        )

    baseline_hashes = checkpoint_receipts(baseline, "baseline")
    candidate_hashes = checkpoint_receipts(candidate, "candidate")
    position = [model for model in revalidation["models"] if model.get("arm") == "position"]
    if (
        len(position) != 3
        or {model.get("seed") for model in position} != set(SEEDS)
        or any(model.get("training_sufficient") is not True for model in position)
    ):
        raise ValueError("all three N1-120 position gates must pass before confirmation planning")

    cpu_seconds = round(time.process_time() - started, 6)
    complete = cpu_seconds <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan121-v11-after-n120-prequery-readiness-v1",
        "status": "ready_for_owner_approval" if complete else "stopped",
        "complete": complete,
        "baseline_report_sha256": BASELINE_REPORT_SHA256,
        "candidate_report_sha256": CANDIDATE_REPORT_SHA256,
        "candidate_receipt_revalidation_sha256": CANDIDATE_REVALIDATION_SHA256,
        "n110_plan_sha256": N110_PLAN_SHA256,
        "n121_plan_sha256": N121_PLAN_SHA256,
        "baseline_checkpoint_hashes": baseline_hashes,
        "candidate_checkpoint_hashes": candidate_hashes,
        "checkpoint_files_hashed": len(baseline_hashes) + len(candidate_hashes),
        "baseline_endpoints": 9,
        "candidate_endpoints": 9,
        "position_training_gates_passed": 3,
        "confirm_face_seed": 111,
        "confirm_face_generated": False,
        "confirm_face_consumed": False,
        "range_queries": 0,
        "confirmation_queries": 0,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "can_promote": False,
        "planned_max_queries_if_approved": 540,
        "planned_range_gate": "at least 6/9 usable candidate endpoints",
        "planned_confirmation_gates": {
            "position_seed_exact_minimum": "10/20",
            "position_seed_kind_exact_minimum": "2/5",
            "advantage_over_terminal_minimum": "12/60",
            "advantage_over_cut_minimum": "12/60",
        },
        "cpu_seconds": cpu_seconds,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
    }
    with READINESS.open("x", encoding="utf-8", newline="\n") as stream:
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
                    "checkpoint_files_hashed",
                    "position_training_gates_passed",
                    "confirm_face_generated",
                    "range_queries",
                    "confirmation_queries",
                    "cpu_seconds",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
