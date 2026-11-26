"""PLAN-N1-88 batch close-out: re-verify sealed run88c evidence without loading models.

The producing controller attests its own prefix equivalence at seal time; this
instrument recomputes the same claims from the sealed artifacts instead, and
extends the score-row prefix requirement to the 18432 milestone, which the
controller only compared strictly below 18432.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import _taiji_n1_transition_factorial_training_receipt as receipt
from scripts.training import run_taiji_n1_transition_factorial_lr_decay_run88 as controller

OUTPUT = ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88c_batch_verification.json"
CPU_LIMIT_SECONDS = 300
KINDS = ("lookup", "revision", "transition", "unknown")
VIEWS = ("original", "pair", "rename", "rename_pair")
GATES = (
    ("original_min", 173),
    ("original_kind_min", 40),
    ("pair_min", 87),
    ("rename_min", 173),
    ("rename_kind_min", 40),
    ("rename_pair_min", 87),
    ("joint_min", 44),
)
IDENTITY_FILES = {
    "controller_sha256": controller.__file__,
    "launcher_sha256": controller.LAUNCHER,
    "frozen_runner_sha256": controller.FROZEN_RUNNER,
    "plan_sha256": controller.PLAN,
    "offline_audit_sha256": controller.AUDIT_REPORT,
    "control_report_sha256": controller.BASE_REPORT,
}


class Guard(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Guard(message)


def guard_cpu(started: float) -> None:
    used = time.process_time() - started
    if used > CPU_LIMIT_SECONDS:
        raise Guard(f"batch verification exceeded its {CPU_LIMIT_SECONDS}-second CPU limit")


def recompute_gates(summary: dict) -> dict:
    return {
        "original_min": summary["free_exact"],
        "original_kind_min": min(summary["free_by_kind"][kind] for kind in KINDS),
        "pair_min": summary["pair_exact"],
        "rename_min": summary["rename_free_exact"],
        "rename_kind_min": min(summary["rename_free_by_kind"][kind] for kind in KINDS),
        "rename_pair_min": summary["rename_pair_exact"],
        "joint_min": min(
            summary["factorial"][view]["joint_four_exact"] for view in ("original", "rename")
        ),
    }


def rows_by_milestone(record: dict, steps: tuple[int, ...]) -> dict:
    return {
        step: [row for row in record["rows"] if row["completed_updates"] == step] for step in steps
    }


def prefix_readings(run_record: dict, base_record: dict) -> dict:
    decay = controller.DECAY_START
    prefix_steps = tuple(step for step in controller.MILESTONES if step <= decay)
    current_rows = rows_by_milestone(run_record, prefix_steps)
    baseline_rows = rows_by_milestone(base_record, prefix_steps)
    per_milestone = {}
    for step in sorted(current_rows):
        current = current_rows[step]
        baseline = baseline_rows.get(step, [])
        equal = current == baseline
        detail = None
        if not equal and len(current) == len(baseline):
            for index, (a, b) in enumerate(zip(current, baseline, strict=True)):
                if a != b:
                    detail = {
                        "first_differing_index": index,
                        "differing_keys": sorted(k for k in a if a.get(k) != b.get(k)),
                        "run88c_id": a.get("id"),
                    }
                    break
        per_milestone[str(step)] = {
            "rows": len(current),
            "baseline_rows": len(baseline),
            "equal": equal,
            "mismatch_detail": detail,
        }
    terminal_current = [
        r for r in run_record["rows"] if r["completed_updates"] == controller.UPDATES
    ]
    terminal_base = [r for r in base_record["rows"] if r["completed_updates"] == controller.UPDATES]
    return {
        "losses_prefix_equal": run_record["losses"][:decay] == base_record["losses"][:decay],
        "losses_prefix_updates": decay,
        "measurements_prefix_equal": (
            run_record["measurements"][:decay] == base_record["measurements"][:decay]
        ),
        "score_rows_by_milestone": per_milestone,
        "controller_checked_rows_strictly_below_decay": (
            [r for r in run_record["rows"] if r["completed_updates"] < decay]
            == [r for r in base_record["rows"] if r["completed_updates"] < decay]
        ),
        "terminal_rows_equal": terminal_current == terminal_base,
        "terminal_rows": len(terminal_current),
    }


def verify() -> dict:
    started = time.process_time()
    require(not OUTPUT.exists(), "batch verification output already exists; single attempt file")
    report = json.loads(controller.REPORT.read_text(encoding="utf-8"))
    completion = json.loads(controller.COMPLETION.read_text(encoding="utf-8"))
    preflight = json.loads(controller.PREFLIGHT.read_text(encoding="utf-8"))
    base_report = json.loads(controller.BASE_REPORT.read_text(encoding="utf-8"))

    report_sha = controller.sha256(controller.REPORT)
    require(
        completion.get("report_sha256") == report_sha
        and completion.get("exit_code") == 0
        and completion.get("error") is None
        and completion.get("task_name") == controller.TASK_RUN,
        "completion receipt does not bind this report",
    )
    for key, path in IDENTITY_FILES.items():
        require(
            report["code_identity"].get(key) == controller.sha256(Path(path)),
            f"code identity {key} differs from the file on disk",
        )

    batch = {
        "status": report.get("status"),
        "complete": report.get("complete"),
        "training_sufficient": report.get("training_sufficient"),
        "can_promote": report.get("can_promote"),
        "code_unchanged": report.get("code_unchanged"),
        "sources_unchanged": report.get("sources_unchanged"),
        "cpu_accounting_complete": report.get("cpu_accounting_complete"),
        "train_sha_key_present": "train_sha" in report or "TRAIN_SHA" in report,
        "unseen_query_counts": {
            key: report.get(key)
            for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        },
    }
    require(
        batch["status"] == "complete"
        and batch["complete"] is True
        and batch["can_promote"] is False
        and batch["code_unchanged"] is True
        and batch["sources_unchanged"] is True
        and batch["cpu_accounting_complete"] is True
        and batch["train_sha_key_present"] is False
        and all(value == 0 for value in batch["unseen_query_counts"].values()),
        "batch-level discipline fields failed",
    )
    require(len(report["models"]) == len(controller.SEEDS), "frozen three-seed set required")

    cpu = {
        "prior_cpu_reserve_seconds": report["prior_cpu_reserve_seconds"],
        "preflight_cpu_seconds": report["preflight_cpu_seconds"],
        "setup_cpu_seconds": report["setup_cpu_seconds"],
        "training_cpu_seconds": report["training_cpu_seconds"],
        "sum_recomputed": round(
            report["prior_cpu_reserve_seconds"]
            + report["preflight_cpu_seconds"]
            + report["setup_cpu_seconds"]
            + report["training_cpu_seconds"],
            6,
        ),
        "reported_cpu_seconds": report["cpu_seconds"],
        "cpu_limit_seconds": report["protocol"]["cpu_limit_seconds"],
        "os_process_cpu_seconds": completion.get("process_cpu_seconds"),
    }
    require(
        cpu["sum_recomputed"] == cpu["reported_cpu_seconds"]
        and cpu["reported_cpu_seconds"] <= cpu["cpu_limit_seconds"],
        "CPU accounting does not recompute inside the frozen limit",
    )

    guards = {}
    for raw_path, digest in report["source_guards"].items():
        path = Path(raw_path)
        require(path.is_file(), f"guarded input disappeared: {raw_path}")
        observed = controller.guard_sha256(path)
        if observed != digest:
            guards[raw_path] = {"recorded": digest, "observed": observed}
        guard_cpu(started)
    require(not guards, "guarded inputs changed since the attempt was written")

    preflight_zero = {}
    for entry in preflight["zero_checkpoints"]:
        path = controller.CHECKPOINT_ROOT / entry["path_name"]
        require(path.is_file(), f"preflight zero checkpoint missing: {path}")
        require(
            controller.sha256(path) == entry["sha256"]
            and entry["restore_equal"] is True
            and entry["matches_run86c_position_zero"] is True,
            f"preflight zero checkpoint failed its seal: {path}",
        )
        preflight_zero[entry["path_name"]] = entry["sha256"]
    require(len(preflight_zero) == 3, "three fresh zero checkpoints required")

    _, _, views = receipt.train.known_views()
    base_position = {
        model["seed"]: model for model in base_report["models"] if model["arm"] == controller.ARM
    }
    require(
        set(base_position) == set(controller.SEEDS),
        "run86c position baselines missing for a frozen seed",
    )

    models = []
    base_protocol = receipt.train.PROTOCOL
    run_protocol = report["protocol"]
    try:
        for summary in report["models"]:
            seed = summary["seed"]
            receipt.train.PROTOCOL = base_protocol
            base_record = receipt.read_model(base_position[seed])
            receipt.train.PROTOCOL = run_protocol
            record = receipt.read_model(summary)
            receipt.validate_model(summary, record, views=views)
            guard_cpu(started)

            final = record["milestones"][-1]
            recomputed = recompute_gates(final)
            sufficient = receipt.train.calibration.sufficient(final)
            require(
                sufficient == summary["training_sufficient"] == record["training_sufficient"],
                f"stored six-gate verdict differs from recomputation for seed {seed}",
            )
            restores = record["restore_checks"]
            require(
                len(restores) == 2
                and all(check["equal"] is True for check in restores)
                and {Path(c["path"]).name for c in restores}
                == {f"s{seed}_position_zero.pt", f"s{seed}_position_final.pt"}
                and controller.sha256(Path(restores[0]["path"]))
                == preflight_zero[f"s{seed}_position_zero.pt"],
                f"independent restore ledger failed for seed {seed}",
            )
            schedule = summary["learning_rate_schedule"]
            require(
                schedule
                == {
                    "base_lr": controller.BASE_LR,
                    "final_cycle_lr": controller.FINAL_LR,
                    "base_updates": controller.DECAY_START,
                    "final_cycle_updates": controller.UPDATES - controller.DECAY_START,
                },
                f"learning-rate schedule evidence differs for seed {seed}",
            )
            models.append(
                {
                    "seed": seed,
                    "arm": summary["arm"],
                    "completed_updates": record["completed_updates"],
                    "losses": len(record["losses"]),
                    "measurements": len(record["measurements"]),
                    "rows": len(record["rows"]),
                    "qa_sequences": sum(entry["qa_sequences"] for entry in record["losses"]),
                    "score_queries": record["score_queries"],
                    "new_session_count": record["new_session_count"],
                    "signature_clone_updates": record["signature_clone_updates"],
                    "milestones_by_step": {
                        str(entry["completed_updates"]): {
                            "original": entry["free_exact"],
                            "pair": entry["pair_exact"],
                            "rename": entry["rename_free_exact"],
                            "rename_pair": entry["rename_pair_exact"],
                            "joint_original": entry["factorial"]["original"]["joint_four_exact"],
                            "joint_rename": entry["factorial"]["rename"]["joint_four_exact"],
                        }
                        for entry in record["milestones"]
                    },
                    "terminal_gates": {
                        "observed": recomputed,
                        "thresholds": dict(GATES),
                        "passed": {k: recomputed[k] >= t for k, t in GATES},
                    },
                    "training_sufficient": sufficient,
                    "independent_restores": [
                        {"path": Path(c["path"]).name, "equal": c["equal"]} for c in restores
                    ],
                    "prefix_vs_run86c": prefix_readings(record, base_record),
                    "run86c_terminal_gates": recompute_gates(base_position[seed]["milestones"][-1]),
                    "run86c_training_sufficient": base_position[seed]["training_sufficient"],
                }
            )
            guard_cpu(started)
    finally:
        receipt.train.PROTOCOL = base_protocol

    for model in models:
        prefix = model["prefix_vs_run86c"]
        require(
            prefix["losses_prefix_equal"]
            and prefix["measurements_prefix_equal"]
            and all(entry["equal"] is True for entry in prefix["score_rows_by_milestone"].values()),
            f"prefix equivalence to run86c failed for seed {model['seed']}",
        )

    totals = {
        "completed_updates": sum(m["completed_updates"] for m in models),
        "losses": sum(m["losses"] for m in models),
        "qa_sequences": sum(m["qa_sequences"] for m in models),
        "new_session_count": sum(m["new_session_count"] for m in models),
        "score_rows": sum(m["rows"] for m in models),
        "signature_clone_updates": sum(m["signature_clone_updates"] for m in models),
        "formal_independent_restores": sum(len(m["independent_restores"]) for m in models),
        "preflight_independent_restores": len(preflight_zero),
        "guarded_inputs": len(report["source_guards"]),
        "seeds_passing_six_gates": sum(bool(m["training_sufficient"]) for m in models),
    }
    require(
        totals["completed_updates"] == report["completed_updates"]
        and totals["losses"] == report["completed_updates"]
        and totals["qa_sequences"] == report["qa_sequences"]
        and totals["new_session_count"] == report["new_session_count"]
        and totals["score_rows"] == report["score_queries"]
        and totals["signature_clone_updates"] == report["signature_clone_updates"],
        "shard-level totals differ from the batch receipt",
    )

    cpu_seconds = round(time.process_time() - started, 6)
    require(cpu_seconds <= CPU_LIMIT_SECONDS, "verification CPU limit exceeded")
    result = {
        "schema": "n1-run88c-batch-verification-v1",
        "plan": "PLAN-N1-88",
        "complete": True,
        "verdict": (
            "batch_sufficient_three_position_six_gates"
            if all(model["training_sufficient"] for model in models)
            else "batch_not_sufficient"
        ),
        "report_sha256": report_sha,
        "task_name": completion["task_name"],
        "process_id": completion.get("process_id"),
        "batch": batch,
        "cpu_accounting": cpu,
        "guarded_inputs_changed": guards,
        "preflight_zero_checkpoints": preflight_zero,
        "learning_rate_contract": preflight["learning_rate_contract"],
        "protocol_schema": run_protocol["schema"],
        "models": models,
        "totals": totals,
        "execution": {
            "cpu_seconds": cpu_seconds,
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "model_loads": 0,
            "checkpoint_loads": 0,
            "model_queries": 0,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "signature_clone_updates": 0,
            "dev_queries": 0,
            "v3_holdout_queries": 0,
            "v4_queries": 0,
            "v5_queries": 0,
            "can_promote": False,
        },
        "interpretation_limit": (
            "Six-gate sufficiency on the frozen train-only views is a training-adequacy reading. "
            "It is not admission, not an unseen-set capability claim, and does not bind PLAN-N1-85 "
            "or promote any checkpoint; no holdout, v3, v4, v5 or dev query was issued here."
        ),
    }
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


if __name__ == "__main__":
    outcome = verify()
    print(
        json.dumps(
            {
                "complete": outcome["complete"],
                "verdict": outcome["verdict"],
                "report": str(OUTPUT),
                "seeds_passing_six_gates": outcome["totals"]["seeds_passing_six_gates"],
                "terminal": {
                    str(model["seed"]): model["terminal_gates"]["observed"]
                    for model in outcome["models"]
                },
                "cpu_seconds": outcome["execution"]["cpu_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
