"""PLAN-N1-89 batch verification: recompute a control-arm batch from sealed artifacts.

Zero training, zero model loads, zero unseen queries. The report's own booleans are
never trusted: the six gates, the run86c same-arm prefix equivalence (score rows at
steps <= 18432 included), the restore ledger, the source guards, the code identity and
the CPU arithmetic are all recomputed from the sealed shards and files on disk.
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
from scripts.training import run_taiji_n1_transition_factorial_lr_decay_run89 as controller

CPU_LIMIT_SECONDS = 300
KINDS = ("lookup", "revision", "transition", "unknown")
GATES = (
    ("original_min", 173),
    ("original_kind_min", 40),
    ("pair_min", 87),
    ("rename_min", 173),
    ("rename_kind_min", 40),
    ("rename_pair_min", 87),
    ("joint_min", 44),
)


class Guard(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Guard(message)


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


def prefix_readings(record: dict, baseline: dict) -> dict:
    decay = controller.DECAY_START
    steps = controller.PREFIX_ROW_STEPS
    per_milestone = {}
    for step in steps:
        current = [row for row in record["rows"] if row["completed_updates"] == step]
        base = [row for row in baseline["rows"] if row["completed_updates"] == step]
        per_milestone[str(step)] = {
            "rows": len(current),
            "baseline_rows": len(base),
            "equal": current == base,
        }
    strict_below = tuple(step for step in steps if step < decay)
    return {
        "losses_prefix_equal": record["losses"][:decay] == baseline["losses"][:decay],
        "measurements_prefix_equal": (
            record["measurements"][:decay] == baseline["measurements"][:decay]
        ),
        "score_rows_by_milestone": per_milestone,
        "strictly_below_decay_rows_equal": [
            row for row in record["rows"] if row["completed_updates"] in strict_below
        ]
        == [row for row in baseline["rows"] if row["completed_updates"] in strict_below],
        "decay_milestone_rows_equal": all(
            entry["equal"] for entry in per_milestone.values() if entry["rows"]
        )
        and per_milestone[str(decay)]["equal"],
        "terminal_rows_equal": [
            row for row in record["rows"] if row["completed_updates"] == controller.UPDATES
        ]
        == [row for row in baseline["rows"] if row["completed_updates"] == controller.UPDATES],
    }


def execute(arm: str, out_path: Path | None = None) -> dict:
    controller.configure(arm)
    started = time.process_time()
    target = out_path or (
        ROOT
        / f"reports/taiji_n1_transition_factorial_lr_decay_{controller.RUN_STEM}_batch_verification.json"
    )
    require(not target.exists(), "batch verification output already exists; single attempt file")
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
    require(
        report.get("code_identity") == controller.code_identity(report["protocol"]),
        "code identity does not recompute from the files on disk",
    )
    require(
        report.get("status") == "complete"
        and report.get("complete") is True
        and report.get("can_promote") is False
        and report.get("code_unchanged") is True
        and report.get("sources_unchanged") is True
        and report.get("cpu_accounting_complete") is True
        and "train_sha" not in report
        and all(
            report.get(key) == 0
            for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        ),
        "batch-level discipline fields failed",
    )
    require(len(report["models"]) == len(controller.SEEDS), "frozen three-seed set required")
    require(
        all(model["arm"] == arm for model in report["models"]),
        "every model in this batch must carry the configured arm",
    )

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

    changed = {}
    for raw_path, digest in report["source_guards"].items():
        path = Path(raw_path)
        require(path.is_file(), f"guarded input disappeared: {raw_path}")
        observed = controller.guard_sha256(path)
        if observed != digest:
            changed[raw_path] = {"recorded": digest, "observed": observed}
        if time.process_time() - started > CPU_LIMIT_SECONDS:
            raise Guard("batch verification exceeded its CPU limit")
    require(not changed, "guarded inputs changed since the attempt was written")

    preflight_zero = {}
    for entry in preflight["zero_checkpoints"]:
        path = controller.CHECKPOINT_ROOT / entry["path_name"]
        require(
            path.is_file()
            and controller.sha256(path) == entry["sha256"]
            and entry["restore_equal"] is True
            and entry["matches_run86c_arm_zero"] is True
            and entry["arm"] == arm,
            f"preflight zero checkpoint failed its seal: {path}",
        )
        preflight_zero[entry["path_name"]] = entry["sha256"]
    require(len(preflight_zero) == len(controller.SEEDS), "one fresh zero checkpoint per seed")

    base_position = {model["seed"]: model for model in base_report["models"] if model["arm"] == arm}
    require(set(base_position) == set(controller.SEEDS), "run86c same-arm baselines incomplete")

    _, _, views = receipt.train.known_views()
    models = []
    base_protocol = receipt.train.PROTOCOL
    try:
        for summary in report["models"]:
            seed = summary["seed"]
            receipt.train.PROTOCOL = base_protocol
            baseline = receipt.read_model(base_position[seed])
            receipt.train.PROTOCOL = report["protocol"]
            record = receipt.read_model(summary)
            receipt.validate_model(summary, record, views=views)

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
                and {Path(check["path"]).name for check in restores}
                == {f"s{seed}_{arm}_zero.pt", f"s{seed}_{arm}_final.pt"}
                and controller.sha256(Path(restores[0]["path"]))
                == preflight_zero[f"s{seed}_{arm}_zero.pt"],
                f"independent restore ledger failed for seed {seed}",
            )
            require(
                summary["learning_rate_schedule"]
                == {
                    "base_lr": controller.BASE_LR,
                    "final_cycle_lr": controller.FINAL_LR,
                    "base_updates": controller.DECAY_START,
                    "final_cycle_updates": controller.UPDATES - controller.DECAY_START,
                },
                f"learning-rate schedule evidence differs for seed {seed}",
            )
            prefix = prefix_readings(record, baseline)
            require(
                prefix["losses_prefix_equal"]
                and prefix["measurements_prefix_equal"]
                and prefix["decay_milestone_rows_equal"],
                f"prefix equivalence to run86c failed for seed {seed}",
            )
            models.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "completed_updates": record["completed_updates"],
                    "losses": len(record["losses"]),
                    "measurements": len(record["measurements"]),
                    "rows": len(record["rows"]),
                    "qa_sequences": sum(entry["qa_sequences"] for entry in record["losses"]),
                    "score_queries": record["score_queries"],
                    "new_session_count": record["new_session_count"],
                    "signature_clone_updates": record["signature_clone_updates"],
                    "terminal_gates": {
                        "observed": recomputed,
                        "thresholds": dict(GATES),
                        "passed": {name: recomputed[name] >= line for name, line in GATES},
                    },
                    "training_sufficient": sufficient,
                    "run86c_terminal_gates": recompute_gates(base_position[seed]["milestones"][-1]),
                    "independent_restores": [
                        {"path": Path(check["path"]).name, "equal": check["equal"]}
                        for check in restores
                    ],
                    "prefix_vs_run86c": prefix,
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
                }
            )
            if time.process_time() - started > CPU_LIMIT_SECONDS:
                raise Guard("batch verification exceeded its CPU limit")
    finally:
        receipt.train.PROTOCOL = base_protocol

    totals = {
        "completed_updates": sum(entry["completed_updates"] for entry in models),
        "losses": sum(entry["losses"] for entry in models),
        "qa_sequences": sum(entry["qa_sequences"] for entry in models),
        "new_session_count": sum(entry["new_session_count"] for entry in models),
        "score_rows": sum(entry["rows"] for entry in models),
        "signature_clone_updates": sum(entry["signature_clone_updates"] for entry in models),
        "formal_independent_restores": sum(len(entry["independent_restores"]) for entry in models),
        "preflight_independent_restores": len(preflight_zero),
        "guarded_inputs": len(report["source_guards"]),
        "seeds_passing_six_gates": sum(bool(entry["training_sufficient"]) for entry in models),
    }
    require(
        totals["completed_updates"] == report["completed_updates"] == len(controller.SEEDS) * 24576
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
        "schema": f"n1-{controller.RUN_STEM}-batch-verification-v1",
        "plan": "PLAN-N1-89",
        "arm": arm,
        "complete": True,
        "verdict": (
            "control_arm_batch_engineering_complete"
            if report["complete"] is True
            else "batch_not_complete"
        ),
        "gate_note": (
            "Six-gate values are published for the control arm but not used to grade it; "
            "PLAN-N1-85 requires the controls to be preserved and engineering complete."
        ),
        "report_sha256": report_sha,
        "task_name": completion["task_name"],
        "process_id": completion.get("process_id"),
        "cpu_accounting": cpu,
        "guarded_inputs_changed": changed,
        "preflight_zero_checkpoints": preflight_zero,
        "learning_rate_contract": preflight["learning_rate_contract"],
        "protocol_schema": report["protocol"]["schema"],
        "bound_position_batch": {
            "report_sha256": controller.sha256(controller.POSITION_REPORT),
            "verification_sha256": controller.sha256(controller.POSITION_VERIFICATION),
            "training_sufficient": json.loads(
                controller.POSITION_REPORT.read_text(encoding="utf-8")
            )["training_sufficient"],
        },
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
            "Engineering completeness of a control-arm batch is not a capability reading. "
            "No v5, v3 holdout, v4 or dev query was issued here, and the arm comparison only "
            "becomes interpretable once all three families are bound by PLAN-N1-89 section 4."
        ),
    }
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "complete": True,
        "verdict": result["verdict"],
        "report": str(target),
        "cpu_seconds": cpu_seconds,
        "seeds_passing_six_gates": totals["seeds_passing_six_gates"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(controller.FAMILIES))
    parser.add_argument("--out-report", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(execute(arguments.arm, arguments.out_path), ensure_ascii=False, sort_keys=True)
    )
