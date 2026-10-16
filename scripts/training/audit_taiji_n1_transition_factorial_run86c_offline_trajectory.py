"""PLAN-N1-87: inspect sealed run86c losses without loading models or checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import _taiji_n1_transition_factorial_training_receipt as receipt

SOURCE = ROOT / "reports/taiji_n1_transition_factorial_run86c.json"
COMPLETION = ROOT / "reports/taiji_n1_transition_factorial_run86c_completion.json"
OUTPUT = ROOT / "reports/taiji_n1_transition_factorial_run86c_offline_trajectory_audit.json"
EXPECTED_REPORT_SHA256 = "0a8fc62e9aa00577d67001c70008773e6e5cee54cebad4115956e0360446d785"
CPU_LIMIT_SECONDS = 60
WINDOW = 512
GROUPS = ("reader", "writer", "local", "context", "relation")
VIEWS = ("original", "pair", "rename", "rename_pair")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_cpu(start: float) -> None:
    if time.process_time() - start > CPU_LIMIT_SECONDS:
        raise TimeoutError("PLAN-N1-87 exceeded its 60-second process CPU limit")


def validate_inputs(start: float) -> tuple[dict, list[dict], list[dict]]:
    report_bytes = SOURCE.read_bytes()
    report_sha = sha256_bytes(report_bytes)
    if report_sha != EXPECTED_REPORT_SHA256:
        raise ValueError("run86c report SHA differs from the approved PLAN-N1-87 input")
    report = json.loads(report_bytes)
    completion = json.loads(COMPLETION.read_text(encoding="utf-8"))
    if (
        completion.get("report_sha256") != report_sha
        or completion.get("exit_code") != 0
        or completion.get("engineering_complete") is not True
        or completion.get("training_sufficient") is not False
    ):
        raise ValueError("run86c completion does not match the adjudicated failed-gate batch")
    if (
        report.get("complete") is not True
        or report.get("training_sufficient") is not False
        or report.get("can_promote") is not False
        or report.get("code_unchanged") is not True
        or report.get("sources_unchanged") is not True
        or report.get("cpu_accounting_complete") is not True
        or len(report.get("models", [])) != 9
        or any(report.get(key) != 0 for key in ("dev_queries", "holdout_queries", "confirmation_queries"))
    ):
        raise ValueError("run86c report does not meet frozen offline-audit preconditions")

    expected = {
        (seed, arm)
        for seed in (11, 12, 13)
        for arm in ("position", "terminal", "cut")
    }
    models = report["models"]
    if {(model["seed"], model["arm"]) for model in models} != expected:
        raise ValueError("run86c model matrix differs from the frozen nine-model set")

    for raw_path, expected_sha in report["source_guards"].items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"run86c guarded source changed or disappeared: {raw_path}")
        guard_cpu(start)

    records = []
    for model in models:
        record = receipt.read_model(model)
        if (
            record.get("complete") is not True
            or record.get("status") != "complete"
            or record.get("completed_updates") != 24576
            or record.get("qa_sequences") != 55296
            or record.get("new_session_count") != 43008
            or record.get("score_queries") != 4800
            or record.get("signature_clone_updates") != 8
            or len(record.get("milestones", [])) != 5
            or len(record.get("rows", [])) != 4800
            or len(record.get("losses", [])) != 24576
            or len(record.get("measurements", [])) != 24576
        ):
            raise ValueError(f"incomplete fixed shard: s{model['seed']}_{model['arm']}")
        if len(record.get("restore_checks", [])) != 2 or any(
            check.get("equal") is not True for check in record["restore_checks"]
        ):
            raise ValueError(f"restore receipt mismatch: s{model['seed']}_{model['arm']}")
        records.append(record)
        guard_cpu(start)
    return report, models, records


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def final_view_rows(record: dict) -> dict[str, list[dict]]:
    return {
        view: [
            row
            for row in record["rows"]
            if row["completed_updates"] == 24576 and row["evaluation_set"] == view
        ]
        for view in VIEWS
    }


def window_summary(record: dict, items: list[dict], model: dict, start: int) -> dict:
    stop = start + WINDOW
    losses = record["losses"][start:stop]
    measurements = record["measurements"][start:stop]
    if len(losses) != WINDOW or len(measurements) != WINDOW:
        raise ValueError("incomplete 512-update analysis window")

    kind_losses: dict[str, list[float]] = defaultdict(list)
    phase_losses: dict[str, list[float]] = defaultdict(list)
    for offset, loss in enumerate(losses):
        step = start + offset
        local_step = step % 6144
        item_index = (local_step // 2) % 192
        item = items[item_index]
        kind_losses[item["kind"]].append(loss["qa"])
        phase = "original_names" if local_step % 2 == 0 else "renamed_names"
        phase_losses[phase].append(loss["qa"])

    group_stats = {}
    for group in GROUPS:
        values = [measurement["groups"][group] for measurement in measurements]
        ratios = [entry["actual_l2"] / entry["cap"] for entry in values]
        group_stats[group] = {
            "mean_actual_l2": round(statistics.fmean(entry["actual_l2"] for entry in values), 10),
            "mean_actual_to_cap": round(statistics.fmean(ratios), 8),
            "clipped_updates": sum(entry["clipped"] is True for entry in values),
        }

    return {
        "seed": model["seed"],
        "arm": model["arm"],
        "start_update": start,
        "end_update_exclusive": stop,
        "mean_qa_loss": round(statistics.fmean(loss["qa"] for loss in losses), 10),
        "mean_observation_loss": round(
            statistics.fmean(loss["observation"] for loss in losses), 10
        ),
        "mean_total_loss": round(statistics.fmean(loss["total"] for loss in losses), 10),
        "mean_qa_loss_by_kind": {
            kind: round(statistics.fmean(values), 10)
            for kind, values in sorted(kind_losses.items())
        },
        "mean_qa_loss_by_name_phase": {
            phase: round(statistics.fmean(values), 10)
            for phase, values in sorted(phase_losses.items())
        },
        "update_groups": group_stats,
    }


def position_item_loss_link(record: dict, items: list[dict], model: dict) -> dict:
    rows = final_view_rows(record)
    if model["seed"] != 12 or model["arm"] != "position":
        raise ValueError("item-level loss link is preregistered for seed12/position only")
    item_by_id = {item["id"]: item for item in items}
    output = {}
    start = 3 * 6144
    for view, phase in (("original", 0), ("rename", 1)):
        final_rows = {row["id"]: row for row in rows[view]}
        aggregates: dict[str, list[float]] = defaultdict(list)
        for offset, loss in enumerate(record["losses"][start : start + 6144]):
            local_step = (start + offset) % 6144
            if local_step % 2 != phase:
                continue
            item_index = (local_step // 2) % 192
            item_id = items[item_index]["id"]
            aggregates[item_id].append(loss["qa"])

        by_outcome: dict[str, list[float]] = defaultdict(list)
        per_item = []
        for item_id, values in aggregates.items():
            row = final_rows[item_id]
            outcome = "final_error" if row["free_exact"] is False else "final_correct"
            item = item_by_id[item_id]
            item_mean = statistics.fmean(values)
            by_outcome[outcome].append(item_mean)
            per_item.append(
                {
                    "id": item_id,
                    "kind": item["kind"],
                    "mean_cycle4_qa_loss": round(item_mean, 10),
                    "final_free_exact": row["free_exact"],
                }
            )
        output[view] = {
            "item_count": len(per_item),
            "loss_mean_by_final_outcome": {
                key: round(statistics.fmean(values), 10)
                for key, values in sorted(by_outcome.items())
            },
            "highest_cycle4_item_losses": sorted(
                per_item, key=lambda item: item["mean_cycle4_qa_loss"], reverse=True
            )[:16],
            "final_error_ids": sorted(
                item["id"] for item in per_item if item["final_free_exact"] is False
            ),
        }
    return output


def model_result(model: dict, record: dict) -> dict:
    final = record["milestones"][-1]
    rows = final_view_rows(record)
    errors = {}
    for view, selected in rows.items():
        failed = [row for row in selected if row["free_exact"] is False]
        errors[view] = {
            "items": len(selected),
            "exact": len(selected) - len(failed),
            "errors_by_kind": {
                kind: sum(row["kind"] == kind for row in failed)
                for kind in ("lookup", "revision", "transition", "unknown")
            },
            "error_ids": sorted(row["id"] for row in failed),
        }
    return {
        "seed": model["seed"],
        "arm": model["arm"],
        "training_sufficient": model["training_sufficient"],
        "final": {
            "original_exact": final["free_exact"],
            "original_by_kind": final["free_by_kind"],
            "pair_exact": final["pair_exact"],
            "rename_exact": final["rename_free_exact"],
            "rename_by_kind": final["rename_free_by_kind"],
            "rename_pair_exact": final["rename_pair_exact"],
            "factorial_original": final["factorial"]["original"],
            "factorial_rename": final["factorial"]["rename"],
        },
        "view_errors": errors,
    }


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def execute() -> dict:
    started_cpu = time.process_time()
    if OUTPUT.exists():
        raise FileExistsError("PLAN-N1-87 is a single attempt; output already exists")
    report, models, records = validate_inputs(started_cpu)
    items = receipt.train.runner.core.checked_train()
    if len(items) != 192 or len({item["id"] for item in items}) != 192:
        raise ValueError("frozen train schedule requires 192 unique items")

    windows = []
    summaries = []
    item_link = None
    for model, record in zip(models, records, strict=True):
        summaries.append(model_result(model, record))
        for cycle in (2, 3):
            cycle_start = cycle * 6144
            for start in range(cycle_start, cycle_start + 6144, WINDOW):
                windows.append(window_summary(record, items, model, start))
                guard_cpu(started_cpu)
        if model["seed"] == 12 and model["arm"] == "position":
            item_link = position_item_loss_link(record, items, model)
        guard_cpu(started_cpu)

    if len(windows) != 9 * 2 * 12 or item_link is None:
        raise ValueError("all nine fixed trajectories and seed12 item audit are required")
    cpu_seconds = time.process_time() - started_cpu
    if cpu_seconds > CPU_LIMIT_SECONDS:
        raise TimeoutError("PLAN-N1-87 exceeded its 60-second process CPU limit")

    result = {
        "schema": "n1-run86c-offline-trajectory-audit-v1",
        "plan": "PLAN-N1-87",
        "complete": True,
        "input_report_sha256": EXPECTED_REPORT_SHA256,
        "completion_report_sha256": json.loads(COMPLETION.read_text(encoding="utf-8"))["report_sha256"],
        "model_count": len(models),
        "sealed_model_count": len(records),
        "analysis_windows": len(windows),
        "window_updates": WINDOW,
        "analysed_cycles": [3, 4],
        "models": summaries,
        "windows": windows,
        "seed12_position_cycle4_item_loss_link": item_link,
        "execution": {
            "cpu_seconds": round(cpu_seconds, 6),
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "model_loads": 0,
            "checkpoint_loads": 0,
            "model_queries": 0,
            "evaluations": 0,
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
            "This is an association audit of already sealed losses, update measurements, and final scores; "
            "it does not establish the causal source of endpoint regression."
        ),
        "report_source_guards": report["source_guards"],
    }
    if any(
        not math.isfinite(window["mean_qa_loss"])
        or not math.isfinite(window["mean_observation_loss"])
        for window in windows
    ):
        raise ValueError("non-finite loss in audited windows")
    write_new(OUTPUT, result)
    return result


if __name__ == "__main__":
    completed = execute()
    print(
        json.dumps(
            {
                "complete": completed["complete"],
                "report": str(OUTPUT),
                "input_report_sha256": completed["input_report_sha256"],
                "models": completed["model_count"],
                "analysis_windows": completed["analysis_windows"],
                "cpu_seconds": completed["execution"]["cpu_seconds"],
                "seed12_position_cycle4": completed[
                    "seed12_position_cycle4_item_loss_link"
                ]["original"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
