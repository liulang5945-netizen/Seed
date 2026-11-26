"""Read-only attribution of the failed PLAN-N1-110 position training gate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_REPORT = ROOT / "reports/taiji_n1_length_balanced_training_20261018.json"
PLAN = ROOT / "plans/reference/PLAN-N1-110_execution_closeout_20261018.md"
OUT_REPORT = ROOT / "reports/taiji_n1_length_balance_failure_attribution_20261018.json"
SEEDS = (11, 12, 13)
CONDITIONS = ("original", "turn_flip", "direction_opposite", "both")
DIRECTIONS = ("north", "east", "south", "west")
TURNS = ("left", "right")
CPU_LIMIT_SECONDS = 30


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _decode_hex(value: str) -> str:
    return bytes.fromhex(value).decode("utf-8", errors="replace")


def summarize_factorial_view(rows: list[dict]) -> dict:
    """Summarize one already-scored 48-parent × four-condition training view."""

    if len(rows) != 192:
        raise ValueError(f"expected 192 factorial score rows, got {len(rows)}")
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        parent = row.get("parent_id")
        condition = row.get("condition")
        if (
            not isinstance(parent, str)
            or condition not in CONDITIONS
            or row.get("kind") != "transition"
            or type(row.get("free_exact")) is not bool
            or type(row.get("teacher_exact")) is not bool
            or condition in grouped[parent]
        ):
            raise ValueError("invalid, duplicate, or non-transition factorial row")
        grouped[parent][condition] = row
    if len(grouped) != 48 or any(set(group) != set(CONDITIONS) for group in grouped.values()):
        raise ValueError("all 48 parents must have exactly one row per factorial condition")

    cells: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {condition: [] for condition in CONDITIONS}
    )
    baseline_cell_counts: Counter[str] = Counter()
    parent_correct: dict[str, dict[str, int]] = {"free": {}, "teacher": {}}
    for parent, conditions in grouped.items():
        baseline = conditions["original"]
        direction = baseline.get("initial_direction")
        turn = baseline.get("turn")
        if direction not in DIRECTIONS or turn not in TURNS:
            raise ValueError("original transition row lacks a valid baseline direction/turn")
        cell = f"{direction}/{turn}"
        baseline_cell_counts[cell] += 1
        for condition, row in conditions.items():
            cells[cell][condition].append(row)
        parent_correct["free"][parent] = sum(int(row["free_exact"]) for row in conditions.values())
        parent_correct["teacher"][parent] = sum(
            int(row["teacher_exact"]) for row in conditions.values()
        )

    if sum(baseline_cell_counts.values()) != 48 or any(
        len(cells[cell][condition]) != expected
        for cell, expected in baseline_cell_counts.items()
        for condition in CONDITIONS
    ):
        raise ValueError("baseline direction × turn cell denominators changed across conditions")

    condition_counts = {}
    wrong_outputs = {}
    for condition in CONDITIONS:
        subset = [conditions[condition] for conditions in grouped.values()]
        incorrect = [row for row in subset if not row["free_exact"]]
        wrong_outputs[condition] = dict(
            Counter(_decode_hex(row["output_hex"]) for row in incorrect)
        )
        condition_counts[condition] = {
            "items": len(subset),
            "free_exact": sum(int(row["free_exact"]) for row in subset),
            "teacher_exact": sum(int(row["teacher_exact"]) for row in subset),
            "free_teacher_disagreements": sum(
                row["free_exact"] != row["teacher_exact"] for row in subset
            ),
        }

    distributions = {}
    joint_parent_ids = {}
    for mode in ("free", "teacher"):
        counts = Counter(parent_correct[mode].values())
        distributions[mode] = {str(number): counts[number] for number in range(5)}
        joint_parent_ids[mode] = sorted(
            parent for parent, count in parent_correct[mode].items() if count == 4
        )

    cell_summary = {
        cell_name: {
            "parents": baseline_cell_counts[cell_name],
            **{
                condition: {
                    "items": len(cell[condition]),
                    "free_exact": sum(int(row["free_exact"]) for row in cell[condition]),
                    "teacher_exact": sum(int(row["teacher_exact"]) for row in cell[condition]),
                }
                for condition in CONDITIONS
            },
        }
        for cell_name, cell in sorted(cells.items())
    }

    return {
        "items": len(rows),
        "parents": len(grouped),
        "conditions": list(CONDITIONS),
        "condition_counts": condition_counts,
        "joint_four_exact": {
            "free": len(joint_parent_ids["free"]),
            "teacher": len(joint_parent_ids["teacher"]),
            "free_parent_ids": joint_parent_ids["free"],
            "teacher_parent_ids": joint_parent_ids["teacher"],
            "free_correct_count_distribution": distributions["free"],
            "teacher_correct_count_distribution": distributions["teacher"],
        },
        "baseline_direction_turn_cells": cell_summary,
        "wrong_free_output_counts_by_condition": wrong_outputs,
    }


def _load_shard(model: dict) -> tuple[dict, dict]:
    seal = model["shard"]
    path = Path(seal["path"])
    compressed = path.read_bytes()
    if sha256_bytes(compressed) != seal["compressed_sha256"]:
        raise ValueError(f"training shard compressed digest mismatch: {path}")
    payload_bytes = gzip.decompress(compressed)
    if sha256_bytes(payload_bytes) != seal["uncompressed_sha256"]:
        raise ValueError(f"training shard uncompressed digest mismatch: {path}")
    payload = json.loads(payload_bytes)
    record = payload.get("record")
    if (
        payload.get("schema") != "n1-lossless-training-evidence-shard-v1"
        or not isinstance(record, dict)
        or record.get("seed") != model["seed"]
        or record.get("arm") != "position"
        or record.get("complete") is not True
        or record.get("completed_updates") != 24576
    ):
        raise ValueError(f"complete fixed position training shard required: {path}")
    return {"path": str(path), "compressed_sha256": seal["compressed_sha256"]}, record


def analyze() -> dict:
    started_cpu = time.process_time()
    if OUT_REPORT.exists():
        raise FileExistsError("PLAN-N1-110 failure attribution is write-once")
    report_bytes = TRAIN_REPORT.read_bytes()
    training_report = json.loads(report_bytes)
    source_guards = training_report.get("source_guards", {})
    if (
        training_report.get("status") != "complete"
        or training_report.get("complete") is not True
        or training_report.get("training_sufficient") is not False
        or training_report.get("length_balance_valid") is not True
        or training_report.get("dev_queries") != 0
        or training_report.get("holdout_queries") != 0
        or training_report.get("confirmation_queries") != 0
        or not source_guards
    ):
        raise ValueError("complete, insufficient, train-only PLAN-N1-110 report required")
    before_guards = {
        path: sha256_file(Path(path))
        for path, expected in source_guards.items()
        if Path(path).is_file() and sha256_file(Path(path)) == expected
    }
    if len(before_guards) != len(source_guards):
        raise ValueError("training report source guard is stale before shard analysis")

    models = []
    for seed in SEEDS:
        model = next(
            (
                record
                for record in training_report["models"]
                if record.get("seed") == seed and record.get("arm") == "position"
            ),
            None,
        )
        if model is None:
            raise ValueError(f"missing position model for seed {seed}")
        shard_identity, record = _load_shard(model)
        final_rows = [
            row
            for row in record["rows"]
            if row.get("completed_updates") == 24576
            and row.get("evaluation_set") in {"factorial_original", "factorial_rename"}
        ]
        views = {}
        for label in ("original", "rename"):
            key = f"factorial_{label}"
            rows = [row for row in final_rows if row["evaluation_set"] == key]
            summary = summarize_factorial_view(rows)
            recorded = model["milestones"][-1]["factorial"][label]
            if (
                summary["joint_four_exact"]["free"] != recorded["joint_four_exact"]
                or {
                    condition: summary["condition_counts"][condition]["free_exact"]
                    for condition in CONDITIONS
                }
                != recorded["by_condition"]
            ):
                raise ValueError(
                    f"row-level factorial summary differs from report for seed {seed}/{label}"
                )
            views[label] = summary
        original_joint = set(views["original"]["joint_four_exact"]["free_parent_ids"])
        rename_joint = set(views["rename"]["joint_four_exact"]["free_parent_ids"])
        models.append(
            {
                "seed": seed,
                "training_sufficient": model["training_sufficient"],
                "source_shard": shard_identity,
                "final_factorial_rows_reused": len(final_rows),
                "views": views,
                "joint_success_overlap": {
                    "both": sorted(original_joint & rename_joint),
                    "original_only": sorted(original_joint - rename_joint),
                    "rename_only": sorted(rename_joint - original_joint),
                },
            }
        )

    after_guards = {
        path: sha256_file(Path(path))
        for path, expected in source_guards.items()
        if Path(path).is_file() and sha256_file(Path(path)) == expected
    }
    if before_guards != after_guards or sha256_bytes(report_bytes) != sha256_file(TRAIN_REPORT):
        raise ValueError("a frozen training source changed during offline analysis")

    cpu_seconds = round(time.process_time() - started_cpu, 6)
    return {
        "schema": "taiji-n1-length-balance-training-failure-attribution-v1",
        "status": "complete",
        "exploratory_train_only_analysis": True,
        "training_report_path": str(TRAIN_REPORT.relative_to(ROOT)).replace("\\", "/"),
        "training_report_sha256": sha256_bytes(report_bytes),
        "plan_path": str(PLAN.relative_to(ROOT)).replace("\\", "/"),
        "plan_sha256": sha256_file(PLAN),
        "source_guard_count": len(source_guards),
        "sources_unchanged": True,
        "models": models,
        "model_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "cpu_seconds": cpu_seconds,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "cpu_limit_passed": cpu_seconds <= CPU_LIMIT_SECONDS,
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_new(path: Path, record: dict) -> None:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "read_only": True,
                    "analysis": "sealed PLAN-N1-110 training score shards only",
                    "output": str(OUT_REPORT.relative_to(ROOT)).replace("\\", "/"),
                }
            )
        )
        return 0
    result = analyze()
    write_new(OUT_REPORT, result)
    print(
        json.dumps(
            {
                "complete": result["status"] == "complete",
                "cpu_seconds": result["cpu_seconds"],
                "seeds": len(result["models"]),
                "model_queries": result["model_queries"],
                "output": str(OUT_REPORT.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
