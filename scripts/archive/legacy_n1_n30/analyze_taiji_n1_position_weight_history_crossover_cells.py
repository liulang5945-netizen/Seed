"""Recompute PLAN-N1-112 cell summaries from its sealed rows without model access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPORT = ROOT / "reports/taiji_n1_position_weight_history_crossover_s13_r1_20261018.json"
OUTPUT = (
    ROOT
    / "reports/taiji_n1_position_weight_history_crossover_s13_r1_cell_summary_correction_20261018.json"
)
SOURCE_REPORT_SHA256 = "e38b08a857ffe69acb8fbf57e040ed672df538b88988dc333e1182e58ffb16bf"
CONDITIONS = ("original", "turn_flip", "direction_opposite", "both")
VIEWS = ("original", "rename")
DIRECTIONS = ("north", "east", "south", "west")
TURNS = ("left", "right")
COMBINATIONS = (
    "baseline_weights+baseline_history",
    "baseline_weights+candidate_history",
    "candidate_weights+baseline_history",
    "candidate_weights+candidate_history",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_combination(rows: list[dict]) -> dict:
    if len(rows) != 384:
        raise ValueError("exactly384 rows required per weight/history combination")
    if len({(row["view"], row["case_id"]) for row in rows}) != 384:
        raise ValueError("duplicate view/case identity")
    summaries = {}
    for view in VIEWS:
        view_rows = [row for row in rows if row["view"] == view]
        if len(view_rows) != 192:
            raise ValueError("each original/rename view must contain192 rows")
        grouped: dict[str, dict[str, dict]] = {}
        for row in view_rows:
            grouped.setdefault(row["parent_id"], {})[row["condition"]] = row
        if len(grouped) != 48 or any(set(group) != set(CONDITIONS) for group in grouped.values()):
            raise ValueError("all48 parents require one row per frozen condition")

        source_cells: dict[str, set[str]] = {}
        for parent, group in grouped.items():
            baseline = group["original"]
            direction, turn = baseline["initial_direction"], baseline["turn"]
            if direction not in DIRECTIONS or turn not in TURNS:
                raise ValueError("original-condition row must define the parent source cell")
            source_cells.setdefault(f"{direction}/{turn}", set()).add(parent)
        if sum(map(len, source_cells.values())) != 48:
            raise ValueError("each parent must belong to one original-condition cell")

        cell_summaries = {}
        for direction in DIRECTIONS:
            for turn in TURNS:
                cell_name = f"{direction}/{turn}"
                parents = source_cells.get(cell_name, set())
                cell_summaries[cell_name] = {
                    "parents": len(parents),
                    "items": len(parents) * len(CONDITIONS),
                    "by_condition": {
                        condition: sum(
                            grouped[parent][condition]["free_exact"] for parent in parents
                        )
                        for condition in CONDITIONS
                    },
                    "joint_four_exact": sum(
                        all(grouped[parent][condition]["free_exact"] for condition in CONDITIONS)
                        for parent in parents
                    ),
                }

        by_condition = {
            condition: sum(row["free_exact"] for row in view_rows if row["condition"] == condition)
            for condition in CONDITIONS
        }
        summaries[view] = {
            "items": 192,
            "parents": 48,
            "by_condition": by_condition,
            "joint_four_exact": sum(
                all(group[condition]["free_exact"] for condition in CONDITIONS)
                for group in grouped.values()
            ),
            "cells": cell_summaries,
        }
    return summaries


def validate_source_report(report: dict) -> None:
    if (
        report.get("schema") != "n1-position-weight-history-crossover-diagnostic-r1-v1"
        or report.get("status") != "complete"
        or report.get("complete") is not True
        or report.get("model_queries") != 1_536
        or report.get("parameter_updates") != 0
        or report.get("optimizer_steps") != 0
        or report.get("checkpoint_saves") != 0
        or any(
            report.get(key) != 0
            for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
        or len(report.get("rows", [])) != 1_536
        or set(report.get("summaries", {})) != set(COMBINATIONS)
    ):
        raise ValueError("complete frozen 1,536-row PLAN-N1-112 report required")


def correction_record() -> dict:
    if sha256_file(SOURCE_REPORT) != SOURCE_REPORT_SHA256:
        raise ValueError("frozen r1 source report hash mismatch")
    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    validate_source_report(report)
    corrected = {}
    for combination in COMBINATIONS:
        rows = [row for row in report["rows"] if row["combination"] == combination]
        summary = summarize_combination(rows)
        stored = report["summaries"][combination]
        for view in VIEWS:
            if any(
                summary[view][key] != stored[view][key]
                for key in ("items", "parents", "by_condition", "joint_four_exact")
            ):
                raise ValueError("corrected cell pass changed sealed overall counts")
            if (
                sum(cell["parents"] for cell in summary[view]["cells"].values()) != 48
                or sum(cell["items"] for cell in summary[view]["cells"].values()) != 192
                or sum(cell["joint_four_exact"] for cell in summary[view]["cells"].values())
                != summary[view]["joint_four_exact"]
            ):
                raise ValueError("corrected source-cell denominators do not reconcile")
        corrected[combination] = summary
    if sha256_file(SOURCE_REPORT) != SOURCE_REPORT_SHA256:
        raise ValueError("frozen r1 source report changed during offline correction")
    return {
        "schema": "n1-position-weight-history-crossover-cell-summary-correction-v1",
        "status": "complete",
        "source_report": {
            "path": str(SOURCE_REPORT.relative_to(ROOT)),
            "sha256": SOURCE_REPORT_SHA256,
        },
        "correction": {
            "field": "summaries[*][original|rename].cells",
            "reason": "the runner grouped each counterfactual row by its condition-mutated direction/turn instead of grouping all four rows by the parent's original-condition direction/turn",
            "method": "assign every parent using its original-condition row, then aggregate all four condition outcomes within that fixed cell",
            "overall_condition_and_joint_counts_unchanged": True,
        },
        "corrected_summaries": corrected,
        "model_loads": 0,
        "checkpoint_payload_reads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "checkpoint_saves": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "sources_unchanged": True,
    }


def write_new(path: Path, record: dict) -> None:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("cell correction is write-once")
    record = correction_record()
    write_new(OUTPUT, record)
    print(
        json.dumps(
            {
                "complete": True,
                "report": str(OUTPUT.relative_to(ROOT)),
                "source_report_sha256": SOURCE_REPORT_SHA256,
                "model_queries": 0,
                "parameter_updates": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
