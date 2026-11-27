"""PLAN-N1-156：对N1-155A已消费的train-only评分做离线逐题配对归因。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder

PLAN = ROOT / "plans/reference/PLAN-N1-156_plan155a_train_score_itemwise_attribution_20261020.md"
PLAN_SHA256 = "6d48c54b45cf6cb61c3bb63c68e8d02520fa6e9e81980d1d740490595d71aa03"
N155A_RESULT = ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_result_20261020.json"
N155A_RESULT_SHA256 = "e577d4dda753d016a46e11ab2e0202b2845206af107c54f87aff7f9fdf1ebb11"
PROGRESS = ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_progress_20261020.jsonl"
PROGRESS_SHA256 = "83e951930272d30ab3b70a0164dba0ef628d8876a8138708be08a97d3a7acf73"
REPORT = ROOT / "reports/taiji_n1_plan156_plan155a_train_score_itemwise_attribution_20261020.json"
TEST = (
    ROOT
    / "tests/taiji_native/test_n1_plan156_plan155a_train_score_itemwise_attribution_contract.py"
)

BRANCHES = ("unknown8_rebalanced", "uniform_length_replay")
GATE_EVENT = "original_train_gate_row"
LENGTH_EVENT = "length_score_row"
EXPECTED_COUNTS = {
    "training_update": 55_296,
    GATE_EVENT: 5_760,
    LENGTH_EVENT: 1_728,
}
GATE_KEY_FIELDS = ("seed", "evaluation_set", "id", "kind", "condition", "case_id")
LENGTH_KEY_FIELDS = ("seed", "context_length", "kind", "id", "split")
GATE_GROUP_FIELDS = ("seed", "evaluation_set", "kind", "condition")
LENGTH_GROUP_FIELDS = ("seed", "context_length", "kind")
LENGTH_TRANSITIONS = ((8, 16), (16, 32), (8, 32))
CPU_LIMIT_SECONDS = 60.0


class Refuse(RuntimeError):
    """来源、分母或配对合同不成立。"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _field_key(row: dict, fields: tuple[str, ...]) -> tuple:
    return tuple(row.get(field) for field in fields)


def paired_summary(
    rows: list[dict], *, key_fields: tuple[str, ...], group_fields: tuple[str, ...]
) -> list[dict]:
    keyed: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        branch = str(row.get("branch"))
        if branch not in BRANCHES:
            raise Refuse(f"unexpected branch in scored row: {branch}")
        key = _field_key(row, key_fields)
        arms = keyed.setdefault(key, {})
        if branch in arms:
            raise Refuse(f"duplicate paired train-score key: {key}")
        arms[branch] = row

    groups: dict[tuple, Counter] = defaultdict(Counter)
    for key, arms in keyed.items():
        if set(arms) != set(BRANCHES):
            raise Refuse(f"candidate/control key coverage differs: {key}")
        candidate = arms["unknown8_rebalanced"].get("free_exact") is True
        control = arms["uniform_length_replay"].get("free_exact") is True
        group = _field_key(arms["unknown8_rebalanced"], group_fields)
        counts = groups[group]
        counts["items"] += 1
        counts["candidate_exact"] += candidate
        counts["control_exact"] += control
        counts["both_exact"] += candidate and control
        counts["candidate_only_correct"] += candidate and not control
        counts["control_only_correct"] += control and not candidate
        counts["both_wrong"] += not candidate and not control

    output = []
    for group in sorted(
        groups, key=lambda key: tuple("" if value is None else str(value) for value in key)
    ):
        row = dict(zip(group_fields, group, strict=True))
        row.update({key: int(value) for key, value in sorted(groups[group].items())})
        output.append(row)
    return output


def _length_transition_summary(rows: list[dict]) -> list[dict]:
    by_item: dict[tuple, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (int(row["seed"]), str(row["kind"]), str(row["id"]), str(row["branch"]))
        length = int(row["context_length"])
        if row["branch"] in by_item[key][length]:
            raise Refuse(f"duplicate length-score item: {key}/{length}")
        by_item[key][length][str(row["branch"])] = row

    output = []
    counts: dict[tuple, Counter] = defaultdict(Counter)
    for (seed, kind, _item_id, branch), by_length in by_item.items():
        for before, after in LENGTH_TRANSITIONS:
            if before not in by_length or after not in by_length:
                raise Refuse(f"nested length-score item missing a context: {seed}/{kind}/{branch}")
            start = by_length[before][branch].get("free_exact") is True
            end = by_length[after][branch].get("free_exact") is True
            row_counts = counts[(seed, branch, kind, before, after)]
            row_counts["items"] += 1
            row_counts["wrong_to_right"] += not start and end
            row_counts["right_to_wrong"] += start and not end
            row_counts["right_both"] += start and end
            row_counts["wrong_both"] += not start and not end

    for key in sorted(counts):
        seed, branch, kind, before, after = key
        row = {
            "seed": seed,
            "branch": branch,
            "kind": kind,
            "from_length": before,
            "to_length": after,
        }
        row.update({field: int(value) for field, value in sorted(counts[key].items())})
        output.append(row)
    return output


def analyze() -> dict:
    started = time.process_time()
    if REPORT.exists():
        raise FileExistsError(f"N1-156 report is write-once: {REPORT}")
    if sha256(PLAN) != PLAN_SHA256 or sha256(N155A_RESULT) != N155A_RESULT_SHA256:
        raise Refuse("N1-156 plan or N1-155A result hash changed")
    result = read_json(N155A_RESULT)
    if (
        result.get("complete") is not True
        or result.get("source_hashes_unchanged") is not True
        or result.get("training_updates") != EXPECTED_COUNTS["training_update"]
        or result.get("train_score_queries") != 7_488
        or result.get("progress_integrity", {}).get("complete") is not True
        or result.get("can_promote") is not False
        or result.get("fresh_diagnostic_queries") != 0
    ):
        raise Refuse("N1-155A execution result is not eligible for train-only attribution")
    if sha256(PROGRESS) != PROGRESS_SHA256 or sha256(PROGRESS) != result["progress_integrity"].get(
        "progress_sha256"
    ):
        raise Refuse("N1-155A progress ledger SHA changed")
    if not n150_builder.verify_hashes(result.get("source_hashes", {})):
        raise Refuse("N1-155A bound source/checkpoint hashes changed")

    events = Counter()
    gate_rows = []
    length_rows = []
    malformed = []
    total_rows = 0
    with PROGRESS.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            total_rows += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                malformed.append({"line": number, "error": type(exc).__name__})
                continue
            event = str(row.get("event"))
            events[event] += 1
            if event == GATE_EVENT:
                gate_rows.append(row)
            elif event == LENGTH_EVENT:
                length_rows.append(row)
    if malformed or dict(events) != EXPECTED_COUNTS or total_rows != sum(EXPECTED_COUNTS.values()):
        raise Refuse(
            f"N1-155A progress denominator/schema differs: {dict(events)} malformed={len(malformed)}"
        )
    if (
        len(gate_rows) != EXPECTED_COUNTS[GATE_EVENT]
        or len(length_rows) != EXPECTED_COUNTS[LENGTH_EVENT]
    ):
        raise Refuse("N1-156 scored-row extraction differs from the frozen denominator")

    gate_paired = paired_summary(
        gate_rows,
        key_fields=GATE_KEY_FIELDS,
        group_fields=GATE_GROUP_FIELDS,
    )
    length_paired = paired_summary(
        length_rows,
        key_fields=LENGTH_KEY_FIELDS,
        group_fields=LENGTH_GROUP_FIELDS,
    )
    length_transitions = _length_transition_summary(length_rows)
    if sha256(PROGRESS) != PROGRESS_SHA256 or sha256(N155A_RESULT) != N155A_RESULT_SHA256:
        raise Refuse("N1-155A result or progress changed during analysis")

    source_hashes = {
        str(PLAN.resolve()): PLAN_SHA256,
        str(N155A_RESULT.resolve()): N155A_RESULT_SHA256,
        str(PROGRESS.resolve()): PROGRESS_SHA256,
        str(Path(__file__).resolve()): sha256(Path(__file__)),
        str(TEST.resolve()): sha256(TEST),
    }
    report = {
        "schema": "taiji-n1-plan156-plan155a-train-score-itemwise-attribution-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "n155a_result_sha256": N155A_RESULT_SHA256,
        "progress_sha256": PROGRESS_SHA256,
        "n155a_bound_sources_unchanged": True,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": n150_builder.verify_hashes(source_hashes),
        "event_counts": dict(sorted(events.items())),
        "scored_rows": {
            "original_train_gate": len(gate_rows),
            "length_score": len(length_rows),
            "paired_original_train_gate": len(gate_paired),
            "paired_length_score": len(length_paired),
        },
        "gate_paired_by_seed_evaluation_set_kind_condition": gate_paired,
        "length_paired_by_seed_context_length_kind": length_paired,
        "length_transitions_by_seed_branch_kind": length_transitions,
        "cpu_seconds": round(time.process_time() - started, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "inference_calls": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "summary_zh": (
            f"完成N1-155A的{len(gate_rows)+len(length_rows)}条train-only评分逐题配对归因；"
            f"CPU{time.process_time()-started:.3f}/{CPU_LIMIT_SECONDS:.0f}秒，0模型/检查点加载、"
            "0推理/查询/更新/dev/holdout/confirmation，can_promote=false。"
        ),
    }
    if report["cpu_seconds"] > CPU_LIMIT_SECONDS or report["source_hashes_unchanged"] is not True:
        raise Refuse("N1-156 CPU ceiling or output source binding failed")
    write_new(REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-156 N1-155A train-only itemwise attribution"
    )
    parser.parse_args()
    report = analyze()
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "status",
                    "complete",
                    "scored_rows",
                    "cpu_seconds",
                    "source_hashes_unchanged",
                    "model_loads",
                    "checkpoint_loads",
                    "model_queries",
                    "parameter_updates",
                    "dev_queries",
                    "holdout_queries",
                    "confirmation_queries",
                    "can_promote",
                    "summary_zh",
                )
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
