"""PLAN-N1-162C：对N1-162B训练门与长度评分做离线逐题归因。"""

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

from scripts.training import build_taiji_n1_plan162_question_surface_schedule_and_face as builder
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder

PLAN = ROOT / "plans/reference/PLAN-N1-162C_train_only_itemwise_gate_attribution_20261020.md"
PLAN_SHA256 = "2da15aa040cf94b33795d5eabdd3b88c327570d828c91ccb6a088da48e4887ca"
N162_PLAN = builder.PLAN
N162_PLAN_SHA256 = builder.PLAN_SHA256
N162A_PLAN = builder.BUDGET_ADDENDUM
N162A_PLAN_SHA256 = builder.BUDGET_ADDENDUM_SHA256
N162B_PLAN = ROOT / "plans/reference/PLAN-N1-162B_首步解析器修正与隔离续执行_20261020.md"
N162B_PLAN_SHA256 = "70f84638da596795e2c3c06e86839f56cba07232b975332b19907076d8f76847"
N162B_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-162B_execution_closeout_20261020.md"
N162B_RESULT = ROOT / "reports/taiji_n1_plan162b_question_surface_result_20261020.json"
N162B_RESULT_SHA256 = "19dfaeb2086abaa503ee5150dba3bbfc9756d0892e8b6b54222fbfc4ecba2c31"
PROGRESS = ROOT / "reports/taiji_n1_plan162b_question_surface_progress_20261020.jsonl"
PROGRESS_SHA256 = "48212bc24ff9267390f7568142a0afd99dfeb39ab787e36c420d6d5d28ed4614"
SCHEDULE = builder.SCHEDULE
REPORT = ROOT / "reports/taiji_n1_plan162b_train_only_itemwise_attribution_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan162b_train_only_itemwise_attribution_contract.py"

SEEDS = (11, 12, 13)
ARMS = ("surface_control", "surface_augmented")
ENDPOINTS = tuple((seed, arm) for seed in SEEDS for arm in ARMS)
KINDS = ("lookup", "revision", "transition", "unknown")
GATE_ROWS_PER_ENDPOINT = 960
LENGTH_ROWS_PER_ENDPOINT = 288
EXPECTED_EVENTS = {
    "training_update": 55_296,
    "original_train_gate_row": 5_760,
    "length_score_row": 1_728,
}
GATE_KEY_FIELDS = ("evaluation_set", "kind", "id", "condition", "case_id")
LENGTH_KEY_FIELDS = ("context_length", "kind", "id", "split")
CPU_LIMIT_SECONDS = 240.0


class Refuse(RuntimeError):
    """封存来源、progress分母或逐题配对合同不成立。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


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


def _pair_key(row: dict, fields: tuple[str, ...]) -> tuple:
    return tuple(row.get(field) for field in fields)


def bind_endpoint_chunks(rows: list[dict], *, rows_per_endpoint: int, event: str) -> tuple[list[dict], int]:
    require(len(rows) == rows_per_endpoint * len(ENDPOINTS), f"{event} denominator differs")
    bound = []
    overwritten_arm_rows = 0
    for endpoint_index, (seed, arm) in enumerate(ENDPOINTS):
        start = endpoint_index * rows_per_endpoint
        chunk = rows[start : start + rows_per_endpoint]
        require(len(chunk) == rows_per_endpoint, f"{event} endpoint chunk is truncated: {seed}/{arm}")
        require(all(int(row.get("seed", -1)) == seed for row in chunk), f"{event} endpoint order/seed mismatch: {seed}/{arm}")
        for row in chunk:
            item = dict(row)
            logged_arm = item.get("arm")
            if event == "original_train_gate_row":
                require(logged_arm == "position", f"unexpected inherited N1-132 arm field: {logged_arm}")
                overwritten_arm_rows += 1
            elif logged_arm is not None:
                require(logged_arm == arm, f"{event} external arm label differs: {seed}/{arm}/{logged_arm}")
            item["experiment_arm"] = arm
            bound.append(item)
    return bound, overwritten_arm_rows


def _summarize_gate_endpoint(rows: list[dict]) -> dict:
    by_set = defaultdict(list)
    for row in rows:
        by_set[str(row.get("evaluation_set"))].append(row)
    required = {"original": 192, "rename": 192, "factorial_original": 192, "factorial_rename": 192, "pair": 96, "rename_pair": 96}
    require({key: len(by_set.get(key, [])) for key in required} == required, "gate evaluation-set denominators differ")
    original = by_set["original"]
    rename = by_set["rename"]
    pair = by_set["pair"]
    rename_pair = by_set["rename_pair"]

    def kind_counts(selected: list[dict]) -> dict[str, int]:
        values = Counter(str(row["kind"]) for row in selected if row.get("free_exact") is True)
        return {kind: int(values[kind]) for kind in KINDS}

    def factorial(view: str) -> dict:
        selected = by_set[f"factorial_{view}"]
        condition_counts = Counter(
            str(row["condition"]) for row in selected if row.get("free_exact") is True
        )
        by_parent = defaultdict(list)
        for row in selected:
            by_parent[str(row["id"])].append(row)
        require(len(by_parent) == 48 and all(len(items) == 4 for items in by_parent.values()), f"factorial parent/condition coverage differs: {view}")
        joint = sum(all(row.get("free_exact") is True for row in items) for items in by_parent.values())
        return {
            "items": len(selected),
            "parents": len(by_parent),
            "by_condition": {name: int(condition_counts[name]) for name in ("original", "direction_opposite", "turn_flip", "both")},
            "joint_four_exact": int(joint),
        }

    return {
        "free_exact": sum(row.get("free_exact") is True for row in original),
        "items": len(original),
        "free_by_kind": kind_counts(original),
        "rename_free_exact": sum(row.get("free_exact") is True for row in rename),
        "rename_free_by_kind": kind_counts(rename),
        "pair_exact": sum(row.get("free_exact") is True for row in pair),
        "rename_pair_exact": sum(row.get("free_exact") is True for row in rename_pair),
        "factorial": {"original": factorial("original"), "rename": factorial("rename")},
    }


def _sufficient(summary: dict) -> bool:
    return bool(
        summary["free_exact"] >= 173
        and all(summary["free_by_kind"].get(kind, 0) >= 40 for kind in KINDS)
        and summary["pair_exact"] >= 87
        and summary["rename_free_exact"] >= 173
        and all(summary["rename_free_by_kind"].get(kind, 0) >= 40 for kind in KINDS)
        and summary["rename_pair_exact"] >= 87
        and all(summary["factorial"][view]["joint_four_exact"] >= 44 for view in ("original", "rename"))
    )


def _paired_gate_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    keyed: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (int(row["seed"]), *_pair_key(row, GATE_KEY_FIELDS))
        pair = keyed.setdefault(key, {})
        arm = str(row["experiment_arm"])
        require(arm not in pair, f"duplicate gate row pair: {key}/{arm}")
        pair[arm] = row
    require(all(set(pair) == set(ARMS) for pair in keyed.values()), "control/augmented gate key coverage differs")
    groups = defaultdict(Counter)
    errors = []
    for key, pair in keyed.items():
        control = pair["surface_control"]
        augmented = pair["surface_augmented"]
        group = (int(control["seed"]), str(control["evaluation_set"]), str(control["kind"]), control.get("condition"))
        counts = groups[group]
        control_exact = control.get("free_exact") is True
        augmented_exact = augmented.get("free_exact") is True
        counts["items"] += 1
        counts["control_exact"] += control_exact
        counts["augmented_exact"] += augmented_exact
        counts["both_exact"] += control_exact and augmented_exact
        counts["augmented_only_correct"] += augmented_exact and not control_exact
        counts["control_only_correct"] += control_exact and not augmented_exact
        counts["both_wrong"] += not control_exact and not augmented_exact
        counts["raw_output_changed"] += control.get("raw_hex") != augmented.get("raw_hex")
        if control_exact != augmented_exact:
            errors.append({
                "seed": int(control["seed"]),
                "evaluation_set": control["evaluation_set"],
                "kind": control["kind"],
                "condition": control.get("condition"),
                "parent_id": control["id"],
                "control_free_exact": control_exact,
                "augmented_free_exact": augmented_exact,
                "control_raw_hex": control.get("raw_hex"),
                "augmented_raw_hex": augmented.get("raw_hex"),
                "control_stop_reason": control.get("stop_reason"),
                "augmented_stop_reason": augmented.get("stop_reason"),
            })
    summary = []
    for key in sorted(groups, key=lambda value: tuple("" if item is None else str(item) for item in value)):
        seed, view, kind, condition = key
        row = {"seed": seed, "evaluation_set": view, "kind": kind, "condition": condition}
        row.update({name: int(value) for name, value in sorted(groups[key].items())})
        summary.append(row)
    return summary, errors


def _paired_length_rows(rows: list[dict]) -> list[dict]:
    keyed: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (int(row["seed"]), *_pair_key(row, LENGTH_KEY_FIELDS))
        pair = keyed.setdefault(key, {})
        arm = str(row["experiment_arm"])
        require(arm not in pair, f"duplicate length row pair: {key}/{arm}")
        pair[arm] = row
    require(all(set(pair) == set(ARMS) for pair in keyed.values()), "control/augmented length key coverage differs")
    groups = defaultdict(Counter)
    for key, pair in keyed.items():
        control, augmented = pair["surface_control"], pair["surface_augmented"]
        group = (int(control["seed"]), str(control["kind"]), int(control["context_length"]))
        counts = groups[group]
        control_exact = control.get("free_exact") is True
        augmented_exact = augmented.get("free_exact") is True
        counts["items"] += 1
        counts["control_exact"] += control_exact
        counts["augmented_exact"] += augmented_exact
        counts["augmented_only_correct"] += augmented_exact and not control_exact
        counts["control_only_correct"] += control_exact and not augmented_exact
        counts["both_wrong"] += not control_exact and not augmented_exact
    output = []
    for key in sorted(groups):
        seed, kind, length = key
        row = {"seed": seed, "kind": kind, "context_length": length}
        row.update({name: int(value) for name, value in sorted(groups[key].items())})
        output.append(row)
    return output


def _template_exposure(schedule: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    counts = defaultdict(Counter)
    for row in schedule.get("schedule_rows", []):
        counts[str(row["parent_id"])][f"{row['context_length']}/{row['augmented_template_id']}"] += 1
    by_parent = {parent: dict(sorted(counter.items())) for parent, counter in sorted(counts.items())}
    by_kind_length = Counter()
    parent_kind = {}
    for row in schedule["schedule_rows"]:
        parent_kind[str(row["parent_id"])] = str(row["kind"])
        by_kind_length[(str(row["kind"]), int(row["context_length"]), str(row["augmented_template_id"]))] += 1
    aggregate = {
        f"{kind}/{length}/{template}": int(value)
        for (kind, length, template), value in sorted(by_kind_length.items())
    }
    require(len(by_parent) == 192 and all(sum(values.values()) == 48 for values in by_parent.values()), "template exposure parent schedule differs")
    return by_parent, aggregate


def _read_progress() -> tuple[dict[str, int], list[dict], list[dict], int]:
    counts = Counter()
    gate_rows = []
    length_rows = []
    total = 0
    malformed = []
    with PROGRESS.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            total += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                malformed.append({"line": number, "error": type(exc).__name__})
                continue
            event = str(row.get("event"))
            counts[event] += 1
            if event == "original_train_gate_row":
                gate_rows.append(row)
            elif event == "length_score_row":
                length_rows.append(row)
    require(not malformed, f"N1-162B progress malformed rows: {malformed[:3]}")
    require(dict(counts) == EXPECTED_EVENTS and total == sum(EXPECTED_EVENTS.values()), f"N1-162B progress event counts differ: {dict(counts)}")
    return dict(sorted(counts.items())), gate_rows, length_rows, total


def analyze() -> dict:
    started = time.process_time()
    if REPORT.exists():
        raise FileExistsError(f"N1-162C report is write-once: {REPORT}")
    require(sha256(PLAN) == PLAN_SHA256, "PLAN-N1-162C plan SHA changed")
    require(sha256(N162_PLAN) == N162_PLAN_SHA256, "PLAN-N1-162 plan SHA changed")
    require(sha256(N162A_PLAN) == N162A_PLAN_SHA256, "PLAN-N1-162A budget addendum SHA changed")
    require(sha256(N162B_PLAN) == N162B_PLAN_SHA256, "PLAN-N1-162B correction SHA changed")
    require(N162B_CLOSEOUT.is_file(), "PLAN-N1-162B closeout is missing")
    require(sha256(N162B_RESULT) == N162B_RESULT_SHA256, "N1-162B result SHA changed")
    require(sha256(PROGRESS) == PROGRESS_SHA256, "N1-162B progress SHA changed")
    result = read_json(N162B_RESULT)
    require(
        result.get("complete") is True
        and result.get("status") == "complete"
        and result.get("training_updates") == EXPECTED_EVENTS["training_update"]
        and result.get("train_score_queries") == 7_488
        and result.get("fresh_queries") == 0
        and result.get("dev_queries") == 0
        and result.get("holdout_queries") == 0
        and result.get("confirmation_queries") == 0
        and result.get("can_promote") is False
        and result.get("source_hashes_unchanged") is True
        and result.get("progress_integrity", {}).get("sha256") == PROGRESS_SHA256,
        "N1-162B result is not eligible for train-only attribution",
    )
    require(len(result.get("models", [])) == 6, "N1-162B model endpoint count differs")
    for model in result["models"]:
        restore = model.get("final_independent_restore", {})
        require(
            restore.get("complete") is True
            and restore.get("equal") is True
            and restore.get("fingerprint") == model.get("final_bundle_fingerprint"),
            f"N1-162B final checkpoint restore evidence differs: {model.get('seed')}/{model.get('arm')}",
        )

    source_hashes = dict(result["source_hashes"])
    schedule_path = str(SCHEDULE.resolve())
    require(schedule_path in source_hashes and sha256(SCHEDULE) == source_hashes[schedule_path], "N1-162B frozen schedule SHA changed")
    require(n150_builder.verify_hashes(source_hashes), "N1-162B bound input/checkpoint hashes changed")
    counts, raw_gate_rows, raw_length_rows, line_count = _read_progress()
    bound_gate_rows, overwritten_arm_rows = bind_endpoint_chunks(
        raw_gate_rows, rows_per_endpoint=GATE_ROWS_PER_ENDPOINT, event="original_train_gate_row"
    )
    bound_length_rows, length_arm_mismatches = bind_endpoint_chunks(
        raw_length_rows, rows_per_endpoint=LENGTH_ROWS_PER_ENDPOINT, event="length_score_row"
    )
    schedule = read_json(SCHEDULE)
    require(builder.validate_schedule(schedule)["accepted"] is True, "N1-162 schedule contract changed")
    exposure_by_parent, exposure_by_kind_length = _template_exposure(schedule)

    gate_endpoint_summaries = []
    gate_recalculation_matches = True
    for seed, arm in ENDPOINTS:
        endpoint_rows = [row for row in bound_gate_rows if (int(row["seed"]), str(row["experiment_arm"])) == (seed, arm)]
        require(len(endpoint_rows) == GATE_ROWS_PER_ENDPOINT, f"N1-162 gate chunk count differs: {seed}/{arm}")
        recomputed = _summarize_gate_endpoint(endpoint_rows)
        model = next(row for row in result["models"] if int(row["seed"]) == seed and str(row["arm"]) == arm)
        stored = model["original_train_gate"]
        fields = ("free_exact", "items", "free_by_kind", "rename_free_exact", "rename_free_by_kind", "pair_exact", "rename_pair_exact", "factorial")
        matches = all(recomputed[field] == stored.get(field) for field in fields)
        sufficient = _sufficient(recomputed)
        gate_recalculation_matches = gate_recalculation_matches and matches and sufficient == model["training_sufficient"]
        gate_endpoint_summaries.append({
            "seed": seed,
            "arm": arm,
            "training_sufficient_stored": model["training_sufficient"],
            "training_sufficient_recomputed": sufficient,
            "gate_summary_matches_stored": matches,
            **recomputed,
        })
    require(gate_recalculation_matches, "N1-132 gate recomputation differs from frozen endpoint summaries")

    paired_gate_summary, gate_disagreements = _paired_gate_rows(bound_gate_rows)
    paired_length_summary = _paired_length_rows(bound_length_rows)
    error_rows = []
    for row in gate_disagreements:
        exposure = exposure_by_parent.get(str(row["parent_id"]), {})
        error_rows.append({**row, "augmented_template_exposure_by_length": exposure})
    class_gate_deltas = []
    grouped = defaultdict(dict)
    for row in paired_gate_summary:
        key = (row["seed"], row["evaluation_set"], row["kind"], row["condition"])
        grouped[key] = row
    for (seed, view, kind, condition), row in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        class_gate_deltas.append({
            "seed": seed,
            "evaluation_set": view,
            "kind": kind,
            "condition": condition,
            "items": row["items"],
            "control_exact": row["control_exact"],
            "augmented_exact": row["augmented_exact"],
            "delta": row["augmented_exact"] - row["control_exact"],
            "control_only_correct": row["control_only_correct"],
            "augmented_only_correct": row["augmented_only_correct"],
            "raw_output_changed": row["raw_output_changed"],
        })
    require(sha256(N162B_RESULT) == N162B_RESULT_SHA256 and sha256(PROGRESS) == PROGRESS_SHA256 and n150_builder.verify_hashes(source_hashes), "N1-162B evidence changed during attribution")
    cpu = time.process_time() - started
    complete = cpu <= CPU_LIMIT_SECONDS
    report_sources = {
        **source_hashes,
        str(PLAN.resolve()): PLAN_SHA256,
        str(N162B_PLAN.resolve()): N162B_PLAN_SHA256,
        str(N162B_CLOSEOUT.resolve()): sha256(N162B_CLOSEOUT),
        str(N162B_RESULT.resolve()): N162B_RESULT_SHA256,
        str(PROGRESS.resolve()): PROGRESS_SHA256,
        str(SCHEDULE.resolve()): source_hashes[schedule_path],
        str(Path(__file__).resolve()): sha256(Path(__file__)),
        str(TEST.resolve()): sha256(TEST),
    }
    report = {
        "schema": "taiji-n1-plan162b-train-only-itemwise-attribution-v1",
        "status": "complete" if complete else "stopped_cpu_limit",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "n162b_plan_sha256": N162B_PLAN_SHA256,
        "n162b_result_sha256": N162B_RESULT_SHA256,
        "progress_sha256": PROGRESS_SHA256,
        "schedule_sha256": source_hashes[schedule_path],
        "source_hashes": report_sources,
        "source_hashes_unchanged": n150_builder.verify_hashes(report_sources),
        "progress_line_count": line_count,
        "event_counts": counts,
        "scored_rows": {
            "original_train_gate": len(raw_gate_rows),
            "length_score": len(raw_length_rows),
            "gate_rows_per_endpoint": GATE_ROWS_PER_ENDPOINT,
            "length_rows_per_endpoint": LENGTH_ROWS_PER_ENDPOINT,
        },
        "gate_arm_field_recovery": {
            "binding_method": "frozen endpoint execution order and 960-row chunks; gate-row inherited arm field is position",
            "inherited_arm_value": "position",
            "overwritten_arm_rows": overwritten_arm_rows,
            "length_arm_mismatches": length_arm_mismatches,
            "endpoint_order": [{"seed": seed, "arm": arm} for seed, arm in ENDPOINTS],
        },
        "gate_recalculation_matches_all_six": gate_recalculation_matches,
        "gate_sufficiency_by_endpoint": gate_endpoint_summaries,
        "paired_gate_by_seed_view_kind_condition": paired_gate_summary,
        "paired_gate_delta_by_seed_view_kind_condition": class_gate_deltas,
        "paired_gate_disagreements_with_template_exposure": error_rows,
        "augmented_template_exposure_by_parent": exposure_by_parent,
        "augmented_template_exposure_by_kind_length": exposure_by_kind_length,
        "paired_length_by_seed_kind_length": paired_length_summary,
        "cpu_seconds": round(cpu, 6),
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
        "summary_zh": f"N1-162B训练门逐题归因{'完成' if complete else '触及CPU停止线'}：配对门差异{len(gate_disagreements)}条；六endpoint门复算一致={gate_recalculation_matches}；0加载/查询/更新/新面，CPU{cpu:.3f}/{CPU_LIMIT_SECONDS:.0f}秒。",
    }
    write_new(REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-162C train-only paired gate attribution")
    parser.add_argument("--analyze", action="store_true", required=True)
    args = parser.parse_args()
    del args
    report = analyze()
    print(json.dumps({key: report.get(key) for key in ("status", "complete", "cpu_seconds", "progress_line_count", "gate_recalculation_matches_all_six", "model_loads", "model_queries", "parameter_updates", "can_promote", "summary_zh")}, ensure_ascii=True, sort_keys=True))
    return 0 if report.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
