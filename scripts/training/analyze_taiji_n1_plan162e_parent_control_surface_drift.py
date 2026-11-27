"""PLAN-N1-162E：离线分解N1-155A→N1-162B的续训与表面增量。"""

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

from scripts.training import analyze_taiji_n1_plan162b_train_only_itemwise_attribution as n162c
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import (
    run_taiji_n1_plan155_short_unknown_context_length_training as n155_runner,
)
from scripts.training import seal_ast  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-162E_parent_control_surface_drift_attribution_20261020.md"
PLAN_SHA256 = "f5f66410ab357d51d543d871ba697e19396da5b8a942d99d232a1c292efa60a5"
N155A_RESULT = n155_runner.RESULT
N155A_RESULT_SHA256 = "e577d4dda753d016a46e11ab2e0202b2845206af107c54f87aff7f9fdf1ebb11"
N155A_PROGRESS = n155_runner.PROGRESS
N155A_PROGRESS_SHA256 = "83e951930272d30ab3b70a0164dba0ef628d8876a8138708be08a97d3a7acf73"
N162B_RESULT = n162c.N162B_RESULT
N162B_RESULT_SHA256 = n162c.N162B_RESULT_SHA256
N162B_PROGRESS = n162c.PROGRESS
N162B_PROGRESS_SHA256 = n162c.PROGRESS_SHA256
N162B_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-162B_execution_closeout_20261020.md"
N162C_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-162C_execution_closeout_20261020.md"
N162C_REPORT = n162c.REPORT
N162D_REPORT = ROOT / "reports/taiji_n1_plan162d_transition_output_type_attribution_20261020.json"
N162D_REPORT_SHA256 = "27fc0b88ea53fb3d9f211cbaee9997b2372e2bf56dc9f04c93190e98380ef6a2"
N162C_REPORT_SHA256 = "f98fa027a55e83a03eb18a1288b61168f619b6ace0e31662d596ce82bf584e63"
N162B_CLOSEOUT_SHA256 = "25e78d238e30a29e0778109339a4f569d6082dc1dc71dc70ed28bf968be953ba"
N162C_CLOSEOUT_SHA256 = "d61d9c02f79693fcee7eb67f17f84915af86296aa4bcfc7b484c7977ff55374f"
REPORT = ROOT / "reports/taiji_n1_plan162e_parent_control_surface_drift_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan162e_parent_control_surface_drift_contract.py"

SEEDS = (11, 12, 13)
ARMS = ("n155a_candidate", "surface_control", "surface_augmented")
GATE_ROWS_PER_ENDPOINT = 960
LENGTH_ROWS_PER_ENDPOINT = 288
KINDS = ("lookup", "revision", "transition", "unknown")
GATE_KEY_FIELDS = ("evaluation_set", "kind", "id", "condition", "case_id")
LENGTH_KEY_FIELDS = ("context_length", "kind", "id", "split")
CPU_LIMIT_SECONDS = 180.0


class Refuse(RuntimeError):
    """三方续训历史、gate行或length配对合同不成立。"""


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


def _merge_hashes(target: dict[str, str], incoming: dict[str, str]) -> None:
    for path, digest in incoming.items():
        old = target.get(path)
        if old is not None and old != digest and not seal_ast.recorded_value_attested(path, old):
            raise Refuse(f"conflicting source hash: {path}")
        target[path] = digest


def _read_events(
    path: Path, *, expected_counts: dict[str, int]
) -> tuple[Counter, list[dict], list[dict], int]:
    counts = Counter()
    gate = []
    length = []
    total = 0
    malformed = []
    with path.open(encoding="utf-8") as stream:
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
                gate.append(row)
            elif event == "length_score_row":
                length.append(row)
    require(not malformed, f"malformed progress rows: {malformed[:3]}")
    require(dict(counts) == expected_counts, f"progress event count mismatch: {dict(counts)}")
    require(total == sum(expected_counts.values()), "progress line count mismatch")
    return counts, gate, length, total


def _tag_n155a(rows: list[dict], *, event: str, per_seed: int) -> list[dict]:
    bound = []
    require(len(rows) == per_seed * len(SEEDS), f"N1-155A {event} candidate row count differs")
    for seed_index, seed in enumerate(SEEDS):
        chunk = rows[seed_index * per_seed : (seed_index + 1) * per_seed]
        require(
            len(chunk) == per_seed and all(int(row.get("seed", -1)) == seed for row in chunk),
            f"N1-155A {event} seed chunk differs: {seed}",
        )
        require(
            all(row.get("branch") == "unknown8_rebalanced" for row in chunk),
            f"N1-155A {event} branch differs: {seed}",
        )
        bound.extend({**row, "lineage_arm": "n155a_candidate"} for row in chunk)
    return bound


def _n155a_gate_rows(rows: list[dict]) -> list[dict]:
    candidates = [row for row in rows if row.get("branch") == "unknown8_rebalanced"]
    return _tag_n155a(candidates, event="gate", per_seed=GATE_ROWS_PER_ENDPOINT)


def _n155a_length_rows(rows: list[dict]) -> list[dict]:
    candidates = [row for row in rows if row.get("branch") == "unknown8_rebalanced"]
    return _tag_n155a(candidates, event="length", per_seed=LENGTH_ROWS_PER_ENDPOINT)


def _three_way_gate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (int(row["seed"]), *(row.get(field) for field in GATE_KEY_FIELDS))
        lineage = str(row.get("lineage_arm", row.get("experiment_arm")))
        pair = by_key.setdefault(key, {})
        require(lineage not in pair, f"duplicate three-way gate item: {key}/{lineage}")
        pair[lineage] = row
    require(
        all(set(pair) == set(ARMS) for pair in by_key.values()),
        "three-way gate key coverage differs",
    )
    groups = defaultdict(Counter)
    trajectories = Counter()
    examples = []
    for _key, items in by_key.items():
        parent, control, augmented = (items[name] for name in ARMS)
        p_ok = parent.get("free_exact") is True
        c_ok = control.get("free_exact") is True
        a_ok = augmented.get("free_exact") is True
        trajectory = (p_ok, c_ok, a_ok)
        trajectories[trajectory] += 1
        group = (
            int(parent["seed"]),
            str(parent["evaluation_set"]),
            str(parent["kind"]),
            parent.get("condition"),
        )
        counts = groups[group]
        counts["items"] += 1
        counts["n155a_exact"] += p_ok
        counts["control_exact"] += c_ok
        counts["augmented_exact"] += a_ok
        counts["parent_to_control_regression"] += p_ok and not c_ok
        counts["parent_to_control_improvement"] += not p_ok and c_ok
        counts["control_to_augmented_regression"] += c_ok and not a_ok
        counts["control_to_augmented_improvement"] += not c_ok and a_ok
        counts["all_exact"] += p_ok and c_ok and a_ok
        counts["all_wrong"] += not p_ok and not c_ok and not a_ok
        if p_ok != c_ok or c_ok != a_ok:
            examples.append(
                {
                    "seed": int(parent["seed"]),
                    "evaluation_set": parent["evaluation_set"],
                    "kind": parent["kind"],
                    "condition": parent.get("condition"),
                    "parent_id": parent["id"],
                    "n155a_exact": p_ok,
                    "surface_control_exact": c_ok,
                    "surface_augmented_exact": a_ok,
                    "n155a_raw_hex": parent.get("raw_hex"),
                    "surface_control_raw_hex": control.get("raw_hex"),
                    "surface_augmented_raw_hex": augmented.get("raw_hex"),
                }
            )
    summary = []
    for key, counts in sorted(
        groups.items(), key=lambda item: tuple("" if x is None else str(x) for x in item[0])
    ):
        seed, view, kind, condition = key
        row = {"seed": seed, "evaluation_set": view, "kind": kind, "condition": condition}
        row.update({name: int(value) for name, value in sorted(counts.items())})
        summary.append(row)
    trajectory_rows = []
    for (p_ok, c_ok, a_ok), count in sorted(trajectories.items()):
        trajectory_rows.append(
            {
                "n155a_correct": p_ok,
                "surface_control_correct": c_ok,
                "surface_augmented_correct": a_ok,
                "items": int(count),
            }
        )
    return summary, trajectory_rows, examples


def _three_way_length(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (int(row["seed"]), *(row.get(field) for field in LENGTH_KEY_FIELDS))
        lineage = str(row.get("lineage_arm", row.get("experiment_arm")))
        pair = by_key.setdefault(key, {})
        require(lineage not in pair, f"duplicate three-way length item: {key}/{lineage}")
        pair[lineage] = row
    require(
        all(set(pair) == set(ARMS) for pair in by_key.values()),
        "three-way length key coverage differs",
    )
    groups = defaultdict(Counter)
    trajectories = Counter()
    for _key, items in by_key.items():
        p_ok = items["n155a_candidate"].get("free_exact") is True
        c_ok = items["surface_control"].get("free_exact") is True
        a_ok = items["surface_augmented"].get("free_exact") is True
        trajectories[(p_ok, c_ok, a_ok)] += 1
        row = items["surface_control"]
        group = (int(row["seed"]), str(row["kind"]), int(row["context_length"]))
        counts = groups[group]
        counts["items"] += 1
        counts["n155a_exact"] += p_ok
        counts["control_exact"] += c_ok
        counts["augmented_exact"] += a_ok
        counts["parent_to_control_regression"] += p_ok and not c_ok
        counts["parent_to_control_improvement"] += not p_ok and c_ok
        counts["control_to_augmented_regression"] += c_ok and not a_ok
        counts["control_to_augmented_improvement"] += not c_ok and a_ok
    summary = []
    for key, counts in sorted(groups.items()):
        seed, kind, length = key
        row = {"seed": seed, "kind": kind, "context_length": length}
        row.update({name: int(value) for name, value in sorted(counts.items())})
        summary.append(row)
    return summary


def analyze() -> dict:
    started = time.process_time()
    if REPORT.exists():
        raise FileExistsError(f"N1-162E report is write-once: {REPORT}")
    require(sha256(PLAN) == PLAN_SHA256, "PLAN-N1-162E plan SHA changed")
    require(sha256(N155A_RESULT) == N155A_RESULT_SHA256, "N1-155A result SHA changed")
    require(sha256(N155A_PROGRESS) == N155A_PROGRESS_SHA256, "N1-155A progress SHA changed")
    require(sha256(N162B_RESULT) == N162B_RESULT_SHA256, "N1-162B result SHA changed")
    require(sha256(N162B_PROGRESS) == N162B_PROGRESS_SHA256, "N1-162B progress SHA changed")
    require(sha256(N162B_CLOSEOUT) == N162B_CLOSEOUT_SHA256, "N1-162B closeout SHA changed")
    require(sha256(N162C_CLOSEOUT) == N162C_CLOSEOUT_SHA256, "N1-162C closeout SHA changed")
    require(sha256(N162C_REPORT) == N162C_REPORT_SHA256, "N1-162C report SHA changed")
    n155a_result = read_json(N155A_RESULT)
    n162b_result = read_json(N162B_RESULT)
    require(
        n155a_result.get("complete") is True
        and n155a_result.get("source_hashes_unchanged") is True,
        "N1-155A result is not complete",
    )
    require(
        n162b_result.get("complete") is True
        and n162b_result.get("source_hashes_unchanged") is True,
        "N1-162B result is not complete",
    )
    require(
        n150_builder.verify_hashes(n155a_result.get("source_hashes", {})),
        "N1-155A bound sources/checkpoints changed",
    )
    require(
        n150_builder.verify_hashes(n162b_result.get("source_hashes", {})),
        "N1-162B bound sources/checkpoints changed",
    )
    n155_counts, n155_gate, n155_length, n155_lines = _read_events(
        N155A_PROGRESS,
        expected_counts={
            "training_update": 55_296,
            "original_train_gate_row": 5_760,
            "length_score_row": 1_728,
        },
    )
    n162b_counts, n162b_gate_raw, n162b_length_raw, n162b_lines = _read_events(
        N162B_PROGRESS, expected_counts=n162c.EXPECTED_EVENTS
    )
    n155_gate = _n155a_gate_rows(n155_gate)
    n155_length = _n155a_length_rows(n155_length)
    n162b_gate, _overwritten = n162c.bind_endpoint_chunks(
        n162b_gate_raw, rows_per_endpoint=GATE_ROWS_PER_ENDPOINT, event="original_train_gate_row"
    )
    n162b_length, _mismatch = n162c.bind_endpoint_chunks(
        n162b_length_raw, rows_per_endpoint=LENGTH_ROWS_PER_ENDPOINT, event="length_score_row"
    )
    combined_gate = n155_gate + n162b_gate
    combined_length = n155_length + n162b_length
    gate_summary, trajectory_report, disagreement_examples = _three_way_gate(combined_gate)
    length_summary = _three_way_length(combined_length)
    require(
        sha256(N155A_PROGRESS) == N155A_PROGRESS_SHA256
        and sha256(N162B_PROGRESS) == N162B_PROGRESS_SHA256,
        "progress changed during three-way attribution",
    )
    require(
        sha256(N155A_RESULT) == N155A_RESULT_SHA256 and sha256(N162B_RESULT) == N162B_RESULT_SHA256,
        "result changed during three-way attribution",
    )
    cpu = time.process_time() - started
    complete = cpu <= CPU_LIMIT_SECONDS
    source_hashes: dict[str, str] = {}
    _merge_hashes(source_hashes, n155a_result.get("source_hashes", {}))
    _merge_hashes(source_hashes, n162b_result.get("source_hashes", {}))
    _merge_hashes(
        source_hashes,
        {
            str(N155A_RESULT.resolve()): N155A_RESULT_SHA256,
            str(N155A_PROGRESS.resolve()): N155A_PROGRESS_SHA256,
            str(N162B_RESULT.resolve()): N162B_RESULT_SHA256,
            str(N162B_PROGRESS.resolve()): N162B_PROGRESS_SHA256,
            str(N162B_CLOSEOUT.resolve()): N162B_CLOSEOUT_SHA256,
            str(N162C_REPORT.resolve()): N162C_REPORT_SHA256,
            str(N162D_REPORT.resolve()): N162D_REPORT_SHA256,
            str(PLAN.resolve()): PLAN_SHA256,
            str(Path(__file__).resolve()): sha256(Path(__file__)),
            str(TEST.resolve()): sha256(TEST),
        },
    )
    report = {
        "schema": "taiji-n1-plan162e-parent-control-surface-drift-v1",
        "status": "complete" if complete else "stopped_cpu_limit",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "n155a_result_sha256": N155A_RESULT_SHA256,
        "n155a_progress_sha256": N155A_PROGRESS_SHA256,
        "n162b_result_sha256": N162B_RESULT_SHA256,
        "n162b_progress_sha256": N162B_PROGRESS_SHA256,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": n150_builder.verify_hashes(source_hashes),
        "progress_line_counts": {"n155a": n155_lines, "n162b": n162b_lines},
        "event_counts": {"n155a": dict(n155_counts), "n162b": dict(n162b_counts)},
        "matched_gate_rows": len(combined_gate),
        "matched_length_rows": len(combined_length),
        "gate_rows_by_seed_endpoint": gate_summary,
        "gate_trajectories": trajectory_report,
        "gate_disagreement_examples": disagreement_examples[:240],
        "gate_disagreement_count": len(disagreement_examples),
        "length_rows_by_seed_kind_context_length": length_summary,
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
        "summary_zh": f"N1-155A父档→N1-162B control→augmented三方归因{'完成' if complete else '触及CPU停止线'}：gate配对{len(combined_gate)}行、length配对{len(combined_length)}行，三方门分解{len(gate_summary)}组；0模型加载/查询/更新，CPU{cpu:.3f}/{CPU_LIMIT_SECONDS:.0f}秒。",
    }
    write_new(REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-162E parent/control/surface drift attribution"
    )
    parser.add_argument("--analyze", action="store_true", required=True)
    args = parser.parse_args()
    del args
    result = analyze()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "cpu_seconds",
                    "matched_gate_rows",
                    "matched_length_rows",
                    "gate_disagreement_count",
                    "model_loads",
                    "model_queries",
                    "parameter_updates",
                    "can_promote",
                    "summary_zh",
                )
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
