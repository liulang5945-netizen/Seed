"""PLAN-N1-162D：对封存transition gate输出做方向类型离线归因。"""

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
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132

PLAN = ROOT / "plans/reference/PLAN-N1-162D_transition_output_type_attribution_20261020.md"
PLAN_SHA256 = "7d5b74366df22d5682b6a398511ff7a35a55ca1adbc0ae2371f50c66285336a6"
N162B_RESULT = n162c.N162B_RESULT
N162B_RESULT_SHA256 = n162c.N162B_RESULT_SHA256
PROGRESS = n162c.PROGRESS
PROGRESS_SHA256 = n162c.PROGRESS_SHA256
N162C_PLAN = n162c.PLAN
N162C_PLAN_SHA256 = n162c.PLAN_SHA256
N162C_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-162C_execution_closeout_20261020.md"
N162C_CLOSEOUT_SHA256 = "d61d9c02f79693fcee7eb67f17f84915af86296aa4bcfc7b484c7977ff55374f"
N162C_REPORT = n162c.REPORT
N162C_REPORT_SHA256 = "f98fa027a55e83a03eb18a1288b61168f619b6ace0e31662d596ce82bf584e63"
SCHEDULE = n162c.SCHEDULE
REPORT = ROOT / "reports/taiji_n1_plan162d_transition_output_type_attribution_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan162d_transition_output_type_attribution_contract.py"

SEEDS = n162c.SEEDS
ARMS = n162c.ARMS
ENDPOINTS = n162c.ENDPOINTS
KINDS = n162c.KINDS
GATE_ROWS_PER_ENDPOINT = n162c.GATE_ROWS_PER_ENDPOINT
TRANSITION_ROWS_PER_ENDPOINT = 480
EXPECTED_TRANSITION_ROWS = TRANSITION_ROWS_PER_ENDPOINT * len(ENDPOINTS)
DIRECTIONS = ("north", "east", "south", "west")
TURNS = ("left", "right")
CONDITIONS = ("original", "direction_opposite", "turn_flip", "both")
CPU_LIMIT_SECONDS = 120.0


class Refuse(RuntimeError):
    """来源、方向派生、progress分母或输出复算不一致。"""


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


def _turned(initial: str, turn: str) -> str:
    require(initial in DIRECTIONS and turn in TURNS, f"invalid direction/turn pair: {initial}/{turn}")
    offset = 1 if turn == "right" else -1
    return DIRECTIONS[(DIRECTIONS.index(initial) + offset) % len(DIRECTIONS)]


def _opposite_turn_answer(initial: str, turn: str) -> str:
    return _turned(initial, "left" if turn == "right" else "right")


def _input_transition(item: dict) -> tuple[str, str, str]:
    observations = item.get("observations", [])
    require(len(observations) == 2, f"transition task must have two observations: {item.get('id')}")
    first = observations[0].removesuffix(".").split()
    second = observations[1].removesuffix(".").split()
    require(len(first) == 3 and first[1] == "faces", f"invalid face observation: {observations[0]}")
    require(len(second) == 3 and second[1] == "turns", f"invalid turn observation: {observations[1]}")
    initial, turn = first[2], second[2]
    answer = _turned(initial, turn)
    require(item.get("answer") == answer, f"v3 transition target formula mismatch: {item.get('id')}")
    return initial, turn, answer


def _classify_output(raw_hex: str, stop_reason: str, expected: str, initial: str, turn: str) -> str:
    raw = bytes.fromhex(raw_hex)
    target = expected.encode("utf-8")
    if raw == target:
        return "correct_exact" if stop_reason == "boundary" else "target_bytes_bad_stop"
    if raw == b"unknown":
        return "literal_unknown"
    if not raw:
        return "empty"
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "non_utf8_bytes"
    if value in DIRECTIONS:
        if value == initial:
            return "initial_direction_copy"
        if value == _opposite_turn_answer(initial, turn):
            return "inverse_turn_answer"
        opposite_initial = DIRECTIONS[(DIRECTIONS.index(initial) + 2) % len(DIRECTIONS)]
        if value == opposite_initial:
            return "opposite_initial_direction"
        return "other_wrong_direction"
    return "other_bytes"


def _progress_transition_rows() -> tuple[list[dict], dict[str, int], int]:
    counts = Counter()
    transitions = []
    malformed = []
    total = 0
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
            if event == "original_train_gate_row" and row.get("kind") == "transition":
                transitions.append(row)
    require(not malformed, f"malformed progress rows: {malformed[:3]}")
    require(dict(counts) == n162c.EXPECTED_EVENTS and total == sum(n162c.EXPECTED_EVENTS.values()), f"N1-162B progress counts differ: {dict(counts)}")
    require(len(transitions) == EXPECTED_TRANSITION_ROWS, "N1-162B transition row count differs")
    return transitions, dict(sorted(counts.items())), total


def _endpoint_bind(rows: list[dict]) -> list[dict]:
    expected_views = {
        "original": 48,
        "rename": 48,
        "factorial_original": 192,
        "factorial_rename": 192,
    }
    if len(rows) != EXPECTED_TRANSITION_ROWS:
        raise Refuse("N1-162B transition progress endpoint denominator differs")
    bound = []
    cursor = 0
    for seed, arm in ENDPOINTS:
        chunk = rows[cursor : cursor + TRANSITION_ROWS_PER_ENDPOINT]
        cursor += TRANSITION_ROWS_PER_ENDPOINT
        require(len(chunk) == TRANSITION_ROWS_PER_ENDPOINT, f"transition endpoint chunk is truncated: {seed}/{arm}")
        require(all(int(row.get("seed", -1)) == seed and row.get("arm") == "position" for row in chunk), f"transition endpoint source order differs: {seed}/{arm}")
        view_counts = Counter(str(row.get("evaluation_set")) for row in chunk)
        require(dict(view_counts) == expected_views, f"transition view denominators differ: {seed}/{arm}/{dict(view_counts)}")
        for row in chunk:
            value = dict(row)
            value["experiment_arm"] = arm
            bound.append(value)
    return bound


def _expected_lookup(inputs: dict, factorial) -> dict[tuple[int, str, str, str], tuple[str, str, str]]:
    expected = {}
    for view in ("original", "rename"):
        items = [item for item in inputs["views"][view] if item.get("kind") == "transition"]
        require(len(items) == 48, f"training input transition count differs in {view}")
        for item in items:
            initial, turn, answer = _input_transition(item)
            expected[(view, str(item["id"]), "", "")] = (initial, turn, answer)
            for variant in factorial.variants(item):
                condition = str(variant["condition"])
                initial_variant = str(variant["initial_direction"])
                turn_variant = str(variant["turn"])
                variant_answer = str(variant["item"]["answer"])
                formula_answer = _turned(initial_variant, turn_variant)
                require(variant_answer == formula_answer, f"factorial source formula differs: {item['id']}/{condition}")
                key = (f"factorial_{view}", str(item["id"]), condition, f"{item['id']}:{condition}")
                expected[key] = (initial_variant, turn_variant, variant_answer)
    return expected


def _paired_type_summary(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    keyed = {}
    group_counts = defaultdict(Counter)
    for row in rows:
        key = (int(row["seed"]), str(row["evaluation_set"]), str(row["id"]), row.get("condition"), row.get("case_id"))
        cell = keyed.setdefault(key, {})
        arm = str(row["experiment_arm"])
        require(arm not in cell, f"duplicate transition output row: {key}/{arm}")
        cell[arm] = row
    require(all(set(cell) == set(ARMS) for cell in keyed.values()), "transition output pair coverage differs")
    transitions = Counter()
    for key, cell in keyed.items():
        control = cell["surface_control"]["output_type"]
        augmented = cell["surface_augmented"]["output_type"]
        transitions[(key[1], control, augmented)] += 1
        group = (key[0], key[1], str(cell["surface_control"]["kind"]), cell["surface_control"].get("condition"))
        counts = group_counts[group]
        counts["items"] += 1
        counts["control_exact"] += control == "correct_exact"
        counts["augmented_exact"] += augmented == "correct_exact"
        counts["output_type_changed"] += control != augmented
    paired = []
    for (view, control_type, augmented_type), count in sorted(transitions.items()):
        paired.append({"evaluation_set": view, "control_output_type": control_type, "augmented_output_type": augmented_type, "items": count})
    grouped = []
    for key, counts in sorted(group_counts.items(), key=lambda pair: tuple("" if value is None else str(value) for value in pair[0])):
        seed, view, kind, condition = key
        row = {"seed": seed, "evaluation_set": view, "kind": kind, "condition": condition}
        row.update({name: int(value) for name, value in sorted(counts.items())})
        grouped.append(row)
    return paired, grouped


def analyze() -> dict:
    started = time.process_time()
    if REPORT.exists():
        raise FileExistsError(f"PLAN-N1-162D report is write-once: {REPORT}")
    require(sha256(PLAN) == PLAN_SHA256, "PLAN-N1-162D plan SHA changed")
    require(sha256(N162B_RESULT) == N162B_RESULT_SHA256, "N1-162B result SHA changed")
    require(sha256(PROGRESS) == PROGRESS_SHA256, "N1-162B progress SHA changed")
    require(sha256(N162C_PLAN) == N162C_PLAN_SHA256, "PLAN-N1-162C plan SHA changed")
    require(sha256(N162C_CLOSEOUT) == N162C_CLOSEOUT_SHA256, "PLAN-N1-162C closeout SHA changed")
    require(sha256(N162C_REPORT) == N162C_REPORT_SHA256, "N1-162C attribution report SHA changed")
    result = read_json(N162B_RESULT)
    analysis = read_json(N162C_REPORT)
    require(
        result.get("complete") is True
        and result.get("training_updates") == 55_296
        and result.get("train_score_queries") == 7_488
        and result.get("fresh_queries") == 0
        and result.get("can_promote") is False
        and analysis.get("complete") is True
        and analysis.get("gate_recalculation_matches_all_six") is True,
        "N1-162B/C sealed training evidence is invalid",
    )
    source_hashes = dict(result.get("source_hashes", {}))
    require(n150_builder.verify_hashes(source_hashes), "N1-162B bound sources/checkpoints changed")
    transitions, event_counts, progress_lines = _progress_transition_rows()
    bound = _endpoint_bind(transitions)
    inputs_bound = n132._build_context(load_inputs=True)
    inputs = inputs_bound["inputs"]
    require(inputs is not None and len(inputs.get("items", [])) == 192, "N1-132 frozen training inputs are required")
    factorial = inputs_bound["g"]["calibration"].engineering.factorial
    expected = _expected_lookup(inputs, factorial)
    expected_transition_rows = 0
    category_counts = Counter()
    output_rows = []
    exact_reproduced = 0
    for row in bound:
        view = str(row["evaluation_set"])
        item_id = str(row["id"])
        condition = str(row.get("condition") or "")
        case_id = str(row.get("case_id") or "")
        key = (view, item_id, condition, case_id)
        if view in ("original", "rename"):
            key = (view, item_id, "", "")
        if key not in expected:
            raise Refuse(f"transition expected-answer binding missing: {key}")
        initial, turn, answer = expected[key]
        if view.startswith("factorial_"):
            require(row.get("initial_direction") == initial and row.get("turn") == turn, f"factorial condition metadata differs: {key}")
        output_type = _classify_output(str(row["raw_hex"]), str(row["stop_reason"]), answer, initial, turn)
        recomputed = output_type == "correct_exact"
        require(recomputed is bool(row.get("free_exact")), f"transition free_exact recomputation differs: {key}")
        exact_reproduced += 1
        expected_transition_rows += 1
        category_counts[(int(row["seed"]), str(row["experiment_arm"]), view, str(row.get("condition") or ""), output_type)] += 1
        output_rows.append({
            "seed": int(row["seed"]),
            "arm": str(row["experiment_arm"]),
            "experiment_arm": str(row["experiment_arm"]),
            "evaluation_set": view,
            "kind": "transition",
            "id": item_id,
            "parent_id": item_id,
            "condition": row.get("condition"),
            "case_id": row.get("case_id"),
            "initial_direction": initial,
            "turn": turn,
            "expected_answer": answer,
            "raw_hex": row["raw_hex"],
            "output_type": output_type,
            "free_exact_stored": row["free_exact"],
            "stop_reason": row["stop_reason"],
        })
    require(expected_transition_rows == EXPECTED_TRANSITION_ROWS == 2_880 and exact_reproduced == 2_880, "transition row/answer-reconstruction denominator differs")
    template_exposure = analysis.get("augmented_template_exposure_by_parent", {})
    for row in output_rows:
        row["augmented_template_exposure_by_length"] = template_exposure.get(row["parent_id"], {})
    paired_output_types, paired_gate_groups = _paired_type_summary(output_rows)
    counts_report = []
    for key, count in sorted(category_counts.items(), key=lambda item: tuple(str(v) for v in item[0])):
        seed, arm, view, condition, output_type = key
        counts_report.append({"seed": seed, "arm": arm, "evaluation_set": view, "condition": condition or None, "output_type": output_type, "items": int(count)})
    require(sha256(N162B_RESULT) == N162B_RESULT_SHA256 and sha256(PROGRESS) == PROGRESS_SHA256 and n150_builder.verify_hashes(source_hashes), "N1-162B source evidence changed during attribution")
    cpu = time.process_time() - started
    complete = cpu <= CPU_LIMIT_SECONDS
    report_sources = {
        **source_hashes,
        str(PLAN.resolve()): sha256(PLAN),
        str(N162B_RESULT.resolve()): N162B_RESULT_SHA256,
        str(PROGRESS.resolve()): PROGRESS_SHA256,
        str(N162C_PLAN.resolve()): N162C_PLAN_SHA256,
        str(N162C_CLOSEOUT.resolve()): sha256(N162C_CLOSEOUT),
        str(N162C_REPORT.resolve()): N162C_REPORT_SHA256,
        str(Path(__file__).resolve()): sha256(Path(__file__)),
        str(TEST.resolve()): sha256(TEST),
    }
    report = {
        "schema": "taiji-n1-plan162d-transition-output-type-attribution-v1",
        "status": "complete" if complete else "stopped_cpu_limit",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "n162b_result_sha256": N162B_RESULT_SHA256,
        "progress_sha256": PROGRESS_SHA256,
        "n162c_report_sha256": N162C_REPORT_SHA256,
        "source_hashes": report_sources,
        "source_hashes_unchanged": n150_builder.verify_hashes(report_sources),
        "progress_line_count": progress_lines,
        "event_counts": event_counts,
        "transition_rows": expected_transition_rows,
        "answer_reconstruction_rows": exact_reproduced,
        "output_type_counts": counts_report,
        "paired_output_type_transitions": paired_output_types,
        "paired_gate_group_summary": paired_gate_groups,
        "transition_rows_by_endpoint": output_rows,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "training_input_views_read": 384,
        "factorial_expected_variants_recomputed": 384,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "inference_calls": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "summary_zh": f"transition输出类型归因{'完成' if complete else '触及CPU停止线'}：2,880行答案重建及free_exact复算一致；0模型/ checkpoint加载、0查询、0更新，CPU{cpu:.3f}/{CPU_LIMIT_SECONDS:.0f}秒。",
    }
    write_new(REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-162D transition output type attribution")
    parser.add_argument("--analyze", action="store_true", required=True)
    args = parser.parse_args()
    del args
    report = analyze()
    print(json.dumps({key: report.get(key) for key in ("status", "complete", "cpu_seconds", "transition_rows", "answer_reconstruction_rows", "model_loads", "model_queries", "parameter_updates", "can_promote", "summary_zh")}, ensure_ascii=True, sort_keys=True))
    return 0 if report.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
