"""PLAN-N1-131 offline semantic attribution for saved transition outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (
    eval_taiji_n1_v13_v14_native_question_binding_mass as n129,  # noqa: E402
)

PLAN = ROOT / "plans/reference/PLAN-N1-131_transition_operator_error_attribution_draft_20261019.md"
PLAN_SHA256 = "f239c2a4b052ac03c05060a1ee8060d4c50d6fbaf38efce8c0d87c3be1970cba"
N130_PLAN = (
    ROOT / "plans/reference/PLAN-N1-130_native_relation_row_binding_mass_audit_draft_20261019.md"
)
N130_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-130_execution_closeout_20261019.md"
N130_READINESS = (
    ROOT / "reports/taiji_n1_v13_v14_native_relation_row_binding_mass_readiness_20261019.json"
)
N130_ATTEMPT = (
    ROOT / "reports/taiji_n1_v13_v14_native_relation_row_binding_mass_attempt_20261019.json"
)
N130_RESULT = (
    ROOT / "reports/taiji_n1_v13_v14_native_relation_row_binding_mass_result_20261019.json"
)
N130_CLOSEOUT_SHA256 = "397fc005125707f9452ce23cf00cd9ce252f030dfeb41c3bf480ee4da870f813"
N130_READINESS_SHA256 = "24e256f5cb9d2c1443bb29921d418c9cd7167208a1ecf8912dd35f343e1d1979"
N130_ATTEMPT_SHA256 = "7fc0c25aa2e0119cda0579fd7d9583eccdd81a1733e162f2692c918b6bbc5fb7"
N130_RESULT_SHA256 = "edd426464ec01dd961d8d27be0b829f5db2b2d1c5d72223e9b6f2039dece94a1"
REPORT = ROOT / "reports/taiji_n1_transition_operator_error_attribution_20261019.json"
CPU_LIMIT_SECONDS = 30
SEEDS = n129.SEEDS
ARMS = n129.ARMS
CONDITIONS = (
    "native_full_context",
    "source_pruned",
    "decoder_source_mask",
    "question_binding_mask",
)
DIRECTIONS = ("north", "east", "south", "west")
TURNS = ("left", "right")
CATEGORIES = (
    "correct",
    "expected_text_non_boundary",
    "inverse_turn",
    "unchanged_base",
    "other_cardinal",
    "other_output",
)


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


def rotate(direction: str, turn: str) -> str:
    if direction not in DIRECTIONS or turn not in TURNS:
        raise ValueError(f"invalid cardinal transition {direction!r} / {turn!r}")
    offset = 1 if turn == "right" else -1
    return DIRECTIONS[(DIRECTIONS.index(direction) + offset) % len(DIRECTIONS)]


def _task_operator(item: dict) -> dict:
    subject = item["entities"][0]
    faces = [row for row in item["observations"] if row.startswith(subject + " faces ")]
    turns = [row for row in item["observations"] if row.startswith(subject + " turns ")]
    if len(faces) != 1 or len(turns) != 1:
        raise ValueError(
            f"transition item must contain one target faces and turns row: {item['id']}"
        )
    base = faces[0][len(subject + " faces ") :].rstrip(".")
    turn = turns[0][len(subject + " turns ") :].rstrip(".")
    answer = item["answer"]
    expected = rotate(base, turn)
    if answer != expected:
        raise ValueError(
            f"manifest transition operator mismatch for {item['id']}: {base} {turn} -> {answer}"
        )
    inverse = rotate(base, "left" if turn == "right" else "right")
    return {
        "base_direction": base,
        "turn": turn,
        "gold_direction": answer,
        "inverse_turn_direction": inverse,
    }


def _source_set() -> dict:
    for path, expected in (
        (PLAN, PLAN_SHA256),
        (N130_CLOSEOUT, N130_CLOSEOUT_SHA256),
        (N130_READINESS, N130_READINESS_SHA256),
        (N130_ATTEMPT, N130_ATTEMPT_SHA256),
        (N130_RESULT, N130_RESULT_SHA256),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen N1-130/N1-131 source changed: {path}")
    n130 = read_json(N130_RESULT)
    if (
        n130.get("complete") is not True
        or n130.get("model_binding_evaluations") != 180
        or n130.get("generated_answer_queries") != 0
        or n130.get("parameter_updates") != 0
        or n130.get("source_hashes_unchanged") is not True
    ):
        raise ValueError("complete N1-130 relation-row activation audit is required")
    bound = n129._source_set()
    source_hashes = dict(bound["source_hashes"])
    for path in (
        N130_PLAN,
        N130_CLOSEOUT,
        N130_READINESS,
        N130_ATTEMPT,
        N130_RESULT,
        PLAN,
        Path(__file__).resolve(),
    ):
        source_hashes[str(Path(path).resolve())] = sha256(Path(path))

    transition_items = {}
    for face_label, (_, _, items) in bound["face_data"].items():
        selected = [item for item in items if item["kind"] == "transition"]
        if len(selected) != 5:
            raise ValueError(f"exactly five transition confirm items required for {face_label}")
        for item in selected:
            transition_items[face_label, item["id"]] = {**_task_operator(item), "item": item}
    if len(transition_items) != 10:
        raise ValueError("the two frozen faces must provide ten transition items")
    return {**bound, "source_hashes": source_hashes, "transition_items": transition_items}


def _saved_output(row: dict) -> str:
    raw = row.get("raw_hex", row.get("output_hex"))
    if not isinstance(raw, str):
        raise ValueError(f"saved transition output is missing for {row.get('id')}")
    return bytes.fromhex(raw).decode("utf-8", errors="replace")


def _classify(row: dict, operator: dict) -> str:
    output = _saved_output(row)
    if output == operator["gold_direction"]:
        return "correct" if row.get("stop_reason") == "boundary" else "expected_text_non_boundary"
    if output == operator["inverse_turn_direction"]:
        return "inverse_turn"
    if output == operator["base_direction"]:
        return "unchanged_base"
    if output in DIRECTIONS:
        return "other_cardinal"
    return "other_output"


def run() -> dict:
    if REPORT.exists():
        raise FileExistsError("PLAN-N1-131 attribution report is write-once")
    started = time.process_time()
    bound = _source_set()
    rows = []
    for face_label, (_, _, items) in bound["face_data"].items():
        for item in items:
            if item["kind"] != "transition":
                continue
            operator = bound["transition_items"][face_label, item["id"]]
            for seed in SEEDS:
                for arm in ARMS:
                    key = (seed, arm, item["id"])
                    condition_rows = {
                        "native_full_context": bound["native_rows"][face_label][key],
                        "source_pruned": bound["pruned_rows"][face_label][key],
                        "decoder_source_mask": bound["question_rows"][
                            face_label, seed, arm, item["id"]
                        ],
                        "question_binding_mask": bound["prior_rows"][
                            face_label, seed, arm, item["id"]
                        ],
                    }
                    for condition in CONDITIONS:
                        row = condition_rows[condition]
                        category = _classify(row, operator)
                        output = _saved_output(row)
                        exact = bool(row.get("free_exact"))
                        semantically_equal = output == operator["gold_direction"]
                        if exact and not semantically_equal:
                            raise ValueError("saved exactness conflicts with transition semantics")
                        rows.append(
                            {
                                "face": face_label,
                                "item_id": item["id"],
                                "seed": seed,
                                "arm": arm,
                                "condition": condition,
                                **{key: value for key, value in operator.items() if key != "item"},
                                "output": output,
                                "stop_reason": row.get("stop_reason"),
                                "free_exact": exact,
                                "semantic_class": category,
                            }
                        )

    expected_rows = 2 * 5 * len(SEEDS) * len(ARMS) * len(CONDITIONS)
    if len(rows) != expected_rows or expected_rows != 240:
        raise ValueError(f"expected 240 saved transition rows, got {len(rows)}")
    summaries = {}
    for condition in CONDITIONS:
        summaries[condition] = {}
        for arm in ARMS:
            selected = [row for row in rows if row["condition"] == condition and row["arm"] == arm]
            summaries[condition][arm] = {
                "rows": len(selected),
                "semantic_classes": dict(Counter(row["semantic_class"] for row in selected)),
                "by_seed": {
                    str(seed): dict(
                        Counter(row["semantic_class"] for row in selected if row["seed"] == seed)
                    )
                    for seed in SEEDS
                },
            }
    hashes_unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in bound["source_hashes"].items()
    )
    cpu = time.process_time() - started
    complete = len(rows) == 240 and hashes_unchanged and cpu <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan131-saved-transition-operator-error-attribution-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "source_hashes": bound["source_hashes"],
        "source_hashes_unchanged": hashes_unchanged,
        "transition_items_per_face": 5,
        "faces": ["v13", "v14"],
        "conditions": list(CONDITIONS),
        "rows": rows,
        "summary": summaries,
        "inputs_read_offline": 10,
        "saved_output_rows_reclassified": len(rows),
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "generated_answer_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "unknown_queries": 0,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "descriptive_audit_is_capability": False,
        "can_promote": False,
    }
    write_new(REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    result = run()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "saved_output_rows_reclassified",
                    "model_loads",
                    "model_queries",
                    "cpu_seconds",
                    "can_promote",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
