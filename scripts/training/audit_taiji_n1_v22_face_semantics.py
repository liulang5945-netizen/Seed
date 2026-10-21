"""对已生成的v22 face做只读事实/答案/位置语义审计。"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v22_confirmation_after_n142 as builder

PLAN = builder.PLAN
FACE = builder.FACE
SOURCE_EXCLUSIONS = builder.SOURCE_EXCLUSIONS
FACE_CHECKS = builder.FACE_CHECKS
PREPARED = builder.PREPARED
REPORT = ROOT / "reports/taiji_n1_v22_face_semantic_audit_20261019.json"


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


def _transition_answer(face: dict, item: dict, rows: list[str]) -> str:
    if len(rows) != 2 or " faces " not in rows[0] or " turns " not in rows[1]:
        raise ValueError(f"transition target facts malformed: {item['id']}")
    start = rows[0].split(" faces ", 1)[1].removesuffix(".")
    turn = rows[1].split(" turns ", 1)[1].removesuffix(".")
    directions = builder.DIRECTIONS
    if start not in directions or turn not in ("left", "right"):
        raise ValueError(f"transition start/turn invalid: {item['id']}")
    delta = 1 if turn == "right" else -1
    result = directions[(directions.index(start) + delta) % len(directions)]
    if result != item.get("answer") or result != item.get("answer_direction"):
        raise ValueError(f"transition answer does not follow facts: {item['id']}")
    if turn != item.get("turn_template"):
        raise ValueError(f"transition turn template disagrees with target facts: {item['id']}")
    return result


def audit_face(face: dict, exclusions: dict) -> dict:
    excluded_names = set(exclusions["excluded_entities"])
    excluded_words = set(exclusions["excluded_word_values"])
    builder.validate_face(face, excluded_names, excluded_words)
    item_ids = set()
    material_answer_lengths = Counter()
    target_starts = Counter()
    direction_counts = Counter()
    turn_counts = Counter()
    target_answer_values = set()
    stale_values = set()
    filler_material_values = set()
    unknown_answer_counts = Counter()
    target_row_count = Counter()
    for item in face["episodes"]:
        item_id = item["id"]
        item_ids.add(item_id)
        kind = item["kind"]
        subset = item["subset"]
        subject = item["entities"][0]
        rows = item["observations"]
        target_rows = [
            (index, row)
            for index, row in enumerate(rows)
            if row.startswith(subject + " ")
        ]
        target_indices = [index for index, _row in target_rows]
        if target_indices != item.get("target_fact_indices"):
            raise ValueError(f"target fact index ledger mismatch: {item_id}")
        target_row_count[kind] += len(target_rows)
        if kind == "lookup":
            if len(target_rows) != 1 or " uses " not in target_rows[0][1]:
                raise ValueError(f"lookup target fact missing: {item_id}")
            value = target_rows[0][1].split(" uses ", 1)[1].removesuffix(".")
            if value != item["answer"]:
                raise ValueError(f"lookup answer differs from target fact: {item_id}")
            material_answer_lengths[(subset, len(value))] += 1
            target_answer_values.add(value)
            target_starts[(subset, kind, target_rows[0][0])] += 1
        elif kind == "revision":
            if len(target_rows) != 2 or " uses " not in target_rows[0][1] or " now uses " not in target_rows[1][1]:
                raise ValueError(f"revision target facts missing or misordered: {item_id}")
            stale = target_rows[0][1].split(" uses ", 1)[1].removesuffix(".")
            current = target_rows[1][1].split(" now uses ", 1)[1].removesuffix(".")
            if current != item["answer"] or stale == current:
                raise ValueError(f"revision answer/stale fact mismatch: {item_id}")
            material_answer_lengths[(subset, len(current))] += 1
            material_answer_lengths[(subset, len(stale))] += 1
            target_answer_values.add(current)
            stale_values.add(stale)
            target_starts[(subset, kind, target_rows[0][0])] += 1
        elif kind == "transition":
            if _transition_answer(face, item, [row for _index, row in target_rows]) != item["answer"]:
                raise ValueError(f"transition target semantics mismatch: {item_id}")
            direction_counts[(subset, item["answer_direction"])] += 1
            turn_counts[(subset, item["turn_template"])] += 1
            target_starts[(subset, kind, target_rows[0][0])] += 1
        elif kind == "unknown":
            if target_rows or item.get("answer") != builder.ABSTENTION:
                raise ValueError(f"unknown item contains a target fact or wrong abstention: {item_id}")
            unknown_answer_counts[(subset, item["answer"])] += 1
        else:
            raise ValueError(f"unexpected item kind: {kind}")

        for row in rows:
            if " uses " in row:
                value = row.split(" uses ", 1)[1].removesuffix(".")
                if len(value) not in (6, 7):
                    raise ValueError(f"material word length differs: {item_id}")
                if not row.startswith(subject + " "):
                    filler_material_values.add(value)
        if kind != "transition":
            for value in (
                row.split(" uses ", 1)[1].removesuffix(".")
                for _index, row in target_rows
                if " uses " in row
            ):
                if len(value) not in (6, 7):
                    raise ValueError(f"target material length differs: {item_id}")

    if len(item_ids) != 160:
        raise ValueError("v22 item identifiers must be unique")
    if any(count != 20 for count in unknown_answer_counts.values()) or set(unknown_answer_counts) != {
        (subset, "unknown") for subset in builder.SUBSETS
    }:
        raise ValueError("each subset must retain twenty literal unknown answers")
    if len(target_answer_values) != 80 or len(stale_values) != 40 or len(filler_material_values) != 1_800:
        raise ValueError("v22 material target/filler inventory does not match the frozen design")
    if len(target_answer_values | stale_values | filler_material_values) != 1_920:
        raise ValueError("v22 material strings must be globally unique")
    if any(len(value) not in (6, 7) for value in target_answer_values | stale_values | filler_material_values):
        raise ValueError("v22 material vocabulary has an invalid length")
    if any(direction_counts[(subset, direction)] != 5 for subset in builder.SUBSETS for direction in builder.DIRECTIONS):
        raise ValueError("v22 transition answer directions are not balanced")
    if any(turn_counts[(subset, turn)] != 10 for subset in builder.SUBSETS for turn in ("left", "right")):
        raise ValueError("v22 transition turn templates are not balanced")
    return {
        "items": len(item_ids),
        "unknown_literal_answer_counts": {
            f"{subset}/{answer}": count
            for (subset, answer), count in sorted(unknown_answer_counts.items())
        },
        "target_row_count_by_kind": dict(target_row_count),
        "material_target_answer_count": len(target_answer_values),
        "material_revision_stale_value_count": len(stale_values),
        "material_filler_value_count": len(filler_material_values),
        "material_value_count": len(target_answer_values | stale_values | filler_material_values),
        "material_target_answer_length_counts": {
            f"{subset}/{length}": count
            for (subset, length), count in sorted(material_answer_lengths.items())
        },
        "transition_direction_counts": {
            f"{subset}/{direction}": direction_counts[(subset, direction)]
            for subset in builder.SUBSETS
            for direction in builder.DIRECTIONS
        },
        "transition_turn_counts": {
            f"{subset}/{turn}": turn_counts[(subset, turn)]
            for subset in builder.SUBSETS
            for turn in ("left", "right")
        },
        "target_start_counts": {
            f"{subset}/{kind}/{position}": count
            for (subset, kind, position), count in sorted(target_starts.items())
        },
    }


def audit() -> dict:
    if REPORT.exists():
        raise FileExistsError("v22 face semantic audit is write-once")
    started = time.process_time()
    if sha256(PLAN) != builder.PLAN_SHA256 or not FACE.is_file() or not SOURCE_EXCLUSIONS.is_file():
        raise FileNotFoundError("frozen N1-143 plan, face and source exclusion manifest are required")
    face = read_json(FACE)
    exclusions = read_json(SOURCE_EXCLUSIONS)
    semantic = audit_face(face, exclusions)
    source_paths = (PLAN, FACE, SOURCE_EXCLUSIONS, FACE_CHECKS, PREPARED, Path(__file__))
    source_hashes = {str(path.resolve()): sha256(path) for path in source_paths}
    unchanged = all(Path(path).is_file() and sha256(Path(path)) == digest for path, digest in source_hashes.items())
    if not unchanged:
        raise ValueError("v22 face semantic audit source changed")
    report = {
        "schema": "n1-plan143-v22-face-semantic-audit-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": sha256(PLAN),
        "face_sha256": sha256(FACE),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "prepared_readiness_sha256": sha256(PREPARED),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": unchanged,
        "semantic_checks": semantic,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "v3_holdout_queries": 0,
        "cpu_seconds": round(time.process_time() - started, 6),
        "can_promote": False,
        "summary_zh": "逐题语义审计通过：lookup/revision目标材料与事实匹配，transition方向可由两条目标事实复算，unknown无目标事实且固定弃权；该报告仅为零查询静态核验。",
    }
    write_new(REPORT, report)
    return report


def main() -> int:
    result = audit()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "complete",
                    "cpu_seconds",
                    "semantic_checks",
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
