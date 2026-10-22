"""Read-only aggregation of stored teacher-forced and free exactness in PLAN-N1-105."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/reference/PLAN-N1-108_v10_teacher_free_agreement_20261017.md"
PLAN105 = ROOT / "plans/reference/PLAN-N1-105_counterbalanced_input_feature_diagnostic_20261017.md"
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v10_factorial_diagnostic_20261017.json"
RESULT = ROOT / "reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_20261017.json"
REPORT = ROOT / "reports/taiji_n1_hard_neighbor_v10_teacher_free_agreement_20261017.json"
CONTRACT_TEST = ROOT / "tests/taiji_native/test_n1_run108_v10_teacher_free_agreement_contract.py"

PLAN_SHA256 = "e07641a038c27b91b229312dfe45685606d27c5e1623e5cbdbcee627d3f31960"
PLAN105_SHA256 = "af6eebc3e3fb01fc8c409f762176aefa8b91f246e9ea7325f42c5fa3267537ca"
FACE_SHA256 = "9b23a8b4ad1508ab0c2b100b9eb98b9c4dcf572b031bb0782b1267737173d5ca"
RESULT_SHA256 = "287cc5c4a38693d8e055f1290d801b1ec6494768386ff3ad1963e3059852a822"

SCHEMA = "n1-hard-neighbor-v10-teacher-free-agreement-v1"
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")
KINDS = ("lookup", "revision", "transition", "unknown")
PAIRED_CELLS = ("both_exact", "free_only_exact", "teacher_only_exact", "both_wrong")
CPU_LIMIT_SECONDS = 15.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _paired_cell(free_exact: bool, teacher_exact: bool) -> str:
    if free_exact and teacher_exact:
        return "both_exact"
    if free_exact:
        return "free_only_exact"
    if teacher_exact:
        return "teacher_only_exact"
    return "both_wrong"


def _validate_and_collect(face: dict, result: dict) -> list[dict]:
    if (
        face.get("schema") != "n1-hard-neighbor-v10-factorial-diagnostic-v1"
        or face.get("split") != "diagnostic"
        or face.get("items") != 248
    ):
        raise ValueError("input face identity differs from frozen v10 diagnostic")
    episodes = face.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 248:
        raise ValueError("frozen face does not contain 248 episodes")
    face_items = {item.get("id"): item for item in episodes}
    if len(face_items) != 248 or None in face_items:
        raise ValueError("face item identifiers are missing or duplicated")
    if Counter(item.get("kind") for item in episodes) != {
        "lookup": 96,
        "revision": 88,
        "transition": 44,
        "unknown": 20,
    }:
        raise ValueError("face kind counts differ from PLAN-N1-105")

    if (
        result.get("status") != "complete"
        or result.get("complete") is not True
        or result.get("face_sha256") != FACE_SHA256
        or result.get("plan_sha256") != PLAN105_SHA256
        or result.get("model_queries") != 2232
        or result.get("planned_model_queries") != 2232
        or result.get("parameter_updates") != 0
        or result.get("optimizer_steps") != 0
        or result.get("dev_queries") != 0
        or result.get("existing_confirmation_face_queries") != 0
        or result.get("gate_recomputed") is not False
        or result.get("can_promote") is not False
    ):
        raise ValueError("v10 result metadata differs from the frozen diagnostic contract")
    result_rows = result.get("rows")
    expected_keys = {
        (seed, arm, item_id) for seed in SEEDS for arm in ARMS for item_id in face_items
    }
    if not isinstance(result_rows, list) or len(result_rows) != len(expected_keys):
        raise ValueError("result does not have the complete 2,232-row endpoint × face design")

    seen = set()
    collected = []
    for row in result_rows:
        key = (row.get("seed"), row.get("arm"), row.get("id"))
        if key not in expected_keys or key in seen:
            raise ValueError(f"unexpected or duplicated result row {key}")
        seen.add(key)
        item = face_items[row["id"]]
        for field in (
            "kind",
            "template_index",
            "target_row_position_1based",
            "material_answer_length_chars",
        ):
            if row.get(field) != item.get(field):
                raise ValueError(f"result/face mismatch for {row['id']} field {field}")
        if row.get("output_hex") != row.get("raw_hex"):
            raise ValueError(f"raw/output mismatch for {row['id']}")
        try:
            output = bytes.fromhex(row["raw_hex"]).decode("ascii")
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid raw output for {row['id']}") from exc
        if row.get("output_hex") != row.get("raw_hex"):
            raise ValueError(f"raw/output mismatch for {row['id']}")
        free_exact = output == item.get("answer")
        if row.get("free_exact") is not free_exact:
            raise ValueError(f"stored free_exact disagrees with output for {row['id']}")
        if type(row.get("teacher_exact")) is not bool:
            raise ValueError(f"teacher_exact is not a boolean for {row['id']}")
        collected.append(
            {
                "seed": row["seed"],
                "arm": row["arm"],
                "id": row["id"],
                "kind": item["kind"],
                "target_row_position_1based": item.get("target_row_position_1based"),
                "material_answer_length_chars": item.get("material_answer_length_chars"),
                "free_exact": free_exact,
                "teacher_exact": row["teacher_exact"],
                "paired_cell": _paired_cell(free_exact, row["teacher_exact"]),
            }
        )
    if seen != expected_keys:
        raise ValueError("one or more endpoint/item result rows are missing")
    return collected


def _summary(rows: list[dict]) -> dict:
    counts = Counter(row["paired_cell"] for row in rows)
    return {
        "denominator": len(rows),
        **{cell: counts[cell] for cell in PAIRED_CELLS},
        "free_exact": sum(row["free_exact"] for row in rows),
        "teacher_exact": sum(row["teacher_exact"] for row in rows),
        "disagreements": counts["free_only_exact"] + counts["teacher_only_exact"],
    }


def _strata(rows: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[dimension] for dimension in dimensions)].append(row)
    output = []
    for key, group in sorted(
        grouped.items(), key=lambda pair: tuple(str(value) for value in pair[0])
    ):
        output.append({**dict(zip(dimensions, key, strict=False)), **_summary(group)})
    return output


def build_report(face: dict, result: dict, before: dict[str, str], cpu_seconds: float) -> dict:
    rows = _validate_and_collect(face, result)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "complete": True,
        "exploratory_diagnostic_only": True,
        "plan_sha256": PLAN_SHA256,
        "plan105_sha256": PLAN105_SHA256,
        "face_sha256": FACE_SHA256,
        "result_sha256": RESULT_SHA256,
        "source_hashes": {
            "analyzer": sha256_file(Path(__file__)),
            "contract_test": sha256_file(CONTRACT_TEST),
        },
        "input_hashes_before": before,
        "input_hashes_after": {
            "face": sha256_file(FACE),
            "result": sha256_file(RESULT),
            "plan108": sha256_file(PLAN),
            "plan105": sha256_file(PLAN105),
        },
        "inputs_unchanged": True,
        "rows_analyzed": len(rows),
        "overall": _summary(rows),
        "marginals": {
            "seed_arm": _strata(rows, ("seed", "arm")),
            "seed_arm_kind": _strata(rows, ("seed", "arm", "kind")),
            "seed_arm_kind_answer_length": _strata(
                rows, ("seed", "arm", "kind", "material_answer_length_chars")
            ),
            "seed_arm_kind_position": _strata(
                rows, ("seed", "arm", "kind", "target_row_position_1based")
            ),
        },
        "paired_rows": rows,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "cpu_seconds": cpu_seconds,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "gate_recomputed": False,
        "can_promote": False,
        "interpretation_limit": "aggregation of stored teacher-forced/free exactness on matched synthetic diagnostic items; no rerun, causal, generalization, or promotion claim",
    }


def run() -> dict:
    started = time.process_time()
    for path, expected in (
        (PLAN, PLAN_SHA256),
        (PLAN105, PLAN105_SHA256),
        (FACE, FACE_SHA256),
        (RESULT, RESULT_SHA256),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen source hash mismatch: {path.relative_to(ROOT)}")
    before = {
        "face": sha256_file(FACE),
        "result": sha256_file(RESULT),
        "plan108": sha256_file(PLAN),
        "plan105": sha256_file(PLAN105),
    }
    face = read_json(FACE)
    result = read_json(RESULT)
    report = build_report(face, result, before, time.process_time() - started)
    if before != report["input_hashes_after"]:
        raise ValueError("one or more read-only inputs changed during execution")
    report["inputs_unchanged"] = True
    report["cpu_seconds"] = time.process_time() - started
    if report["cpu_seconds"] > CPU_LIMIT_SECONDS:
        raise ValueError("PLAN-N1-108 CPU ceiling exceeded")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if REPORT.exists():
        raise FileExistsError(f"write-once report already exists: {REPORT}")
    report = run()
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(REPORT.relative_to(ROOT)),
                "rows_analyzed": report["rows_analyzed"],
                "overall": report["overall"],
                "cpu_seconds": report["cpu_seconds"],
                "model_loads": report["model_loads"],
                "model_queries": report["model_queries"],
                "parameter_updates": report["parameter_updates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
