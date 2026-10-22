"""Read-only lexical attribution for the consumed PLAN-N1-105 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/reference/PLAN-N1-106_v10_output_error_attribution_20261017.md"
PLAN105 = ROOT / "plans/reference/PLAN-N1-105_counterbalanced_input_feature_diagnostic_20261017.md"
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v10_factorial_diagnostic_20261017.json"
RESULT = ROOT / "reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_20261017.json"
REPORT = ROOT / "reports/taiji_n1_hard_neighbor_v10_output_error_attribution_20261017.json"
CONTRACT_TEST = ROOT / "tests/taiji_native/test_n1_run106_v10_output_error_attribution_contract.py"

PLAN_SHA256 = "556e73501143cea141138ced44e3e2ff52c2521150a9461a0a362d32331880fc"
PLAN105_SHA256 = "af6eebc3e3fb01fc8c409f762176aefa8b91f246e9ea7325f42c5fa3267537ca"
FACE_SHA256 = "9b23a8b4ad1508ab0c2b100b9eb98b9c4dcf572b031bb0782b1267737173d5ca"
RESULT_SHA256 = "287cc5c4a38693d8e055f1290d801b1ec6494768386ff3ad1963e3059852a822"

SCHEMA = "n1-hard-neighbor-v10-output-error-attribution-v1"
FACE_SCHEMA = "n1-hard-neighbor-v10-factorial-diagnostic-v1"
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")
KINDS = ("lookup", "revision", "transition", "unknown")
CATEGORIES = (
    "exact_answer",
    "revision_stale_material",
    "other_target_observed_material",
    "distractor_observed_material",
    "unknown_abstention",
    "empty_output",
    "other_unobserved_output",
    "nontext_output",
)
CPU_LIMIT_SECONDS = 30.0

_MATERIAL_RE = re.compile(r"^(?P<entity>[a-z]+) (?:(?:now )?uses )(?P<value>[a-z]+)\.$")
_QUESTION_RE = re.compile(r"^material of (?P<entity>[a-z]+)\?$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_material_observations(item: dict) -> dict[str, list[str]]:
    by_entity: dict[str, list[str]] = defaultdict(list)
    for observation in item.get("observations", []):
        match = _MATERIAL_RE.fullmatch(observation)
        if match:
            by_entity[match.group("entity")].append(match.group("value"))
    return dict(by_entity)


def target_entity_for(item: dict) -> str | None:
    match = _QUESTION_RE.fullmatch(item.get("question", ""))
    return match.group("entity") if match else None


def classify_output(item: dict, output: str | None) -> str:
    if output is None:
        return "nontext_output"
    answer = item.get("answer")
    if output == answer:
        return "exact_answer"
    if output == "":
        return "empty_output"
    if output == "unknown":
        return "unknown_abstention"

    by_entity = parse_material_observations(item)
    target_entity = target_entity_for(item)
    target_values = by_entity.get(target_entity, []) if target_entity else []
    if item.get("kind") == "revision" and output in target_values[:-1]:
        return "revision_stale_material"
    if target_entity and output in target_values:
        return "other_target_observed_material"

    observed_elsewhere = {
        value for entity, values in by_entity.items() if entity != target_entity for value in values
    }
    if output in observed_elsewhere or (
        target_entity is None and any(output in v for v in by_entity.values())
    ):
        return "distractor_observed_material"
    return "other_unobserved_output"


def decode_output(raw_hex: str) -> str | None:
    try:
        return bytes.fromhex(raw_hex).decode("ascii")
    except (ValueError, UnicodeDecodeError, TypeError):
        return None


def _validate_source_metadata(face: dict, result: dict) -> None:
    if face.get("schema") != FACE_SCHEMA or face.get("split") != "diagnostic":
        raise ValueError("input face is not the frozen v10 diagnostic face")
    items = face.get("episodes")
    if not isinstance(items, list) or len(items) != 248 or face.get("items") != 248:
        raise ValueError("the frozen 248-item face is incomplete")
    if result.get("status") != "complete" or result.get("complete") is not True:
        raise ValueError("the v10 diagnostic result is not complete")
    if (
        result.get("face_sha256") != FACE_SHA256
        or result.get("plan_sha256") != PLAN105_SHA256
        or result.get("model_queries") != 2232
        or result.get("planned_model_queries") != 2232
        or result.get("parameter_updates") != 0
        or result.get("optimizer_steps") != 0
        or result.get("dev_queries") != 0
        or result.get("existing_confirmation_face_queries") != 0
        or result.get("gate_recomputed") is not False
        or result.get("can_promote") is not False
        or result.get("checkpoints_unchanged") is not True
        or result.get("sources_unchanged") is not True
        or result.get("code_identity_unchanged") is not True
        or result.get("v7_v8_confirmation_payload_fields_decoded") is not False
    ):
        raise ValueError("v10 result metadata differs from its bounded diagnostic contract")


def _validate_and_classify(face: dict, result: dict) -> list[dict]:
    _validate_source_metadata(face, result)
    face_items = {item["id"]: item for item in face["episodes"]}
    if len(face_items) != 248:
        raise ValueError("face item identifiers are not unique")

    expected_keys = {
        (seed, arm, item_id) for seed in SEEDS for arm in ARMS for item_id in face_items
    }
    result_rows = result.get("rows")
    if not isinstance(result_rows, list) or len(result_rows) != len(expected_keys):
        raise ValueError("result row count does not match the complete endpoint × face design")

    seen: set[tuple[int, str, str]] = set()
    classified: list[dict] = []
    for row in result_rows:
        key = (row.get("seed"), row.get("arm"), row.get("id"))
        if key not in expected_keys or key in seen:
            raise ValueError(f"unexpected or duplicated result key: {key}")
        seen.add(key)
        item = face_items[row["id"]]
        if row.get("split") != "diagnostic" or row.get("kind") != item.get("kind"):
            raise ValueError(f"result/face identity mismatch for {row['id']}")
        if (
            row.get("template_index") != item.get("template_index")
            or row.get("target_row_position_1based") != item.get("target_row_position_1based")
            or row.get("material_answer_length_chars") != item.get("material_answer_length_chars")
        ):
            raise ValueError(f"result/face factor mismatch for {row['id']}")
        if row.get("output_hex") != row.get("raw_hex"):
            raise ValueError(f"output_hex/raw_hex mismatch for {row['id']}")
        output = decode_output(row.get("raw_hex"))
        free_exact = output == item.get("answer")
        if row.get("free_exact") is not free_exact:
            raise ValueError(f"free_exact does not match decoded output for {row['id']}")
        classified.append(
            {
                "seed": row["seed"],
                "arm": row["arm"],
                "id": row["id"],
                "kind": item["kind"],
                "template_index": item.get("template_index"),
                "target_row_position_1based": item.get("target_row_position_1based"),
                "material_answer_length_chars": item.get("material_answer_length_chars"),
                "free_exact": free_exact,
                "output_hex": row.get("raw_hex"),
                "output_class": classify_output(item, output),
            }
        )
    if seen != expected_keys:
        raise ValueError(f"missing result keys: {sorted(expected_keys - seen)[:3]}")
    return sorted(classified, key=lambda row: (row["seed"], ARMS.index(row["arm"]), row["id"]))


def _summary(rows: list[dict]) -> dict:
    counts = Counter(row["output_class"] for row in rows)
    return {
        "denominator": len(rows),
        "exact": counts["exact_answer"],
        "errors": len(rows) - counts["exact_answer"],
        "output_classes": {category: counts[category] for category in CATEGORIES},
    }


def _strata(rows: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[dimension] for dimension in dimensions)].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        output.append({**dict(zip(dimensions, key, strict=False)), **_summary(group)})
    return output


def _paired(rows: list[dict], comparison_arm: str) -> list[dict]:
    indexed = {(row["seed"], row["arm"], row["id"]): row for row in rows}
    episodes = {(row["seed"], row["id"]): row for row in rows if row["arm"] == "position"}
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for (seed, item_id), position in episodes.items():
        comparison = indexed[(seed, comparison_arm, item_id)]
        key = (seed, position["kind"], position["material_answer_length_chars"])
        grouped[key].append(
            {
                "position_exact": position["free_exact"],
                "comparison_exact": comparison["free_exact"],
                "position_class": position["output_class"],
            }
        )
    output = []
    for (seed, kind, answer_length), group in sorted(
        grouped.items(), key=lambda pair: (pair[0][0], pair[0][1], str(pair[0][2]))
    ):
        both = sum(row["position_exact"] and row["comparison_exact"] for row in group)
        position_only = sum(row["position_exact"] and not row["comparison_exact"] for row in group)
        comparison_only_rows = [
            row for row in group if not row["position_exact"] and row["comparison_exact"]
        ]
        both_wrong = sum(not row["position_exact"] and not row["comparison_exact"] for row in group)
        output.append(
            {
                "seed": seed,
                "kind": kind,
                "material_answer_length_chars": answer_length,
                "comparison_arm": comparison_arm,
                "denominator": len(group),
                "both_exact": both,
                "position_only_exact": position_only,
                "comparison_only_exact": len(comparison_only_rows),
                "both_wrong": both_wrong,
                "position_error_classes_when_comparison_exact": {
                    category: sum(row["position_class"] == category for row in comparison_only_rows)
                    for category in CATEGORIES
                },
            }
        )
    return output


def build_report(
    face: dict, result: dict, input_hashes: dict[str, str], cpu_seconds: float
) -> dict:
    rows = _validate_and_classify(face, result)
    dims = ("seed", "arm", "kind", "material_answer_length_chars", "target_row_position_1based")
    return {
        "schema": SCHEMA,
        "status": "complete",
        "complete": True,
        "exploratory_diagnostic_only": True,
        "plan_sha256": PLAN_SHA256,
        "plan105_sha256": PLAN105_SHA256,
        "source_hashes": {
            "analyzer": sha256_file(Path(__file__)),
            "contract_test": sha256_file(CONTRACT_TEST),
        },
        "face_sha256": FACE_SHA256,
        "result_sha256": RESULT_SHA256,
        "input_hashes_before": input_hashes,
        "input_hashes_after": {
            "face": sha256_file(FACE),
            "result": sha256_file(RESULT),
            "plan106": sha256_file(PLAN),
            "plan105": sha256_file(PLAN105),
        },
        "inputs_unchanged": True,
        "items": len(face["episodes"]),
        "rows_analyzed": len(rows),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "category_definitions": {
            "exact_answer": "decoded free output exactly equals the face answer",
            "revision_stale_material": "output is a material observed earlier for the same revision target entity",
            "other_target_observed_material": "output is observed for the target entity but is neither current nor stale revision material",
            "distractor_observed_material": "output matches a material observed for another entity, or any observed material on an item with no target row",
            "unknown_abstention": "the output is unknown for an item whose expected answer is not unknown",
            "empty_output": "decoded output contains no characters",
            "other_unobserved_output": "output matches no lexical category above and is absent from uses-observations",
            "nontext_output": "raw output bytes are not valid ASCII text",
        },
        "overall": _summary(rows),
        "marginals": {
            "seed_arm_kind": _strata(rows, ("seed", "arm", "kind")),
            "seed_arm_kind_answer_length": _strata(
                rows, ("seed", "arm", "kind", "material_answer_length_chars")
            ),
            "seed_arm_kind_target_position": _strata(
                rows, ("seed", "arm", "kind", "target_row_position_1based")
            ),
            "fully_crossed": _strata(rows, dims),
        },
        "paired_position_vs_cut": _paired(rows, "cut"),
        "paired_position_vs_terminal": _paired(rows, "terminal"),
        "classified_rows": rows,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "cpu_seconds": cpu_seconds,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "gate_recomputed": False,
        "can_promote": False,
        "interpretation_limit": "lexical output attribution on matched synthetic diagnostic items only; it does not reveal internal retrieval or establish a causal mechanism, generalization, or capability promotion",
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
        "plan106": sha256_file(PLAN),
        "plan105": sha256_file(PLAN105),
    }
    face = read_json(FACE)
    result = read_json(RESULT)
    report = build_report(face, result, before, time.process_time() - started)
    after = report["input_hashes_after"]
    if before != after:
        raise ValueError("one or more read-only analysis inputs changed during execution")
    report["inputs_unchanged"] = True
    report["cpu_seconds"] = time.process_time() - started
    if report["cpu_seconds"] > CPU_LIMIT_SECONDS:
        raise ValueError("PLAN-N1-106 CPU ceiling exceeded")
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
