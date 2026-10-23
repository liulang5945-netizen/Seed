"""Read-only output-distance and within-template stability analysis for PLAN-N1-105."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/reference/PLAN-N1-107_v10_output_geometry_and_position_stability_20261017.md"
PLAN105 = ROOT / "plans/reference/PLAN-N1-105_counterbalanced_input_feature_diagnostic_20261017.md"
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v10_factorial_diagnostic_20261017.json"
RESULT = ROOT / "reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_20261017.json"
REPORT = ROOT / "reports/taiji_n1_hard_neighbor_v10_output_geometry_20261017.json"
CONTRACT_TEST = ROOT / "tests/taiji_native/test_n1_run107_v10_output_geometry_contract.py"
sys.path.insert(0, str(ROOT))
from scripts.training import seal_ast  # noqa: E402

PLAN_SHA256 = "9504cce61f748ce4cad748d39762be1ef0cfe12b51ec698a20d852768ad818ac"
PLAN105_SHA256 = "af6eebc3e3fb01fc8c409f762176aefa8b91f246e9ea7325f42c5fa3267537ca"
FACE_SHA256 = "2d912941baecaa4811266ace64b56fa9bb2ace65d657691834911db16ed369d8"
RESULT_SHA256 = "287cc5c4a38693d8e055f1290d801b1ec6494768386ff3ad1963e3059852a822"

SCHEMA = "n1-hard-neighbor-v10-output-geometry-v1"
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")
KINDS = ("lookup", "revision", "transition", "unknown")
CPU_LIMIT_SECONDS = 30.0

_MATERIAL_RE = re.compile(r"^(?P<entity>[a-z]+) (?:(?:now )?uses )(?P<value>[a-z]+)\.$")
_RELATION_RE = re.compile(r"^(?P<entity>[a-z]+) (?P<relation>faces|turns) (?P<value>[a-z]+)\.$")
_MATERIAL_QUESTION_RE = re.compile(r"^material of (?P<entity>[a-z]+)\?$", re.IGNORECASE)
_DIRECTION_QUESTION_RE = re.compile(r"^direction of (?P<entity>[a-z]+)\?$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def levenshtein_distance(left: str, right: str) -> int:
    """Return character-level edit distance without optional dependencies."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _question_entity(item: dict) -> str | None:
    for pattern in (_MATERIAL_QUESTION_RE, _DIRECTION_QUESTION_RE):
        match = pattern.fullmatch(item.get("question", ""))
        if match:
            return match.group("entity")
    return None


def observed_candidates(item: dict) -> list[dict]:
    target_entity = _question_entity(item)
    kind = item.get("kind")
    by_entity: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for observation in item.get("observations", []):
        match = _MATERIAL_RE.fullmatch(observation)
        if match:
            by_entity[match.group("entity")].append((match.group("value"), "material"))
            continue
        match = _RELATION_RE.fullmatch(observation)
        if match:
            by_entity[match.group("entity")].append((match.group("value"), "relation"))

    candidates = []
    for entity, values in by_entity.items():
        for index, (value, value_kind) in enumerate(values):
            if entity != target_entity:
                role = "other_entity_observed_value"
            elif kind == "revision" and value_kind == "material" and index < len(values) - 1:
                role = "stale_target_material"
            elif kind in {"lookup", "revision"} and value_kind == "material":
                role = (
                    "current_target_material"
                    if value == item.get("answer")
                    else "other_target_material"
                )
            else:
                role = "target_entity_relation_value"
            candidates.append({"value": value, "role": role, "entity": entity})
    return candidates


def nearest_observed_values(output: str, item: dict) -> dict:
    candidates = observed_candidates(item)
    if not candidates:
        return {"distance": None, "values": []}
    distances = [(levenshtein_distance(output, item["value"]), item) for item in candidates]
    minimum = min(distance for distance, _ in distances)
    nearest = sorted(
        {
            tuple(sorted(candidate.items()))
            for distance, candidate in distances
            if distance == minimum
        }
    )
    return {
        "distance": minimum,
        "values": [dict(candidate) for candidate in nearest],
    }


def validate_matched_family(items: list[dict], expected_positions: list[int]) -> None:
    if not items:
        raise ValueError("matched family is empty")
    positions = [item.get("target_row_position_1based") for item in items]
    if sorted(positions) != expected_positions or len(set(positions)) != len(positions):
        raise ValueError("matched family does not contain the complete frozen position range")
    signature_fields = ("answer", "question", "entities")
    first = items[0]
    for item in items[1:]:
        if any(item.get(field) != first.get(field) for field in signature_fields):
            raise ValueError(
                "matched family answer/question/entity identity changed across positions"
            )
        if sorted(item.get("observations", [])) != sorted(first.get("observations", [])):
            raise ValueError("matched family observation multiset changed across positions")


def _expected_positions(kind: str) -> list[int]:
    if kind == "lookup":
        return list(range(1, 13))
    if kind in {"revision", "transition"}:
        return list(range(1, 12))
    raise ValueError(f"no position factor is defined for {kind}")


def _validate_face_and_result(face: dict, result: dict) -> dict[str, dict]:
    if face.get("schema") != "n1-hard-neighbor-v10-factorial-diagnostic-v1":
        raise ValueError("unexpected diagnostic face schema")
    if face.get("split") != "diagnostic" or face.get("items") != 248:
        raise ValueError("diagnostic face identity/count mismatch")
    episodes = face.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 248:
        raise ValueError("diagnostic face is not the frozen 248-item face")
    face_items = {item.get("id"): item for item in episodes}
    if len(face_items) != 248 or None in face_items:
        raise ValueError("face item identifiers are missing or duplicated")

    expected_kind_counts = {"lookup": 96, "revision": 88, "transition": 44, "unknown": 20}
    if Counter(item.get("kind") for item in episodes) != expected_kind_counts:
        raise ValueError("diagnostic face kind counts differ from PLAN-N1-105")
    families: dict[tuple, list[dict]] = defaultdict(list)
    for item in episodes:
        if item.get("kind") != "unknown":
            family_key = (
                item.get("kind"),
                item.get("template_index"),
                item.get("material_answer_length_chars"),
            )
            families[family_key].append(item)
    family_counts = {"lookup": 12, "revision": 11, "transition": 11}
    expected_family_keys = (
        {("lookup", template, length) for template in range(4) for length in (6, 7)}
        | {("revision", template, length) for template in range(4) for length in (6, 7)}
        | {("transition", template, None) for template in range(4)}
    )
    for (kind, _, _), items in families.items():
        validate_matched_family(items, _expected_positions(kind))
        if len(items) != family_counts[kind]:
            raise ValueError("matched family has an unexpected item count")
    expected_family_count = 4 * (2 + 2 + 1)
    if len(families) != expected_family_count or set(families) != expected_family_keys:
        raise ValueError("diagnostic face has an incomplete matched-family design")

    if (
        result.get("status") != "complete"
        or result.get("complete") is not True
        or result.get("face_sha256") != FACE_SHA256
        or result.get("plan_sha256") != PLAN105_SHA256
        or result.get("model_queries") != 2232
        or result.get("planned_model_queries") != 2232
        or result.get("parameter_updates") != 0
        or result.get("optimizer_steps") != 0
        or result.get("gate_recomputed") is not False
        or result.get("can_promote") is not False
    ):
        raise ValueError("v10 diagnostic result metadata differs from its frozen contract")
    rows = result.get("rows")
    expected_keys = {
        (seed, arm, item_id) for seed in SEEDS for arm in ARMS for item_id in face_items
    }
    if not isinstance(rows, list) or len(rows) != len(expected_keys):
        raise ValueError("result does not have the complete 2,232-row endpoint × face design")
    seen = set()
    for row in rows:
        key = (row.get("seed"), row.get("arm"), row.get("id"))
        if key not in expected_keys or key in seen:
            raise ValueError(f"unexpected or duplicate result row {key}")
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
        if row.get("raw_hex") != row.get("output_hex"):
            raise ValueError(f"raw/output identity mismatch for {row['id']}")
        try:
            decoded = bytes.fromhex(row["raw_hex"]).decode("ascii")
        except (ValueError, UnicodeDecodeError, TypeError) as exc:
            raise ValueError(f"non-text output in frozen result for {row['id']}") from exc
        if row.get("free_exact") is not (decoded == item.get("answer")):
            raise ValueError(f"free_exact flag disagrees with output for {row['id']}")
    if seen != expected_keys:
        raise ValueError("one or more frozen endpoint/item rows are missing")
    return face_items


def _analyze_rows(
    face_items: dict[str, dict], result_rows: list[dict]
) -> tuple[list[dict], list[dict]]:
    geometry_rows = []
    template_rows = []
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in result_rows:
        item = face_items[row["id"]]
        output = bytes.fromhex(row["raw_hex"]).decode("ascii")
        answer = item["answer"]
        distance = levenshtein_distance(output, answer)
        nearest = nearest_observed_values(output, item)
        geometry_row = {
            "seed": row["seed"],
            "arm": row["arm"],
            "id": row["id"],
            "kind": row["kind"],
            "template_index": row["template_index"],
            "target_row_position_1based": row["target_row_position_1based"],
            "material_answer_length_chars": row["material_answer_length_chars"],
            "free_exact": row["free_exact"],
            "output_hex": row["raw_hex"],
            "output_length_chars": len(output),
            "answer_length_chars": len(answer),
            "output_minus_answer_length": len(output) - len(answer),
            "answer_edit_distance": distance,
            "nearest_observed_value_distance": nearest["distance"],
            "nearest_observed_values": nearest["values"],
        }
        geometry_rows.append(geometry_row)
        if item["kind"] != "unknown":
            key = (
                row["seed"],
                row["arm"],
                item["kind"],
                item["template_index"],
                item.get("material_answer_length_chars"),
            )
            grouped[key].append(geometry_row)

    for (seed, arm, kind, template_index, answer_length), group in sorted(
        grouped.items(), key=lambda pair: tuple(str(v) for v in pair[0])
    ):
        group.sort(key=lambda row: row["target_row_position_1based"])
        outputs = [row["output_hex"] for row in group]
        output_counts = Counter(outputs)
        template_rows.append(
            {
                "seed": seed,
                "arm": arm,
                "kind": kind,
                "template_index": template_index,
                "material_answer_length_chars": answer_length,
                "position_variants": len(group),
                "unique_output_count": len(output_counts),
                "modal_output_support": max(output_counts.values()),
                "modal_output_fraction": max(output_counts.values()) / len(group),
                "all_outputs_identical": len(output_counts) == 1,
                "exact_positions": sum(row["free_exact"] for row in group),
                "output_length_matches_answer": sum(
                    row["output_length_chars"] == row["answer_length_chars"] for row in group
                ),
                "answer_edit_distance_histogram": _integer_histogram(
                    row["answer_edit_distance"] for row in group
                ),
                "position_outputs": [
                    {
                        "position": row["target_row_position_1based"],
                        "free_exact": row["free_exact"],
                        "output_hex": row["output_hex"],
                        "answer_edit_distance": row["answer_edit_distance"],
                    }
                    for row in group
                ],
            }
        )
    return geometry_rows, template_rows


def _integer_histogram(values) -> dict[str, int]:
    counts = Counter(values)
    return {str(value): counts[value] for value in sorted(counts)}


def _distance_strata(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["seed"],
            row["arm"],
            row["kind"],
            row["material_answer_length_chars"],
        )
        grouped[key].append(row)
    summary = []
    for (seed, arm, kind, answer_length), group in sorted(
        grouped.items(), key=lambda pair: tuple(str(v) for v in pair[0])
    ):
        errors = [row for row in group if not row["free_exact"]]
        distances = [row["answer_edit_distance"] for row in errors]
        deltas = Counter(row["output_minus_answer_length"] for row in group)
        nearest_roles = Counter(
            candidate["role"] for row in errors for candidate in row["nearest_observed_values"]
        )
        summary.append(
            {
                "seed": seed,
                "arm": arm,
                "kind": kind,
                "material_answer_length_chars": answer_length,
                "denominator": len(group),
                "exact": len(group) - len(errors),
                "errors": len(errors),
                "error_answer_edit_distance_histogram": _integer_histogram(distances),
                "median_error_answer_edit_distance": (
                    statistics.median(distances) if distances else None
                ),
                "output_minus_answer_length_histogram": {
                    str(value): deltas[value] for value in sorted(deltas)
                },
                "output_length_histogram": _integer_histogram(
                    row["output_length_chars"] for row in group
                ),
                "nearest_observed_value_roles_on_errors": {
                    role: nearest_roles[role] for role in sorted(nearest_roles)
                },
            }
        )
    return summary


def _template_summary(template_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in template_rows:
        key = (
            row["seed"],
            row["arm"],
            row["kind"],
            row["material_answer_length_chars"],
        )
        grouped[key].append(row)
    summary = []
    for (seed, arm, kind, answer_length), group in sorted(
        grouped.items(), key=lambda pair: tuple(str(v) for v in pair[0])
    ):
        summary.append(
            {
                "seed": seed,
                "arm": arm,
                "kind": kind,
                "material_answer_length_chars": answer_length,
                "template_count": len(group),
                "templates_with_one_repeated_output": sum(
                    row["all_outputs_identical"] for row in group
                ),
                "mean_unique_output_count": statistics.mean(
                    row["unique_output_count"] for row in group
                ),
                "exact_position_variants": sum(row["exact_positions"] for row in group),
                "total_position_variants": sum(row["position_variants"] for row in group),
            }
        )
    return summary


def build_report(
    face: dict, result: dict, input_hashes: dict[str, str], cpu_seconds: float
) -> dict:
    face_items = _validate_face_and_result(face, result)
    geometry_rows, template_rows = _analyze_rows(face_items, result["rows"])
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
        "input_hashes_before": input_hashes,
        "input_hashes_after": {
            "face": sha256_file(FACE),
            "result": sha256_file(RESULT),
            "plan107": sha256_file(PLAN),
            "plan105": sha256_file(PLAN105),
        },
        "inputs_unchanged": True,
        "items": len(face_items),
        "rows_analyzed": len(geometry_rows),
        "distance_strata": _distance_strata(geometry_rows),
        "template_stability": template_rows,
        "template_stability_summary": _template_summary(template_rows),
        "rows": geometry_rows,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "cpu_seconds": cpu_seconds,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "gate_recomputed": False,
        "can_promote": False,
        "interpretation_limit": "character distances and output stability on matched synthetic diagnostic items only; not evidence of internal retrieval, causality, generalization, or capability promotion",
    }


def run() -> dict:
    started = time.process_time()
    for path, expected in (
        (PLAN, PLAN_SHA256),
        (PLAN105, PLAN105_SHA256),
        (FACE, FACE_SHA256),
        (RESULT, RESULT_SHA256),
    ):
        if not seal_ast.recorded_value_attested(path, expected):
            raise ValueError(f"frozen source hash mismatch: {path.relative_to(ROOT)}")
    before = {
        "face": sha256_file(FACE),
        "result": sha256_file(RESULT),
        "plan107": sha256_file(PLAN),
        "plan105": sha256_file(PLAN105),
    }
    face = read_json(FACE)
    result = read_json(RESULT)
    report = build_report(face, result, before, time.process_time() - started)
    if before != report["input_hashes_after"]:
        raise ValueError("a read-only analysis input changed during execution")
    report["inputs_unchanged"] = True
    report["cpu_seconds"] = time.process_time() - started
    if report["cpu_seconds"] > CPU_LIMIT_SECONDS:
        raise ValueError("PLAN-N1-107 CPU ceiling exceeded")
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
                "template_families": len(report["template_stability"]),
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
