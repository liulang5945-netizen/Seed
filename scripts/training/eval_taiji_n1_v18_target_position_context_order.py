"""对已批准的PLAN-N1-136一次性评估12个冻结endpoint。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from scripts.training import build_taiji_n1_v18_target_position_context_order as builder
from scripts.training import eval_taiji_n1_v11_after_n120 as n120_eval
from scripts.training import eval_taiji_n1_v15_after_n132 as n133_evaluator
from scripts.training import run_taiji_n1_expanded_normalized_rwl_length_balanced as n120

ROOT = builder.ROOT
FACE = builder.FACE
FACE_ATTEMPT = builder.FACE_ATTEMPT
FACE_CHECKS = builder.FACE_CHECKS
READINESS = builder.READINESS
SOURCE_EXCLUSIONS = builder.SOURCE_EXCLUSIONS
RUN_REPORT = builder.RUN_REPORT
RUN_ATTEMPT = builder.RUN_ATTEMPT
CPU_LIMIT_SECONDS = builder.CPU_LIMIT_SECONDS
SEEDS = (11, 12, 13)
ENDPOINTS = tuple(
    [{"source": "N1-132", "seed": seed, "arm": "position"} for seed in SEEDS]
    + [
        {"source": "N1-120", "seed": seed, "arm": arm}
        for arm in ("position", "terminal", "cut")
        for seed in SEEDS
    ]
)
CONDITIONS = ((2, "baseline"), (8, "early"), (8, "late"), (16, "early"), (16, "late"))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _verify_inputs() -> tuple[dict, dict, dict, dict, dict]:
    readiness = builder._verify_readiness()
    if not FACE.is_file() or not FACE_CHECKS.is_file() or not FACE_ATTEMPT.is_file():
        raise FileNotFoundError("必须先生成并核验经批准的v18诊断面")
    face = read_json(FACE)
    checks = read_json(FACE_CHECKS)
    attempt = read_json(FACE_ATTEMPT)
    manifest = read_json(SOURCE_EXCLUSIONS)
    if (
        checks.get("accepted") is not True
        or checks.get("face_sha256") != sha256(FACE)
        or checks.get("plan_sha256") != builder.PLAN_SHA256
        or checks.get("readiness_sha256") != sha256(READINESS)
        or checks.get("source_exclusion_manifest_sha256") != sha256(SOURCE_EXCLUSIONS)
        or checks.get("model_queries") != 0
        or checks.get("parameter_updates") != 0
        or attempt.get("status") != "started"
        or attempt.get("plan_sha256") != builder.PLAN_SHA256
        or attempt.get("readiness_sha256") != sha256(READINESS)
        or attempt.get("source_exclusion_manifest_sha256") != sha256(SOURCE_EXCLUSIONS)
        or attempt.get("planned_items") != builder.ITEMS
        or attempt.get("planned_queries") != builder.ITEMS * len(ENDPOINTS)
        or attempt.get("model_queries") != 0
        or attempt.get("parameter_updates") != 0
        or face.get("schema") != "n1-transition-target-position-context-order-diagnostic-v1"
        or face.get("seed") != builder.SEED
        or face.get("items") != builder.ITEMS
        or face.get("context_lengths") != list(builder.LENGTHS)
        or len(face.get("episodes", [])) != builder.ITEMS
        or manifest.get("entity_count") != builder.EXPECTED_EXCLUDED_ENTITIES
        or manifest.get("word_value_count") != builder.EXPECTED_EXCLUDED_WORDS
    ):
        raise ValueError("v18题面、readiness或来源排除manifest失效")

    contract = builder.audit_v18_face(
        face,
        excluded_names=set(manifest.get("excluded_entities", [])),
        excluded_words=set(manifest.get("excluded_word_values", [])),
    )
    if contract.get("accepted") is not True:
        raise ValueError("v18题面未通过重新执行的合同检查")

    source_hashes = dict(readiness.get("source_hashes", {}))
    source_hashes.update(face.get("exclusion_sources", {}).get("source_hashes", {}))
    source_hashes[str(SOURCE_EXCLUSIONS.resolve())] = sha256(SOURCE_EXCLUSIONS)
    source_hashes[str(FACE.resolve())] = sha256(FACE)
    source_hashes[str(FACE_CHECKS.resolve())] = sha256(FACE_CHECKS)
    source_hashes[str(FACE_ATTEMPT.resolve())] = sha256(FACE_ATTEMPT)
    source_hashes[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    builder._verify_source_hashes(source_hashes)
    return readiness, face, checks, manifest, source_hashes


def _paired(left: dict[str, bool], right: dict[str, bool]) -> dict:
    if set(left) != set(right):
        raise ValueError("匹配条件的模板ID不一致")
    cells = Counter((left[item], right[item]) for item in left)
    return {
        "matched_templates": len(left),
        "both_correct": cells[(True, True)],
        "first_only_correct": cells[(True, False)],
        "second_only_correct": cells[(False, True)],
        "both_wrong": cells[(False, False)],
        "second_minus_first": sum(right.values()) - sum(left.values()),
    }


def _summarize_conditions(face: dict, endpoint_rows: list[dict]) -> tuple[dict, dict]:
    item_by_template: dict[str, dict[tuple[int, str], dict]] = {}
    direction_by_template = {}
    for item in face["episodes"]:
        key = (item["context_length"], item["placement"])
        item_by_template.setdefault(item["template_id"], {})[key] = item
        direction_by_template[item["template_id"]] = item["answer_direction"]

    by_endpoint = {}
    comparisons = {}
    pair_specs = (
        ("8_early_vs_late", (8, "early"), (8, "late")),
        ("16_early_vs_late", (16, "early"), (16, "late")),
        ("baseline_vs_8_early", (2, "baseline"), (8, "early")),
        ("baseline_vs_8_late", (2, "baseline"), (8, "late")),
        ("baseline_vs_16_early", (2, "baseline"), (16, "early")),
        ("baseline_vs_16_late", (2, "baseline"), (16, "late")),
    )

    for endpoint in endpoint_rows:
        endpoint_key = (endpoint["source"], endpoint["seed"], endpoint["arm"])
        row_map = {row["id"]: row for row in endpoint["rows"]}
        if len(row_map) != builder.ITEMS:
            raise ValueError(f"endpoint结果行数不符：{endpoint_key}")
        if set(row_map) != {item["id"] for item in face["episodes"]}:
            raise ValueError(f"endpoint结果ID与v18题面不一致：{endpoint_key}")

        condition_summary = {}
        direction_summary = {}
        for length, placement in CONDITIONS:
            selected = [
                row
                for row in row_map.values()
                if row["context_length"] == length and row["placement"] == placement
            ]
            condition_summary[f"{length}/{placement}"] = {
                "items": len(selected),
                "free_exact": sum(row["free_exact"] is True for row in selected),
                "errors": sum(row["free_exact"] is not True for row in selected),
            }
        for direction in builder.DIRECTIONS:
            direction_summary[direction] = {}
            for length, placement in CONDITIONS:
                selected = [
                    row
                    for row in row_map.values()
                    if row["answer_direction"] == direction
                    and row["context_length"] == length
                    and row["placement"] == placement
                ]
                direction_summary[direction][f"{length}/{placement}"] = {
                    "items": len(selected),
                    "free_exact": sum(row["free_exact"] is True for row in selected),
                }

        by_template = {}
        by_pair = {name: {} for name, _, _ in pair_specs}
        for template_id, conditions in item_by_template.items():
            template_rows = {}
            for condition in CONDITIONS:
                item = conditions[condition]
                row = row_map[item["id"]]
                template_rows[f"{condition[0]}/{condition[1]}"] = {
                    "id": item["id"],
                    "target_start_index": item["target_start_index"],
                    "free_exact": row["free_exact"],
                    "generated_direction": row["generated_direction"],
                    "stop_reason": row["stop_reason"],
                }
            by_template[template_id] = {
                "answer_direction": direction_by_template[template_id],
                "by_condition": template_rows,
            }
            exact = {
                condition: template_rows[f"{condition[0]}/{condition[1]}"]["free_exact"] is True
                for condition in CONDITIONS
            }
            for pair_name, first, second in pair_specs:
                by_pair[pair_name][template_id] = {
                    "first_correct": exact[first],
                    "second_correct": exact[second],
                    "both_correct": exact[first] and exact[second],
                    "first_only_correct": exact[first] and not exact[second],
                    "second_only_correct": not exact[first] and exact[second],
                    "both_wrong": not exact[first] and not exact[second],
                }

        label = f"{endpoint['source']}/{endpoint['seed']}/{endpoint['arm']}"
        by_endpoint[label] = {
            "items": len(row_map),
            "by_condition": condition_summary,
            "by_direction_and_condition": direction_summary,
            "by_matched_template": by_template,
        }
        comparisons[label] = {
            name: _paired(
                {template: value["first_correct"] for template, value in rows.items()},
                {template: value["second_correct"] for template, value in rows.items()},
            )
            for name, rows in by_pair.items()
        }
        comparisons[label]["by_template"] = by_pair

    return by_endpoint, comparisons


def _attach_item_trace(row: dict, item: dict) -> dict:
    raw = bytes.fromhex(row["raw_hex"])
    decoded = raw.decode("utf-8", errors="replace")
    candidate = decoded.strip().lower()
    return {
        **row,
        "template_id": item["template_id"],
        "matched_template_id": item["template_id"],
        "context_length": item["context_length"],
        "placement": item["placement"],
        "target_start_index": item["target_start_index"],
        "target_fact_indices": item["target_fact_indices"],
        "answer_direction": item["answer_direction"],
        "target_direction": item["answer"],
        "turn_word": item["turn_word"],
        "target_bytes_hex": item["answer"].encode("ascii").hex(),
        "generated_text_utf8": decoded,
        "generated_direction": candidate if candidate in builder.DIRECTIONS else None,
    }


def run_diagnostic(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-136新测面查询需要owner授权")
    if RUN_REPORT.exists() or RUN_ATTEMPT.exists():
        raise FileExistsError("PLAN-N1-136查询阶段一次性，禁止重试")

    started = time.process_time()
    write_new(
        RUN_ATTEMPT,
        {
            "schema": "n1-plan136-target-position-diagnostic-attempt-v1",
            "status": "started",
            "plan_sha256": builder.PLAN_SHA256,
            "planned_endpoints": len(ENDPOINTS),
            "planned_items_per_endpoint": builder.ITEMS,
            "planned_queries": len(ENDPOINTS) * builder.ITEMS,
            "parameter_updates": 0,
            "dev_queries": 0,
            "can_promote": False,
            "说明_zh": "对12个冻结endpoint各评估40题，并保存逐题目标位置与生成字节；只执行一次，不重试。",
        },
    )

    readiness = {}
    face = {}
    checks = {}
    source_hashes: dict[str, str] = {}
    endpoint_rows = []
    total_queries = 0
    model_loads = 0
    prior_cpu = 0.0
    errors = []
    try:
        readiness, face, checks, _manifest, source_hashes = _verify_inputs()
        prior_cpu = float(readiness["cpu_seconds"]) + float(checks["build_cpu_seconds"])
        source_hashes[str(RUN_ATTEMPT.resolve())] = sha256(RUN_ATTEMPT)
        builder._verify_source_hashes(source_hashes)
        if len(ENDPOINTS) != 12:
            raise ValueError("PLAN-N1-136冻结endpoint数必须为12")
        remaining = CPU_LIMIT_SECONDS - prior_cpu - (time.process_time() - started)
        if remaining <= 0:
            raise TimeoutError("PLAN-N1-136累计CPU额度不足")

        budget = n120_eval.CpuDeadline(remaining)
        n120.configure()
        _, n120_globals, _ = n120._BASE.make_runner(load_inputs=False)
        n120_report = read_json(n133_evaluator.face_builder.N120_RESULT)
        n132_report = read_json(n133_evaluator.face_builder.N132_RESULT)
        items_by_id = {item["id"]: item for item in face["episodes"]}

        for endpoint in ENDPOINTS:
            budget.guard(reserve=5)
            spec = n133_evaluator._endpoint_spec(
                endpoint["source"], endpoint["seed"], endpoint["arm"], n120_report, n132_report
            )
            rows, queries, loaded, error = n133_evaluator._score_endpoint(
                spec, face["episodes"], budget, n120_globals
            )
            total_queries += queries
            model_loads += int(loaded)
            traced_rows = [_attach_item_trace(row, items_by_id[row["id"]]) for row in rows]
            endpoint_rows.append(
                {
                    "source": spec["source"],
                    "seed": spec["seed"],
                    "arm": spec["arm"],
                    "role": spec["role"],
                    "rows": traced_rows,
                }
            )
            if error:
                errors.append(f"{spec['source']}/{spec['seed']}/{spec['arm']}: {error}")
                break
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    unchanged = True
    for raw_path, expected in source_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            unchanged = False
            errors.append(f"诊断期间来源发生变化：{path}")
            break
    try:
        by_endpoint, comparisons = _summarize_conditions(face, endpoint_rows)
    except BaseException as exc:
        by_endpoint, comparisons = {}, {}
        errors.append(f"{type(exc).__name__}: {exc}")

    stage_cpu = time.process_time() - started
    cumulative_cpu = prior_cpu + stage_cpu
    row_count = sum(len(endpoint.get("rows", [])) for endpoint in endpoint_rows)
    unique_rows = {
        (endpoint["source"], endpoint["seed"], endpoint["arm"], row["id"])
        for endpoint in endpoint_rows
        for row in endpoint["rows"]
    }
    complete = (
        not errors
        and len(endpoint_rows) == 12
        and total_queries == 480
        and row_count == 480
        and len(unique_rows) == 480
        and unchanged
        and cumulative_cpu <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan136-target-position-context-order-diagnostic-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": builder.PLAN_SHA256,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": unchanged,
        "readiness_sha256": sha256(READINESS) if READINESS.is_file() else None,
        "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS)
        if SOURCE_EXCLUSIONS.is_file()
        else None,
        "face_sha256": sha256(FACE) if FACE.is_file() else None,
        "face_checks_sha256": sha256(FACE_CHECKS) if FACE_CHECKS.is_file() else None,
        "run_attempt_sha256": sha256(RUN_ATTEMPT),
        "items": builder.ITEMS,
        "context_lengths": list(builder.LENGTHS),
        "placements": ["baseline", "early", "late"],
        "endpoint_count": len(endpoint_rows),
        "diagnostic_queries": total_queries,
        "model_loads": model_loads,
        "checkpoint_loads": model_loads,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "previous_face_queries": 0,
        "by_endpoint": by_endpoint,
        "matched_position_comparisons_by_endpoint": comparisons,
        "endpoint_results": endpoint_rows,
        "range_or_confirmation_gate": None,
        "diagnostic_only": True,
        "cpu_seconds": round(stage_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "can_promote": False,
        "errors": errors,
        "summary_zh": "N1-136完成同事实、固定8/16行数下的目标位置前置/后置匹配诊断，并保存480条逐题生成字节；仅作描述性对照，不设能力门、不作模型晋升。"
        if complete
        else "N1-136诊断未完整完成；保留已执行查询与失败回执，不重试。",
    }
    write_new(RUN_REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    result = run_diagnostic(owner_approved=args.owner_approved)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "endpoint_count",
                    "diagnostic_queries",
                    "cumulative_cpu_seconds",
                    "can_promote",
                    "summary_zh",
                    "errors",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("complete") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
