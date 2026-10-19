"""对已批准的PLAN-N1-134一次性评估12个冻结endpoint。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from scripts.training import build_taiji_n1_v16_transition_direction_position_diagnostic as builder
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
        raise FileNotFoundError("必须先生成并核验经批准的v16诊断面")
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
        or attempt.get("model_queries") != 0
        or attempt.get("parameter_updates") != 0
        or face.get("schema") != "n1-transition-direction-position-diagnostic-v1"
        or face.get("seed") != builder.SEED
        or face.get("items") != builder.ITEMS
        or face.get("observations_per_item") != builder.OBSERVATIONS
        or face.get("name_neighbor_distance") != builder.DISTANCE
        or len(face.get("episodes", [])) != builder.ITEMS
        or len(face.get("subsets", {}).get("diagnostic", [])) != builder.ITEMS
        or manifest.get("entity_count") != 7938
        or manifest.get("word_value_count") != 6288
    ):
        raise ValueError("v16题面、readiness或来源排除manifest失效")

    source_hashes = dict(readiness.get("source_hashes", {}))
    source_hashes.update(face.get("exclusion_sources", {}).get("source_hashes", {}))
    source_hashes[str(SOURCE_EXCLUSIONS.resolve())] = sha256(SOURCE_EXCLUSIONS)
    source_hashes[str(FACE.resolve())] = sha256(FACE)
    source_hashes[str(FACE_CHECKS.resolve())] = sha256(FACE_CHECKS)
    source_hashes[str(FACE_ATTEMPT.resolve())] = sha256(FACE_ATTEMPT)
    builder._verify_source_hashes(source_hashes)
    return readiness, face, checks, manifest, source_hashes


def _paired(left: dict[str, bool], right: dict[str, bool]) -> dict:
    if set(left) != set(right):
        raise ValueError("配对endpoint的诊断题目ID不同")
    cells = Counter((left[item], right[item]) for item in left)
    return {
        "both_correct": cells[(True, True)],
        "candidate_only_correct": cells[(True, False)],
        "control_only_correct": cells[(False, True)],
        "both_wrong": cells[(False, False)],
        "candidate_minus_control": sum(left.values()) - sum(right.values()),
    }


def _summarize_direction_position(endpoints: list[dict], face: dict) -> tuple[dict, dict]:
    episode_by_id = {item["id"]: item for item in face["episodes"]}
    cells_by_endpoint = {}
    rows_by_endpoint = {}
    for endpoint in endpoints:
        key = (endpoint["source"], endpoint["seed"], endpoint["arm"])
        row_map = {row["id"]: row for row in endpoint["rows"]}
        if len(row_map) != builder.ITEMS or set(row_map) != set(episode_by_id):
            raise ValueError(f"endpoint诊断行与冻结v16题面不一致：{key}")
        rows_by_endpoint[key] = row_map
        cells = {}
        for direction in builder.DIRECTIONS:
            for target_start in builder.TARGET_STARTS:
                selected = [
                    item_id
                    for item_id, item in episode_by_id.items()
                    if item["answer_direction"] == direction
                    and item["target_start_index"] == target_start
                ]
                exact = sum(row_map[item_id]["free_exact"] is True for item_id in selected)
                cells[f"{direction}/{target_start}"] = {
                    "items": len(selected),
                    "free_exact": exact,
                    "errors": len(selected) - exact,
                }
        cells_by_endpoint[f"{endpoint['source']}/{endpoint['seed']}/{endpoint['arm']}"] = {
            "total_items": len(row_map),
            "free_exact": sum(row["free_exact"] is True for row in row_map.values()),
            "errors": sum(row["free_exact"] is not True for row in row_map.values()),
            "by_direction_and_target_start": cells,
        }

    pairwise = {}
    for seed in SEEDS:
        candidate = rows_by_endpoint[("N1-132", seed, "position")]
        for arm in ("position", "terminal", "cut"):
            control = rows_by_endpoint[("N1-120", seed, arm)]
            pairwise[f"s{seed}_candidate_vs_{arm}"] = {
                "overall": _paired(
                    {item_id: row["free_exact"] is True for item_id, row in candidate.items()},
                    {item_id: row["free_exact"] is True for item_id, row in control.items()},
                ),
                "by_direction_and_target_start": {},
            }
            for direction in builder.DIRECTIONS:
                for target_start in builder.TARGET_STARTS:
                    ids = [
                        item_id
                        for item_id, item in episode_by_id.items()
                        if item["answer_direction"] == direction
                        and item["target_start_index"] == target_start
                    ]
                    pairwise[f"s{seed}_candidate_vs_{arm}"]["by_direction_and_target_start"][
                        f"{direction}/{target_start}"
                    ] = _paired(
                        {item_id: candidate[item_id]["free_exact"] is True for item_id in ids},
                        {item_id: control[item_id]["free_exact"] is True for item_id in ids},
                    )
    return cells_by_endpoint, pairwise


def run_diagnostic(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-134新测面查询需要owner授权")
    if RUN_REPORT.exists() or RUN_ATTEMPT.exists():
        raise FileExistsError("PLAN-N1-134查询阶段一次性，禁止重试")
    started = time.process_time()
    readiness, face, checks, _manifest, source_hashes = _verify_inputs()
    if len(ENDPOINTS) != 12:
        raise ValueError("PLAN-N1-134终点数必须固定为12")
    prior_cpu = float(readiness["cpu_seconds"]) + float(checks["build_cpu_seconds"])
    remaining = CPU_LIMIT_SECONDS - prior_cpu
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-134累计CPU额度不足")

    write_new(
        RUN_ATTEMPT,
        {
            "schema": "n1-plan134-transition-diagnostic-attempt-v1",
            "status": "started",
            "plan_sha256": builder.PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
            "face_sha256": sha256(FACE),
            "planned_queries": len(ENDPOINTS) * builder.ITEMS,
            "parameter_updates": 0,
            "dev_queries": 0,
            "can_promote": False,
            "说明_zh": "对12个冻结endpoint各评估40题；本阶段只执行一次，不重试。",
        },
    )
    source_hashes[str(RUN_ATTEMPT.resolve())] = sha256(RUN_ATTEMPT)
    builder._verify_source_hashes(source_hashes)
    budget = n120_eval.CpuDeadline(remaining)
    n120.configure()
    _, n120_globals, _ = n120._BASE.make_runner(load_inputs=False)
    n120_report = read_json(n133_evaluator.face_builder.N120_RESULT)
    n132_report = read_json(n133_evaluator.face_builder.N132_RESULT)
    endpoint_rows = []
    total_queries = 0
    model_loads = 0
    errors = []
    try:
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
            endpoint_rows.append(
                {
                    "source": spec["source"],
                    "seed": spec["seed"],
                    "arm": spec["arm"],
                    "role": spec["role"],
                    "rows": rows,
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
        direction_position, pairwise = _summarize_direction_position(endpoint_rows, face)
    except BaseException as exc:
        direction_position, pairwise = {}, {}
        errors.append(f"{type(exc).__name__}: {exc}")
    stage_cpu = time.process_time() - started
    cumulative_cpu = prior_cpu + stage_cpu
    complete = (
        not errors
        and len(endpoint_rows) == 12
        and total_queries == 480
        and sum(len(endpoint.get("rows", [])) for endpoint in endpoint_rows) == 480
        and unchanged
        and cumulative_cpu <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan134-transition-direction-position-diagnostic-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": builder.PLAN_SHA256,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": unchanged,
        "readiness_sha256": sha256(READINESS),
        "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "run_attempt_sha256": sha256(RUN_ATTEMPT),
        "items": builder.ITEMS,
        "endpoint_count": len(endpoint_rows),
        "diagnostic_queries": total_queries,
        "model_loads": model_loads,
        "checkpoint_loads": model_loads,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "previous_face_queries": 0,
        "direction_position_summary": direction_position,
        "paired_by_seed": pairwise,
        "range_or_confirmation_gate": None,
        "diagnostic_only": True,
        "cpu_seconds": round(stage_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "can_promote": False,
        "errors": errors,
        "summary_zh": "N1-134对已批准的seed116方向×位置诊断面完成一次性计分；读数仅作描述，不设晋升门，不隔离上下文长度效应。"
        if complete
        else "N1-134诊断未完整完成；保留已执行查询与失败回执，不重试。",
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
