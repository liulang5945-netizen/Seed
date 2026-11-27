"""对已完成的PLAN-N1-133 v15 confirm回执做零查询逐题归因。"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys as _sys

_sys.path.insert(0, str(ROOT))

from scripts.training import seal_ast  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-133_v15_binding_supervision_confirmation_draft_20261019.md"
READINESS = (
    ROOT / "reports/taiji_n1_v15_after_n132_confirmation_readiness_revalidation2_20261019.json"
)
SOURCE_EXCLUSIONS = (
    ROOT / "plans/manifests/n1_v15_after_n132_source_exclusions_revalidation2_20261019.json"
)
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v15_confirmation_seed115_20261019.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_v15_after_n132_face_checks_20261019.json"
RANGE_REPORT = ROOT / "reports/taiji_n1_v15_after_n132_range_20261019.json"
CONFIRM_REPORT = ROOT / "reports/taiji_n1_v15_after_n132_confirmation_20261019.json"
OUTPUT = ROOT / "reports/taiji_n1_v15_after_n132_itemwise_attribution_20261019.json"

SEEDS = (11, 12, 13)
KINDS = ("lookup", "revision", "transition", "unknown")
EXPECTED_ENDPOINTS = {
    *(("N1-132", seed, "position") for seed in SEEDS),
    *(("N1-120", seed, arm) for seed in SEEDS for arm in ("position", "terminal", "cut")),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _paired(left: dict[str, bool], right: dict[str, bool]) -> dict:
    if set(left) != set(right):
        raise ValueError("配对端点的confirm题目ID不一致")
    cells = Counter((left[item_id], right[item_id]) for item_id in left)
    return {
        "both_correct": cells[(True, True)],
        "candidate_only_correct": cells[(True, False)],
        "control_only_correct": cells[(False, True)],
        "both_wrong": cells[(False, False)],
        "candidate_minus_control": sum(left.values()) - sum(right.values()),
    }


def build_report() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"一次性逐题归因报告已存在：{OUTPUT}")
    started = time.process_time()
    readiness = read_json(READINESS)
    exclusions = read_json(SOURCE_EXCLUSIONS)
    face = read_json(FACE)
    face_checks = read_json(FACE_CHECKS)
    range_report = read_json(RANGE_REPORT)
    confirmation = read_json(CONFIRM_REPORT)
    if (
        readiness.get("complete") is not True
        or readiness.get("model_queries") != 0
        or readiness.get("parameter_updates") != 0
        or face.get("schema") != "n1-hard-neighbor-v15-confirmation-v1"
        or face.get("seed") != 115
        or face.get("items") != 40
        or face.get("observations_per_item") != 16
        or face_checks.get("accepted") is not True
        or not seal_ast.receipt_binding_matches(face_checks.get("face_sha256"), FACE)
        or range_report.get("complete") is not True
        or range_report.get("range_gate", {}).get("range_check_passed") is not True
        or range_report.get("usable_endpoints") not in (None, 7)
        or range_report.get("model_queries") != 180
        or range_report.get("confirmation_queries") != 0
        or confirmation.get("complete") is not True
        or confirmation.get("confirmation_queries") != 240
        or confirmation.get("total_model_queries") != 420
        or confirmation.get("parameter_updates") != 0
        or confirmation.get("dev_queries") != 0
        or confirmation.get("source_hashes_unchanged") is not True
        or confirmation.get("can_promote") is not False
        or not seal_ast.receipt_binding_matches(confirmation.get("face_sha256"), FACE)
        or not seal_ast.receipt_binding_matches(confirmation.get("face_checks_sha256"), FACE_CHECKS)
        or not seal_ast.receipt_binding_matches(
            confirmation.get("range_report_sha256"), RANGE_REPORT
        )
        or not seal_ast.receipt_binding_matches(confirmation.get("readiness_sha256"), READINESS)
        or exclusions.get("entity_count") != 7308
        or exclusions.get("word_value_count") != 5808
    ):
        raise ValueError("PLAN-N1-133已消费证据不完整或哈希/查询账不一致")

    confirm_ids = set(face.get("subsets", {}).get("confirm", []))
    items = {item["id"]: item for item in face.get("episodes", []) if item["id"] in confirm_ids}
    if len(confirm_ids) != 20 or set(items) != confirm_ids:
        raise ValueError("冻结confirm题面必须恰有20个唯一题目")

    rows_by_endpoint: dict[tuple[str, int, str], dict[str, dict]] = {}
    for endpoint in confirmation.get("endpoint_results", []):
        key = (endpoint.get("source"), endpoint.get("seed"), endpoint.get("arm"))
        if key not in EXPECTED_ENDPOINTS or key in rows_by_endpoint:
            raise ValueError(f"confirm endpoint重复或越界：{key}")
        rows = endpoint.get("rows", [])
        row_map = {row.get("id"): row for row in rows}
        if (
            len(rows) != 20
            or len(row_map) != 20
            or set(row_map) != confirm_ids
            or any(row.get("split") != "holdout" for row in rows)
            or any(not isinstance(row.get("free_exact"), bool) for row in rows)
        ):
            raise ValueError(f"confirm endpoint行集与冻结题面不一致：{key}")
        rows_by_endpoint[key] = row_map
    if set(rows_by_endpoint) != EXPECTED_ENDPOINTS:
        raise ValueError("confirm结果必须覆盖冻结的12个endpoint")

    arm_keys = {
        "N1-132_position": ("N1-132", "position"),
        "N1-120_position": ("N1-120", "position"),
        "N1-120_terminal": ("N1-120", "terminal"),
        "N1-120_cut": ("N1-120", "cut"),
    }
    scores_by_seed = {}
    scores_by_arm_kind = {}
    for label, (source, arm) in arm_keys.items():
        scores_by_seed[label] = {}
        scores_by_arm_kind[label] = {
            kind: {"items": 0, "free_exact": 0, "errors": 0} for kind in KINDS
        }
        for seed in SEEDS:
            row_map = rows_by_endpoint[(source, seed, arm)]
            scores_by_seed[label][str(seed)] = {
                "total": sum(row["free_exact"] for row in row_map.values()),
                "by_kind": {
                    kind: {
                        "items": sum(items[item_id]["kind"] == kind for item_id in row_map),
                        "free_exact": sum(
                            items[item_id]["kind"] == kind and row["free_exact"]
                            for item_id, row in row_map.items()
                        ),
                    }
                    for kind in KINDS
                },
            }
            for item_id, row in row_map.items():
                kind = items[item_id]["kind"]
                scores_by_arm_kind[label][kind]["items"] += 1
                scores_by_arm_kind[label][kind]["free_exact"] += row["free_exact"]
        for kind in KINDS:
            cell = scores_by_arm_kind[label][kind]
            cell["errors"] = cell["items"] - cell["free_exact"]

    comparisons = (
        ("N1-132_position_vs_N1-120_position", "N1-120", "position"),
        ("N1-132_position_vs_N1-120_terminal", "N1-120", "terminal"),
        ("N1-132_position_vs_N1-120_cut", "N1-120", "cut"),
    )
    paired_by_seed = {}
    paired_aggregate_by_kind = {}
    for comparison, control_source, control_arm in comparisons:
        paired_by_seed[comparison] = {}
        paired_aggregate_by_kind[comparison] = {}
        for seed in SEEDS:
            candidate_rows = rows_by_endpoint[("N1-132", seed, "position")]
            control_rows = rows_by_endpoint[(control_source, seed, control_arm)]
            by_kind = {
                kind: _paired(
                    {
                        item_id: row["free_exact"]
                        for item_id, row in candidate_rows.items()
                        if items[item_id]["kind"] == kind
                    },
                    {
                        item_id: row["free_exact"]
                        for item_id, row in control_rows.items()
                        if items[item_id]["kind"] == kind
                    },
                )
                for kind in KINDS
            }
            paired_by_seed[comparison][str(seed)] = {
                "overall": _paired(
                    {item_id: row["free_exact"] for item_id, row in candidate_rows.items()},
                    {item_id: row["free_exact"] for item_id, row in control_rows.items()},
                ),
                "by_kind": by_kind,
            }
        for kind in KINDS:
            paired_aggregate_by_kind[comparison][kind] = _paired(
                {
                    f"{seed}:{item_id}": row["free_exact"]
                    for seed in SEEDS
                    for item_id, row in rows_by_endpoint[("N1-132", seed, "position")].items()
                    if items[item_id]["kind"] == kind
                },
                {
                    f"{seed}:{item_id}": row["free_exact"]
                    for seed in SEEDS
                    for item_id, row in rows_by_endpoint[
                        (control_source, seed, control_arm)
                    ].items()
                    if items[item_id]["kind"] == kind
                },
            )

    itemwise = []
    for item_id in face["subsets"]["confirm"]:
        item = items[item_id]
        candidate = {
            str(seed): rows_by_endpoint[("N1-132", seed, "position")][item_id]["free_exact"]
            for seed in SEEDS
        }
        same_seed_position = {
            str(seed): rows_by_endpoint[("N1-120", seed, "position")][item_id]["free_exact"]
            for seed in SEEDS
        }
        same_seed_cut = {
            str(seed): rows_by_endpoint[("N1-120", seed, "cut")][item_id]["free_exact"]
            for seed in SEEDS
        }
        itemwise.append(
            {
                "id": item_id,
                "kind": item["kind"],
                "N1_132_position_correct_seeds": [
                    seed for seed, correct in candidate.items() if correct
                ],
                "N1_132_position_error_seeds": [
                    seed for seed, correct in candidate.items() if not correct
                ],
                "N1_120_position_correct_seeds": [
                    seed for seed, correct in same_seed_position.items() if correct
                ],
                "N1_120_cut_correct_seeds": [
                    seed for seed, correct in same_seed_cut.items() if correct
                ],
            }
        )

    gate_summary = confirmation.get("confirmation_summary", {}).get("package_vs_n120_controls", {})
    if gate_summary.get("all_four_gates") is not False:
        raise ValueError("逐题归因仅适用于本次冻结确认门未通过的结果")
    transition_errors = scores_by_arm_kind["N1-132_position"]["transition"]["errors"]
    if transition_errors != 12:
        raise ValueError("N1-132 transition错误数与确认摘要不一致")

    source_paths = (
        PLAN,
        READINESS,
        SOURCE_EXCLUSIONS,
        FACE,
        FACE_CHECKS,
        RANGE_REPORT,
        CONFIRM_REPORT,
        Path(__file__).resolve(),
    )
    source_hashes = {str(path.resolve()): sha256(path) for path in source_paths}
    cpu_seconds = round(time.process_time() - started, 6)
    report = {
        "schema": "n1-plan133-v15-confirmation-itemwise-attribution-v1",
        "status": "complete",
        "plan_sha256": sha256(PLAN),
        "source_hashes": source_hashes,
        "face_sha256": sha256(FACE),
        "range_report_sha256": sha256(RANGE_REPORT),
        "confirmation_report_sha256": sha256(CONFIRM_REPORT),
        "confirm_items": 20,
        "confirm_endpoint_rows": 240,
        "query_counts_for_this_analysis": 0,
        "parameter_updates": 0,
        "model_loads": 0,
        "model_queries": 0,
        "dev_queries": 0,
        "can_promote": False,
        "原确认门状态": gate_summary,
        "按seed及类别准确数": scores_by_seed,
        "四类跨seed汇总": scores_by_arm_kind,
        "候选对照逐题配对": {
            "按seed": paired_by_seed,
            "跨seed按类别": paired_aggregate_by_kind,
        },
        "逐题正确与错误seed": itemwise,
        "interpretation_zh": [
            "N1-132 position为41/60，terminal为15/60，cut为45/60；相对terminal领先26题，相对cut落后4题，原四项合取门未通过。",
            "N1-132三个seed总分均达到10/20，但seed12与seed13的transition分别为0/5与1/5，导致逐seed逐类别门失败。",
            "N1-132 position在transition类共3/15正确、12/15错误；N1-120 cut为6/15、terminal为2/15。revision三组均为15/15；因此总体cut差距主要集中在transition，同时lookup也低于cut。",
            "同seed N1-120 position为描述性父档；本报告只对已消费确认结果做逐题配对统计，不新增门、不作因果结论，也不证明L_bind单独效果。",
        ],
        "summary_zh": "v15 confirm逐题离线归因完成；0模型加载、0查询、0参数更新。原确认门未通过，结果只用于定位已消费题面的错误分布。",
        "cpu_seconds": cpu_seconds,
    }
    write_new(OUTPUT, report)
    return report


if __name__ == "__main__":
    result = build_report()
    print(
        json.dumps(
            {
                "status": result["status"],
                "confirm_items": result["confirm_items"],
                "confirm_endpoint_rows": result["confirm_endpoint_rows"],
                "query_counts_for_this_analysis": result["query_counts_for_this_analysis"],
                "parameter_updates": result["parameter_updates"],
                "summary_zh": result["summary_zh"],
                "cpu_seconds": result["cpu_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
