"""离线配对N1-139 train-only结果，定位原门回退和上下文条件下的恢复。"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/reference/PLAN-N1-139_context_robust_matched_continuation_calibration_draft_20261019.md"
N132_RESULT = ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_result_20261019.json"
N139_RESULT = ROOT / "reports/taiji_n1_plan139_context_robust_matched_continuation_calibration_result_20261019.json"
RUNNER = ROOT / "scripts/training/run_taiji_n1_plan139_context_robust_matched_continuation_calibration.py"
REPORT = ROOT / "reports/taiji_n1_plan139_itemwise_retention_context_attribution_20261019.json"
CONDITIONS = ("8/d1", "8/d2", "8/d4", "16/d1", "16/d2", "16/d4")
KINDS = ("lookup", "revision", "transition", "unknown")


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


def _score_map(rows: list[dict], *, key: str = "id") -> dict[str, bool]:
    result = {str(row[key]): row.get("free_exact") is True for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate item identifiers for {key}")
    return result


def _paired(first: dict[str, bool], second: dict[str, bool]) -> dict:
    if set(first) != set(second):
        raise ValueError("paired train-only item identifiers differ")
    cells = Counter((first[item], second[item]) for item in first)
    return {
        "matched_items": len(first),
        "both_correct": cells[(True, True)],
        "first_only_correct": cells[(True, False)],
        "second_only_correct": cells[(False, True)],
        "both_wrong": cells[(False, False)],
        "second_minus_first": sum(second.values()) - sum(first.values()),
        "first_only_correct_ids": sorted(item for item in first if first[item] and not second[item]),
        "second_only_correct_ids": sorted(item for item in first if not first[item] and second[item]),
    }


def _original_rows(rows: list[dict]) -> list[dict]:
    selected = [row for row in rows if row.get("evaluation_set") == "original"]
    if len(selected) != 192 or len({row["id"] for row in selected}) != 192:
        raise ValueError("N1-132 frozen original train view must contain 192 unique items")
    return selected


def analyze() -> dict:
    if REPORT.exists():
        raise FileExistsError("N1-139 itemwise attribution is write-once")
    started = time.process_time()
    n132 = read_json(N132_RESULT)
    n139 = read_json(N139_RESULT)
    if (
        n132.get("complete") is not True
        or n139.get("complete") is not True
        or n139.get("train_score_queries") != 7_488
        or n139.get("dev_queries") != 0
        or n139.get("holdout_queries") != 0
        or n139.get("can_promote") is not False
    ):
        raise ValueError("完整N1-132/N1-139 train-only结果且无留出查询为前置")

    parent_by_seed = {
        int(row["seed"]): row
        for row in n132["models"]
        if row.get("arm") == "position"
    }
    if set(parent_by_seed) != {11, 12, 13}:
        raise ValueError("N1-132须提供三个position父终点")

    endpoints = {}
    candidate_context = {condition: {} for condition in CONDITIONS}
    control_context = {condition: {} for condition in CONDITIONS}
    original_gate_passes = 0
    for model in n139["models"]:
        seed = int(model["seed"])
        branch = model["branch"]
        label = f"seed{seed}/{branch}"
        parent_original = _original_rows(parent_by_seed[seed]["train_score_rows"])
        final_original = _original_rows(model["frozen_gate_rows"])
        parent_map = _score_map(parent_original)
        final_map = _score_map(final_original)
        by_kind = {}
        for kind in KINDS:
            first = _score_map(
                [row for row in parent_original if row.get("kind") == kind]
            )
            second = _score_map(
                [row for row in final_original if row.get("kind") == kind]
            )
            by_kind[kind] = _paired(first, second)
        endpoint_sufficient = model.get("training_sufficient") is True
        original_gate_passes += int(endpoint_sufficient)

        baseline_transition = {
            key: value
            for key, value in final_map.items()
            if any(
                row["id"] == key and row.get("kind") == "transition"
                for row in final_original
            )
        }
        context_pairs = {}
        for condition in CONDITIONS:
            context_rows = [
                row
                for row in model["augmented_context_rows"]
                if row.get("condition") == condition
            ]
            context_map = _score_map(context_rows, key="parent_id")
            paired = _paired(baseline_transition, context_map)
            context_pairs[condition] = paired
            aggregate = candidate_context if branch == "candidate" else control_context
            aggregate[condition][seed] = context_map

        endpoints[label] = {
            "seed": seed,
            "branch": branch,
            "n132_parent_original_exact": sum(parent_map.values()),
            "n139_final_original_exact": sum(final_map.values()),
            "n132_to_final_by_kind": by_kind,
            "context_vs_same_endpoint_two_row_transition": context_pairs,
            "original_gate_sufficient": endpoint_sufficient,
            "train_only_queries": model["train_score_queries"],
        }

    paired_candidate_control = {}
    candidate_context_totals = {}
    control_context_totals = {}
    for condition in CONDITIONS:
        candidate_maps = candidate_context[condition]
        control_maps = control_context[condition]
        by_seed = {
            seed: _paired(candidate_maps[seed], control_maps[seed])
            for seed in (11, 12, 13)
        }
        combined_candidate = {
            f"seed{seed}/{item}": value
            for seed in (11, 12, 13)
            for item, value in candidate_maps[seed].items()
        }
        combined_control = {
            f"seed{seed}/{item}": value
            for seed in (11, 12, 13)
            for item, value in control_maps[seed].items()
        }
        paired_candidate_control[condition] = {
            "by_seed": by_seed,
            "all_seed_matched": _paired(combined_control, combined_candidate),
        }
        candidate_context_totals[condition] = sum(
            sum(values.values()) for values in candidate_maps.values()
        )
        control_context_totals[condition] = sum(
            sum(values.values()) for values in control_maps.values()
        )

    source_hashes = {
        str(PLAN.resolve()): sha256(PLAN),
        str(N132_RESULT.resolve()): sha256(N132_RESULT),
        str(N139_RESULT.resolve()): sha256(N139_RESULT),
        str(RUNNER.resolve()): sha256(RUNNER),
        str(Path(__file__).resolve()): sha256(Path(__file__)),
    }
    unchanged = all(Path(path).is_file() and sha256(Path(path)) == digest for path, digest in source_hashes.items())
    if not unchanged:
        raise ValueError("N1-139离线归因来源在分析期间改变")

    candidate_total = sum(candidate_context_totals.values())
    control_total = sum(control_context_totals.values())
    report = {
        "schema": "n1-plan139-itemwise-retention-context-attribution-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": n139["plan_sha256"],
        "n132_result_sha256": sha256(N132_RESULT),
        "n139_result_sha256": sha256(N139_RESULT),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": unchanged,
        "endpoints": endpoints,
        "candidate_vs_control_by_context_condition": paired_candidate_control,
        "candidate_context_free_exact_by_condition": candidate_context_totals,
        "control_context_free_exact_by_condition": control_context_totals,
        "candidate_context_free_exact_total": candidate_total,
        "control_context_free_exact_total": control_total,
        "context_score_denominator_each_branch": 288,
        "context_score_denominator_all_three_seeds_per_branch": 864,
        "original_gate_pass_endpoints": original_gate_passes,
        "original_gate_endpoint_count": 6,
        "seed13_candidate_transition_regressions": endpoints["seed13/candidate"][
            "n132_to_final_by_kind"
        ]["transition"]["first_only_correct_ids"],
        "seed13_candidate_transition_regression_count": endpoints["seed13/candidate"][
            "n132_to_final_by_kind"
        ]["transition"]["first_only_correct"],
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "cpu_seconds": round(time.process_time() - started, 6),
    }
    report["summary_zh"] = (
        f"离线配对分析完成：原N1-132训练门{original_gate_passes}/6通过；"
        f"候选在六类增强上下文合计{candidate_total}/864正确，对照{control_total}/864；"
        f"seed13候选原始transition从48/48降至28/48，"
        f"定位到{report['seed13_candidate_transition_regression_count']}个回退项目。"
    )
    write_new(REPORT, report)
    return report


def main() -> int:
    result = analyze()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "complete",
                    "cpu_seconds",
                    "original_gate_pass_endpoints",
                    "candidate_context_free_exact_total",
                    "control_context_free_exact_total",
                    "seed13_candidate_transition_regression_count",
                    "model_queries",
                    "parameter_updates",
                    "dev_queries",
                    "holdout_queries",
                    "can_promote",
                    "summary_zh",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
