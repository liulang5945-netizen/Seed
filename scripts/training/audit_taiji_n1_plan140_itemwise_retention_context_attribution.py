"""离线配对N1-132/139/140的train-only输出与上下文计分。"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-140_baseline_retention_balanced_matched_continuation_draft_20261019.md"
)
N132_RESULT = ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_result_20261019.json"
N139_RESULT = (
    ROOT
    / "reports/taiji_n1_plan139_context_robust_matched_continuation_calibration_result_20261019.json"
)
N140_RESULT = (
    ROOT
    / "reports/taiji_n1_plan140_baseline_retention_balanced_matched_continuation_result_20261019.json"
)
RUNNER = (
    ROOT
    / "scripts/training/run_taiji_n1_plan140_baseline_retention_balanced_matched_continuation.py"
)
REPORT = ROOT / "reports/taiji_n1_plan140_itemwise_retention_context_attribution_20261019.json"
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
        "first_only_correct_ids": sorted(
            item for item in first if first[item] and not second[item]
        ),
        "second_only_correct_ids": sorted(
            item for item in first if not first[item] and second[item]
        ),
    }


def _original_rows(rows: list[dict]) -> list[dict]:
    selected = [row for row in rows if row.get("evaluation_set") == "original"]
    if len(selected) != 192 or len({row["id"] for row in selected}) != 192:
        raise ValueError("frozen original train view must contain 192 unique items")
    return selected


def _paired_by_kind(first: list[dict], second: list[dict]) -> dict[str, dict]:
    return {
        kind: _paired(
            _score_map([row for row in first if row.get("kind") == kind]),
            _score_map([row for row in second if row.get("kind") == kind]),
        )
        for kind in KINDS
    }


def analyze() -> dict:
    if REPORT.exists():
        raise FileExistsError("N1-140 itemwise attribution is write-once")
    started = time.process_time()
    n132 = read_json(N132_RESULT)
    n139 = read_json(N139_RESULT)
    n140 = read_json(N140_RESULT)
    if (
        n132.get("complete") is not True
        or n139.get("complete") is not True
        or n139.get("train_score_queries") != 7_488
        or n139.get("dev_queries") != 0
        or n139.get("holdout_queries") != 0
        or n139.get("can_promote") is not False
        or n140.get("complete") is not True
        or n140.get("training_updates") != 55_296
        or n140.get("train_score_queries") != 7_488
        or n140.get("dev_queries") != 0
        or n140.get("holdout_queries") != 0
        or n140.get("can_promote") is not False
    ):
        raise ValueError("complete train-only N1-132/139/140 reports are required")

    parent_by_seed = {
        int(row["seed"]): row for row in n132["models"] if row.get("arm") == "position"
    }
    if set(parent_by_seed) != {11, 12, 13}:
        raise ValueError("N1-132 must provide all three position parent endpoints")
    n139_by_endpoint = {(int(row["seed"]), row["branch"]): row for row in n139["models"]}
    n140_by_endpoint = {(int(row["seed"]), row["branch"]): row for row in n140["models"]}
    expected_endpoints = {
        (seed, branch) for seed in (11, 12, 13) for branch in ("candidate", "control")
    }
    if set(n139_by_endpoint) != expected_endpoints or set(n140_by_endpoint) != expected_endpoints:
        raise ValueError("N1-139 and N1-140 must each contain six matched endpoints")

    endpoints = {}
    candidate_context = {condition: {} for condition in CONDITIONS}
    control_context = {condition: {} for condition in CONDITIONS}
    gate_passes = 0
    for (seed, branch), model in sorted(n140_by_endpoint.items()):
        label = f"seed{seed}/{branch}"
        parent_rows = _original_rows(parent_by_seed[seed]["train_score_rows"])
        prior_rows = _original_rows(n139_by_endpoint[(seed, branch)]["frozen_gate_rows"])
        final_rows = _original_rows(model["frozen_gate_rows"])
        parent_map = _score_map(parent_rows)
        prior_map = _score_map(prior_rows)
        final_map = _score_map(final_rows)
        sufficient = model.get("training_sufficient") is True
        gate_passes += int(sufficient)

        baseline_transition = _score_map(
            [row for row in final_rows if row.get("kind") == "transition"]
        )
        context_pairs = {}
        for condition in CONDITIONS:
            context_rows = [
                row for row in model["augmented_context_rows"] if row.get("condition") == condition
            ]
            context_map = _score_map(context_rows, key="parent_id")
            if len(context_map) != 48:
                raise ValueError(f"{label} {condition} context view must contain 48 items")
            context_pairs[condition] = _paired(baseline_transition, context_map)
            aggregate = candidate_context if branch == "candidate" else control_context
            aggregate[condition][seed] = context_map

        endpoints[label] = {
            "seed": seed,
            "branch": branch,
            "n132_parent_original_exact": sum(parent_map.values()),
            "n139_final_original_exact": sum(prior_map.values()),
            "n140_final_original_exact": sum(final_map.values()),
            "n132_to_n140_by_kind": _paired_by_kind(parent_rows, final_rows),
            "n139_to_n140_by_kind": _paired_by_kind(prior_rows, final_rows),
            "context_vs_n140_original_transition": context_pairs,
            "original_gate_sufficient": sufficient,
            "train_only_queries": model["train_score_queries"],
        }

    candidate_control = {}
    candidate_totals = {}
    control_totals = {}
    for condition in CONDITIONS:
        candidate_maps = candidate_context[condition]
        control_maps = control_context[condition]
        by_seed = {seed: _paired(control_maps[seed], candidate_maps[seed]) for seed in (11, 12, 13)}
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
        candidate_control[condition] = {
            "by_seed": by_seed,
            "all_seed_matched": _paired(combined_control, combined_candidate),
        }
        candidate_totals[condition] = sum(
            sum(values.values()) for values in candidate_maps.values()
        )
        control_totals[condition] = sum(sum(values.values()) for values in control_maps.values())

    prior_seed13 = n139_by_endpoint[(13, "candidate")]
    final_seed13 = n140_by_endpoint[(13, "candidate")]
    prior_transition = _score_map(
        [
            row
            for row in _original_rows(prior_seed13["frozen_gate_rows"])
            if row["kind"] == "transition"
        ]
    )
    final_transition = _score_map(
        [
            row
            for row in _original_rows(final_seed13["frozen_gate_rows"])
            if row["kind"] == "transition"
        ]
    )
    n139_errors = {item for item, correct in prior_transition.items() if not correct}
    n140_errors = {item for item, correct in final_transition.items() if not correct}

    source_paths = (PLAN, N132_RESULT, N139_RESULT, N140_RESULT, RUNNER, Path(__file__))
    source_hashes = {str(path.resolve()): sha256(path) for path in source_paths}
    unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in source_hashes.items()
    )
    if not unchanged:
        raise ValueError("a bound N1-140 offline attribution source changed during analysis")

    candidate_total = sum(candidate_totals.values())
    control_total = sum(control_totals.values())
    report = {
        "schema": "n1-plan140-itemwise-retention-context-attribution-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": n140["plan_sha256"],
        "n132_result_sha256": sha256(N132_RESULT),
        "n139_result_sha256": sha256(N139_RESULT),
        "n140_result_sha256": sha256(N140_RESULT),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": unchanged,
        "endpoints": endpoints,
        "candidate_vs_control_by_context_condition": candidate_control,
        "candidate_context_free_exact_by_condition": candidate_totals,
        "control_context_free_exact_by_condition": control_totals,
        "candidate_context_free_exact_total": candidate_total,
        "control_context_free_exact_total": control_total,
        "context_score_denominator_each_branch": 288,
        "context_score_denominator_all_three_seeds_per_branch": 864,
        "original_gate_pass_endpoints": gate_passes,
        "original_gate_endpoint_count": 6,
        "seed13_candidate_transition_n139_error_ids": sorted(n139_errors),
        "seed13_candidate_transition_n140_error_ids": sorted(n140_errors),
        "seed13_candidate_transition_error_overlap": sorted(n139_errors & n140_errors),
        "seed13_candidate_transition_new_errors_in_n140": sorted(n140_errors - n139_errors),
        "seed13_candidate_transition_fixed_in_n140": sorted(n139_errors - n140_errors),
        "seed13_candidate_transition_n139_exact": sum(prior_transition.values()),
        "seed13_candidate_transition_n140_exact": sum(final_transition.values()),
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
        f"离线配对分析完成：N1-140原门{gate_passes}/6通过；"
        f"候选上下文{candidate_total}/864、两行控制{control_total}/864；"
        f"seed13 candidate transition由N1-139的{sum(prior_transition.values())}/48"
        f"变为N1-140的{sum(final_transition.values())}/48，"
        f"错误重叠{len(n139_errors & n140_errors)}项、新增{len(n140_errors - n139_errors)}项。"
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
                    "seed13_candidate_transition_n139_exact",
                    "seed13_candidate_transition_n140_exact",
                    "seed13_candidate_transition_error_overlap",
                    "seed13_candidate_transition_new_errors_in_n140",
                    "model_queries",
                    "parameter_updates",
                    "dev_queries",
                    "holdout_queries",
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
