"""离线配对N1-132/140/141的train-only输出与上下文计分。"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-141_interleaved_context_schedule_matched_continuation_draft_20261019.md"
)
N132_RESULT = ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_result_20261019.json"
N140_RESULT = (
    ROOT
    / "reports/taiji_n1_plan140_baseline_retention_balanced_matched_continuation_result_20261019.json"
)
N141_RESULT = (
    ROOT
    / "reports/taiji_n1_plan141_interleaved_context_schedule_matched_continuation_result_20261019.json"
)
RUNNER = (
    ROOT
    / "scripts/training/run_taiji_n1_plan141_interleaved_context_schedule_matched_continuation.py"
)
REPORT = ROOT / "reports/taiji_n1_plan141_itemwise_retention_context_attribution_20261019.json"
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
        raise FileExistsError("N1-141 itemwise attribution is write-once")
    started = time.process_time()
    n132 = read_json(N132_RESULT)
    n140 = read_json(N140_RESULT)
    n141 = read_json(N141_RESULT)
    if (
        n132.get("complete") is not True
        or n140.get("complete") is not True
        or n140.get("train_score_queries") != 7_488
        or n140.get("dev_queries") != 0
        or n140.get("holdout_queries") != 0
        or n140.get("can_promote") is not False
        or n141.get("complete") is not True
        or n141.get("training_updates") != 55_296
        or n141.get("train_score_queries") != 7_488
        or n141.get("training_sufficient_all_endpoints") is not True
        or n141.get("dev_queries") != 0
        or n141.get("holdout_queries") != 0
        or n141.get("can_promote") is not False
    ):
        raise ValueError("complete train-only N1-132/140/141 reports are required")

    parents = {int(row["seed"]): row for row in n132["models"] if row.get("arm") == "position"}
    old = {(int(row["seed"]), row["branch"]): row for row in n140["models"]}
    new = {(int(row["seed"]), row["branch"]): row for row in n141["models"]}
    expected = {(seed, branch) for seed in (11, 12, 13) for branch in ("candidate", "control")}
    if set(parents) != {11, 12, 13} or set(old) != expected or set(new) != expected:
        raise ValueError("N1-132/140/141 must contain all six seed-matched endpoints")

    endpoints = {}
    candidate_context = {condition: {} for condition in CONDITIONS}
    control_context = {condition: {} for condition in CONDITIONS}
    gate_passes = 0
    for (seed, branch), model in sorted(new.items()):
        label = f"seed{seed}/{branch}"
        parent_rows = _original_rows(parents[seed]["train_score_rows"])
        n140_rows = _original_rows(old[(seed, branch)]["frozen_gate_rows"])
        n141_rows = _original_rows(model["frozen_gate_rows"])
        parent_map = _score_map(parent_rows)
        n140_map = _score_map(n140_rows)
        n141_map = _score_map(n141_rows)
        gate = model.get("training_sufficient") is True
        gate_passes += int(gate)

        transition_original = _score_map(
            [row for row in n141_rows if row.get("kind") == "transition"]
        )
        context_pairs = {}
        for condition in CONDITIONS:
            n140_context_rows = [
                row
                for row in old[(seed, branch)]["augmented_context_rows"]
                if row.get("condition") == condition
            ]
            n141_context_rows = [
                row for row in model["augmented_context_rows"] if row.get("condition") == condition
            ]
            n140_context = _score_map(n140_context_rows, key="parent_id")
            n141_context = _score_map(n141_context_rows, key="parent_id")
            if len(n141_context) != 48:
                raise ValueError(f"{label} {condition} must contain 48 matched items")
            context_pairs[condition] = {
                "n141_vs_original_transition": _paired(transition_original, n141_context),
                "n140_to_n141": _paired(n140_context, n141_context),
            }
            aggregate = candidate_context if branch == "candidate" else control_context
            aggregate[condition][seed] = n141_context

        endpoints[label] = {
            "seed": seed,
            "branch": branch,
            "n132_parent_original_exact": sum(parent_map.values()),
            "n140_final_original_exact": sum(n140_map.values()),
            "n141_final_original_exact": sum(n141_map.values()),
            "n132_to_n141_by_kind": _paired_by_kind(parent_rows, n141_rows),
            "n140_to_n141_by_kind": _paired_by_kind(n140_rows, n141_rows),
            "context_by_condition": context_pairs,
            "original_gate_sufficient": gate,
            "train_only_queries": model["train_score_queries"],
        }

    n141_candidate_vs_control = {}
    n141_candidate_vs_n140 = {}
    candidate_totals = {}
    control_totals = {}
    for condition in CONDITIONS:
        c_maps = candidate_context[condition]
        k_maps = control_context[condition]
        by_seed = {seed: _paired(k_maps[seed], c_maps[seed]) for seed in (11, 12, 13)}
        c_all = {
            f"seed{seed}/{item}": value
            for seed in (11, 12, 13)
            for item, value in c_maps[seed].items()
        }
        k_all = {
            f"seed{seed}/{item}": value
            for seed in (11, 12, 13)
            for item, value in k_maps[seed].items()
        }
        n141_candidate_vs_control[condition] = {
            "by_seed": by_seed,
            "all_seed_matched": _paired(k_all, c_all),
        }
        prior_c = {
            seed: _score_map(
                [
                    row
                    for row in old[(seed, "candidate")]["augmented_context_rows"]
                    if row.get("condition") == condition
                ],
                key="parent_id",
            )
            for seed in (11, 12, 13)
        }
        prior_all = {
            f"seed{seed}/{item}": value
            for seed in (11, 12, 13)
            for item, value in prior_c[seed].items()
        }
        n141_candidate_vs_n140[condition] = _paired(prior_all, c_all)
        candidate_totals[condition] = sum(sum(values.values()) for values in c_maps.values())
        control_totals[condition] = sum(sum(values.values()) for values in k_maps.values())

    n140_13 = _score_map(
        [
            row
            for row in _original_rows(old[(13, "candidate")]["frozen_gate_rows"])
            if row.get("kind") == "transition"
        ]
    )
    n141_13 = _score_map(
        [
            row
            for row in _original_rows(new[(13, "candidate")]["frozen_gate_rows"])
            if row.get("kind") == "transition"
        ]
    )
    old_errors = {item for item, correct in n140_13.items() if not correct}
    new_errors = {item for item, correct in n141_13.items() if not correct}

    source_paths = (PLAN, N132_RESULT, N140_RESULT, N141_RESULT, RUNNER, Path(__file__))
    source_hashes = {str(path.resolve()): sha256(path) for path in source_paths}
    unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in source_hashes.items()
    )
    if not unchanged:
        raise ValueError("an N1-141 offline attribution source changed during analysis")

    candidate_total = sum(candidate_totals.values())
    control_total = sum(control_totals.values())
    report = {
        "schema": "n1-plan141-itemwise-retention-context-attribution-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": n141["plan_sha256"],
        "n132_result_sha256": sha256(N132_RESULT),
        "n140_result_sha256": sha256(N140_RESULT),
        "n141_result_sha256": sha256(N141_RESULT),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": unchanged,
        "endpoints": endpoints,
        "n141_candidate_vs_control_by_context": n141_candidate_vs_control,
        "n141_candidate_vs_n140_by_context": n141_candidate_vs_n140,
        "n141_candidate_free_exact_by_condition": candidate_totals,
        "n141_control_free_exact_by_condition": control_totals,
        "n141_candidate_free_exact_total": candidate_total,
        "n141_control_free_exact_total": control_total,
        "context_denominator_each_branch": 288,
        "context_denominator_all_seeds_each_branch": 864,
        "n141_original_gate_pass_endpoints": gate_passes,
        "endpoint_count": 6,
        "seed13_candidate_transition_n140_error_ids": sorted(old_errors),
        "seed13_candidate_transition_n141_error_ids": sorted(new_errors),
        "seed13_candidate_transition_error_overlap": sorted(old_errors & new_errors),
        "seed13_candidate_transition_fixed_in_n141": sorted(old_errors - new_errors),
        "seed13_candidate_transition_new_errors_in_n141": sorted(new_errors - old_errors),
        "seed13_candidate_transition_n140_exact": sum(n140_13.values()),
        "seed13_candidate_transition_n141_exact": sum(n141_13.values()),
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
        f"离线配对分析完成：N1-141原训练门{gate_passes}/6通过；"
        f"candidate增强上下文{candidate_total}/864，control{control_total}/864；"
        f"seed13 candidate transition由N1-140的{sum(n140_13.values())}/48"
        f"变为N1-141的{sum(n141_13.values())}/48，"
        f"修复{len(old_errors - new_errors)}项、回退{len(new_errors - old_errors)}项。"
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
                    "n141_original_gate_pass_endpoints",
                    "n141_candidate_free_exact_total",
                    "n141_control_free_exact_total",
                    "seed13_candidate_transition_n140_exact",
                    "seed13_candidate_transition_n141_exact",
                    "seed13_candidate_transition_error_overlap",
                    "seed13_candidate_transition_fixed_in_n141",
                    "seed13_candidate_transition_new_errors_in_n141",
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
