"""只读复核N1-163已消费训练评分的逐题配对变化。"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "reports/taiji_n1_plan163_transition_template_rebalanced_result_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan163_transition_template_rebalanced_progress_20261020.jsonl"
READINESS = ROOT / "reports/taiji_n1_plan163a_transition_template_rebalanced_readiness_20261020.json"
MANIFEST = ROOT / "plans/manifests/n1_plan163a_transition_template_rebalanced_schedule_20261020.json"
PLAN = ROOT / "plans/reference/PLAN-N1-163_transition_template_rebalanced_matched_continuation_20261020.md"
RUNNER = ROOT / "scripts/training/run_taiji_n1_plan163_transition_template_rebalanced_continuation.py"
AUDITOR = Path(__file__).resolve()
INITIAL_REPORT = ROOT / "reports/taiji_n1_plan163_itemwise_attribution_20261020.json"
REPORT = ROOT / "reports/taiji_n1_plan163_itemwise_attribution_v2_20261020.json"
INITIAL_REPORT_SHA256 = "83ce0037a44a040d8a9f241baf2ae319c1cdf6eadb2c9a4d507e623905f864b6"

SEEDS = (11, 12, 13)
ARMS = ("surface_replay_control", "preferred_turn_replacement")
GATE_ROWS_PER_ENDPOINT = 960
LENGTH_ROWS_PER_ENDPOINT = 288
TRAINING_STEPS_PER_ENDPOINT = 18_432
CPU_LIMIT_SECONDS = 120.0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_view = {
        view: [row for row in rows if row["evaluation_set"] == view]
        for view in ("original", "pair", "rename", "rename_pair")
    }
    factorial = {
        view: [row for row in rows if row["evaluation_set"] == view]
        for view in ("factorial_original", "factorial_rename")
    }
    joint_four: dict[str, int] = {}
    factorial_rows_per_parent: dict[str, dict[str, int]] = {}
    for view, view_rows in factorial.items():
        grouped: dict[str, list[bool]] = defaultdict(list)
        for row in view_rows:
            grouped[str(row["id"])].append(bool(row["free_exact"]))
        if len(grouped) != 48 or any(len(values) != 4 for values in grouped.values()):
            raise ValueError(f"bad factorial parent grouping for {view}")
        joint_four[view] = sum(all(values) for values in grouped.values())
        factorial_rows_per_parent[view] = {key: len(value) for key, value in grouped.items()}

    free_by_kind = {
        view: dict(
            sorted(
                Counter(
                    row["kind"]
                    for row in by_view[view]
                    if row["free_exact"]
                ).items()
            )
        )
        for view in ("original", "rename")
    }
    return {
        "free_exact": sum(bool(row["free_exact"]) for row in by_view["original"]),
        "rename_free_exact": sum(bool(row["free_exact"]) for row in by_view["rename"]),
        "pair_exact": sum(bool(row["free_exact"]) for row in by_view["pair"]),
        "rename_pair_exact": sum(bool(row["free_exact"]) for row in by_view["rename_pair"]),
        "free_by_kind": free_by_kind["original"],
        "rename_free_by_kind": free_by_kind["rename"],
        "joint_four_original": joint_four["factorial_original"],
        "joint_four_rename": joint_four["factorial_rename"],
        "rows_by_view": {key: len(value) for key, value in by_view.items()}
        | {key: len(value) for key, value in factorial.items()},
    }


def _moves(
    control: dict[tuple[str, str], dict[str, Any]],
    candidate: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    if set(control) != set(candidate):
        raise ValueError("paired control/candidate item keys differ")
    grouped: dict[tuple[str, str], list[dict[str, int]]] = defaultdict(list)
    for key in sorted(control):
        left = control[key]
        right = candidate[key]
        if left["kind"] != right["kind"]:
            raise ValueError(f"paired task-kind mismatch: {key}")
        if left.get("target_symbols") != right.get("target_symbols"):
            raise ValueError(f"paired target mismatch: {key}")
        was_correct = bool(left["free_exact"])
        is_correct = bool(right["free_exact"])
        view = left.get("evaluation_set")
        if view is None:
            view = f"context_length_{left['context_length']}"
        grouped[(str(view), str(left["kind"]))].append(
            {
                "control_correct": int(was_correct),
                "candidate_correct": int(is_correct),
                "improved": int(not was_correct and is_correct),
                "regressed": int(was_correct and not is_correct),
                "unchanged_correct": int(was_correct and is_correct),
                "unchanged_wrong": int(not was_correct and not is_correct),
                "same_output": int(left.get("output_hex") == right.get("output_hex")),
            }
        )
    output = []
    for (view, kind), values in sorted(grouped.items()):
        sums = {key: sum(row[key] for row in values) for key in values[0]}
        output.append(
            {
                "evaluation_set": view,
                "kind": kind,
                "items": len(values),
                **sums,
                "net_correct_change": sums["improved"] - sums["regressed"],
            }
        )
    return {"items": len(control), "by_view_and_kind": output}


def _indexed(rows: list[dict[str, Any]], *, length: bool) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    occurrences: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if length:
            base = (f"{row['context_length']}:{row['kind']}", str(row["id"]))
        else:
            base = (str(row["evaluation_set"]), str(row["id"]))
        occurrence = occurrences[base]
        occurrences[base] += 1
        key = (base[0], f"{base[1]}#{occurrence}")
        if key in indexed:
            raise ValueError(f"duplicate paired score key: {key}")
        indexed[key] = row
    return indexed


def main() -> int:
    started = time.process_time()
    if REPORT.exists():
        raise FileExistsError(f"write-once report already exists: {REPORT}")
    if not INITIAL_REPORT.is_file() or sha256_file(INITIAL_REPORT) != INITIAL_REPORT_SHA256:
        raise ValueError("initial N1-164 report differs from the frozen supersession binding")
    result_bytes = RESULT.read_bytes()
    result = json.loads(result_bytes)
    if result.get("status") != "complete" or result.get("complete") is not True:
        raise ValueError("N1-163 result is not complete")
    if result.get("source_hashes_unchanged") is not True:
        raise ValueError("N1-163 source hashes did not remain stable")
    if result.get("can_promote") is not False:
        raise ValueError("N1-163 promotion state changed")
    if result.get("seeds") != list(SEEDS) or result.get("arms") != list(ARMS):
        raise ValueError("N1-163 seed/arm order differs from the frozen audit contract")

    historical_hashes = result.get("source_hashes", {})
    expected_sources = {
        PLAN: result["plan_sha256"],
        RUNNER: None,
        READINESS: result["readiness_sha256"],
        MANIFEST: result["manifest_sha256"],
    }
    current_hashes = {path.resolve(): sha256_file(path) for path in expected_sources}
    runner_recorded = None
    for path, digest in historical_hashes.items():
        resolved = Path(path).resolve()
        if resolved == RUNNER.resolve():
            runner_recorded = digest
    if runner_recorded is None:
        raise ValueError("N1-163 result has no runner source binding")
    expected_sources[RUNNER] = runner_recorded
    source_checks = {}
    for path, expected in expected_sources.items():
        actual = current_hashes[path.resolve()]
        if actual != expected:
            raise ValueError(f"source binding differs for {path}: {actual} != {expected}")
        source_checks[str(path.relative_to(ROOT))] = actual

    models = result["models"]
    expected_model_order = [(seed, arm) for seed in SEEDS for arm in ARMS]
    actual_model_order = [(row["seed"], row["arm"]) for row in models]
    if actual_model_order != expected_model_order:
        raise ValueError(f"N1-163 result endpoint order differs: {actual_model_order}")

    gate_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    training_steps: dict[tuple[int, str], set[int]] = defaultdict(set)
    progress_events = Counter()
    gate_arm_labels = Counter()
    progress_digest = hashlib.sha256()
    with PROGRESS.open("rb") as handle:
        for raw in handle:
            progress_digest.update(raw)
            row = json.loads(raw)
            event = row.get("event")
            progress_events[event] += 1
            if event == "training_update":
                training_steps[(int(row["seed"]), str(row["arm"]))].add(int(row["step"]))
            elif event == "original_train_gate_row":
                gate_rows.append(row)
                gate_arm_labels[str(row.get("arm"))] += 1
            elif event == "length_score_row":
                length_rows.append(row)
    progress_sha = progress_digest.hexdigest()

    if progress_events["training_update"] != result["total_updates"]:
        raise ValueError("training progress row count differs from the result")
    for key in expected_model_order:
        steps = training_steps[key]
        if len(steps) != TRAINING_STEPS_PER_ENDPOINT or min(steps) != 0 or max(steps) != TRAINING_STEPS_PER_ENDPOINT - 1:
            raise ValueError(f"training update sequence is incomplete for {key}")
    if len(gate_rows) != result["gate_queries_per_endpoint"] * len(models):
        raise ValueError("train-gate progress row count differs from the result")
    if len(length_rows) != result["length_queries_per_endpoint"] * len(models):
        raise ValueError("length progress row count differs from the result")
    if progress_events["original_train_gate_row"] != len(gate_rows) or progress_events["length_score_row"] != len(length_rows):
        raise ValueError("progress event counts differ from loaded row counts")

    gate_blocks: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for index, model_row in enumerate(models):
        block = gate_rows[index * GATE_ROWS_PER_ENDPOINT : (index + 1) * GATE_ROWS_PER_ENDPOINT]
        key = (int(model_row["seed"]), str(model_row["arm"]))
        if len(block) != GATE_ROWS_PER_ENDPOINT or any(int(row["seed"]) != key[0] for row in block):
            raise ValueError(f"gate progress block does not match result order for {key}")
        recomputed = _summary(block)
        expected = model_row["original_train_gate"]
        fields = {
            "free_exact": expected["free_exact"],
            "rename_free_exact": expected["rename_free_exact"],
            "pair_exact": expected["pair_exact"],
            "rename_pair_exact": expected["rename_pair_exact"],
            "free_by_kind": expected["free_by_kind"],
            "rename_free_by_kind": expected["rename_free_by_kind"],
            "joint_four_original": expected["factorial"]["original"]["joint_four_exact"],
            "joint_four_rename": expected["factorial"]["rename"]["joint_four_exact"],
        }
        for field, value in fields.items():
            if recomputed[field] != value:
                raise ValueError(f"recomputed {field} differs from result for {key}: {recomputed[field]} != {value}")
        if recomputed["rows_by_view"] != {
            "original": 192,
            "pair": 96,
            "rename": 192,
            "rename_pair": 96,
            "factorial_original": 192,
            "factorial_rename": 192,
        }:
            raise ValueError(f"train-gate row strata differ from frozen shape for {key}")
        if model_row["training_sufficient"] is not bool(model_row["training_sufficient"]):
            raise ValueError(f"invalid training gate value for {key}")
        gate_blocks[key] = block

    length_blocks: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in length_rows:
        length_blocks[(int(row["seed"]), str(row["arm"]))].append(row)
    for key in expected_model_order:
        block = length_blocks[key]
        if len(block) != LENGTH_ROWS_PER_ENDPOINT:
            raise ValueError(f"length score count differs for {key}")
        if any(row.get("split") != "train" for row in block):
            raise ValueError(f"non-train row appeared in length scores for {key}")

    pairwise_gate = []
    pairwise_length = []
    for seed in SEEDS:
        control_key = (seed, ARMS[0])
        candidate_key = (seed, ARMS[1])
        pairwise_gate.append(
            {
                "seed": seed,
                "control_training_sufficient": next(r["training_sufficient"] for r in models if (r["seed"], r["arm"]) == control_key),
                "candidate_training_sufficient": next(r["training_sufficient"] for r in models if (r["seed"], r["arm"]) == candidate_key),
                **_moves(
                    _indexed(gate_blocks[control_key], length=False),
                    _indexed(gate_blocks[candidate_key], length=False),
                ),
            }
        )
        pairwise_length.append(
            {
                "seed": seed,
                **_moves(
                    _indexed(length_blocks[control_key], length=True),
                    _indexed(length_blocks[candidate_key], length=True),
                ),
            }
        )

    stage_cpu = time.process_time() - started
    if stage_cpu > CPU_LIMIT_SECONDS:
        raise TimeoutError(f"N1-164 CPU limit exceeded: {stage_cpu:.6f}s")
    report = {
        "schema": "taiji-n1-plan163-itemwise-attribution-v1",
        "status": "complete",
        "complete": True,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "cpu_seconds": round(stage_cpu, 6),
        "result_sha256": sha256_bytes(result_bytes),
        "progress_sha256": progress_sha,
        "readiness_sha256": result["readiness_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "plan_sha256": result["plan_sha256"],
        "historical_runner_sha256": runner_recorded,
        "auditor_sha256": sha256_file(AUDITOR),
        "supersedes_report_sha256": INITIAL_REPORT_SHA256,
        "source_hashes_current_and_matched": source_checks,
        "progress_event_counts": dict(sorted(progress_events.items())),
        "endpoint_training_update_counts": {
            f"seed{seed}/{arm}": len(training_steps[(seed, arm)])
            for seed, arm in expected_model_order
        },
        "gate_progress_arm_labels": dict(sorted(gate_arm_labels.items())),
        "gate_row_arm_label_limitation_zh": "逐题原门行的arm字段均为父档position，来自runner将score_row展开在续训arm之后。按result顺序和runner的固定seed/arm循环分成六个960行块；六块原门摘要均逐项复算匹配result后才接受归因。原始progress未被修改。",
        "endpoint_gate_recomputed_matches_result": True,
        "gate_pairwise_candidate_vs_control": pairwise_gate,
        "length_pairwise_candidate_vs_control": pairwise_length,
        "training_updates": 0,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "checkpoint_writes": 0,
        "model_queries": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "summary_zh": "N1-163全部六个原门评分块可由逐题progress复算并与封存result一致。candidate与control的逐题变化按既有train输入作配对统计；该读数只解释训练面内的变化。",
        "limitations_zh": [
            "训练gate和长度评分均复用既有train题，不能证明新题泛化或模型能力晋级。",
            "原门progress的续训arm标签覆盖为position；归因依赖已通过六个result摘要核验的固定分块顺序。",
            "因子原门的progress未保存四种条件名称；候选/对照逐行配对按同一父题下的出现顺序重建，只报告四行合计，不归因到具名条件。",
            "本报告不作RNN架构因果判断。",
        ],
    }
    with REPORT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "cpu_seconds": report["cpu_seconds"], "endpoints": len(models), "gate_rows": len(gate_rows), "length_rows": len(length_rows), "all_summaries_match": True}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
