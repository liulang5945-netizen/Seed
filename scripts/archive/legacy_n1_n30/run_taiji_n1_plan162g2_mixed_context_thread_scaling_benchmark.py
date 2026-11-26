"""N1-162G2补全8/16/32行上下文混合配比的CPU线程基准。"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import run_taiji_n1_plan162g_training_thread_scaling_benchmark as g

PLAN = ROOT / "plans/reference/PLAN-N1-162G2_mixed_context_thread_scaling_benchmark_20261020.md"
PLAN_SHA256 = "be391eb5cdc676963ef83e79cc3a4baa6da8e6a92f2264c217c36dca1c82c2e1"
G1_RESULT = ROOT / "reports/taiji_n1_plan162g1_training_thread_scaling_20261020.json"
G1_RESULT_SHA256 = "2659d8aaf9a3c034b1bff18dfe65b5582c5001664ff4483d3fe137395f07933c"
G1_PLAN = (
    ROOT / "plans/reference/PLAN-N1-162G1_training_thread_scaling_preflight_repair_20261020.md"
)
G1_PLAN_SHA256 = "b56b5eec8d2b2f9b7b11051f0d3e0404c0cd882afdc9b9a67cb463d2331018af"
G1_RUNNER = ROOT / "scripts/training/run_taiji_n1_plan162g1_training_thread_scaling_benchmark.py"
G1_RUNNER_SHA256 = "f9638d79228679de3b16390dd8698d4aece18f9e44682c58bc7af6a683c75cdb"
G1_ATTEMPT = ROOT / "reports/taiji_n1_plan162g1_training_thread_scaling_attempt_20261020.json"
G1_ATTEMPT_SHA256 = "02814963e71a7cabef40236913555b2c0da1682b14eecbb1d7adb018135dbd26"
G1_PROGRESS = ROOT / "reports/taiji_n1_plan162g1_training_thread_scaling_progress_20261020.jsonl"
G1_PROGRESS_SHA256 = "80889e23de38ec929480480fcb8fa28f5a00b3a51a3424143e3e93e5888ab878"
G_RUNNER = ROOT / "scripts/training/run_taiji_n1_plan162g_training_thread_scaling_benchmark.py"
G_RUNNER_SHA256 = "91c5ca22a6c7db69692319b1d1ac274acebae932bd83c8234ec3749767aa0ed3"
G_PLAN = ROOT / "plans/reference/PLAN-N1-162G_training_thread_scaling_benchmark_20261020.md"
G_PLAN_SHA256 = "4d427329db7705f32ba30b3a23e868efd49c18dc79d8018812617d44a44584f7"
G_ATTEMPT = ROOT / "reports/taiji_n1_plan162g_training_thread_scaling_attempt_20261020.json"
G_ATTEMPT_SHA256 = "5859304fff227aea2cbbcf761e3a1eb787853e73a98b40261b5411f830f25702"
G_PROGRESS = ROOT / "reports/taiji_n1_plan162g_training_thread_scaling_progress_20261020.jsonl"
G_PROGRESS_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
G_RESULT = ROOT / "reports/taiji_n1_plan162g_training_thread_scaling_20261020.json"
G_RESULT_SHA256 = "c84874ec479f43046a3c5962b4918e414dcd3c34eedd345fa150cbc4e8bffa63"
G2_ATTEMPT = ROOT / "reports/taiji_n1_plan162g2_mixed_context_thread_scaling_attempt_20261020.json"
G2_PROGRESS = (
    ROOT / "reports/taiji_n1_plan162g2_mixed_context_thread_scaling_progress_20261020.jsonl"
)
G2_RESULT = ROOT / "reports/taiji_n1_plan162g2_mixed_context_thread_scaling_20261020.json"

KINDS = ("lookup", "revision", "transition", "unknown")
LENGTHS = (8, 16, 32)
WARMUP_TARGETS = {
    ("lookup", 8): 3,
    ("revision", 8): 3,
    ("transition", 8): 3,
    ("unknown", 8): 3,
    ("lookup", 16): 3,
    ("revision", 16): 3,
    ("transition", 16): 2,
    ("unknown", 16): 2,
    ("lookup", 32): 3,
    ("revision", 32): 3,
    ("transition", 32): 2,
    ("unknown", 32): 2,
}
MEASURED_TARGETS = {
    ("lookup", 8): 21,
    ("revision", 8): 21,
    ("transition", 8): 21,
    ("unknown", 8): 33,
    ("lookup", 16): 21,
    ("revision", 16): 21,
    ("transition", 16): 21,
    ("unknown", 16): 17,
    ("lookup", 32): 21,
    ("revision", 32): 21,
    ("transition", 32): 21,
    ("unknown", 32): 17,
}
SAMPLE_DESIGN: dict = {}
CPU_LIMIT_SECONDS = 1_200.0

g.PLAN = PLAN
g.PLAN_SHA256 = PLAN_SHA256
g.ATTEMPT = G2_ATTEMPT
g.PROGRESS = G2_PROGRESS
g.RESULT = G2_RESULT
g.CPU_LIMIT_SECONDS = CPU_LIMIT_SECONDS


def _sha256(path: Path) -> str:
    return g.sha256(path)


def _verify_parent_chain() -> dict[str, str]:
    expected = {
        PLAN: PLAN_SHA256,
        G1_RESULT: G1_RESULT_SHA256,
        G1_PLAN: G1_PLAN_SHA256,
        G1_RUNNER: G1_RUNNER_SHA256,
        G1_ATTEMPT: G1_ATTEMPT_SHA256,
        G1_PROGRESS: G1_PROGRESS_SHA256,
        G_RUNNER: G_RUNNER_SHA256,
        G_PLAN: G_PLAN_SHA256,
        G_ATTEMPT: G_ATTEMPT_SHA256,
        G_PROGRESS: G_PROGRESS_SHA256,
        G_RESULT: G_RESULT_SHA256,
        Path(__file__).resolve(): _sha256(Path(__file__).resolve()),
    }
    for path, expected_hash in expected.items():
        if not path.is_file() or _sha256(path) != expected_hash:
            raise g.Refuse(f"N1-162G2 parent-chain file changed: {path}")
    prior = json.loads(G1_RESULT.read_text(encoding="utf-8"))
    if (
        prior.get("status") != "complete"
        or prior.get("complete") is not True
        or prior.get("live_independent_restores") != 6
        or prior.get("parameter_updates_in_memory") != 1_728
        or prior.get("checkpoint_writes") != 0
        or prior.get("source_hashes_unchanged") is not True
        or prior.get("can_promote") is not False
    ):
        raise g.Refuse("N1-162G1 performance-only result is incomplete")
    prior_sources = prior.get("source_hashes", {})
    if not prior_sources or not n150_builder.verify_hashes(prior_sources):
        raise g.Refuse("N1-162G1 bound sources drifted before G2")
    prior_sources[str(G1_RESULT.resolve())] = G1_RESULT_SHA256
    return prior_sources


def _verify_hashes_with_g2_sources(sources: dict[str, str]) -> bool:
    sources.update(_verify_parent_chain())
    sources[str(PLAN.resolve())] = PLAN_SHA256
    sources[str(Path(__file__).resolve())] = _sha256(Path(__file__).resolve())
    return n150_builder.verify_hashes(sources)


def _spread(indices: list[int], count: int) -> list[int]:
    if count < 0 or count > len(indices):
        raise g.Refuse(f"stratified sample count is infeasible: {count}/{len(indices)}")
    if count == 0:
        return []
    positions = [math.floor((offset + 0.5) * len(indices) / count) for offset in range(count)]
    chosen = [indices[position] for position in positions]
    if len(set(chosen)) != count:
        raise g.Refuse("stratified schedule positions are not unique")
    return chosen


def select_schedule_indices(rows: list[dict]) -> tuple[list[int], list[int]]:
    groups = {
        key: [
            index
            for index, row in enumerate(rows)
            if (str(row["kind"]), int(row["context_length"])) == key
        ]
        for key in WARMUP_TARGETS
    }
    warmup: list[int] = []
    measured: list[int] = []
    for key in WARMUP_TARGETS:
        group = groups[key]
        total = WARMUP_TARGETS[key] + MEASURED_TARGETS[key]
        selected = _spread(group, total)
        warmup.extend(_spread(selected, WARMUP_TARGETS[key]))
        warmup_set = set(warmup[-WARMUP_TARGETS[key] :])
        measured.extend(index for index in selected if index not in warmup_set)
    warmup.sort()
    measured.sort()
    if len(warmup) != 32 or len(measured) != 256 or set(warmup) & set(measured):
        raise g.Refuse("G2 stratified warmup/measurement denominators differ")
    return warmup, measured


def _counts(rows: list[dict], indices: list[int]) -> dict[str, dict[str, int]]:
    counts = Counter(
        (str(rows[index]["kind"]), int(rows[index]["context_length"])) for index in indices
    )
    return {
        length: {kind: counts[(kind, int(length))] for kind in KINDS}
        for length in map(str, LENGTHS)
    }


def _verify_sources_with_stratified_schedule() -> tuple[dict, dict, dict, dict[str, str]]:
    runtime, full_schedule, endpoint, sources = _ORIGINAL_VERIFY_SOURCES()
    warmup, measured = select_schedule_indices(full_schedule["rows"])
    indices = warmup + measured
    sample_rows = [full_schedule["rows"][index] for index in indices]
    sample_slots = [full_schedule["slots"][index] for index in indices]
    global SAMPLE_DESIGN
    SAMPLE_DESIGN = {
        "method": "evenly_spread_within_kind_by_context_length",
        "full_schedule_rows": len(full_schedule["rows"]),
        "warmup_indices": warmup,
        "measured_indices": measured,
        "warmup_index_sha256": g.builder.canonical_sha(warmup),
        "measured_index_sha256": g.builder.canonical_sha(measured),
        "warmup_count_by_length_kind": _counts(full_schedule["rows"], warmup),
        "measured_count_by_length_kind": _counts(full_schedule["rows"], measured),
        "selected_count_by_length_kind": _counts(full_schedule["rows"], indices),
        "selected_schedule_rows_sha256": g.builder.canonical_sha(sample_rows),
    }
    sample_schedule = dict(full_schedule, rows=sample_rows, slots=sample_slots)
    return runtime, sample_schedule, endpoint, sources


_ORIGINAL_VERIFY_SOURCES = g._verify_sources
g.n150.verify_hashes = _verify_hashes_with_g2_sources
g._verify_sources = _verify_sources_with_stratified_schedule
_ORIGINAL_WRITE_NEW = g.write_new


def _write_new_with_sample(path: Path, value: dict) -> None:
    if path == G2_ATTEMPT:
        value = dict(value)
        value["sample_contract"] = {
            "warmup_targets_by_length_kind": {
                f"{kind}:{length}": count for (kind, length), count in WARMUP_TARGETS.items()
            },
            "measured_targets_by_length_kind": {
                f"{kind}:{length}": count for (kind, length), count in MEASURED_TARGETS.items()
            },
            "parent_g1_result_sha256": G1_RESULT_SHA256,
        }
    elif path == G2_RESULT:
        value = dict(value)
        value["sample_design"] = SAMPLE_DESIGN
        value["parent_g1_result_sha256"] = G1_RESULT_SHA256
        value["summary_zh"] = (
            "完整8/16/32行长度配比线程基准完成；测量样本按原N1-162B schedule的kind×长度分层，"
            + str(value.get("summary_zh", ""))
        )
    _ORIGINAL_WRITE_NEW(path, value)


g.write_new = _write_new_with_sample


def main() -> int:
    try:
        g.torch.set_num_threads(1)
        report = g.execute()
    except Exception as exc:
        print(f"PLAN-N1-162G2 refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "cpu_seconds",
                    "parameter_updates_in_memory",
                    "summary_by_threads",
                    "best_median_thread_count",
                    "stable_recommendation_thread_count",
                    "sample_design",
                    "summary_zh",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
