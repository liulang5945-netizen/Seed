"""N1-162G只读前置修复包装器；复用冻结的吞吐设计和训练实现。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import run_taiji_n1_plan162g_training_thread_scaling_benchmark as g

PLAN = ROOT / "plans/reference/PLAN-N1-162G1_training_thread_scaling_preflight_repair_20261020.md"
PLAN_SHA256 = "015e364a7a7e2dd2c48bac83abd40eec950c51f953bb144524b6e7ac8e42994d"
PARENT_PLAN = ROOT / "plans/reference/PLAN-N1-162G_training_thread_scaling_benchmark_20261020.md"
PARENT_PLAN_SHA256 = "a6ff0f26ae3548c706e0e99c464a496d14dde298a75aa116943c9e5624c26ad9"
G_RUNNER = ROOT / "scripts/training/run_taiji_n1_plan162g_training_thread_scaling_benchmark.py"
G_RUNNER_SHA256 = "91c5ca22a6c7db69692319b1d1ac274acebae932bd83c8234ec3749767aa0ed3"
FAILED_ATTEMPT = ROOT / "reports/taiji_n1_plan162g_training_thread_scaling_attempt_20261020.json"
FAILED_ATTEMPT_SHA256 = "5859304fff227aea2cbbcf761e3a1eb787853e73a98b40261b5411f830f25702"
FAILED_PROGRESS = ROOT / "reports/taiji_n1_plan162g_training_thread_scaling_progress_20261020.jsonl"
FAILED_PROGRESS_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
FAILED_RESULT = ROOT / "reports/taiji_n1_plan162g_training_thread_scaling_20261020.json"
FAILED_RESULT_SHA256 = "c84874ec479f43046a3c5962b4918e414dcd3c34eedd345fa150cbc4e8bffa63"
PRIOR_FAILED_CPU_SECONDS = 38.265625
G1_CPU_LIMIT_SECONDS = 861.734375
LINKED_CPU_LIMIT_SECONDS = 900.0

g.PLAN = PLAN
g.PLAN_SHA256 = PLAN_SHA256
g.ATTEMPT = ROOT / "reports/taiji_n1_plan162g1_training_thread_scaling_attempt_20261020.json"
g.PROGRESS = ROOT / "reports/taiji_n1_plan162g1_training_thread_scaling_progress_20261020.jsonl"
g.RESULT = ROOT / "reports/taiji_n1_plan162g1_training_thread_scaling_20261020.json"
g.CPU_LIMIT_SECONDS = G1_CPU_LIMIT_SECONDS


def _sha256(path: Path) -> str:
    return g.sha256(path)


def _verify_failed_parent_attempt() -> None:
    expected = {
        PLAN: PLAN_SHA256,
        PARENT_PLAN: PARENT_PLAN_SHA256,
        G_RUNNER: G_RUNNER_SHA256,
        FAILED_ATTEMPT: FAILED_ATTEMPT_SHA256,
        FAILED_PROGRESS: FAILED_PROGRESS_SHA256,
        FAILED_RESULT: FAILED_RESULT_SHA256,
    }
    for path, expected_hash in expected.items():
        if not path.is_file() or _sha256(path) != expected_hash:
            raise g.Refuse(f"N1-162G1 failed-parent evidence changed: {path}")


def _verify_hashes_with_g1_sources(sources: dict[str, str]) -> bool:
    sources[str(PLAN.resolve())] = PLAN_SHA256
    sources[str(PARENT_PLAN.resolve())] = PARENT_PLAN_SHA256
    sources[str(G_RUNNER.resolve())] = G_RUNNER_SHA256
    sources[str(Path(__file__).resolve())] = _sha256(Path(__file__).resolve())
    sources[str(FAILED_ATTEMPT.resolve())] = FAILED_ATTEMPT_SHA256
    sources[str(FAILED_PROGRESS.resolve())] = FAILED_PROGRESS_SHA256
    sources[str(FAILED_RESULT.resolve())] = FAILED_RESULT_SHA256
    return n150_builder.verify_hashes(sources)


def configure() -> None:
    """Bind G's frozen engine to G1's plan, verifier, and write-once outputs."""
    _verify_failed_parent_attempt()
    if getattr(g, "_plan162g1_configured", False):
        return
    # G1 repair: verifier belongs to the N1-150 build module, not its runner.
    g.n150.verify_hashes = _verify_hashes_with_g1_sources
    original_write_new = g.write_new

    def _write_with_linked_budget(path: Path, value: dict) -> None:
        if path in {g.ATTEMPT, g.RESULT}:
            value = dict(value)
            value["prior_failed_cpu_seconds"] = PRIOR_FAILED_CPU_SECONDS
            value["linked_cpu_limit_seconds"] = LINKED_CPU_LIMIT_SECONDS
            if "cpu_seconds" in value:
                value["linked_cpu_seconds"] = round(PRIOR_FAILED_CPU_SECONDS + float(value["cpu_seconds"]), 6)
        original_write_new(path, value)

    g.write_new = _write_with_linked_budget
    g._plan162g1_configured = True


def main() -> int:
    try:
        configure()
        g.torch.set_num_threads(1)
        result = g.execute()
    except Exception as exc:
        print(f"PLAN-N1-162G1 refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in (
        "status", "cpu_seconds", "parameter_updates_in_memory", "summary_by_threads",
        "best_median_thread_count", "stable_recommendation_thread_count", "summary_zh",
    )}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
