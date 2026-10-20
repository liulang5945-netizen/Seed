from __future__ import annotations

from collections import Counter

from scripts.training import (
    run_taiji_n1_plan162g2_mixed_context_thread_scaling_benchmark as benchmark,
)


def test_length_stratified_schedule_selection_is_balanced_and_disjoint() -> None:
    rows = [
        {"kind": kind, "context_length": length}
        for kind in benchmark.KINDS
        for length in benchmark.LENGTHS
        for _ in range(1_200)
    ]

    warmup, measured = benchmark.select_schedule_indices(rows)

    assert len(warmup) == 32
    assert len(measured) == 256
    assert not (set(warmup) & set(measured))
    assert Counter(rows[index]["context_length"] for index in measured) == Counter({8: 96, 16: 80, 32: 80})
    assert benchmark._counts(rows, measured) == {
        "8": {"lookup": 21, "revision": 21, "transition": 21, "unknown": 33},
        "16": {"lookup": 21, "revision": 21, "transition": 21, "unknown": 17},
        "32": {"lookup": 21, "revision": 21, "transition": 21, "unknown": 17},
    }
    assert len(warmup + measured) == 288


def test_g2_keeps_thread_comparison_and_parent_outputs_separate() -> None:
    assert benchmark.PLAN.is_file()
    assert benchmark._sha256(benchmark.PLAN) == benchmark.PLAN_SHA256
    assert benchmark.g.THREAD_COUNTS == (1, 2, 4)
    assert benchmark.g.TRIAL_ORDER == ((1, 1), (2, 1), (4, 1), (4, 2), (2, 2), (1, 2))
    assert benchmark.G2_ATTEMPT.name.startswith("taiji_n1_plan162g2_")
    assert benchmark.G2_PROGRESS.name.startswith("taiji_n1_plan162g2_")
    assert benchmark.G2_RESULT.name.startswith("taiji_n1_plan162g2_")
    assert benchmark.CPU_LIMIT_SECONDS == 1_200.0
