from __future__ import annotations

from scripts.training import run_taiji_n1_plan162g_training_thread_scaling_benchmark as benchmark


def test_thread_scaling_uses_balanced_repeated_runs_and_fixed_update_count() -> None:
    assert benchmark.THREAD_COUNTS == (1, 2, 4)
    assert benchmark.TRIAL_ORDER == ((1, 1), (2, 1), (4, 1), (4, 2), (2, 2), (1, 2))
    assert all(sum(1 for threads, _ in benchmark.TRIAL_ORDER if threads == count) == 2 for count in benchmark.THREAD_COUNTS)
    assert benchmark.WARMUP_UPDATES == 32
    assert benchmark.MEASURED_UPDATES == 256
    assert len(benchmark.TRIAL_ORDER) * benchmark.SCHEDULE_ROWS == 1_728


def test_summary_requires_repeated_ten_percent_speedup_before_recommendation() -> None:
    trials = [
        {"threads": 1, "updates_per_second": 10.0, "cpu_seconds_per_update": 0.10},
        {"threads": 1, "updates_per_second": 10.2, "cpu_seconds_per_update": 0.098},
        {"threads": 2, "updates_per_second": 11.5, "cpu_seconds_per_update": 0.087},
        {"threads": 2, "updates_per_second": 11.7, "cpu_seconds_per_update": 0.085},
        {"threads": 4, "updates_per_second": 14.0, "cpu_seconds_per_update": 0.071},
        {"threads": 4, "updates_per_second": 10.8, "cpu_seconds_per_update": 0.092},
    ]

    summary, winner, stable_winner = benchmark._summarize_trials(trials)

    assert summary["2"]["stable_gain_10_percent"] is True
    assert summary["4"]["stable_gain_10_percent"] is False
    assert winner == 4
    assert stable_winner == 2


def test_summary_rejects_missing_replicate() -> None:
    trials = [{"threads": 1, "updates_per_second": 10.0, "cpu_seconds_per_update": 0.1}]

    try:
        benchmark._summarize_trials(trials)
    except benchmark.Refuse as exc:
        assert "expected 2" in str(exc)
    else:
        raise AssertionError("missing timing replicate must fail closed")
