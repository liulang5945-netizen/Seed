from __future__ import annotations

from scripts.training import benchmark_taiji_r2_cpu_thread_scaling as benchmark


def test_thread_scaling_reserves_cpu_before_starting_a_trial() -> None:
    assert benchmark.CPU_LIMIT_SECONDS == 900.0
    assert benchmark.TRIAL_CPU_RESERVE_SECONDS == 300.0
    assert benchmark._trial_fits_cpu_budget(599.0)
    assert not benchmark._trial_fits_cpu_budget(600.1)
    assert not benchmark._trial_fits_cpu_budget(948.4375)
