from __future__ import annotations

from scripts.training import run_taiji_n1_plan162g1_training_thread_scaling_benchmark as repair


def test_repair_binds_new_plan_and_preserves_failed_parent_paths() -> None:
    repair.configure()

    assert repair.g.PLAN == repair.PLAN
    assert repair.g.PLAN_SHA256 == repair.PLAN_SHA256
    assert repair.g.ATTEMPT.name.startswith("taiji_n1_plan162g1_")
    assert repair.g.PROGRESS.name.startswith("taiji_n1_plan162g1_")
    assert repair.g.RESULT.name.startswith("taiji_n1_plan162g1_")
    assert repair.g.ATTEMPT != repair.g.ROOT / "reports/taiji_n1_plan162g_training_thread_scaling_attempt_20261020.json"
    assert repair.g.CPU_LIMIT_SECONDS == repair.G1_CPU_LIMIT_SECONDS
    assert repair.PRIOR_FAILED_CPU_SECONDS + repair.G1_CPU_LIMIT_SECONDS == repair.LINKED_CPU_LIMIT_SECONDS


def test_correct_verifier_is_added_to_g_runner_module() -> None:
    repair.configure()

    assert repair.g.n150.verify_hashes is repair._verify_hashes_with_g1_sources
