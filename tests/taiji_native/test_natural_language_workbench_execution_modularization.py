"""P5-3 Workbench execution boundary modularization Gate."""

from scripts.training.eval_taiji_natural_language_workbench_execution_modularization import (
    evaluate,
)


def test_natural_language_workbench_execution_modularization_gate_passes() -> None:
    report = evaluate()
    # 失败时必须说出是哪一条指标：这个门由 4 条独立判据合取而成，其中一条要在同一进程里
    # 读别的模块源码，历史上它在整套里红、单跑绿过（2026-09-18），只报 "False is True" 无法归因。
    failed = [key for key, passed in report["metrics"].items() if not passed]
    assert report["gate"]["passed"] is True, failed
