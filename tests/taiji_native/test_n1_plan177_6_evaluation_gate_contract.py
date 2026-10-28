"""N177.6 评价器的开评门契约册（holdout 只读一次）。

评价器的状态机本身就是主守卫：评完会置 `evaluation_complete` / `partial_evaluation_complete` /
`evaluation_budget_stop`，三者都不在可评集合里。但这留了两个缝，本册分别钉住：

* 报告写出是无条件覆盖 —— 于是"重跑一次"会静默换掉一份已发表的读数；
* 进程中途崩溃会停在 `evaluation_in_progress`（可评），重跑会**再读一遍 holdout** 而没人记得读过几次。

两条都只做加性记账与显式开关，不改任何阈值、臂选择或评分语义。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.training import evaluate_taiji_n1_plan177_6_t1_pilot as scorer  # noqa: E402


@pytest.mark.parametrize(
    "status",
    [
        "evaluation_complete",
        "partial_evaluation_complete",
        "evaluation_budget_stop",
        "paused_by_user",
        "running",
        "engineering_failure",
    ],
)
def test_gate_refuses_after_training_or_scoring_finished(tmp_path: Path, status: str) -> None:
    reason = scorer.evaluation_gate({"status": status}, tmp_path / "new.json")
    assert reason is not None
    assert "fixed boundary" in reason


@pytest.mark.parametrize(
    "status",
    ["training_complete_evaluation_pending", "budget_stop", "evaluation_in_progress"],
)
def test_gate_allows_the_three_operable_statuses(tmp_path: Path, status: str) -> None:
    assert scorer.evaluation_gate({"status": status}, tmp_path / "new.json") is None


def test_gate_refuses_to_silently_overwrite_an_existing_report(tmp_path: Path) -> None:
    report = tmp_path / "already.json"
    report.write_text('{"status": "complete"}', encoding="utf-8")
    state = {"status": "training_complete_evaluation_pending"}

    reason = scorer.evaluation_gate(state, report)
    assert reason is not None and "holdout 只读一次" in reason
    #: 显式开关才放行——旧报告另存留证的动作在 main 里，开关本身就是"我知道我在重读"。
    assert scorer.evaluation_gate(state, report, overwrite_report=True) is None


def test_attempt_counter_is_recorded_before_any_scoring() -> None:
    state: dict[str, object] = {"status": "training_complete_evaluation_pending"}
    assert scorer.record_evaluation_attempt(state) == 1
    assert state["evaluation_attempts"] == 1
    assert isinstance(state["evaluation_last_start_utc"], str)
    #: 崩溃恢复那次会被记成 2——正常路径下这个数应当恒为 1，报告里也照原样出版。
    assert scorer.record_evaluation_attempt(state) == 2


def test_scorer_holds_the_same_run_root_lock_as_the_trainer() -> None:
    """所有权只有一把锁：评价器必须复用训练器那份 acquire/release，不许自己再写一份。"""

    from scripts.training import run_taiji_n1_plan177_6_t1_pilot as trainer

    assert scorer.acquire_run_lock is trainer.acquire_run_lock
    assert scorer.release_run_lock is trainer.release_run_lock
